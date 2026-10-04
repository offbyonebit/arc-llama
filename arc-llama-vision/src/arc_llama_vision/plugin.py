"""Optional Arc Llama dashboard adapter for the vision companion.

The dashboard route acquires an exclusive GPU lease from the core's
resource-lease manager for the duration of the outbound generation request,
so image generation never runs concurrently with resident llama-server
models (or another exclusive plugin task) on the same GPU. The manager
waits for active router requests to drain and stops resident models
itself; this adapter only asks for the lease.
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from arc_llama.plugins import Plugin

# The companion's default API base, matching VisionConfig defaults.
_VISION_API_BASE = "http://127.0.0.1:11440"
_VISION_REQUEST_TIMEOUT = 1500.0


class VisionPlugin(Plugin):
    name = "vision"

    def register(self, app: FastAPI) -> None:
        api_base = _VISION_API_BASE

        @app.get("/plugins/vision")
        async def vision_plugin_info() -> JSONResponse:
            return JSONResponse(
                {
                    "name": self.name,
                    "service": "arc-llama-vision",
                    "message": "Start the vision companion on port 11440 to use image generation.",
                    "openai_endpoint": f"{api_base}/v1/images/generations",
                }
            )

        @app.post("/plugins/vision/generate")
        async def vision_generate(request: Request) -> JSONResponse:
            try:
                body = await request.json()
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc
            if not isinstance(body, dict) or not isinstance(body.get("prompt"), str) or not body["prompt"].strip():
                raise HTTPException(status_code=400, detail="prompt must be a nonempty string")
            if "model" in body and (not isinstance(body["model"], str) or not body["model"].strip()):
                raise HTTPException(status_code=400, detail="model must be a nonempty string")
            resources = getattr(request.app.state, "resources", None)
            if resources is None:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "GPU resource arbitration is unavailable on this server; "
                        "the vision adapter requires arc-llama's resource lease "
                        "manager to run image generation safely."
                    ),
                )
            try:
                async with httpx.AsyncClient(timeout=_VISION_REQUEST_TIMEOUT) as client:
                    # Discover before eviction: a missing companion or invalid
                    # model must not stop a healthy text backend. The fake and
                    # ComfyUI adapters advertise different model identifiers.
                    catalog = await client.get(f"{api_base}/v1/models")
                    catalog.raise_for_status()
                    try:
                        data = catalog.json()
                    except ValueError as exc:
                        raise HTTPException(status_code=503, detail="Vision companion returned an invalid model catalog") from exc
                    entries = data.get("data") if isinstance(data, dict) else None
                    ids = [entry["id"] for entry in entries if isinstance(entry, dict) and isinstance(entry.get("id"), str) and entry["id"].strip()] if isinstance(entries, list) else []
                    if not ids:
                        raise HTTPException(status_code=503, detail="Vision companion advertises no available image models")
                    model = body.get("model", ids[0])
                    if model not in ids:
                        raise HTTPException(status_code=404, detail=f"Unknown vision model: {model}")
                    payload = {"model": model, "prompt": body["prompt"], "size": body.get("size", "512x512")}
                    async with resources.acquire(self.name, exclusive=True):
                        response = await client.post(f"{api_base}/v1/images/generations", json=payload)
                        response.raise_for_status()
                        return JSONResponse(response.json())
            except httpx.HTTPStatusError as exc:
                # An actual backend error must not be presented as a missing
                # service. Preserve its status and explain the rejected request.
                try:
                    error = exc.response.json()
                    detail = error.get("detail") if isinstance(error, dict) else None
                    if isinstance(detail, dict):
                        detail = detail.get("error", {}).get("message")
                except (ValueError, AttributeError):
                    detail = None
                raise HTTPException(
                    status_code=exc.response.status_code,
                    detail=detail if isinstance(detail, str) else f"Vision companion returned HTTP {exc.response.status_code}",
                ) from exc
            except httpx.RequestError as exc:
                raise HTTPException(status_code=503, detail="Vision companion is not running or responding on port 11440") from exc
            except TimeoutError as exc:
                raise HTTPException(status_code=503, detail="GPU work did not drain for image generation; retry after it finishes") from exc

    def info(self) -> dict[str, Any]:
        return {
            "version": "0.1.0",
            "description": "Image-generation companion integration",
            "ui": {
                "actions": [
                    {
                        "id": "vision.open",
                        "label": "Vision companion",
                        "description": "Generate an image without competing for GPU memory",
                        "icon": "image",
                        "placement": "plugins",
                        "route": "/plugins/vision/generate",
                        "method": "POST",
                        "composer": {
                            "mode": "text",
                            "result": "image",
                            "placeholder": "Describe the image you want to create…",
                        },
                    }
                ],
            },
            "api": ["/plugins/vision", f"{_VISION_API_BASE}/v1/images/generations"],
        }


def create_plugin() -> VisionPlugin:
    return VisionPlugin()
