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
            body = await request.json()
            if not isinstance(body, dict) or not str(body.get("prompt", "")).strip():
                raise HTTPException(status_code=400, detail="prompt is required")
            payload = {
                "model": body.get("model", "arc-vision-diffusion"),
                "prompt": body["prompt"],
                "size": body.get("size", "512x512"),
            }

            # Present on every standard arc-llama server (app.state.resources,
            # a ResourceLeaseManager). Its absence means the adapter was
            # registered against a server without GPU arbitration; generating
            # anyway would let the vision companion contend with llama-server
            # for VRAM with nothing serialising them, so refuse clearly.
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

            # Exclusive lease: active router requests drain, resident local
            # models stop, and other exclusive plugin tasks serialize on the
            # same gate. Released on return, error, or cancellation.
            async with resources.acquire(self.name, exclusive=True):
                try:
                    # Real diffusion backends may take many minutes on CPU or
                    # during a cold-start. Keep the proxy alive for the same
                    # bounded window as the companion's render contract.
                    async with httpx.AsyncClient(timeout=_VISION_REQUEST_TIMEOUT) as client:
                        response = await client.post(
                            f"{api_base}/v1/images/generations", json=payload
                        )
                        response.raise_for_status()
                        return JSONResponse(response.json())
                except httpx.HTTPError as exc:
                    raise HTTPException(
                        status_code=503,
                        detail="Vision companion is not running on port 11440",
                    ) from exc

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
