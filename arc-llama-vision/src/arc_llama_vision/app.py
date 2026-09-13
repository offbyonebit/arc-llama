"""FastAPI application for the arc-llama vision companion.

Endpoints
---------

- ``GET /health`` — liveness probe reflecting backend readiness;
- ``GET /v1/models`` — OpenAI model list with image modality metadata;
- ``POST /v1/images/generations`` — OpenAI-compatible image generation.

The app is backend-neutral: all rendering work crosses the adapter seam in
``arc_llama_vision.backend``, and the two built-in adapters keep the scaffold
deterministic and local-only (no model weights, no heavyweight ML deps).
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from arc_llama_vision.backend import (
    BackendUnavailableError,
    CapabilityError,
    GeneratedImage,
    ImageBackend,
    ModelInfo,
    validate_size,
)

if TYPE_CHECKING:
    from arc_llama_vision.config import VisionConfig

log = logging.getLogger("arc_llama_vision")

_SERVICE_NAME = "vision-companion"


class GenerationRequest(BaseModel):
    """Parsed body for POST /v1/images/generations.

    Mirrors the published OpenAI image-generation parameters; ``model`` and
    ``prompt`` are the only required fields. Unknown fields are ignored by
    pydantic so clients can pass newer OpenAI options without breaking.
    """

    model_config = {"extra": "ignore"}

    model: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    n: int = Field(default=1, ge=1)
    size: str | None = Field(default=None)
    quality: str | None = Field(default=None)
    response_format: str | None = Field(default=None)  # "url" or "b64_json"
    style: str | None = Field(default=None)
    user: str | None = Field(default=None)


def create_app(cfg: VisionConfig, backend: ImageBackend) -> FastAPI:
    """Build the companion app; the caller owns the backend instance.

    Receiving the concrete backend keeps the app factory pure and lets tests
    inject either built-in adapter. Generation errors are translated per the
    documented contract: invalid parameters -> 400, unknown models -> 404,
    unusable backends -> 503, unexpected adapter failures -> 500.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await backend.startup()
        except Exception as exc:  # noqa: BLE001 - startup failures are surfaced via /health
            log.warning("backend %s startup failed: %s", backend.backend_name, exc)
            app.state.backend_ready = False
        else:
            app.state.backend_ready = True
        try:
            yield
        finally:
            await backend.shutdown()

    app = FastAPI(
        title="arc-llama-vision",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.config = cfg
    app.state.backend = backend

    @app.exception_handler(RequestValidationError)
    async def _validation_to_400(request: Any, exc: RequestValidationError) -> JSONResponse:
        # OpenAI-compatible clients expect 400 for malformed requests, not
        # FastAPI's default 422. The error body uses the same stable schema
        # as every other failure: {"detail": {"error": {type, message}}}.
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", []))
        message = first.get("msg", "invalid request")
        detail = f"{loc}: {message}" if loc else str(message)
        return JSONResponse(
            status_code=400,
            content={"detail": {"error": {"type": "invalid_request_error", "message": detail}}},
        )

    def _model_by_id(model_id: str) -> ModelInfo:
        """Return the backend's model metadata or raise the 404 error."""
        for info in backend.models:
            if info.id == model_id:
                return info
        # OpenAI uses 404 for unknown models on /v1/models/{id} but 400 on
        # generations; the audio companion doc pins 404 for unknown models on
        # generation endpoints, so follow the doc.
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "type": "invalid_request_error",
                    "message": f"Unknown model {model_id!r}",
                }
            },
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        status: str = "ok" if app.state.backend_ready else "degraded"
        return {"status": status, "backend": backend.backend_name}

    @app.get("/v1/models")
    async def models_list() -> dict[str, Any]:
        data = []
        for info in backend.models:
            data.append(
                {
                    "id": info.id,
                    "object": "model",
                    "owned_by": _SERVICE_NAME,
                    "created": 0,
                    "metadata": {
                        "modality": info.metadata.get("modality", "image->image"),
                        "backend": info.backend,
                        **{k: v for k, v in info.metadata.items() if k != "modality"},
                    },
                }
            )
        return {"object": "list", "data": data}

    @app.post("/v1/images/generations")
    async def create_images(req: GenerationRequest) -> dict[str, Any]:
        # Parameter guards documented in the README's Limits section.
        if req.n > cfg.max_batch_images:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"n exceeds the configured maximum of {cfg.max_batch_images}",
                    }
                },
            )
        if len(req.prompt) > cfg.max_prompt_chars:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "type": "invalid_request_error",
                        "message": (
                            f"prompt exceeds the configured maximum of "
                            f"{cfg.max_prompt_chars} characters"
                        ),
                    }
                },
            )

        # b64_json is the only guaranteed output format (see README): serving
        # persistent URLs would require an image store, which conflicts with
        # the companion staying stateless and local-first. Reject "url"
        # before any backend work happens.
        if req.response_format == "url":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "type": "invalid_request_error",
                        "message": (
                            "response_format 'url' is not supported; use 'b64_json' (the default)"
                        ),
                    }
                },
            )

        model_info = _model_by_id(req.model)

        # Size syntax/bounds are validated at the endpoint so no adapter work
        # happens for malformed input; adapters enforce their own supported
        # sizes on top of this.
        try:
            validate_size(req.size)
        except CapabilityError as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": {"type": "invalid_request_error", "message": str(exc)}},
            ) from exc

        if not app.state.backend_ready:
            raise HTTPException(
                status_code=503,
                detail={"error": {"type": "server_error", "message": "Backend unavailable"}},
            )

        try:
            images: list[GeneratedImage] = await backend.generate(
                req.prompt,
                model_info,
                n=req.n,
                size=req.size,
                quality=req.quality,
                style=req.style,
            )
        except CapabilityError as exc:
            # Invalid sizes/formats are request errors (400).
            raise HTTPException(
                status_code=400,
                detail={"error": {"type": "invalid_request_error", "message": str(exc)}},
            ) from exc
        except BackendUnavailableError as exc:
            raise HTTPException(
                status_code=503,
                detail={"error": {"type": "server_error", "message": str(exc)}},
            ) from exc
        except Exception as exc:  # noqa: BLE001 - adapters may raise anything
            log.exception("backend %s generation failed", backend.backend_name)
            raise HTTPException(
                status_code=500,
                detail={"error": {"type": "server_error", "message": str(exc)}},
            ) from exc

        results: list[dict[str, Any]] = [
            img.to_response_dict() for img in images[: cfg.max_batch_images]
        ]

        return {
            "created": int(time.time()),
            "data": results,
        }

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


def _cors_origins() -> list[str]:
    """Same opt-in policy idea as arc-llama core (localhost browser UIs)."""
    import os

    configured = os.environ.get("ARV_CORS_ORIGINS")
    if configured is not None:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return ["http://localhost:3000", "http://127.0.0.1:3000"]
