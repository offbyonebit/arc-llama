"""arc-llama-vision: bounded image-generation companion scaffold for arc-llama.

A standalone, local-first service exposing a small OpenAI-compatible
image-generation surface (health, models with modality metadata, generations)
through a backend-neutral adapter seam. Reuses the integration boundary and
lifecycle conventions of arc-llama's documented plugin and audio-companion
contracts: no imports of private arc-llama runtime internals, no model
weights, no heavyweight ML dependencies.
"""

from arc_llama_vision.backend import (
    BACKEND_REGISTRY,
    BackendUnavailableError,
    CapabilityError,
    FakeImageBackend,
    GeneratedImage,
    ImageBackend,
    ModelInfo,
    UnavailableBackend,
    create_backend,
)
from arc_llama_vision.config import VisionConfig

__version__ = "0.1.0"

__all__ = [
    "BACKEND_REGISTRY",
    "BackendUnavailableError",
    "CapabilityError",
    "FakeImageBackend",
    "GeneratedImage",
    "ImageBackend",
    "ModelInfo",
    "UnavailableBackend",
    "VisionConfig",
    "create_backend",
    "__version__",
]
