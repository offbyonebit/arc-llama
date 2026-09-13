"""Reference configuration for the vision companion.

The companion keeps no state on disk. Configuration values are applied in the
documented precedence order:

1. built-in defaults (shown below);
2. ``ARV_*`` environment variables; and then
3. command-line options, which always win.

Tests and embedding callers can also construct :class:`VisionConfig` directly
and pass it to ``create_app``.

Windows-safe by construction: ports are integers and paths never appear here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

# The default service port lives alongside arc-llama's default 11437 and the
# audio companion's 11438 shown in docs/integrations/audio-companion.md.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11440

# "fake" keeps the scaffold local-first and dependency-free; a real backend
# adapter plugs in behind the same adapter seam instead.
DEFAULT_BACKEND = "fake"

# Bounded, documented generation limits. The scaffold deliberately refuses to
# grow work with request size: batches beyond this size or prompts over this
# length are rejected with 400 before any backend work happens.
DEFAULT_MAX_BATCH_IMAGES = 1
DEFAULT_MAX_PROMPT_CHARS = 8192


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class VisionConfig:
    """Runtime configuration for the companion service."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    backend: str = DEFAULT_BACKEND
    max_batch_images: int = DEFAULT_MAX_BATCH_IMAGES
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS
    # Free-form per-adapter options, merged from ARV_BACKEND_OPTIONS (JSON).
    backend_options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> VisionConfig:
        """Build a config from the ARV_* environment variables listed in the README."""
        cfg = cls()
        cfg.host = os.environ.get("ARV_HOST", cfg.host)
        cfg.port = _env_int("ARV_PORT", cfg.port)
        cfg.backend = os.environ.get("ARV_BACKEND", cfg.backend)
        cfg.max_batch_images = _env_int("ARV_MAX_BATCH_IMAGES", cfg.max_batch_images)
        cfg.max_prompt_chars = _env_int("ARV_MAX_PROMPT_CHARS", cfg.max_prompt_chars)

        # Per-backend knobs stay structured JSON so each adapter documents its
        # own keys, e.g. ARV_BACKEND_OPTIONS='{"api_base": "http://127.0.0.1:7860"}'.
        # Malformed JSON falls back to empty options rather than crashing the
        # server at startup; adapter validation surfaces the mistake.
        raw = os.environ.get("ARV_BACKEND_OPTIONS")
        if raw:
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                cfg.backend_options = parsed
        return cfg

    def apply_overrides(self, **overrides: Any) -> VisionConfig:
        """Return a copy with explicitly passed values replaced.

        ``None`` means "leave as configured", letting the CLI keep command-line
        precedence over environment variables without tracking which flags were
        parsed.
        """
        from dataclasses import replace

        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **clean)
