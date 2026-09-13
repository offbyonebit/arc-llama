"""Command-line entry point: ``python -m arc_llama_vision``.

Cross-platform rules from the integration doc apply: no POSIX-only paths or
signals, platform-neutral host/port handling, and graceful shutdown through
uvicorn's own lifespan handling (SIGTERM/SIGINT included on Windows).
"""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from arc_llama_vision import __version__
from arc_llama_vision.app import create_app
from arc_llama_vision.backend import CapabilityError, create_backend
from arc_llama_vision.config import VisionConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arc-llama-vision",
        description=(
            "Standalone image-generation companion for arc-llama. "
            "Serves an OpenAI-compatible /v1/images API over a backend-neutral adapter seam."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--host", default=None, help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: 11440)")
    parser.add_argument(
        "--backend",
        default=None,
        help="Image backend adapter name (default: fake; known: fake, unavailable, comfyui)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Uvicorn log level",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(argv)

    logging.basicConfig(level={"debug": 10, "info": 20, "warning": 30, "error": 40}[args.log_level])

    # Precedence: CLI flags > ARV_* env vars > defaults.
    cfg = VisionConfig.from_env().apply_overrides(
        host=args.host,
        port=args.port,
        backend=args.backend,
    )

    try:
        backend = create_backend(cfg.backend, cfg.backend_options)
    except CapabilityError as exc:
        # Unknown adapter is a clear, actionable startup error.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - don't die with a raw traceback
        print(f"error: backend {cfg.backend!r} failed to initialize: {exc}", file=sys.stderr)
        return 1

    app = create_app(cfg, backend)
    print(
        f"arc-llama-vision {__version__} listening on http://{cfg.host}:{cfg.port} "
        f"(backend: {cfg.backend})",
        file=sys.stderr,
    )
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
