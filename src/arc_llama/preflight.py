"""Fast checks performed before replacing a healthy resident model."""

from __future__ import annotations

import os
import shutil
import socket
from pathlib import Path
from urllib.parse import urlsplit

from arc_llama.config import GPUConfig, ModelConfig
from arc_llama.failures import StartupFailureError
from arc_llama.launcher import LaunchPlan


def _argv_value(argv: list[str], flag: str) -> str | None:
    try:
        index = argv.index(flag)
    except ValueError:
        return None
    return argv[index + 1] if index + 1 < len(argv) else None


def _check_gguf(path_value: str, *, draft: bool = False) -> None:
    path = Path(path_value).expanduser()
    label = "Draft model" if draft else "Model"
    category = "draft_missing" if draft else "model_missing"
    action = (
        "Update or disable the speculative draft configuration."
        if draft
        else "Update the model path or remove this registration."
    )
    if not path.exists():
        raise StartupFailureError(
            category,
            f"{label} file not found: {path}. {action}",
            action,
            details={"path": str(path)},
        )
    if not path.is_file():
        raise StartupFailureError(
            category,
            f"{label} path is not a file: {path}. {action}",
            action,
            details={"path": str(path)},
        )
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        raise StartupFailureError(
            category,
            f"{label} file is not readable: {path}. {action}",
            action,
            details={"path": str(path), "reason": str(exc)},
        ) from exc


def _runtime_path(command: str) -> Path | None:
    expanded = Path(command).expanduser()
    if expanded.is_absolute() or expanded.parent != Path("."):
        return expanded
    resolved = shutil.which(command)
    return Path(resolved) if resolved else None


def _check_runtime(command: str) -> None:
    path = _runtime_path(command)
    action = "Install a compatible runtime or update paths.llama_server."
    if path is None or not path.exists():
        raise StartupFailureError(
            "runtime_missing",
            f"llama-server runtime not found: {command}. {action}",
            action,
            details={"runtime": command},
        )
    if not path.is_file() or (os.name != "nt" and not os.access(path, os.X_OK)):
        raise StartupFailureError(
            "runtime_incompatible",
            f"llama-server runtime is not executable: {path}. {action}",
            action,
            details={"runtime": str(path)},
        )


def _check_port(host: str, port: int) -> None:
    bind_host = host
    if host in {"localhost", ""}:
        bind_host = "127.0.0.1"
    try:
        addresses = socket.getaddrinfo(
            bind_host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
        )
    except OSError as exc:
        raise StartupFailureError(
            "port_in_use",
            f"Cannot check model port {host}:{port}: {exc}.",
            "Choose another model port or correct the configured host.",
            details={"host": host, "port": port, "reason": str(exc)},
        ) from exc
    for family, socktype, protocol, _canonname, sockaddr in addresses:
        probe = socket.socket(family, socktype, protocol)
        try:
            if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(sockaddr)
        except OSError as exc:
            raise StartupFailureError(
                "port_in_use",
                f"Model port is already in use: {host}:{port}.",
                "Stop the process using the port or configure another model port.",
                details={"host": host, "port": port, "reason": str(exc)},
            ) from exc
        finally:
            probe.close()


def preflight_launch(model: ModelConfig, gpu: GPUConfig, plan: LaunchPlan) -> None:
    """Reject predictable launch failures without spawning a subprocess."""
    if not gpu.enabled:
        raise StartupFailureError(
            "gpu_unavailable",
            f"Configured GPU is disabled: {gpu.pci_slot}.",
            "Enable the GPU in the configuration or assign the model to an available GPU.",
            details={"gpu": gpu.pci_slot, "backend": gpu.backend},
        )
    if not plan.argv:
        raise StartupFailureError(
            "runtime_missing",
            "No llama-server runtime is configured.",
            "Install a compatible runtime or update paths.llama_server.",
        )
    _check_runtime(plan.argv[0])
    _check_gguf(model.path)
    draft_path = _argv_value(plan.argv, "--spec-draft-model")
    if draft_path:
        _check_gguf(draft_path, draft=True)
    host = urlsplit(plan.backend_url).hostname or "127.0.0.1"
    _check_port(host, model.port)
