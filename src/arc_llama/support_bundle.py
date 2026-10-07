"""Create small, privacy-conscious diagnostic bundles."""

from __future__ import annotations

import platform
import re
import sys
import zipfile
from pathlib import Path

from arc_llama import __version__
from arc_llama.config import Config
from arc_llama.detect import DetectedGPU

_SECRET_LINE = re.compile(
    r"(?im)^(\s*(?:admin_token|hf_token|token|password|secret|api_key)\s*=\s*).*$"
)
_TOKEN_QUERY = re.compile(r"([?&](?:token|api[_-]?key|password|secret)=)[^&\s]+", re.I)


def _safe_text(value: str) -> str:
    """Remove common credential-bearing query values and local home paths."""
    home = str(Path.home())
    # Config files can carry either slash style regardless of the current OS.
    # Redact equivalent home spellings so a Windows support bundle also hides
    # a path copied in POSIX form (and vice versa).
    home_variants = {home, home.replace("\\", "/"), home.replace("/", "\\")}
    flags = re.IGNORECASE if sys.platform == "win32" else 0
    for variant in sorted(home_variants, key=len, reverse=True):
        value = re.sub(re.escape(variant), "$HOME", value, flags=flags)
    return _TOKEN_QUERY.sub(r"\1[REDACTED]", value)


def redact_config(text: str) -> str:
    """Redact credential-shaped TOML assignments before bundling."""
    return _SECRET_LINE.sub(r"\1\"[REDACTED]\"", _safe_text(text))


def _gpu_text(gpus: list[DetectedGPU]) -> str:
    lines = []
    for gpu in gpus:
        lines.append(
            " | ".join(
                (
                    f"name={gpu.name}",
                    f"pci={gpu.pci_slot}",
                    f"device_id=0x{gpu.device_id:04x}",
                    f"arch={gpu.arch.value}",
                    f"driver={gpu.driver or 'unknown'}",
                    f"vram_mb={gpu.vram_mb or 'unknown'}",
                )
            )
        )
    return "\n".join(lines) if lines else "No Intel GPU detected."


def _model_text(cfg: Config | None) -> str:
    if cfg is None or not cfg.models:
        return "No registered models."
    lines = []
    for model in cfg.models:
        path = _safe_text(model.path)
        try:
            size = str(Path(model.path).expanduser().stat().st_size)
        except OSError:
            size = "missing"
        lines.append(f"name={model.name} | path={path} | size_bytes={size} | port={model.port}")
    return "\n".join(lines)


def create_support_bundle(
    output: Path,
    *,
    config_path: Path,
    cfg: Config | None,
    gpus: list[DetectedGPU],
) -> Path:
    """Write a diagnostic zip without model files, secrets, or environment dumps."""
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    config_text = "config file not found\n"
    if config_path.exists():
        try:
            config_text = redact_config(config_path.read_text(encoding="utf-8"))
        except OSError as exc:
            config_text = f"could not read config: {exc}\n"
    manifest = "\n".join(
        (
            f"arc-llama={__version__}",
            f"python={sys.version.split()[0]}",
            f"platform={platform.platform()}",
            "contents=config.toml (redacted), hardware.txt, models.txt",
            "model files, environment variables, and credentials are excluded",
        )
    ) + "\n"
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.txt", manifest)
        archive.writestr("config.toml", config_text)
        archive.writestr("hardware.txt", _gpu_text(gpus) + "\n")
        archive.writestr("models.txt", _model_text(cfg) + "\n")
    return output
