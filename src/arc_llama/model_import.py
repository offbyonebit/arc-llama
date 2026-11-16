"""Read other apps' local model catalogs without modifying their files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from arc_llama.config import Config, ModelConfig
from arc_llama.gguf_meta import check_gguf_header, read_gguf_meta, split_info
from arc_llama.models import add_local_model, find_mmproj, is_auxiliary_gguf


@dataclass
class ImportCandidate:
    name: str
    path: Path
    projector: Path | None = None


def _is_windows() -> bool:
    return sys.platform == "win32"


def default_roots(source: str) -> list[Path]:
    home = Path(os.environ.get("USERPROFILE") or Path.home()) if _is_windows() else Path.home()
    if source == "ollama":
        return [Path(os.environ["OLLAMA_MODELS"]).expanduser()] if os.environ.get("OLLAMA_MODELS") else [home / ".ollama" / "models"]
    return [home / ".lmstudio" / "models", home / ".cache" / "lm-studio" / "models"]


def _single_gguf(path: Path) -> None:
    with path.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError("not a GGUF model")
    split = split_info(path)
    if (split is not None and split[1] > 1) or read_gguf_meta(path).get("split_count", 1) > 1:
        raise ValueError("multi-part GGUF models are not imported")
    check_gguf_header(path)


def _digest_path(root: Path, layer: dict) -> Path:
    digest = layer.get("digest", "")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[a-fA-F0-9]{64}", digest):
        raise ValueError("invalid blob digest")
    return root / "blobs" / digest.lower().replace(":", "-", 1)


def _blob(root: Path, layer: dict) -> Path:
    path = _digest_path(root, layer)
    _single_gguf(path)
    return path.resolve()


def _manifest_files(root: Path, data: dict, depth: int = 0) -> tuple[Path, Path | None]:
    if depth > 4:
        raise ValueError("manifest list nesting exceeds the supported depth")
    if "manifests" in data:
        preferences = {"llamacpp": 0, "ggml": 1}
        children = sorted(data["manifests"], key=lambda child: preferences.get(child.get("runner"), 2))
        for child in children:
            if child.get("format") not in (None, "", "gguf"):
                continue
            try:
                variant = child if "layers" in child or "manifests" in child else json.loads(
                    _digest_path(root, child).read_text(encoding="utf-8")
                )
                return _manifest_files(root, variant, depth + 1)
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                continue
        raise ValueError("no installed single-file GGUF variant in manifest list")
    layers = data["layers"]
    models = [layer for layer in layers if layer.get("mediaType") == "application/vnd.ollama.image.model"]
    projectors = [layer for layer in layers if layer.get("mediaType") == "application/vnd.ollama.image.projector"]
    if len(models) != 1 or len(projectors) > 1:
        raise ValueError("multi-part or non-GGUF model entry")
    return _blob(root, models[0]), _blob(root, projectors[0]) if projectors else None


def discover(source: str, roots: list[Path] | None = None) -> tuple[list[ImportCandidate], list[str]]:
    found: list[ImportCandidate] = []
    skipped = []
    for raw in roots if roots is not None else default_roots(source):
        root = raw.expanduser().resolve()
        if not root.is_dir():
            skipped.append(f"{root}: folder not found")
            continue
        if source == "ollama":
            manifests = root / "manifests"
            if not manifests.is_dir():
                skipped.append(f"{root}: no manifests folder")
                continue
            for manifest in sorted(path for path in manifests.rglob("*") if path.is_file()):
                try:
                    data = json.loads(manifest.read_text(encoding="utf-8"))
                    model, projector = _manifest_files(root, data)
                    parts = list(manifest.relative_to(manifests).parts)
                    if len(parts) < 4:
                        raise ValueError("unrecognized manifest name layout")
                    tag = parts.pop()
                    if parts[0] == "registry.ollama.ai":
                        parts.pop(0)
                        if parts[0] == "library":
                            parts.pop(0)
                    name = "/".join(parts) + ":" + tag
                    found.append(ImportCandidate(name, model, projector))
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                    skipped.append(f"{manifest}: {exc}")
        else:
            for path in sorted(path for path in root.rglob("*") if path.is_file()):
                if path.suffix.lower() != ".gguf":
                    if path.suffix.lower() in {".safetensors", ".bin", ".onnx", ".pt", ".pth"}:
                        skipped.append(f"{path}: non-GGUF model file")
                    continue
                try:
                    _single_gguf(path)
                    if is_auxiliary_gguf(path.name):
                        raise ValueError("auxiliary/projector GGUF, not a standalone model")
                    found.append(ImportCandidate(path.stem, path.resolve(), find_mmproj(path)))
                except (OSError, ValueError) as exc:
                    skipped.append(f"{path}: {exc}")
    return found, skipped


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve()))


def _registration_path(candidate: ImportCandidate, cfg: Config, source: str) -> Path:
    if not _is_windows():
        return candidate.path
    # Linking does not copy weights or need symlink privileges. Cross-volume,
    # permissions and unsupported filesystems fall back to the source reference.
    key = hashlib.sha256(str(candidate.path).encode()).hexdigest()[:12]
    name = re.sub(r"[^a-z0-9._-]+", "-", candidate.name.lower()).strip("-.")[:80] or "model"
    target = Path(cfg.paths.models_dir).expanduser() / "imports" / source / f"{name}-{key}.gguf"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return target if _same_file(target, candidate.path) else candidate.path
        os.link(candidate.path, target)
        return target.resolve()
    except OSError:
        return candidate.path


def import_models(cfg: Config, source: str, candidates: list[ImportCandidate], gpu: str | None = None) -> list[ModelConfig]:
    enabled = [item for item in cfg.gpus if item.enabled]
    if gpu is None:
        if not enabled:
            raise ValueError("No enabled GPU in config; run arc-llama init first")
        gpu = enabled[0].pci_slot
    if cfg.find_gpu(gpu) is None:
        raise ValueError(f"GPU {gpu} is not configured")
    added = []
    for candidate in candidates:
        existing = next((model for model in cfg.models if _same_file(Path(model.path), candidate.path)), None)
        if existing:
            existing.import_source = source
            if candidate.name not in [existing.name, *existing.aliases]:
                existing.aliases.append(candidate.name)
            continue
        base = re.sub(r"[^a-z0-9._-]+", "-", candidate.name.lower()).strip("-._")[:80] or "imported-model"
        name, index = base, 2
        while any(model.name == name for model in cfg.models):
            name, index = f"{base}-{index}", index + 1
        path = _registration_path(candidate, cfg, source)
        model = add_local_model(cfg, name=name, path=str(path), gpu_pci_slot=gpu,
                                display_name=f"{source}: {candidate.name}", aliases=[candidate.name],
                                recipe_overrides={"mmproj": str(candidate.projector)} if candidate.projector else None)
        model.import_source = source
        added.append(model)
    return added
