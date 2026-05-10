"""Model registry — adding, removing, and (optionally) downloading GGUFs from HF.

The downloader is intentionally a thin shim around `huggingface_hub.hf_hub_download`
so users who already have models on disk never need network access.

Discovery (`discover_ggufs` / `register_discovered`) is the plug-and-play
entrypoint: walk a few directories, infer reasonable recipes from filename
heuristics, and register everything found. Users on a fresh box should never
need to type `arc-llama add` for a GGUF they already have on disk.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arc_llama.config import (
    Config,
    ModelConfig,
)
from arc_llama.recipes import default_recipe

log = logging.getLogger("arc_llama.models")

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
HF_SPEC_RE = re.compile(
    r"^(?P<repo>[^@:\s/]+/[^@:\s/]+)(?::(?P<file>[^@\s]+))?$"
)


@dataclass
class HFModelSpec:
    """Parsed user input like `unsloth/gemma-4-31B-it-GGUF:Q4_K_M`."""
    repo: str
    file: str | None  # exact filename; if None, we glob the repo for a match
    quant: str | None # short hint like "Q4_K_M", used when file is None


def parse_hf_spec(spec: str) -> HFModelSpec:
    m = HF_SPEC_RE.match(spec)
    if not m:
        raise ValueError(
            f"Invalid HF spec '{spec}'. Expected 'org/repo' or 'org/repo:filename' "
            f"or 'org/repo:Q4_K_M'."
        )
    repo = m.group("repo")
    file = m.group("file")
    quant = None
    if file and not file.endswith(".gguf"):
        # Treat short tokens like Q4_K_M as a quant hint, not a filename.
        if re.fullmatch(r"(IQ|Q|UD-)?[A-Z0-9_]+", file):
            quant = file
            file = None
    return HFModelSpec(repo=repo, file=file, quant=quant)


def _next_free_port(used: set[int], start: int = 18080) -> int:
    p = start
    while p in used:
        p += 1
    return p


def _short_name_from(repo: str, file: str | None) -> str:
    """Generate a short, slug-friendly name from an HF repo/filename."""
    base = repo.split("/")[-1].lower()
    # Strip common GGUF suffixes
    for suffix in ("-gguf", "_gguf"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    base = re.sub(r"[^a-z0-9._-]+", "-", base).strip("-")
    if file:
        # Append the quant tier if obvious from the filename
        m = re.search(r"(IQ\d[A-Z_]*|Q\d[A-Z_]*|UD-[A-Z0-9_]+)", file, re.IGNORECASE)
        if m:
            base = f"{base}-{m.group(1).lower()}"
    return base or "model"


def add_local_model(
    cfg: Config,
    *,
    name: str,
    path: str,
    gpu_pci_slot: str,
    port: int | None = None,
    display_name: str = "",
    kv_class: str = "default",
    aliases: list[str] | None = None,
    recipe_overrides: dict[str, Any] | None = None,
) -> ModelConfig:
    """Register an already-downloaded GGUF in the config.

    Picks a recipe based on the bound GPU's arch and VRAM, then applies any overrides.
    """
    if not NAME_RE.match(name):
        raise ValueError(
            f"Model name '{name}' must match [a-z0-9][a-z0-9._-]*"
        )
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    gpu = cfg.find_gpu(gpu_pci_slot)
    if gpu is None:
        raise ValueError(f"GPU {gpu_pci_slot} not in config — run `arc-llama init` first.")
    used_ports = {m.port for m in cfg.models}
    port = port or _next_free_port(used_ports)
    if port in used_ports:
        raise ValueError(f"Port {port} already in use by another model.")
    if any(m.name == name for m in cfg.models):
        raise ValueError(f"Model name '{name}' already registered.")
    # Build a recipe that fits this GPU.
    from arc_llama.arch import Arch
    arch = Arch(gpu.arch) if gpu.arch else Arch.UNKNOWN
    recipe = default_recipe(
        arch=arch,
        vram_mb=gpu.vram_mb or 8192,
        model_file_mb=p.stat().st_size // (1024 * 1024),
        kv_class=kv_class,
    )
    recipe_dict: dict[str, Any] = {
        "n_gpu_layers": recipe.n_gpu_layers,
        "ctx": recipe.ctx,
        "parallel": recipe.parallel,
        "cache_type_k": recipe.cache_type_k.value,
        "cache_type_v": recipe.cache_type_v.value,
    }
    if recipe_overrides:
        recipe_dict.update(recipe_overrides)
    mc = ModelConfig(
        name=name,
        path=str(p),
        port=port,
        gpu_pci_slot=gpu_pci_slot,
        display_name=display_name or name,
        kv_class=kv_class,
        recipe=recipe_dict,
        aliases=aliases or [p.name],
    )
    cfg.models.append(mc)
    return mc


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

# Filename → kv_class hints, evaluated in order. First match wins.
_KV_CLASS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"gemma[\W_-]*[34]", re.IGNORECASE), "gemma_swa"),
    (re.compile(r"qwen[\W_-]*3[.\W_-]*6?[\W_-]*27b(?!.*a3b)", re.IGNORECASE), "qwen3_27b_dense"),
    (re.compile(r"(qwen[\W_-]*3.*a3b|qwen[\W_-]*3.*moe|carnice|huihui.*30b.*a3b)", re.IGNORECASE), "moe_a3b"),
]

# Common quant tier markers that look good in display names.
_QUANT_TIER_RE = re.compile(
    r"\b(IQ\d_[A-Z_]*|Q\d_[KS]_[A-Z]+|Q\d_K|Q\d_\d|Q8_0|UD-[A-Z0-9_]+)\b",
    re.IGNORECASE,
)


def infer_kv_class(filename: str) -> str:
    """Guess the kv_class for VRAM estimation from the GGUF filename."""
    for pattern, kv_class in _KV_CLASS_PATTERNS:
        if pattern.search(filename):
            return kv_class
    return "default"


def short_name_from_path(path: Path, used: set[str]) -> str:
    """Generate a unique [a-z0-9._-]+ slug for a discovered GGUF.

    We prefer the file *stem* (e.g. `Qwen3.6-27B-Q4_K_M`) over the parent
    directory name — stems are more descriptive and survive the common case
    where multiple GGUFs share a directory like `/mnt/storage/models/`.
    """
    base = re.sub(r"[^a-z0-9._-]+", "-", path.stem.lower()).strip(".-")
    # Strip noise suffixes Unsloth/quanters tend to bolt on.
    base = re.sub(r"[._-](gguf|imatrix)$", "", base)
    if not base or not NAME_RE.match(base):
        # Last-ditch fallback — parent dir + stem.
        base = re.sub(r"[^a-z0-9._-]+", "-",
                      f"{path.parent.name.lower()}-{path.stem.lower()}").strip("-")
    if not base:
        base = "model"
    if base not in used:
        return base
    n = 2
    while f"{base}-{n}" in used:
        n += 1
    return f"{base}-{n}"


def infer_display_name(path: Path) -> str:
    """A friendlier display name from a GGUF filename — stem + quant tier."""
    stem = path.stem
    return stem.replace("_", " ").replace("-", " ").strip()


def _resolve_scan_paths(cfg: Config, extra: list[Path] | None = None) -> list[Path]:
    """Build the deduped, existing list of dirs to walk for GGUFs."""
    paths: list[Path] = []
    seen: set[Path] = set()
    candidates = [Path(cfg.paths.models_dir).expanduser()]
    candidates.extend(Path(p).expanduser() for p in cfg.paths.scan_paths)
    if extra:
        candidates.extend(Path(p).expanduser() for p in extra)
    for c in candidates:
        try:
            r = c.resolve()
        except OSError:
            continue
        if r in seen or not r.is_dir():
            continue
        seen.add(r)
        paths.append(r)
    return paths


def discover_ggufs(
    cfg: Config, extra_paths: list[Path] | None = None, max_depth: int = 4
) -> list[Path]:
    """Walk the configured + extra scan paths and return every *.gguf file found.

    Hidden dirs and symlinked dirs are skipped to avoid loops. Bounded depth
    keeps a stray scan of `/` from running forever.
    """
    found: dict[Path, None] = {}
    for root in _resolve_scan_paths(cfg, extra_paths):
        for path in _walk_for_ggufs(root, max_depth):
            found[path] = None
    return list(found.keys())


def _walk_for_ggufs(root: Path, max_depth: int) -> list[Path]:
    out: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        cur, depth = stack.pop()
        try:
            entries = list(cur.iterdir())
        except (OSError, PermissionError):
            continue
        for e in entries:
            name = e.name
            if name.startswith("."):
                continue
            try:
                if e.is_symlink():
                    continue
                if e.is_dir() and depth < max_depth:
                    stack.append((e, depth + 1))
                elif e.is_file() and name.endswith(".gguf"):
                    out.append(e.resolve())
            except OSError:
                continue
    return out


def register_discovered(
    cfg: Config,
    paths: list[Path],
    *,
    gpu_pci_slot: str | None = None,
    port_start: int = 18080,
) -> list[ModelConfig]:
    """Auto-register every newly-found GGUF in `paths` against the cfg.

    Skips files already registered (by absolute path). Picks a recipe via
    `default_recipe(arch, vram, file_size, kv_class)` so context length is
    sized to the bound GPU's VRAM. Returns the list of newly-added entries.
    """
    if not cfg.gpus:
        raise ValueError("No GPUs in config — run `arc-llama init` first.")
    if gpu_pci_slot is None:
        enabled = next((g for g in cfg.gpus if g.enabled), None)
        if enabled is None:
            enabled = cfg.gpus[0]
        gpu_pci_slot = enabled.pci_slot
    gpu = cfg.find_gpu(gpu_pci_slot)
    if gpu is None:
        raise ValueError(f"Unknown GPU: {gpu_pci_slot}")
    from arc_llama.arch import Arch
    arch = Arch(gpu.arch) if gpu.arch else Arch.UNKNOWN
    existing_paths = {Path(m.path).resolve() for m in cfg.models}
    used_names = {m.name for m in cfg.models}
    used_ports = {m.port for m in cfg.models}
    added: list[ModelConfig] = []
    for p in paths:
        try:
            rp = p.resolve()
        except OSError:
            continue
        if rp in existing_paths:
            continue
        if not rp.exists():
            continue
        kv_class = infer_kv_class(rp.name)
        recipe = default_recipe(
            arch=arch,
            vram_mb=gpu.vram_mb or 8192,
            model_file_mb=rp.stat().st_size // (1024 * 1024),
            kv_class=kv_class,
        )
        name = short_name_from_path(rp, used_names)
        used_names.add(name)
        port = port_start
        while port in used_ports:
            port += 1
        used_ports.add(port)
        mc = ModelConfig(
            name=name,
            path=str(rp),
            port=port,
            gpu_pci_slot=gpu_pci_slot,
            display_name=infer_display_name(rp),
            kv_class=kv_class,
            recipe={
                "n_gpu_layers": recipe.n_gpu_layers,
                "ctx": recipe.ctx,
                "parallel": recipe.parallel,
                "cache_type_k": recipe.cache_type_k.value,
                "cache_type_v": recipe.cache_type_v.value,
            },
            aliases=[rp.name],
        )
        cfg.models.append(mc)
        added.append(mc)
        existing_paths.add(rp)
        log.info(
            "discovered %s → %s (kv=%s, ctx=%d, port=%d)",
            rp.name, name, kv_class, recipe.ctx, port,
        )
    return added


def download_from_hf(
    spec: HFModelSpec,
    *,
    target_dir: Path,
    token: str | None = None,
    progress: bool = True,
) -> Path:
    """Resolve a HFModelSpec to a concrete file path under `target_dir`.

    Imported lazily so users without huggingface-hub can still use arc-llama
    with already-downloaded files.
    """
    target_dir = Path(target_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface-hub is required for downloads. "
            "Install with `pip install huggingface-hub` or pre-download the GGUF "
            "and use `arc-llama add --path /path/to/model.gguf`."
        ) from e

    file = spec.file
    if file is None:
        api = HfApi(token=token)
        files = [
            f for f in api.list_repo_files(spec.repo)
            if f.endswith(".gguf")
        ]
        if spec.quant:
            ql = spec.quant.lower()
            matches = [f for f in files if ql in f.lower()]
            if not matches:
                raise FileNotFoundError(
                    f"No GGUF in {spec.repo} matched quant hint '{spec.quant}'. "
                    f"Available: {', '.join(sorted(files))}"
                )
            # Prefer uniform quants (no "_xl" / "ud-") if multiple matched.
            uniform = [f for f in matches if "_xl" not in f.lower() and "ud-" not in f.lower()]
            file = sorted(uniform or matches)[0]
        elif len(files) == 1:
            file = files[0]
        else:
            raise ValueError(
                f"Repo {spec.repo} has {len(files)} GGUF files; specify one with "
                f"`{spec.repo}:<filename>` or `{spec.repo}:Q4_K_M`."
            )
    return Path(hf_hub_download(
        repo_id=spec.repo,
        filename=file,
        local_dir=str(target_dir),
        token=token,
    ))
