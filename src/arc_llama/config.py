"""On-disk config schema (TOML) at $XDG_CONFIG_HOME/arc-llama/config.toml.

Schema (v1):

```toml
[server]
host = "127.0.0.1"
port = 11436
single_resident = true   # only one llama-server alive at a time across all GPUs

[paths]
llama_server = "/usr/local/bin/llama-server"
models_dir   = "~/.local/share/arc-llama/models"
state_dir    = "~/.local/state/arc-llama"

[[gpus]]
pci_slot   = "0000:03:00.0"
sycl_index = 0          # passed as ONEAPI_DEVICE_SELECTOR=level_zero:N
arch       = "battlemage"
vram_mb    = 24480
enabled    = true

[[models]]
name             = "qwen3.6-27b"
display_name     = "Qwen 3.6 27B (dense)"
path             = "/mnt/storage/models/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf"
gpu_pci_slot     = "0000:03:00.0"
port             = 8083
kv_class         = "default"
[models.recipe]
ctx              = 131072
cache_type_k     = "q8_0"
cache_type_v     = "q8_0"
n_gpu_layers     = 999
parallel         = 1
extra_flags      = ["--reasoning", "off"]
```
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

if sys.version_info >= (3, 11):
    import tomllib  # noqa: F401
    _toml_load = tomllib.load  # type: ignore[attr-defined]
else:
    import tomli  # type: ignore[import-not-found]
    _toml_load = tomli.load

from arc_llama.arch import Arch
from arc_llama.recipes import KVCacheType, LaunchRecipe

CONFIG_VERSION = 1


def _xdg_config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def _xdg_data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


def _xdg_state_home() -> Path:
    return Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")


def default_config_path() -> Path:
    return _xdg_config_home() / "arc-llama" / "config.toml"


def default_models_dir() -> Path:
    return _xdg_data_home() / "arc-llama" / "models"


def default_state_dir() -> Path:
    return _xdg_state_home() / "arc-llama"


def default_skills_dir() -> Path:
    return _xdg_config_home() / "arc-llama" / "skills"


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 11437
    single_resident: bool = True
    admin_token: str | None = None
    """Bearer token required for destructive admin endpoints (load, stop, edit, scan).
    Set via `arc-llama serve --admin-token` or env var ARC_LLAMA_ADMIN_TOKEN."""
    """Why 11437? Ollama owns 11434 by default, and IPEX-LLM-Ollama installs
    sometimes use 11435/11436. 11437 is the first port in that neighbourhood
    that nobody else seems to claim."""
    """If True, only one llama-server runs at a time. If False, models share VRAM
    on a best-effort basis — set this only if you have generous VRAM headroom."""


@dataclass
class UpstreamConfig:
    """An OpenAI-compatible API endpoint whose models are merged into
    arc-llama's model list and forwarded transparently.

    Models from upstreams are shown in the web UI with an "upstream" source
    tag. When a request targets an upstream model, arc-llama proxies it to
    the upstream instead of starting a local llama-server.
    """
    url: str
    """Base URL of the upstream, e.g. 'http://127.0.0.1:11435' or 'http://192.168.1.50:8080'."""
    name: str = ""
    """Short label for the UI (e.g. 'ollama', 'proxy'). Default: hostname from url."""


@dataclass
class PathsConfig:
    llama_server: str = "llama-server"
    """Path to the llama-server binary. Plain `llama-server` resolves via PATH."""
    models_dir: str = field(default_factory=lambda: str(default_models_dir()))
    state_dir: str = field(default_factory=lambda: str(default_state_dir()))
    skills_dir: str = field(default_factory=lambda: str(default_skills_dir()))
    """Directory containing user skill Python files."""
    scan_paths: list[str] = field(default_factory=list)
    """Extra directories `arc-llama scan` walks looking for GGUFs. The
    `models_dir` is always scanned in addition to these."""


@dataclass
class GPUConfig:
    pci_slot: str
    sycl_index: int
    arch: str  # Arch.value
    vram_mb: int | None = None
    enabled: bool = True
    name: str = ""


@dataclass
class ModelConfig:
    name: str                  # short id, also URL-safe (e.g. "qwen3.6-27b")
    path: str                  # absolute path to the GGUF
    port: int                  # backend port for this model's llama-server
    gpu_pci_slot: str          # which detected GPU to bind to
    display_name: str = ""
    kv_class: str = "default"  # used for VRAM estimation
    recipe: dict[str, Any] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    """Extra strings that should match this model in the OpenAI `model` field."""

    def launch_recipe(self) -> LaunchRecipe:
        r = self.recipe or {}
        return LaunchRecipe(
            n_gpu_layers=int(r.get("n_gpu_layers", 999)),
            ctx=int(r.get("ctx", 8192)),
            parallel=int(r.get("parallel", 1)),
            cache_type_k=KVCacheType(r.get("cache_type_k", "f16")),
            cache_type_v=KVCacheType(r.get("cache_type_v", "f16")),
            threads=r.get("threads"),
            temp=r.get("temp"),
            top_p=r.get("top_p"),
            top_k=r.get("top_k"),
            spec_type=r.get("spec_type"),
            ubatch_size=r.get("ubatch_size"),
            batch_size=r.get("batch_size"),
            flash_attn=r.get("flash_attn"),
            no_mmap=bool(r.get("no_mmap", False)),
            mlock=bool(r.get("mlock", False)),
            extra_flags=list(r.get("extra_flags", [])),
        )


@dataclass
class Config:
    version: int = CONFIG_VERSION
    server: ServerConfig = field(default_factory=ServerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    gpus: list[GPUConfig] = field(default_factory=list)
    models: list[ModelConfig] = field(default_factory=list)
    upstreams: list[UpstreamConfig] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def find_model(self, query: str) -> ModelConfig | None:
        """Match a user-supplied model id against name/display_name/aliases.

        Match is exact-name first, then substring-on-aliases, then case-insensitive
        substring on display_name and basename(path) — so the OpenAI request body's
        `model` field can be the GGUF filename, the short name, or a friendly alias.
        """
        if not query:
            return None
        for m in self.models:
            if m.name == query:
                return m
        for m in self.models:
            if query in m.aliases:
                return m
        ql = query.lower()
        for m in self.models:
            haystacks = [
                m.name.lower(),
                m.display_name.lower(),
                Path(m.path).name.lower(),
                *(a.lower() for a in m.aliases),
            ]
            if any(ql in h for h in haystacks):
                return m
        return None

    def find_gpu(self, pci_slot: str) -> GPUConfig | None:
        for g in self.gpus:
            if g.pci_slot == pci_slot:
                return g
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_toml_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": self.version,
            "server": asdict(self.server),
            "paths": asdict(self.paths),
            "gpus": [asdict(g) for g in self.gpus],
            "models": [asdict(m) for m in self.models],
            "upstreams": [asdict(u) for u in self.upstreams],
        }
        return _strip_none(d)

    def save(self, path: Path | None = None) -> Path:
        path = path or default_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            tomli_w.dump(self.to_toml_dict(), f)
        return path


def _strip_none(obj: Any) -> Any:
    """Recursively remove dict keys whose value is None.

    TOML has no null type, so None values crash tomli_w. This is applied
    to the whole config tree before persistence.
    """
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj]
    return obj


def load_config(path: Path | None = None) -> Config:
    path = path or default_config_path()
    if not path.exists():
        return Config()
    with open(path, "rb") as f:
        raw = _toml_load(f)
    return Config(
        version=int(raw.get("version", CONFIG_VERSION)),
        server=ServerConfig(**raw.get("server", {})),
        paths=PathsConfig(**raw.get("paths", {})),
        gpus=[GPUConfig(**g) for g in raw.get("gpus", [])],
        models=[ModelConfig(**m) for m in raw.get("models", [])],
        upstreams=[UpstreamConfig(**u) for u in raw.get("upstreams", [])],
    )


def init_config_from_detection(detected_gpus, llama_server_path: str | None = None) -> Config:
    """Build a fresh Config from a detect.detect_gpus() result."""
    cfg = Config()
    if llama_server_path:
        cfg.paths.llama_server = llama_server_path
    enabled_set = False
    for g in detected_gpus:
        gc = GPUConfig(
            pci_slot=g.pci_slot,
            sycl_index=g.sycl_index_hint,
            arch=g.arch.value if hasattr(g.arch, "value") else str(g.arch),
            vram_mb=g.vram_mb,
            enabled=False,
            name=g.name,
        )
        # Prefer the highest-VRAM Battlemage / Alchemist as the default GPU.
        if not enabled_set and g.arch in (Arch.BATTLEMAGE, Arch.ALCHEMIST) and g.vram_mb:
            gc.enabled = True
            enabled_set = True
        cfg.gpus.append(gc)
    if cfg.gpus and not enabled_set:
        cfg.gpus[0].enabled = True
    return cfg
