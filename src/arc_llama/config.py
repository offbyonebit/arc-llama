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

import logging
import os
import secrets
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import tomli_w

if sys.version_info >= (3, 11):
    import tomllib  # noqa: F401

    _toml_load = tomllib.load  # type: ignore[attr-defined]
else:
    import tomli  # type: ignore[import-not-found]

    _toml_load = tomli.load

from arc_llama.arch import Arch, Backend
from arc_llama.recipes import KVCacheType, LaunchRecipe

CONFIG_VERSION = 2

_save_locks: dict[str, threading.Lock] = {}


def _xdg_config_home() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def _xdg_data_home() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


def _xdg_state_home() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
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
    tts_python: str = ""
    """Interpreter used to run a Python-based TTS backend (e.g. OmniVoice).

    OmniVoice pulls in torch, transformers and torchaudio; arc-llama depends on
    none of those and has no business forcing them into its own environment. So
    a speech engine runs out of its own virtualenv and arc-llama only needs to
    know which interpreter that is — `~/git/OmniVoice/.venv/bin/python`, say.

    Empty means "use the interpreter running arc-llama", which is correct only
    when the two happen to share an environment. A model's
    `recipe.python` overrides this per entry.
    """
    models_dir: str = field(default_factory=lambda: str(default_models_dir()))
    state_dir: str = field(default_factory=lambda: str(default_state_dir()))
    skills_dir: str = field(default_factory=lambda: str(default_skills_dir()))
    """Directory containing user skill Python files."""
    scan_paths: list[str] = field(default_factory=list)
    """Extra directories `arc-llama scan` walks looking for GGUFs. The
    `models_dir` is always scanned in addition to these."""
    oneapi_setvars: str = ""
    """Path to Intel oneAPI's setvars script. When set and a SYCL llama-server
    is launched in an environment that lacks the oneAPI runtime libraries,
    arc-llama sources this script and merges the resulting variables into the
    subprocess environment. Leave empty to auto-detect."""


@dataclass
class TuneConfig:
    """Background auto-tuning policy.

    Tuning is an idle-time, single-model sweep. These knobs decide whether the
    auto-tuner runs at all, how quiet the router has to be before it starts, and
    the benchmark shape it uses.
    """

    auto: bool = True
    """Run background sweeps when models are idle and eligible."""
    idle_seconds: int = 120
    """Seconds of router inactivity required before a sweep may start."""
    target: str = "balanced"
    """Target balance passed to tune_model: balanced, generation, or prompt."""
    prompt_tokens: int = 1024
    gen_tokens: int = 128
    min_uses: int = 1
    """Requests a model must serve before it becomes eligible for auto-tune."""
    retune_on_fingerprint_change: bool = True
    """Treat a fingerprint mismatch as a fresh untuned state."""


@dataclass
class WorkloadConfig:
    """Declared usage profile, gathered by `arc-llama init`.

    Every field may be empty ("not sure" / never asked), in which case the
    tuner behaves exactly as if no profile existed. The answers steer what the
    auto-tuner measures: which context depth rankings are taken at, which KV
    types are even eligible, and how prompt-eval vs generation is weighted.
    """

    context_length: str = ""
    """"" | short (<8k) | long (~32k) | very_long (100k+)."""
    style: str = ""
    """"" | agentic (tool-calling loops) | conversational (chat)."""
    priority: str = ""
    """"" | first_token (time to first token) | throughput (steady-state tok/s)."""


@dataclass
class AgentConfig:
    root: str = "."
    """Default filesystem root for the agent file/shell tools.

    Request-level `root` overrides this. Use an absolute path if you access
    arc-llama from another machine and `.` (the server's working directory)
    is not what you want.
    """
    profile: str | None = None
    """Default profile name for selecting which MCP servers are active.

    A profile references MCP servers by name. When a profile is active,
    only the MCP servers listed in that profile are loaded. If unset, all
    configured MCP servers are loaded.
    """


@dataclass
class ProfileConfig:
    """Named whitelist of MCP servers."""

    name: str = ""
    mcp_servers: list[str] = field(default_factory=list)


@dataclass
class MCPServerConfig:
    """Configuration for one stdio MCP server."""

    name: str = ""
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class GPUConfig:
    pci_slot: str
    sycl_index: int
    arch: str  # Arch.value
    vram_mb: int | None = None
    enabled: bool = True
    name: str = ""
    backend: str = Backend.SYCL.value  # Backend.value
    vulkan_index: int | None = None
    """Vulkan device index, which is NOT the same number as sycl_index.

    SYCL/Level-Zero enumerates Intel devices only, so sycl_index 0 is the first
    Arc card. Vulkan enumerates every vendor, so on a machine with a discrete
    NVIDIA or AMD card the Arc may be Vulkan1 while sycl_index is still 0.
    Resolved from `llama-server --list-devices`; set it by hand to override.
    """


@dataclass
class ModelConfig:
    name: str  # short id, also URL-safe (e.g. "qwen3.6-27b")
    path: str  # absolute path to the GGUF
    port: int  # backend port for this model's llama-server
    gpu_pci_slot: str  # which detected GPU to bind to
    display_name: str = ""
    kv_class: str = "default"  # used for VRAM estimation
    recipe: dict[str, Any] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    """Extra strings that should match this model in the OpenAI `model` field."""
    tune_state: str = "untuned"  # untuned | tuned | failed | skipped
    tuned_at: float | None = None
    tune_fingerprint: str = ""
    tune_error: str = ""

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
            spec_draft_n_max=(
                int(r["spec_draft_n_max"]) if r.get("spec_draft_n_max") is not None else None
            ),
            spec_draft_model=r.get("spec_draft_model"),
            spec_draft_ngl=(
                int(r["spec_draft_ngl"]) if r.get("spec_draft_ngl") is not None else None
            ),
            ubatch_size=r.get("ubatch_size"),
            batch_size=r.get("batch_size"),
            flash_attn=r.get("flash_attn"),
            no_mmap=bool(r.get("no_mmap", False)),
            mlock=bool(r.get("mlock", False)),
            n_cpu_moe=r.get("n_cpu_moe"),
            override_tensor=list(r.get("override_tensor", []))
            if r.get("override_tensor")
            else None,
            extra_flags=list(r.get("extra_flags", [])),
        )


AUDIO_ENGINE_LLAMACPP = "llamacpp"

ASR_ENGINES = (AUDIO_ENGINE_LLAMACPP,)
"""Runtimes that can serve `/v1/audio/transcriptions`.

Only llama-server. It is the binary an Arc box already has, it is the only one
with a SYCL build, and it inherits the arch env profiles and the device
selector. TTS engines are not listed here: they are discovered from
``arc_llama.tts``, which config cannot import without a cycle and does not need
to — nothing in this module dispatches on an engine name.
"""

DEFAULT_ASR_CTX = 4096
"""Context length for a transcription model when the recipe doesn't say.

This must be set explicitly, because llama-server's `-c` default is `0`,
meaning "whatever the GGUF was trained for" — and Qwen3-ASR advertises 65536.
That sizes a ~7 GB KV cache for a 1.7B model whose weights are under 2 GB,
which looks like a runaway leak and is really just an unasked-for context.
A single utterance is a few hundred audio tokens plus its transcript, so 4096
covers minutes of speech; raise it in the recipe for long-form dictation.
"""


_AUDIO_RECIPE_KEYS = (
    "mmproj",
    "ctx",
    "n_gpu_layers",
    "cache_type_k",
    "cache_type_v",
    "threads",
    "extra_flags",
    "python",
    "device",
    "dtype",
    "default_voice",
    "default_language",
    "default_response_format",
    "options",
)


@dataclass
class AudioRecipe:
    """Resolved launch knobs for one audio model.

    Engine-specific fields are inert for the other engine rather than an
    error: swapping `engine` on an existing entry should not mean rewriting
    the recipe from scratch.
    """

    mmproj: str = ""
    ctx: int = DEFAULT_ASR_CTX
    n_gpu_layers: int = 999
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    threads: int = 1
    extra_flags: list[str] = field(default_factory=list)

    # -- TTS --------------------------------------------------------
    python: str = ""
    """Interpreter for a Python TTS backend, overriding `paths.tts_python`."""
    device: str = ""
    """Compute device as the engine names it (`xpu`, `cuda:0`, `cpu`).

    Empty lets the engine choose from the GPU's configured backend, which is
    `xpu` for the SYCL cards this project exists for.
    """
    dtype: str = ""
    """Weight dtype for a torch-based engine (e.g. `float16`, `bfloat16`).

    Empty lets the engine pick, which it needs to do: a quantized checkpoint
    dictates the dtype of the base model it was derived from, and getting that
    wrong is a dtype mismatch on the first matmul rather than a slow path.
    """
    default_voice: str = ""
    """Voice used when a request's `voice` field matches nothing registered."""
    default_language: str = ""
    """Language used when a request does not say (e.g. `English`)."""
    default_response_format: str = "mp3"
    """Encoding used when a request omits `response_format`, matching OpenAI."""
    options: dict[str, Any] = field(default_factory=dict)
    """Engine-specific knobs, passed through untouched.

    Anything only one engine understands lives here rather than becoming a
    field: OmniVoice's `num_step` and `normalize_text` mean nothing to a
    `llama-tts` backend, and a shared dataclass that grows a field per engine
    stops being a shared dataclass. Adding an engine should not need an edit
    to this file.
    """


@dataclass
class AudioModelConfig:
    """One audio model, served by its own backend subprocess.

    Covers both directions, because they share everything except the endpoint
    they answer on: the same registry, ports, GPU binding, launch/health/evict
    lifecycle and VRAM accounting.

      * **`task = "asr"`** — speech to text on `/v1/audio/transcriptions`,
        always `engine = "llamacpp"`: `llama-server -m model.gguf --mmproj
        proj.gguf`. That is the binary an Arc box already has, it is the only
        transcription runtime with a **SYCL** build, and it inherits the arch
        env profiles and the device selector.
      * **`task = "tts"`** — text to speech on `/v1/audio/speech`, served by
        whichever engine is named in ``engine``. Those come from
        ``arc_llama.tts``, which owns both how the backend is launched and how
        an OpenAI request is translated for it; `omnivoice` is the one shipped
        today.

    Deliberately not a ``ModelConfig``: the tuner's whole surface (KV-cache
    sweeps, ctx-vs-VRAM recipes, benchmark prompts) is meaningless for a
    transcription model, and giving audio models fields the tuner reads would
    let a sweep pick up something it cannot benchmark. Keeping them in their
    own table also stops `arc-llama scan` from ever treating one as an LLM.
    """

    name: str  # short id, also URL-safe (e.g. "qwen3-asr")
    path: str  # model directory or .gguf file
    port: int  # backend port for this model's backend process
    gpu_pci_slot: str  # which detected GPU to bind to
    engine: str = "llamacpp"
    """Which runtime serves this model.

    `llamacpp` for `task = "asr"`; for `task = "tts"` it is the name of a
    registered TTS engine (see ``arc_llama.tts``), e.g. `omnivoice`.
    """
    task: str = "asr"
    """`asr` or `tts`. Decides which OpenAI endpoint routes here."""
    mode: str = "offline"
    """`offline` or `streaming`. Streaming is required for `stream=true`
    transcriptions, and only for models whose backend can produce incremental
    deltas."""
    recipe: dict[str, Any] = field(default_factory=dict)
    """How to launch this model, in the same place `[models.recipe]` keeps it.

    Everything that shapes the process goes here — `mmproj`, `ctx`,
    `n_gpu_layers`, `cache_type_k`/`cache_type_v`, `extra_flags` for ASR;
    `python`, `device`, `dtype`, the `default_*` request fallbacks and an
    `options` bag for TTS. The body above stays identity and routing policy,
    so the two model tables read the same way. See ``audio_recipe`` for the
    defaults.
    """
    strip_asr_markers: bool = True
    """Strip Qwen3-ASR's native output framing from the transcript.

    Qwen3-ASR emits `language English<asr_text>the actual words`, and
    llama.cpp forwards it verbatim (ggml-org/llama.cpp#26749, still open).
    A Home Assistant voice pipeline then tries to match that prefix as part
    of the command and fails. Stripping is on by default and is a no-op for
    any model that does not emit the marker.
    """
    display_name: str = ""
    aliases: list[str] = field(default_factory=list)
    """Extra strings that should match this model in the OpenAI `model` field.
    Register `whisper-1` here if a client hardcodes OpenAI's STT model id."""
    always_resident: bool = True
    """Exempt this model from single-resident eviction.

    An ASR model is small (Qwen3-ASR-0.6B q8 is well under 1 GB) and is used
    in short bursts between LLM turns. Letting it evict a 20 GB LLM — which
    then pays a full cold start on the next reply — trades seconds of VRAM
    saving for tens of seconds of latency on every utterance. Its footprint
    still counts against the GPU budget when an LLM load is checked for fit.
    """
    vram_mb: int | None = None
    """Declared VRAM footprint, overriding the load-time fit guard's estimate.

    A TTS model has no llama.cpp-style tensor table to measure, and a
    safetensors directory's on-disk size is a poor proxy once weights are
    requantized at load. None means "estimate from the path and the recipe".
    """

    def audio_recipe(self) -> AudioRecipe:
        """The launch recipe with defaults filled in."""
        r = self.recipe or {}
        return AudioRecipe(
            mmproj=str(r.get("mmproj", "")),
            ctx=int(r.get("ctx", DEFAULT_ASR_CTX)),
            n_gpu_layers=int(r.get("n_gpu_layers", 999)),
            cache_type_k=str(r.get("cache_type_k", "f16")),
            cache_type_v=str(r.get("cache_type_v", "f16")),
            threads=int(r.get("threads", 1)),
            extra_flags=list(r.get("extra_flags", [])),
            python=str(r.get("python", "")),
            device=str(r.get("device", "")),
            dtype=str(r.get("dtype", "")),
            default_voice=str(r.get("default_voice", "")),
            default_language=str(r.get("default_language", "")),
            default_response_format=str(r.get("default_response_format", "mp3")),
            options=dict(r.get("options", {})),
        )


@dataclass
class VoiceConfig:
    """A named voice, resolvable from the OpenAI `voice` request field.

    Kept in its own top-level table rather than nested under a model, because a
    voice is a property of the speaker and not of the runtime: the same
    reference clip should still name the same voice after switching engines,
    and a client that says `voice = "glados"` should not have to know which
    backend is loaded. ``models`` narrows it when that is not true.

    Which fields are set decides the synthesis mode. `ref_audio` (with or
    without `ref_text`) clones; `instruct` alone designs a voice from
    attributes; neither lets the model pick one itself.
    """

    name: str
    ref_audio: str = ""
    """Reference clip to clone, 3–10 s of clean speech."""
    ref_text: str = ""
    """Transcript of `ref_audio`. Empty makes the engine transcribe it with
    Whisper on first use, which costs a second model on the GPU — so supplying
    it is worth the typing."""
    instruct: str = ""
    """Voice-design attributes, e.g. `female, low pitch, british accent`.
    Ignored when `ref_audio` is set: cloning already fixes the speaker."""
    language: str = ""
    """Language this voice is meant to speak, e.g. `English`."""
    prompt_file: str = ""
    """Where the engine caches the encoded reference.

    Encoding a reference clip is not free and its result never changes, so the
    first use writes it here and later starts load it back instead of decoding
    (and possibly re-transcribing) the audio again. Empty means the engine
    picks a path under the state dir.
    """
    models: list[str] = field(default_factory=list)
    """TTS model names this voice applies to. Empty means all of them."""
    display_name: str = ""
    aliases: list[str] = field(default_factory=list)
    """Extra strings that resolve to this voice. Register `alloy` here for
    clients that hardcode one of OpenAI's voice ids."""


@dataclass
class Config:
    version: int = CONFIG_VERSION
    server: ServerConfig = field(default_factory=ServerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    gpus: list[GPUConfig] = field(default_factory=list)
    models: list[ModelConfig] = field(default_factory=list)
    audio_models: list[AudioModelConfig] = field(default_factory=list)
    voices: list[VoiceConfig] = field(default_factory=list)
    upstreams: list[UpstreamConfig] = field(default_factory=list)
    mcp_servers: list[MCPServerConfig] = field(default_factory=list)
    profiles: list[ProfileConfig] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def find_profile(self, name: str | None) -> ProfileConfig | None:
        if not name:
            return None
        for p in self.profiles:
            if p.name == name:
                return p
        return None

    def active_profile_name(self, profile_name: str | None = None) -> str | None:
        """Return the effective profile name.

        Explicit ``profile_name`` wins, then ``agent.profile`` in config,
        then None (meaning all MCP servers are active).
        """
        if profile_name:
            return profile_name
        if self.agent.profile:
            return self.agent.profile
        return None

    def active_mcp_servers(self, profile_name: str | None = None) -> list[MCPServerConfig]:
        """Return MCP servers that belong to the active profile.

        If no profile is active, all configured servers are returned. Unknown
        server names listed in a profile are ignored with a warning.
        """
        name = self.active_profile_name(profile_name)
        if not name:
            return list(self.mcp_servers)
        profile = self.find_profile(name)
        if profile is None:
            logging.getLogger("arc_llama.config").warning(
                "Profile %r not found; loading all MCP servers", name
            )
            return list(self.mcp_servers)
        allowed = set(profile.mcp_servers)
        found: dict[str, MCPServerConfig] = {}
        for server in self.mcp_servers:
            if server.name in allowed:
                found[server.name] = server
        for missing in allowed - set(found):
            logging.getLogger("arc_llama.config").warning(
                "Profile %r references unknown MCP server %r", name, missing
            )
        return [found[name] for name in profile.mcp_servers if name in found]

    def find_model(self, query: str) -> ModelConfig | None:
        """Match a user-supplied model id against name/display_name/aliases.

        Match is exact-name first, then substring-on-aliases, then case-insensitive
        substring on display_name and basename(path) — so the OpenAI request body's
        `model` field can be the GGUF filename, the short name, or a friendly alias.
        """
        return _find_entry(self.models, query)

    def find_audio_model(self, query: str) -> AudioModelConfig | None:
        """Same matching rules as ``find_model``, over the audio registry."""
        return _find_entry(self.audio_models, query)

    def find_voice(self, query: str) -> VoiceConfig | None:
        """Match a `voice` request field against name/display_name/aliases."""
        if not query:
            return None
        for v in self.voices:
            if v.name == query:
                return v
        ql = query.lower()
        for v in self.voices:
            if ql == v.name.lower() or any(ql == a.lower() for a in v.aliases):
                return v
        return None

    def voices_for(self, model_name: str) -> list[VoiceConfig]:
        """Voices usable by the TTS model called *model_name*."""
        return [v for v in self.voices if not v.models or model_name in v.models]

    def find_any_model(self, query: str) -> ModelConfig | AudioModelConfig | None:
        """Resolve *query* against both registries, LLMs first.

        Names are unique across the two tables (registration refuses a
        collision), so the order only decides which one wins a loose
        substring match — and an LLM is the commoner intent.
        """
        return self.find_model(query) or self.find_audio_model(query)

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
            "tune": asdict(self.tune),
            "workload": asdict(self.workload),
            "agent": asdict(self.agent),
            "gpus": [asdict(g) for g in self.gpus],
            "models": [asdict(m) for m in self.models],
            "audio_models": [asdict(m) for m in self.audio_models],
            "voices": [asdict(v) for v in self.voices],
            "upstreams": [asdict(u) for u in self.upstreams],
            "mcp_servers": [asdict(s) for s in self.mcp_servers],
            "profiles": [asdict(p) for p in self.profiles],
        }
        return _strip_none(d)

    def save(self, path: Path | None = None) -> Path:
        """Persist the config, atomically.

        Writing in place with ``open(path, "wb")`` truncates first, so any
        failure between that and the last byte -- a full disk, a kill, an
        exception raised while serialising -- left a truncated file behind.
        That is not a hypothetical corner: the surviving fragment is usually
        still *valid TOML*, so the next start loads it without complaint and
        the user silently comes up with no models, no GPUs and no admin token.
        A visible crash would be kinder than that.

        Write to a temporary file beside the target, fsync it, then rename
        over the original. Rename is atomic, so a reader either sees the whole
        old file or the whole new one and never a partial write. On Windows
        concurrent renames of the same target can collide; a per-path lock
        plus a small retry loop keeps multi-threaded saves reliable.

        The file carries ``server.admin_token``, so it is created 0600 and
        chmod'ed before the rename rather than after: it must never be
        briefly readable by other users under its final name.
        """
        path = path or default_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        # The directory holds the admin token too, so keep it private. Belt and
        # braces with the 0600 below: it survives the file's mode being lost.
        if os.name != "nt":
            try:
                os.chmod(path.parent, 0o700)
            except OSError:
                logging.getLogger("arc_llama.config").debug(
                    "could not chmod config directory", exc_info=True
                )

        # Same directory, so the rename stays on one filesystem and is atomic.
        # The random suffix keeps two writers from picking the same scratch
        # name; pid alone is not enough, since a single process can save from
        # more than one thread.
        tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(4)}")

        # Serialize concurrent saves targeting the same path. On Windows the
        # atomic-rename collision is visible as PermissionError; on POSIX it
        # is harmless but the lock still prevents temp-name starvation.
        key = str(path.resolve())
        lock = _save_locks.setdefault(key, threading.Lock())
        with lock:
            try:
                with open(tmp, "wb") as f:
                    tomli_w.dump(self.to_toml_dict(), f)
                    f.flush()
                    os.fsync(f.fileno())
                try:
                    os.chmod(tmp, 0o600)
                except OSError:
                    # Windows and some filesystems have limited chmod. Not fatal.
                    logging.getLogger("arc_llama.config").debug(
                        "could not chmod config temp file", exc_info=True
                    )
                # Windows can briefly deny the rename when another handle is
                # closing; a short retry absorbs the race without weakening
                # atomicity.
                if os.name == "nt":
                    deadline = time.monotonic() + 0.5
                    while True:
                        try:
                            os.replace(tmp, path)
                            break
                        except PermissionError:
                            if time.monotonic() > deadline:
                                raise
                            time.sleep(0.01)
                else:
                    os.replace(tmp, path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise

        # Best-effort durability for the rename itself. POSIX only: opening a
        # directory for fsync is not permitted on Windows.
        if os.name != "nt":
            try:
                dir_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                logging.getLogger("arc_llama.config").debug(
                    "could not fsync config directory", exc_info=True
                )
        return path


def _find_entry(entries: list[Any], query: str) -> Any | None:
    """Match *query* against a list of model entries.

    Exact name first, then exact alias, then case-insensitive substring over
    name / display_name / basename(path) / aliases. Shared by the LLM and
    audio registries so a client's `model` field resolves the same way in
    both — an id that works on /v1/chat/completions should not need
    different spelling on /v1/audio/transcriptions.
    """
    if not query:
        return None
    for m in entries:
        if m.name == query:
            return m
    for m in entries:
        if query in m.aliases:
            return m
    ql = query.lower()
    for m in entries:
        haystacks = [
            m.name.lower(),
            m.display_name.lower(),
            Path(m.path).name.lower(),
            *(a.lower() for a in m.aliases),
        ]
        if any(ql in h for h in haystacks):
            return m
    return None


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


def _filter_fields(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    """Return only the keys recognised by ``cls``, warning about extras.

    Keeps forward-compatible loading: a config written by a newer
    arc-llama may contain fields this version does not know about.
    """
    known = {f.name for f in fields(cls)}
    filtered: dict[str, Any] = {}
    for k, v in raw.items():
        if k in known:
            filtered[k] = v
        else:
            logging.getLogger("arc_llama.config").warning(
                "Ignoring unknown config key %r in %s", k, cls.__name__
            )
    return filtered


def migrate_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Bump an on-disk config dict to the current schema version.

    Applies field-level migrations so older configs (0.1 → 0.2 → 0.3) pick up
    new defaults without losing user edits.
    """
    version = int(raw.get("version", 1))
    if version > CONFIG_VERSION:
        raise ValueError(
            f"config version {version} is newer than the supported version "
            f"{CONFIG_VERSION}; upgrade arc-llama"
        )

    # Ensure all top-level sections exist so downstream code can assume them.
    raw.setdefault("server", {})
    raw.setdefault("paths", {})
    raw.setdefault("tune", {})
    raw.setdefault("workload", {})
    raw.setdefault("agent", {})
    raw.setdefault("gpus", [])
    raw.setdefault("models", [])
    raw.setdefault("audio_models", [])
    raw.setdefault("voices", [])
    raw.setdefault("upstreams", [])
    raw.setdefault("mcp_servers", [])
    raw.setdefault("profiles", [])

    # 0.2 → 0.3: GPU backend field (SYCL vs Vulkan). Default to SYCL to match
    # pre-0.3 behaviour, but log so the user knows they can set it explicitly.
    for gpu in raw.get("gpus", []):
        if not isinstance(gpu, dict):
            continue
        if "backend" not in gpu:
            gpu["backend"] = Backend.SYCL.value
            logging.getLogger("arc_llama.config").warning(
                "GPU %s is missing the 'backend' field; defaulting to '%s'. "
                "Set it to '%s' if you are using a Vulkan llama-server build.",
                gpu.get("pci_slot", "?"),
                Backend.SYCL.value,
                Backend.VULKAN.value,
            )

    # 0.3: agent settings default to a server-side project root of ".".
    agent = raw.get("agent", {})
    if "root" not in agent:
        agent["root"] = "."
    if "profile" not in agent:
        agent["profile"] = None

    # Ensure newer server fields exist with safe defaults.
    server = raw.get("server", {})
    if "admin_token" not in server:
        server["admin_token"] = None

    # Audio launch knobs moved from the entry body into [audio_models.recipe]
    # so the two model tables read the same way. Lift them rather than making
    # anyone who tried the first cut hand-edit their config.
    for audio in raw.get("audio_models", []):
        if not isinstance(audio, dict):
            continue
        recipe = audio.setdefault("recipe", {})
        for key in _AUDIO_RECIPE_KEYS:
            if key in audio:
                recipe.setdefault(key, audio.pop(key))

    # Ensure model defaults that were introduced across releases.
    for model in raw.get("models", []):
        if not isinstance(model, dict):
            continue
        if "kv_class" not in model:
            model["kv_class"] = "default"
        if "display_name" not in model:
            model["display_name"] = ""
        if "aliases" not in model:
            model["aliases"] = []

    raw["version"] = CONFIG_VERSION
    return raw


def validate_config(raw: dict[str, Any]) -> None:
    """Basic structural validation for a loaded config dict."""
    if not isinstance(raw.get("version"), int):
        raise ValueError("config 'version' must be an integer")
    if not isinstance(raw.get("server", {}), dict):
        raise ValueError("config 'server' must be a table")
    if not isinstance(raw.get("paths", {}), dict):
        raise ValueError("config 'paths' must be a table")
    if not isinstance(raw.get("agent", {}), dict):
        raise ValueError("config 'agent' must be a table")
    if not isinstance(raw.get("gpus", []), list):
        raise ValueError("config 'gpus' must be an array")
    if not isinstance(raw.get("models", []), list):
        raise ValueError("config 'models' must be an array")
    if not isinstance(raw.get("audio_models", []), list):
        raise ValueError("config 'audio_models' must be an array")
    if not isinstance(raw.get("voices", []), list):
        raise ValueError("config 'voices' must be an array")
    if not isinstance(raw.get("upstreams", []), list):
        raise ValueError("config 'upstreams' must be an array")
    if not isinstance(raw.get("mcp_servers", []), list):
        raise ValueError("config 'mcp_servers' must be an array")
    if not isinstance(raw.get("profiles", []), list):
        raise ValueError("config 'profiles' must be an array")
    if not isinstance(raw.get("tune", {}), dict):
        raise ValueError("config 'tune' must be a table")
    if not isinstance(raw.get("workload", {}), dict):
        raise ValueError("config 'workload' must be a table")


def _resolve_admin_token(cfg: Config, path: Path, *, persist: bool) -> None:
    """Fill in cfg.server.admin_token so admin/auto_confirm auth is never a no-op.

    ARC_LLAMA_ADMIN_TOKEN always wins and is never written to disk (so it can
    be overridden per-invocation, e.g. in containers). Otherwise, if no token
    is configured yet, generate one and persist it so it survives restarts --
    admin endpoints and `auto_confirm` agent runs would otherwise be
    unauthenticated by default.
    """
    env_token = os.environ.get("ARC_LLAMA_ADMIN_TOKEN")
    if env_token:
        cfg.server.admin_token = env_token
        return
    if cfg.server.admin_token:
        return
    cfg.server.admin_token = secrets.token_urlsafe(32)
    if not persist:
        # Callers pass persist=False when no config file exists yet. The old
        # message claimed the token was "saved to <path>" on this branch too,
        # sending users hunting for a file that was never written -- and hiding
        # that the token rotates on every restart until one is configured.
        logging.getLogger("arc_llama.config").warning(
            "No admin_token was configured -- generated an in-memory one for "
            "this run only (no config file exists at %s, so nothing was "
            "saved). Admin endpoints and auto_confirm agent runs require an "
            "'Authorization: Bearer <token>' header, and the token changes on "
            "every restart until one is persisted. Set ARC_LLAMA_ADMIN_TOKEN "
            "or create a config to pin it.",
            path,
        )
        return
    try:
        cfg.save(path)
    except OSError as exc:
        logging.getLogger("arc_llama.config").warning(
            "No admin_token was configured -- generated one but could not "
            "save it to %s: %s. The token is in-memory only for this run; "
            "admin endpoints and auto_confirm agent runs will use a new "
            "token after restart. Set ARC_LLAMA_ADMIN_TOKEN to use your "
            "own token without persisting it to disk.",
            path,
            exc,
        )
        return
    logging.getLogger("arc_llama.config").warning(
        "No admin_token was configured -- generated one and saved it to %s. "
        "Admin endpoints and auto_confirm agent runs now require an "
        "'Authorization: Bearer <token>' header. Set ARC_LLAMA_ADMIN_TOKEN "
        "to use your own token without persisting it to disk.",
        path,
    )


def load_config(path: Path | None = None) -> Config:
    path = path or default_config_path()
    if not path.exists():
        cfg = Config()
        _resolve_admin_token(cfg, path, persist=False)
        return cfg
    with open(path, "rb") as f:
        raw = _toml_load(f)
    raw = migrate_config(raw)
    validate_config(raw)
    top = _filter_fields(Config, raw)
    cfg = Config(
        version=int(top.get("version", CONFIG_VERSION)),
        server=ServerConfig(**_filter_fields(ServerConfig, top.get("server", {}))),
        paths=PathsConfig(**_filter_fields(PathsConfig, top.get("paths", {}))),
        tune=TuneConfig(**_filter_fields(TuneConfig, top.get("tune", {}))),
        workload=WorkloadConfig(**_filter_fields(WorkloadConfig, top.get("workload", {}))),
        agent=AgentConfig(**_filter_fields(AgentConfig, top.get("agent", {}))),
        gpus=[GPUConfig(**_filter_fields(GPUConfig, g)) for g in top.get("gpus", [])],
        models=[ModelConfig(**_filter_fields(ModelConfig, m)) for m in top.get("models", [])],
        audio_models=[
            AudioModelConfig(**_filter_fields(AudioModelConfig, m))
            for m in top.get("audio_models", [])
        ],
        voices=[VoiceConfig(**_filter_fields(VoiceConfig, v)) for v in top.get("voices", [])],
        upstreams=[
            UpstreamConfig(**_filter_fields(UpstreamConfig, u)) for u in top.get("upstreams", [])
        ],
        mcp_servers=[
            MCPServerConfig(**_filter_fields(MCPServerConfig, s))
            for s in top.get("mcp_servers", [])
        ],
        profiles=[
            ProfileConfig(**_filter_fields(ProfileConfig, p)) for p in top.get("profiles", [])
        ],
    )
    _resolve_admin_token(cfg, path, persist=True)
    return cfg


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
            backend=Backend.SYCL.value,
        )
        # Prefer the highest-VRAM Battlemage / Alchemist as the default GPU.
        if not enabled_set and g.arch in (Arch.BATTLEMAGE, Arch.ALCHEMIST) and g.vram_mb:
            gc.enabled = True
            enabled_set = True
        cfg.gpus.append(gc)
    if cfg.gpus and not enabled_set:
        cfg.gpus[0].enabled = True
    return cfg
