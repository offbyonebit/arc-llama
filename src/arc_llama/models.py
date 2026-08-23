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
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arc_llama.config import (
    ASR_ENGINES,
    AUDIO_ENGINE_LLAMACPP,
    AudioModelConfig,
    Config,
    ModelConfig,
    VoiceConfig,
)
from arc_llama.gguf_meta import (
    has_mtp_heads,
    is_hybrid_ssm,
    is_moe,
    read_gguf_meta,
    trained_context_length,
)
from arc_llama.recipes import default_recipe, recipe_to_dict

log = logging.getLogger("arc_llama.models")


def _suggest_moe_offload(
    path: Path,
    *,
    vram_mb: int,
    recipe_dict: dict[str, Any],
    kv_class: str,
) -> int | None:
    """Smallest ``--n-cpu-moe`` layer count estimated to fit the card, or None.

    ``--n-cpu-moe N`` keeps the routed-expert tensors of the first N *layers*
    on the host (not N experts), so the suggestion must come from per-layer
    expert-tensor accounting against the recipe's own ctx/KV — anything else
    disagrees with the load-time VRAM guard and the two cancel out: ``add``
    enables offload, then the guard counts full weights and refuses the load.
    This shares the router's estimator so registration and the guard agree by
    construction. Returns None when no offload is needed or when expert
    tensor bytes cannot be determined (in which case the guard skips rather
    than refuses, so no blind guess is needed here either).
    """
    from arc_llama.router import min_moe_offload_layers

    probe = ModelConfig(
        name="(offload-probe)",
        path=str(path),
        port=0,
        gpu_pci_slot="",
        kv_class=kv_class,
        recipe=dict(recipe_dict),
    )
    n = min_moe_offload_layers(probe, vram_mb)
    return n if n and n > 0 else None


NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
HF_SPEC_RE = re.compile(r"^(?P<repo>[^@:\s/]+/[^@:\s/]+)(?::(?P<file>[^@\s]+))?$")


@dataclass
class HFModelSpec:
    """Parsed user input like `unsloth/gemma-4-31B-it-GGUF:Q4_K_M`."""

    repo: str
    file: str | None  # exact filename; if None, we glob the repo for a match
    quant: str | None  # short hint like "Q4_K_M", used when file is None


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
        m = re.search(r"(IQ\d[A-Z0-9_]*|Q\d[A-Z0-9_]*|UD-[A-Z0-9_]+)", file, re.IGNORECASE)
        if m:
            base = f"{base}-{m.group(1).lower()}"
    return base or "model"


def add_audio_model(
    cfg: Config,
    *,
    name: str,
    path: str,
    gpu_pci_slot: str,
    engine: str = AUDIO_ENGINE_LLAMACPP,
    mmproj: str = "",
    task: str = "asr",
    mode: str = "offline",
    port: int | None = None,
    display_name: str = "",
    aliases: list[str] | None = None,
    always_resident: bool = True,
    vram_mb: int | None = None,
    ctx: int = 0,
    recipe_overrides: dict[str, Any] | None = None,
    strip_asr_markers: bool = True,
) -> AudioModelConfig:
    """Register an audio model in the config.

    There is no recipe to derive the way there is for an LLM: what an audio
    backend needs is a task, an engine and a model. So this validates,
    allocates a port, and stores what the user told us — with per-engine
    validation delegated to the engine itself, so that adding one does not mean
    editing this function.

    Ports and names are allocated across *both* registries. They share one
    `/v1/models` namespace and one router process map, so a collision would
    make one of the two unreachable in a way that only shows up at request
    time.
    """
    from arc_llama.tts import engine_names, get_engine

    if not NAME_RE.match(name):
        raise ValueError(f"Model name '{name}' must match [a-z0-9][a-z0-9._-]*")
    if task not in ("asr", "tts"):
        raise ValueError(f"task must be 'asr' or 'tts', not {task!r}")
    if mode not in ("offline", "streaming"):
        raise ValueError(f"mode must be 'offline' or 'streaming', not {mode!r}")

    tts_engine = None
    if task == "tts":
        tts_engine = get_engine(engine)
        if tts_engine is None:
            known = ", ".join(engine_names()) or "(none)"
            raise ValueError(
                f"Unknown TTS engine {engine!r}. Registered engines: {known}."
            )
    elif engine not in ASR_ENGINES:
        raise ValueError(
            f"engine must be one of {list(ASR_ENGINES)} for task='asr', not {engine!r}."
        )

    if task == "asr":
        if not mmproj:
            raise ValueError(
                "A transcription model needs --mmproj, the audio projector GGUF "
                "published beside the weights (mmproj-*.gguf). Without it "
                "llama-server loads a plain text LLM and transcription returns "
                "confident nonsense rather than failing."
            )
        mm = Path(mmproj).expanduser().resolve()
        if not mm.exists():
            raise FileNotFoundError(f"mmproj not found: {mm}")
        mmproj = str(mm)

    p = Path(path).expanduser()
    if p.exists():
        p = p.resolve()
        stored_path = str(p)
    elif tts_engine is not None and tts_engine.accepts_remote_path:
        # OmniVoice is normally addressed by Hugging Face repo id, which the
        # engine resolves (and downloads) itself on first start. Refusing it
        # here would make the documented way of registering the model fail.
        stored_path = path
    else:
        raise FileNotFoundError(f"Model path not found: {p}")
    gpu = cfg.find_gpu(gpu_pci_slot)
    if gpu is None:
        raise ValueError(f"GPU {gpu_pci_slot} not in config — run `arc-llama init` first.")
    used_names = {m.name for m in cfg.models} | {m.name for m in cfg.audio_models}
    if name in used_names:
        raise ValueError(f"Model name '{name}' already registered.")
    used_ports = {m.port for m in cfg.models} | {m.port for m in cfg.audio_models}
    port = port or _next_free_port(used_ports)
    if port in used_ports:
        raise ValueError(f"Port {port} already in use by another model.")
    recipe: dict[str, Any] = {}
    if mmproj:
        recipe["mmproj"] = mmproj
    if ctx:
        recipe["ctx"] = ctx
    for key, value in (recipe_overrides or {}).items():
        if value not in (None, "", {}, []):
            recipe[key] = value
    entry = AudioModelConfig(
        name=name,
        path=stored_path,
        port=port,
        gpu_pci_slot=gpu_pci_slot,
        engine=engine,
        task=task,
        mode=mode,
        recipe=recipe,
        display_name=display_name or name,
        aliases=list(aliases or []),
        always_resident=always_resident,
        vram_mb=vram_mb,
        strip_asr_markers=strip_asr_markers,
    )
    if tts_engine is not None:
        # Engine-specific checks run while the user is still at the prompt and
        # can fix the problem, rather than at first request.
        tts_engine.validate(entry)
    cfg.audio_models.append(entry)
    return entry


def add_voice(
    cfg: Config,
    *,
    name: str,
    ref_audio: str = "",
    ref_text: str = "",
    instruct: str = "",
    language: str = "",
    auto: bool = False,
    models: list[str] | None = None,
    display_name: str = "",
    aliases: list[str] | None = None,
) -> VoiceConfig:
    """Register a named voice for `/v1/audio/speech`.

    A voice is cloned from a reference clip, designed from attributes, or
    ``auto`` — the model's own voice, with no prompt at all. That last one is
    what a *fine-tuned* model wants: its speaker is in the weights, so adding a
    clone or design prompt on top fights the training rather than helping.

    ``auto`` has to be asked for explicitly, because a voice with no reference
    and no attributes is otherwise indistinguishable from one whose fields were
    forgotten — and silently registering a name that does nothing is a much
    worse outcome than an error.
    """
    if not NAME_RE.match(name):
        raise ValueError(f"Voice name '{name}' must match [a-z0-9][a-z0-9._-]*")
    if any(v.name == name for v in cfg.voices):
        raise ValueError(f"Voice '{name}' already registered.")
    if auto and (ref_audio or instruct):
        raise ValueError(
            "--auto means the model supplies the voice itself, so it cannot be "
            "combined with --ref-audio or --instruct."
        )
    if not auto and not ref_audio and not instruct:
        raise ValueError(
            "A voice needs --ref-audio (clone a reference clip), --instruct "
            "(design one from attributes, e.g. 'female, low pitch, british "
            "accent'), or --auto (the model's own voice, for a fine-tune whose "
            "speaker is already in the weights)."
        )
    resolved_ref = ""
    if ref_audio:
        clip = Path(ref_audio).expanduser()
        if not clip.exists():
            raise FileNotFoundError(f"Reference audio not found: {clip}")
        resolved_ref = str(clip.resolve())
    for model_name in models or []:
        if cfg.find_audio_model(model_name) is None:
            raise ValueError(f"Unknown audio model: {model_name}")
    voice = VoiceConfig(
        name=name,
        ref_audio=resolved_ref,
        ref_text=ref_text,
        instruct=instruct,
        language=language,
        models=list(models or []),
        display_name=display_name or name,
        aliases=list(aliases or []),
    )
    cfg.voices.append(voice)
    return voice


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
        raise ValueError(f"Model name '{name}' must match [a-z0-9][a-z0-9._-]*")
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    gpu = cfg.find_gpu(gpu_pci_slot)
    if gpu is None:
        raise ValueError(f"GPU {gpu_pci_slot} not in config — run `arc-llama init` first.")
    # Prefer GGUF metadata over the filename heuristic for kv_class unless the
    # caller passed an explicit (non-default) hint. MoE models in particular
    # (Gemma-4 A4B, Qwen3 A3B) share a filename family with dense variants.
    if kv_class == "default":
        inferred = infer_kv_class_from_path(p)
        if inferred is not None:
            kv_class = inferred
            log.info("model %s: kv_class=%s from GGUF metadata", name, kv_class)
    used_ports = {m.port for m in cfg.models}
    port = port or _next_free_port(used_ports)
    if port in used_ports:
        raise ValueError(f"Port {port} already in use by another model.")
    if any(m.name == name for m in cfg.models):
        raise ValueError(f"Model name '{name}' already registered.")
    # Build a recipe that fits this GPU.
    from arc_llama.arch import Arch, Backend

    arch = Arch(gpu.arch) if gpu.arch else Arch.UNKNOWN
    backend = Backend(gpu.backend) if gpu.backend else Backend.SYCL
    trained_ctx = trained_context_length(p)
    recipe = default_recipe(
        arch=arch,
        vram_mb=gpu.vram_mb or 8192,
        model_file_mb=p.stat().st_size // (1024 * 1024),
        kv_class=kv_class,
        backend=backend,
        trained_ctx=trained_ctx,
    )
    recipe_dict: dict[str, Any] = recipe_to_dict(recipe)
    # Auto-enable draft-mtp for models that actually carry MTP heads.
    # Measured B60/Qwen3.6-27B-MTP: draft-mtp n_max 1–4 ≈ +20% gen vs none;
    # n_max 5–6 regresses. Pin n_max=3 (llama default / mid of the good band).
    if has_mtp_heads(p):
        if is_hybrid_ssm(p):
            # Hybrid SSM+attention families (qwen35*) carry MTP heads but are
            # documented right on is_hybrid_ssm() as performing poorly with
            # SYCL MTP speculative decoding on Xe2 — enabling it here would
            # auto-configure a known regression. Leave spec_type unset; the
            # user can still opt in per model.
            log.info(
                "model %s has MTP heads but is a hybrid SSM architecture; "
                "NOT auto-enabling draft-mtp (known slow on Xe2)",
                name,
            )
        else:
            recipe_dict["spec_type"] = "draft-mtp"
            recipe_dict["spec_draft_n_max"] = DEFAULT_SPEC_DRAFT_N_MAX
            log.info(
                "model %s has embedded MTP heads; auto-enabling spec_type=draft-mtp "
                "spec_draft_n_max=%d (B60 measured band 1–4)",
                name,
                DEFAULT_SPEC_DRAFT_N_MAX,
            )
    else:
        draft = find_draft_model(p)
        if draft is not None:
            recipe_dict["spec_type"] = "draft-mtp"
            recipe_dict["spec_draft_model"] = str(draft)
            recipe_dict["spec_draft_ngl"] = DEFAULT_SPEC_DRAFT_NGL
            recipe_dict["spec_draft_n_max"] = DEFAULT_SPEC_DRAFT_N_MAX
            log.info(
                "model %s: found sidecar draft %s; auto-enabling draft-mtp "
                "with --spec-draft-model (spec_draft_n_max=%d)",
                name,
                draft.name,
                DEFAULT_SPEC_DRAFT_N_MAX,
            )
    # Suggest MoE expert offload when the estimated footprint needs it.
    if is_moe(p):
        n_cpu = _suggest_moe_offload(
            p,
            vram_mb=gpu.vram_mb or 8192,
            recipe_dict=recipe_dict,
            kv_class=kv_class,
        )
        if n_cpu:
            recipe_dict["n_cpu_moe"] = n_cpu
            log.info(
                "model %s is MoE; offloading expert tensors of %d layer(s) to CPU",
                name,
                n_cpu,
            )
        else:
            log.debug(
                "model %s is MoE and fits without offload (or its expert "
                "tensors could not be accounted)",
                name,
            )
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
    (re.compile(r"gemma[\W_-]*[234]", re.IGNORECASE), "gemma_swa"),
    (re.compile(r"phi[\W_-]*4", re.IGNORECASE), "phi4"),
    (re.compile(r"deepseek[\W_-]*r1[\W_-]*distill", re.IGNORECASE), "deepseek_r1_distill"),
    (re.compile(r"llama[\W_-]*(3|4)", re.IGNORECASE), "llama3"),
    (re.compile(r"qwen[\W_-]*2[.\W_-]*5", re.IGNORECASE), "qwen2_5"),
    (re.compile(r"qwen[\W_-]*3[.\W_-]*6?[\W_-]*27b(?!.*a3b)", re.IGNORECASE), "qwen3_dense"),
    (re.compile(r"qwen[\W_-]*3(?!.*a3b)(?!.*moe)", re.IGNORECASE), "qwen3_dense"),
    (
        re.compile(
            r"(qwen[\W_-]*3.*a3b|qwen[\W_-]*3.*moe|carnice|huihui.*30b.*a3b)", re.IGNORECASE
        ),
        "moe_a3b",
    ),
]

# Common quant tier markers that look good in display names.
_QUANT_TIER_RE = re.compile(
    r"\b(IQ\d_[A-Z_]*|Q\d_[KS]_[A-Z]+|Q\d_K|Q\d_\d|Q8_0|UD-[A-Z0-9_]+)\b",
    re.IGNORECASE,
)


# --- Speculative sidecar-draft detection -----------------------------------
# Some models ship their MTP/EAGLE draft heads as a separate small GGUF next to
# the main weights (e.g. `mtp-gemma-4-26B-A4B-it.gguf`). We auto-pair them so
# the fast speculative recipe works without the user knowing the file exists.

# A draft carries the marker as a *prefix*. This deliberately does NOT match a
# mid-name 'MTP' (e.g. `Qwen3.6-27B-MTP-...`), which is a full model with
# *embedded* heads, not a sidecar draft.
_DRAFT_PREFIX_RE = re.compile(r"^(mtp|draft|eagle|medusa)[-_.]", re.IGNORECASE)

DEFAULT_SPEC_DRAFT_N_MAX = 3  # B60-measured good band is 1–4; 5–6 regress.
DEFAULT_SPEC_DRAFT_NGL = 999  # fully offload the (small) draft to the GPU.


def _sibling_ggufs(directory: Path) -> list[Path]:
    """List *.gguf files in `directory` via os.listdir.

    Deliberately avoids Path.glob, which stats the directory's mode and so
    trips test stat-mocks; os.listdir needs no stat on the entries.
    """
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return sorted(directory / n for n in names if n.lower().endswith(".gguf"))


def _model_key(name: str) -> str:
    """Normalise a GGUF filename to a comparable base key.

    Drops the extension, a leading draft marker, quant-tier tokens, and every
    non-alphanumeric, so `mtp-gemma-4-26B-A4B-it.gguf` and
    `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` reduce to comparable stems
    (`gemma426ba4bit` vs `gemma426ba4bitqat`).
    """
    stem = name[:-5] if name.lower().endswith(".gguf") else name
    stem = _DRAFT_PREFIX_RE.sub("", stem)
    stem = _QUANT_TIER_RE.sub("", stem)
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def find_draft_model(main_path: Path | str, siblings: list[Path] | None = None) -> Path | None:
    """Return a sidecar speculative-draft GGUF for `main_path`, or None.

    A sibling in the same directory qualifies when it carries a draft-marker
    prefix, is smaller than the main model, and shares its base name.
    """
    main = Path(main_path)
    try:
        main_size = main.stat().st_size
    except OSError:
        return None
    if siblings is None:
        siblings = _sibling_ggufs(main.parent)
    main_key = _model_key(main.name)
    if not main_key:
        return None
    for s in siblings:
        if s.name == main.name or not _DRAFT_PREFIX_RE.match(s.name):
            continue
        try:
            if s.stat().st_size >= main_size:  # a draft is smaller than its target
                continue
        except OSError:
            continue
        draft_key = _model_key(s.name)
        if draft_key and main_key.startswith(draft_key):
            return s
    return None


def looks_like_draft(path: Path | str, siblings: list[Path] | None = None) -> bool:
    """True if `path` is a sidecar draft belonging to some larger sibling.

    Used by scan to avoid registering a draft GGUF as a standalone model.
    """
    p = Path(path)
    if not _DRAFT_PREFIX_RE.match(p.name):
        return False
    if siblings is None:
        siblings = _sibling_ggufs(p.parent)
    for s in siblings:
        if s.name == p.name:
            continue
        d = find_draft_model(s, siblings)
        if d is not None and d.name == p.name:
            return True
    return False


def is_audio_gguf(path: Path | str, cfg: Config | None = None) -> bool:
    """True if `path` is an audio model's GGUF rather than a chat LLM's.

    A scan of the models directory happily finds Qwen3-ASR next to Qwen3.
    Registering one as an LLM is not a harmless mistake: it gets a KV-cache
    class, a context length sized to VRAM and a place in the auto-tuner's
    candidate list, and every one of those is meaningless.

    Four signals, cheapest first:

    1. Already registered as (or living under) an audio model's path.
    2. Named `mmproj-*.gguf` — a projector is never a standalone model,
       whatever it projects for.
    3. No `general.architecture` in its metadata: llama.cpp requires that key
       to pick an implementation, so a GGUF without one is not something the
       LLM path can load at all.
    4. An ASR-looking name *with* a projector sibling. Qwen3-ASR reports
       `architecture: qwen3vl`, so metadata alone cannot tell it from a
       vision-language chat model — but a vision model registered for chat is
       a legitimate thing to do, while an ASR model registered for chat is
       not, and only the latter's name says `asr`.
    """
    p = Path(path)
    if cfg is not None:
        for am in cfg.audio_models:
            for known in (am.path, am.audio_recipe().mmproj):
                if not known:
                    continue
                candidate = Path(known).expanduser()
                try:
                    if p == candidate.resolve() or candidate in p.parents:
                        return True
                except OSError:
                    continue
    if p.name.lower().startswith("mmproj-"):
        return True
    if "asr" in p.stem.lower() and (p.parent / f"mmproj-{p.name}").exists():
        return True
    meta = read_gguf_meta(p)
    if not meta:
        # Unreadable is not the same as audio; leave that judgement to the
        # existing registration path, which already tolerates thin metadata.
        return False
    return not meta.get("architecture")


def infer_kv_class(filename: str) -> str:
    """Guess the kv_class for VRAM estimation from the GGUF filename."""
    for pattern, kv_class in _KV_CLASS_PATTERNS:
        if pattern.search(filename):
            return kv_class
    return "default"


def infer_kv_class_from_path(path: Path) -> str | None:
    """Infer kv_class from GGUF metadata when available.

    More reliable than the filename heuristic for families that share a name
    across dense and MoE variants (e.g. Gemma-4, Qwen3): the architecture
    string is identical, so ``expert_count`` metadata is what actually tells
    dense and MoE apart. Returns None when metadata can't decide, so the
    caller falls back to ``infer_kv_class`` (filename) or ``"default"``.
    """
    meta = read_gguf_meta(path)
    if not meta:
        return None
    # MoE is the highest-value signal: dense and MoE variants of the same
    # family share an architecture string, so expert_count distinguishes them.
    if is_moe(path):
        return "moe_a3b"
    arch = str(meta.get("architecture", "")).lower()
    if arch.startswith("gemma"):
        return "gemma_swa"
    if arch == "phi4" or arch.startswith("phi4"):
        return "phi4"
    return None


def resolve_kv_class(path: Path, explicit: str = "default") -> str:
    """Pick the kv_class for VRAM sizing.

    Order of precedence:
      1. an explicit, non-default override from the caller (CLI flag),
      2. GGUF metadata (MoE / Gemma / Phi-4 — the cases the filename gets wrong),
      3. the filename heuristic,
      4. ``"default"``.
    """
    if explicit != "default":
        return explicit
    meta_kv = infer_kv_class_from_path(path)
    if meta_kv is not None:
        return meta_kv
    return infer_kv_class(path.name)


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
        base = re.sub(
            r"[^a-z0-9._-]+", "-", f"{path.parent.name.lower()}-{path.stem.lower()}"
        ).strip("-")
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
    from arc_llama.arch import Arch, Backend

    arch = Arch(gpu.arch) if gpu.arch else Arch.UNKNOWN
    backend = Backend(gpu.backend) if gpu.backend else Backend.SYCL
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
        if looks_like_draft(rp):
            log.info("skipping %s (speculative draft for a sibling model)", rp.name)
            continue
        if is_audio_gguf(rp, cfg):
            log.info("skipping %s (audio model, not a chat LLM)", rp.name)
            continue
        kv_class = resolve_kv_class(rp)
        trained_ctx = trained_context_length(rp)
        recipe = default_recipe(
            arch=arch,
            vram_mb=gpu.vram_mb or 8192,
            model_file_mb=rp.stat().st_size // (1024 * 1024),
            kv_class=kv_class,
            backend=backend,
            trained_ctx=trained_ctx,
        )
        recipe_dict: dict[str, Any] = recipe_to_dict(recipe)
        # Auto-enable draft-mtp for discovered models that carry MTP heads.
        # n_max=3 pinned from B60 measurements (see bench_results/SUMMARY.md).
        if has_mtp_heads(rp):
            if is_hybrid_ssm(rp):
                # Same guard as add_local_model: hybrid SSM + SYCL MTP is a
                # documented regression on Xe2, so discovery must not
                # auto-enable it either.
                log.info(
                    "discovered %s has MTP heads but is a hybrid SSM architecture; "
                    "NOT auto-enabling draft-mtp (known slow on Xe2)",
                    rp.name,
                )
            else:
                recipe_dict["spec_type"] = "draft-mtp"
                recipe_dict["spec_draft_n_max"] = DEFAULT_SPEC_DRAFT_N_MAX
                log.info(
                    "discovered %s has embedded MTP heads; auto-enabling draft-mtp n_max=%d",
                    rp.name,
                    DEFAULT_SPEC_DRAFT_N_MAX,
                )
        else:
            draft = find_draft_model(rp)
            if draft is not None:
                recipe_dict["spec_type"] = "draft-mtp"
                recipe_dict["spec_draft_model"] = str(draft)
                recipe_dict["spec_draft_ngl"] = DEFAULT_SPEC_DRAFT_NGL
                recipe_dict["spec_draft_n_max"] = DEFAULT_SPEC_DRAFT_N_MAX
                log.info(
                    "discovered %s: sidecar draft %s; auto-enabling draft-mtp n_max=%d",
                    rp.name,
                    draft.name,
                    DEFAULT_SPEC_DRAFT_N_MAX,
                )
        # Suggest MoE expert offload when the estimated footprint needs it.
        if is_moe(rp):
            n_cpu = _suggest_moe_offload(
                rp,
                vram_mb=gpu.vram_mb or 8192,
                recipe_dict=recipe_dict,
                kv_class=kv_class,
            )
            if n_cpu:
                recipe_dict["n_cpu_moe"] = n_cpu
                log.info(
                    "discovered %s is MoE; offloading expert tensors of %d layer(s) to CPU",
                    rp.name,
                    n_cpu,
                )
            else:
                log.debug(
                    "discovered %s is MoE and fits without offload (or its "
                    "expert tensors could not be accounted)",
                    rp.name,
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
            recipe=recipe_dict,
            aliases=[rp.name],
        )
        cfg.models.append(mc)
        added.append(mc)
        existing_paths.add(rp)
        log.info(
            "discovered %s → %s (kv=%s, ctx=%d, port=%d)",
            rp.name,
            name,
            kv_class,
            recipe.ctx,
            port,
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
        files = [f for f in api.list_repo_files(spec.repo) if f.endswith(".gguf")]
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
    return Path(
        hf_hub_download(
            repo_id=spec.repo,
            filename=file,
            local_dir=str(target_dir),
            token=token,
        )
    )


def download_asr_from_hf(
    spec: HFModelSpec,
    *,
    target_dir: Path,
    token: str | None = None,
) -> tuple[Path, Path]:
    """Download an ASR repo's weights *and* its audio projector.

    Returns ``(model_path, mmproj_path)``. Separate from
    ``download_from_hf`` because an ASR repo holds two files per quant
    (`Qwen3-ASR-0.6B-Q8_0.gguf` and `mmproj-Qwen3-ASR-0.6B-Q8_0.gguf`) and
    the generic path would return whichever sorted first — a coin flip
    between the model and its projector.
    """
    target_dir = Path(target_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface-hub is required for downloads. Install it with "
            "`pip install huggingface-hub`, or download the weights and the "
            "mmproj yourself and pass --mmproj."
        ) from e

    api = HfApi(token=token)
    files = [f for f in api.list_repo_files(spec.repo) if f.endswith(".gguf")]
    weights = [f for f in files if not Path(f).name.lower().startswith("mmproj")]
    projectors = [f for f in files if Path(f).name.lower().startswith("mmproj")]
    if not projectors:
        raise FileNotFoundError(
            f"{spec.repo} has no mmproj-*.gguf, so it is not an ASR/multimodal "
            "repo. Register it as a normal model with `arc-llama add`."
        )

    def _pick(candidates: list[str], what: str) -> str:
        if spec.file:
            exact = [f for f in candidates if Path(f).name == Path(spec.file).name]
            if exact:
                return exact[0]
        if spec.quant:
            ql = spec.quant.lower()
            matches = [f for f in candidates if ql in f.lower()]
            if not matches:
                raise FileNotFoundError(
                    f"No {what} in {spec.repo} matched quant hint "
                    f"'{spec.quant}'. Available: {', '.join(sorted(candidates))}"
                )
            return sorted(matches)[0]
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(
            f"{spec.repo} has {len(candidates)} {what} files; pick a quant with "
            f"`{spec.repo}:Q8_0`. Available: {', '.join(sorted(candidates))}"
        )

    model_file = _pick(weights, "weights")
    # Match the projector to the chosen quant where possible: a bf16 mmproj
    # beside q8_0 weights works but wastes VRAM for no accuracy anyone asked
    # for. Fall back to the sole projector when names don't line up.
    stem_quant = spec.quant or ""
    if not stem_quant:
        for candidate in ("q8_0", "bf16", "f16"):
            if candidate in Path(model_file).name.lower():
                stem_quant = candidate
                break
    matched = [f for f in projectors if stem_quant and stem_quant in f.lower()]
    mmproj_file = sorted(matched)[0] if matched else sorted(projectors)[0]

    paths = []
    for filename in (model_file, mmproj_file):
        paths.append(
            Path(
                hf_hub_download(
                    repo_id=spec.repo,
                    filename=filename,
                    local_dir=str(target_dir),
                    token=token,
                )
            )
        )
    return paths[0], paths[1]
