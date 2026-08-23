"""OmniVoice as an arc-llama TTS engine.

OmniVoice (k2-fsa) is a zero-shot multilingual TTS model shipped as a Python
library — a `model.generate(...)` call, no server and no binary. So this engine
launches :mod:`arc_llama.tts.omnivoice_server`, a small script that wraps one
loaded model in the `/v1/audio/speech` route, under whatever interpreter has
OmniVoice installed.

Running it as a subprocess rather than importing it here buys three things that
matter more than the extra hop: torch never enters arc-llama's environment,
stopping the model actually returns its VRAM (the router's existing evict path
just works), and a synthesis that wedges the GPU takes down a child process
instead of the router.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from arc_llama.arch import Arch, Backend, profile_for
from arc_llama.config import AudioModelConfig, Config, GPUConfig, VoiceConfig
from arc_llama.launcher import LaunchPlan, build_env
from arc_llama.tts.base import TTSEngine, register_engine

log = logging.getLogger("arc_llama.tts.omnivoice")

TTS_ENGINE_OMNIVOICE = "omnivoice"

DEFAULT_OMNIVOICE_REPO = "k2-fsa/OmniVoice"

SERVER_SCRIPT = Path(__file__).parent / "sidecars" / "omnivoice_server.py"

QUANTIZED_STATE_NAME = "quantized_state.pt"
"""Filename torchao quantization scripts write the weights to.

`torchao`'s `quantize_()` replaces Linear weights with tensor subclasses, which
`save_pretrained` cannot serialise — so the established practice is a plain
`torch.save` of the state dict beside an otherwise ordinary model directory
(config, tokenizer, audio_tokenizer/, but no `model.safetensors`). Its presence
is what marks a directory as quantized.
"""


def quantized_state_path(model_path: str) -> Path | None:
    """The quantized weights inside *model_path*, or None if it is an ordinary model.

    Detection rather than configuration: a directory holding
    `quantized_state.pt` cannot be loaded by `from_pretrained` at all, so
    treating it as unquantized has no valid interpretation — there is nothing
    for the user to choose here.
    """
    if not model_path:
        return None
    # `is_dir()` is also what rules out a Hugging Face repo id, which is not a
    # path on this machine and so can never be a quantized directory.
    directory = Path(model_path).expanduser()
    if not directory.is_dir():
        return None
    candidate = directory / QUANTIZED_STATE_NAME
    return candidate if candidate.is_file() else None


def tts_state_dir(cfg: Config) -> Path:
    """Where generated voice tables and cached voice prompts live."""
    base = Path(cfg.paths.state_dir).expanduser() if cfg.paths.state_dir else Path(".")
    return base / "tts"


def resolve_python(cfg: Config, model: AudioModelConfig) -> str:
    """The interpreter that runs the sidecar, most specific first.

    Falls back to the one running arc-llama, which is right only when the two
    share an environment — common in a container image built for both, wrong
    for the usual `pip install arc-llama` plus a separate OmniVoice checkout.
    The failure is loud either way: the child exits with ModuleNotFoundError
    and the router's log-tail hint says what to set.
    """
    recipe = model.audio_recipe()
    for candidate in (recipe.python, cfg.paths.tts_python):
        if candidate:
            return str(Path(candidate).expanduser())
    return sys.executable


def _hf_cache_dir(repo_id: str) -> Path | None:
    """Local cache directory for an HF repo id, if it has been downloaded.

    Resolves to the *snapshot* for the checked-out revision rather than the
    repo root, because only that revision is ever loaded. A cache holding two
    revisions has two sets of blobs, and measuring the root would charge the
    fit guard for weights that will never reach the GPU.
    """
    # `org/name` and nothing else. The obvious spelling of this test —
    # "contains os.sep, so it is a path" — is silently always true on POSIX,
    # where os.sep *is* the separator a repo id uses, so it rejected every repo
    # id and the estimate came back unknown for exactly the models it exists to
    # measure.
    parts = repo_id.split("/")
    if len(parts) != 2 or not all(parts) or repo_id[0] in "./~\\":
        return None
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        base = Path(hub).expanduser()
    else:
        home = os.environ.get("HF_HOME")
        root = Path(home).expanduser() if home else Path.home() / ".cache" / "huggingface"
        base = root / "hub"
    candidate = base / ("models--" + repo_id.replace("/", "--"))
    if not candidate.is_dir():
        return None
    snapshot = _current_snapshot(candidate)
    return snapshot if snapshot is not None else candidate


def _current_snapshot(repo_dir: Path) -> Path | None:
    """The snapshot directory `refs/main` points at, if it can be resolved."""
    try:
        revision = (repo_dir / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        # No refs (a manually assembled cache, or a revision pinned by hash).
        # A single snapshot is unambiguous; more than one is not, so fall back
        # to the repo root and let the inode de-duplication bound the damage.
        snapshots = sorted((repo_dir / "snapshots").glob("*")) if (
            repo_dir / "snapshots"
        ).is_dir() else []
        return snapshots[0] if len(snapshots) == 1 else None
    snapshot = repo_dir / "snapshots" / revision
    return snapshot if snapshot.is_dir() else None


class OmniVoiceEngine(TTSEngine):
    name = TTS_ENGINE_OMNIVOICE
    description = "k2-fsa OmniVoice — zero-shot multilingual TTS with voice cloning"
    speech_path = "/v1/audio/speech"
    accepts_remote_path = True

    # -- registration -------------------------------------------------

    def validate(self, model: AudioModelConfig) -> None:
        recipe = model.audio_recipe()
        if recipe.dtype and not recipe.dtype.replace("_", "").isalnum():
            raise ValueError(f"invalid dtype {recipe.dtype!r}")
        path = Path(model.path).expanduser()
        if not path.exists() and "/" not in model.path:
            raise FileNotFoundError(
                f"{model.path!r} is neither a local directory nor a Hugging Face "
                f"repo id. Use a path, or the repo id {DEFAULT_OMNIVOICE_REPO!r}."
            )

    # -- lifecycle ----------------------------------------------------

    def build_plan(
        self, cfg: Config, model: AudioModelConfig, gpu: GPUConfig, host: str = "127.0.0.1"
    ) -> LaunchPlan:
        recipe = model.audio_recipe()
        python = resolve_python(cfg, model)
        if not SERVER_SCRIPT.exists():  # pragma: no cover - broken install
            raise RuntimeError(f"the OmniVoice sidecar is missing from {SERVER_SCRIPT}")

        device = recipe.device or default_device(gpu)
        voices_path = write_voices_file(cfg, model)
        # torchao int8 checkpoints are produced from a bf16 base, and the
        # quantized tensors carry that as their compute dtype. Loading the base
        # as fp16 instead gives a dtype mismatch on the first matmul, so the
        # default follows the checkpoint rather than the engine.
        dtype = recipe.dtype or (
            "bfloat16" if quantized_state_path(model.path) is not None else "float16"
        )

        argv: list[str] = [
            python,
            str(SERVER_SCRIPT),
            "--model", str(Path(model.path).expanduser()) if Path(model.path).expanduser().exists()
            else model.path,
            "--host", host,
            "--port", str(model.port),
            "--device", device,
            "--dtype", dtype,
            "--voices", str(voices_path),
            "--default-response-format", recipe.default_response_format or "mp3",
        ]
        if recipe.default_language:
            argv += ["--default-language", recipe.default_language]

        options = recipe.options or {}
        if options.get("num_step"):
            argv += ["--num-step", str(int(options["num_step"]))]
        if options.get("asr_model"):
            argv += ["--asr-model", str(options["asr_model"])]
        if options.get("asr_device"):
            argv += ["--asr-device", str(options["asr_device"])]
        if options.get("normalize_text"):
            argv.append("--normalize-text")
        if options.get("compile"):
            argv.append("--compile")

        state_path = quantized_state_path(model.path)
        if state_path is not None:
            # The quantized directory has no weights `from_pretrained` can read,
            # so the sidecar rebuilds the structure from the base model and then
            # loads these tensors into it.
            argv += [
                "--quantize", str(options.get("quantization", "int8")),
                "--quantized-state", str(state_path),
                "--base-model", str(options.get("base_model") or DEFAULT_OMNIVOICE_REPO),
            ]
        argv.extend(recipe.extra_flags)

        backend_url = f"http://{host}:{model.port}"
        return LaunchPlan(
            argv=argv,
            env=build_env_for(cfg, gpu, device),
            backend_url=backend_url,
            health_url=f"{backend_url}/health",
        )

    def estimate_vram_mb(self, model: AudioModelConfig) -> int | None:
        if model.vram_mb:
            return int(model.vram_mb)
        path = Path(model.path).expanduser()
        if not path.exists():
            # The model is named by repo id, so its bytes are in the HF cache
            # rather than at `path`. Measuring there keeps the fit guard from
            # treating a perfectly measurable ~3 GB model as unknown-sized and
            # letting an LLM load on top of it.
            cached = _hf_cache_dir(model.path)
            if cached is None:
                return None
            path = cached
        return super().estimate_vram_mb(
            _with_path(model, str(path))
        )

    # -- requests -----------------------------------------------------

    def build_payload(self, model: AudioModelConfig, body: dict[str, Any]) -> dict[str, Any]:
        """Fill in the model's request defaults; the sidecar is OpenAI-native.

        The defaults are applied here rather than in the sidecar so they follow
        the config: editing `recipe.default_voice` takes effect on the next
        request instead of the next model reload.
        """
        recipe = model.audio_recipe()
        payload = dict(body)
        payload.pop("model", None)
        if not payload.get("voice") and recipe.default_voice:
            payload["voice"] = recipe.default_voice
        if not payload.get("response_format") and recipe.default_response_format:
            payload["response_format"] = recipe.default_response_format
        if not payload.get("language") and recipe.default_language:
            payload["language"] = recipe.default_language
        for key, value in (recipe.options or {}).items():
            # Per-request wins; these are the model's defaults for knobs the
            # sidecar accepts (num_step, guidance_scale, ...).
            if key in _REQUEST_OPTIONS and payload.get(key) is None:
                payload[key] = value
        return payload

    # -- diagnostics --------------------------------------------------

    def preflight(self, cfg: Config, model: AudioModelConfig) -> list[str]:
        problems: list[str] = []
        python = resolve_python(cfg, model)
        resolved = shutil.which(python) if os.sep not in python else (
            python if Path(python).exists() else None
        )
        if resolved is None:
            problems.append(
                f"TTS interpreter not found: {python}. Set it with "
                "`arc-llama audio set-python /path/to/omnivoice/venv/bin/python`."
            )
            return problems
        if not _can_import(resolved, "omnivoice"):
            problems.append(
                f"{resolved} cannot import `omnivoice`. Install OmniVoice into "
                "that environment, or point `arc-llama audio set-python` at the "
                "virtualenv that has it."
            )
        recipe = model.audio_recipe()
        fmt = recipe.default_response_format or "mp3"
        if fmt not in ("wav", "pcm") and shutil.which("ffmpeg") is None:
            problems.append(
                f"default_response_format is {fmt!r} but ffmpeg is not installed. "
                "The backend will try libsndfile first and fall back to an error; "
                "install ffmpeg or set the default to wav."
            )
        return problems


_REQUEST_OPTIONS = frozenset({
    "num_step", "guidance_scale", "duration", "t_shift", "class_temperature",
})


def _with_path(model: AudioModelConfig, path: str) -> AudioModelConfig:
    """A shallow copy of *model* pointing at *path*, for size measurement."""
    import dataclasses

    return dataclasses.replace(model, path=path)


def default_device(gpu: GPUConfig) -> str:
    """The torch device string for a GPU with no explicit `recipe.device`.

    `xpu` for the Intel cards this project targets — torch's XPU backend goes
    through the same Level Zero runtime as the SYCL llama.cpp build, so it
    honours the `ONEAPI_DEVICE_SELECTOR` pin set below and index 0 inside the
    process is the card the model was bound to.
    """
    backend = Backend(gpu.backend) if gpu.backend else Backend.SYCL
    return "xpu" if backend == Backend.SYCL else "cpu"


def build_env_for(cfg: Config, gpu: GPUConfig, device: str) -> dict[str, str]:
    """Environment for the sidecar, pinned to this model's GPU.

    Reuses the llama.cpp env builder for the `xpu` case purely for the device
    selector and the known-bad-variable stripping; the `GGML_*` settings it also
    sets are inert in a torch process. A non-Intel device gets the ambient
    environment, since neither the SYCL selector nor the Vulkan one means
    anything to it.
    """
    if not device.startswith("xpu"):
        return os.environ.copy()
    arch = Arch(gpu.arch) if gpu.arch else Arch.UNKNOWN
    return build_env(
        profile_for(arch),
        gpu,
        llama_server=cfg.paths.llama_server,
        oneapi_setvars=getattr(cfg.paths, "oneapi_setvars", None),
        backend_override=Backend.SYCL,
    )


def voice_entry(cfg: Config, model: AudioModelConfig, voice: VoiceConfig) -> dict[str, Any]:
    """One voice, as the sidecar's JSON expects it."""
    prompt_file = voice.prompt_file
    if not prompt_file and voice.ref_audio:
        # Cache the encoded reference per model: the encoding is produced by
        # that model's audio tokenizer, so it is not portable between models
        # and a shared filename would hand one model another's prompt.
        prompt_file = str(tts_state_dir(cfg) / "prompts" / f"{model.name}-{voice.name}.pt")
    return {
        "ref_audio": str(Path(voice.ref_audio).expanduser()) if voice.ref_audio else "",
        "ref_text": voice.ref_text,
        "instruct": voice.instruct,
        "language": voice.language,
        "prompt_file": prompt_file,
        "aliases": list(voice.aliases),
    }


def write_voices_file(cfg: Config, model: AudioModelConfig) -> Path:
    """Write this model's voice table and return its path.

    Rewritten on every plan build, and re-read by the sidecar whenever its
    mtime changes, so `arc-llama audio voice add` reaches a running backend
    without a restart.
    """
    recipe = model.audio_recipe()
    payload = {
        "default_voice": recipe.default_voice,
        "voices": {
            v.name: voice_entry(cfg, model, v) for v in cfg.voices_for(model.name)
        },
    }
    directory = tts_state_dir(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model.name}.voices.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _can_import(python: str, module: str) -> bool:
    """Whether *python* can import *module*, without paying for the import.

    `find_spec` only resolves the module on the path; actually importing
    omnivoice would drag in torch and take tens of seconds, which is far too
    slow for `arc-llama doctor`.
    """
    try:
        proc = subprocess.run(
            [python, "-c",
             f"import importlib.util,sys; sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"],
            capture_output=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


register_engine(OmniVoiceEngine())
