"""A minimal OpenAI `/v1/audio/speech` server in front of an OmniVoice model.

Run as a standalone script by whatever interpreter has OmniVoice installed:

    <python> .../arc_llama/tts/omnivoice_server.py --model k2-fsa/OmniVoice \
        --host 127.0.0.1 --port 8090 --device xpu --voices /path/voices.json

**Nothing here may import `arc_llama`.** OmniVoice pulls in torch, transformers
and torchaudio, which arc-llama deliberately does not depend on, so it lives in
its own virtualenv and this file is executed by that virtualenv's interpreter.
The only contract with the parent is the command line, the voices JSON, and the
two HTTP routes below — which is also what makes the engine swappable.

It sits in `sidecars/` rather than beside the engine module for a reason:
running a script puts its own directory at the front of `sys.path`, so a
neighbour named after one of the imports below would shadow it. Living next to
`omnivoice.py` meant `from omnivoice import OmniVoice` resolved to arc-llama's
engine module and failed on a machine with OmniVoice plainly installed.

Only the standard library is used for the HTTP side. The alternative (FastAPI,
matching the parent) would be a dependency the OmniVoice environment has no
reason to carry, and a TTS backend serialises on one GPU anyway, so there is
nothing for an async server to overlap.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

log = logging.getLogger("omnivoice_server")

# OpenAI's documented set. `wav`/`pcm` are written by hand below, the rest go
# through soundfile and then ffmpeg, so which of them actually work depends on
# the environment; the request path reports what it could not produce.
RESPONSE_FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")

# OpenAI's `pcm` is documented as 24 kHz 16-bit signed little-endian mono,
# which is exactly OmniVoice's native output rate — so `pcm` is a raw dump and
# never resamples. A model with a different rate is served correctly in every
# container format that carries a rate, and only `pcm` would mislead a client;
# that case warns at startup rather than silently retuning the audio.
OPENAI_PCM_RATE = 24000

MAX_INPUT_CHARS = 8192


class VoiceBook:
    """The voice table, reloaded from disk whenever the file changes.

    Re-reading on each request is what lets `arc-llama audio voice add` take
    effect without restarting the backend — the alternative, baking voices into
    argv at launch, would make adding a voice cost a model reload (tens of
    seconds and a VRAM round-trip) for what is a one-line config edit.

    Encoded clone prompts are cached in memory, keyed by the voice definition
    itself, so an edited voice re-encodes and an untouched one never does.
    """

    def __init__(self, path: str | None):
        self.path = Path(path).expanduser() if path else None
        self._mtime: float | None = None
        self._data: dict[str, Any] = {}
        self._prompts: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _reload_if_stale(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return
        if mtime == self._mtime:
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Keep serving the last good table: a half-written file during an
            # `arc-llama audio voice add` must not take TTS down.
            log.warning("could not reload voices from %s: %s", self.path, exc)
            return
        self._mtime = mtime
        self._data = raw if isinstance(raw, dict) else {}
        self._prompts.clear()
        log.info("loaded %d voice(s) from %s", len(self.voices), self.path)

    @property
    def voices(self) -> dict[str, Any]:
        entries = self._data.get("voices", {})
        return entries if isinstance(entries, dict) else {}

    @property
    def default_voice(self) -> str:
        return str(self._data.get("default_voice", "") or "")

    def lookup(self, name: str) -> tuple[str, dict[str, Any]] | None:
        """Resolve a `voice` field to (canonical name, definition).

        Matching is exact, then case-insensitive, then over each voice's
        aliases. A client that hardcodes one of OpenAI's voice ids ("alloy")
        gets whatever the user registered under that alias, or falls through to
        the default voice rather than an error — an unknown voice is a much
        worse failure for a speech client than a substituted one.
        """
        with self._lock:
            self._reload_if_stale()
            entries = self.voices
            if name:
                if name in entries:
                    return name, dict(entries[name])
                lowered = {k.lower(): k for k in entries}
                if name.lower() in lowered:
                    key = lowered[name.lower()]
                    return key, dict(entries[key])
                for key, entry in entries.items():
                    aliases = entry.get("aliases") or []
                    if any(str(a).lower() == name.lower() for a in aliases):
                        return key, dict(entry)
            fallback = self.default_voice
            if fallback and fallback in entries:
                return fallback, dict(entries[fallback])
            return None

    def cached_prompt(self, key: str, definition: dict[str, Any]) -> Any | None:
        with self._lock:
            hit = self._prompts.get(key)
            if hit is not None and hit[0] == definition:
                return hit[1]
            return None

    def store_prompt(self, key: str, definition: dict[str, Any], prompt: Any) -> None:
        with self._lock:
            self._prompts[key] = (dict(definition), prompt)


class Engine:
    """The loaded OmniVoice model plus the generation lock around it."""

    def __init__(self, args: argparse.Namespace, voices: VoiceBook):
        self.args = args
        self.voices = voices
        self.model: Any = None
        self.sampling_rate = OPENAI_PCM_RATE
        self.load_error: str | None = None
        # One GPU, one model, and `generate` is not re-entrant. Serialise here
        # rather than in the HTTP layer so a threaded server stays correct.
        self.gpu_lock = threading.Lock()
        self._asr_loaded = False

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        import torch
        from omnivoice import OmniVoice

        dtype = getattr(torch, self.args.dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"unknown torch dtype: {self.args.dtype!r}")
        log.info("loading %s onto %s (%s)", self.args.model, self.args.device, self.args.dtype)
        kwargs: dict[str, Any] = {"device_map": self.args.device, "dtype": dtype}
        if self.args.asr_model:
            kwargs["asr_model_name"] = self.args.asr_model
        if self.args.asr_device:
            kwargs["asr_device"] = self.args.asr_device
        if self.args.quantize:
            self.model = self._load_quantized(kwargs)
        else:
            self.model = OmniVoice.from_pretrained(self.args.model, **kwargs)
        if self.args.compile:
            # Opt-in: the first request pays the compile, and Inductor on XPU is
            # newer than the eager path, so it is not something to impose by
            # default on a backend whose first call is a user waiting for audio.
            log.info("compiling the model graph (first request will be slower)")
            self.model = torch.compile(self.model)
        self.sampling_rate = int(getattr(self.model, "sampling_rate", None) or OPENAI_PCM_RATE)
        if self.sampling_rate != OPENAI_PCM_RATE:
            log.warning(
                "model sampling rate is %d Hz, but OpenAI's `pcm` response format "
                "is defined as %d Hz mono s16le. Clients decoding raw pcm will play "
                "this back at the wrong speed; use response_format=wav instead.",
                self.sampling_rate, OPENAI_PCM_RATE,
            )
        log.info("ready: %s at %d Hz", self.args.model, self.sampling_rate)

    def _load_quantized(self, kwargs: dict[str, Any]) -> Any:
        """Load a torchao-quantized checkpoint.

        `torchao`'s `quantize_()` swaps Linear weights for tensor subclasses, so
        the saved file is a state dict over a structure that does not exist yet
        at load time. It cannot be rebuilt by `from_pretrained` alone: the base
        model has to be materialised first, quantized to create the same module
        shapes, and only then can the weights be read into it. That is why a
        quantized directory holds `quantized_state.pt` rather than the
        `model.safetensors` `from_pretrained` looks for.
        """
        # Checked before the imports and before the base model is materialised:
        # both cost real time (tens of seconds of weights onto the GPU), and
        # neither can turn a bad scheme or an absent checkpoint into a good one.
        if self.args.quantize != "int8":
            raise ValueError(
                f"unsupported quantization {self.args.quantize!r}; expected 'int8'"
            )
        state_path = Path(self.args.quantized_state).expanduser()
        if not state_path.exists():
            # Emphatically not a warning. Continuing here would serve the base
            # model's voice under the fine-tune's name — audio that sounds
            # plausible and is simply the wrong speaker, which is far harder to
            # notice than a backend that refuses to start.
            raise FileNotFoundError(
                f"quantized weights not found at {state_path}. The model is "
                "registered as int8, so refusing to start on the base weights."
            )

        import torch
        from omnivoice import OmniVoice
        from torchao.quantization import Int8WeightOnlyConfig, quantize_

        base = self.args.base_model or self.args.model
        log.info("loading base model %s to rebuild the int8 structure", base)
        model = OmniVoice.from_pretrained(base, **kwargs)

        # Same two submodules the quantization script targeted; quantizing a
        # different set would produce different keys and load nothing.
        quantize_(model.llm, Int8WeightOnlyConfig())
        quantize_(model.audio_heads, Int8WeightOnlyConfig())

        state = torch.load(str(state_path), map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)

        # `strict=False` is required (the audio tokenizer and feature extractor
        # are not in this file), but it also means a checkpoint that matches
        # nothing loads "successfully" and leaves the base weights in place. So
        # check that it actually applied rather than trusting the call.
        applied = len(state) - len(unexpected)
        if applied == 0:
            raise RuntimeError(
                f"{state_path} shares no parameter names with the model — "
                "nothing was loaded. Check that it was quantized from this "
                f"same base model ({base})."
            )
        log.info(
            "loaded %d/%d quantized tensors (%d unexpected, %d left at base)",
            applied, len(state), len(unexpected), len(missing),
        )
        if unexpected:
            log.warning(
                "%d tensor(s) in the checkpoint had no home in the model, e.g. %s",
                len(unexpected), ", ".join(sorted(unexpected)[:3]),
            )
        return model

    def _ensure_asr(self) -> None:
        """Load Whisper, needed only to transcribe a reference clip for us.

        Deferred because it is a second model on the GPU that is pure waste for
        the common case: a voice registered with its `ref_text` never needs it,
        and `arc-llama audio voice add` asks for that text up front.
        """
        if self._asr_loaded:
            return
        self.model.load_asr_model()
        self._asr_loaded = True

    def prompt_for(self, name: str, definition: dict[str, Any]) -> Any | None:
        """Build (or fetch) the encoded clone prompt for one voice.

        Returns None for a voice that has no reference to encode — a designed
        or auto voice — which is the common case and must not cost anything.
        """
        cached = self.voices.cached_prompt(name, definition)
        if cached is not None:
            return cached

        prompt_file = str(definition.get("prompt_file", "") or "")
        ref_audio = str(definition.get("ref_audio", "") or "")
        if not prompt_file and not ref_audio:
            return None

        from omnivoice import VoiceClonePrompt

        # A saved prompt is the encoded reference, so loading one skips both the
        # audio decode and any Whisper pass. Treat a stale/corrupt file as a
        # cache miss and re-encode rather than failing the request.
        if prompt_file and Path(prompt_file).expanduser().exists():
            try:
                prompt = VoiceClonePrompt.load(str(Path(prompt_file).expanduser()))
                self.voices.store_prompt(name, definition, prompt)
                return prompt
            except Exception:
                log.warning("ignoring unreadable voice prompt %s", prompt_file, exc_info=True)

        if not ref_audio:
            return None
        ref_path = Path(ref_audio).expanduser()
        if not ref_path.exists():
            raise FileNotFoundError(f"voice {name!r}: reference audio not found at {ref_path}")
        ref_text = str(definition.get("ref_text", "") or "") or None
        if ref_text is None:
            self._ensure_asr()
        prompt = self.model.create_voice_clone_prompt(str(ref_path), ref_text=ref_text)

        if prompt_file:
            # Persist so the next cold start of this backend skips the encode.
            try:
                target = Path(prompt_file).expanduser()
                target.parent.mkdir(parents=True, exist_ok=True)
                prompt.save(str(target))
            except Exception:
                log.warning("could not cache voice prompt to %s", prompt_file, exc_info=True)
        self.voices.store_prompt(name, definition, prompt)
        return prompt

    def synthesize(self, body: dict[str, Any]) -> tuple[bytes, str]:
        """Turn one OpenAI speech request into encoded audio bytes."""
        text = body.get("input")
        if not isinstance(text, str) or not text.strip():
            raise BadRequestError("'input' must be a non-empty string")
        if len(text) > MAX_INPUT_CHARS:
            raise BadRequestError(f"'input' must be at most {MAX_INPUT_CHARS} characters")

        fmt = str(body.get("response_format") or self.args.default_response_format).lower()
        if fmt not in RESPONSE_FORMATS:
            raise BadRequestError(
                f"unsupported response_format {fmt!r}; expected one of {', '.join(RESPONSE_FORMATS)}"
            )

        kwargs: dict[str, Any] = {"text": text}
        # `instructions` is OpenAI's per-request style field and maps exactly
        # onto OmniVoice's voice-design `instruct`, so a request carrying one
        # overrides whatever style the registered voice implies.
        instruct = str(body.get("instructions") or "")
        language = str(body.get("language") or self.args.default_language or "")

        resolved = self.voices.lookup(str(body.get("voice") or ""))
        voice_name = ""
        if resolved is not None:
            voice_name, definition = resolved
            prompt = self.prompt_for(voice_name, definition)
            if prompt is not None:
                kwargs["voice_clone_prompt"] = prompt
            if not instruct:
                instruct = str(definition.get("instruct", "") or "")
            if not language:
                language = str(definition.get("language", "") or "")
        # Voice cloning wins when both are present: the reference audio already
        # fixes the speaker, and OmniVoice's own guidance is that cloning is the
        # stable mode. Design attributes only apply when there is no clone.
        if instruct and "voice_clone_prompt" not in kwargs:
            kwargs["instruct"] = instruct
        if language:
            kwargs["language"] = language

        speed = body.get("speed")
        if speed is not None:
            try:
                speed = float(speed)
            except (TypeError, ValueError):
                raise BadRequestError("'speed' must be a number") from None
            # OpenAI's documented range. OmniVoice accepts more, but a value
            # outside this band is nearly always a client bug.
            if not 0.25 <= speed <= 4.0:
                raise BadRequestError("'speed' must be between 0.25 and 4.0")
            kwargs["speed"] = speed

        for key, caster in (
            ("num_step", int),
            ("guidance_scale", float),
            ("duration", float),
            ("t_shift", float),
            ("class_temperature", float),
        ):
            if body.get(key) is not None:
                try:
                    kwargs[key] = caster(body[key])
                except (TypeError, ValueError):
                    raise BadRequestError(f"'{key}' must be a number") from None
        kwargs.setdefault("num_step", self.args.num_step)
        if self.args.normalize_text:
            kwargs.setdefault("normalize_text", True)

        log.info(
            "speech: %d chars, voice=%s, format=%s", len(text), voice_name or "(auto)", fmt
        )
        with self.gpu_lock:
            audios = self.model.generate(**kwargs)
        if not audios:
            raise RuntimeError("OmniVoice returned no audio")
        return encode_audio(audios[0], self.sampling_rate, fmt)


class BadRequestError(Exception):
    """A client error worth reporting verbatim; anything else is a 500."""


def _to_int16(samples: Any) -> bytes:
    import numpy as np

    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    # OmniVoice can overshoot 1.0 slightly; clipping first keeps that from
    # wrapping around into loud noise when it becomes int16.
    clipped = np.clip(array, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _wav_bytes(samples: Any, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(_to_int16(samples))
    return buf.getvalue()


# soundfile format/subtype per requested container, for the formats libsndfile
# can write. mp3 and aac depend on the libsndfile build, so both stay in the
# table and simply fall through to ffmpeg when the write raises.
_SOUNDFILE_FORMATS = {
    "flac": ("FLAC", "PCM_16"),
    "opus": ("OGG", "OPUS"),
    "mp3": ("MP3", None),
}

_FFMPEG_ARGS = {
    "mp3": ["-f", "mp3", "-b:a", "128k"],
    "opus": ["-f", "ogg", "-c:a", "libopus", "-b:a", "64k"],
    "aac": ["-f", "adts", "-c:a", "aac", "-b:a", "128k"],
    "flac": ["-f", "flac"],
}

_MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "application/octet-stream",
}


def encode_audio(samples: Any, rate: int, fmt: str) -> tuple[bytes, str]:
    """Encode float samples into *fmt*, returning (bytes, media type).

    `wav` and `pcm` are written from the standard library so the two formats
    every client can decode never depend on an optional codec. The compressed
    formats try libsndfile first (already present, since OmniVoice uses it) and
    fall back to ffmpeg, because libsndfile's mp3/aac support is a build option
    that is off in many wheels — and mp3 is what OpenAI clients ask for by
    default, so failing there would break the common case.
    """
    if fmt == "pcm":
        return _to_int16(samples), _MEDIA_TYPES["pcm"]
    wav = _wav_bytes(samples, rate)
    if fmt == "wav":
        return wav, _MEDIA_TYPES["wav"]

    errors: list[str] = []
    spec = _SOUNDFILE_FORMATS.get(fmt)
    if spec is not None:
        try:
            import soundfile as sf

            container, subtype = spec
            buf = io.BytesIO()
            sf.write(buf, _sf_array(samples), rate, format=container, subtype=subtype)
            return buf.getvalue(), _MEDIA_TYPES[fmt]
        except Exception as exc:
            errors.append(f"soundfile: {exc}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg and fmt in _FFMPEG_ARGS:
        try:
            proc = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                 *_FFMPEG_ARGS[fmt], "pipe:1"],
                input=wav, capture_output=True, check=True, timeout=120,
            )
            return proc.stdout, _MEDIA_TYPES[fmt]
        except subprocess.SubprocessError as exc:
            stderr = getattr(exc, "stderr", b"") or b""
            errors.append(f"ffmpeg: {stderr.decode('utf-8', 'replace').strip() or exc}")
    elif not ffmpeg:
        errors.append("ffmpeg: not installed")

    raise BadRequestError(
        f"cannot encode {fmt!r} in this environment ({'; '.join(errors)}). "
        "Install ffmpeg, or ask for response_format=wav."
    )


def _sf_array(samples: Any):
    import numpy as np

    return np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """A server that reports a dropped connection as one line, not a traceback.

    ``socketserver`` prints a full traceback to stderr whenever a client goes
    away mid-request. The router health-checks every 1.5 s with a 2 s timeout
    while the model loads, and every one of those timeouts is a reset
    connection — so a cold start that legitimately takes minutes would write
    hundreds of tracebacks into the backend log and bury the single message
    that explains a real failure. arc-llama shows the tail of this log when a
    backend fails to start, which is exactly when that matters.
    """

    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, TimeoutError)):
            log.debug("client %s disconnected: %s", client_address, exc)
            return
        log.exception("error handling request from %s", client_address)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    engine: Engine  # set on the server class before serving

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: N803
        log.debug("%s - %s", self.address_string(), fmt % args)


    def _send(self, status: int, body: bytes, media_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            log.debug("client disconnected before the response was written")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": {"message": message, "type": "invalid_request_error"}})

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/health":
            # The parent gates "safe to forward to" on this exact shape, so it
            # must stay 503 until the weights are actually resident — a 200 at
            # bind time would put the whole model load inside a user's first
            # request, past the timeout the router is enforcing.
            if self.engine.ready:
                self._send_json(200, {"status": "ok"})
            elif self.engine.load_error:
                self._send_json(500, {"status": "error", "error": self.engine.load_error})
            else:
                self._send_json(503, {"status": "loading"})
            return
        self._error(404, f"unknown path {path!r}")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path != "/v1/audio/speech":
            self._error(404, f"unknown path {path!r}")
            return
        if not self.engine.ready:
            self._error(503, self.engine.load_error or "model is still loading")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._error(400, "invalid Content-Length")
            return
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            self._error(400, f"invalid JSON: {exc}")
            return
        if not isinstance(body, dict):
            self._error(400, "body must be a JSON object")
            return
        try:
            audio, media_type = self.engine.synthesize(body)
        except BadRequestError as exc:
            self._error(400, str(exc))
            return
        except FileNotFoundError as exc:
            self._error(400, str(exc))
            return
        except Exception as exc:
            log.exception("speech synthesis failed")
            self._error(500, f"speech synthesis failed: {exc}")
            return
        self._send(200, audio, media_type)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="HF repo id or local model directory.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--device", default="xpu", help="torch device_map, e.g. xpu, cuda:0, cpu.")
    p.add_argument("--dtype", default="float16", help="torch dtype name.")
    p.add_argument("--voices", default="", help="Path to the voices JSON written by arc-llama.")
    p.add_argument("--num-step", type=int, default=32, help="Default diffusion steps.")
    p.add_argument("--default-language", default="", help="Language used when none is given.")
    p.add_argument(
        "--default-response-format", default="mp3",
        help="Format used when the request omits response_format.",
    )
    p.add_argument("--asr-model", default="", help="Whisper model for reference auto-transcription.")
    p.add_argument("--asr-device", default="", help="Device for the Whisper model.")
    p.add_argument(
        "--normalize-text", action="store_true",
        help="Expand numbers and dates to their spoken form before synthesis.",
    )
    p.add_argument(
        "--quantize", default="",
        help="Quantization scheme of the checkpoint ('int8'). Empty for an "
        "ordinary unquantized model.",
    )
    p.add_argument(
        "--quantized-state", default="",
        help="Path to the quantized state dict (quantized_state.pt).",
    )
    p.add_argument(
        "--base-model", default="",
        help="Model the quantized checkpoint was derived from, whose structure "
        "is rebuilt before the weights are read in.",
    )
    p.add_argument(
        "--compile", action="store_true",
        help="torch.compile the model after loading.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("ARC_LLAMA_TTS_LOG", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    engine = Engine(args, VoiceBook(args.voices or None))

    class BoundHandler(Handler):
        pass

    BoundHandler.engine = engine
    httpd = QuietThreadingHTTPServer((args.host, args.port), BoundHandler)

    # Bind before loading so /health can answer 503 "loading" instead of
    # refusing the connection: the router distinguishes a backend that is slow
    # from one that never came up, and only the former is worth waiting on.
    def _load() -> None:
        try:
            engine.load()
        except Exception as exc:
            log.exception("model load failed")
            engine.load_error = f"{type(exc).__name__}: {exc}"

    loader = threading.Thread(target=_load, name="omnivoice-load", daemon=True)
    loader.start()

    log.info("listening on http://%s:%d", args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
