"""Audio integration: registry, launch plans, residency, and both endpoints.

Covers speech-to-text on llama-server and text-to-speech through the pluggable
TTS engine layer, plus the OmniVoice sidecar's own request handling.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arc_llama.config import (
    DEFAULT_ASR_CTX,
    AudioModelConfig,
    Config,
    GPUConfig,
    ModelConfig,
    PathsConfig,
    ServerConfig,
    VoiceConfig,
    load_config,
)
from arc_llama.launcher import build_audio_plan, resolve_binary
from arc_llama.models import add_audio_model, add_voice, is_audio_gguf
from arc_llama.router import Router, _estimate_audio_vram_mb
from arc_llama.server import create_app, strip_asr_markers
from arc_llama.server_caps import ServerCaps
from arc_llama.tts import engine_names, get_engine, require_engine
from arc_llama.tts.omnivoice import write_voices_file


def _sidecar():
    """Load the sidecar the way arc-llama launches it: by path.

    It deliberately is not importable as `arc_llama.tts.<something>` — it lives
    in `sidecars/`, which has no `__init__.py`, because it runs under a foreign
    interpreter that has no arc_llama at all. Loading it from its file here
    exercises that same standalone contract: if it ever grows an arc_llama
    import, every test below fails.
    """
    import importlib.util

    from arc_llama.tts import omnivoice

    spec = importlib.util.spec_from_file_location(
        "omnivoice_server", omnivoice.SERVER_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SIDECAR = _sidecar()
Engine = _SIDECAR.Engine
VoiceBook = _SIDECAR.VoiceBook
BadRequestError = _SIDECAR.BadRequestError
build_parser = _SIDECAR.build_parser
encode_audio = _SIDECAR.encode_audio


def _gpu(**kw):
    defaults = dict(
        pci_slot="0000:03:00.0",
        sycl_index=0,
        arch="battlemage",
        vram_mb=24480,
        name="Arc Pro B60",
        backend="sycl",
    )
    defaults.update(kw)
    return GPUConfig(**defaults)


def _audio_model(tmp_path, recipe=None, **kw):
    """An ASR model with weights and a projector on disk."""
    weights = tmp_path / "Qwen3-ASR-0.6B-Q8_0.gguf"
    weights.write_bytes(b"GGUF")
    mmproj = tmp_path / "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"
    mmproj.write_bytes(b"GGUF")
    full_recipe = {"mmproj": str(mmproj)}
    if recipe is not None:
        full_recipe.update(recipe)
    defaults = dict(
        name="qwen3-asr",
        path=str(weights),
        port=18090,
        gpu_pci_slot="0000:03:00.0",
        engine="llamacpp",
        recipe=full_recipe,
        task="asr",
    )
    defaults.update(kw)
    return AudioModelConfig(**defaults)


def _tts_model(tmp_path, recipe=None, **kw):
    """An OmniVoice-engine TTS model backed by a real directory on disk."""
    model_dir = tmp_path / "OmniVoice"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "model.safetensors").write_bytes(b"0" * 2048)
    defaults = dict(
        name="omnivoice",
        path=str(model_dir),
        port=18091,
        gpu_pci_slot="0000:03:00.0",
        engine="omnivoice",
        task="tts",
        recipe=dict(recipe or {}),
    )
    defaults.update(kw)
    return AudioModelConfig(**defaults)


def _fake_hf_cache(tmp_path, repo_id, *, blob_bytes, stale_revision_bytes=0):
    """A Hugging Face cache laid out the way the real one is.

    Weights live once in `blobs/` and are exposed through `snapshots/<rev>/`
    as symlinks — the detail that made a naive directory walk double every
    model's measured size.
    """
    cache = tmp_path / "hf-hub"
    repo = cache / ("models--" + repo_id.replace("/", "--"))
    blobs = repo / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    (repo / "refs").mkdir(exist_ok=True)
    (repo / "refs" / "main").write_text("rev-current", encoding="utf-8")

    blob = blobs / "sha-current"
    blob.write_bytes(b"\0" * blob_bytes)
    snapshot = repo / "snapshots" / "rev-current"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "model.safetensors").symlink_to(blob)
    (snapshot / "config.json").symlink_to(blob)  # a second link to the same bytes

    if stale_revision_bytes:
        stale_blob = blobs / "sha-stale"
        stale_blob.write_bytes(b"\0" * stale_revision_bytes)
        old = repo / "snapshots" / "rev-old"
        old.mkdir(parents=True, exist_ok=True)
        (old / "model.safetensors").symlink_to(stale_blob)
    return cache


def _fake_binary(tmp_path, name, help_text=""):
    """An executable file, so resolve_binary() finds something real.

    Windows has no executable bit and does not run extensionless files: a
    PATH lookup there only considers names ending in a PATHEXT suffix, so a
    bare `llama-server` is invisible to `shutil.which` no matter where it
    sits. Give it a real extension on Windows and a shebang on POSIX, so the
    helper models an actual install on both.
    """
    if sys.platform == "win32":
        p = tmp_path / f"{name}.cmd"
        body = "\r\n".join(f"echo {line}" for line in (help_text.splitlines() or [""]))
        p.write_text(f"@echo off\r\n{body}\r\n")
        return p
    p = tmp_path / name
    p.write_text(f"#!/bin/sh\ncat <<'EOF'\n{help_text}\nEOF\n")
    p.chmod(0o755)
    return p


def _fake_llama_server(tmp_path):
    """A stand-in whose --help advertises a multimodal build."""
    return _fake_binary(
        tmp_path,
        "llama-server",
        "-fa, --flash-attn FA  set Flash Attention use (default: 'auto')\n--mmproj FILE\n",
    )


def _fake_python(tmp_path):
    return _fake_binary(tmp_path, "python3-tts")


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


class TestConfigRoundTrip:
    def test_audio_models_survive_save_and_load(self, tmp_path):
        cfg = Config(gpus=[_gpu()], audio_models=[_audio_model(tmp_path, aliases=["whisper-1"])])
        path = tmp_path / "config.toml"
        cfg.save(path)

        loaded = load_config(path)
        assert len(loaded.audio_models) == 1
        entry = loaded.audio_models[0]
        assert entry.name == "qwen3-asr"
        assert entry.engine == "llamacpp"
        assert entry.aliases == ["whisper-1"]
        assert entry.always_resident is True

    def test_recipe_round_trips_as_a_subtable(self, tmp_path):
        """Launch knobs live under [audio_models.recipe], as they do for LLMs."""
        cfg = Config(
            gpus=[_gpu()],
            audio_models=[_audio_model(tmp_path, recipe={"ctx": 8192})],
        )
        path = tmp_path / "config.toml"
        cfg.save(path)

        text = path.read_text(encoding="utf-8")
        assert "[audio_models.recipe]" in text
        entry = load_config(path).audio_models[0]
        assert entry.audio_recipe().ctx == 8192
        assert entry.audio_recipe().mmproj.endswith("mmproj-Qwen3-ASR-0.6B-Q8_0.gguf")

    def test_tts_recipe_round_trips(self, tmp_path):
        cfg = Config(
            gpus=[_gpu()],
            audio_models=[
                _tts_model(
                    tmp_path,
                    recipe={
                        "device": "xpu",
                        "default_voice": "glados",
                        "options": {"num_step": 16},
                    },
                )
            ],
        )
        path = tmp_path / "config.toml"
        cfg.save(path)

        recipe = load_config(path).audio_models[0].audio_recipe()
        assert recipe.device == "xpu"
        assert recipe.default_voice == "glados"
        assert recipe.options == {"num_step": 16}
        assert recipe.default_response_format == "mp3"

    def test_voices_survive_save_and_load(self, tmp_path):
        clip = tmp_path / "ref.wav"
        clip.write_bytes(b"RIFF")
        cfg = Config(
            gpus=[_gpu()],
            voices=[
                VoiceConfig(
                    name="glados",
                    ref_audio=str(clip),
                    ref_text="the transcript",
                    aliases=["alloy"],
                )
            ],
        )
        path = tmp_path / "config.toml"
        cfg.save(path)

        loaded = load_config(path)
        assert loaded.voices[0].name == "glados"
        assert loaded.voices[0].ref_text == "the transcript"
        assert loaded.find_voice("alloy").name == "glados"

    def test_launch_knobs_in_the_body_are_migrated_into_the_recipe(self, tmp_path):
        """The first cut put ctx/mmproj in the body; don't make anyone re-edit."""
        path = tmp_path / "config.toml"
        path.write_text(
            "version = 1\n\n"
            "[[audio_models]]\n"
            'name = "qwen3-asr"\n'
            'path = "/models/asr.gguf"\n'
            "port = 18090\n"
            'gpu_pci_slot = "0000:03:00.0"\n'
            'mmproj = "/models/mmproj-asr.gguf"\n'
            "ctx = 2048\n",
            encoding="utf-8",
        )
        entry = load_config(path).audio_models[0]
        assert entry.audio_recipe().ctx == 2048
        assert entry.audio_recipe().mmproj == "/models/mmproj-asr.gguf"

    def test_ctx_defaults_to_something_sane(self, tmp_path):
        """Not to the GGUF's 65536, which is ~7 GB of KV for a 1.7B model."""
        assert _audio_model(tmp_path).audio_recipe().ctx == DEFAULT_ASR_CTX
        assert DEFAULT_ASR_CTX <= 8192

    def test_old_config_without_audio_table_still_loads(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text("version = 1\n[server]\nport = 11437\n", encoding="utf-8")
        loaded = load_config(path)
        assert loaded.audio_models == []
        assert loaded.voices == []

    def test_find_audio_model_matches_alias(self, tmp_path):
        cfg = Config(audio_models=[_audio_model(tmp_path, aliases=["whisper-1"])])
        assert cfg.find_audio_model("whisper-1").name == "qwen3-asr"
        # The LLM registry must not see it, or `scan`/tune would pick it up.
        assert cfg.find_model("whisper-1") is None
        assert cfg.find_any_model("whisper-1").name == "qwen3-asr"

    def test_voices_are_scoped_to_their_models(self, tmp_path):
        cfg = Config(
            voices=[
                VoiceConfig(name="everywhere"),
                VoiceConfig(name="only-here", models=["omnivoice"]),
                VoiceConfig(name="elsewhere", models=["other"]),
            ]
        )
        names = [v.name for v in cfg.voices_for("omnivoice")]
        assert names == ["everywhere", "only-here"]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _weights_and_mmproj(tmp_path):
    weights = tmp_path / "Qwen3-ASR-0.6B-Q8_0.gguf"
    weights.write_bytes(b"GGUF")
    mmproj = tmp_path / "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"
    mmproj.write_bytes(b"GGUF")
    return weights, mmproj


class TestAddAudioModel:
    def test_registers_with_auto_port(self, tmp_path):
        cfg = Config(gpus=[_gpu()])
        weights, mmproj = _weights_and_mmproj(tmp_path)
        entry = add_audio_model(
            cfg,
            name="qwen3-asr",
            path=str(weights),
            mmproj=str(mmproj),
            gpu_pci_slot="0000:03:00.0",
        )
        assert entry.port >= 18080
        assert cfg.audio_models == [entry]
        assert entry.always_resident is True

    def test_llamacpp_is_the_default_engine(self, tmp_path):
        cfg = Config(gpus=[_gpu()])
        weights, mmproj = _weights_and_mmproj(tmp_path)
        entry = add_audio_model(
            cfg,
            name="qwen3-asr",
            path=str(weights),
            mmproj=str(mmproj),
            gpu_pci_slot="0000:03:00.0",
        )
        assert entry.engine == "llamacpp"
        assert entry.strip_asr_markers is True

    def test_asr_requires_an_mmproj(self, tmp_path):
        """Without a projector llama-server loads a text LLM and invents words."""
        cfg = Config(gpus=[_gpu()])
        weights, _ = _weights_and_mmproj(tmp_path)
        with pytest.raises(ValueError, match="mmproj"):
            add_audio_model(
                cfg,
                name="qwen3-asr",
                path=str(weights),
                gpu_pci_slot="0000:03:00.0",
            )

    def test_unknown_asr_engine_is_refused(self, tmp_path):
        cfg = Config(gpus=[_gpu()])
        weights, mmproj = _weights_and_mmproj(tmp_path)
        with pytest.raises(ValueError, match="engine must be"):
            add_audio_model(
                cfg,
                name="qwen3-asr",
                path=str(weights),
                mmproj=str(mmproj),
                engine="whisper.cpp",
                gpu_pci_slot="0000:03:00.0",
            )

    def test_unknown_tts_engine_lists_the_registered_ones(self, tmp_path):
        cfg = Config(gpus=[_gpu()])
        model_dir = tmp_path / "voice"
        model_dir.mkdir()
        with pytest.raises(ValueError, match="omnivoice"):
            add_audio_model(
                cfg,
                name="some-tts",
                path=str(model_dir),
                engine="piper",
                task="tts",
                gpu_pci_slot="0000:03:00.0",
            )

    def test_tts_accepts_a_hugging_face_repo_id(self, tmp_path):
        """OmniVoice resolves the repo itself; a path check would refuse it."""
        cfg = Config(gpus=[_gpu()])
        entry = add_audio_model(
            cfg,
            name="omnivoice",
            path="k2-fsa/OmniVoice",
            engine="omnivoice",
            task="tts",
            gpu_pci_slot="0000:03:00.0",
        )
        assert entry.path == "k2-fsa/OmniVoice"

    def test_asr_still_needs_a_real_path(self, tmp_path):
        cfg = Config(gpus=[_gpu()])
        _, mmproj = _weights_and_mmproj(tmp_path)
        with pytest.raises(FileNotFoundError):
            add_audio_model(
                cfg,
                name="qwen3-asr",
                path=str(tmp_path / "nope.gguf"),
                mmproj=str(mmproj),
                gpu_pci_slot="0000:03:00.0",
            )

    def test_recipe_overrides_land_in_the_recipe(self, tmp_path):
        cfg = Config(gpus=[_gpu()])
        entry = add_audio_model(
            cfg,
            name="omnivoice",
            path="k2-fsa/OmniVoice",
            engine="omnivoice",
            task="tts",
            gpu_pci_slot="0000:03:00.0",
            recipe_overrides={"device": "cpu", "dtype": None, "options": {"num_step": 16}},
        )
        assert entry.recipe["device"] == "cpu"
        assert entry.recipe["options"] == {"num_step": 16}
        # None means "not given" and must not overwrite the dataclass default.
        assert "dtype" not in entry.recipe

    def test_port_does_not_collide_with_an_llm(self, tmp_path):
        """Both registries share one router process map, so ports are one pool."""
        cfg = Config(
            gpus=[_gpu()],
            models=[
                ModelConfig(
                    name="qwen",
                    path=str(tmp_path / "q.gguf"),
                    port=18080,
                    gpu_pci_slot="0000:03:00.0",
                )
            ],
        )
        weights, mmproj = _weights_and_mmproj(tmp_path)
        entry = add_audio_model(
            cfg,
            name="qwen3-asr",
            path=str(weights),
            mmproj=str(mmproj),
            gpu_pci_slot="0000:03:00.0",
        )
        assert entry.port != 18080

    def test_name_collision_with_an_llm_is_refused(self, tmp_path):
        cfg = Config(
            gpus=[_gpu()],
            models=[
                ModelConfig(
                    name="shared",
                    path=str(tmp_path / "q.gguf"),
                    port=18080,
                    gpu_pci_slot="0000:03:00.0",
                )
            ],
        )
        weights, mmproj = _weights_and_mmproj(tmp_path)
        with pytest.raises(ValueError, match="already registered"):
            add_audio_model(
                cfg,
                name="shared",
                path=str(weights),
                mmproj=str(mmproj),
                gpu_pci_slot="0000:03:00.0",
            )


class TestAddVoice:
    def test_clone_voice_resolves_the_reference(self, tmp_path):
        cfg = Config()
        clip = tmp_path / "ref.wav"
        clip.write_bytes(b"RIFF")
        voice = add_voice(cfg, name="glados", ref_audio=str(clip), ref_text="hello")
        assert voice.ref_audio == str(clip.resolve())
        assert cfg.voices == [voice]

    def test_design_voice_needs_no_reference(self, tmp_path):
        cfg = Config()
        voice = add_voice(cfg, name="narrator", instruct="male, low pitch")
        assert voice.instruct == "male, low pitch"
        assert voice.ref_audio == ""

    def test_a_voice_with_neither_is_refused(self):
        """Otherwise a forgotten flag registers a name that does nothing."""
        with pytest.raises(ValueError, match="ref-audio"):
            add_voice(Config(), name="empty")

    def test_auto_voice_needs_no_reference_or_attributes(self):
        """A fine-tune's speaker is in the weights, not in a prompt."""
        cfg = Config()
        voice = add_voice(cfg, name="glados", auto=True)
        assert voice.ref_audio == ""
        assert voice.instruct == ""
        assert cfg.find_voice("glados") is voice

    def test_auto_cannot_be_combined_with_a_prompt(self):
        with pytest.raises(ValueError, match="cannot be combined"):
            add_voice(Config(), name="glados", auto=True, instruct="female")

    def test_missing_reference_audio_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            add_voice(Config(), name="glados", ref_audio=str(tmp_path / "nope.wav"))

    def test_duplicate_name_is_refused(self, tmp_path):
        cfg = Config()
        add_voice(cfg, name="narrator", instruct="male")
        with pytest.raises(ValueError, match="already registered"):
            add_voice(cfg, name="narrator", instruct="female")

    def test_scoping_to_an_unknown_model_is_refused(self):
        with pytest.raises(ValueError, match="Unknown audio model"):
            add_voice(Config(), name="narrator", instruct="male", models=["nope"])


# ---------------------------------------------------------------------------
# Launch plan — transcription
# ---------------------------------------------------------------------------


class TestBuildLlamacppAudioPlan:
    def _cfg(self, tmp_path, **paths):
        binary = _fake_llama_server(tmp_path)
        defaults = dict(llama_server=str(binary), state_dir=str(tmp_path / "state"))
        defaults.update(paths)
        return Config(paths=PathsConfig(**defaults), gpus=[_gpu()])

    def test_argv_carries_weights_and_projector(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "arc_llama.launcher.probe_server_caps",
            lambda _p: ServerCaps(supports_mmproj=True, probed=True),
        )
        cfg = self._cfg(tmp_path)
        model = _audio_model(tmp_path, recipe={"ctx": 8192})
        plan = build_audio_plan(cfg, model, cfg.gpus[0])

        assert Path(plan.argv[0]).stem == "llama-server"
        assert "--mmproj" in plan.argv
        assert plan.argv[plan.argv.index("--mmproj") + 1].endswith(
            "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"
        )
        assert plan.argv[plan.argv.index("-m") + 1].endswith("Qwen3-ASR-0.6B-Q8_0.gguf")
        assert plan.argv[plan.argv.index("-c") + 1] == "8192"
        assert plan.health_url == "http://127.0.0.1:18090/health"

    def test_context_is_always_pinned(self, tmp_path, monkeypatch):
        """llama-server's -c default is 0 = the GGUF's trained context.

        Qwen3-ASR advertises 65536, which buys a ~7 GB KV cache for a 1.7B
        model. Never leave it unset.
        """
        monkeypatch.setattr(
            "arc_llama.launcher.probe_server_caps",
            lambda _p: ServerCaps(supports_mmproj=True, probed=True),
        )
        cfg = self._cfg(tmp_path)
        plan = build_audio_plan(cfg, _audio_model(tmp_path), cfg.gpus[0])
        assert "-c" in plan.argv
        assert plan.argv[plan.argv.index("-c") + 1] == str(DEFAULT_ASR_CTX)
        # One slot, so the context budget is not silently multiplied.
        assert plan.argv[plan.argv.index("-np") + 1] == "1"

    def test_sycl_gpu_keeps_the_sycl_path(self, tmp_path, monkeypatch):
        """The reason ASR runs on llama.cpp: it is the only SYCL-capable one."""
        monkeypatch.setattr(
            "arc_llama.launcher.probe_server_caps",
            lambda _p: ServerCaps(supports_mmproj=True, probed=True),
        )
        cfg = self._cfg(tmp_path)
        plan = build_audio_plan(cfg, _audio_model(tmp_path), cfg.gpus[0])
        assert plan.env["ONEAPI_DEVICE_SELECTOR"] == "level_zero:0"

    def test_missing_mmproj_is_a_clear_error(self, tmp_path):
        cfg = self._cfg(tmp_path)
        model = _audio_model(tmp_path, recipe={"mmproj": ""})
        with pytest.raises(RuntimeError, match="mmproj"):
            build_audio_plan(cfg, model, cfg.gpus[0])

    def test_mmproj_path_that_does_not_exist_is_refused(self, tmp_path):
        cfg = self._cfg(tmp_path)
        model = _audio_model(tmp_path, recipe={"mmproj": str(tmp_path / "nope.gguf")})
        with pytest.raises(RuntimeError, match="not found"):
            build_audio_plan(cfg, model, cfg.gpus[0])

    def test_binary_without_multimodal_support_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "arc_llama.launcher.probe_server_caps",
            lambda _p: ServerCaps(supports_mmproj=False, probed=True),
        )
        cfg = self._cfg(tmp_path)
        with pytest.raises(RuntimeError, match="multimodal"):
            build_audio_plan(cfg, _audio_model(tmp_path), cfg.gpus[0])


# ---------------------------------------------------------------------------
# Launch plan — speech
# ---------------------------------------------------------------------------


class TestTTSEngineRegistry:
    def test_omnivoice_is_registered(self):
        assert "omnivoice" in engine_names()
        assert require_engine("omnivoice").speech_path == "/v1/audio/speech"

    def test_unknown_engine_names_the_known_ones(self):
        assert get_engine("piper") is None
        with pytest.raises(ValueError, match="omnivoice"):
            require_engine("piper")

    def test_a_new_engine_needs_no_edits_elsewhere(self, tmp_path):
        """The point of the registry: register, and the whole path works."""
        from arc_llama.launcher import LaunchPlan
        from arc_llama.tts.base import _ENGINES, TTSEngine, register_engine

        class ToyEngine(TTSEngine):
            name = "toy"
            speech_path = "/speak"

            def build_plan(self, cfg, model, gpu, host="127.0.0.1"):
                return LaunchPlan(argv=["toy"], env={}, backend_url=f"http://{host}:1")

            def build_payload(self, model, body):
                return {"say": body["input"]}

        register_engine(ToyEngine())
        try:
            cfg = Config(gpus=[_gpu()], paths=PathsConfig(state_dir=str(tmp_path)))
            model = _tts_model(tmp_path, engine="toy")
            plan = build_audio_plan(cfg, model, cfg.gpus[0])
            assert plan.argv == ["toy"]
            assert require_engine("toy").build_payload(model, {"input": "hi"}) == {"say": "hi"}
        finally:
            _ENGINES.pop("toy", None)


class TestBuildOmniVoicePlan:
    def _cfg(self, tmp_path, **paths):
        defaults = dict(
            llama_server=str(_fake_llama_server(tmp_path)),
            state_dir=str(tmp_path / "state"),
            tts_python=str(_fake_python(tmp_path)),
        )
        defaults.update(paths)
        return Config(paths=PathsConfig(**defaults), gpus=[_gpu()])

    def test_argv_runs_the_sidecar_under_the_tts_interpreter(self, tmp_path):
        cfg = self._cfg(tmp_path)
        plan = build_audio_plan(cfg, _tts_model(tmp_path), cfg.gpus[0])

        assert Path(plan.argv[0]).stem == "python3-tts"
        assert plan.argv[1].endswith("omnivoice_server.py")
        assert plan.argv[plan.argv.index("--port") + 1] == "18091"
        assert plan.health_url == "http://127.0.0.1:18091/health"

    def test_recipe_python_beats_the_global_one(self, tmp_path):
        cfg = self._cfg(tmp_path)
        other = _fake_binary(tmp_path, "other-python")
        plan = build_audio_plan(
            cfg, _tts_model(tmp_path, recipe={"python": str(other)}), cfg.gpus[0]
        )
        assert plan.argv[0] == str(other)

    def test_sycl_gpu_defaults_to_xpu_and_keeps_the_device_pin(self, tmp_path):
        """torch's XPU backend goes through the same Level Zero runtime."""
        cfg = self._cfg(tmp_path)
        plan = build_audio_plan(cfg, _tts_model(tmp_path), cfg.gpus[0])
        assert plan.argv[plan.argv.index("--device") + 1] == "xpu"
        assert plan.env["ONEAPI_DEVICE_SELECTOR"] == "level_zero:0"

    def test_explicit_cpu_device_drops_the_gpu_pin(self, tmp_path):
        cfg = self._cfg(tmp_path)
        plan = build_audio_plan(cfg, _tts_model(tmp_path, recipe={"device": "cpu"}), cfg.gpus[0])
        assert plan.argv[plan.argv.index("--device") + 1] == "cpu"
        assert "ONEAPI_DEVICE_SELECTOR" not in plan.env

    def test_engine_options_become_flags(self, tmp_path):
        cfg = self._cfg(tmp_path)
        model = _tts_model(
            tmp_path, recipe={"options": {"num_step": 16, "normalize_text": True}}
        )
        plan = build_audio_plan(cfg, model, cfg.gpus[0])
        assert plan.argv[plan.argv.index("--num-step") + 1] == "16"
        assert "--normalize-text" in plan.argv

    def test_health_budget_is_larger_than_an_llm_cold_start(self, tmp_path):
        """A first run downloads several GB before it can answer /health."""
        cfg = self._cfg(tmp_path)
        plan = build_audio_plan(cfg, _tts_model(tmp_path), cfg.gpus[0])
        assert plan.health_timeout is not None
        assert plan.health_timeout >= 600

    def test_a_repo_id_is_passed_through_unresolved(self, tmp_path):
        cfg = self._cfg(tmp_path)
        model = _tts_model(tmp_path, path="k2-fsa/OmniVoice")
        plan = build_audio_plan(cfg, model, cfg.gpus[0])
        assert plan.argv[plan.argv.index("--model") + 1] == "k2-fsa/OmniVoice"


class TestQuantizedModel:
    """A torchao int8 checkpoint is a state dict, not a loadable model dir."""

    def _cfg(self, tmp_path):
        return Config(
            paths=PathsConfig(
                llama_server=str(_fake_llama_server(tmp_path)),
                state_dir=str(tmp_path / "state"),
                tts_python=str(_fake_python(tmp_path)),
            ),
            gpus=[_gpu()],
        )

    def _quantized_dir(self, tmp_path):
        d = tmp_path / "OmniVoice_INT8"
        d.mkdir(exist_ok=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
        (d / "quantized_state.pt").write_bytes(b"\x80\x02")  # contents never read here
        return d

    def test_a_quantized_directory_is_detected(self, tmp_path):
        from arc_llama.tts.omnivoice import quantized_state_path

        d = self._quantized_dir(tmp_path)
        assert quantized_state_path(str(d)) == d / "quantized_state.pt"

    def test_an_ordinary_model_is_not_flagged(self, tmp_path):
        from arc_llama.tts.omnivoice import quantized_state_path

        assert quantized_state_path(str(_tts_model(tmp_path).path)) is None
        assert quantized_state_path("k2-fsa/OmniVoice") is None
        assert quantized_state_path("") is None

    def test_plan_rebuilds_from_the_base_model(self, tmp_path):
        """from_pretrained cannot read the quantized dir, so a base is required."""
        cfg = self._cfg(tmp_path)
        d = self._quantized_dir(tmp_path)
        plan = build_audio_plan(cfg, _tts_model(tmp_path, path=str(d)), cfg.gpus[0])

        assert plan.argv[plan.argv.index("--quantize") + 1] == "int8"
        assert plan.argv[plan.argv.index("--quantized-state") + 1] == str(
            d / "quantized_state.pt"
        )
        assert plan.argv[plan.argv.index("--base-model") + 1] == "k2-fsa/OmniVoice"

    def test_base_model_is_overridable(self, tmp_path):
        cfg = self._cfg(tmp_path)
        d = self._quantized_dir(tmp_path)
        model = _tts_model(
            tmp_path, path=str(d), recipe={"options": {"base_model": "me/OmniVoice-ft"}}
        )
        plan = build_audio_plan(cfg, model, cfg.gpus[0])
        assert plan.argv[plan.argv.index("--base-model") + 1] == "me/OmniVoice-ft"

    def test_quantized_defaults_to_bfloat16(self, tmp_path):
        """int8 checkpoints come from a bf16 base; fp16 mismatches on first matmul."""
        cfg = self._cfg(tmp_path)
        d = self._quantized_dir(tmp_path)
        plan = build_audio_plan(cfg, _tts_model(tmp_path, path=str(d)), cfg.gpus[0])
        assert plan.argv[plan.argv.index("--dtype") + 1] == "bfloat16"

    def test_unquantized_still_defaults_to_float16(self, tmp_path):
        cfg = self._cfg(tmp_path)
        plan = build_audio_plan(cfg, _tts_model(tmp_path), cfg.gpus[0])
        assert plan.argv[plan.argv.index("--dtype") + 1] == "float16"

    def test_explicit_dtype_wins(self, tmp_path):
        cfg = self._cfg(tmp_path)
        d = self._quantized_dir(tmp_path)
        model = _tts_model(tmp_path, path=str(d), recipe={"dtype": "float32"})
        plan = build_audio_plan(cfg, model, cfg.gpus[0])
        assert plan.argv[plan.argv.index("--dtype") + 1] == "float32"

    def test_compile_is_opt_in(self, tmp_path):
        cfg = self._cfg(tmp_path)
        assert "--compile" not in build_audio_plan(
            cfg, _tts_model(tmp_path), cfg.gpus[0]
        ).argv
        model = _tts_model(tmp_path, recipe={"options": {"compile": True}})
        assert "--compile" in build_audio_plan(cfg, model, cfg.gpus[0]).argv

    def test_missing_weights_refuse_to_start_on_the_base_model(self, tmp_path):
        """Serving stock OmniVoice under a fine-tune's name is worse than failing.

        The upstream reference server warns and continues here, which yields
        audio that sounds plausible and is simply the wrong speaker.
        """

        args = build_parser().parse_args([
            "--model", "m", "--port", "1",
            "--quantize", "int8",
            "--quantized-state", str(tmp_path / "absent.pt"),
        ])
        engine = Engine(args, VoiceBook(None))
        with pytest.raises(FileNotFoundError, match="refusing to start"):
            engine._load_quantized({})


class TestVoicesFile:
    def _cfg(self, tmp_path, voices):
        return Config(
            paths=PathsConfig(state_dir=str(tmp_path / "state")),
            gpus=[_gpu()],
            voices=voices,
        )

    def test_voices_are_written_for_the_backend(self, tmp_path):
        clip = tmp_path / "ref.wav"
        clip.write_bytes(b"RIFF")
        cfg = self._cfg(
            tmp_path,
            [VoiceConfig(name="glados", ref_audio=str(clip), ref_text="hi", aliases=["alloy"])],
        )
        model = _tts_model(tmp_path, recipe={"default_voice": "glados"})
        written = json.loads(write_voices_file(cfg, model).read_text(encoding="utf-8"))

        assert written["default_voice"] == "glados"
        assert written["voices"]["glados"]["ref_text"] == "hi"
        assert written["voices"]["glados"]["aliases"] == ["alloy"]

    def test_prompt_cache_is_per_model(self, tmp_path):
        """The encoding comes from the model's own tokenizer, so it can't be shared."""
        clip = tmp_path / "ref.wav"
        clip.write_bytes(b"RIFF")
        cfg = self._cfg(tmp_path, [VoiceConfig(name="glados", ref_audio=str(clip))])
        first = json.loads(
            write_voices_file(cfg, _tts_model(tmp_path)).read_text(encoding="utf-8")
        )
        second = json.loads(
            write_voices_file(cfg, _tts_model(tmp_path, name="other", port=18092)).read_text(
                encoding="utf-8"
            )
        )
        assert (
            first["voices"]["glados"]["prompt_file"]
            != second["voices"]["glados"]["prompt_file"]
        )

    def test_an_auto_voice_carries_no_prompt_to_the_backend(self, tmp_path):
        """Resolvable by name, but it must not inject instruct/clone/language."""
        cfg = self._cfg(tmp_path, [VoiceConfig(name="glados")])
        written = json.loads(
            write_voices_file(cfg, _tts_model(tmp_path)).read_text(encoding="utf-8")
        )
        entry = written["voices"]["glados"]
        assert entry["ref_audio"] == ""
        assert entry["instruct"] == ""
        assert entry["prompt_file"] == ""

    def test_voices_scoped_to_another_model_are_left_out(self, tmp_path):
        cfg = self._cfg(
            tmp_path,
            [
                VoiceConfig(name="mine", instruct="male", models=["omnivoice"]),
                VoiceConfig(name="theirs", instruct="female", models=["other"]),
            ],
        )
        written = json.loads(
            write_voices_file(cfg, _tts_model(tmp_path)).read_text(encoding="utf-8")
        )
        assert list(written["voices"]) == ["mine"]


class TestSpeechPayload:
    def test_model_defaults_fill_in_missing_fields(self, tmp_path):
        model = _tts_model(
            tmp_path,
            recipe={
                "default_voice": "glados",
                "default_language": "English",
                "default_response_format": "wav",
                "options": {"num_step": 16, "asr_model": "whisper"},
            },
        )
        payload = require_engine("omnivoice").build_payload(model, {"input": "hi"})
        assert payload["voice"] == "glados"
        assert payload["language"] == "English"
        assert payload["response_format"] == "wav"
        assert payload["num_step"] == 16
        # Load-time knobs are flags, not request fields.
        assert "asr_model" not in payload

    def test_request_fields_win_over_defaults(self, tmp_path):
        model = _tts_model(
            tmp_path, recipe={"default_voice": "glados", "options": {"num_step": 16}}
        )
        payload = require_engine("omnivoice").build_payload(
            model, {"input": "hi", "voice": "narrator", "num_step": 32}
        )
        assert payload["voice"] == "narrator"
        assert payload["num_step"] == 32

    def test_the_model_field_is_not_forwarded(self, tmp_path):
        """The sidecar serves exactly one model and has no id to match."""
        payload = require_engine("omnivoice").build_payload(
            _tts_model(tmp_path), {"input": "hi", "model": "omnivoice"}
        )
        assert "model" not in payload


# ---------------------------------------------------------------------------
# Residency policy
# ---------------------------------------------------------------------------


class FakeRunningServer:
    def __init__(self):
        self.is_running = True
        self.ready = True
        self.stopped = False

    async def astop(self, drain_seconds: float = 3.0):
        self.stopped = True
        self.is_running = False


class TestResidencyPolicy:
    def _router(self, tmp_path, *, always_resident=True):
        cfg = Config(
            server=ServerConfig(single_resident=True),
            paths=PathsConfig(
                llama_server=str(_fake_llama_server(tmp_path)),
                state_dir=str(tmp_path / "state"),
            ),
            gpus=[_gpu()],
            models=[
                ModelConfig(
                    name="qwen",
                    path=str(tmp_path / "q.gguf"),
                    port=18080,
                    gpu_pci_slot="0000:03:00.0",
                )
            ],
            audio_models=[_audio_model(tmp_path, always_resident=always_resident)],
        )
        return Router(cfg)

    def test_pinned_audio_model_survives_an_llm_load(self, tmp_path):
        rt = self._router(tmp_path)
        audio_srv = FakeRunningServer()
        rt._servers["qwen3-asr"] = audio_srv

        asyncio.run(rt._evict_for(rt.cfg.models[0], rt.cfg.gpus[0]))

        assert audio_srv.stopped is False, "pinned audio model must not be evicted"

    def test_swappable_audio_model_is_evicted(self, tmp_path):
        rt = self._router(tmp_path, always_resident=False)
        audio_srv = FakeRunningServer()
        rt._servers["qwen3-asr"] = audio_srv

        asyncio.run(rt._evict_for(rt.cfg.models[0], rt.cfg.gpus[0]))

        assert audio_srv.stopped is True

    def test_loading_a_pinned_model_does_not_evict_the_llm(self, tmp_path):
        """The symptom pinning exists to prevent: a voice command cold-starting the LLM.

        Being exempt from eviction is only half of it — if loading the ASR
        model displaced the LLM, the utterance would pay exactly the cold start
        pinning is for, and the ASR model would just be pinned in its place.
        """
        rt = self._router(tmp_path)
        llm_srv = FakeRunningServer()
        rt._servers["qwen"] = llm_srv

        asyncio.run(rt._evict_for(rt.cfg.audio_models[0], rt.cfg.gpus[0]))

        assert llm_srv.stopped is False

    def test_loading_a_swappable_audio_model_still_evicts(self, tmp_path):
        """--swappable opts back into the single-resident policy, both ways."""
        rt = self._router(tmp_path, always_resident=False)
        llm_srv = FakeRunningServer()
        rt._servers["qwen"] = llm_srv

        asyncio.run(rt._evict_for(rt.cfg.audio_models[0], rt.cfg.gpus[0]))

        assert llm_srv.stopped is True

    def test_a_pinned_model_that_cannot_fit_says_what_to_do(self, tmp_path):
        """It cannot evict its way out, so the error has to name the options."""
        rt = self._router(tmp_path)
        rt._servers["qwen"] = FakeRunningServer()
        rt.cfg.models[0].recipe = {"ctx": 4096}
        rt.cfg.audio_models[0].vram_mb = 24000

        with pytest.raises(RuntimeError, match="pinned") as exc:
            rt._check_vram_fit(rt.cfg.audio_models[0], rt.cfg.gpus[0])
        assert "--swappable" in str(exc.value)

    def test_pinned_model_is_charged_to_the_vram_budget(self, tmp_path):
        """A pinned model survives eviction, so an LLM has to fit alongside it."""
        rt = self._router(tmp_path)
        rt._servers["qwen3-asr"] = FakeRunningServer()
        rt.cfg.audio_models[0].vram_mb = 24000  # nearly the whole card
        gpu = rt.cfg.gpus[0]

        with pytest.raises(RuntimeError, match="but only"):
            rt._check_vram_fit(rt.cfg.models[0], gpu)

    def test_audio_backends_are_registered_in_the_server_map(self, tmp_path):
        rt = self._router(tmp_path)
        assert "qwen3-asr" in rt._servers
        assert Path(rt._servers["qwen3-asr"].plan.argv[0]).stem == "llama-server"

    def test_tts_backends_join_the_same_map(self, tmp_path):
        cfg = Config(
            paths=PathsConfig(
                llama_server=str(_fake_llama_server(tmp_path)),
                state_dir=str(tmp_path / "state"),
                tts_python=str(_fake_python(tmp_path)),
            ),
            gpus=[_gpu()],
            audio_models=[_tts_model(tmp_path)],
        )
        rt = Router(cfg)
        assert "omnivoice" in rt._servers

    def test_unlaunchable_audio_model_is_skipped_not_fatal(self, tmp_path):
        cfg = Config(
            paths=PathsConfig(
                llama_server=str(tmp_path / "nope" / "llama-server"),
                state_dir=str(tmp_path),
            ),
            gpus=[_gpu()],
            audio_models=[_audio_model(tmp_path)],
        )
        rt = Router(cfg)
        assert "qwen3-asr" not in rt._servers
        assert rt.all_audio_models()[0].name == "qwen3-asr"
        # The reason has to survive, or the UI can only say "not launchable".
        assert "not found" in rt.audio_launch_errors["qwen3-asr"]


class TestAudioVramEstimate:
    def test_declared_value_wins(self, tmp_path):
        assert _estimate_audio_vram_mb(_audio_model(tmp_path, vram_mb=1234)) == 1234

    def test_unmeasurable_path_returns_none(self, tmp_path):
        model = _audio_model(tmp_path, path=str(tmp_path / "nope"))
        assert _estimate_audio_vram_mb(model) is None

    def test_llamacpp_estimate_grows_with_context(self, tmp_path):
        """The KV cache is the term that made an ASR model outweigh a 27B."""
        small = _estimate_audio_vram_mb(_audio_model(tmp_path, recipe={"ctx": 4096}))
        large = _estimate_audio_vram_mb(_audio_model(tmp_path, recipe={"ctx": 65536}))
        assert large > small

    def test_tts_model_is_measured_by_its_engine(self, tmp_path):
        assert _estimate_audio_vram_mb(_tts_model(tmp_path)) is not None

    def test_tts_model_with_an_unresolvable_repo_id_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-cache"))
        model = _tts_model(tmp_path, path="k2-fsa/DoesNotExist")
        assert _estimate_audio_vram_mb(model) is None

    def test_a_cached_repo_id_is_measured(self, tmp_path, monkeypatch):
        """`os.sep in repo_id` is always true on POSIX, so this used to be None."""
        cache = _fake_hf_cache(tmp_path, "k2-fsa/OmniVoice", blob_bytes=8 * 1024 * 1024)
        monkeypatch.setenv("HF_HUB_CACHE", str(cache))
        model = _tts_model(tmp_path, path="k2-fsa/OmniVoice")
        assert _estimate_audio_vram_mb(model) is not None

    def test_hf_snapshot_symlinks_are_not_counted_twice(self, tmp_path, monkeypatch):
        """blobs/ holds the bytes and snapshots/ symlinks to them.

        `Path.stat()` follows symlinks, so the naive walk charged the fit guard
        for every weight twice and refused loads that fit comfortably.
        """
        from arc_llama.tts.base import _VRAM_OVERHEAD_MB

        blob_mb = 8
        cache = _fake_hf_cache(
            tmp_path, "k2-fsa/OmniVoice", blob_bytes=blob_mb * 1024 * 1024
        )
        monkeypatch.setenv("HF_HUB_CACHE", str(cache))
        model = _tts_model(tmp_path, path="k2-fsa/OmniVoice")

        estimate = _estimate_audio_vram_mb(model)
        assert estimate == blob_mb + _VRAM_OVERHEAD_MB, (
            "expected the blob counted once, not once per symlink"
        )

    def test_only_the_checked_out_revision_is_measured(self, tmp_path, monkeypatch):
        """A stale revision's blobs are never loaded, so they must not be charged."""
        from arc_llama.tts.base import _VRAM_OVERHEAD_MB

        blob_mb = 8
        cache = _fake_hf_cache(
            tmp_path, "k2-fsa/OmniVoice", blob_bytes=blob_mb * 1024 * 1024,
            stale_revision_bytes=32 * 1024 * 1024,
        )
        monkeypatch.setenv("HF_HUB_CACHE", str(cache))
        model = _tts_model(tmp_path, path="k2-fsa/OmniVoice")
        assert _estimate_audio_vram_mb(model) == blob_mb + _VRAM_OVERHEAD_MB


class TestResolveBinary:
    def test_bare_name_is_found_on_path(self, tmp_path, monkeypatch):
        """A build dropped on PATH must count as installed."""
        binary = _fake_binary(tmp_path, "llama-server")
        monkeypatch.setenv("PATH", str(tmp_path))
        resolved = resolve_binary("llama-server")
        assert resolved is not None
        # Compared by identity, not by string: on Windows the name carries a
        # PATHEXT suffix and `shutil.which` returns it in whatever case
        # PATHEXT spells, so the two paths point at one file without
        # matching character for character.
        assert os.path.samefile(resolved, binary)

    def test_missing_bare_name_is_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path))
        assert resolve_binary("llama-server") is None

    def test_explicit_path_is_taken_literally(self, tmp_path):
        binary = _fake_binary(tmp_path, "llama-server")
        assert resolve_binary(str(binary)) == str(binary)
        assert resolve_binary(str(tmp_path / "nope" / "llama-server")) is None


# ---------------------------------------------------------------------------
# Output sanitisation
# ---------------------------------------------------------------------------


class TestStripAsrMarkers:
    def test_strips_language_announcement_and_marker(self):
        raw = "language English<asr_text>turn off the kitchen lights"
        assert strip_asr_markers(raw) == "turn off the kitchen lights"

    def test_plain_transcript_is_untouched(self):
        assert strip_asr_markers("turn off the lights") == "turn off the lights"

    def test_audio_tokens_are_removed(self):
        assert strip_asr_markers("<|audio_start|>hello<|audio_end|>") == "hello"

    def test_empty_transcript_stays_empty(self):
        assert strip_asr_markers("language English<asr_text>") == ""


# ---------------------------------------------------------------------------
# Scan safety
# ---------------------------------------------------------------------------


class TestIsAudioGguf:
    def test_projector_is_never_an_llm(self, tmp_path):
        p = tmp_path / "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"
        p.write_bytes(b"GGUF")
        assert is_audio_gguf(p) is True

    def test_asr_weights_with_a_projector_sibling_are_skipped(self, tmp_path):
        weights, _ = _weights_and_mmproj(tmp_path)
        assert is_audio_gguf(weights) is True

    def test_registered_audio_model_is_skipped(self, tmp_path):
        weights, mmproj = _weights_and_mmproj(tmp_path)
        cfg = Config(audio_models=[_audio_model(tmp_path)])
        assert is_audio_gguf(weights, cfg) is True
        assert is_audio_gguf(mmproj, cfg) is True

    def test_ordinary_llm_gguf_is_not_audio(self, tmp_path, monkeypatch):
        p = tmp_path / "qwen3-27b-q4_k_m.gguf"
        p.write_bytes(b"GGUF")
        monkeypatch.setattr(
            "arc_llama.models.read_gguf_meta", lambda _p: {"architecture": "qwen3"}
        )
        assert is_audio_gguf(p) is False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


class FakeAudioBackendPlan:
    backend_url = "http://fake-audio"


class FakeAudioBackend:
    plan = FakeAudioBackendPlan()
    is_running = True
    ready = True


class FakeAudioResponse:
    status_code = 200
    headers = {"content-type": "application/json"}
    content = b'{"text": "hello world"}'


class FakeQwen3AsrResponse:
    """What llama-server actually returns for Qwen3-ASR (llama.cpp#26749)."""

    status_code = 200
    headers = {"content-type": "application/json"}
    content = b'{"text": "language English<asr_text>turn off the kitchen lights"}'


class FakeSpeechResponse:
    status_code = 200
    headers = {"content-type": "audio/mpeg"}
    content = b"ID3\x04\x00fake mp3 bytes"


class RecordingAsyncClient:
    """Captures the request the audio proxy builds for the backend."""

    last_call: dict = {}
    response_cls = FakeAudioResponse

    def __init__(self, timeout=None):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url, **kwargs):
        RecordingAsyncClient.last_call = {"url": url, **kwargs}
        cls = RecordingAsyncClient.response_cls
        if url.endswith("/v1/audio/speech"):
            return FakeSpeechResponse()
        return cls()

    async def aclose(self):
        return None


class FakeAudioRouter:
    last_activity = 0.0
    inflight = 0

    def __init__(self, cfg, log_dir=None):
        self.cfg = cfg
        self.model_inflight = {}
        self._servers = {m.name: FakeAudioBackend() for m in cfg.audio_models}
        self.metrics = {
            "loads": 0,
            "stops": 0,
            "load_errors": 0,
            "last_load_at": None,
            "last_error": None,
        }

    def acquire_model(self, name):
        self.model_inflight[name] = self.model_inflight.get(name, 0) + 1

    def release_model(self, name):
        self.model_inflight.pop(name, None)

    def all_models(self):
        return list(self.cfg.models)

    def all_audio_models(self):
        return list(self.cfg.audio_models)

    async def ensure_active(self, query, *, acquire=False):
        model = self.cfg.find_any_model(query)
        if model is None or model.name not in self._servers:
            raise KeyError(query)
        if acquire:
            self.acquire_model(model.name)
        return model, self._servers[model.name]

    async def shutdown(self):
        return None


class FakeUpstreamManagerNoop:
    def __init__(self, upstreams=None):
        pass

    async def models(self):
        return []

    def find_model(self, name):
        return None

    def upstreams_status(self):
        return []


def _app(monkeypatch, tmp_path, audio_models, voices=None):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeAudioRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManagerNoop)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", RecordingAsyncClient)
    cfg = Config(
        paths=PathsConfig(state_dir=str(tmp_path / "state")),
        gpus=[_gpu()],
        audio_models=audio_models,
        voices=list(voices or []),
    )
    return create_app(cfg)


def _audio_app(monkeypatch, tmp_path, **model_kw):
    return _app(monkeypatch, tmp_path, [_audio_model(tmp_path, **model_kw)])


def _speech_app(monkeypatch, tmp_path, voices=None, **model_kw):
    return _app(monkeypatch, tmp_path, [_tts_model(tmp_path, **model_kw)], voices=voices)


class TestTranscriptionsEndpoint:
    def test_multipart_upload_is_forwarded(self, monkeypatch, tmp_path):
        app = _audio_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                data={"model": "qwen3-asr", "language": "en"},
                files={"file": ("speech.wav", b"RIFF....", "audio/wav")},
            )

        assert response.status_code == 200
        assert response.json() == {"text": "hello world"}
        call = RecordingAsyncClient.last_call
        assert call["url"] == "http://fake-audio/v1/audio/transcriptions"
        assert call["data"]["model"] == "qwen3-asr"
        assert call["data"]["language"] == "en"
        assert call["files"]["file"][0] == "speech.wav"
        assert call["files"]["file"][1] == b"RIFF...."

    def test_alias_is_rewritten_to_the_backend_id(self, monkeypatch, tmp_path):
        """The backend only answers to its own id, not to our aliases."""
        app = _audio_app(monkeypatch, tmp_path, aliases=["whisper-1"])
        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                data={"model": "whisper-1"},
                files={"file": ("speech.wav", b"RIFF", "audio/wav")},
            )

        assert response.status_code == 200
        assert RecordingAsyncClient.last_call["data"]["model"] == "qwen3-asr"

    def test_unknown_model_falls_back_to_the_only_asr_model(self, monkeypatch, tmp_path):
        app = _audio_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                data={"model": "some-client-default"},
                files={"file": ("speech.wav", b"RIFF", "audio/wav")},
            )

        assert response.status_code == 200
        assert RecordingAsyncClient.last_call["data"]["model"] == "qwen3-asr"

    def test_json_body_path_is_supported(self, monkeypatch, tmp_path):
        app = _audio_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                json={"model": "qwen3-asr", "audio": "/srv/audio/clip.wav"},
            )

        assert response.status_code == 200
        assert RecordingAsyncClient.last_call["json"]["audio"] == "/srv/audio/clip.wav"

    def test_missing_file_is_rejected(self, monkeypatch, tmp_path):
        app = _audio_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            response = client.post("/v1/audio/transcriptions", data={"model": "qwen3-asr"})
        assert response.status_code == 400

    def test_stream_needs_a_streaming_model(self, monkeypatch, tmp_path):
        app = _audio_app(monkeypatch, tmp_path, mode="offline")
        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                data={"model": "qwen3-asr", "stream": "true"},
                files={"file": ("speech.wav", b"RIFF", "audio/wav")},
            )
        assert response.status_code == 400
        assert "streaming" in response.json()["detail"]

    def test_no_audio_models_returns_501(self, monkeypatch, tmp_path):
        app = _app(monkeypatch, tmp_path, [])
        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                data={"model": "whatever"},
                files={"file": ("speech.wav", b"RIFF", "audio/wav")},
            )
        assert response.status_code == 501

    def test_a_tts_model_does_not_answer_transcriptions(self, monkeypatch, tmp_path):
        """Routing is by task, so a box with only TTS still says 501 for STT."""
        app = _speech_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                data={"model": "omnivoice"},
                files={"file": ("speech.wav", b"RIFF", "audio/wav")},
            )
        assert response.status_code == 501

    def test_inflight_is_released_after_the_request(self, monkeypatch, tmp_path):
        app = _audio_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            client.post(
                "/v1/audio/transcriptions",
                data={"model": "qwen3-asr"},
                files={"file": ("speech.wav", b"RIFF", "audio/wav")},
            )
            rt = app.state.router
        assert rt.inflight == 0
        assert rt.model_inflight == {}

    def test_qwen3_asr_framing_is_stripped(self, monkeypatch, tmp_path):
        """A HA voice pipeline must not have to match 'language English<asr_text>'."""
        monkeypatch.setattr(RecordingAsyncClient, "response_cls", FakeQwen3AsrResponse)
        app = _audio_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                data={"model": "qwen3-asr"},
                files={"file": ("speech.wav", b"RIFF", "audio/wav")},
            )
        assert response.json() == {"text": "turn off the kitchen lights"}

    def test_framing_is_kept_when_stripping_is_disabled(self, monkeypatch, tmp_path):
        monkeypatch.setattr(RecordingAsyncClient, "response_cls", FakeQwen3AsrResponse)
        app = _audio_app(monkeypatch, tmp_path, strip_asr_markers=False)
        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                data={"model": "qwen3-asr"},
                files={"file": ("speech.wav", b"RIFF", "audio/wav")},
            )
        assert "<asr_text>" in response.json()["text"]

    def test_audio_models_appear_in_v1_models(self, monkeypatch, tmp_path):
        app = _audio_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            data = client.get("/v1/models").json()["data"]
        entry = next(m for m in data if m["id"] == "qwen3-asr")
        assert entry["owned_by"] == "arc-llama-audio"
        assert entry["metadata"]["task"] == "asr"
        assert entry["metadata"]["engine"] == "llamacpp"


class TestSpeechEndpoint:
    def test_input_is_forwarded_and_audio_comes_back(self, monkeypatch, tmp_path):
        app = _speech_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/speech",
                json={"model": "omnivoice", "input": "hello there", "voice": "glados"},
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.content == FakeSpeechResponse.content
        call = RecordingAsyncClient.last_call
        assert call["url"] == "http://fake-audio/v1/audio/speech"
        assert call["json"]["input"] == "hello there"
        assert call["json"]["voice"] == "glados"

    def test_model_defaults_are_applied_by_the_engine(self, monkeypatch, tmp_path):
        app = _speech_app(
            monkeypatch, tmp_path, recipe={"default_voice": "glados", "default_language": "English"}
        )
        with TestClient(app) as client:
            client.post("/v1/audio/speech", json={"input": "hello"})
        assert RecordingAsyncClient.last_call["json"]["voice"] == "glados"
        assert RecordingAsyncClient.last_call["json"]["language"] == "English"

    def test_an_omitted_model_resolves_to_the_only_tts_model(self, monkeypatch, tmp_path):
        """Clients hardcode `tts-1`, or send nothing at all."""
        app = _speech_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            response = client.post("/v1/audio/speech", json={"input": "hello", "model": "tts-1"})
        assert response.status_code == 200

    def test_empty_input_is_rejected(self, monkeypatch, tmp_path):
        app = _speech_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            response = client.post("/v1/audio/speech", json={"model": "omnivoice", "input": "  "})
        assert response.status_code == 400

    def test_oversized_body_is_rejected(self, monkeypatch, tmp_path):
        app = _speech_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/speech", json={"model": "omnivoice", "input": "x" * 200_000}
            )
        assert response.status_code == 413

    def test_no_tts_models_returns_501_with_the_fix(self, monkeypatch, tmp_path):
        app = _audio_app(monkeypatch, tmp_path)  # ASR only
        with TestClient(app) as client:
            response = client.post("/v1/audio/speech", json={"input": "hello"})
        assert response.status_code == 501
        assert "omnivoice" in response.json()["detail"]

    def test_unknown_model_among_several_is_a_404(self, monkeypatch, tmp_path):
        app = _app(
            monkeypatch,
            tmp_path,
            [_tts_model(tmp_path), _tts_model(tmp_path, name="other", port=18092)],
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/speech", json={"model": "nope", "input": "hello"}
            )
        assert response.status_code == 404

    def test_inflight_is_released_after_the_request(self, monkeypatch, tmp_path):
        app = _speech_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            client.post("/v1/audio/speech", json={"model": "omnivoice", "input": "hello"})
            rt = app.state.router
        assert rt.inflight == 0
        assert rt.model_inflight == {}

    def test_tts_models_appear_in_v1_models(self, monkeypatch, tmp_path):
        app = _speech_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            data = client.get("/v1/models").json()["data"]
        entry = next(m for m in data if m["id"] == "omnivoice")
        assert entry["metadata"]["task"] == "tts"
        assert entry["metadata"]["engine"] == "omnivoice"


# ---------------------------------------------------------------------------
# The OmniVoice sidecar
# ---------------------------------------------------------------------------


class TestVoiceBook:
    def _book(self, tmp_path, payload):

        path = tmp_path / "voices.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return VoiceBook(str(path)), path

    def test_exact_and_case_insensitive_lookup(self, tmp_path):
        book, _ = self._book(tmp_path, {"voices": {"GLaDOS": {"instruct": "female"}}})
        assert book.lookup("GLaDOS")[0] == "GLaDOS"
        assert book.lookup("glados")[0] == "GLaDOS"

    def test_alias_lookup(self, tmp_path):
        book, _ = self._book(
            tmp_path, {"voices": {"glados": {"instruct": "female", "aliases": ["alloy"]}}}
        )
        assert book.lookup("alloy")[0] == "glados"

    def test_unknown_voice_falls_back_to_the_default(self, tmp_path):
        """A substituted voice beats a failed request for a speech client."""
        book, _ = self._book(
            tmp_path,
            {"default_voice": "glados", "voices": {"glados": {"instruct": "female"}}},
        )
        assert book.lookup("nova")[0] == "glados"

    def test_no_match_and_no_default_is_none(self, tmp_path):
        book, _ = self._book(tmp_path, {"voices": {"glados": {"instruct": "female"}}})
        assert book.lookup("nova") is None

    def test_an_edited_file_is_picked_up_without_a_restart(self, tmp_path):
        book, path = self._book(tmp_path, {"voices": {"glados": {"instruct": "female"}}})
        assert book.lookup("narrator") is None

        import os

        payload = {"voices": {"glados": {"instruct": "female"}, "narrator": {"instruct": "male"}}}
        path.write_text(json.dumps(payload), encoding="utf-8")
        os.utime(path, (0, 0))  # force a different mtime

        assert book.lookup("narrator")[0] == "narrator"

    def test_a_corrupt_file_keeps_the_last_good_table(self, tmp_path):
        """A half-written voices file must not take TTS down."""
        book, path = self._book(tmp_path, {"voices": {"glados": {"instruct": "female"}}})
        assert book.lookup("glados") is not None

        import os

        path.write_text("{not json", encoding="utf-8")
        os.utime(path, (0, 0))
        assert book.lookup("glados") is not None


class TestAutoVoiceSynthesis:
    """A fine-tuned model must reach generate() with no prompt attached."""

    def _generate_kwargs(self, voices_json, voice_field):
        import json as _json
        import tempfile
        from pathlib import Path as _Path

        d = _Path(tempfile.mkdtemp())
        (d / "v.json").write_text(_json.dumps(voices_json), encoding="utf-8")
        args = build_parser().parse_args(
            ["--model", "m", "--port", "1", "--voices", str(d / "v.json")]
        )
        engine = Engine(args, VoiceBook(str(d / "v.json")))
        captured: dict = {}

        class FakeModel:
            def generate(self, **kw):
                captured.update(kw)
                return [[0.0]]

        engine.model = FakeModel()
        original = _SIDECAR.encode_audio
        _SIDECAR.encode_audio = lambda s, r, f: (b"", "audio/wav")
        try:
            engine.synthesize({"input": "hi", "voice": voice_field})
        finally:
            _SIDECAR.encode_audio = original
        return captured

    def test_no_registered_voices_means_the_models_own_voice(self):
        """Clients must send a `voice`; with none registered it is ignored."""
        kw = self._generate_kwargs({"voices": {}}, "alloy")
        assert kw["text"] == "hi"
        assert "voice_clone_prompt" not in kw
        assert "instruct" not in kw

    def test_a_registered_auto_voice_adds_nothing(self):
        kw = self._generate_kwargs({"voices": {"glados": {}}}, "glados")
        assert "voice_clone_prompt" not in kw
        assert "instruct" not in kw

    def test_a_design_voice_would_override_the_finetune(self):
        """The failure mode to avoid: a prompt layered on baked-in weights."""
        kw = self._generate_kwargs(
            {"default_voice": "narrator", "voices": {"narrator": {"instruct": "male"}}},
            "alloy",
        )
        assert kw["instruct"] == "male"


class TestSidecarImportIsolation:
    """A sidecar's own directory is on `sys.path`, so its neighbours matter.

    Running a script puts its directory at the front of `sys.path` — ahead of
    both site-packages and the stdlib. These scripts therefore live alone in
    `sidecars/`; when `omnivoice_server.py` sat beside the engine module,
    `from omnivoice import OmniVoice` resolved to `arc_llama/tts/omnivoice.py`
    and failed with "cannot import name 'OmniVoice'" on machines where
    OmniVoice was installed and importable.
    """

    def _sidecar_dir(self):
        from pathlib import Path

        from arc_llama.tts import omnivoice

        return Path(omnivoice.SERVER_SCRIPT).parent

    def test_no_neighbour_shadows_anything_a_sidecar_imports(self):
        import ast

        directory = self._sidecar_dir()
        scripts = sorted(directory.glob("*.py"))
        assert scripts, f"no sidecar scripts found in {directory}"

        # Every top-level module name any sidecar imports, however it imports it.
        imported: set[str] = set()
        for script in scripts:
            tree = ast.parse(script.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imported.add(node.module.split(".")[0])

        neighbours = {p.stem for p in directory.glob("*.py")}
        clashes = sorted(neighbours & imported)
        assert not clashes, (
            f"{directory} contains {clashes}, which would shadow the same-named "
            "module for every sidecar run from that directory"
        )

    def test_the_engine_module_is_not_a_neighbour(self):
        """The specific collision that broke `from omnivoice import OmniVoice`."""
        from pathlib import Path

        from arc_llama.tts import omnivoice

        engine_module = Path(omnivoice.__file__)
        assert engine_module.stem == "omnivoice"
        assert engine_module.parent != self._sidecar_dir()

    def test_running_as_a_script_resolves_the_installed_package(self, tmp_path):
        """End to end: launch it the way arc-llama does and see what wins."""
        import os
        import shutil
        import subprocess
        import sys

        from arc_llama.tts import omnivoice

        script_dir = tmp_path / "sidecars"
        script_dir.mkdir()
        shutil.copy(omnivoice.SERVER_SCRIPT, script_dir / "omnivoice_server.py")

        # The real package, as a virtualenv would provide it.
        site = tmp_path / "site"
        site.mkdir()
        (site / "omnivoice.py").write_text(
            "WHICH = 'installed'\nOmniVoice = object\n", encoding="utf-8"
        )

        driver = script_dir / "driver.py"
        driver.write_text(
            "import runpy, sys\n"
            "runpy.run_path(str(sys.argv[1]), run_name='not_main')\n"
            "import omnivoice\n"
            "print(omnivoice.WHICH)\n",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [sys.executable, str(driver), str(script_dir / "omnivoice_server.py")],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(site)}, timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "installed"


class TestSidecarEncoding:
    def _samples(self):
        np = pytest.importorskip("numpy")
        return np.linspace(-1.0, 1.0, 480, dtype="float32")

    def test_wav_is_a_real_riff_file(self):
        import io
        import wave


        data, media_type = encode_audio(self._samples(), 24000, "wav")
        assert media_type == "audio/wav"
        with wave.open(io.BytesIO(data)) as w:
            assert w.getframerate() == 24000
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert w.getnframes() == 480

    def test_pcm_is_raw_s16le_at_the_model_rate(self):

        data, media_type = encode_audio(self._samples(), 24000, "pcm")
        assert media_type == "application/octet-stream"
        assert len(data) == 480 * 2  # no container, 16-bit mono

    def test_out_of_range_samples_are_clipped_not_wrapped(self):
        """A sample above 1.0 wrapping through int16 is loud noise, not audio."""
        np = pytest.importorskip("numpy")


        loud = np.array([2.0, -2.0], dtype="float32")
        data, _ = encode_audio(loud, 24000, "pcm")
        assert np.frombuffer(data, dtype="<i2").tolist() == [32767, -32767]


class TestSidecarRequestValidation:
    def _engine(self, tmp_path, **overrides):

        argv = ["--model", "k2-fsa/OmniVoice", "--port", "18091"]
        for key, value in overrides.items():
            argv += [f"--{key.replace('_', '-')}", str(value)]
        args = build_parser().parse_args(argv)
        engine = Engine(args, VoiceBook(None))
        engine.model = object()  # never reached by the validation paths below
        return engine

    def test_empty_input_is_a_client_error(self, tmp_path):

        engine = self._engine(tmp_path)
        with pytest.raises(BadRequestError, match="input"):
            engine.synthesize({"input": "   "})

    def test_unknown_response_format_lists_the_valid_ones(self, tmp_path):

        engine = self._engine(tmp_path)
        with pytest.raises(BadRequestError, match="response_format"):
            engine.synthesize({"input": "hi", "response_format": "ogg-vorbis"})

    def test_speed_outside_openais_range_is_refused(self, tmp_path):

        engine = self._engine(tmp_path)
        with pytest.raises(BadRequestError, match="speed"):
            engine.synthesize({"input": "hi", "speed": 12.0})

    def test_generation_arguments_are_assembled(self, tmp_path, monkeypatch):
        """Voice cloning wins over a design instruction; both never apply at once."""

        path = tmp_path / "voices.json"
        path.write_text(
            json.dumps({"voices": {"glados": {"instruct": "female", "language": "English"}}}),
            encoding="utf-8",
        )
        args = build_parser().parse_args(
            ["--model", "m", "--port", "1", "--voices", str(path)]
        )
        engine = Engine(args, VoiceBook(str(path)))

        captured = {}

        class FakeModel:
            def generate(self, **kwargs):
                captured.update(kwargs)
                return [[0.0, 0.1]]

        engine.model = FakeModel()
        monkeypatch.setattr(
            _SIDECAR, "encode_audio", lambda samples, rate, fmt: (b"audio", "audio/wav")
        )

        engine.synthesize({"input": "hello", "voice": "glados", "speed": 1.2})

        assert captured["text"] == "hello"
        assert captured["instruct"] == "female"
        assert captured["language"] == "English"
        assert captured["speed"] == 1.2
        assert captured["num_step"] == 32
