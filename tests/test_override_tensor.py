"""Tests for --override-tensor (AUTOTUNE_ROUND7).

All tests use fakes. No llama-server process, no GPU probing, no systemd.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from helpers import FakeTensor as _FakeTensor
from helpers import FakeTensorReader as _FakeReader

from arc_llama.benchmark import BenchmarkResult
from arc_llama.config import Config, GPUConfig, ModelConfig, PathsConfig
from arc_llama.gguf_meta import (
    override_tensor_saved_bytes,
    propose_override_tensor_patterns,
    validate_override_patterns,
    weight_tensor_table,
)
from arc_llama.recipes import LaunchRecipe, recipe_to_dict
from arc_llama.tune import tune_model

# ---------------------------------------------------------------------------
# Shared fixtures/helpers
# ---------------------------------------------------------------------------


def _patch_reader(monkeypatch: pytest.MonkeyPatch, tensors, arch: str = "gemma4"):
    import gguf

    monkeypatch.setattr(
        "arc_llama.gguf_meta.gguf.GGUFReader",
        lambda _path: _FakeReader(tensors, arch),
    )
    monkeypatch.setattr(
        gguf, "GGUFReader",
        lambda _path: _FakeReader(tensors, arch),
    )


def _gemma_fused_tensors() -> list[_FakeTensor]:
    """Real-like fused Gemma layout including router and shared expert."""
    return [
        _FakeTensor("token_embd.weight", 1000),
        _FakeTensor("output.weight", 400),
        _FakeTensor("blk.0.attn_norm.weight", 10),
        _FakeTensor("blk.0.attn_q.weight", 100),
        _FakeTensor("blk.0.ffn_gate_up_exps.weight", 800),
        _FakeTensor("blk.0.ffn_down_exps.weight", 400),
        _FakeTensor("blk.0.ffn_down_exps.scale", 20),
        _FakeTensor("blk.0.ffn_gate_inp.weight", 30),
        _FakeTensor("blk.0.ffn_up_shexp.weight", 60),
        _FakeTensor("blk.1.ffn_gate_up_exps.weight", 800),
        _FakeTensor("blk.1.ffn_down_exps.weight", 400),
        _FakeTensor("blk.1.ffn_down_exps.scale", 20),
        _FakeTensor("blk.1.ffn_gate_inp.weight", 30),
    ]


def _unfused_tensors() -> list[_FakeTensor]:
    """Unfused routed-expert layout."""
    return [
        _FakeTensor("token_embd.weight", 1000),
        _FakeTensor("blk.0.attn_q.weight", 50),
        _FakeTensor("blk.0.ffn_gate_exps.weight", 300),
        _FakeTensor("blk.0.ffn_up_exps.weight", 300),
        _FakeTensor("blk.0.ffn_down_exps.weight", 300),
        _FakeTensor("blk.0.ffn_gate_inp.weight", 50),
        _FakeTensor("blk.1.ffn_gate_exps.weight", 300),
        _FakeTensor("blk.1.ffn_up_exps.weight", 300),
        _FakeTensor("blk.1.ffn_down_exps.weight", 300),
    ]


_MB = 1_048_576


def _moe_scale_tensors() -> list[_FakeTensor]:
    """Fused-Gemma layout at realistic byte scale, sized so offload is needed.

    The byte-scale fixtures above are fine for pattern generation, but the
    tuner's offload stage only engages when the model does not fit, and
    ``vram_mb`` is an integer count of MiB. A 4 KB model fits any budget, so a
    tuner test built on those tensors silently exercises nothing.

    8 expert layers at 920 MiB of routed experts each (600 fused gate+up, 300
    down, 20 scale) plus 1028 MiB of non-expert weight. Against the 8000 MiB
    budget used below that needs a 2-layer offload, and the cheapest generated
    pattern (the down class, 2560 MiB across all layers) frees more than those
    2 layers do, so both -ot candidates are viable and the refinement actually
    runs.
    """
    tensors = [
        _FakeTensor("token_embd.weight", 500 * _MB),
        _FakeTensor("output.weight", 400 * _MB),
    ]
    for i in range(8):
        tensors += [
            _FakeTensor(f"blk.{i}.attn_q.weight", 10 * _MB),
            _FakeTensor(f"blk.{i}.ffn_gate_up_exps.weight", 600 * _MB),
            _FakeTensor(f"blk.{i}.ffn_down_exps.weight", 300 * _MB),
            _FakeTensor(f"blk.{i}.ffn_down_exps.scale", 20 * _MB),
            _FakeTensor(f"blk.{i}.ffn_gate_inp.weight", 1 * _MB),
            _FakeTensor(f"blk.{i}.ffn_up_shexp.weight", 5 * _MB),
        ]
    return tensors


# The cheapest candidate the generator proposes for the fixture above: the
# down-projection class is the smallest by total bytes, so it is tried first.
# The generator is spelling-agnostic (_exps / _chexps), hence the optional "ch".
_CHEAPEST_PATTERN = r"blk\.\d+\.ffn_down_(?:ch)?exps\."
_CATCH_ALL_PATTERN = r"blk\.\d+\.ffn_.*_(?:ch)?exps\."


class _FakeCaps:
    supports_flash_attn = True
    flash_attn_takes_value = True


# ---------------------------------------------------------------------------
# Pattern generation from real tensor names
# ---------------------------------------------------------------------------


def test_fused_pattern_generation_orders_by_bytes(tmp_path, monkeypatch):
    f = tmp_path / "gemma.gguf"
    f.write_bytes(b"x")
    _patch_reader(monkeypatch, _gemma_fused_tensors())

    table = weight_tensor_table(f)
    assert table is not None
    # Down is 2 layers * 420 bytes, fused gate_up is 2 layers * 800 bytes.
    assert table["blk.0.ffn_down_exps.weight"] == 400
    assert table["blk.0.ffn_gate_up_exps.weight"] == 800

    pats = propose_override_tensor_patterns(table)
    assert len(pats) == 3
    assert "down" in pats[0]  # smaller bytes => cheaper => first
    assert "gate_up" in pats[1]
    assert re.search(pats[0], "blk.0.ffn_down_exps.weight")
    assert not re.search(pats[0], "blk.0.ffn_gate_up_exps.weight")
    assert re.search(pats[1], "blk.0.ffn_gate_up_exps.weight")


def test_unfused_pattern_generation(tmp_path, monkeypatch):
    f = tmp_path / "qwen.gguf"
    f.write_bytes(b"x")
    _patch_reader(monkeypatch, _unfused_tensors())

    table = weight_tensor_table(f)
    assert table is not None
    pats = propose_override_tensor_patterns(table)
    # gate/up/down all same bytes, plus catch-all => 4 projection regexes.
    assert len(pats) == 4
    assert re.search(pats[0], "blk.0.ffn_gate_exps.weight")
    assert re.search(pats[3], "blk.0.ffn_gate_exps.weight")
    assert not re.search(pats[0], "token_embd.weight")


def test_validate_patterns_rejects_zero_match(tmp_path, monkeypatch):
    f = tmp_path / "gemma.gguf"
    f.write_bytes(b"x")
    _patch_reader(monkeypatch, _gemma_fused_tensors())

    table = weight_tensor_table(f)
    assert table is not None
    ok, err = validate_override_patterns(table, [r"no\.such\.tensor"])
    assert not ok
    assert "matches zero tensors" in err


def test_validate_patterns_rejects_invalid_regex(tmp_path, monkeypatch):
    f = tmp_path / "gemma.gguf"
    f.write_bytes(b"x")
    _patch_reader(monkeypatch, _gemma_fused_tensors())

    table = weight_tensor_table(f)
    ok, err = validate_override_patterns(table, ["("])
    assert not ok
    assert "invalid" in err


def test_saved_bytes_agrees_with_table(tmp_path, monkeypatch):
    f = tmp_path / "gemma.gguf"
    f.write_bytes(b"x")
    _patch_reader(monkeypatch, _gemma_fused_tensors())

    table = weight_tensor_table(f)
    # fused gate_up for blk.0 and blk.1 only
    saved = override_tensor_saved_bytes(table, [r"blk\.\d+\.ffn_gate_up_exps\."])
    assert saved == 1600

    # catch-all matches gate_up, down, and scales but not router/inp/shexp
    saved_all = override_tensor_saved_bytes(table, [r"blk\.\d+\.ffn_\w*exps\."])
    expected = 800 + 400 + 20 + 800 + 400 + 20
    assert saved_all == expected


def _chexps_tensors() -> list[_FakeTensor]:
    """Upstream checkpoint where routed experts use the ``chexps`` spelling."""
    return [
        _FakeTensor("token_embd.weight", 1000),
        _FakeTensor("blk.0.ffn_gate_chexps.weight", 700),
        _FakeTensor("blk.0.ffn_down_chexps.weight", 400),
        _FakeTensor("blk.0.ffn_up_shexp.weight", 60),
        _FakeTensor("blk.0.ffn_gate_inp.weight", 30),
        _FakeTensor("blk.1.ffn_gate_chexps.weight", 700),
        _FakeTensor("blk.1.ffn_down_chexps.weight", 400),
    ]


def test_chexps_pattern_generation_and_validation(tmp_path, monkeypatch):
    """Round 9: ``chexps`` tensors are grouped by projection class and match real names."""
    f = tmp_path / "chexps.gguf"
    f.write_bytes(b"x")
    _patch_reader(monkeypatch, _chexps_tensors())

    table = weight_tensor_table(f)
    pats = propose_override_tensor_patterns(table)
    assert len(pats) == 3  # down, gate, catch-all
    assert "down" in pats[0]
    assert re.search(pats[0], "blk.0.ffn_down_chexps.weight")
    assert not re.search(pats[0], "blk.0.ffn_gate_chexps.weight")
    assert not re.search(pats[0], "blk.0.ffn_up_shexp.weight")

    # validate every generated pattern against the exact tensor set
    ok, err = validate_override_patterns(table, pats)
    assert ok, err
    assert override_tensor_saved_bytes(table, [pats[-1]]) == 700 + 400 + 700 + 400


# ---------------------------------------------------------------------------
# Recipe/argv rendering
# ---------------------------------------------------------------------------


def test_override_tensor_rendered_as_repeated_flags():
    r = LaunchRecipe(override_tensor=[r"blk\.\d+\.ffn_down_exps\.", r"blk\.\d+\.ffn_gate_exps\."])
    argv = r.to_argv()
    assert "--override-tensor" in argv
    idx = argv.index("--override-tensor")
    assert argv[idx + 1] == r"blk\.\d+\.ffn_down_exps\.=CPU"
    assert argv[idx + 3] == r"blk\.\d+\.ffn_gate_exps\.=CPU"


def test_override_tensor_suppresses_n_cpu_moe():
    """If both are set, -ot wins and --n-cpu-moe is not emitted."""
    r = LaunchRecipe(n_cpu_moe=4, override_tensor=[r"exps"])
    argv = r.to_argv()
    assert "--n-cpu-moe" not in argv
    assert "--override-tensor" in argv


def test_recipe_to_dict_round_trip():
    r = LaunchRecipe(n_cpu_moe=4)
    assert "override_tensor" not in recipe_to_dict(r)
    r2 = LaunchRecipe(override_tensor=[r"exps"])
    d2 = recipe_to_dict(r2)
    assert d2["override_tensor"] == [r"exps"]
    assert "n_cpu_moe" not in d2


# ---------------------------------------------------------------------------
# Tuner integration with faked backend
# ---------------------------------------------------------------------------


def _moe_cfg(tmp_path: Path, *, vram_mb: int) -> Config:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\x00" * 64)
    return Config(
        paths=PathsConfig(llama_server=str(tmp_path / "no-such-server")),
        gpus=[GPUConfig(
            pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=vram_mb
        )],
        models=[ModelConfig(
            name="m", path=str(gguf), port=18080, gpu_pci_slot="0000:03:00.0",
            recipe={"ctx": 8192, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
        )],
    )


def _patch_tune_threads(monkeypatch) -> None:
    async def inline_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("arc_llama.tune.asyncio.to_thread", inline_to_thread)


class _MoeMeasurements:
    def __init__(self, cfg: Config, table: dict[str, int] | None, perf: dict):
        self.cfg = cfg
        self.table = table
        self.perf = perf
        self.state = dict(cfg.models[0].recipe)
        self.edits_seen: list[dict[str, Any]] = []
        self.calls: list[str] = []

    async def apply(self, _client, _name, edits):
        self.edits_seen.append(dict(edits))
        for k, v in edits.items():
            if v is None:
                self.state.pop(k, None)
            else:
                self.state[k] = v

    async def bench(self, _server_url, _model_name, **kw):
        n = self.state.get("n_cpu_moe") or 0
        ot = self.state.get("override_tensor")
        self.calls.append(f"n={n},ot={ot}")
        if ot:
            key = str(ot)
        else:
            key = f"n={n}"
        pp, gen = self.perf.get(key, (500.0, 40.0))
        return BenchmarkResult(
            model="m", ctx=8192, cache_type_k="q8_0", cache_type_v="q8_0",
            prompt_tokens=1024, gen_tokens=128,
            prompt_eval_tok_s=pp, generation_tok_s=gen,
        )


async def test_tuner_refines_n_cpu_moe_with_override_tensor(tmp_path, monkeypatch):
    """When -ot scores higher than n_cpu_moe, the recipe uses -ot."""
    import arc_llama.server_caps as caps_mod

    cfg = _moe_cfg(tmp_path, vram_mb=8000)
    _patch_tune_threads(monkeypatch)
    _patch_reader(monkeypatch, _moe_scale_tensors())
    table = weight_tensor_table(cfg.models[0].path)

    # The 2-layer n_cpu_moe winner is slower than the cheap -ot pattern.
    perf = {
        "n=2": (400.0, 30.0),
        str([_CHEAPEST_PATTERN]): (600.0, 35.0),
        str([_CATCH_ALL_PATTERN]): (300.0, 25.0),
    }
    fake = _MoeMeasurements(cfg, table, perf)
    monkeypatch.setattr("arc_llama.tune._apply_edits", fake.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", fake.bench)
    monkeypatch.setattr(caps_mod, "probe_server_caps", lambda _p: _FakeCaps())

    report = await tune_model("http://127.0.0.1", "m", cfg=cfg)

    assert report.error is None
    assert "override_tensor" in fake.state
    assert fake.state.get("n_cpu_moe") is None


async def test_tuner_keeps_n_cpu_moe_when_ot_does_not_beat(tmp_path, monkeypatch):
    import arc_llama.server_caps as caps_mod

    cfg = _moe_cfg(tmp_path, vram_mb=8000)
    _patch_tune_threads(monkeypatch)
    _patch_reader(monkeypatch, _moe_scale_tensors())

    # Every -ot candidate is slower than the n_cpu_moe winner here.
    perf = {
        "n=2": (900.0, 60.0),
        str([_CHEAPEST_PATTERN]): (400.0, 30.0),
        str([_CATCH_ALL_PATTERN]): (300.0, 25.0),
    }
    fake = _MoeMeasurements(cfg, None, perf)
    monkeypatch.setattr("arc_llama.tune._apply_edits", fake.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", fake.bench)
    monkeypatch.setattr(caps_mod, "probe_server_caps", lambda _p: _FakeCaps())

    report = await tune_model("http://127.0.0.1", "m", cfg=cfg)

    assert report.error is None
    assert fake.state.get("n_cpu_moe") == 2
    assert "override_tensor" not in fake.state


async def test_override_tensor_skipped_when_no_offload_needed(tmp_path, monkeypatch):
    """A small MoE that fits with zero offload never runs the -ot refinement."""
    import arc_llama.server_caps as caps_mod

    cfg = _moe_cfg(tmp_path, vram_mb=64 * 1024)
    _patch_tune_threads(monkeypatch)
    _patch_reader(monkeypatch, _gemma_fused_tensors())

    fake = _MoeMeasurements(cfg, None, {})
    monkeypatch.setattr("arc_llama.tune._apply_edits", fake.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", fake.bench)
    monkeypatch.setattr(caps_mod, "probe_server_caps", lambda _p: _FakeCaps())

    report = await tune_model("http://127.0.0.1", "m", cfg=cfg)

    assert report.error is None
    # Clearing the axis (None) is expected: every apply sends the full
    # canonical state. What must never happen is a pattern being set.
    assert not any(e.get("override_tensor") for e in fake.edits_seen)
