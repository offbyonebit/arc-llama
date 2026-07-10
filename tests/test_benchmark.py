"""Tests for arc_llama.benchmark — measurement, formatting, sweep."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from arc_llama.benchmark import (
    BenchmarkResult,
    _build_prompt,
    _find_drm_card,
    _fmt_speed,
    _fmt_time,
    _fmt_vram,
    _read_pid_vram_mb,
    _read_vram_total,
    _read_vram_used,
    benchmark_model,
    benchmark_sweep,
    print_result,
)


class TestBuildPrompt:
    def test_non_empty(self):
        p = _build_prompt(100)
        assert len(p) > 0
        assert isinstance(p, str)

    def test_approximate_length(self):
        p = _build_prompt(512)
        # ~4 chars per token => ~2048 chars
        assert len(p) >= 1800
        assert len(p) <= 2200

    def test_repeats_word(self):
        p = _build_prompt(100)
        assert "fox" in p


class TestVramHelpers:
    def test_read_vram_used_missing(self, tmp_path: Path):
        assert _read_vram_used(tmp_path) is None

    def test_read_vram_used_valid(self, tmp_path: Path):
        d = tmp_path / "device"
        d.mkdir()
        (d / "mem_info_vram_used").write_text("268435456\n")
        assert _read_vram_used(tmp_path) == 256  # 256 MiB

    def test_read_vram_total_valid(self, tmp_path: Path):
        d = tmp_path / "device"
        d.mkdir()
        (d / "mem_info_vram_total").write_text("25769803776\n")
        assert _read_vram_total(tmp_path) == 24576  # 24 GiB

    def test_read_vram_total_xe_layout(self, tmp_path: Path):
        # Intel xe driver: total is per-tile, no amdgpu-style attrs.
        tile = tmp_path / "device" / "tile0"
        tile.mkdir(parents=True)
        (tile / "physical_vram_size_bytes").write_text("25769803776\n")
        assert _read_vram_total(tmp_path) == 24576

    def test_read_vram_total_i915_layout(self, tmp_path: Path):
        d = tmp_path / "device"
        d.mkdir()
        (d / "lmem_total_bytes").write_text("17179869184\n")
        assert _read_vram_total(tmp_path) == 16384

    def test_read_vram_used_absent_on_intel(self, tmp_path: Path):
        # xe/i915 expose no global used counter — must return None, not 0.
        tile = tmp_path / "device" / "tile0"
        tile.mkdir(parents=True)
        (tile / "physical_vram_size_bytes").write_text("25769803776\n")
        assert _read_vram_used(tmp_path) is None

    def test_find_drm_card_no_sys(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("arc_llama.benchmark.Path", lambda p: tmp_path / p)
        assert _find_drm_card("0000:03:00.0") is None

    def test_find_drm_card_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        drm = tmp_path / "sys" / "class" / "drm"
        card = drm / "card1"
        card.mkdir(parents=True)
        device = card / "device"
        device.symlink_to(tmp_path / "sys" / "bus" / "pci" / "devices" / "0000:03:00.0")
        real_device = tmp_path / "sys" / "bus" / "pci" / "devices" / "0000:03:00.0"
        real_device.mkdir(parents=True)

        monkeypatch.setattr("arc_llama.benchmark.Path", lambda p: tmp_path / "sys" / "class" / "drm")
        found = _find_drm_card("0000:03:00.0")
        assert found is not None
        assert found.name == "card1"


class TestPidVram:
    def _fdinfo(self, tmp_path: Path, pid: int, files: dict[str, str]) -> Path:
        d = tmp_path / str(pid) / "fdinfo"
        d.mkdir(parents=True)
        for name, content in files.items():
            (d / name).write_text(content)
        return tmp_path

    def test_xe_fdinfo(self, tmp_path: Path):
        proc = self._fdinfo(tmp_path, 42, {
            "5": (
                "drm-driver:\txe\n"
                "drm-client-id:\t7\n"
                "drm-total-vram0:\t8192 MiB\n"
                "drm-resident-vram0:\t6144 MiB\n"
            ),
        })
        assert _read_pid_vram_mb(42, proc_root=proc) == 6144

    def test_i915_fdinfo_total_fallback(self, tmp_path: Path):
        proc = self._fdinfo(tmp_path, 42, {
            "5": (
                "drm-driver:\ti915\n"
                "drm-client-id:\t3\n"
                "drm-total-local0:\t4194304 KiB\n"
            ),
        })
        assert _read_pid_vram_mb(42, proc_root=proc) == 4096

    def test_duplicate_clients_counted_once(self, tmp_path: Path):
        fd = (
            "drm-driver:\txe\n"
            "drm-client-id:\t7\n"
            "drm-resident-vram0:\t1024 MiB\n"
        )
        proc = self._fdinfo(tmp_path, 42, {"5": fd, "6": fd, "7": fd})
        assert _read_pid_vram_mb(42, proc_root=proc) == 1024

    def test_distinct_clients_summed(self, tmp_path: Path):
        proc = self._fdinfo(tmp_path, 42, {
            "5": "drm-client-id:\t1\ndrm-resident-vram0:\t1024 MiB\n",
            "6": "drm-client-id:\t2\ndrm-resident-vram0:\t512 MiB\n",
        })
        assert _read_pid_vram_mb(42, proc_root=proc) == 1536

    def test_no_drm_fds(self, tmp_path: Path):
        proc = self._fdinfo(tmp_path, 42, {"0": "pos:\t0\nflags:\t0100002\n"})
        assert _read_pid_vram_mb(42, proc_root=proc) is None

    def test_missing_pid(self, tmp_path: Path):
        assert _read_pid_vram_mb(99999, proc_root=tmp_path) is None


class TestBenchmarkResult:
    def test_to_dict(self):
        r = BenchmarkResult(
            model="qwen3-7b", ctx=8192, cache_type_k="q8_0", cache_type_v="q8_0",
            prompt_tokens=512, gen_tokens=128,
            prompt_eval_tok_s=847.3, generation_tok_s=34.2,
        )
        d = r.to_dict()
        assert d["model"] == "qwen3-7b"
        assert d["prompt_eval_tok_s"] == 847.3

    def test_vram_pct(self):
        r = BenchmarkResult(
            model="m", ctx=4096, cache_type_k="f16", cache_type_v="f16",
            prompt_tokens=1, gen_tokens=1,
            vram_used_mb=12288, vram_total_mb=24576,
        )
        assert r.vram_pct == 50.0

    def test_vram_pct_none(self):
        r = BenchmarkResult(
            model="m", ctx=4096, cache_type_k="f16", cache_type_v="f16",
            prompt_tokens=1, gen_tokens=1,
        )
        assert r.vram_pct is None


class TestFormatting:
    def test_fmt_speed(self):
        assert _fmt_speed(847.3) == " 847.3 tok/s"
        assert _fmt_speed(None) == "—"

    def test_fmt_time_ms(self):
        assert _fmt_time(604.0) == "  604 ms"

    def test_fmt_time_s(self):
        assert _fmt_time(3760.0) == " 3.76 s"

    def test_fmt_vram(self):
        assert _fmt_vram(12288, 24576) == "12.0 GB / 24.0 GB  (50.0%)"
        assert _fmt_vram(None, 24576) == "—"


class FakeHttpxResponse:
    def __init__(self, status_code: int = 200, json_data: Any = None, text: str = ""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        return self._json or {}


class FakeStreamContext:
    """Async context manager that yields a fake streamed response."""
    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def aiter_text(self):
        for c in self._chunks:
            yield c


class FakeHttpxClient:
    """Minimal async httpx client mock for benchmark tests."""
    def __init__(self, base_url: str = "", timeout: float = 30.0):
        self.base_url = base_url
        self.timeout = timeout
        self.calls: list[tuple[str, str, Any]] = []
        self._load_ok = True
        self._edit_ok = True
        self._stream_chunks = ["data: {}\n\n"]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, path: str, **kwargs):
        self.calls.append(("POST", path, kwargs))
        if path.startswith("/admin/load"):
            status = 200 if self._load_ok else 404
            return FakeHttpxResponse(status_code=status, json_data={"loaded": True})
        if path.startswith("/admin/models/") and "/edit" in path:
            status = 200 if self._edit_ok else 400
            return FakeHttpxResponse(status_code=status, json_data={"changed": ["ctx"]})
        return FakeHttpxResponse(200)

    def stream(self, method: str, path: str, **kwargs):
        self.calls.append((method, path, kwargs))
        return FakeStreamContext(self._stream_chunks)

    async def aclose(self):
        pass


@pytest.fixture
def sample_cfg_with_model():
    from arc_llama.config import Config, GPUConfig, ModelConfig
    return Config(
        server=type("S", (), {"host": "127.0.0.1", "port": 11437, "single_resident": True})(),
        paths=type("P", (), {"llama_server": "/bin/llama-server"})(),
        gpus=[
            GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=24 * 1024),
        ],
        models=[
            ModelConfig(
                name="qwen3-7b",
                path="/models/qwen3-7b.gguf",
                port=18080,
                gpu_pci_slot="0000:03:00.0",
                recipe={"ctx": 32768, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
            ),
        ],
    )


class TestBenchmarkModel:
    @pytest.mark.asyncio
    async def test_success(self, sample_cfg_with_model, monkeypatch: pytest.MonkeyPatch):
        fake_client = FakeHttpxClient()
        monkeypatch.setattr("arc_llama.benchmark.httpx.AsyncClient", lambda **kw: fake_client)
        # Mock VRAM reading to avoid needing real sysfs
        monkeypatch.setattr("arc_llama.benchmark._find_drm_card", lambda slot: None)
        monkeypatch.setattr("arc_llama.benchmark._read_vram_used", lambda p: None)
        monkeypatch.setattr("arc_llama.benchmark._read_vram_total", lambda p: None)

        result = await benchmark_model(
            "http://127.0.0.1:11437", "qwen3-7b",
            prompt_tokens=32, gen_tokens=8, load=True, cfg=sample_cfg_with_model,
        )
        assert result.model == "qwen3-7b"
        assert result.ctx == 32768
        assert result.error is None
        assert result.prompt_eval_tok_s is not None
        assert result.generation_tok_s is not None
        # Should have called load, then streamed warmup, prompt-eval, generation
        assert any("/admin/load" in c[1] for c in fake_client.calls)
        assert sum(1 for c in fake_client.calls if c[0] == "POST" and "/v1/chat/completions" in c[1]) >= 3

    @pytest.mark.asyncio
    async def test_model_not_found(self, sample_cfg_with_model, monkeypatch: pytest.MonkeyPatch):
        result = await benchmark_model(
            "http://127.0.0.1:11437", "ghost",
            cfg=sample_cfg_with_model,
        )
        assert result.error is not None
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_load_failure(self, sample_cfg_with_model, monkeypatch: pytest.MonkeyPatch):
        fake_client = FakeHttpxClient()
        fake_client._load_ok = False
        monkeypatch.setattr("arc_llama.benchmark.httpx.AsyncClient", lambda **kw: fake_client)
        result = await benchmark_model(
            "http://127.0.0.1:11437", "qwen3-7b",
            load=True, cfg=sample_cfg_with_model,
        )
        assert result.error is not None
        assert "Failed to load" in result.error


class TestBenchmarkSweep:
    @pytest.mark.asyncio
    async def test_sweep_two_configs(self, sample_cfg_with_model, monkeypatch: pytest.MonkeyPatch):
        fake_client = FakeHttpxClient()
        monkeypatch.setattr("arc_llama.benchmark.httpx.AsyncClient", lambda **kw: fake_client)
        monkeypatch.setattr("arc_llama.benchmark._find_drm_card", lambda slot: None)
        monkeypatch.setattr("arc_llama.benchmark._read_vram_used", lambda p: None)
        monkeypatch.setattr("arc_llama.benchmark._read_vram_total", lambda p: None)

        results = await benchmark_sweep(
            "http://127.0.0.1:11437", "qwen3-7b",
            ctx_values=[4096, 8192],
            kv_types=["q8_0"],
            prompt_tokens=32, gen_tokens=8,
            cfg=sample_cfg_with_model,
        )
        assert len(results) == 2
        assert results[0].ctx == 4096
        assert results[1].ctx == 8192
        # Each benchmark run + 2 edits per config + 1 restore edit = 5 edits
        edit_calls = [c for c in fake_client.calls if "/edit" in c[1]]
        assert len(edit_calls) >= 3  # 2 sweep edits + 1 restore

    @pytest.mark.asyncio
    async def test_sweep_edit_failure(self, sample_cfg_with_model, monkeypatch: pytest.MonkeyPatch):
        fake_client = FakeHttpxClient()
        fake_client._edit_ok = False
        monkeypatch.setattr("arc_llama.benchmark.httpx.AsyncClient", lambda **kw: fake_client)

        results = await benchmark_sweep(
            "http://127.0.0.1:11437", "qwen3-7b",
            ctx_values=[4096],
            kv_types=["q8_0"],
            cfg=sample_cfg_with_model,
        )
        assert len(results) == 1
        assert results[0].error is not None
        assert "Edit failed" in results[0].error


class TestPrintResult:
    def test_success(self, capsys):
        r = BenchmarkResult(
            model="qwen3-7b", ctx=8192, cache_type_k="q8_0", cache_type_v="q8_0",
            prompt_tokens=512, gen_tokens=128,
            prompt_eval_tok_s=847.0, prompt_eval_ms=604.0,
            generation_tok_s=34.0, generation_ms=3760.0,
            vram_used_mb=12288, vram_total_mb=24576,
            jit_warmup_s=18.4,
        )
        print_result(r)
        out = capsys.readouterr().out
        assert "qwen3-7b" in out
        assert "847.0" in out or "847" in out
        assert "18.4" in out

    def test_error(self, capsys):
        r = BenchmarkResult(
            model="x", ctx=0, cache_type_k="?", cache_type_v="?",
            prompt_tokens=1, gen_tokens=1,
            error="something broke",
        )
        print_result(r)
        out = capsys.readouterr().out
        assert "something broke" in out
