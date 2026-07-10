"""Tests for arc_llama.launcher — env construction, command-line building."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from arc_llama.arch import Arch
from arc_llama.config import Config, GPUConfig, ModelConfig
from arc_llama.launcher import LlamaServer, build_env, build_plan


class TestBuildEnv:
    def test_sets_device_selector(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin"})
        from arc_llama.arch import profile_for
        profile = profile_for(Arch.BATTLEMAGE)
        env = build_env(profile, sycl_index=2)
        assert env["ONEAPI_DEVICE_SELECTOR"] == "level_zero:2"

    def test_strips_bad_vars(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(os, "environ", {
            "PATH": "/usr/bin",
            "GGML_SYCL_DISABLE_OPT": "1",
            "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "1",
        })
        from arc_llama.arch import profile_for
        profile = profile_for(Arch.BATTLEMAGE)
        env = build_env(profile, sycl_index=0)
        assert "GGML_SYCL_DISABLE_OPT" not in env
        assert "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS" not in env

    def test_applies_arch_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin"})
        from arc_llama.arch import profile_for
        profile = profile_for(Arch.BATTLEMAGE)
        env = build_env(profile, sycl_index=0)
        assert env["SYCL_CACHE_PERSISTENT"] == "0"
        assert env["ZES_ENABLE_SYSMAN"] == "1"


class TestBuildPlan:
    def test_includes_model_and_port(self):
        cfg = Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})())
        model = ModelConfig(name="m", path="/m.gguf", port=18080, gpu_pci_slot="00:00.0")
        gpu = GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage")
        plan = build_plan(cfg, model, gpu)
        assert plan.argv[0] == "/bin/llama-server"
        assert "-m" in plan.argv
        assert "/m.gguf" in plan.argv
        assert "--port" in plan.argv
        assert "18080" in plan.argv
        assert plan.backend_url == "http://127.0.0.1:18080"

    def test_uses_custom_host(self):
        cfg = Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})())
        model = ModelConfig(name="m", path="/m.gguf", port=18080, gpu_pci_slot="00:00.0")
        gpu = GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage")
        plan = build_plan(cfg, model, gpu, host="0.0.0.0")
        assert plan.backend_url == "http://0.0.0.0:18080"
        assert "--host" in plan.argv
        assert "0.0.0.0" in plan.argv

    def test_env_has_device_selector(self):
        cfg = Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})())
        model = ModelConfig(name="m", path="/m.gguf", port=18080, gpu_pci_slot="00:00.0")
        gpu = GPUConfig(pci_slot="00:00.0", sycl_index=2, arch="battlemage")
        plan = build_plan(cfg, model, gpu)
        assert plan.env["ONEAPI_DEVICE_SELECTOR"] == "level_zero:2"

    def test_fake_path_no_mtp_no_ub_injected(self):
        """Non-existent GGUF → no MTP heads → -ub should NOT appear."""
        cfg = Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})())
        model = ModelConfig(name="m", path="/m.gguf", port=18080, gpu_pci_slot="00:00.0")
        gpu = GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage")
        plan = build_plan(cfg, model, gpu)
        assert "-ub" not in plan.argv


class TestBuildPlanFlashAttn:
    def _plan(self, recipe, caps, monkeypatch):
        from arc_llama.server_caps import ServerCaps
        monkeypatch.setattr(
            "arc_llama.launcher.probe_server_caps", lambda path: ServerCaps(**caps)
        )
        cfg = Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})())
        model = ModelConfig(
            name="m", path="/m.gguf", port=18080, gpu_pci_slot="00:00.0", recipe=recipe,
        )
        gpu = GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage")
        return build_plan(cfg, model, gpu)

    def test_modern_binary_gets_fa_with_value(self, monkeypatch):
        plan = self._plan(
            {"flash_attn": "on"},
            {"supports_flash_attn": True, "flash_attn_takes_value": True, "probed": True},
            monkeypatch,
        )
        idx = plan.argv.index("-fa")
        assert plan.argv[idx + 1] == "on"

    def test_old_binary_gets_bare_fa_for_on(self, monkeypatch):
        plan = self._plan(
            {"flash_attn": "on"},
            {"supports_flash_attn": True, "flash_attn_takes_value": False, "probed": True},
            monkeypatch,
        )
        idx = plan.argv.index("-fa")
        # bare flag: next token (if any) is another option, not a value
        assert idx == len(plan.argv) - 1 or plan.argv[idx + 1].startswith("-")

    def test_old_binary_auto_omitted(self, monkeypatch):
        plan = self._plan(
            {"flash_attn": "auto"},
            {"supports_flash_attn": True, "flash_attn_takes_value": False, "probed": True},
            monkeypatch,
        )
        assert "-fa" not in plan.argv

    def test_unsupported_binary_omits_fa(self, monkeypatch):
        plan = self._plan(
            {"flash_attn": "on"},
            {"supports_flash_attn": False, "flash_attn_takes_value": False, "probed": True},
            monkeypatch,
        )
        assert "-fa" not in plan.argv

    def test_batch_flags_from_recipe(self, monkeypatch):
        plan = self._plan(
            {"ubatch_size": 1024, "batch_size": 2048},
            {"supports_flash_attn": True, "flash_attn_takes_value": True, "probed": True},
            monkeypatch,
        )
        assert plan.argv[plan.argv.index("-ub") + 1] == "1024"
        assert plan.argv[plan.argv.index("-b") + 1] == "2048"


# Real GGUF fixtures for MTP integration tests.
_MTP_QWEN = Path("/mnt/storage/models/qwen3.6-27b/Qwen3.6-27B-MTP-UD-Q4_K_XL.gguf")


def _have_mtp_fixture() -> bool:
    return _MTP_QWEN.exists()


class TestBuildPlanMtp:
    @pytest.mark.skipif(not _have_mtp_fixture(), reason="MTP fixture GGUF not on disk")
    def test_auto_injects_ub_8_for_mtp(self):
        cfg = Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})())
        model = ModelConfig(
            name="mtp-qwen",
            path=str(_MTP_QWEN),
            port=18080,
            gpu_pci_slot="00:00.0",
        )
        gpu = GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage")
        plan = build_plan(cfg, model, gpu)
        assert "-ub" in plan.argv
        idx = plan.argv.index("-ub")
        assert plan.argv[idx + 1] == "8"

    @pytest.mark.skipif(not _have_mtp_fixture(), reason="MTP fixture GGUF not on disk")
    def test_user_ubatch_size_not_overridden(self):
        """If the recipe already has ubatch_size, don't stomp it."""
        cfg = Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})())
        model = ModelConfig(
            name="mtp-qwen",
            path=str(_MTP_QWEN),
            port=18080,
            gpu_pci_slot="00:00.0",
            recipe={"ubatch_size": 16},
        )
        gpu = GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage")
        plan = build_plan(cfg, model, gpu)
        idx = plan.argv.index("-ub")
        assert plan.argv[idx + 1] == "16"

    @pytest.mark.skipif(not _have_mtp_fixture(), reason="MTP fixture GGUF not on disk")
    def test_mtp_on_lunar_lake_also_gets_ub_8(self):
        """Xe2 iGPU (Lunar Lake) is the same generation as Battlemage — same fix."""
        cfg = Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})())
        model = ModelConfig(
            name="mtp-qwen",
            path=str(_MTP_QWEN),
            port=18080,
            gpu_pci_slot="00:00.0",
        )
        gpu = GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="lunar_lake")
        plan = build_plan(cfg, model, gpu)
        assert "-ub" in plan.argv
        idx = plan.argv.index("-ub")
        assert plan.argv[idx + 1] == "8"


class TestLlamaServerLifecycle:
    def test_not_running_before_start(self):
        plan = build_plan(
            Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})()),
            ModelConfig(name="m", path="/m.gguf", port=18080, gpu_pci_slot="00:00.0"),
            GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage"),
        )
        srv = LlamaServer(plan)
        assert srv.is_running is False

    def test_start_log_dir_creates_parents(self, tmp_path: Path):
        plan = build_plan(
            Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})()),
            ModelConfig(name="m", path="/m.gguf", port=18080, gpu_pci_slot="00:00.0"),
            GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage"),
        )
        srv = LlamaServer(plan)
        log_dir = tmp_path / "deep" / "logs"
        # We can't actually start a fake binary without mocking Popen,
        # but we can at least assert the log_dir path would be used.
        assert not log_dir.exists()
        # Mock Popen to avoid actually spawning
        import subprocess
        original_popen = subprocess.Popen
        called = {}

        def _fake_popen(*args, **kwargs):
            called["args"] = args
            called["kwargs"] = kwargs
            class FakeProc:
                pid = 12345
                def poll(self):
                    return None
            return FakeProc()

        subprocess.Popen = _fake_popen
        try:
            srv.start(log_dir=log_dir)
            assert log_dir.exists()
        finally:
            subprocess.Popen = original_popen
        assert srv.is_running is True

    @pytest.mark.asyncio
    async def test_wait_ready_true_when_healthy(self, monkeypatch: pytest.MonkeyPatch):
        plan = build_plan(
            Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})()),
            ModelConfig(name="m", path="/m.gguf", port=18080, gpu_pci_slot="00:00.0"),
            GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage"),
        )
        srv = LlamaServer(plan)
        # Pretend it's running
        srv.process = type("P", (), {"poll": lambda self: None, "pid": 1})()
        srv.started_at = 0.0

        import httpx
        original_get = httpx.AsyncClient.get

        async def _fake_get(self, url):
            if "/health" in url:
                return type("R", (), {"status_code": 200, "json": lambda self: {"status": "ok"}})()
            return type("R", (), {"status_code": 404})()

        monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
        ready = await srv.wait_ready(timeout=2.0)
        assert ready is True

    @pytest.mark.asyncio
    async def test_wait_ready_false_on_crash(self):
        plan = build_plan(
            Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})()),
            ModelConfig(name="m", path="/m.gguf", port=18080, gpu_pci_slot="00:00.0"),
            GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage"),
        )
        srv = LlamaServer(plan)
        # Simulate crashed process
        srv.process = type("P", (), {"poll": lambda self: 1})()
        ready = await srv.wait_ready(timeout=1.0)
        assert ready is False

    def test_stop_idempotent(self):
        plan = build_plan(
            Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})()),
            ModelConfig(name="m", path="/m.gguf", port=18080, gpu_pci_slot="00:00.0"),
            GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage"),
        )
        srv = LlamaServer(plan)
        # Should not raise when not running
        srv.stop()
        assert srv.is_running is False
