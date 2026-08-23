"""Tests for arc_llama.launcher — env construction, command-line building."""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from arc_llama.arch import Arch, Backend
from arc_llama.config import Config, GPUConfig, ModelConfig
from arc_llama.launcher import LlamaServer, build_env, build_plan
from arc_llama.recipes import KVCacheType


def _gpu(sycl_index: int = 0, backend: Backend = Backend.SYCL) -> GPUConfig:
    return GPUConfig(
        pci_slot="00:00.0",
        sycl_index=sycl_index,
        arch="battlemage",
        backend=backend.value,
    )


class TestBuildEnv:
    def test_sets_device_selector(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin"})
        from arc_llama.arch import profile_for
        profile = profile_for(Arch.BATTLEMAGE)
        env = build_env(profile, _gpu(sycl_index=2))
        assert env["ONEAPI_DEVICE_SELECTOR"] == "level_zero:2"

    def test_strips_bad_vars(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(os, "environ", {
            "PATH": "/usr/bin",
            "GGML_SYCL_DISABLE_OPT": "1",
            "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "1",
        })
        from arc_llama.arch import profile_for
        profile = profile_for(Arch.BATTLEMAGE)
        env = build_env(profile, _gpu(sycl_index=0))
        assert "GGML_SYCL_DISABLE_OPT" not in env
        assert "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS" not in env

    def test_applies_arch_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin"})
        from arc_llama.arch import profile_for
        profile = profile_for(Arch.BATTLEMAGE)
        env = build_env(profile, _gpu(sycl_index=0))
        assert env["SYCL_CACHE_PERSISTENT"] == "0"
        assert env["ZES_ENABLE_SYSMAN"] == "1"

    def test_vulkan_uses_visible_devices(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin"})
        from arc_llama.arch import profile_for
        profile = profile_for(Arch.BATTLEMAGE)
        gpu = _gpu(sycl_index=2, backend=Backend.VULKAN)
        gpu.vulkan_index = 3
        env = build_env(profile, gpu)
        assert env["GGML_VK_VISIBLE_DEVICES"] == "3"
        assert "ONEAPI_DEVICE_SELECTOR" not in env

    def test_vulkan_never_falls_back_to_sycl_index(self, monkeypatch: pytest.MonkeyPatch):
        """sycl_index is a Level-Zero index and must never be used as a Vulkan one.

        SYCL enumerates Intel devices only; Vulkan enumerates every vendor. On a
        machine with a discrete NVIDIA card the Arc is Vulkan1 while sycl_index
        is still 0, so passing sycl_index ran models on the NVIDIA GPU. With no
        way to resolve the real index we must leave the variable unset rather
        than guess.
        """
        monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin"})
        from arc_llama.arch import profile_for
        profile = profile_for(Arch.BATTLEMAGE)
        env = build_env(profile, _gpu(sycl_index=2, backend=Backend.VULKAN))
        assert "GGML_VK_VISIBLE_DEVICES" not in env

    def test_vulkan_strips_sycl_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(os, "environ", {
            "PATH": "/usr/bin",
            "ONEAPI_DEVICE_SELECTOR": "level_zero:0",
            "SYCL_CACHE_PERSISTENT": "1",
            "ZES_ENABLE_SYSMAN": "1",
        })
        from arc_llama.arch import profile_for
        profile = profile_for(Arch.BATTLEMAGE)
        env = build_env(profile, _gpu(sycl_index=0, backend=Backend.VULKAN))
        assert "ONEAPI_DEVICE_SELECTOR" not in env
        assert "SYCL_CACHE_PERSISTENT" not in env
        assert "ZES_ENABLE_SYSMAN" not in env

    def test_sycl_sources_setvars_when_runtime_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        if sys.platform == "win32":
            pytest.skip("bash setvars sourcing is not exercised on Windows")
        monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin"})
        setvars = tmp_path / "setvars.sh"
        setvars.write_text("export ONEAPI_ROOT=/fake/oneapi\nexport LD_LIBRARY_PATH=/fake/lib\n")

        from arc_llama.arch import profile_for
        profile = profile_for(Arch.BATTLEMAGE)

        # Force the missing-runtime path and point it at our fake script.
        monkeypatch.setattr(
            "arc_llama.launcher.oneapi_runtime_env_needed", lambda: True
        )
        monkeypatch.setattr(
            "arc_llama.launcher.oneapi_setvars_path", lambda: setvars
        )

        env = build_env(profile, _gpu(sycl_index=0))
        assert env["ONEAPI_ROOT"] == "/fake/oneapi"
        assert env["LD_LIBRARY_PATH"] == "/fake/lib"
        # Our device selector must still win.
        assert env["ONEAPI_DEVICE_SELECTOR"] == "level_zero:0"

    def test_sycl_does_not_source_when_runtime_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin"})
        setvars = tmp_path / "setvars.sh"
        setvars.write_text("export ONEAPI_ROOT=/should-not-apply\n")

        from arc_llama.arch import profile_for
        profile = profile_for(Arch.BATTLEMAGE)

        monkeypatch.setattr(
            "arc_llama.launcher.oneapi_runtime_env_needed", lambda: False
        )
        monkeypatch.setattr(
            "arc_llama.launcher.oneapi_setvars_path", lambda: setvars
        )

        env = build_env(profile, _gpu(sycl_index=0))
        assert "ONEAPI_ROOT" not in env


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

    def test_sycl_q8_does_not_inject_flash_attn(self):
        # Matches production: q8 V, no --flash-attn, SYCL serves fine.
        cfg = Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})())
        model = ModelConfig(
            name="m",
            path="/m.gguf",
            port=18080,
            gpu_pci_slot="00:00.0",
            recipe={
                "cache_type_k": KVCacheType.Q8_0.value,
                "cache_type_v": KVCacheType.Q8_0.value,
            },
        )
        gpu = GPUConfig(
            pci_slot="00:00.0", sycl_index=0, arch="battlemage", backend=Backend.SYCL.value
        )
        plan = build_plan(cfg, model, gpu)
        assert "--flash-attn" not in plan.argv
        assert "-fa" not in plan.argv

    def test_vulkan_q8_auto_injects_flash_attn(self, tmp_path):
        # Must be a path that cannot be a working llama-server. Naming a real
        # location like /bin/llama-server made this test pass only on machines
        # that don't have llama.cpp installed — i.e. CI but not a dev box.
        missing = tmp_path / "not-a-llama-server"
        cfg = Config(paths=type("P", (), {"llama_server": str(missing)})())
        model = ModelConfig(
            name="m",
            path="/m.gguf",
            port=18080,
            gpu_pci_slot="00:00.0",
            recipe={
                "cache_type_k": KVCacheType.Q8_0.value,
                "cache_type_v": KVCacheType.Q8_0.value,
            },
        )
        gpu = GPUConfig(
            pci_slot="00:00.0", sycl_index=0, arch="battlemage", backend=Backend.VULKAN.value
        )
        plan = build_plan(cfg, model, gpu)
        assert "--flash-attn" in plan.argv
        # The binary cannot be queried for its device list, so the Vulkan index
        # is deliberately left unset rather than guessed from sycl_index.
        assert "GGML_VK_VISIBLE_DEVICES" not in plan.env

    def test_vulkan_q8_with_flash_attn_already_set(self):
        cfg = Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})())
        model = ModelConfig(
            name="m",
            path="/m.gguf",
            port=18080,
            gpu_pci_slot="00:00.0",
            recipe={
                "cache_type_k": KVCacheType.Q8_0.value,
                "cache_type_v": KVCacheType.Q8_0.value,
                "extra_flags": ["--flash-attn", "on"],
            },
        )
        gpu = GPUConfig(
            pci_slot="00:00.0", sycl_index=0, arch="battlemage", backend=Backend.VULKAN.value
        )
        plan = build_plan(cfg, model, gpu)
        assert plan.argv.count("--flash-attn") == 1


# Real GGUF fixtures for MTP integration tests.
_MTP_QWEN = Path("/mnt/storage/models/qwen3.6-27b/Qwen3.6-27B-MTP-UD-Q4_K_XL.gguf")


def _have_mtp_fixture() -> bool:
    return _MTP_QWEN.exists()


class TestBuildPlanMtp:
    @pytest.mark.skipif(not _have_mtp_fixture(), reason="MTP fixture GGUF not on disk")
    def test_no_auto_ub_for_mtp(self):
        """MTP detection must not force -ub 8; it regresses prompt-eval throughput."""
        cfg = Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})())
        model = ModelConfig(
            name="mtp-qwen",
            path=str(_MTP_QWEN),
            port=18080,
            gpu_pci_slot="00:00.0",
        )
        gpu = GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage")
        plan = build_plan(cfg, model, gpu)
        assert "-ub" not in plan.argv

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
    def test_no_auto_ub_for_mtp_on_lunar_lake(self):
        """Xe2 iGPU (Lunar Lake) should also avoid the forced micro-ubatch."""
        cfg = Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})())
        model = ModelConfig(
            name="mtp-qwen",
            path=str(_MTP_QWEN),
            port=18080,
            gpu_pci_slot="00:00.0",
        )
        gpu = GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="lunar_lake")
        plan = build_plan(cfg, model, gpu)
        assert "-ub" not in plan.argv


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

    @pytest.mark.asyncio
    async def test_wait_ready_cancellation_calls_astop(self, monkeypatch):
        plan = build_plan(
            Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})()),
            ModelConfig(name="m", path="/m.gguf", port=18080, gpu_pci_slot="00:00.0"),
            GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage"),
        )
        srv = LlamaServer(plan)
        srv.process = type("P", (), {"poll": lambda self: None, "pid": 1})()
        srv.started_at = 0.0

        astops = []
        async def _recording_astop(drain_seconds=3.0):
            astops.append(drain_seconds)
        monkeypatch.setattr(srv, "astop", _recording_astop)

        import httpx
        async def _fake_get(self, url):
            return type("R", (), {"status_code": 503})()
        monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)

        async def _sleep_then_cancel(delay):
            raise asyncio.CancelledError("mock cancel")
        monkeypatch.setattr(asyncio, "sleep", _sleep_then_cancel)

        with pytest.raises(asyncio.CancelledError):
            await srv.wait_ready(timeout=2.0)
        assert astops == [3.0]

    @pytest.mark.asyncio
    async def test_wait_ready_cancellation_falls_back_to_blocking_stop(self, monkeypatch):
        """If astop() is itself cancelled, the blocking stop() must still run.

        CancelledError is a BaseException, so a naive `except Exception` around
        the async cleanup would let it escape and orphan the subprocess.
        """
        plan = build_plan(
            Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})()),
            ModelConfig(name="m", path="/m.gguf", port=18080, gpu_pci_slot="00:00.0"),
            GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage"),
        )
        srv = LlamaServer(plan)
        srv.process = type("P", (), {"poll": lambda self: None, "pid": 1})()
        srv.started_at = 0.0

        async def _cancelled_astop(drain_seconds=3.0):
            raise asyncio.CancelledError("cancelled during cleanup")
        monkeypatch.setattr(srv, "astop", _cancelled_astop)

        stops = []
        monkeypatch.setattr(srv, "stop", lambda *a, **k: stops.append(True))

        import httpx
        async def _fake_get(self, url):
            return type("R", (), {"status_code": 503})()
        monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)

        async def _sleep_then_cancel(delay):
            raise asyncio.CancelledError("mock cancel")
        monkeypatch.setattr(asyncio, "sleep", _sleep_then_cancel)

        with pytest.raises(asyncio.CancelledError):
            await srv.wait_ready(timeout=2.0)
        assert stops == [True], "blocking stop() must run when astop() is cancelled"

    @pytest.mark.asyncio
    async def test_astop_offloads_to_thread(self, monkeypatch):
        plan = build_plan(
            Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})()),
            ModelConfig(name="m", path="/m.gguf", port=18080, gpu_pci_slot="00:00.0"),
            GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage"),
        )
        srv = LlamaServer(plan)

        to_thread_calls = []
        async def _fake_to_thread(func, *args, **kwargs):
            to_thread_calls.append((func, args, kwargs))
            return func(*args, **kwargs)
        monkeypatch.setattr(asyncio, "to_thread", _fake_to_thread)

        stop_calls = []
        def _recording_stop(drain_seconds=3.0):
            stop_calls.append(drain_seconds)
        monkeypatch.setattr(srv, "stop", _recording_stop)

        await srv.astop(drain_seconds=1.5)
        assert to_thread_calls
        assert stop_calls == [1.5]


class TestLogHandling:
    def test_log_rotation_renames_existing_large_log(self, tmp_path):
        from arc_llama import launcher as launcher_mod

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_path = log_dir / "m.log"
        log_path.write_bytes(b"x" * (launcher_mod._MAX_LOG_BYTES + 1))
        launcher_mod._rotate_log(log_path)
        assert not log_path.exists()
        assert (log_dir / "m.log.1").exists()

    def test_tail_log_returns_last_lines(self, tmp_path):
        from arc_llama.config import Config, GPUConfig, ModelConfig

        plan = build_plan(
            Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})()),
            ModelConfig(name="m", path="/m.gguf", port=18080, gpu_pci_slot="00:00.0"),
            GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage"),
        )
        srv = LlamaServer(plan)
        log_dir = tmp_path / "logs"
        original_popen = subprocess.Popen

        def _fake_popen(*args, **kwargs):
            class FakeProc:
                pid = 12345
                def poll(self):
                    return None
                def send_signal(self, sig):
                    pass
                def wait(self, timeout):
                    self._waited = True
            return FakeProc()

        subprocess.Popen = _fake_popen
        try:
            srv.start(log_dir=log_dir)
            srv._log_file.write(b"line1\nline2\nline3\n")
            srv._log_file.flush()
            assert srv.tail_log(lines=2) == "line2\nline3"
        finally:
            subprocess.Popen = original_popen
        srv.stop()

    def test_start_closes_log_file_on_popen_failure(self, monkeypatch, tmp_path):
        plan = build_plan(
            Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})()),
            ModelConfig(name="m", path="/m.gguf", port=18080, gpu_pci_slot="00:00.0"),
            GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage"),
        )
        srv = LlamaServer(plan)
        log_dir = tmp_path / "logs"

        closes = []
        class FakeFile:
            def write(self, data):
                pass
            def flush(self):
                pass
            def close(self):
                closes.append(True)

        real_open = open
        def _fake_open(path, mode="r", *args, **kwargs):
            if mode == "ab" and str(path).endswith(".log"):
                return FakeFile()
            return real_open(path, mode, *args, **kwargs)
        monkeypatch.setattr("builtins.open", _fake_open)

        def _fake_popen(*args, **kwargs):
            raise FileNotFoundError("llama-server not found")
        monkeypatch.setattr(subprocess, "Popen", _fake_popen)

        with pytest.raises(FileNotFoundError):
            srv.start(log_dir=log_dir)
        assert closes
        assert srv._log_path is None


class TestWindowsLifecycle:
    def test_start_uses_create_new_process_group(self, monkeypatch, tmp_path):
        from arc_llama import launcher as launcher_mod

        monkeypatch.setattr(launcher_mod, "_IS_WINDOWS", True)
        plan = build_plan(
            Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})()),
            ModelConfig(name="m", path="/m.gguf", port=18080, gpu_pci_slot="00:00.0"),
            GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage"),
        )
        srv = LlamaServer(plan)
        log_dir = tmp_path / "logs"
        called = {}
        original_popen = subprocess.Popen

        def _fake_popen(*args, **kwargs):
            called["kwargs"] = kwargs
            class FakeProc:
                pid = 12345
                def poll(self):
                    return None
            return FakeProc()

        subprocess.Popen = _fake_popen
        try:
            srv.start(log_dir=log_dir)
        finally:
            subprocess.Popen = original_popen
        assert called["kwargs"]["creationflags"] == getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
        assert "preexec_fn" not in called["kwargs"]

    def test_stop_sends_ctrl_break_then_force_kills_tree_on_timeout(self, monkeypatch):
        from arc_llama import launcher as launcher_mod

        monkeypatch.setattr(launcher_mod, "_IS_WINDOWS", True)
        plan = build_plan(
            Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})()),
            ModelConfig(name="m", path="/m.gguf", port=18080, gpu_pci_slot="00:00.0"),
            GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage"),
        )
        srv = LlamaServer(plan)
        calls = []

        class FakeProc:
            pid = 12345
            def poll(self):
                return None
            def send_signal(self, sig):
                calls.append(("send_signal", sig))
            def wait(self, timeout):
                if not any(c[0] == "taskkill" for c in calls):
                    raise subprocess.TimeoutExpired("cmd", timeout)

        def _fake_run(cmd, **kwargs):
            calls.append(("taskkill", cmd))
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", _fake_run)
        srv.process = FakeProc()
        srv.stop(drain_seconds=0.1)
        assert calls[0] == ("send_signal", launcher_mod._CTRL_BREAK_EVENT)
        assert calls[1][0] == "taskkill"
        assert calls[1][1] == ["taskkill", "/F", "/T", "/PID", "12345"]

    def test_stop_skips_force_kill_when_ctrl_break_succeeds(self, monkeypatch):
        from arc_llama import launcher as launcher_mod

        monkeypatch.setattr(launcher_mod, "_IS_WINDOWS", True)
        plan = build_plan(
            Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})()),
            ModelConfig(name="m", path="/m.gguf", port=18080, gpu_pci_slot="00:00.0"),
            GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage"),
        )
        srv = LlamaServer(plan)
        calls = []

        class FakeProc:
            pid = 12345
            def poll(self):
                return None
            def send_signal(self, sig):
                calls.append(("send_signal", sig))
            def wait(self, timeout):
                return None

        def _fake_run(cmd, **kwargs):
            calls.append(("taskkill", cmd))
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", _fake_run)
        srv.process = FakeProc()
        srv.stop(drain_seconds=0.1)
        assert calls == [("send_signal", launcher_mod._CTRL_BREAK_EVENT)]


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
