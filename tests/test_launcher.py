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
from arc_llama.launcher import LaunchPlan, LlamaServer, build_env, build_plan
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

    def test_vulkan_q8_auto_injects_flash_attn(self):
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
            pci_slot="00:00.0", sycl_index=0, arch="battlemage", backend=Backend.VULKAN.value
        )
        plan = build_plan(cfg, model, gpu)
        assert "--flash-attn" in plan.argv
        # /bin/llama-server is not a real llama-server, so the Vulkan index
        # cannot be resolved and the variable is deliberately left unset
        # rather than guessed from sycl_index.
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


class TestBuildPlanMtp:
    def test_no_auto_ub_for_mtp(self, metadata_ggufs):
        """MTP detection must not force -ub 8; it regresses prompt-eval throughput."""
        cfg = Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})())
        model = ModelConfig(
            name="mtp-qwen",
            path=str(metadata_ggufs["mtp"]),
            port=18080,
            gpu_pci_slot="00:00.0",
        )
        gpu = GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage")
        plan = build_plan(cfg, model, gpu)
        assert "-ub" not in plan.argv

    def test_user_ubatch_size_not_overridden(self, metadata_ggufs):
        """If the recipe already has ubatch_size, don't stomp it."""
        cfg = Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})())
        model = ModelConfig(
            name="mtp-qwen",
            path=str(metadata_ggufs["mtp"]),
            port=18080,
            gpu_pci_slot="00:00.0",
            recipe={"ubatch_size": 16},
        )
        gpu = GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage")
        plan = build_plan(cfg, model, gpu)
        idx = plan.argv.index("-ub")
        assert plan.argv[idx + 1] == "16"

    def test_no_auto_ub_for_mtp_on_lunar_lake(self, metadata_ggufs):
        """Xe2 iGPU (Lunar Lake) should also avoid the forced micro-ubatch."""
        cfg = Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})())
        model = ModelConfig(
            name="mtp-qwen",
            path=str(metadata_ggufs["mtp"]),
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

    def test_tail_log_returns_last_lines(self, tmp_path, monkeypatch):
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
        import signal
        from unittest.mock import Mock

        signals = Mock()
        monkeypatch.setattr(os, "killpg", signals, raising=False)
        srv.stop()
        if sys.platform != "win32":
            signals.assert_called_once_with(12345, signal.SIGTERM)

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


@pytest.mark.parametrize("windows", [False, True])
def test_stop_timeout_retains_child_lock_and_log_until_exit(monkeypatch, tmp_path, windows):
    from unittest.mock import Mock

    from arc_llama import launcher
    from arc_llama.failures import StartupFailureError
    from arc_llama.launcher import LaunchPlan

    monkeypatch.setattr(launcher, "_IS_WINDOWS", windows)
    # This parameter exercises the POSIX process-group path even when the
    # test suite itself runs on Windows, where SIGKILL is not defined.
    monkeypatch.setattr(
        launcher.signal, "SIGKILL", getattr(launcher.signal, "SIGKILL", 9), raising=False
    )
    monkeypatch.setattr(launcher.os, "killpg", Mock(), raising=False)
    monkeypatch.setattr(launcher.subprocess, "run", Mock())
    srv = LlamaServer(LaunchPlan(argv=["unused"], env={}), "stuck")
    proc = Mock(pid=12345)
    proc.poll.return_value = None
    proc.wait.side_effect = subprocess.TimeoutExpired("fake", 0)
    srv.process = proc
    srv.ready = True
    srv.started_at = 123.0
    srv._log_file = open(tmp_path / "child.log", "wb")
    log_file = srv._log_file
    lock_file = open(tmp_path / "resident.lock", "a+")
    srv._resident_lock = lock_file

    with pytest.raises(StartupFailureError) as caught:
        srv.stop(drain_seconds=0)
    assert caught.value.details["reason"] == "shutdown_timeout"
    assert proc.wait.call_count == 2
    assert srv.process is proc and srv.is_running
    assert srv._resident_lock is lock_file and not lock_file.closed
    assert srv._log_file is log_file and not log_file.closed
    assert srv.started_at == 123.0
    assert not srv.ready

    # Later cleanup reaps the child and then releases resources, idempotently.
    proc.poll.return_value = -9
    srv.stop()
    srv.stop()
    assert srv.process is None and srv._resident_lock is None
    assert lock_file.closed and log_file.closed


def test_windows_taskkill_timeout_keeps_ownership(monkeypatch):
    from unittest.mock import Mock

    from arc_llama import launcher
    from arc_llama.failures import StartupFailureError
    from arc_llama.launcher import LaunchPlan

    monkeypatch.setattr(launcher, "_IS_WINDOWS", True)
    proc = Mock(pid=12345)
    proc.poll.return_value = None
    proc.wait.side_effect = subprocess.TimeoutExpired("fake", 0)
    taskkill = Mock(side_effect=subprocess.TimeoutExpired("taskkill", 0))
    monkeypatch.setattr(launcher.subprocess, "run", taskkill)
    srv = LlamaServer(LaunchPlan(argv=["unused"], env={}))
    srv.process = proc
    lock = Mock()
    srv._resident_lock = lock
    with pytest.raises(StartupFailureError) as caught:
        srv.stop(drain_seconds=0)
    assert caught.value.details["reason"] == "shutdown_timeout"
    assert srv.process is proc and srv._resident_lock is lock
    lock.close.assert_not_called()


@pytest.mark.asyncio
async def test_memory_guard_interrupts_a_stalled_health_request(monkeypatch):
    from unittest.mock import Mock

    from arc_llama import launcher
    from arc_llama.failures import StartupFailureError
    from arc_llama.launcher import LaunchPlan

    srv = LlamaServer(LaunchPlan(argv=["unused"], env={}), "target")
    srv.process = Mock(pid=12345)
    srv.process.poll.return_value = None
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def stalled_health(timeout):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    def pressure():
        return {"available_host_mb": 100, "reserved_host_mb": 4096} if entered.is_set() else None

    stops = []

    async def stop(drain_seconds=3.0):
        stops.append(drain_seconds)
        srv.process = None

    monkeypatch.setattr(srv, "_wait_ready_health", stalled_health)
    monkeypatch.setattr(srv, "astop", stop)
    monkeypatch.setattr(launcher, "host_memory_pressure", pressure)
    with pytest.raises(StartupFailureError) as caught:
        await asyncio.wait_for(srv.wait_ready(), timeout=2)
    assert caught.value.category == "out_of_memory"
    assert stops == [0.25]
    assert cancelled.is_set() and not srv.is_running


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_pressure_cleanup_wins_over_health_success(monkeypatch, cancelled):
    from arc_llama import launcher
    from arc_llama.failures import StartupFailureError
    from arc_llama.launcher import LaunchPlan

    srv = LlamaServer(LaunchPlan(argv=["unused"], env={}))
    stopping = asyncio.Event()
    release = asyncio.Event()

    async def health(timeout):
        await stopping.wait()
        srv.ready = True
        return True

    async def stop(drain_seconds=3.0):
        stopping.set()
        await release.wait()

    monkeypatch.setattr(srv, "_wait_ready_health", health)
    monkeypatch.setattr(srv, "astop", stop)
    monkeypatch.setattr(launcher, "host_memory_pressure", lambda: {
        "available_host_mb": 100, "reserved_host_mb": 4096,
    })
    waiter = asyncio.create_task(srv.wait_ready())
    await asyncio.wait_for(stopping.wait(), 2)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not waiter.done(), "Readiness cannot return during pressure cleanup"
    if cancelled:
        waiter.cancel()
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.sleep(0)
        assert not waiter.done(), "Cancellation cannot detach pressure cleanup"
    release.set()
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiter, 2)
    else:
        with pytest.raises(StartupFailureError) as caught:
            await asyncio.wait_for(waiter, 2)
        assert caught.value.category == "out_of_memory"
    assert not srv.ready


@pytest.mark.asyncio
async def test_astop_cancellation_waits_for_worker(monkeypatch):
    from arc_llama.launcher import LaunchPlan

    srv = LlamaServer(LaunchPlan(argv=["unused"], env={}))
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = []

    async def worker(func, *args):
        entered.set()
        await release.wait()
        finished.append(True)

    monkeypatch.setattr(asyncio, "to_thread", worker)
    stopper = asyncio.create_task(srv.astop())
    await asyncio.wait_for(entered.wait(), 2)
    stopper.cancel()
    await asyncio.sleep(0)
    stopper.cancel()  # repeated cancellation also must not detach cleanup
    await asyncio.sleep(0)
    assert not stopper.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(stopper, 2)
    assert finished == [True]


def test_concurrent_stops_share_one_cleanup(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import Mock

    from arc_llama.launcher import LaunchPlan

    srv = LlamaServer(LaunchPlan(argv=["unused"], env={}))
    entered = threading.Event()
    release = threading.Event()
    proc = Mock(pid=12345)
    proc.poll.return_value = None

    def waited(timeout):
        entered.set()
        assert release.wait(2)
        proc.poll.return_value = 0
        return 0

    proc.wait.side_effect = waited
    srv.process = proc
    signals = Mock()
    monkeypatch.setattr(os, "killpg", signals, raising=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(srv.stop)
        assert entered.wait(2)
        second = pool.submit(srv.stop)
        release.set()
        first.result(timeout=2)
        second.result(timeout=2)
    assert proc.wait.call_count == 1
    assert srv.process is None
    if sys.platform != "win32":
        assert signals.call_count == 1


def test_unit_test_signal_barriers():
    # These are fixture sentinels, not native signaling calls. The runner
    # additionally isolates process IDs from the user's desktop.
    for signal_call in (os.kill, os.killpg):
        with pytest.raises(AssertionError, match="signaling is prohibited"):
            signal_call(12345, 15)
    with pytest.raises(AssertionError, match="signaling is prohibited"):
        subprocess.Popen(["taskkill", "/F", "/PID", "12345"])


@pytest.mark.asyncio
async def test_wait_ready_cancelled_before_health_starts(monkeypatch):
    srv = LlamaServer(LaunchPlan(argv=[], env={}, backend_url="", health_url=""))
    started = []
    stopped = []

    async def health(timeout):
        started.append(True)
        return True

    async def cancel_before_yield(*args, **kwargs):
        raise asyncio.CancelledError()

    async def stop(drain_seconds=3.0):
        stopped.append(True)

    monkeypatch.setattr(srv, "_wait_ready_health", health)
    monkeypatch.setattr(srv, "astop", stop)
    monkeypatch.setattr(asyncio, "wait", cancel_before_yield)
    with pytest.raises(asyncio.CancelledError):
        await srv.wait_ready()
    assert started == []
    assert stopped == [True]
    assert srv.ready is False


@pytest.mark.asyncio
async def test_wait_ready_cancelled_during_final_gather(monkeypatch):
    srv = LlamaServer(LaunchPlan(argv=[], env={}, backend_url="", health_url=""))
    draining = asyncio.Event()
    release = asyncio.Event()
    stopped = []

    async def health(timeout):
        srv.ready = True
        return True

    async def memory(pressure_seen):
        try:
            await asyncio.Future()
        finally:
            draining.set()
            await release.wait()

    async def stop(drain_seconds=3.0):
        srv.ready = False
        stopped.append(True)

    monkeypatch.setattr(srv, "_wait_ready_health", health)
    monkeypatch.setattr(srv, "_guard_host_memory", memory)
    monkeypatch.setattr(srv, "astop", stop)
    waiter = asyncio.create_task(srv.wait_ready())
    await asyncio.wait_for(draining.wait(), 1)
    waiter.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert stopped == [True]
    assert srv.ready is False
