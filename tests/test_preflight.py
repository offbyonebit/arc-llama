from __future__ import annotations

import os
import socket

import pytest

from arc_llama.config import GPUConfig, ModelConfig
from arc_llama.failures import StartupFailureError
from arc_llama.launcher import LaunchPlan
from arc_llama.preflight import preflight_launch


def _inputs(tmp_path):
    runtime = tmp_path / ("llama-server.exe" if os.name == "nt" else "llama-server")
    runtime.write_bytes(b"runtime")
    if os.name != "nt":
        runtime.chmod(0o755)
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")
    gpu = GPUConfig("0000:03:00.0", 0, "battlemage", enabled=True)
    model = ModelConfig("model", str(model_path), 18080, gpu.pci_slot)
    plan = LaunchPlan(
        argv=[str(runtime), "-m", str(model_path), "--port", str(model.port)],
        env={},
        backend_url=f"http://127.0.0.1:{model.port}",
        health_url=f"http://127.0.0.1:{model.port}/health",
    )
    return runtime, model, gpu, plan


def test_missing_model_fails_without_starting_a_process(tmp_path, monkeypatch):
    _runtime, model, gpu, plan = _inputs(tmp_path)
    os.unlink(model.path)
    popen_called = False

    def unexpected_popen(*args, **kwargs):
        nonlocal popen_called
        popen_called = True
        raise AssertionError("preflight must not spawn")

    monkeypatch.setattr("subprocess.Popen", unexpected_popen)
    with pytest.raises(StartupFailureError) as caught:
        preflight_launch(model, gpu, plan)

    assert caught.value.category == "model_missing"
    assert "Update the model path" in str(caught.value)
    assert not popen_called


def test_missing_draft_is_identified_separately(tmp_path):
    _runtime, model, gpu, plan = _inputs(tmp_path)
    plan.argv += ["--spec-draft-model", str(tmp_path / "missing-draft.gguf")]

    with pytest.raises(StartupFailureError) as caught:
        preflight_launch(model, gpu, plan)

    assert caught.value.category == "draft_missing"
    assert "Draft model file not found" in str(caught.value)


def test_missing_runtime_is_actionable(tmp_path):
    _runtime, model, gpu, plan = _inputs(tmp_path)
    plan.argv[0] = str(tmp_path / "missing-runtime")

    with pytest.raises(StartupFailureError) as caught:
        preflight_launch(model, gpu, plan)

    assert caught.value.category == "runtime_missing"
    assert caught.value.http_status == 503


def test_disabled_gpu_is_rejected_before_filesystem_checks(tmp_path):
    _runtime, model, gpu, plan = _inputs(tmp_path)
    gpu.enabled = False

    with pytest.raises(StartupFailureError) as caught:
        preflight_launch(model, gpu, plan)

    assert caught.value.category == "gpu_unavailable"


def test_occupied_port_is_rejected(tmp_path):
    _runtime, model, gpu, plan = _inputs(tmp_path)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        model.port = listener.getsockname()[1]
        plan.backend_url = f"http://127.0.0.1:{model.port}"

        with pytest.raises(StartupFailureError) as caught:
            preflight_launch(model, gpu, plan)

    assert caught.value.category == "port_in_use"
    assert caught.value.http_status == 409
