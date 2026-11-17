"""A fake resident llama-server exercises change checks without model/GPU inference."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from click.testing import CliRunner
from helpers import FakeServer, python_executable

import arc_llama.router as router_mod
import arc_llama.runtime_correctness as checks
from arc_llama.config import Config, GPUConfig, ModelConfig, ServerConfig
from arc_llama.router import Router


@pytest.fixture
def rig(tmp_path, monkeypatch):
    binary = tmp_path / "old" / "llama-server"
    binary.parent.mkdir()
    binary.write_bytes(b"controlled runtime one")
    (binary.parent / ".arc-llama-runtime.json").write_text(json.dumps({"tag": "b100"}), encoding="utf-8")
    cfg = Config(gpus=[GPUConfig("slot", 0, "battlemage", backend="sycl")],
                 models=[ModelConfig("m", str(tmp_path / "missing.gguf"), 18080, "slot")])
    cfg.paths.state_dir = str(tmp_path / "state")
    cfg.paths.llama_server = str(binary)
    FakeServer.starts = []
    FakeServer.stops = []
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)
    monkeypatch.setattr(router_mod, "build_plan", lambda *a, **k: SimpleNamespace(
        backend_url="http://127.0.0.1:18080", argv=[cfg.paths.llama_server], env={}))
    calls = []
    state = {"reply": "correct"}

    def serve(request):
        body = json.loads(request.content)
        calls.append(body)
        if state["reply"] == "timeout":
            raise httpx.ReadTimeout("controlled timeout", request=request)
        answer = ("The answer is 4." if "2 + 2" in body["messages"][0]["content"] else "Lantern!")
        content = "<think>4 lantern are in reasoning only</think> " + (answer if state["reply"] == "correct" else "garbage")
        return httpx.Response(200, json={"choices": [{"message": {"content": content, "reasoning_content": "4 lantern"}}]})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(checks.httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(serve), **kwargs))
    return SimpleNamespace(cfg=cfg, binary=binary, router=Router(cfg), calls=calls, state=state)


async def test_change_passes_once_and_survives_reload_and_restart(rig):
    cfg, rt = rig.cfg, rig.router
    await rt.ensure_active("m")
    assert rig.calls == []
    assert checks.read_checks(cfg)["m"]["status"] == "baseline"
    await rt.stop_one("m")
    rig.binary.write_bytes(b"controlled runtime changed and longer")
    _, server = await rt.ensure_active("m")
    assert len(rig.calls) == 2 and server.is_running
    record = checks.read_checks(cfg)["m"]
    assert record["status"] == "passed" and record["build"] == "b100"
    for body in rig.calls:
        assert body["temperature"] == 0 and body["max_tokens"] == 32
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
    await rt.stop_one("m")
    await rt.ensure_active("m")
    assert len(rig.calls) == 2
    await rt.shutdown()
    restarted = Router(cfg)
    await restarted.ensure_active("m")
    assert len(rig.calls) == 2
    assert checks.read_checks(cfg)["m"]["checked_at"] == record["checked_at"]
    await restarted.shutdown()


@pytest.mark.parametrize("reply,requests", [("garbage", 2), ("timeout", 1)])
async def test_bad_or_unavailable_answers_warn_without_stopping(rig, reply, requests):
    rig.state["reply"] = reply
    checks.mark_runtime_changed(rig.cfg)
    _, server = await rig.router.ensure_active("m")
    assert server.is_running and server.ready and FakeServer.stops == []
    warning = rig.router.correctness.warning("m")
    assert warning["status"] == "warning" and warning["build"] == "b100"
    assert warning["detail"]
    assert len(rig.calls) == requests
    if reply == "garbage":
        assert warning["detail"].count("Expected") == 2  # reasoning tokens cannot make the answers pass
    await rig.router.shutdown()


async def test_backend_change_checks_but_recipe_reload_does_not(rig):
    await rig.router.ensure_active("m")
    await rig.router.stop_one("m")
    rig.cfg.gpus[0].backend = "vulkan"
    await rig.router.ensure_active("m")
    assert len(rig.calls) == 2
    await rig.router.stop_one("m")
    rig.cfg.models[0].recipe["ctx"] = 16384
    await rig.router.ensure_active("m")
    assert len(rig.calls) == 2
    await rig.router.shutdown()


async def test_path_runtime_with_model_environment_is_checked(rig, tmp_path, monkeypatch):
    import os

    binary = python_executable(tmp_path / "tools" / "runtime", "print('version: 102 (fixture)')\n")
    path = Path(binary)
    (path.parent / ".arc-llama-runtime.json").write_text(json.dumps({"tag": "b102"}), encoding="utf-8")
    monkeypatch.setenv("PATH", str(path.parent) + os.pathsep + os.environ.get("PATH", ""))
    rig.cfg.paths.llama_server = path.name
    rig.cfg.models[0].env = {"BACKEND_FLAG": "literal"}
    await rig.router.ensure_active("m")
    assert len(rig.calls) == 2
    assert checks.read_checks(rig.cfg)["m"]["build"] == "b102"
    await rig.router.shutdown()


async def test_switch_off_makes_no_inference_requests(rig):
    rig.cfg.server.runtime_self_check = False
    checks.mark_runtime_changed(rig.cfg)
    await rig.router.ensure_active("m")
    assert rig.calls == []
    assert checks.read_checks(rig.cfg)["m"]["status"] == "pending"
    await rig.router.shutdown()


@pytest.mark.parametrize("kind", ["embedding", "reranking"])
async def test_non_chat_models_are_not_sent_chat_prompts(rig, kind):
    rig.cfg.models[0].recipe[kind] = True
    checks.mark_runtime_changed(rig.cfg)
    await rig.router.ensure_active("m")
    assert rig.calls == []
    await rig.router.shutdown()


async def test_update_then_rollback_each_schedule_one_check(rig, tmp_path):
    from arc_llama.runtime_update import rollback_runtime, update_runtime
    from arc_llama.server_caps import ServerCaps

    await rig.router.ensure_active("m")
    await rig.router.stop_one("m")
    candidate = tmp_path / "new" / "llama-server"
    candidate.parent.mkdir()
    candidate.write_bytes(b"runtime two")
    (candidate.parent / ".arc-llama-runtime.json").write_text(json.dumps({"tag": "b101"}), encoding="utf-8")
    result = update_runtime(rig.cfg, tmp_path / "config.toml", canary=False,
                            installer=lambda **_: SimpleNamespace(binary_path=candidate, tag="b101"),
                            caps_probe=lambda _: ServerCaps(probed=True))
    assert result.switched and checks.read_checks(rig.cfg)["m"]["pending"]
    await rig.router.rebuild_model("m")
    await rig.router.ensure_active("m")
    assert len(rig.calls) == 2 and checks.read_checks(rig.cfg)["m"]["build"] == "b101"
    await rig.router.stop_one("m")
    assert rollback_runtime(rig.cfg, tmp_path / "config.toml")
    await rig.router.rebuild_model("m")
    await rig.router.ensure_active("m")
    assert len(rig.calls) == 4 and checks.read_checks(rig.cfg)["m"]["build"] == "b100"
    await rig.router.shutdown()


def test_doctor_reports_saved_warning_with_build_and_rollback(rig, tmp_path, monkeypatch):
    import arc_llama.cli as cli_mod
    import arc_llama.cli_diagnostics as diagnostics
    from arc_llama.arch import Backend
    from arc_llama.cli import cli

    checks._save(rig.cfg, {"m": {"status": "warning", "build": "b100", "detail": "Expected a correct short answer"}})
    path = tmp_path / "config.toml"
    rig.cfg.save(path)
    monkeypatch.setattr(cli_mod, "_IS_WINDOWS", True)
    monkeypatch.setattr(cli_mod, "detect_gpus", lambda **_: [])
    monkeypatch.setattr(diagnostics, "detect_backends", lambda _: {Backend.SYCL})
    monkeypatch.setattr(diagnostics, "detect_llama_server_backend", lambda _: Backend.SYCL)
    result = CliRunner().invoke(cli, ["--config", str(path), "doctor"])
    text = " ".join(result.output.split())
    assert "runtime self-check" in text and "b100" in text and "rollback" in text


def test_self_check_config_default_and_validation():
    assert ServerConfig().runtime_self_check is True
    with pytest.raises(ValueError, match="runtime_self_check"):
        ServerConfig(runtime_self_check="yes")
