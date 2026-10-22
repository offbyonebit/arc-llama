"""Per-model runtime isolation with synthetic executables and configurations."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from helpers import FakeServer

from arc_llama import model_compatibility as compatibility
from arc_llama.autotune import compute_fingerprint
from arc_llama.cli import cli
from arc_llama.config import Config, GPUConfig, ModelConfig, load_config
from arc_llama.launcher import build_plan
from arc_llama.router import Router
from arc_llama.runtime_update import pick_canary_model, required_capabilities, run_canary
from arc_llama.server import create_app
from arc_llama.server_caps import probe_server_caps


def executable(path: Path, help_text: str = "--flash-attn on off auto") -> str:
    path.write_text(
        f"#!{sys.executable}\nimport os, sys, json\n"
        f"if '--help' in sys.argv: print({help_text!r})\n"
        "else: print(json.dumps({'argv': sys.argv[1:], 'env': os.environ.get('ARC_TEST_ONLY')}))\n"
    )
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def runtime_cfg(tmp_path, monkeypatch):
    monkeypatch.setattr("arc_llama.launcher.oneapi_runtime_env_needed", lambda: False)
    cfg = Config()
    cfg.tune.auto = False
    cfg.paths.state_dir = str(tmp_path / "state")
    cfg.paths.llama_server = executable(tmp_path / "global-server", "usage")
    cfg.gpus = [GPUConfig(pci_slot="test-gpu", sycl_index=0, arch="battlemage")]
    cfg.models = [ModelConfig(name=n, path=str(tmp_path / (n + ".gguf")), port=18100+i,
                              gpu_pci_slot="test-gpu", recipe={"flash_attn": "on"})
                  for i, n in enumerate(("ornith", "other"))]
    return cfg


def test_model_runtime_launch_isolation_and_literal_arguments(runtime_cfg, tmp_path):
    cfg = runtime_cfg
    model, other = cfg.models
    model.llama_server = executable(tmp_path / "patched server")
    model.env = {"ARC_TEST_ONLY": "literal $(touch BAD); value=1", "SYCL_CACHE_PERSISTENT": "1"}
    before = dict(os.environ)
    patched = build_plan(cfg, model, cfg.gpus[0])
    normal = build_plan(cfg, other, cfg.gpus[0])
    assert patched.argv[0] == model.llama_server
    assert normal.argv[0] == cfg.paths.llama_server
    assert "-fa" in patched.argv and "-fa" not in normal.argv
    assert patched.env["SYCL_CACHE_PERSISTENT"] == "1"
    assert normal.env["SYCL_CACHE_PERSISTENT"] == "0"
    assert "ARC_TEST_ONLY" not in normal.env
    assert dict(os.environ) == before
    result = subprocess.run(patched.argv, env=patched.env, capture_output=True, text=True, check=True)
    actual = json.loads(result.stdout)
    assert actual["env"] == model.env["ARC_TEST_ONLY"]
    assert actual["argv"] == patched.argv[1:]
    assert not (tmp_path / "BAD").exists()


def test_config_roundtrip_old_defaults_and_model_path(runtime_cfg, tmp_path):
    cfg = runtime_cfg
    model = cfg.models[0]
    model.llama_server = "private-server"
    model.env = {"PATH": str(tmp_path), "LD_LIBRARY_PATH": "/private/lib", "EMPTY": ""}
    binary = executable(tmp_path / "private-server")
    path = cfg.save(tmp_path / "config.toml")
    loaded = load_config(path)
    assert loaded.model_llama_server(loaded.models[0]) == binary
    assert loaded.models[0].env == model.env
    assert loaded.models[1].env == {} and loaded.models[1].llama_server == ""
    assert loaded.paths.llama_server == cfg.paths.llama_server
    # Legacy model records need no migration or overrides.
    legacy = ModelConfig("legacy", "/model.gguf", 18080, "gpu")
    assert legacy.llama_server == "" and legacy.env == {}


@pytest.mark.parametrize("binary,env", [(None, {}), ([], {}), ("x\0", {}), ("", []),
    ("", {"": "x"}), ("", {"A=B": "x"}), ("", {"A": 1}), ("", {"A": "x\0"})])
def test_invalid_runtime_config_is_rejected(binary, env):
    with pytest.raises(ValueError):
        ModelConfig("x", "/model.gguf", 18080, "gpu", llama_server=binary, env=env)


def test_probe_environment_and_cache_are_model_specific(tmp_path):
    binary = tmp_path / "env-server"
    binary.write_text(f"#!{sys.executable}\nimport os\nprint(os.environ.get('HELP', 'usage'))\n")
    binary.chmod(0o755)
    first = probe_server_caps(str(binary), env={"HELP": "usage"})
    second = probe_server_caps(str(binary), env={"HELP": "--flash-attn on off auto"})
    assert first.probed and not first.supports_flash_attn
    assert second.probed and second.supports_flash_attn


def test_fingerprint_tracks_model_runtime_and_env(runtime_cfg, tmp_path):
    cfg = runtime_cfg
    model, other = cfg.models
    def fp(m):
        return compute_fingerprint(m, cfg.model_llama_server(m), cfg.gpus[0], "test", stack_key="test")
    original, independent = fp(model), fp(other)
    model.llama_server = executable(tmp_path / "patched")
    changed = fp(model)
    assert changed != original and fp(other) == independent
    model.env["EXTRA"] = "1"
    assert fp(model) != changed and fp(other) == independent


async def test_rebuild_replaces_only_selected_model(runtime_cfg, monkeypatch, tmp_path):
    monkeypatch.setattr("arc_llama.router.LlamaServer", FakeServer)
    cfg = runtime_cfg
    router = Router(cfg)
    first, other = router._servers["ornith"], router._servers["other"]
    first.start()
    other.start()
    cfg.models[0].llama_server = executable(tmp_path / "patched")
    cfg.models[0].env = {"ARC_TEST_ONLY": "ornith"}
    assert await router.rebuild_model("ornith") == (True, True)
    assert not first.is_running
    assert router._servers["other"] is other and other.is_running
    assert router._servers["ornith"].plan.argv[0] == cfg.models[0].llama_server
    assert router._servers["ornith"].plan.env["ARC_TEST_ONLY"] == "ornith"
    await router.shutdown()


def test_admin_edit_is_atomic_and_does_not_change_other_models(runtime_cfg, monkeypatch, tmp_path):
    cfg = runtime_cfg
    cfg.server.admin_token = "test-token"
    path = tmp_path / "config.toml"
    app = create_app(cfg, config_path=path, plugins=[])
    router = app.state.router if hasattr(app.state, "router") else None
    # Lifespan constructs the router, but no model is started by this test.
    with TestClient(app) as client:
        router = app.state.router
        original_other = router._servers["other"]
        binary = executable(tmp_path / "patched")
        body = {"llama_server": binary, "env": {"ARC_TEST_ONLY": "private"}}
        assert client.post("/admin/models/ornith/edit", json=body).status_code == 401
        auth = {"Authorization": "Bearer test-token"}
        result = client.post("/admin/models/ornith/edit", json=body, headers=auth)
        assert result.status_code == 200
        assert router._servers["other"] is original_other
        assert load_config(path).models[0].env == body["env"]
        assert cfg.paths.llama_server != binary and cfg.models[1].llama_server == ""
        assert client.post("/admin/models/ornith/edit", json={"ctx": 4096}, headers=auth).status_code == 200
        assert cfg.models[0].env == body["env"] and cfg.models[0].llama_server == binary
        before = router._servers["ornith"]
        monkeypatch.setattr(cfg, "save", lambda *_a: (_ for _ in ()).throw(OSError("synthetic write failure")))
        assert client.post("/admin/models/ornith/edit", json={"env": {"BAD": "lost"}}, headers=auth).status_code == 500
        assert cfg.models[0].env == body["env"]
        assert router._servers["ornith"] is before
        assert client.post("/admin/models/ornith/edit", json={"env": {"BAD": 2}}, headers=auth).status_code == 400


async def test_local_compatibility_uses_model_build_and_env(runtime_cfg, monkeypatch, tmp_path):
    cfg = runtime_cfg
    model = cfg.models[0]
    Path(model.path).write_bytes(b"synthetic header")
    model.llama_server = executable(tmp_path / "patched")
    model.env = {"ARC_TEST_ONLY": "compatibility"}
    monkeypatch.setattr(compatibility, "header_architecture", lambda _data: "testarch")
    probe = AsyncMock(return_value="recognized")
    monkeypatch.setattr(compatibility, "probe_architecture", probe)
    answer = await compatibility.assess_compatibility(cfg, {"name": model.name})
    assert answer["status"] == "recognized"
    assert probe.call_args.args[0] == model.llama_server
    assert probe.call_args.kwargs["env"]["ARC_TEST_ONLY"] == "compatibility"
    identity = compatibility.model_runtime_identity(cfg, model)
    model.env["ARC_TEST_ONLY"] = "different"
    assert compatibility.model_runtime_identity(cfg, model) != identity


def test_cli_changes_only_one_model_and_never_prints_env_values(runtime_cfg, tmp_path):
    cfg = runtime_cfg
    path = cfg.save(tmp_path / "config.toml")
    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(path), "model-runtime", "ornith", "--llama-server", "/patched/server", "--env", "SECRET=do-not-print"])
    assert result.exit_code == 0, result.output
    assert "do-not-print" not in result.output
    saved = load_config(path)
    assert saved.models[0].llama_server == "/patched/server"
    assert saved.models[0].env == {"SECRET": "do-not-print"}
    assert saved.models[1].env == {} and saved.paths.llama_server == cfg.paths.llama_server
    reset = runner.invoke(cli, ["--config", str(path), "model-runtime", "ornith", "--reset"])
    assert reset.exit_code == 0
    assert load_config(path).models[0].llama_server == "" and load_config(path).models[0].env == {}


def test_global_runtime_update_ignores_pinned_model_requirements(runtime_cfg):
    cfg = runtime_cfg
    cfg.models[0].llama_server = "/patched/server"
    cfg.models[0].recipe["mmproj"] = "/projector.gguf"
    Path(cfg.models[0].path).touch()
    Path(cfg.models[1].path).touch()
    assert "supports_mmproj" not in required_capabilities(cfg)
    assert pick_canary_model(cfg) is cfg.models[1]


async def test_canary_tests_candidate_even_when_model_is_pinned(runtime_cfg, monkeypatch, tmp_path):
    cfg = runtime_cfg
    model = cfg.models[0]
    model.llama_server = "/old/patched"
    candidate = Path(executable(tmp_path / "candidate"))
    captured = []

    class RefuseStart:
        def __init__(self, plan, name):
            captured.append(plan)
        def start(self, *args, **kwargs):
            raise OSError("synthetic refusal")
        async def astop(self):
            pass

    monkeypatch.setattr("arc_llama.launcher.LlamaServer", RefuseStart)
    await run_canary(cfg, candidate, model)
    assert captured[0].argv[0] == str(candidate)
    assert model.llama_server == "/old/patched"
