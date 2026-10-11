from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from arc_llama.cli import cli
from arc_llama.config import Config, GPUConfig, ModelConfig, load_config
from arc_llama.runtime_update import (
    CanaryResult,
    capability_problems,
    pick_canary_model,
    read_history,
    required_capabilities,
    rollback_runtime,
    run_canary,
    update_runtime,
)
from arc_llama.server_caps import ServerCaps

FULL_CAPS = ServerCaps(
    probed=True,
    supports_speculative=True,
    supports_draft_model=True,
    supports_ngram=True,
    supports_load_mode=True,
    supports_mmproj=True,
    supports_reranking=True,
    supports_tensor_split=True,
)


def _cfg(tmp_path: Path, **recipe) -> Config:
    model_path = tmp_path / "small.gguf"
    model_path.write_bytes(b"GGUF")
    old = tmp_path / "old" / "llama-server"
    old.parent.mkdir()
    old.write_bytes(b"old")
    cfg = Config(
        gpus=[GPUConfig("0000:03:00.0", 0, "battlemage", vram_mb=24576, backend="sycl")],
        models=[ModelConfig("small", str(model_path), 18080, "0000:03:00.0", recipe=dict(recipe))],
    )
    cfg.paths.state_dir = str(tmp_path / "state")
    cfg.paths.llama_server = str(old)
    return cfg


def _installer(binary: Path, tag: str = "b9999"):
    calls = []

    def install(**kwargs):
        calls.append(kwargs)
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(b"new")
        return SimpleNamespace(binary_path=binary, tag=tag)

    install.calls = calls  # type: ignore[attr-defined]
    return install


def test_required_capabilities_and_problems(tmp_path) -> None:
    cfg = _cfg(tmp_path, mmproj="/p.gguf", spec_type="draft-mtp", tensor_split=[1, 1])
    needs = required_capabilities(cfg)
    assert set(needs) == {"supports_mmproj", "supports_speculative", "supports_tensor_split"}
    assert capability_problems(FULL_CAPS, cfg) == []
    problems = capability_problems(ServerCaps(probed=True), cfg)
    assert any("--mmproj" in p and "small" in p for p in problems)
    assert capability_problems(ServerCaps(probed=False), cfg)[0].startswith("the candidate")


def test_update_switches_after_passing_canary(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    config_path = tmp_path / "config.toml"
    new = tmp_path / "new" / "llama-server"
    install = _installer(new)
    seen = []

    def canary(c, binary, model):
        seen.append((binary, model.name))
        return CanaryResult(True, "answered 'OK'", model.name, 3.0)

    result = update_runtime(
        cfg, config_path, installer=install, canary_runner=canary, caps_probe=lambda _: FULL_CAPS
    )
    assert result.switched
    assert install.calls[0]["set_default"] is False  # type: ignore[attr-defined]
    assert install.calls[0]["backend"] == "sycl"  # type: ignore[attr-defined]
    assert seen == [(new, "small")]
    assert load_config(config_path).paths.llama_server == str(new)
    [entry] = read_history(cfg)
    assert entry["installed"] == str(new)
    assert entry["previous_backend"] == "sycl"


def test_failed_canary_keeps_current_runtime(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    old = cfg.paths.llama_server
    config_path = tmp_path / "config.toml"
    result = update_runtime(
        cfg,
        config_path,
        installer=_installer(tmp_path / "new" / "llama-server"),
        canary_runner=lambda c, b, m: CanaryResult(False, "did not become healthy", m.name),
        caps_probe=lambda _: FULL_CAPS,
    )
    assert result.status == "rejected"
    assert "did not become healthy" in result.problems[0]
    assert cfg.paths.llama_server == old
    assert not config_path.exists()
    assert read_history(cfg) == []


def test_missing_capability_rejects_before_canary(tmp_path) -> None:
    cfg = _cfg(tmp_path, mmproj="/p.gguf")

    def canary(*_):
        raise AssertionError("canary must not run")

    result = update_runtime(
        cfg,
        tmp_path / "config.toml",
        installer=_installer(tmp_path / "new" / "llama-server"),
        canary_runner=canary,
        caps_probe=lambda _: ServerCaps(probed=True),
    )
    assert result.status == "rejected"
    assert "--mmproj" in result.problems[0]


def test_same_binary_is_up_to_date(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    result = update_runtime(
        cfg,
        tmp_path / "config.toml",
        installer=_installer(Path(cfg.paths.llama_server)),
        caps_probe=lambda _: FULL_CAPS,
    )
    assert result.status == "up_to_date"


def test_dry_run_and_no_canary(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    old = cfg.paths.llama_server
    result = update_runtime(
        cfg,
        tmp_path / "config.toml",
        installer=_installer(tmp_path / "new" / "llama-server"),
        canary=False,
        dry_run=True,
        caps_probe=lambda _: FULL_CAPS,
    )
    assert result.status == "dry_run"
    assert result.canary is None
    assert cfg.paths.llama_server == old


def test_no_model_for_canary_is_rejected(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.models = []
    result = update_runtime(
        cfg,
        tmp_path / "config.toml",
        installer=_installer(tmp_path / "new" / "llama-server"),
        caps_probe=lambda _: FULL_CAPS,
    )
    assert result.status == "rejected"
    assert "--no-canary" in result.problems[0]


def test_rollback_restores_previous_runtime_and_backend(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    old = cfg.paths.llama_server
    config_path = tmp_path / "config.toml"
    update_runtime(
        cfg,
        config_path,
        backend="vulkan",
        installer=_installer(tmp_path / "new" / "llama-server"),
        canary=False,
        caps_probe=lambda _: FULL_CAPS,
    )
    assert cfg.gpus[0].backend == "vulkan"
    restored = rollback_runtime(cfg, config_path)
    assert restored is not None and restored[1] == old
    on_disk = load_config(config_path)
    assert on_disk.paths.llama_server == old
    assert on_disk.gpus[0].backend == "sycl"
    assert rollback_runtime(cfg, config_path) is None


def test_rollback_skips_deleted_runtime(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    config_path = tmp_path / "config.toml"
    update_runtime(
        cfg,
        config_path,
        installer=_installer(tmp_path / "new" / "llama-server"),
        canary=False,
        caps_probe=lambda _: FULL_CAPS,
    )
    Path(read_history(cfg)[0]["previous"]).unlink()
    assert rollback_runtime(cfg, config_path) is None


def test_pick_canary_model_prefers_smallest_present(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    big = tmp_path / "big.gguf"
    big.write_bytes(b"GGUF" * 100)
    cfg.models.append(ModelConfig("big", str(big), 18081, "0000:03:00.0"))
    cfg.models.append(ModelConfig("gone", str(tmp_path / "gone.gguf"), 18082, "0000:03:00.0"))
    assert pick_canary_model(cfg).name == "small"  # type: ignore[union-attr]
    assert pick_canary_model(cfg, "big").name == "big"  # type: ignore[union-attr]


FAKE_SERVER = textwrap.dedent(
    """
    #!{python}
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    if "--help" in sys.argv:
        print("--flash-attn FA  ('on', 'off', or 'auto')\\n--spec-type --load-mode --mmproj")
        sys.exit(0)
    port = int(sys.argv[sys.argv.index("--port") + 1])
    reply = {reply!r}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send({{"status": "ok"}})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            self._send({{"choices": [{{"message": {{"content": reply}}}}]}})

    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    """
).lstrip()


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX shebang script as llama-server")
@pytest.mark.parametrize(("reply", "ok"), [("OK", True), ("", False)])
async def test_run_canary_against_stand_in_server(tmp_path, monkeypatch, reply, ok) -> None:
    import posix  # the real primitives; conftest has replaced os.kill/os.killpg
    import subprocess

    binary = tmp_path / "candidate" / "llama-server"
    binary.parent.mkdir()
    # conftest forbids real signals; permit them only for the stand-in this
    # test spawns, so the canary's own cleanup path is exercised for real.
    own_pids: set[int] = set()
    guarded_popen = subprocess.Popen
    real_killpg = posix.killpg
    real_kill = posix.kill

    def tracking_popen(args, *a, **kw):
        proc = guarded_popen(args, *a, **kw)
        if isinstance(args, list) and args and args[0] == str(binary):
            own_pids.add(proc.pid)
        return proc

    def scoped(real):
        def send(pid, sig):
            if pid not in own_pids:
                raise AssertionError("signal to a process this test did not start")
            return real(pid, sig)

        return send

    monkeypatch.setattr(subprocess, "Popen", tracking_popen)
    monkeypatch.setattr(os, "kill", scoped(real_kill))
    monkeypatch.setattr(os, "killpg", scoped(real_killpg))
    binary.write_text(FAKE_SERVER.format(python=sys.executable, reply=reply))
    binary.chmod(0o755)
    cfg = _cfg(tmp_path)
    cfg.gpus[0].backend = "vulkan"
    cfg.server.single_resident = False
    result = await run_canary(cfg, binary, cfg.models[0], timeout=20)
    assert result.ok is ok, result.detail
    assert result.model == "small"


def test_runtime_cli_update_and_rollback(tmp_path, monkeypatch) -> None:
    import arc_llama.runtime_update as mod

    cfg = _cfg(tmp_path)
    config_path = tmp_path / "config.toml"
    cfg.save(config_path)
    new = tmp_path / "new" / "llama-server"
    real_update = mod.update_runtime

    def fake_update(c, path, **kwargs):
        return real_update(
            c, path, installer=_installer(new), caps_probe=lambda _: FULL_CAPS, **kwargs
        )

    monkeypatch.setattr(mod, "update_runtime", fake_update)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--config", str(config_path), "runtime", "update", "--no-canary"], obj={}
    )
    assert result.exit_code == 0, result.output
    assert "Switched to b9999" in result.output
    assert load_config(config_path).paths.llama_server == str(new)
    back = runner.invoke(cli, ["--config", str(config_path), "runtime", "rollback"], obj={})
    assert back.exit_code == 0, back.output
    assert load_config(config_path).paths.llama_server == cfg.paths.llama_server
    again = runner.invoke(cli, ["--config", str(config_path), "runtime", "rollback"], obj={})
    assert again.exit_code != 0
