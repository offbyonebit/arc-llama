from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from arc_llama.browser_launch import wait_and_open
from arc_llama.cli import cli
from arc_llama.config import Config, GPUConfig, ModelConfig, load_config


@pytest.mark.parametrize(
    "existing,open_browser", [(False, True), (True, True), (False, False)]
)
def test_start_opens_local_ui_without_requiring_model_or_runtime(
    tmp_path, monkeypatch, existing, open_browser
):
    cfg = Config()
    cfg.paths.models_dir = str(tmp_path / "models")
    cfg.paths.state_dir = str(tmp_path / "state")
    cfg.server.host = "0.0.0.0"
    cfg.gpus = [GPUConfig("gpu", 0, "battlemage", vram_mb=12288)]
    if existing:
        cfg.models = [ModelConfig("mine", "/synthetic/model.gguf", 1, "gpu")]
    path = tmp_path / "config.toml"
    cfg.save(path)
    app = SimpleNamespace(state=SimpleNamespace(router=None))
    browser = Mock()
    run = Mock()
    monkeypatch.setattr("arc_llama.server.create_app", lambda *a, **kw: app)
    monkeypatch.setattr("arc_llama.browser_launch.serve_in_browser", browser)
    monkeypatch.setattr("uvicorn.run", run)
    with patch("signal.signal"), patch("atexit.register"):
        args = ["--config", str(path), "start", "--port", "11438"]
        if not open_browser:
            args += ["--no-open"]
        result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    if open_browser:
        browser.assert_called_once_with(
            app, host="127.0.0.1", port=11438, path="/chat" if existing else "/"
        )
        run.assert_not_called()
    else:
        browser.assert_not_called()
        run.assert_called_once()
    # Binding changes apply to this launch; user's saved LAN configuration stays.
    saved = load_config(path)
    assert saved.server.host == "0.0.0.0" and len(saved.models) == int(existing)


@pytest.mark.parametrize("detection", [[], OSError("synthetic probe failure")])
def test_missing_hardware_still_opens_diagnostics_ui(tmp_path, monkeypatch, detection):
    cfg = Config()
    cfg.paths.models_dir = str(tmp_path / "models")
    path = tmp_path / "cfg.toml"
    monkeypatch.setattr("arc_llama.cli.load_config", lambda p: cfg)
    probe = Mock(
        side_effect=detection if isinstance(detection, Exception) else None,
        return_value=detection,
    )
    monkeypatch.setattr("arc_llama.cli.detect_gpus", probe)
    with patch("arc_llama.cli.serve.callback") as serve:
        result = CliRunner().invoke(cli, ["--config", str(path), "start"])
    assert result.exit_code == 0, result.output
    assert path.exists() and not cfg.gpus
    serve.assert_called_once()


def test_browser_waits_for_own_listening_server(monkeypatch):
    opener = Mock(return_value=True)
    monkeypatch.setattr("arc_llama.browser_launch.webbrowser.open", opener)
    server = SimpleNamespace(started=False, should_exit=False)
    stopped = threading.Event()
    wait_and_open(server, "http://127.0.0.1:11437/", stopped, timeout=0)
    opener.assert_not_called()
    server.started = True
    wait_and_open(server, "http://127.0.0.1:11437/", stopped)
    opener.assert_called_once()
    opener.reset_mock()
    stopped.set()
    wait_and_open(server, "http://127.0.0.1:11437/", stopped)
    opener.assert_not_called()


def test_failed_server_start_does_not_open_another_service(monkeypatch):
    opener = Mock()
    monkeypatch.setattr("arc_llama.browser_launch.webbrowser.open", opener)
    wait_and_open(
        SimpleNamespace(started=False, should_exit=True),
        "http://127.0.0.1:11437/",
        threading.Event(),
    )
    opener.assert_not_called()
