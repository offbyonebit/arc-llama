from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from arc_llama import __version__
from arc_llama.arch import Arch
from arc_llama.config import Config, ModelConfig
from arc_llama.detect import DetectedGPU
from arc_llama.platform_checks import DoctorReport
from arc_llama.runtime_pin import LLAMA_CPP_BUILD


@pytest.fixture
def environment(tmp_path, monkeypatch):
    import arc_llama.cli as cli_mod
    import arc_llama.cli_diagnostics as diagnostics
    import arc_llama.plugins as plugins

    home = tmp_path / "private-home"
    home.mkdir()
    path = home / "config.toml"
    path.touch()
    cfg = Config()
    cfg.server.admin_token = "bare-admin-value"
    cfg.file_admin_token = "bare-file-admin-value"
    cfg.paths.llama_server = str(home / "missing-runtime")
    cfg.models = [ModelConfig("test", str(home / "model.gguf"), 18080, "gpu",
                              env={"PROVIDER_API_KEY": "bare-model-key"})]
    monkeypatch.setenv("DOCTOR_TEST_API_KEY", "bare-environment-key")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(cli_mod, "_IS_WINDOWS", True)
    monkeypatch.setattr(cli_mod, "load_config", lambda _: cfg)
    gpu = DetectedGPU("0000:03:00.0", 0xE212, Arch.BATTLEMAGE, "Test B60", "test-driver",
                      24576, None, None, "")
    monkeypatch.setattr(cli_mod, "detect_gpus", lambda **_: [gpu])
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: str(home / name) if name == "clinfo" else None)
    monkeypatch.setattr(diagnostics, "level_zero_loader_present", lambda: (True, str(home / "loader")))
    monkeypatch.setattr(diagnostics, "oneapi_setvars_path", lambda: None)
    secrets = [cfg.server.admin_token, cfg.file_admin_token, "bare-model-key", "bare-environment-key", "abc.def-123", "query-value"]
    def health():
        # Third-party output must not pollute the JSON document either.
        print("probe output " + cfg.server.admin_token)
        return [{"name": "healthy", "status": "loaded", "version": "1"},
                {"name": "broken", "status": "error", "error":
                 f"{home}/plugin {' '.join(secrets[:4])} Authorization: Bearer abc.def-123 "
                 "https://example.test/?api_key=query-value"}]
    monkeypatch.setattr(plugins, "plugin_health", health)
    return SimpleNamespace(cli=cli_mod.cli, path=path, cfg=cfg, home=home, secrets=secrets, gpu=gpu)


def test_json_is_one_redacted_document_with_shared_check_results(environment, monkeypatch):
    recorded = []
    original = DoctorReport.add
    def record(self, name, ok, detail, **kwargs):
        recorded.append(name)
        return original(self, name, ok, detail, **kwargs)
    monkeypatch.setattr(DoctorReport, "add", record)
    runner = CliRunner()
    args = ["-c", str(environment.path), "doctor"]
    human = runner.invoke(environment.cli, args)
    human_checks = list(recorded)
    recorded.clear()
    result = runner.invoke(environment.cli, [*args, "--json"])
    assert result.exit_code == human.exit_code == 2
    data = json.loads(result.output)
    assert data["schema_version"] == 1
    assert data["arc_llama_version"] == __version__
    assert data["pinned_runtime_build"] == LLAMA_CPP_BUILD
    assert data["os"] and data["python"]
    assert data["gpus"] == [{"name": "Test B60", "pci_slot": "0000:03:00.0", "device_id": "0xe212",
                             "arch": "battlemage", "driver": "test-driver", "vram_mb": 24576}]
    assert [check["name"] for check in data["checks"]] == recorded == human_checks
    checks = {check["name"]: check for check in data["checks"]}
    assert checks["gpus"]["status"] == checks["config"]["status"] == "ok"
    assert checks["plugin_healthy"]["status"] == "ok"
    assert checks["plugin_broken"]["status"] == checks["oneapi_setvars"]["status"] == "warn"
    assert checks["llama_server"]["status"] == "fail"
    assert checks["gpu_driver"]["status"] == checks["tool_lspci"]["status"] == "skipped"
    assert checks["llama_server"]["hint"]
    for check in data["checks"]:
        assert {"name", "status", "message"} <= check.keys()
        assert len(check["message"]) <= 500
        if check["status"] in ("warn", "fail"):
            assert check["name"] in human.output
    assert "~/config.toml" in checks["config"]["message"].replace("\\", "/")
    assert str(environment.home) not in result.output
    assert "probe output" not in result.output
    assert all(secret not in result.output for secret in environment.secrets)
    assert "sysfs_path" not in data["gpus"][0]


def test_json_keeps_success_exit_code(environment, monkeypatch):
    import arc_llama.cli_diagnostics as diagnostics
    from arc_llama.arch import Backend

    environment.cfg.paths.llama_server = str(environment.path)  # existing controlled file
    monkeypatch.setattr(diagnostics, "detect_backends", lambda _: {Backend.VULKAN})
    monkeypatch.setattr(diagnostics, "detect_llama_server_backend", lambda _: Backend.VULKAN)
    args = ["-c", str(environment.path), "doctor"]
    runner = CliRunner()
    human = runner.invoke(environment.cli, args)
    result = runner.invoke(environment.cli, [*args, "--json"])
    assert result.exit_code == human.exit_code == 0
    assert all(check["status"] != "fail" for check in json.loads(result.output)["checks"])


def test_json_redacts_windows_home_spellings_and_url_credentials(monkeypatch):
    import arc_llama.cli_diagnostics as diagnostics
    import arc_llama.issue_report as issue_report
    from arc_llama.cli_diagnostics import _doctor_json

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: PureWindowsPath("C:/Users/Private")))
    monkeypatch.setattr(issue_report.sys, "platform", "win32")
    monkeypatch.setattr(diagnostics.platform, "platform", lambda: "Windows test")
    report = DoctorReport()
    report.add("paths", False,
               r"C:\Users\Private\model c:/users/private/config C:\\Users\\Private\\log "
               "https://user:password@example.test/", hint="Check the path")
    data = _doctor_json(report, [], None)
    message = data["checks"][0]["message"]
    assert "private" not in message.lower()
    assert message.count("~") == 3
    assert "user:password" not in message
    assert "[REDACTED]@example.test" in message


def test_short_environment_flags_do_not_redact_versions_or_messages(environment, monkeypatch):
    from arc_llama.cli_diagnostics import _doctor_json

    monkeypatch.setenv("FOO_TOKEN", "1")
    monkeypatch.setenv("FEATURE_KEY", "true")
    monkeypatch.setenv("SEVEN_SECRET", "abcdefg")
    environment.cfg.models[0].env["MODEL_TOKEN"] = "1"
    report = DoctorReport()
    message = "Version 1.1.1: true flag, abcdefg unchanged"
    report.add("flags", True, message)
    data = _doctor_json(report, [], environment.cfg)
    assert data["arc_llama_version"] == __version__
    assert data["checks"][0]["message"] == message


def test_eight_character_env_secret_and_short_admin_tokens_still_redact(environment, monkeypatch):
    from arc_llama.cli_diagnostics import _doctor_json

    monkeypatch.setenv("EIGHT_TOKEN", "abcdefgh")
    environment.cfg.server.admin_token = "adm"
    environment.cfg.file_admin_token = "fil"
    report = DoctorReport()
    report.add("tokens", True, "abcdefgh adm fil")
    assert _doctor_json(report, [], environment.cfg)["checks"][0]["message"] == "[REDACTED] [REDACTED] [REDACTED]"
