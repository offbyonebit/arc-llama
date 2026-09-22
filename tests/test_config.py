from __future__ import annotations

import logging
from pathlib import Path

import pytest

from arc_llama.config import (
    Config,
    GPUConfig,
    MCPServerConfig,
    ModelConfig,
    ProfileConfig,
    _resolve_admin_token,
    default_config_path,
    load_config,
    migrate_config,
)


def test_config_round_trips_models_gpus_and_paths(tmp_path):
    path = tmp_path / "config.toml"
    cfg = Config()
    cfg.paths.llama_server = "/opt/llama.cpp/llama-server"
    cfg.paths.scan_paths = [str(tmp_path / "extra-models")]
    cfg.gpus = [
        GPUConfig(
            pci_slot="0000:03:00.0",
            sycl_index=0,
            arch="battlemage",
            vram_mb=24576,
            name="Arc Pro B60",
        )
    ]
    cfg.models = [
        ModelConfig(
            name="qwen",
            path=str(tmp_path / "Qwen3-7B-Q4_K_M.gguf"),
            port=18080,
            gpu_pci_slot="0000:03:00.0",
            display_name="Qwen 3 7B",
            aliases=["Qwen3-7B-Q4_K_M.gguf"],
            recipe={"ctx": 32768, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
        )
    ]

    cfg.save(path)
    loaded = load_config(path)

    assert loaded.paths.llama_server == "/opt/llama.cpp/llama-server"
    assert loaded.paths.scan_paths == [str(tmp_path / "extra-models")]
    assert loaded.gpus[0].name == "Arc Pro B60"
    assert loaded.models[0].recipe["ctx"] == 32768


def test_find_model_matches_name_alias_display_name_and_filename(tmp_path):
    cfg = Config(
        models=[
            ModelConfig(
                name="qwen",
                path=str(tmp_path / "Qwen3-7B-Q4_K_M.gguf"),
                port=18080,
                gpu_pci_slot="0000:03:00.0",
                display_name="Qwen 3 7B",
                aliases=["chat-default"],
            )
        ]
    )

    assert cfg.find_model("qwen").name == "qwen"
    assert cfg.find_model("chat-default").name == "qwen"
    assert cfg.find_model("qwen 3").name == "qwen"
    assert cfg.find_model("Q4_K_M").name == "qwen"
    assert cfg.find_model("missing") is None


def test_windows_default_paths_use_appdata(monkeypatch):
    import os

    from arc_llama import config as config_mod

    monkeypatch.setattr(config_mod.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", r"C:\Users\test\AppData\Roaming")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\test\AppData\Local")
    appdata = Path(os.environ["APPDATA"])
    localappdata = Path(os.environ["LOCALAPPDATA"])
    assert config_mod.default_config_path() == appdata / "arc-llama" / "config.toml"
    assert config_mod.default_models_dir() == localappdata / "arc-llama" / "models"
    assert config_mod.default_state_dir() == localappdata / "arc-llama"


def test_migrate_config_adds_missing_sections():
    from arc_llama.config import CONFIG_VERSION, migrate_config

    raw = migrate_config({})
    assert raw["version"] == CONFIG_VERSION
    # Newer server sections carry the model-switch drain and interrupt
    # policy with explicit defaults.
    assert raw["server"] == {
        "admin_token": None,
        "switch_drain_seconds": 30.0,
        "switch_interrupt_policy": "reject_new",
    }
    assert raw["paths"] == {}
    assert raw["agent"] == {"root": ".", "profile": None}
    assert raw["gpus"] == []
    assert raw["models"] == []
    assert raw["upstreams"] == []
    assert raw["profiles"] == []


def test_validate_config_rejects_bad_structure():
    from arc_llama.config import validate_config

    with pytest.raises(ValueError, match="version"):
        validate_config({"version": "not-an-int"})
    with pytest.raises(ValueError, match="gpus"):
        validate_config({"version": 1, "gpus": {}})
    with pytest.raises(ValueError, match="profiles"):
        validate_config({"version": 1, "profiles": {}})


def test_load_config_generates_and_persists_admin_token(tmp_path):
    path = tmp_path / "config.toml"
    Config().save(path)

    loaded = load_config(path)
    assert loaded.server.admin_token

    reloaded = load_config(path)
    assert reloaded.server.admin_token == loaded.server.admin_token


def test_load_config_generates_admin_token_when_no_file_exists(tmp_path):
    path = tmp_path / "does-not-exist.toml"

    cfg = load_config(path)
    assert cfg.server.admin_token
    assert not path.exists()


def test_load_config_env_var_overrides_and_is_not_persisted(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    Config().save(path)
    monkeypatch.setenv("ARC_LLAMA_ADMIN_TOKEN", "env-token")

    cfg = load_config(path)
    assert cfg.server.admin_token == "env-token"

    monkeypatch.delenv("ARC_LLAMA_ADMIN_TOKEN")
    reloaded = load_config(path)
    assert reloaded.server.admin_token != "env-token"


def test_load_config_keeps_existing_admin_token(tmp_path):
    path = tmp_path / "config.toml"
    cfg = Config()
    cfg.server.admin_token = "existing-token"
    cfg.save(path)

    loaded = load_config(path)
    assert loaded.server.admin_token == "existing-token"


def test_profiles_round_trip(tmp_path):
    path = tmp_path / "config.toml"
    cfg = Config()
    cfg.agent.profile = "work"
    cfg.mcp_servers = [
        MCPServerConfig(name="fs", command="npx", args=["-y", "@mcp/fs"]),
        MCPServerConfig(name="gh", command="npx", args=["-y", "@mcp/gh"]),
    ]
    cfg.profiles = [
        ProfileConfig(name="work", mcp_servers=["fs"]),
        ProfileConfig(name="oss", mcp_servers=["fs", "gh"]),
    ]
    cfg.save(path)
    loaded = load_config(path)

    assert loaded.agent.profile == "work"
    assert [p.name for p in loaded.profiles] == ["work", "oss"]
    assert loaded.profiles[0].mcp_servers == ["fs"]
    assert loaded.profiles[1].mcp_servers == ["fs", "gh"]


def test_active_mcp_servers_filters_by_profile():
    cfg = Config()
    cfg.mcp_servers = [
        MCPServerConfig(name="fs", command="npx"),
        MCPServerConfig(name="gh", command="npx"),
    ]
    cfg.profiles = [ProfileConfig(name="work", mcp_servers=["fs"])]

    assert [s.name for s in cfg.active_mcp_servers()] == ["fs", "gh"]
    assert [s.name for s in cfg.active_mcp_servers("work")] == ["fs"]
    assert cfg.active_mcp_servers("missing") == cfg.mcp_servers


def test_migrate_config_adds_profiles_and_agent_profile():
    raw = migrate_config({})
    assert raw["profiles"] == []
    assert "profile" in raw["agent"]


def test_default_config_path_is_isolated(tmp_path):
    """The autouse fixture redirects XDG dirs under tmp_path."""
    assert default_config_path().is_relative_to(tmp_path)


def test_resolve_admin_token_warning_omits_secret(caplog, tmp_path):
    cfg = Config()
    path = tmp_path / "config.toml"
    with caplog.at_level(logging.WARNING, logger="arc_llama.config"):
        _resolve_admin_token(cfg, path, persist=False)
    assert cfg.server.admin_token is not None
    assert len(cfg.server.admin_token) > 16
    assert cfg.server.admin_token not in caplog.text
    assert "generated" in caplog.text
    assert str(path) in caplog.text


def test_load_config_survives_read_only_config(tmp_path, monkeypatch, caplog):
    """A generated admin_token must not crash load_config on a read-only file."""
    path = tmp_path / "config.toml"
    Config().save(path)

    def _raise(_self, _p=None):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(Config, "save", _raise)

    with caplog.at_level(logging.WARNING, logger="arc_llama.config"):
        cfg = load_config(path)

    assert cfg.server.admin_token
    assert "in-memory only" in caplog.text
    assert "Permission denied" in caplog.text


def test_load_config_ignores_unknown_model_keys(tmp_path, caplog):
    path = tmp_path / "config.toml"
    path.write_text(
        """
version = 1
[server]
host = "127.0.0.1"
port = 11437
admin_token = "secret"
[[models]]
name = "qwen"
path = "/models/qwen.gguf"
port = 18080
gpu_pci_slot = "0000:03:00.0"
field_from_a_newer_arc_llama = "whatever"
"""
    )
    with caplog.at_level(logging.WARNING, logger="arc_llama.config"):
        cfg = load_config(path)
    assert cfg.models[0].name == "qwen"
    # Deliberately not a real field: this test must keep exercising the
    # unknown-key path even as new fields are added to ModelConfig.
    assert not hasattr(cfg.models[0], "field_from_a_newer_arc_llama")
    assert "field_from_a_newer_arc_llama" in caplog.text


def test_load_config_ignores_unknown_top_level_keys(tmp_path, caplog):
    path = tmp_path / "config.toml"
    path.write_text(
        """
version = 1
unknown_section = { foo = 1 }
[server]
host = "127.0.0.1"
"""
    )
    with caplog.at_level(logging.WARNING, logger="arc_llama.config"):
        cfg = load_config(path)
    assert cfg.server.host == "127.0.0.1"
    assert "unknown_section" in caplog.text
