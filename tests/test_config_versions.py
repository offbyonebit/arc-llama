"""Upgrade and rollback of config files between released versions (issue #78).

The fixtures in ``fixtures/config-versions`` were written by the published 0.8.0, 0.9.0, 0.9.1.2 and 1.0.0
packages themselves (every field set to a non-default value), not hand-written. ``loaded-by-*.json``
is what each release read back from its own file; ``rollback-resaved-by-*`` is what each release
saved after loading a config from this version. ``scripts/check_config_versions.py`` regenerates
them and repeats the round trips against the real old packages.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import tomli_w

from arc_llama.config import load_config

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

FIXTURES = Path(__file__).parent / "fixtures" / "config-versions"
RELEASES = ["0.8.0", "0.9.0", "0.9.1.2", "1.0.0"]
CURRENT_FIXTURE = "written-by-0.9.2.dev0.toml"

# What a rollback to each release costs once that release saves the file: settings it does not
# know are dropped and come back as defaults on upgrade. Keep this list exact so a new field that
# old releases would drop is a deliberate, documented decision (docs/upgrading.md).
LOST_ON_ROLLBACK = {
    "0.9.1.2": {
        "models[0].env.PROBE_KEY",
        "models[0].llama_server",
        "server.inference_first_response_seconds",
        "server.inference_stream_idle_seconds",
    },
}
LOST_ON_ROLLBACK["0.9.0"] = LOST_ON_ROLLBACK["0.9.1.2"]
LOST_ON_ROLLBACK["0.8.0"] = LOST_ON_ROLLBACK["0.9.1.2"] | {"server.switch_drain_seconds"}
# 1.0.0 keeps settings it does not know, so rolling back from 1.1 (agent.opencode_bin and
# agent.hermes_bin included) costs nothing.
LOST_ON_ROLLBACK["1.0.0"] = set()
# The config each release's rollback fixture was resaved from (the checkout of the day).
ROLLBACK_SOURCE = {"1.0.0": "written-by-1.1.0.dev0.toml"}


def _flat(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out.update(_flat(v, f"{prefix}.{k}" if prefix else k))
        return out
    if isinstance(value, list) and value and isinstance(value[0], dict):
        out = {}
        for i, v in enumerate(value):
            out.update(_flat(v, f"{prefix}[{i}]"))
        return out
    return {prefix: value}


def _load_flat(path: Path) -> dict[str, Any]:
    cfg = load_config(path)
    data = json.loads(json.dumps(asdict(cfg), default=str))
    data.pop("unknown_keys", None)
    return _flat(data)


@pytest.fixture(autouse=True)
def _no_env_token(monkeypatch):
    monkeypatch.delenv("ARC_LLAMA_ADMIN_TOKEN", raising=False)


def _copy(name: str, tmp_path: Path) -> Path:
    target = tmp_path / "config.toml"
    shutil.copyfile(FIXTURES / name, target)
    return target


@pytest.mark.parametrize("release", RELEASES)
def test_upgrade_keeps_every_value_the_release_wrote(release, tmp_path, caplog):
    expected = _flat(json.loads((FIXTURES / f"loaded-by-{release}.json").read_text()))
    with caplog.at_level(logging.WARNING, logger="arc_llama.config"):
        loaded = _load_flat(_copy(f"written-by-{release}.toml", tmp_path))
    assert not caplog.records, caplog.text
    lost = {k: (v, loaded.get(k, "<missing>")) for k, v in expected.items() if loaded.get(k, "<missing>") != v}
    assert lost == {}


@pytest.mark.parametrize("release", RELEASES)
def test_upgraded_config_survives_saving_with_this_version(release, tmp_path):
    path = _copy(f"written-by-{release}.toml", tmp_path)
    first = _load_flat(path)
    load_config(path).save(path)
    assert _load_flat(path) == first


@pytest.mark.parametrize("release", RELEASES)
def test_rollback_loses_only_the_documented_settings(release, tmp_path):
    original = _load_flat(_copy(ROLLBACK_SOURCE.get(release, CURRENT_FIXTURE), tmp_path))
    after = _load_flat(_copy(f"rollback-resaved-by-{release}.toml", tmp_path))
    changed = {k for k in original if after.get(k) != original[k]}
    assert changed == LOST_ON_ROLLBACK[release]


def test_unknown_keys_from_a_newer_version_survive_an_edit_and_save(tmp_path, caplog):
    raw = tomllib.loads((FIXTURES / CURRENT_FIXTURE).read_text())
    raw["future_section"] = {"enabled": True}
    raw["server"]["future_option"] = 5
    raw["gpus"][0]["future_gpu_option"] = "x"
    raw["models"][0]["future_model_option"] = {"nested": [1, 2]}
    path = tmp_path / "config.toml"
    path.write_text(tomli_w.dumps(raw))

    with caplog.at_level(logging.WARNING, logger="arc_llama.config"):
        cfg = load_config(path)
    assert "future_option" in caplog.text and "keeps it when saving" in caplog.text
    cfg.server.port = 4321
    cfg.save(path)

    saved = tomllib.loads(path.read_text())
    assert saved["future_section"] == {"enabled": True}
    assert saved["server"]["future_option"] == 5
    assert saved["server"]["port"] == 4321
    assert saved["gpus"][0]["future_gpu_option"] == "x"
    assert saved["models"][0]["future_model_option"] == {"nested": [1, 2]}


def test_preserved_keys_never_override_current_values_and_follow_their_entry(tmp_path):
    raw = tomllib.loads((FIXTURES / CURRENT_FIXTURE).read_text())
    raw["models"][0]["future_model_option"] = 1
    path = tmp_path / "config.toml"
    path.write_text(tomli_w.dumps(raw))
    cfg = load_config(path)
    cfg.unknown_keys.setdefault("server", {})["port"] = 1  # a key this version knows always wins
    cfg.models[0].name = "renamed"
    cfg.save(path)

    saved = tomllib.loads(path.read_text())
    assert saved["server"]["port"] == raw["server"]["port"]
    assert "future_model_option" not in saved["models"][0]
