"""Deployment defaults must protect active responses and keep waits bounded."""
from __future__ import annotations

import math

import pytest

from arc_llama.config import ServerConfig, load_config


def test_new_server_preserves_active_generations_by_default():
    assert ServerConfig().switch_interrupt_policy == "reject_new"


def test_existing_config_without_switch_fields_gets_safe_default(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[server]\nadmin_token = "test-token"\n')
    cfg = load_config(path)
    assert cfg.server.switch_interrupt_policy == "reject_new"
    assert math.isfinite(cfg.server.switch_drain_seconds)


@pytest.mark.parametrize("seconds", [float("inf"), float("nan"), -1, 0, True, "30"])
def test_drain_deadline_rejects_invalid_or_unbounded_values(seconds):
    with pytest.raises(ValueError, match="switch_drain_seconds"):
        ServerConfig(switch_drain_seconds=seconds)
