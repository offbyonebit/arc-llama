"""Exercise the real launcher with isolated, bounded, synthetic Linux children."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="native Linux process groups; Windows uses separate mocks")
@pytest.mark.parametrize("scenario", ["graceful", "force", "startup", "crash", "concurrent", "cancel"])
def test_native_launcher_repeated_lifecycle(tmp_path, scenario):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(root / "src"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        XDG_STATE_HOME=str(tmp_path / "state"),
        HERMES_HOME=str(tmp_path / "hermes"),
    )
    result = subprocess.run(
        [sys.executable, str(root / "tests/fixtures/lifecycle_probe.py"), str(tmp_path / "probe"), scenario],
        env=env, capture_output=True, text=True, timeout=25,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {"scenario": scenario, "iterations": 3, "port_released": True}
