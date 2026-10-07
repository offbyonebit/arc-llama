from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from arc_llama.arch import Arch
from arc_llama.config import Config, ModelConfig
from arc_llama.detect import DetectedGPU
from arc_llama.support_bundle import create_support_bundle, redact_config


def test_redact_config_removes_tokens_and_home_paths(monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: Path("/users/tester")))
    text = 'admin_token = "secret"\npath = "/users/tester/models/x.gguf"\n'

    redacted = redact_config(text)

    assert "secret" not in redacted
    assert "$HOME/models/x.gguf" in redacted


@pytest.mark.parametrize("model_present", [True, False])
def test_support_bundle_contains_metadata_but_not_model_file(tmp_path, model_present):
    config_path = tmp_path / "config.toml"
    config_path.write_text('admin_token = "secret"\n', encoding="utf-8")
    model_path = tmp_path / "model.gguf"
    if model_present:
        model_path.write_bytes(b"model data")
    cfg = Config(models=[ModelConfig("model", str(model_path), 18080, "0000:03:00.0")])
    gpu = DetectedGPU(
        pci_slot="0000:03:00.0",
        device_id=0xE211,
        arch=Arch.BATTLEMAGE,
        name="Intel Arc Pro B60",
        driver="xe",
        vram_mb=24576,
        drm_card=None,
        drm_render=None,
        sysfs_path=None,
    )
    output = tmp_path / "support.zip"

    create_support_bundle(output, config_path=config_path, cfg=cfg, gpus=[gpu])

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert names == {"manifest.txt", "config.toml", "hardware.txt", "models.txt"}
        assert "secret" not in archive.read("config.toml").decode()
        assert b"model data" not in b"".join(archive.read(name) for name in names)
        assert "Intel Arc Pro B60" in archive.read("hardware.txt").decode()

        expected_size = "10" if model_present else "missing"
        assert f"size_bytes={expected_size}" in archive.read("models.txt").decode()
