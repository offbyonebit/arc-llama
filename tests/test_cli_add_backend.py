"""Model-only runtime selection uses real discovery of controlled local folders."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from arc_llama.arch import Backend
from arc_llama.cli import cli
from arc_llama.config import Config, GPUConfig, ModelConfig, PathsConfig, ServerConfig, load_config
from arc_llama.launcher import model_backend
from arc_llama.recipes import LaunchRecipe
from arc_llama.runtime_pin import LLAMA_CPP_BUILD


def _binary(folder: Path, backend: str) -> Path:
    folder.mkdir(parents=True)
    binary = folder / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
    binary.write_bytes(f"ggml_backend_{backend}_reg".encode())
    return binary


@pytest.fixture
def add_setup(tmp_path, monkeypatch):
    binary = _binary(tmp_path / "Global Runtime", "vulkan")
    gpu = GPUConfig("gpu", 0, "battlemage", vram_mb=24576, backend="vulkan")
    old = ModelConfig("old", str(tmp_path / "old.gguf"), 18080, gpu.pci_slot, recipe={"ctx": 8192})
    cfg = Config(server=ServerConfig(admin_token="fixture-admin-token"),
                 paths=PathsConfig(state_dir=str(tmp_path / "State Folder"), llama_server=str(binary)),
                 gpus=[gpu], models=[old])
    path = tmp_path / "config.toml"
    cfg.save(path)
    model_file = tmp_path / "New Model.gguf"
    model_file.write_bytes(b"controlled model metadata fixture")
    monkeypatch.setattr("arc_llama.models.default_recipe", lambda **_: LaunchRecipe(flash_attn="off"))
    return SimpleNamespace(cfg=cfg, path=path, model_file=model_file, old=old)


def _install(setup, tag: str, backend="sycl") -> Path:
    folder = Path(setup.cfg.paths.state_dir) / "runtime" / f"llama-{tag}-{backend}"
    binary = _binary(folder / "bin", backend)
    (folder / ".arc-llama-runtime.json").write_text(json.dumps({"tag": tag, "backend": backend}), encoding="utf-8")
    return binary


def _add(setup, backend):
    return CliRunner().invoke(cli, ["--config", str(setup.path), "add", str(setup.model_file),
                                  "--name", "new", "--backend", backend])


def test_add_backend_uses_model_runtime_and_preserves_gpu_and_other_models(add_setup):
    s = add_setup
    binary = _install(s, LLAMA_CPP_BUILD)
    result = _add(s, "sycl")
    assert result.exit_code == 0, result.output
    saved = load_config(s.path)
    model = saved.find_model("new")
    assert saved.gpus[0].backend == "vulkan"
    assert saved.paths.llama_server == s.cfg.paths.llama_server
    assert model.llama_server == str(binary)
    assert model_backend(saved, model, saved.gpus[0]) == Backend.SYCL
    assert saved.find_model("old").recipe == s.old.recipe
    assert saved.find_model("old").llama_server == s.old.llama_server
    assert model.recipe["flash_attn"] == "on"
    assert "Flash attention set to on for this model's SYCL runtime." in result.output
    assert "Warning:" not in result.output


@pytest.mark.parametrize("tags,expected", [([LLAMA_CPP_BUILD, "b99999"], LLAMA_CPP_BUILD), (["b9", "b10"], "b10")])
def test_add_backend_prefers_pin_else_newest_numeric_tag(add_setup, tags, expected):
    binaries = {tag: _install(add_setup, tag) for tag in tags}
    result = _add(add_setup, "sycl")
    assert result.exit_code == 0, result.output
    assert load_config(add_setup.path).find_model("new").llama_server == str(binaries[expected])


def test_add_missing_backend_exits_before_registration_or_config_changes(add_setup, monkeypatch):
    before = add_setup.path.read_bytes()
    monkeypatch.setattr("arc_llama.cli.add_local_model", lambda *a, **kw: pytest.fail("registered before finding a runtime"))
    result = _add(add_setup, "sycl")
    assert result.exit_code == 1
    assert "No sycl runtime installed. Run: arc-llama install-runtime --backend sycl --no-set-default" in " ".join(result.output.split())
    assert add_setup.path.read_bytes() == before
    assert len(load_config(add_setup.path).models) == 1


def test_add_matching_backend_keeps_inherited_runtime(add_setup):
    result = _add(add_setup, "vulkan")
    assert result.exit_code == 0, result.output
    saved = load_config(add_setup.path)
    assert not saved.find_model("new").llama_server
    assert saved.gpus[0].backend == "vulkan"
    assert saved.find_model("new").recipe["flash_attn"] == "off"


def test_add_backend_help_is_model_only():
    result = CliRunner().invoke(cli, ["add", "--help"])
    assert "Runtime backend for this model only (default: the GPU's backend)." in " ".join(result.output.split())
