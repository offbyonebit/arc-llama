from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from arc_llama.arch import Backend
from arc_llama.cli import cli
from arc_llama.config import Config, GPUConfig, PathsConfig, load_config
from arc_llama.runtime import older_runtime_notice
from arc_llama.runtime_pin import LLAMA_CPP_BUILD


def _runtime_config(tmp_path: Path) -> tuple[Path, Config]:
    state = tmp_path / "state"
    install = state / "runtime" / "llama-b12345-vulkan"
    install.mkdir(parents=True)
    binary = install / "llama-server"
    binary.write_bytes(b"runtime")
    marker = {"schema": 1, "tag": "b12345", "backend": "vulkan"}
    (install / ".arc-llama-runtime.json").write_text(json.dumps(marker), encoding="utf-8")
    config_path = tmp_path / "config.toml"
    cfg = Config(
        paths=PathsConfig(state_dir=str(state), llama_server="missing-runtime"),
        gpus=[GPUConfig("0000:03:00.0", 0, "battlemage", backend="sycl")],
    )
    cfg.save(config_path)
    return config_path, cfg


def test_runtime_list_shows_complete_install(tmp_path):
    config_path, _cfg = _runtime_config(tmp_path)

    result = CliRunner().invoke(cli, ["--config", str(config_path), "runtime", "list"])

    assert result.exit_code == 0, result.output
    assert "b12345" in result.output
    assert "vulkan" in result.output


def test_runtime_use_switches_configured_binary_and_backend(tmp_path):
    config_path, _cfg = _runtime_config(tmp_path)

    result = CliRunner().invoke(
        cli,
        ["--config", str(config_path), "runtime", "use", "b12345", "--backend", "vulkan"],
    )

    assert result.exit_code == 0, result.output
    saved = load_config(config_path)
    assert Path(saved.paths.llama_server).name == "llama-server"
    assert saved.gpus[0].backend == Backend.VULKAN.value



def test_runtime_list_marks_the_active_runtime(tmp_path):
    config_path, _cfg = _runtime_config(tmp_path)
    runner = CliRunner()
    used = runner.invoke(cli, ["--config", str(config_path), "runtime", "use", "b12345", "--backend", "vulkan"])
    assert used.exit_code == 0, used.output

    result = runner.invoke(cli, ["--config", str(config_path), "runtime", "list"])

    assert result.exit_code == 0, result.output
    assert "yes" in result.output


def test_runtime_use_picks_the_server_exe_over_its_impl_dll(tmp_path):
    # The b12243 Windows zip ships llama-server-impl.dll next to llama-server.exe, and the DLL sorts first.
    config_path, cfg = _runtime_config(tmp_path)
    install = Path(cfg.paths.state_dir) / "runtime" / "llama-b12345-vulkan"
    (install / "llama-server").unlink()
    (install / "llama-server-impl.dll").write_bytes(b"dll")
    (install / "llama-server.exe").write_bytes(b"exe")

    result = CliRunner().invoke(
        cli,
        ["--config", str(config_path), "runtime", "use", "b12345", "--backend", "vulkan"],
    )

    assert result.exit_code == 0, result.output
    assert Path(load_config(config_path).paths.llama_server).name == "llama-server.exe"


def test_runtime_list_never_takes_the_impl_dll_for_the_server(tmp_path):
    config_path, cfg = _runtime_config(tmp_path)
    install = Path(cfg.paths.state_dir) / "runtime" / "llama-b12345-vulkan"
    (install / "llama-server").unlink()
    (install / "llama-server-impl.dll").write_bytes(b"dll")

    result = CliRunner().invoke(cli, ["--config", str(config_path), "runtime", "list"])

    assert result.exit_code == 0, result.output
    assert "No installed Arc Llama runtimes found" in result.output

def test_missing_release_says_why_and_that_nothing_was_installed():
    import httpx

    from arc_llama.optimized_runtime import OptimizedRuntimeError
    from arc_llama.optimized_runtime import _release as fetch_release

    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404)))
    with pytest.raises(OptimizedRuntimeError, match="isn't published yet.*Nothing was installed"):
        fetch_release(client, "latest")


@pytest.mark.parametrize(("tag", "nudged"), [("b12193", True), (LLAMA_CPP_BUILD, False), ("b99999", False)])
def test_older_runtime_notice_only_for_installs_older_than_the_pin(tmp_path, tag, nudged):
    install = tmp_path / "runtime" / f"llama-{tag}-sycl"
    (install / "bin").mkdir(parents=True)
    binary = install / "bin" / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
    binary.write_bytes(b"runtime")
    (install / ".arc-llama-runtime.json").write_text(json.dumps({"tag": tag, "backend": "sycl"}), encoding="utf-8")

    notice = older_runtime_notice(binary)

    assert (notice is not None) == nudged
    if nudged:
        assert f"you have {tag}" in notice and "arc-llama install-runtime --backend sycl" in notice


def test_older_runtime_notice_ignores_unmarked_and_path_binaries(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"own build")

    assert older_runtime_notice(binary) is None
    assert older_runtime_notice("llama-server") is None
