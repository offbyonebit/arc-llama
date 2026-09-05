from __future__ import annotations

import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import pytest

from arc_llama.arch import Backend
from arc_llama.runtime import (
    RuntimeInstallError,
    _resolve_windows_release_with_asset,
    asset_suffix,
    extract_archive,
    host_platform,
    install_runtime,
    select_asset,
)

# ---------------------------------------------------------------------------
# host_platform
# ---------------------------------------------------------------------------


def test_host_platform_maps_arch(monkeypatch):
    import arc_llama.runtime as rt

    monkeypatch.setattr(rt.platform, "system", lambda: "Linux")
    monkeypatch.setattr(rt.platform, "machine", lambda: "x86_64")
    assert host_platform() == ("linux", "x64")

    monkeypatch.setattr(rt.platform, "system", lambda: "Windows")
    monkeypatch.setattr(rt.platform, "machine", lambda: "AMD64")
    assert host_platform() == ("windows", "x64")

    monkeypatch.setattr(rt.platform, "system", lambda: "Linux")
    monkeypatch.setattr(rt.platform, "machine", lambda: "aarch64")
    assert host_platform() == ("linux", "arm64")


# ---------------------------------------------------------------------------
# asset_suffix
# ---------------------------------------------------------------------------


def test_asset_suffix_all_supported():
    assert asset_suffix("linux", "x64", "vulkan") == "bin-ubuntu-vulkan-x64.tar.gz"
    assert asset_suffix("linux", "x64", "sycl") == "bin-ubuntu-sycl-fp16-x64.tar.gz"
    assert asset_suffix("windows", "x64", "vulkan") == "bin-win-vulkan-x64.zip"
    assert asset_suffix("windows", "x64", "sycl") == "bin-win-sycl-x64.zip"


@pytest.mark.parametrize(
    ("os_name", "arch", "backend"),
    [
        ("macos", "x64", "vulkan"),
        ("linux", "arm64", "vulkan"),
        ("linux", "x64", "rocm"),
        ("windows", "arm64", "vulkan"),
    ],
)
def test_asset_suffix_unsupported_raises(os_name, arch, backend):
    with pytest.raises(RuntimeInstallError):
        asset_suffix(os_name, arch, backend)


# ---------------------------------------------------------------------------
# select_asset
# ---------------------------------------------------------------------------


def test_select_asset_matches():
    release = {
        "tag_name": "b10092",
        "assets": [
            {
                "name": "llama-b10092-bin-ubuntu-sycl-fp16-x64.tar.gz",
                "browser_download_url": "http://x/s",
                "size": 500,
            },
            {
                "name": "llama-b10092-bin-ubuntu-vulkan-x64.tar.gz",
                "browser_download_url": "http://x/v",
                "size": 123,
            },
        ],
    }
    asset = select_asset(release, "linux", "x64", "vulkan")
    assert asset.name == "llama-b10092-bin-ubuntu-vulkan-x64.tar.gz"
    assert asset.url == "http://x/v"
    assert asset.size == 123
    assert asset.tag == "b10092"


def test_select_asset_matches_current_windows_release_name():
    release = {
        "tag_name": "b10819",
        "assets": [
            {
                "name": "llama-b10819-bin-win-vulkan-x64.zip",
                "browser_download_url": "http://x/v",
                "size": 123,
            },
        ],
    }
    asset = select_asset(release, "windows", "x64", "vulkan")
    assert asset.name == "llama-b10819-bin-win-vulkan-x64.zip"


def test_windows_latest_fallback_skips_assetless_latest_release():
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        def get(self, url):
            assert url.endswith("/releases?per_page=30")
            return Response(
                [
                    {"tag_name": "v0.4.0", "assets": []},
                    {
                        "tag_name": "b10819",
                        "assets": [
                            {
                                "name": "llama-b10819-bin-win-vulkan-x64.zip",
                                "browser_download_url": "http://x/v",
                                "size": 123,
                            }
                        ],
                    },
                ]
            )

    release = _resolve_windows_release_with_asset(Client(), "x64", "vulkan")
    assert release["tag_name"] == "b10819"


def test_select_asset_missing_raises():
    release = {
        "tag_name": "b10092",
        "assets": [
            {
                "name": "llama-b10092-bin-ubuntu-sycl-fp16-x64.tar.gz",
                "browser_download_url": "http://x/s",
                "size": 500,
            },
        ],
    }
    with pytest.raises(RuntimeInstallError):
        select_asset(release, "linux", "x64", "vulkan")


# ---------------------------------------------------------------------------
# extract_archive
# ---------------------------------------------------------------------------


def _make_fake_server(tmp_path: Path, name: str = "llama-server") -> Path:
    """Create a small fake binary file inside a build/bin directory."""
    src = tmp_path / "src"
    (src / "build" / "bin").mkdir(parents=True)
    server = src / "build" / "bin" / name
    server.write_text("#!/bin/sh\necho hello\n")
    return server


def test_extract_tar_gz_finds_binary(tmp_path):
    server = _make_fake_server(tmp_path)
    archive = tmp_path / "test.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(server, arcname="build/bin/llama-server")

    dest = tmp_path / "extracted"
    found = extract_archive(archive, dest)
    assert found.exists()
    assert found.name == "llama-server"
    if os.name == "posix":
        assert os.access(found, os.X_OK)


def test_extract_zip_finds_exe(tmp_path):
    src = tmp_path / "src"
    (src / "bin").mkdir(parents=True)
    server = src / "bin" / "llama-server.exe"
    server.write_text("fake exe")

    archive = tmp_path / "test.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(server, arcname="bin/llama-server.exe")

    dest = tmp_path / "extracted"
    found = extract_archive(archive, dest)
    assert found.exists()
    assert found.name == "llama-server.exe"


def test_extract_no_binary_raises(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "other.txt").write_text("not a binary")

    archive = tmp_path / "test.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(src / "other.txt", arcname="other.txt")

    dest = tmp_path / "extracted"
    with pytest.raises(RuntimeInstallError, match="no llama-server"):
        extract_archive(archive, dest)


# ---------------------------------------------------------------------------
# install_runtime end-to-end (monkeypatched, no real network)
# ---------------------------------------------------------------------------


def _build_fake_archive(tmp_path) -> Path:
    """Build a real .tar.gz containing a fake llama-server and return its path."""
    server = _make_fake_server(tmp_path)
    archive = tmp_path / "fake.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(server, arcname="build/bin/llama-server")
    return archive


def _fake_release() -> dict:
    return {
        "tag_name": "b10092",
        "assets": [
            {
                "name": "llama-b10092-bin-ubuntu-vulkan-x64.tar.gz",
                "browser_download_url": "http://x/v",
                "size": 123,
            },
        ],
    }


def test_install_runtime_end_to_end(monkeypatch, tmp_path):
    import arc_llama.runtime as rt
    from arc_llama.config import Config, PathsConfig

    fake_archive = _build_fake_archive(tmp_path)

    monkeypatch.setattr(rt, "resolve_release", lambda client, version: _fake_release())

    def fake_download(client, asset, dest_file, on_progress=None):
        shutil.copy(fake_archive, dest_file)
        return dest_file

    monkeypatch.setattr(rt, "download_asset", fake_download)
    monkeypatch.setattr(rt, "detect_llama_server_backend", lambda p: Backend.VULKAN)
    monkeypatch.setattr(rt, "host_platform", lambda: ("linux", "x64"))

    cfg = Config(paths=PathsConfig(state_dir=str(tmp_path), llama_server=""))
    config_path = tmp_path / "config.toml"

    result = install_runtime(
        backend="vulkan",
        cfg=cfg,
        set_default=True,
        config_path=config_path,
    )

    assert result.binary_path.exists()
    assert result.backend == Backend.VULKAN
    assert result.requested_backend == "vulkan"
    assert result.tag == "b10092"
    assert cfg.paths.llama_server == str(result.binary_path)
    assert config_path.exists()


def test_install_runtime_force_false_reuses(monkeypatch, tmp_path):
    import arc_llama.runtime as rt
    from arc_llama.config import Config, PathsConfig

    fake_archive = _build_fake_archive(tmp_path)

    monkeypatch.setattr(rt, "resolve_release", lambda client, version: _fake_release())

    call_count = {"n": 0}

    def fake_download(client, asset, dest_file, on_progress=None):
        call_count["n"] += 1
        shutil.copy(fake_archive, dest_file)
        return dest_file

    monkeypatch.setattr(rt, "download_asset", fake_download)
    monkeypatch.setattr(rt, "detect_llama_server_backend", lambda p: Backend.VULKAN)
    monkeypatch.setattr(rt, "host_platform", lambda: ("linux", "x64"))

    cfg = Config(paths=PathsConfig(state_dir=str(tmp_path), llama_server=""))
    config_path = tmp_path / "config.toml"

    # First call: downloads and installs.
    result1 = install_runtime(
        backend="vulkan",
        cfg=cfg,
        set_default=False,
        config_path=config_path,
    )
    assert call_count["n"] == 1
    assert result1.binary_path.exists()

    # Second call without --force: should reuse, not re-download.
    result2 = install_runtime(
        backend="vulkan",
        cfg=cfg,
        set_default=False,
        config_path=config_path,
    )
    assert call_count["n"] == 1
    assert result2.binary_path == result1.binary_path


def test_install_runtime_force_redownloads(monkeypatch, tmp_path):
    import arc_llama.runtime as rt
    from arc_llama.config import Config, PathsConfig

    fake_archive = _build_fake_archive(tmp_path)

    monkeypatch.setattr(rt, "resolve_release", lambda client, version: _fake_release())

    call_count = {"n": 0}

    def fake_download(client, asset, dest_file, on_progress=None):
        call_count["n"] += 1
        shutil.copy(fake_archive, dest_file)
        return dest_file

    monkeypatch.setattr(rt, "download_asset", fake_download)
    monkeypatch.setattr(rt, "detect_llama_server_backend", lambda p: Backend.VULKAN)
    monkeypatch.setattr(rt, "host_platform", lambda: ("linux", "x64"))

    cfg = Config(paths=PathsConfig(state_dir=str(tmp_path), llama_server=""))
    config_path = tmp_path / "config.toml"

    install_runtime(backend="vulkan", cfg=cfg, set_default=False, config_path=config_path)
    assert call_count["n"] == 1

    install_runtime(
        backend="vulkan", cfg=cfg, set_default=False, config_path=config_path, force=True
    )
    assert call_count["n"] == 2


def test_install_runtime_aligns_gpu_backend(monkeypatch, tmp_path):
    import arc_llama.runtime as rt
    from arc_llama.config import Config, GPUConfig, PathsConfig

    fake_archive = _build_fake_archive(tmp_path)

    monkeypatch.setattr(rt, "resolve_release", lambda client, version: _fake_release())

    def fake_download(client, asset, dest_file, on_progress=None):
        shutil.copy(fake_archive, dest_file)
        return dest_file

    monkeypatch.setattr(rt, "download_asset", fake_download)
    monkeypatch.setattr(rt, "detect_llama_server_backend", lambda p: Backend.VULKAN)
    monkeypatch.setattr(rt, "host_platform", lambda: ("linux", "x64"))

    cfg = Config(
        paths=PathsConfig(state_dir=str(tmp_path), llama_server=""),
        gpus=[
            GPUConfig(
                pci_slot="0000:03:00.0",
                sycl_index=0,
                arch="battlemage",
                vram_mb=24576,
                enabled=True,
                backend="sycl",
            )
        ],
    )

    result = install_runtime(
        backend="vulkan",
        cfg=cfg,
        set_default=True,
        config_path=tmp_path / "config.toml",
    )

    assert cfg.gpus[0].backend == "vulkan"
    assert cfg.paths.llama_server == str(result.binary_path)
