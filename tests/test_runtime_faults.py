"""Synthetic failures must never promote a partial replacement runtime."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import httpx
import pytest

import arc_llama.runtime as runtime
from arc_llama.arch import Backend
from arc_llama.config import Config, PathsConfig


def archive_bytes() -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        data = b"synthetic replacement executable\n"
        entry = tarfile.TarInfo("bin/llama-server")
        entry.mode = 0o755
        entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))
    return output.getvalue()


@pytest.mark.parametrize(
    "failure",
    ["disconnect", "size", "checksum", "extract", "cancel_extract", "backend", "marker", "publish"],
)
def test_failed_replacement_keeps_install_and_config_then_retries(tmp_path, monkeypatch, failure):
    payload = archive_bytes()
    dest = tmp_path / "runtime"
    install = dest / "llama-b12345-vulkan"
    install.mkdir(parents=True)
    original_binary = install / "llama-server"
    original_binary.write_bytes(b"previous verified synthetic executable\n")
    marker = install / ".arc-llama-runtime.json"
    marker.write_text('{"previous":true}\n')
    cfg = Config(paths=PathsConfig(state_dir=str(tmp_path), llama_server=str(original_binary)))
    config_path = tmp_path / "config.toml"
    cfg.save(config_path)
    config_before = config_path.read_bytes()
    install_before = {p.name: p.read_bytes() for p in install.iterdir()}
    release = {"tag_name": "b12345", "assets": [{
        "name": "llama-b12345-bin-ubuntu-vulkan-x64.tar.gz",
        "browser_download_url": "https://synthetic.invalid/runtime",
        "size": len(payload), "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }]}
    monkeypatch.setattr(runtime, "host_platform", lambda: ("linux", "x64"))
    monkeypatch.setattr(runtime, "resolve_release", lambda *_: release)
    monkeypatch.setattr(runtime, "detect_llama_server_backend", lambda _: Backend.VULKAN)

    class Disconnected(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            yield payload[:10]
            raise httpx.ReadError("synthetic interrupted download")

        def close(self):
            self.closed = True

    interrupted = Disconnected()
    with monkeypatch.context() as fault:
        if failure == "size":
            release["assets"][0]["size"] += 1
        elif failure == "checksum":
            release["assets"][0]["digest"] = "sha256:" + "00" * 32
        elif failure in ("extract", "cancel_extract"):
            real_extract = runtime.extract_archive

            def fail_extract(archive, stage):
                real_extract(archive, stage)
                if failure == "cancel_extract":
                    raise KeyboardInterrupt("synthetic cancellation")
                raise OSError("synthetic extraction failure after partial output")

            fault.setattr(runtime, "extract_archive", fail_extract)
        elif failure == "backend":
            fault.setattr(runtime, "detect_llama_server_backend", lambda _: Backend.SYCL)
        elif failure == "marker":
            real_write = Path.write_text

            def fail_marker(path, *args, **kwargs):
                if path.name == ".arc-llama-runtime.json":
                    raise OSError("synthetic disk-full writing marker")
                return real_write(path, *args, **kwargs)

            fault.setattr(Path, "write_text", fail_marker)
        elif failure == "publish":
            real_replace = runtime.os.replace

            def fail_publish(source, target):
                if ".staging-" in Path(source).name and Path(target) == install:
                    raise OSError("synthetic publication failure after old install moved")
                return real_replace(source, target)

            fault.setattr(runtime.os, "replace", fail_publish)

        def response(_request):
            if failure == "disconnect":
                return httpx.Response(200, stream=interrupted)
            return httpx.Response(200, content=payload)

        with httpx.Client(transport=httpx.MockTransport(response)) as client:
            with pytest.raises((runtime.RuntimeInstallError, OSError, KeyboardInterrupt, httpx.ReadError)):
                runtime.install_runtime(dest=dest, cfg=cfg, config_path=config_path, client=client, force=True)

    assert {p.name: p.read_bytes() for p in install.iterdir()} == install_before
    assert config_path.read_bytes() == config_before
    assert cfg.paths.llama_server == str(original_binary)
    assert list(dest.iterdir()) == [install], "failed downloads/staging/backups leaked"
    if failure == "disconnect":
        assert interrupted.closed, "interrupted HTTP response was not closed"

    release["assets"][0].update(size=len(payload), digest="sha256:" + hashlib.sha256(payload).hexdigest())
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=payload))) as client:
        result = runtime.install_runtime(dest=dest, cfg=cfg, config_path=config_path, client=client, force=True)
    assert result.binary_path.read_bytes() == b"synthetic replacement executable\n"
    assert cfg.paths.llama_server == str(result.binary_path)
    assert list(dest.iterdir()) == [install]
