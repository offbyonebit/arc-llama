"""Synthetic runtime installer that dies or pauses at a publication boundary.

Usage: runtime_install_child.py DEST CONFIG POINT ACTION NETWORK FORCE PAYLOAD

POINT is where ACTION happens: ``backup_renamed`` (old install moved aside),
``published`` (staging moved into place), ``trash_renamed`` (backup renamed
for deletion) or ``trash_partial`` (one file of that tree deleted), or
``none``. ACTION is ``exit`` (os._exit(77), no cleanup or finally blocks) or
``pause`` (write DEST/../reached, wait for DEST/../go). NETWORK is ``online``
or ``offline``. PAYLOAD is the synthetic executable's text. Uses only the
temporary directories it is given; prints one JSON result line.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import httpx

import arc_llama.runtime as rt
from arc_llama.arch import Backend
from arc_llama.config import load_config

dest, config_path = Path(sys.argv[1]), Path(sys.argv[2])
point, action, network, force, text = sys.argv[3:8]
signal_dir = dest.parent


def archive(content: bytes) -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as tf:
        for name, data in (("llama-server", content), ("libggml.so", b"synthetic library\n")):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o755
            tf.addfile(entry, io.BytesIO(data))
    return out.getvalue()


payload = archive(text.encode())
release = {"tag_name": "b12345", "assets": [{
    "name": "llama-b12345-bin-ubuntu-vulkan-x64.tar.gz",
    "browser_download_url": "https://synthetic.invalid/runtime",
    "size": len(payload), "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
}]}


def resolve(*_args):
    if network == "offline":
        raise httpx.ConnectError("synthetic offline")
    return release


rt.host_platform = lambda: ("linux", "x64")
rt.resolve_release = resolve
rt.detect_llama_server_backend = lambda _path: Backend.VULKAN


def boundary() -> None:
    if action == "exit":
        os._exit(77)
    (signal_dir / "reached").write_text(point)
    deadline = time.monotonic() + 30
    while not (signal_dir / "go").exists():
        if time.monotonic() > deadline:
            os._exit(78)
        time.sleep(0.02)


real_replace = os.replace


def replace(src, dst):
    real_replace(src, dst)
    name = Path(dst).name
    if point == "backup_renamed" and ".backup-" in name:
        boundary()
    elif point == "published" and name == "llama-b12345-vulkan":
        boundary()
    elif point == "trash_renamed" and ".trash-" in name:
        boundary()
    elif point == "trash_partial" and ".trash-" in name:
        (Path(dst) / "libggml.so").unlink()
        boundary()


rt.os.replace = replace  # type: ignore[assignment]
cfg = load_config(config_path)
with httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=payload))) as client:
    try:
        result = rt.install_runtime(
            dest=dest, cfg=cfg, config_path=config_path, client=client, force=force == "force"
        )
    except Exception as exc:  # noqa: BLE001 - reported to the parent test
        print(json.dumps({"ok": False, "error": type(exc).__name__}))
    else:
        print(json.dumps({"ok": True, "binary": str(result.binary_path)}))
