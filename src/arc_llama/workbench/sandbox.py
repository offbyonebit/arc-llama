"""Optional Linux command execution; there is deliberately no host-shell fallback."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

from arc_llama.workbench import files


def available() -> bool:
    return sys.platform == "linux" and bool(
        shutil.which("bwrap") and shutil.which("prlimit") and shutil.which("systemd-run")
    )


def command_argv(root: Path, command: str) -> list[str]:
    if not available():
        raise ValueError(
            "Isolated commands require Linux, Bubblewrap, prlimit, and a working systemd user session. File-only work remains available."
        )
    args = [
        str(shutil.which("systemd-run")),
        "--user",
        "--scope",
        "--quiet",
        "-p",
        "MemoryMax=1G",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "TasksMax=64",
        "-p",
        "CPUQuota=100%",
        str(shutil.which("prlimit")),
        "--as=1073741824",
        "--cpu=30",
        "--fsize=16777216",
        "--nofile=128",
        "--",
        str(shutil.which("bwrap")),
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
    ]
    for path in ("/usr", "/bin", "/lib", "/lib64"):
        if Path(path).exists():
            args += ["--ro-bind", path, path]
    args += [
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--bind",
        str(root),
        "/workspace",
        "--chdir",
        "/workspace",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "LANG",
        "C.UTF-8",
        "/bin/sh",
        "-c",
        command,
    ]
    return args


async def run(root: Path, command: str) -> tuple[str, bool]:
    proc = await asyncio.create_subprocess_exec(
        *command_argv(root, command),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={
            "PATH": os.defpath,
            **{
                k: os.environ[k]
                for k in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")
                if k in os.environ
            },
        },
    )
    output = bytearray()

    async def collect() -> None:
        assert proc.stdout is not None
        while chunk := await proc.stdout.read(4096):
            output.extend(chunk)
            if len(output) > 128 * 1024:
                raise ValueError("Command exceeded its 128 KiB output limit.")
        await proc.wait()

    async def watch_copy() -> None:
        def check() -> None:
            count = size = 0
            for current, dirs, names in os.walk(root, followlinks=False):
                count += len(dirs) + len(names)
                if count > files.MAX_FILES:
                    raise ValueError("Command exceeded the project copy's file limit")
                for name in names:
                    try:
                        size += (Path(current) / name).lstat().st_size
                    except FileNotFoundError:
                        continue
                    if size > files.MAX_BYTES:
                        raise ValueError("Command exceeded the project copy's size limit")

        while proc.returncode is None:
            await asyncio.to_thread(check)
            await asyncio.sleep(0.25)

    tasks = [asyncio.create_task(collect()), asyncio.create_task(watch_copy())]
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), 60)
        text = output.decode("utf-8", "replace")
        if proc.returncode:
            return (
                text
                + f"\nCommand exited with status {proc.returncode}. The OS sandbox may be unavailable on this host.",
                True,
            )
        return text, False
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
