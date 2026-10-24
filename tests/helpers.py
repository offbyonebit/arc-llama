"""Small shared test doubles; scenario-specific behavior stays in each test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import arc_llama.router as router_mod
from arc_llama.router import Router


@dataclass
class FakeTensor:
    name: str
    n_bytes: int


class FakeField:
    def __init__(self, value: Any):
        self._value = value

    def contents(self) -> Any:
        return self._value


class FakeTensorReader:
    def __init__(self, tensors: list[FakeTensor], arch: str):
        self.tensors = tensors
        self._arch = arch

    def get_field(self, key: str):
        return FakeField(self._arch) if key == "general.architecture" else None


class AsyncContextClient:
    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeServer:
    starts: list[str] = []
    stops: list[str] = []

    def __init__(self, plan, name):
        self.plan = plan
        self.name = name
        self.running = False
        self.ready = False

    @property
    def is_running(self):
        return self.running

    def start(self, log_dir=None):
        self.running = True
        self.ready = False
        self.starts.append(self.name)

    async def wait_ready(self):
        self.ready = True
        return True

    def stop(self):
        self.running = False
        self.ready = False
        self.stops.append(self.name)

    async def astop(self, drain_seconds=3.0):
        # Mirrors LlamaServer.astop, which offloads the blocking stop() to a
        # thread. The router awaits this from the event loop.
        self.stop()


def fake_router(tmp_path, monkeypatch, *, single=True) -> Router:
    from conftest import make_config

    FakeServer.starts = []
    FakeServer.stops = []
    cfg = make_config(tmp_path, single_resident=single)
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)

    async def inline_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(router_mod.asyncio, "to_thread", inline_to_thread)
    return Router(cfg)


class ConfigRouter:
    def __init__(self, cfg, log_dir=None):
        self.cfg = cfg
        self._servers = {}

    def all_models(self):
        return list(self.cfg.models)

    async def shutdown(self):
        pass


def python_executable(path, source):
    """A real executable test double on both POSIX and Windows; no shell.

    Windows does not execute shebang scripts. Use the same native console
    launchers used for installed Python entry points, preserving argv quoting.
    """
    import os
    import sys
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if source.startswith("#!"):
        source = source.split("\n", 1)[1]
    if os.name != "nt":
        path.write_text(f"#!{sys.executable}\n" + source, encoding="utf-8", newline="\n")
        path.chmod(0o755)
        return str(path)

    from distlib.scripts import ScriptMaker

    script = path.with_name(path.name + ".py")
    script.write_text("#!python\n" + source, encoding="utf-8", newline="\n")
    maker = ScriptMaker(str(script.parent), str(path.parent))
    maker.executable = sys.executable
    maker.force = True
    return maker.make(script.name)[0]


def sparse_file(path, size):
    """Create a logical model-sized file without allocating model-sized storage."""
    import os

    with path.open("wb") as stream:
        if os.name == "nt":
            import ctypes
            import msvcrt
            from ctypes import wintypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            control = kernel.DeviceIoControl
            control.argtypes = [
                wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID,
            ]
            control.restype = wintypes.BOOL
            returned = wintypes.DWORD()
            # FSCTL_SET_SPARSE: Windows truncate alone allocates all 26 GiB.
            handle = msvcrt.get_osfhandle(stream.fileno())
            if not control(handle, 0x900C4, None, 0, None, 0, ctypes.byref(returned), None):
                raise ctypes.WinError(ctypes.get_last_error())
            # CRT truncate zero-fills the extension even after SET_SPARSE.
            # Set the logical EOF directly instead of writing those zeros.
            move = kernel.SetFilePointerEx
            move.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wintypes.DWORD]
            move.restype = wintypes.BOOL
            end = kernel.SetEndOfFile
            end.argtypes = [wintypes.HANDLE]
            end.restype = wintypes.BOOL
            if not move(handle, size, None, 0) or not end(handle):
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            stream.truncate(size)
