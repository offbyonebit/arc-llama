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
