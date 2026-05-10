from __future__ import annotations

from arc_llama.router import Router


class FakeServer:
    starts: list[str] = []
    stops: list[str] = []

    def __init__(self, plan, name):
        self.plan = plan
        self.name = name
        self.running = False

    @property
    def is_running(self):
        return self.running

    def start(self, log_dir=None):
        self.running = True
        self.starts.append(self.name)

    async def wait_ready(self):
        return True

    def stop(self):
        self.running = False
        self.stops.append(self.name)


async def test_single_resident_policy_stops_other_models_before_starting_target(tmp_path, monkeypatch):
    from conftest import make_config

    import arc_llama.router as router_mod

    FakeServer.starts = []
    FakeServer.stops = []
    cfg = make_config(tmp_path, single_resident=True)
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)
    rt = Router(cfg)

    await rt.ensure_active("qwen")
    await rt.ensure_active("gemma")

    assert FakeServer.starts == ["qwen", "gemma"]
    assert FakeServer.stops == ["qwen"]


async def test_multi_resident_policy_keeps_models_on_different_gpus_running(tmp_path, monkeypatch):
    from conftest import make_config

    import arc_llama.router as router_mod

    FakeServer.starts = []
    FakeServer.stops = []
    cfg = make_config(tmp_path, single_resident=False)
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)
    rt = Router(cfg)

    await rt.ensure_active("qwen")
    await rt.ensure_active("gemma")

    assert FakeServer.starts == ["qwen", "gemma"]
    assert FakeServer.stops == []
