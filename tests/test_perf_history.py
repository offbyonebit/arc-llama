from __future__ import annotations

import json

from httpx import ASGITransport, AsyncClient

from arc_llama.config import Config, ServerConfig
from arc_llama.perf_history import BUCKET_SECONDS, RETENTION_DAYS, PerfHistory
from arc_llama.router import ModelTimings


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_hourly_medians_are_persisted(tmp_path) -> None:
    clock = Clock(10 * BUCKET_SECONDS + 5)
    history = PerfHistory(tmp_path / "h.jsonl", clock=clock)
    for value in (10.0, 30.0, 20.0):
        history.add("m", "generation_tok_s", value)
    history.add("m", "generation_tok_s", float("nan"))
    history.add("m", "bogus", 1.0)
    [partial] = history.query()
    assert partial["median"] == 20.0 and partial["n"] == 3 and partial["partial"]
    assert not (tmp_path / "h.jsonl").exists()
    clock.t += BUCKET_SECONDS
    history.add("m", "generation_tok_s", 40.0)
    lines = (tmp_path / "h.jsonl").read_text().splitlines()
    assert json.loads(lines[0]) == {
        "t": 10 * BUCKET_SECONDS,
        "model": "m",
        "metric": "generation_tok_s",
        "median": 20.0,
        "n": 3,
    }
    history.flush()
    reloaded = PerfHistory(tmp_path / "h.jsonl", clock=clock)
    assert [p["median"] for p in reloaded.query()] == [20.0, 40.0]
    assert reloaded.query(model="other") == []
    assert reloaded.query(metric="ttft_s") == []


def test_old_and_corrupt_points_are_pruned(tmp_path) -> None:
    now = 400 * 86400
    path = tmp_path / "h.jsonl"
    old = {"t": now - (RETENTION_DAYS + 1) * 86400, "model": "m", "metric": "ttft_s", "median": 1, "n": 1}
    fresh = {"t": now - 86400, "model": "m", "metric": "ttft_s", "median": 2, "n": 1}
    path.write_text("\n".join([json.dumps(old), "not json", json.dumps(fresh)]) + "\n")
    history = PerfHistory(path, clock=Clock(now))
    assert [p["median"] for p in history.query(days=90)] == [2]
    assert len(path.read_text().splitlines()) == 1


def test_model_timings_feed_history(tmp_path) -> None:
    history = PerfHistory(tmp_path / "h.jsonl", clock=Clock(0))
    timings = ModelTimings()
    timings.history = history
    timings.record_generation_tok_s("m", 25.0)
    timings.record_ttft("m", 0.4)
    metrics = {(p["metric"], p["median"]) for p in history.query(days=1e9)}
    assert metrics == {("generation_tok_s", 25.0), ("ttft_s", 0.4)}


async def test_history_endpoint(monkeypatch, tmp_path) -> None:
    import arc_llama.server as server_mod

    try:
        from tests.test_server import FakeRouter, FakeUpstreamManager
    except ModuleNotFoundError:
        from test_server import FakeRouter, FakeUpstreamManager
    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    cfg = Config(server=ServerConfig(admin_token="tok"))
    cfg.paths.state_dir = str(tmp_path)
    app = server_mod.create_app(cfg, plugins=[])
    auth = {"Authorization": "Bearer tok"}
    async with app.router.lifespan_context(app):
        app.state.perf_history.add("qwen", "generation_tok_s", 22.0)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            assert (await client.get("/admin/metrics/history")).status_code == 401
            body = (await client.get("/admin/metrics/history", headers=auth)).json()
            assert body["points"][0]["model"] == "qwen"
            bad = await client.get("/admin/metrics/history?metric=nope", headers=auth)
            assert bad.status_code == 400
    assert (tmp_path / "perf-history.jsonl").exists()
