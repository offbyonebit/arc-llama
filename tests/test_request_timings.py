"""Request timing metrics: bounded, honest, from real traffic only.

The router keeps per-model cold-start, TTFT, generation-speed and
queue-wait samples. Every summary must be derived from actual observed
values; nothing is invented for endpoints without samples, and the sample
lists are capped.
"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from arc_llama.config import Config
from arc_llama.router import ModelTimings, Router, summarise_seconds
from arc_llama.stream_metrics import StreamMetricsObserver, generation_rate

# ---------------------------------------------------------------------------
# ModelTimings: bounded recording and honest snapshots
# ---------------------------------------------------------------------------


def test_snapshot_is_absent_without_samples():
    t = ModelTimings()
    # No traffic at all: per-model map empty, no queue entry.
    snapshot = t.snapshot()
    assert snapshot == {"models": {}}
    assert "queue_wait" not in snapshot


def test_cold_start_summary_reports_count_median_last():
    t = ModelTimings()
    t.record_cold_start("qwen", 21.2)
    t.record_cold_start("qwen", 19.0)
    t.record_cold_start("qwen", 40.0)
    summary = t.snapshot()["models"]["qwen"]["cold_start"]
    assert summary["count"] == 3
    assert summary["median_s"] == 21.2
    assert summary["last_s"] == 40.0
    assert summary["p95_s"] >= summary["median_s"]


def test_per_model_lists_are_capped_and_oldest_dropped():
    t = ModelTimings(cap=3)
    for i in range(10):
        t.record_ttft("qwen", float(i))
    bucket = t.ttft["qwen"]
    assert len(bucket) == 3
    # Oldest dropped; the most recent values survive.
    assert bucket == [7.0, 8.0, 9.0]


def test_nonfinite_and_negative_samples_are_ignored():
    t = ModelTimings()
    t.record_ttft("qwen", math.inf)
    t.record_ttft("qwen", math.nan)
    t.record_ttft("qwen", -1.0)
    t.record_queue_wait(math.nan)
    assert t.snapshot() == {"models": {}}


def test_queue_wait_summary_uses_recent_samples():
    t = ModelTimings()
    t.record_queue_wait(0.01)
    t.record_queue_wait(0.03)
    snapshot = t.snapshot()
    assert snapshot["queue_wait"]["count"] == 2
    assert snapshot["queue_wait"]["last_s"] == 0.03


def test_model_wait_is_attributed_per_model():
    t = ModelTimings()
    t.record_queue_wait(0.4, "qwen")
    t.record_queue_wait(0.02, "gemma")
    models = t.snapshot()["models"]
    assert models["qwen"]["model_wait"]["last_s"] == 0.4
    assert models["gemma"]["model_wait"]["last_s"] == 0.02


def test_generation_speed_summary_renames_units():
    t = ModelTimings()
    t.record_generation_tok_s("qwen", 24.0)
    t.record_generation_tok_s("qwen", 26.0)
    entry = t.snapshot()["models"]["qwen"]["generation_tok_s"]
    # tok/s values keep their unit in every field name.
    assert entry["median_tok_s"] == 25.0
    assert entry["last_tok_s"] == 26.0
    assert "median_s" not in entry


def test_model_names_are_bounded_too():
    t = ModelTimings()
    for i in range(300):
        t.record_cold_start(f"m{i}", 1.0)
    assert len(t.cold_starts) <= 128


# ---------------------------------------------------------------------------
# summarise_seconds: honest stats
# ---------------------------------------------------------------------------


def test_summarise_seconds_empty_is_none():
    assert summarise_seconds([]) is None


# ---------------------------------------------------------------------------
# StreamMetricsObserver: SSE parsing for TTFT and generation rate
# ---------------------------------------------------------------------------


def sse(payload: dict) -> bytes:
    import json

    return f"data: {json.dumps(payload)}\n\n".encode()


def test_observer_records_first_generated_content_as_ttft_origin():
    obs = StreamMetricsObserver()
    obs.feed(b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n", now=1.0)
    assert obs.first_token_at is None, "role-only deltas are not generated content"
    obs.feed(sse({"choices": [{"delta": {"content": "H"}}]}), now=2.5)
    assert obs.first_token_at == 2.5


def test_observer_reads_generation_rate_from_backend_timings():
    obs = StreamMetricsObserver()
    chunk = sse(
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "timings": {"predicted_n": 64, "predicted_ms": 2000},
        }
    )
    obs.feed(chunk, now=1.0)
    assert obs.generation_tok_s == pytest.approx(32.0)


def test_observer_honours_predicted_per_second():
    obs = StreamMetricsObserver()
    obs.feed(sse({"timings": {"predicted_per_second": 41.5}}), now=1.0)
    assert obs.generation_tok_s == pytest.approx(41.5)


def test_observer_ignores_events_without_usable_timings():
    obs = StreamMetricsObserver()
    obs.feed(sse({"choices": [{"delta": {"content": "x"}}]}), now=1.0)
    obs.feed(b"data: [DONE]\n\n", now=1.1)
    obs.feed(b"event: ping\ndata: {}\n\n", now=1.2)
    assert obs.generation_tok_s is None
    assert obs.first_token_at == 1.0


def test_observer_handles_json_split_across_chunks():
    obs = StreamMetricsObserver()
    payload = sse({"choices": [{"delta": {"content": "split"}}]})
    half = len(payload) // 2
    obs.feed(payload[:half], now=1.0)
    assert obs.first_token_at is None
    obs.feed(payload[half:], now=1.5)
    assert obs.first_token_at == 1.5


def test_observer_caps_oversized_lines_without_breaking():
    obs = StreamMetricsObserver(max_line_bytes=64)
    obs.feed(b"data: " + b"x" * 200 + b"\n\n", now=1.0)
    obs.feed(sse({"choices": [{"delta": {"content": "ok"}}]}), now=2.0)
    assert obs.first_token_at == 2.0


def test_generation_rate_rejects_malformed_payloads():
    assert generation_rate("not a dict") is None
    assert generation_rate({"timings": {}}) is None
    assert generation_rate({"timings": {"predicted_n": 0, "predicted_ms": 0}}) is None
    assert generation_rate({"timings": {"predicted_n": 10, "predicted_ms": "x"}}) is None
    # usage-wrapped timings (non-streaming shape) also accepted
    assert generation_rate({"usage": {"timings": {"predicted_per_second": 30.0}}}) == 30.0


# ---------------------------------------------------------------------------
# /admin/metrics endpoint shape
# ---------------------------------------------------------------------------


class _TimingsRouter(Router):
    """Router with pre-seeded timings; construction still builds servers."""

    def __init__(self, cfg, log_dir=None):
        super().__init__(cfg, log_dir=log_dir)
        self.timings = ModelTimings()
        self.timings.record_cold_start("qwen", 18.4)
        self.timings.record_ttft("qwen", 0.3)
        self.timings.record_generation_tok_s("qwen", 24.9)
        self.timings.record_queue_wait(0.05, "qwen")


def test_admin_metrics_endpoint_shape(monkeypatch, tmp_path):
    import arc_llama.server as server_mod
    from arc_llama.config import GPUConfig, ModelConfig

    monkeypatch.setattr(server_mod, "Router", _TimingsRouter)
    monkeypatch.setattr(
        server_mod,
        "UpstreamManager",
        lambda upstreams=None: FakeUpstreamMinimal(),
    )
    cfg = Config()
    cfg.gpus = [GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=24576)]
    cfg.models = [
        ModelConfig(
            name="qwen",
            path=str(tmp_path / "qwen.gguf"),
            port=18080,
            gpu_pci_slot="0000:03:00.0",
        )
    ]
    app = server_mod.create_app(cfg, plugins=[])
    with TestClient(app) as client:
        r = client.get("/admin/metrics")
    assert r.status_code == 200
    data = r.json()
    entry = data["timings"]["models"]["qwen"]
    assert entry["cold_start"]["last_s"] == pytest.approx(18.4, abs=0.01)
    assert entry["ttft"]["last_s"] == pytest.approx(0.3, abs=0.01)
    assert entry["generation_tok_s"]["last_tok_s"] == pytest.approx(24.9)
    assert entry["model_wait"]["last_s"] == pytest.approx(0.05)
    assert data["timings"]["queue_wait"]["count"] == 1
    # Autotune before/after is present but None until a real sweep reported.
    models_autotune = {m["name"]: m for m in data["autotune"]["models"]}
    assert models_autotune["qwen"]["before_after"] is None


class FakeUpstreamMinimal:
    def __init__(self, upstreams=None):
        pass

    async def models(self):
        return []

    def find_model(self, model_id):
        return None

    def upstreams_status(self):
        return []
