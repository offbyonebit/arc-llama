import json

from arc_llama.stream_metrics import StreamMetricsObserver, generation_rate


def event(data):
    return b"data: " + json.dumps(data).encode() + b"\r\n\r\n"


def test_ttft_ignores_role_and_handles_split_network_events():
    observer = StreamMetricsObserver()
    observer.feed(event({"choices": [{"delta": {"role": "assistant"}}]}), 1)
    assert observer.first_token_at is None
    data = event({"choices": [{"delta": {"content": "hello"}}]})
    observer.feed(data[:17], 2)
    assert observer.first_token_at is None
    observer.feed(data[17:], 3)
    assert observer.first_token_at == 3
    observer.feed(event({"choices": [{"delta": {"content": " again"}}]}), 4)
    assert observer.first_token_at == 3


def test_generation_speed_requires_backend_generation_timing():
    assert generation_rate({"usage": {"completion_tokens": 100}}) is None
    assert generation_rate({"timings": {"predicted_n": 100, "predicted_ms": 2000}}) == 50
    assert generation_rate({"timings": {"predicted_per_second": float("inf")}}) is None
    observer = StreamMetricsObserver()
    data = event({"timings": {"predicted_per_second": 42.0}})
    for byte in data:
        observer.feed(bytes([byte]), 5)
    assert observer.generation_tok_s == 42


def test_oversized_or_bad_event_cannot_grow_buffer_or_poison_next_event():
    observer = StreamMetricsObserver(max_line_bytes=128)
    observer.feed(b"data: " + b"x" * 10000, 1)
    assert len(observer._pending) <= 128
    observer.feed(b"\ndata: broken\n\n", 2)
    observer.feed(event({"choices": [{"delta": {"reasoning_content": "hmm"}}]}), 3)
    assert observer.first_token_at == 3


def test_proxy_records_split_sse_without_changing_bytes_or_leaking_slot(monkeypatch):
    from fastapi.testclient import TestClient
    from test_server import FakeAsyncClient, FakeRouter, FakeUpstreamManager, FakeUpstreamStream

    import arc_llama.server as server_mod
    from arc_llama.config import Config
    from arc_llama.router import ModelTimings

    payload = event({"choices": [{"delta": {"role": "assistant"}}]})
    payload += event({"choices": [{"delta": {"content": "hello"}}]})
    payload += event({"usage": {"completion_tokens": 10}, "timings": {"predicted_per_second": 25}})
    payload += b"data: [DONE]\n\n"

    class MeasuredRouter(FakeRouter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.timings = ModelTimings()

    class SplitStream(FakeUpstreamStream):
        async def aiter_raw(self):
            for offset in range(0, len(payload), 7):
                yield payload[offset:offset + 7]

    class SplitClient(FakeAsyncClient):
        async def send(self, request, stream=False):
            self.last_stream = SplitStream()
            return self.last_stream

    monkeypatch.setattr(server_mod, "Router", MeasuredRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", SplitClient)
    app = server_mod.create_app(Config(), plugins=[])
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "qwen", "stream": True, "messages": []})
        assert response.content == payload
        assert app.state.router.inflight == 0
        summary = app.state.router.timings.snapshot()["models"]["qwen"]
        assert summary["ttft"]["count"] == 1
        assert summary["generation_tok_s"]["last_tok_s"] == 25
        assert summary["model_wait"]["count"] == 1
