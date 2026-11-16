from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from helpers import ConfigRouter
from starlette.responses import JSONResponse, StreamingResponse

from arc_llama.api_keys import ApiKeyStore
from arc_llama.config import Config, ServerConfig, TuneConfig
from arc_llama.messages_api import from_openai, to_openai
from arc_llama.server import create_app
from arc_llama.upstream import UpstreamModel


def request(**fields):
    return {"model": "local", "max_tokens": 128, "messages": [{"role": "user", "content": "Hello"}], **fields}


def answer(**fields):
    return {"id": "chatcmpl-1", "model": "local", "choices": [{"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 12, "completion_tokens": 3}, **fields}


@pytest.fixture
def upstream_app(tmp_path, monkeypatch):
    import arc_llama.server as mod

    state = SimpleNamespace(bodies=[], response=answer(), status=200, chunks=None, closed=0, clients_closed=0)

    class Response:
        headers = {"content-type": "application/json"}

        @property
        def status_code(self):
            return state.status

        async def aread(self):
            return json.dumps(state.response).encode()

        async def aclose(self):
            state.closed += 1

        async def aiter_raw(self):
            for chunk in state.chunks:
                yield chunk

    class Manager:
        def __init__(self, *args):
            pass

        async def models(self):
            return [UpstreamModel("local", "fixture", "http://fixture")]

        def find_model(self, name):
            return (UpstreamModel("local", "fixture", "http://fixture") if name == "local" else None)

        async def proxy(self, model, path, body, headers, **kwargs):
            assert path == "/v1/chat/completions"
            assert headers == {"Content-Type": "application/json"}  # caller credentials never forwarded
            state.bodies.append(json.loads(body))

            async def close():
                state.clients_closed += 1

            return SimpleNamespace(aclose=close), Response()

    monkeypatch.setattr(mod, "Router", ConfigRouter)
    monkeypatch.setattr(mod, "UpstreamManager", Manager)
    cfg = Config(server=ServerConfig(admin_token="admin"), tune=TuneConfig(auto=False))
    cfg.paths.state_dir = str(tmp_path)
    app = create_app(cfg, plugins=[])
    return app, state


async def call(app, body, headers=None, peer="127.0.0.1"):
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app, client=(peer, 1234)), base_url="http://test") as client:
            return await client.post("/v1/messages", json=body, headers=headers)


async def test_text_system_images_samplers_and_metadata(upstream_app):
    app, state = upstream_app
    result = await call(app, request(system=[{"type": "text", "text": "First", "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "Second"}], messages=[{"role": "user", "content": [{"type": "text", "text": "Look"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}}]}], temperature=0.2, top_p=0.9, top_k=40, stop_sequences=["END"], metadata={"user_id": "fixture"}, cache_control={"type": "ephemeral"}, service_tier="auto", thinking={"type": "adaptive"}, output_config={"effort": "high"}), {"anthropic-beta": "prompt-caching-2024-07-31", "anthropic-version": "2023-06-01"})
    assert result.status_code == 200
    data = result.json()
    assert data["content"] == [{"type": "text", "text": "Hi"}]
    assert data["stop_reason"] == "end_turn"
    assert data["usage"] == {"input_tokens": 12, "output_tokens": 3}
    sent = state.bodies[0]
    assert sent["messages"][0] == {"role": "system", "content": "First\n\nSecond"}
    assert sent["messages"][1]["content"][1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
    assert sent["temperature"] == 0.2 and sent["top_p"] == 0.9 and sent["top_k"] == 40
    assert sent["stop"] == ["END"] and sent["max_tokens"] == 128
    assert sent["chat_template_kwargs"] == {"enable_thinking": True}
    assert sent["reasoning_effort"] == "high"
    assert "metadata" not in sent and "cache_control" not in sent
    assert state.closed == state.clients_closed == 1


async def test_tool_round_trip(upstream_app):
    app, state = upstream_app
    state.response = answer(choices=[{"message": {"content": "", "tool_calls": [{"id": "call_1", "function": {"name": "weather", "arguments": '{"city":"York"}'}}]}, "finish_reason": "tool_calls"}])
    tool = {"name": "weather", "description": "Weather lookup", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}, "cache_control": {"type": "ephemeral"}}
    first = await call(app, request(tools=[tool], tool_choice={"type": "tool", "name": "weather", "disable_parallel_tool_use": True}))
    block = first.json()["content"][0]
    assert block == {"type": "tool_use", "id": "call_1", "name": "weather", "input": {"city": "York"}}
    assert first.json()["stop_reason"] == "tool_use"
    assert state.bodies[0]["tools"][0]["function"]["parameters"] == tool["input_schema"]
    assert state.bodies[0]["tool_choice"] == {"type": "function", "function": {"name": "weather"}}
    assert state.bodies[0]["parallel_tool_calls"] is False
    state.response = answer()
    second = await call(app, request(messages=[{"role": "user", "content": "Weather?"}, {"role": "assistant", "content": [block]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": [{"type": "text", "text": "sunny"}]}, {"type": "text", "text": "Summarize"}]}]))
    assert second.status_code == 200
    messages = state.bodies[1]["messages"]
    assert json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]) == {"city": "York"}
    assert messages[2] == {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
    assert messages[3]["role"] == "user"


@pytest.mark.parametrize("reason,extra,expected,sequence", [("stop", {}, "end_turn", None), ("length", {}, "max_tokens", None), ("tool_calls", {}, "tool_use", None), ("stop", {"stop_sequence": "END"}, "stop_sequence", "END")])
async def test_stop_reasons(upstream_app, reason, extra, expected, sequence):
    app, state = upstream_app
    state.response = answer(choices=[{"message": {"content": "Hi"}, "finish_reason": reason, **extra}])
    data = (await call(app, request())).json()
    assert data["stop_reason"] == expected
    assert data["stop_sequence"] == sequence


def sse(obj):
    return b"data: " + json.dumps(obj).encode() + b"\r\n\r\n"


def events(text):
    return [json.loads(part.split("data: ", 1)[1]) for part in text.split("\n\n") if "data: " in part]


async def test_stream_split_parallel_tool_calls_usage_and_framing(upstream_app):
    app, state = upstream_app
    frames = [
        {"id": "c1", "model": "local", "choices": [{"delta": {"content": "Hi 🦙"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_", "function": {"name": "wea"}}, {"index": 1, "id": "call_2", "function": {"name": "clock"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "1", "function": {"name": "ther", "arguments": '{"city":'}}, {"index": 1, "function": {"arguments": "{}"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"York"}'}}]}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 24, "completion_tokens": 7}},
    ]
    wire = b"".join(sse(obj) for obj in frames) + b"data: [DONE]\r\n\r\n"
    state.chunks = [wire[i:i + 7] for i in range(0, len(wire), 7)]
    response = await call(app, request(stream=True))
    assert response.status_code == 200
    data = events(response.text)
    assert data[0]["type"] == "message_start"
    starts = [e["content_block"] for e in data if e["type"] == "content_block_start"]
    assert starts == [{"type": "text", "text": ""}, {"type": "tool_use", "id": "call_1", "name": "weather", "input": {}}, {"type": "tool_use", "id": "call_2", "name": "clock", "input": {}}]
    fragments = [e["delta"]["partial_json"] for e in data if e.get("delta", {}).get("type") == "input_json_delta"]
    assert fragments == ['{"city":', '"York"}', "{}"]
    assert data[-2] == {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"input_tokens": 24, "output_tokens": 7}}
    assert data[-1]["type"] == "message_stop"
    assert [e["index"] for e in data if e["type"] == "content_block_stop"] == [0, 1, 2]
    assert state.bodies[0]["stream_options"] == {"include_usage": True}
    assert state.closed == state.clients_closed == 1


@pytest.mark.parametrize("chunks", [[b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'], [b'data: {"error":{"message":"failed"}}\n\n'], [b'data: invalid\n\n']])
async def test_stream_failure_is_not_success(upstream_app, chunks):
    app, state = upstream_app
    state.chunks = chunks
    response = await call(app, request(stream=True))
    data = events(response.text)
    assert data[-1]["type"] == "error"
    assert not any(e["type"] == "message_stop" for e in data)
    assert state.closed == state.clients_closed == 1


async def test_x_api_key_same_rules_as_bearer(upstream_app, tmp_path):
    app, state = upstream_app
    store = ApiKeyStore.for_state_dir(tmp_path)
    _, key = store.create("cli")
    app.state.api_keys = store
    for headers, status in [({}, 401), ({"x-api-key": "bad"}, 401), ({"x-api-key": key}, 200), ({"Authorization": "Bearer " + key}, 200), ({"x-api-key": "admin"}, 200), ({"Authorization": "Bearer bad", "x-api-key": key}, 401)]:
        result = await call(app, request(), headers, peer="192.168.1.20")
        assert result.status_code == status
    assert len(state.bodies) == 3


async def test_invalid_requests_and_upstream_errors(upstream_app):
    app, state = upstream_app
    for fields in ({"max_tokens": True}, {"messages": []}, {"system": [{"type": "image"}]}, {"messages": [{"role": "user", "content": [{"type": "document"}]}]}, {"thinking": {"type": "enabled", "budget_tokens": 1024}}, {"context_management": {"edits": ["change history"]}}):
        result = await call(app, request(**fields))
        assert result.status_code == 400
        assert result.json()["type"] == "error"
    assert state.bodies == []
    state.response = {"error": {"message": "upstream refused"}}
    state.status = 429
    result = await call(app, request())
    assert result.status_code == 429
    assert result.json()["error"]["message"] == "upstream refused"


async def test_bad_tool_json_is_clear_error(upstream_app):
    app, state = upstream_app
    state.response = answer(choices=[{"message": {"tool_calls": [{"id": "1", "function": {"name": "f", "arguments": "{broken"}}]}, "finish_reason": "tool_calls"}])
    result = await call(app, request())
    assert result.status_code == 502
    assert result.json()["type"] == "error"


@pytest.mark.parametrize("kind,expected", [("auto", "auto"), ("any", "required"), ("none", "none")])
def test_tool_choices(kind, expected):
    assert to_openai(request(tool_choice={"type": kind}))["tool_choice"] == expected


def test_tool_error_and_output_schema_mapping():
    result = to_openai(request(system="System", thinking={"type": "disabled"}, output_config={"effort": "high", "format": {"type": "json_schema", "schema": {"type": "object"}}}, messages=[{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "is_error": True, "content": "Missing"}]}]))
    assert result["messages"][0] == {"role": "system", "content": "System"}
    assert result["messages"][1]["content"] == "Tool error: Missing"
    assert result["reasoning_effort"] == "none"
    assert result["response_format"]["json_schema"]["schema"] == {"type": "object"}


async def test_adapter_closes_unstarted_stream_on_disconnect():
    from starlette.background import BackgroundTask

    closed = []

    async def source():
        yield b"unused"

    response = await from_openai(StreamingResponse(source(), background=BackgroundTask(lambda: closed.append(True))), "local")

    async def send(message):
        raise RuntimeError("disconnected")

    async def receive():
        return {"type": "http.disconnect"}

    with pytest.raises(RuntimeError, match="disconnected"):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert closed == [True]


async def test_missing_upstream_usage_is_zero_not_fabricated():
    response = await from_openai(JSONResponse(answer(usage={})), "local")
    assert json.loads(response.body)["usage"] == {"input_tokens": 0, "output_tokens": 0}


@pytest.mark.parametrize("fields", [
    {"messages": [{"role": {}, "content": "x"}]},
    {"tool_choice": {"type": {}}}, {"thinking": {"type": {}}},
    {"temperature": True}, {"top_p": 2}, {"top_k": 1.5},
    {"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": {}, "data": ""}}]}]},
])
async def test_malformed_field_types_are_client_errors(upstream_app, fields):
    app, state = upstream_app
    result = await call(app, request(**fields))
    assert result.status_code == 400
    assert state.bodies == []


async def test_stream_text_length_finish_and_usage(upstream_app):
    app, state = upstream_app
    state.chunks = [sse({"choices": [{"delta": {"content": "Hello"}}]}),
                    sse({"choices": [{"delta": {}, "finish_reason": "length"}]}),
                    sse({"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 5}}),
                    b"data: [DONE]\n\n"]
    data = events((await call(app, request(stream=True))).text)
    assert data[-2]["delta"]["stop_reason"] == "max_tokens"
    assert data[-2]["usage"] == {"input_tokens": 4, "output_tokens": 5}
    assert data[-1]["type"] == "message_stop"


async def test_api_key_header_does_not_change_other_routes(upstream_app, tmp_path):
    app, _ = upstream_app
    store = ApiKeyStore.for_state_dir(tmp_path)
    _, key = store.create("cli")
    app.state.api_keys = store
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app, client=("192.168.1.20", 1234)), base_url="http://test") as client:
            response = await client.get("/v1/models", headers={"x-api-key": key})
    assert response.status_code == 401


async def test_stream_request_http_error_retains_status_and_closes(upstream_app):
    app, state = upstream_app
    state.status = 429
    state.chunks = [b'{"error":{"message":"rate limited"}}']
    result = await call(app, request(stream=True))
    assert result.status_code == 429
    assert result.json() == {"type": "error", "error": {"type": "rate_limit_error", "message": "rate limited"}}
    assert state.closed == state.clients_closed == 1
