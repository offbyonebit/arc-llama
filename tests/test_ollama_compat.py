import asyncio
import json

from starlette.responses import Response, StreamingResponse

from arc_llama.server import _ollama_to_openai, _openai_response_as_ollama


def test_ollama_options_translate_to_openai_payload() -> None:
    payload = _ollama_to_openai(
        {"model": "demo", "stream": False, "options": {"temperature": 0.2, "num_predict": 12, "stop": ["END"]}},
        messages=[{"role": "user", "content": "hi"}],
    )
    assert payload == {
        "model": "demo",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "temperature": 0.2,
        "stop": ["END"],
        "max_tokens": 12,
    }


def test_openai_chat_response_converts_to_ollama_shape() -> None:
    response = _openai_response_as_ollama(
        Response(json.dumps({"choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}]}).encode()),
        "demo",
        generate=False,
    )
    body = json.loads(response.body)
    assert body["message"] == {"role": "assistant", "content": "hello"}
    assert body["done"] is True


def test_openai_generate_response_converts_to_ollama_shape() -> None:
    response = _openai_response_as_ollama(
        Response(json.dumps({"choices": [{"text": "hello", "finish_reason": "stop"}]}).encode()),
        "demo",
        generate=True,
    )
    body = json.loads(response.body)
    assert body["response"] == "hello"
    assert body["done_reason"] == "stop"


def test_openai_stream_converts_to_ndjson() -> None:
    async def source():
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield b"data: [DONE]\n\n"

    response = _openai_response_as_ollama(StreamingResponse(source()), "demo", generate=False)

    async def collect():
        return b"".join([chunk async for chunk in response.body_iterator])  # type: ignore[attr-defined]

    lines = [json.loads(line) for line in asyncio.run(collect()).splitlines()]
    assert lines[0]["message"]["content"] == "hi"
    assert lines[-1]["done"] is True


def test_ollama_generate_forwards_prompt_to_completion_backend(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import arc_llama.server as server_mod
    from arc_llama.config import Config, PathsConfig

    async def proxy(request, target_path):
        payload = json.loads(await request.body())
        assert target_path == "/v1/completions"
        assert payload["prompt"] == "2 plus 2"
        assert payload["max_tokens"] == 5
        assert "messages" not in payload
        return Response(json.dumps({"choices": [{"text": "4", "finish_reason": "stop"}]}).encode())

    monkeypatch.setattr(server_mod, "_proxy_post", proxy)
    cfg = Config(paths=PathsConfig(state_dir=str(tmp_path)))
    cfg.tune.auto = False
    with TestClient(server_mod.create_app(cfg, plugins=[])) as client:
        response = client.post("/api/generate", json={
            "model": "demo", "prompt": "2 plus 2", "stream": False,
            "options": {"num_predict": 5},
        })
        assert response.status_code == 200
        assert response.json()["response"] == "4"
