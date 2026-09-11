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
