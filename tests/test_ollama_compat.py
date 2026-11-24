import asyncio
import json

import pytest
from starlette.responses import Response, StreamingResponse

from arc_llama.server import _ollama_to_openai, _openai_response_as_ollama


def test_ollama_options_translate_to_openai_payload() -> None:
    payload = _ollama_to_openai(
        {
            "model": "demo",
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 12, "stop": ["END"]},
        },
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
        Response(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "hello"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        ),
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


def test_ollama_generate_forwards_prompt_to_chat_backend(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import arc_llama.server as server_mod
    from arc_llama.config import Config, PathsConfig

    async def proxy(request, target_path):
        payload = json.loads(await request.body())
        assert target_path == "/v1/chat/completions"
        assert payload["messages"] == [{"role": "user", "content": "2 plus 2"}]
        assert payload["max_tokens"] == 5
        assert "prompt" not in payload
        return Response(json.dumps({"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}]}).encode())

    monkeypatch.setattr(server_mod, "_proxy_post", proxy)
    cfg = Config(paths=PathsConfig(state_dir=str(tmp_path)))
    cfg.tune.auto = False
    with TestClient(server_mod.create_app(cfg, plugins=[])) as client:
        response = client.post(
            "/api/generate",
            json={
                "model": "demo",
                "prompt": "2 plus 2",
                "stream": False,
                "options": {"num_predict": 5},
            },
        )
        assert response.status_code == 200
        assert response.json()["response"] == "4"


# Endpoint tests stop at a controlled OpenAI upstream; they exercise the public
# Ollama request/response boundary without loading a model or using a GPU.
@pytest.fixture
def ollama_client(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient
    from helpers import ConfigRouter

    import arc_llama.server as server
    from arc_llama.config import Config, ModelConfig, PathsConfig

    path = tmp_path / 'demo.gguf'
    path.write_bytes(b'GGUF')
    cfg = Config(paths=PathsConfig(state_dir=str(tmp_path / 'state')),
                 models=[ModelConfig('demo', str(path), 18080, 'gpu', aliases=['alias'], recipe={'ctx': 4096}),
                         ModelConfig('cold', str(tmp_path / 'cold.gguf'), 18081, 'gpu')])
    cfg.tune.auto = False
    wire = SimpleNamespace(calls=[], status=200, chunks=None, closed=False,
                           response={'choices': [{'message': {'role': 'assistant', 'content': 'hello'}, 'finish_reason': 'stop'}]})
    async def upstream(request, target_path):
        wire.calls.append((target_path, json.loads(await request.body())))
        if wire.chunks is not None:
            async def source():
                try:
                    for chunk in wire.chunks:
                        yield chunk
                finally:
                    wire.closed = True
            return StreamingResponse(source(), status_code=wire.status)
        return Response(json.dumps(wire.response).encode(), status_code=wire.status)
    monkeypatch.setattr(server, 'Router', ConfigRouter)
    monkeypatch.setattr(server, '_proxy_post', upstream)
    app = server.create_app(cfg, plugins=[])
    with TestClient(app) as client:
        yield client, wire, app


PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a5l0AAAAASUVORK5CYII='
TOOL = {'type': 'function', 'function': {'name': 'weather', 'parameters': {'type': 'object'}}}


def test_chat_images_translate_to_openai_content_parts(ollama_client):
    client, wire, _ = ollama_client
    response = client.post('/api/chat', json={'model': 'demo', 'stream': False,
        'messages': [{'role': 'user', 'content': 'Describe this', 'images': [PNG]}]})
    assert response.status_code == 200
    target, body = wire.calls[0]
    assert target == '/v1/chat/completions'
    assert body['messages'][0]['content'] == [
        {'type': 'text', 'text': 'Describe this'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + PNG}},
    ]
    assert 'images' not in body['messages'][0]


def test_tools_and_tool_results_round_trip(ollama_client):
    client, wire, _ = ollama_client
    wire.response = {'choices': [{'message': {'content': None, 'tool_calls': [
        {'id': 'backend-id', 'type': 'function', 'function': {'name': 'weather', 'arguments': '{"city":"Tokyo"}'}}]},
        'finish_reason': 'tool_calls'}]}
    response = client.post('/api/chat', json={'model': 'demo', 'stream': False, 'tools': [TOOL], 'messages': [
        {'role': 'user', 'content': 'Weather?'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'weather', 'arguments': {'city': 'Toronto'}}}]},
        {'role': 'tool', 'tool_name': 'weather', 'content': '11 C'},
    ]})
    assert response.status_code == 200
    body = wire.calls[0][1]
    assert body['tools'] == [TOOL]
    call = body['messages'][1]['tool_calls'][0]
    assert json.loads(call['function']['arguments']) == {'city': 'Toronto'}
    assert body['messages'][2] == {'role': 'tool', 'content': '11 C', 'tool_call_id': call['id']}
    assert response.json()['message']['tool_calls'] == [{'function': {'name': 'weather', 'arguments': {'city': 'Tokyo'}}}]
    assert response.json()['done_reason'] == 'stop'


@pytest.mark.parametrize('endpoint', ['chat', 'generate'])
@pytest.mark.parametrize('fmt', ['json', {'type': 'object', 'properties': {'age': {'type': 'integer'}}}])
def test_format_json_and_schema_reach_the_upstream(ollama_client, endpoint, fmt):
    client, wire, _ = ollama_client
    body = {'model': 'demo', 'stream': False, 'format': fmt}
    body.update({'messages': [{'role': 'user', 'content': 'JSON please'}]} if endpoint == 'chat' else {'prompt': 'JSON please'})
    assert client.post('/api/' + endpoint, json=body).status_code == 200
    actual = wire.calls[0][1]['response_format']
    assert actual == ({'type': 'json_object'} if fmt == 'json' else
                      {'type': 'json_schema', 'json_schema': {'name': 'ollama_response', 'schema': fmt}})


def test_generate_preserves_system_prompt_and_images(ollama_client):
    client, wire, _ = ollama_client
    response = client.post('/api/generate', json={'model': 'demo', 'prompt': 'What is here?',
        'system': 'Be brief', 'images': [PNG], 'stream': False})
    assert response.json()['response'] == 'hello'
    target, body = wire.calls[0]
    assert target == '/v1/chat/completions'
    assert body['messages'][0] == {'role': 'system', 'content': 'Be brief'}
    assert body['messages'][1]['content'][0]['text'] == 'What is here?'
    assert body['messages'][1]['content'][1]['image_url']['url'].endswith(PNG)


def test_streaming_tools_reassemble_fragments_and_preserve_ndjson(ollama_client):
    client, wire, _ = ollama_client
    events = [
        {'choices': [{'delta': {'content': 'Checking ', 'reasoning_content': 'Plan'}}]},
        {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'name': 'wea', 'arguments': '{"city":'}}]}}]},
        {'choices': [{'delta': {'tool_calls': [{'index': 1, 'function': {'name': 'clock', 'arguments': '{}'}},
                                             {'index': 0, 'function': {'name': 'ther', 'arguments': '"東京"}'}}]}}]},
        {'choices': [{'delta': {}, 'finish_reason': 'tool_calls'}]},
        {'choices': [], 'usage': {'prompt_tokens': 7, 'completion_tokens': 9}},
    ]
    data = ''.join('data: ' + json.dumps(event, ensure_ascii=False) + '\r\n\r\n' for event in events).encode()
    data += b'data: [DONE]\r\n\r\n'
    wire.chunks = [data[i:i+7] for i in range(0, len(data), 7)]
    response = client.post('/api/chat', json={'model': 'demo', 'messages': [{'role': 'user', 'content': 'Weather?'}], 'tools': [TOOL]})
    assert response.headers['content-type'].startswith('application/x-ndjson')
    rows = [json.loads(line) for line in response.text.splitlines()]
    assert rows[0]['message']['content'] == 'Checking '
    assert rows[0]['message']['thinking'] == 'Plan'
    assert rows[1]['message']['tool_calls'] == [
        {'function': {'name': 'weather', 'arguments': {'city': '東京'}}},
        {'function': {'name': 'clock', 'arguments': {}}},
    ]
    assert rows[1]['done'] is False
    assert rows[-1]['done'] is True and rows[-1]['done_reason'] == 'stop'
    assert rows[-1]['prompt_eval_count'] == 7 and rows[-1]['eval_count'] == 9
    assert wire.closed


@pytest.mark.parametrize('stream', [False, True])
def test_invalid_upstream_tool_arguments_are_clear_errors(ollama_client, stream):
    client, wire, _ = ollama_client
    call = {'index': 0, 'function': {'name': 'weather', 'arguments': '{broken'}}
    if stream:
        wire.chunks = [('data: ' + json.dumps({'choices': [{'delta': {'tool_calls': [call]}}]}) + '\n\n').encode(), b'data: [DONE]\n\n']
    else:
        wire.response = {'choices': [{'message': {'tool_calls': [call]}, 'finish_reason': 'tool_calls'}]}
    response = client.post('/api/chat', json={'model': 'demo', 'stream': stream, 'messages': [{'role': 'user', 'content': 'hi'}]})
    if stream:
        rows = [json.loads(line) for line in response.text.splitlines()]
        assert 'error' in rows[-1] and not any(row.get('done') for row in rows)
        assert wire.closed
    else:
        assert response.status_code == 502 and 'error' in response.json()


@pytest.mark.parametrize('endpoint', ['chat', 'generate'])
def test_client_lifetime_and_load_options_are_accepted_without_forwarding(ollama_client, endpoint):
    client, wire, _ = ollama_client
    body = {'model': 'demo', 'stream': False, 'keep_alive': '5m', 'think': 'high', 'options': {
        'num_ctx': 8192, 'num_gpu': -1, 'num_thread': 8, 'num_batch': 512,
        'use_mmap': True, 'use_mlock': False, 'main_gpu': 0, 'numa': False,
        'low_vram': False, 'f16_kv': True, 'logits_all': False, 'vocab_only': False,
        'num_gqa': 1, 'temperature': 0.2, 'num_predict': 42,
    }}
    body.update({'messages': [{'role': 'user', 'content': 'hi'}]} if endpoint == 'chat' else {'prompt': 'hi'})
    response = client.post('/api/' + endpoint, json=body)
    assert response.status_code == 200
    assert wire.calls == [('/v1/chat/completions', {
        'model': 'demo', 'stream': False, 'messages': [{'role': 'user', 'content': 'hi'}],
        'temperature': 0.2, 'max_tokens': 42,
    })]


@pytest.mark.parametrize('extra', [{'format': 'xml'}, {'options': {'unknown_sampling': 1}},
                                 {'options': []}, {'suffix': 'fill'}, {'context': [1]}, {'tools': [{}]}])
def test_unmappable_request_fields_return_errors(ollama_client, extra):
    client, wire, _ = ollama_client
    response = client.post('/api/chat', json={'model': 'demo', 'messages': [{'role': 'user', 'content': 'hi'}], **extra})
    assert response.status_code == 400 and response.json()['error']
    assert wire.calls == []


@pytest.mark.parametrize('message', [
    {'role': 'user', 'content': 'hi', 'images': ['invalid base64!']},
    {'role': [], 'content': 'hi'},
    {'role': 'tool', 'content': 'unmatched', 'tool_name': 'unknown'},
])
def test_invalid_images_roles_and_unmatched_tool_results_are_rejected(ollama_client, message):
    client, wire, _ = ollama_client
    response = client.post('/api/chat', json={'model': 'demo', 'messages': [message]})
    assert response.status_code == 400 and response.json()['error']
    assert not wire.calls


def test_show_uses_known_library_metadata(ollama_client, monkeypatch):
    import arc_llama.server as server

    client, _, _ = ollama_client
    monkeypatch.setattr(server, 'read_gguf_meta', lambda path: {'architecture': 'llama', 'context_length': 8192})
    info = client.post('/api/show', json={'model': 'alias'}).json()
    assert info['details'] == {'format': 'gguf', 'family': 'llama', 'families': ['llama']}
    assert info['model_info']['llama.context_length'] == 8192
    assert info['parameters'] == 'num_ctx 4096'
    assert 'completion' in info['capabilities']
    assert 'template' not in info and 'license' not in info
    assert client.post('/api/show', json={'model': 'missing'}).status_code == 404
    assert client.post('/api/show', json={'model': 'demo', 'verbose': True}).status_code == 400


def test_ps_returns_ready_local_models_without_inventing_vram_or_expiry(ollama_client):
    from types import SimpleNamespace

    client, _, app = ollama_client
    app.state.router._servers = {'demo': SimpleNamespace(is_running=True, ready=True),
                                'cold': SimpleNamespace(is_running=True, ready=False)}
    models = client.get('/api/ps').json()['models']
    assert len(models) == 1 and models[0]['model'] == 'demo'
    assert models[0]['size'] == 4
    assert not {'size_vram', 'expires_at', 'digest'} & models[0].keys()


@pytest.mark.parametrize('chunks', [
    [b'data: {"error":{"message":"backend failed"}}\n\n'],
    [b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'],
])
def test_stream_errors_and_truncation_do_not_claim_completion(ollama_client, chunks):
    client, wire, _ = ollama_client
    wire.chunks = chunks
    response = client.post('/api/chat', json={'model': 'demo', 'messages': [{'role': 'user', 'content': 'hi'}]})
    rows = [json.loads(line) for line in response.text.splitlines()]
    assert rows[-1]['error']
    assert not any(row.get('done') for row in rows)
    assert wire.closed


def test_generate_rejects_raw_templates_instead_of_dropping_them(ollama_client):
    client, wire, _ = ollama_client
    response = client.post('/api/generate', json={'model': 'demo', 'prompt': 'hello', 'raw': True})
    assert response.status_code == 400 and 'raw' in response.json()['error']
    assert not wire.calls
