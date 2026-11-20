"""Messages API shapes over the existing OpenAI chat proxy."""
from __future__ import annotations

import json
import math
import uuid
from collections.abc import AsyncIterator
from typing import Any, NoReturn

from fastapi import HTTPException
from starlette.responses import JSONResponse, Response, StreamingResponse


def _bad(message: str) -> NoReturn:
    raise HTTPException(400, detail=message)


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        _bad(f"{field} must be a string")
    return value


def _parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list) or any(not isinstance(b, dict) for b in content):
        _bad("content must be a string or an array of blocks")
    return content


def _ordinary(block: dict[str, Any]) -> dict[str, Any]:
    kind = block.get("type")
    if kind == "text":
        return {"type": "text", "text": _string(block.get("text"), "text")}
    if kind == "image":
        source = block.get("source")
        if not isinstance(source, dict) or source.get("type") != "base64":
            _bad("Images require a base64 source")
        media = source.get("media_type")
        if media not in ("image/png", "image/jpeg", "image/gif", "image/webp"):
            _bad("Unsupported image media_type")
        data = _string(source.get("data"), "image data")
        return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}
    _bad(f"Content block {kind!r} cannot map to the chat proxy")
    raise AssertionError("unreachable")


def to_openai(body: dict[str, Any]) -> dict[str, Any]:
    model = _string(body.get("model"), "model")
    maximum = body.get("max_tokens")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        _bad("max_tokens must be a positive integer")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        _bad("messages must be a nonempty array")
    harmless = {"metadata", "cache_control", "service_tier", "betas", "speed", "inference_geo"}
    supported = {"model", "max_tokens", "messages", "system", "tools", "tool_choice",
                 "temperature", "top_p", "top_k", "stop_sequences", "stream", "thinking", "output_config"}
    unsupported = set(body) - harmless - supported
    if unsupported:
        _bad(f"Cannot map request field(s): {', '.join(sorted(unsupported))}")
    payload: dict[str, Any] = {"model": model, "max_tokens": maximum, "messages": []}
    if "system" in body:
        blocks = _parts(body["system"])
        if any(b.get("type") != "text" for b in blocks):
            _bad("system supports text blocks only")
        payload["messages"].append({"role": "system", "content": "\n\n".join(
            _string(b.get("text"), "system text") for b in blocks
        )})
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
            _bad("messages require user or assistant roles")
        role = message["role"]
        normal, calls, results = [], [], []
        for block in _parts(message.get("content")):
            kind = block.get("type")
            if kind == "tool_use":
                if role != "assistant" or not isinstance(block.get("input"), dict):
                    _bad("tool_use requires an assistant role and JSON object input")
                calls.append({"id": _string(block.get("id"), "tool_use.id"), "type": "function",
                              "function": {"name": _string(block.get("name"), "tool_use.name"),
                                           "arguments": json.dumps(block["input"])}})
            elif kind == "tool_result":
                if role != "user":
                    _bad("tool_result requires a user role")
                parts = [_ordinary(b) for b in _parts(block.get("content", ""))]
                value: Any = "\n".join(b["text"] for b in parts) if all(b["type"] == "text" for b in parts) else parts
                # is_error is descriptive metadata; preserve it in the tool output.
                if block.get("is_error"):
                    if isinstance(value, str):
                        value = "Tool error: " + value
                    else:
                        value.insert(0, {"type": "text", "text": "Tool error:"})
                results.append({"role": "tool", "tool_call_id": _string(block.get("tool_use_id"), "tool_result.tool_use_id"), "content": value})
            else:
                normal.append(_ordinary(block))
        # OpenAI requires tool results directly after the assistant tool calls,
        # before an accompanying user message.
        payload["messages"].extend(results)
        if normal or calls or not results:
            entry: dict[str, Any] = {"role": role, "content": normal or None}
            if calls:
                entry["tool_calls"] = calls
            payload["messages"].append(entry)
    for field in ("temperature", "top_p", "top_k"):
        if field in body:
            value = body[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                _bad(f"{field} must be a finite number")
            if (field == "top_k" and (not isinstance(value, int) or value < 0)) or (field != "top_k" and not 0 <= value <= 1):
                _bad(f"{field} is outside its supported range")
            payload[field] = value
    if "stop_sequences" in body:
        stops = body["stop_sequences"]
        if not isinstance(stops, list) or any(not isinstance(s, str) for s in stops):
            _bad("stop_sequences must be an array of strings")
        payload["stop"] = stops
    if "stream" in body and not isinstance(body["stream"], bool):
        _bad("stream must be a boolean")
    payload["stream"] = body.get("stream", False)
    if payload["stream"]:
        payload["stream_options"] = {"include_usage": True}
    if "tools" in body:
        tools = body["tools"]
        if not isinstance(tools, list):
            _bad("tools must be an array")
        payload["tools"] = []
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type", "custom") != "custom":
                _bad("Only client-defined tools can map to the chat proxy")
            if not isinstance(tool.get("input_schema"), dict):
                _bad("tools require an input_schema object")
            fn = {"name": _string(tool.get("name"), "tool name"), "parameters": tool["input_schema"]}
            if "description" in tool:
                fn["description"] = _string(tool["description"], "tool description")
            if "strict" in tool:
                fn["strict"] = tool["strict"]
            payload["tools"].append({"type": "function", "function": fn})
    if "tool_choice" in body:
        choice = body["tool_choice"]
        if not isinstance(choice, dict):
            _bad("tool_choice must be an object")
        kind = choice.get("type")
        if kind in ("auto", "none", "any"):
            payload["tool_choice"] = "required" if kind == "any" else kind
        elif kind == "tool":
            payload["tool_choice"] = {"type": "function", "function": {"name": _string(choice.get("name"), "tool_choice.name")}}
        else:
            _bad("Unsupported tool_choice type")
        if "disable_parallel_tool_use" in choice:
            if not isinstance(choice["disable_parallel_tool_use"], bool):
                _bad("disable_parallel_tool_use must be a boolean")
            payload["parallel_tool_calls"] = not choice["disable_parallel_tool_use"]
    output = body.get("output_config")
    if output is not None:
        if not isinstance(output, dict) or set(output) - {"effort", "format"}:
            _bad("output_config supports effort and JSON schema format only")
        if "effort" in output:
            if output["effort"] not in ("minimal", "low", "medium", "high", "xhigh", "max"):
                _bad("Unsupported output_config.effort")
            payload["reasoning_effort"] = output["effort"]
        if "format" in output:
            fmt = output["format"]
            if not isinstance(fmt, dict) or fmt.get("type") != "json_schema" or not isinstance(fmt.get("schema"), dict):
                _bad("output_config.format requires a JSON schema")
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "messages_output", "schema": fmt["schema"], "strict": True}}
    thinking = body.get("thinking")
    if thinking is not None:
        if not isinstance(thinking, dict) or thinking.get("type") not in ("disabled", "adaptive", "enabled"):
            _bad("thinking.type must be enabled, adaptive or disabled")
        payload["chat_template_kwargs"] = {"enable_thinking": thinking["type"] in ("adaptive", "enabled")}
        if thinking["type"] == "disabled":
            payload["reasoning_effort"] = "none"
    return payload


def _usage(obj: dict[str, Any]) -> dict[str, int]:
    value = obj.get("usage") or {}
    if not isinstance(value, dict):
        raise ValueError("Expected an object for token usage")
    def count(key: str) -> int:
        n = value.get(key, 0)
        return n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else 0
    return {"input_tokens": count("prompt_tokens"), "output_tokens": count("completion_tokens")}


def _stop(choice: dict[str, Any]) -> tuple[str, str | None]:
    reason = choice.get("finish_reason")
    if reason == "length":
        return "max_tokens", None
    if reason in {"tool_calls", "function_call"}:
        return "tool_use", None
    matched = choice.get("stop_sequence") or choice.get("stop_string") or choice.get("stopping_word")
    return ("stop_sequence", matched) if matched else ("end_turn", None)


def _envelope(obj: dict[str, Any], model: str) -> dict[str, Any]:
    return {"id": obj.get("id") or "msg_" + uuid.uuid4().hex, "type": "message", "role": "assistant",
            "model": obj.get("model") or model, "content": [], "stop_reason": None,
            "stop_sequence": None, "usage": _usage(obj)}


def _tool(call: dict[str, Any]) -> dict[str, Any]:
    fn = call.get("function") or {}
    data = json.loads(fn.get("arguments") or "{}")
    if not isinstance(data, dict):
        raise ValueError("Tool arguments must be a JSON object")
    if not isinstance(call.get("id"), str) or not call["id"] or not isinstance(fn.get("name"), str) or not fn["name"]:
        raise ValueError("Tool calls require an ID and name")
    return {"type": "tool_use", "id": call["id"], "name": fn["name"], "input": data}


def error_response(message: str, status: int = 502) -> JSONResponse:
    kind = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
            404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error", 529: "overloaded_error"}.get(status, "api_error")
    return JSONResponse({"type": "error", "error": {"type": kind, "message": message}}, status_code=status)


async def _close_response(response: StreamingResponse) -> None:
    iterator_close = getattr(response.body_iterator, "aclose", None)
    try:
        if iterator_close is not None:
            await iterator_close()
    finally:
        if response.background is not None:
            await response.background()


async def from_openai(response: Response, model: str) -> Response:
    if isinstance(response, StreamingResponse) and response.status_code < 400:
        return _stream_response(response, model)
    if isinstance(response, StreamingResponse):
        raw = b""
        try:
            async for chunk in response.body_iterator:
                raw += chunk.encode() if isinstance(chunk, str) else bytes(chunk)
                if len(raw) > 1_048_576:
                    return error_response("Chat proxy error body exceeded one MiB", response.status_code)
        finally:
            await _close_response(response)
        response = Response(raw, status_code=response.status_code)
    try:
        obj = json.loads(bytes(response.body))
        if response.status_code >= 400:
            error = obj.get("error") or obj.get("detail") or "Chat proxy request failed"
            return error_response(str(error.get("message", error) if isinstance(error, dict) else error), response.status_code)
        choice = obj["choices"][0]
        message = choice["message"]
        result = _envelope(obj, model)
        if message.get("content"):
            if not isinstance(message["content"], str):
                raise ValueError("Expected string assistant content")
            result["content"].append({"type": "text", "text": message["content"]})
        result["content"].extend(_tool(call) for call in message.get("tool_calls") or [])
        result["stop_reason"], result["stop_sequence"] = _stop({**obj, **choice})
        return JSONResponse(result)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        return error_response(f"Invalid chat proxy response: {exc}")


def _event(kind: str, **body: Any) -> bytes:
    return (f"event: {kind}\ndata: " + json.dumps({"type": kind, **body}) + "\n\n").encode()


async def _chunks(response: StreamingResponse) -> AsyncIterator[dict[str, Any] | None]:
    buffer = b""
    async for chunk in response.body_iterator:
        buffer += chunk.encode() if isinstance(chunk, str) else bytes(chunk)
        buffer = buffer.replace(b"\r\n", b"\n")
        while b"\n\n" in buffer:
            raw, buffer = buffer.split(b"\n\n", 1)
            data = b"\n".join(line[5:].lstrip() for line in raw.splitlines() if line.startswith(b"data:"))
            if not data:
                continue
            if data.strip() == b"[DONE]":
                yield None
                return
            value = json.loads(data)
            if not isinstance(value, dict):
                raise ValueError("Expected an object in the chat event stream")
            yield value
    # The proxy recognizes a complete DONE line even when its final blank line
    # arrives in a later network chunk, and stops forwarding at that point.
    if buffer.strip() == b"data: [DONE]":
        yield None


def _stream_response(response: StreamingResponse, model: str) -> StreamingResponse:
    closed = False

    async def close() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        await _close_response(response)

    async def stream() -> AsyncIterator[bytes]:
        started = False
        text_open = False
        index = 0
        tools: dict[int, dict[str, Any]] = {}
        finish: dict[str, Any] | None = None
        usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        done = False
        try:
            async for obj in _chunks(response):
                if obj is None:
                    done = True
                    break
                if "error" in obj:
                    raise ValueError(str(obj["error"]))
                if obj.get("usage"):
                    usage = _usage(obj)
                if not started:
                    started = True
                    yield _event("message_start", message=_envelope(obj, model))
                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    if not text_open:
                        text_open = True
                        yield _event("content_block_start", index=index, content_block={"type": "text", "text": ""})
                    yield _event("content_block_delta", index=index, delta={"type": "text_delta", "text": delta["content"]})
                for call in delta.get("tool_calls") or []:
                    slot = tools.setdefault(call.get("index", 0), {"id": "", "function": {"name": "", "arguments": ""}, "fragments": []})
                    slot["id"] += call.get("id") or ""
                    fn = call.get("function") or {}
                    slot["function"]["name"] += fn.get("name") or ""
                    if "arguments" in fn:
                        slot["function"]["arguments"] += fn["arguments"]
                        slot["fragments"].append(fn["arguments"])
                if choice.get("finish_reason"):
                    finish = choice
            if not done or finish is None:
                raise ValueError("Chat stream ended before its completion marker")
            if text_open:
                yield _event("content_block_stop", index=index)
                index += 1
            # Buffer tool metadata until complete so split names/IDs and parallel
            # calls become stable Messages blocks; preserve argument fragments.
            for call in tools.values():
                block = _tool(call)
                yield _event("content_block_start", index=index, content_block={**block, "input": {}})
                for fragment in call["fragments"]:
                    yield _event("content_block_delta", index=index, delta={"type": "input_json_delta", "partial_json": fragment})
                yield _event("content_block_stop", index=index)
                index += 1
            reason, sequence = _stop(finish)
            yield _event("message_delta", delta={"stop_reason": reason, "stop_sequence": sequence}, usage=usage)
            yield _event("message_stop")
        except Exception as exc:
            yield _event("error", error={"type": "api_error", "message": str(exc)})
        finally:
            await close()

    class MessagesStream(StreamingResponse):
        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                await close()

    return MessagesStream(stream(), media_type="text/event-stream")
