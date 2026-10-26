"""Validate model replies before they enter agent history or reach tools."""
from __future__ import annotations

import json
from typing import Any


class ModelResponseError(ValueError):
    """An invalid reply, described without echoing model payloads or credentials."""


def _invalid(path: str, expected: str) -> None:
    raise ModelResponseError(f"Invalid model response at {path}: expected {expected}. Retry the request.")


def completion_message(data: Any, *, call_prefix: str = "call") -> tuple[dict[str, Any], dict[str, Any]]:
    """Select the first choice, as before, and validate its complete call list.

    Choices are independent; no content, finish reason or tool call is taken
    from a peer. Missing call ids get matching assistant/tool ids in history.
    Validate every call before a caller can approve or execute the first one.
    Argument JSON is checked separately so the model can resend a bad call.
    """
    if not isinstance(data, dict):
        _invalid("response", "an object")
    if data.get("error") is not None:
        raise ModelResponseError("The model server returned an error response. Check the backend and retry.")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        _invalid("choices", "a nonempty list")
    choice = choices[0]
    if not isinstance(choice, dict):
        _invalid("choices[0]", "an object")
    if choice.get("finish_reason") is not None and not isinstance(choice["finish_reason"], str):
        _invalid("choices[0].finish_reason", "a string or null")
    message = choice.get("message")
    if not isinstance(message, dict):
        _invalid("choices[0].message", "an object")
    if message.get("role", "assistant") != "assistant":
        _invalid("choices[0].message.role", "assistant")
    if message.get("content") is not None and not isinstance(message["content"], str):
        _invalid("choices[0].message.content", "text or null")
    if message.get("refusal") is not None and not isinstance(message["refusal"], str):
        _invalid("choices[0].message.refusal", "text or null")
    if message.get("refusal"):
        raise ModelResponseError("The model declined the request.")
    calls = message.get("tool_calls")
    if calls is None:
        calls = []
    if not isinstance(calls, list):
        _invalid("choices[0].message.tool_calls", "a list or null")
    ids: set[str] = set()
    for i, call in enumerate(calls):
        path = f"choices[0].message.tool_calls[{i}]"
        if not isinstance(call, dict):
            _invalid(path, "an object")
        if call.get("type", "function") != "function":
            _invalid(path + ".type", "function")
        function = call.get("function")
        if not isinstance(function, dict):
            _invalid(path + ".function", "an object")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            _invalid(path + ".function.name", "a nonempty string")
        if not isinstance(function.get("arguments"), (str, dict)):
            _invalid(path + ".function.arguments", "JSON text or an object")
        call_id = call.get("id")
        if call_id is not None:
            if not isinstance(call_id, str) or not call_id.strip() or call_id in ids:
                _invalid(path + ".id", "a unique nonempty string")
            ids.add(call_id)
    message = dict(message)
    message.setdefault("role", "assistant")
    if calls:
        normalized = []
        for i, call in enumerate(calls):
            call_id = call.get("id")
            if call_id is None:
                call_id = f"{call_prefix}-{i}"
                while call_id in ids:
                    call_id += "-generated"
                ids.add(call_id)
            normalized.append({**call, "id": call_id})
        message["tool_calls"] = normalized
    return choice, message


def completion_problem(reason: Any) -> str | None:
    if reason in (None, "stop", "tool_calls", "function_call"):
        return None
    if reason == "length":
        return "The model reply hit the token limit before it was complete. Send the complete reply again."
    if reason == "content_filter":
        return "The model reply was stopped by a content filter."
    return "The model returned an unsupported completion state. Retry the request."


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate argument key")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError("non-finite argument value")


def parse_tool_call(tc: dict[str, Any], finish_reason: Any) -> tuple[dict[str, Any], str | None]:
    """Never replace unknown, ambiguous or incomplete arguments with a runnable {}."""
    function = tc.get("function") if isinstance(tc, dict) else None
    raw = function.get("arguments") if isinstance(function, dict) else None
    try:
        arguments = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite) if isinstance(raw, str) else raw
        # Also reject non-finite values in object-valued local-server arguments.
        if isinstance(arguments, dict):
            json.dumps(arguments, allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        arguments = None
    problem = completion_problem(finish_reason)
    if problem:
        return arguments if isinstance(arguments, dict) else {}, "Not executed: " + problem
    if not isinstance(arguments, dict):
        return {}, "Not executed: the tool arguments were not a valid JSON object. Send the call again."
    return arguments, None
