"""Research-derived response boundaries through both real agent entry points."""
from __future__ import annotations

import copy
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from arc_llama.agent.interactive import InteractiveAgent
from arc_llama.agent.loop import run_agent
from arc_llama.agent.tools import TOOLS, ToolResult


def answer(content="done", **choice):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop", **choice}]}


def call(arguments='{"path":"marker.txt","content":"changed"}', **extra):
    return {"id": "one", "type": "function", "function": {"name": "write_file", "arguments": arguments}, **extra}


def tools(*calls, **choice):
    return answer(None, message={"content": None, "tool_calls": list(calls)}, finish_reason="tool_calls", **choice)


BAD_RESPONSES = [
    None, [], "not an envelope", 3,
    {}, {"choices": None}, {"choices": []}, {"choices": {}}, {"choices": "wrong"},
    {"choices": [None]}, {"choices": [1]}, {"choices": [{}]},
    answer(message=None), answer(message=[]), answer(message="wrong"),
    answer(content={"secret": "do not echo"}), answer(content=[None]),
    answer(message={"role": "user", "content": "untrusted"}),
    answer(finish_reason=[]),
    answer(message={"content": "plan", "tool_calls": {}}),
    answer(message={"content": "plan", "tool_calls": "wrong"}),
    tools(None), tools(call(function=None)), tools(call(function=[])),
    tools(call(function={"name": None, "arguments": "{}"})),
    tools(call(function={"name": [], "arguments": "{}"})),
    tools(call(function={"name": "", "arguments": "{}"})),
    tools(call(function={"name": "write_file"})),
    tools(call(arguments=None)), tools(call(arguments=[])),
    tools(call(type="custom")), tools(call(id={})),
    tools(call(), call()),
    # A valid first call must not run before a later structural failure is found.
    tools(call(), call(id="two", function=None)),
    {"error": {"message": "SECRET-MODEL-PAYLOAD"}, **answer()},
]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    payloads = []
    requests = []
    real_client = httpx.AsyncClient

    def respond(request):
        requests.append(json.loads(request.content))
        assert payloads, "unexpected extra model request"
        return httpx.Response(200, content=json.dumps(payloads.pop(0)).encode(),
                              headers={"content-type": "application/json"}, request=request)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw))

    async def execute(name, arguments, ctx):
        (tmp_path / "marker.txt").write_text("executed")
        return ToolResult("fixture result")

    execution = AsyncMock(side_effect=execute)
    monkeypatch.setattr(TOOLS, "execute", execution)
    approval = AsyncMock(return_value=True)
    plan_approval = AsyncMock(return_value=True)

    async def run(responses, *, interactive, planning=False, auto_confirm=False):
        payloads.extend(copy.deepcopy(responses))
        if interactive:
            agent = InteractiveAgent("fixture", "http://synthetic.invalid", tmp_path,
                                     plan_mode=planning, auto_confirm=auto_confirm, max_turns=3)
            try:
                events = [e async for e in agent.chat("synthetic task", confirm_callback=approval,
                                                     plan_callback=plan_approval)]
                return events, agent.conversation
            finally:
                await agent.close()
        events = [e async for e in run_agent("synthetic task", "fixture", "http://synthetic.invalid", tmp_path,
                                             plan_mode=planning, auto_confirm=auto_confirm, max_turns=3,
                                             confirm_callback=approval, plan_callback=plan_approval)]
        return events, None

    return run, execution, approval, plan_approval, requests, tmp_path


@pytest.mark.asyncio
@pytest.mark.parametrize("interactive", [False, True], ids=["single-task", "interactive"])
@pytest.mark.parametrize("planning", [False, True], ids=["execution", "planning"])
@pytest.mark.parametrize("payload", BAD_RESPONSES, ids=lambda _: None)
async def test_malformed_response_fails_before_any_action(harness, interactive, planning, payload):
    run, execute, approval, plan_approval, requests, root = harness
    events, conversation = await run([payload], interactive=interactive, planning=planning)
    assert events[-1]["type"] == "error"
    assert "model" in events[-1]["message"].lower()
    assert "SECRET-MODEL-PAYLOAD" not in json.dumps(events)
    assert not any(e["type"] in {"done", "plan", "confirm_required", "tool_call"} for e in events)
    execute.assert_not_awaited()
    approval.assert_not_awaited()
    plan_approval.assert_not_awaited()
    assert not (root / "marker.txt").exists()
    assert len(requests) == 1
    if conversation is not None:
        assert all(m.get("role") != "assistant" for m in conversation)


@pytest.mark.asyncio
@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.parametrize("arguments", ["", "null", "[]", "not json", '{"x":NaN}', '{"path":"one","path":"two"}'])
async def test_bad_arguments_request_resend_without_execution(harness, interactive, arguments):
    run, execute, approval, _, requests, _ = harness
    events, _ = await run([tools(call(arguments)), answer()], interactive=interactive)
    results = [e for e in events if e["type"] == "tool_result"]
    assert len(results) == 1 and results[0]["error"] is True
    assert "Not executed" in results[0]["content"]
    assert events[-1]["type"] == "done"
    execute.assert_not_awaited()
    approval.assert_not_awaited()
    assert len(requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.parametrize("refusal", ["SECRET-REFUSAL", 7])
async def test_refusal_never_becomes_a_success_or_tool_execution(harness, interactive, refusal):
    run, execute, approval, _, _, _ = harness
    payload = tools(call())
    payload["choices"][0]["message"]["refusal"] = refusal
    events, _ = await run([payload], interactive=interactive)
    assert events[-1]["type"] == "error"
    assert "SECRET-REFUSAL" not in json.dumps(events)
    execute.assert_not_awaited()
    approval.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("interactive", [False, True])
async def test_valid_call_denial_preserves_confirmation_behavior(harness, interactive):
    run, execute, approval, _, _, _ = harness
    approval.return_value = False
    events, _ = await run([tools(call()), answer()], interactive=interactive)
    approval.assert_awaited_once()
    execute.assert_not_awaited()
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["error"] is True and "denied" in result["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.parametrize("arguments", ["{}", {}])
async def test_explicit_empty_argument_object_remains_a_valid_call(harness, interactive, arguments):
    run, execute, approval, _, _, _ = harness
    events, _ = await run([tools(call(arguments)), answer()], interactive=interactive)
    assert events[-1]["type"] == "done"
    execute.assert_awaited_once()
    assert execute.call_args.args[1] == {}
    approval.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("interactive", [False, True])
async def test_generated_id_cannot_collide_with_a_model_supplied_id(harness, interactive):
    run, execute, _, _, requests, _ = harness
    missing = call()
    missing.pop("id")
    events, _ = await run([tools(missing, call(id="turn-0-call-0")), answer()],
                         interactive=interactive, auto_confirm=True)
    assert events[-1]["type"] == "done" and execute.await_count == 2
    message = next(m for m in requests[-1]["messages"] if m.get("tool_calls"))
    ids = [tc["id"] for tc in message["tool_calls"]]
    assert len(set(ids)) == 2
    assert ids == [m["tool_call_id"] for m in requests[-1]["messages"] if m["role"] == "tool"]
    assert len(requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.parametrize("finish", ["length", "content_filter", "unknown"])
async def test_incomplete_tool_call_never_executes(harness, interactive, finish):
    run, execute, approval, _, _, _ = harness
    payload = tools(call())
    payload["choices"][0]["finish_reason"] = finish
    events, _ = await run([payload, answer()], interactive=interactive)
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["error"] is True and "Not executed" in result["content"]
    execute.assert_not_awaited()
    approval.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.parametrize("planning", [False, True])
async def test_truncated_text_is_not_reported_as_completed_or_approved(harness, interactive, planning):
    run, execute, _, plan_approval, _, _ = harness
    events, _ = await run([answer("partial", finish_reason="length")], interactive=interactive, planning=planning)
    assert events[-1]["type"] == "error"
    assert not any(e["type"] in {"done", "plan"} for e in events)
    if not planning:
        assert any(e["type"] == "assistant" and e["content"] == "partial" for e in events)
    execute.assert_not_awaited()
    plan_approval.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("interactive", [False, True])
async def test_valid_plan_tools_and_missing_ids_remain_consistent(harness, interactive):
    run, execute, approval, plan_approval, requests, root = harness
    missing = call()
    missing.pop("id")
    events, _ = await run([answer("fixture plan"), tools(missing), answer("finished")],
                         interactive=interactive, planning=True)
    assert events[-1]["type"] == "done"
    execute.assert_awaited_once()
    approval.assert_awaited_once()
    plan_approval.assert_awaited_once_with("fixture plan")
    assert (root / "marker.txt").read_text() == "executed"
    assistant = next(m for m in requests[-1]["messages"] if m.get("tool_calls"))
    result = next(m for m in requests[-1]["messages"] if m["role"] == "tool")
    assert assistant["tool_calls"][0]["id"] == result["tool_call_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("interactive", [False, True])
async def test_choices_are_never_merged_and_optional_usage_cannot_crash(harness, interactive):
    run, execute, _, _, _, _ = harness
    payload = answer("selected")
    payload["choices"].append({"message": {"content": "other choice"}, "finish_reason": "stop"})
    payload["usage"] = ["malformed optional metric"]
    events, _ = await run([payload], interactive=interactive)
    assert [e["content"] for e in events if e["type"] == "assistant"] == ["selected"]
    assert events[-1]["type"] == "done"
    execute.assert_not_awaited()
