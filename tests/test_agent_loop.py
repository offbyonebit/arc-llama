"""Tests for the agent loop."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arc_llama.agent.loop import run_agent


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    (tmp_path / "hello.txt").write_text("world\n")
    return tmp_path


async def collect_events(task: str, responses: list[dict], tmp_root: Path, **kwargs):
    """Run the agent loop against a mocked httpx client and collect events."""
    events = []

    def make_response(data: dict) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = data
        resp.raise_for_status.return_value = None
        return resp

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(side_effect=[make_response(r) for r in responses])

    with patch("arc_llama.agent.loop.httpx.AsyncClient", return_value=mock_client):
        async for event in run_agent(
            task=task,
            model="test-model",
            base_url="http://localhost:11437",
            root=tmp_root,
            **kwargs,
        ):
            events.append(event)

    return events


@pytest.mark.asyncio
async def test_agent_returns_direct_answer(tmp_root: Path) -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                    }
                }
            ]
        }
    ]
    events = await collect_events("say hi", responses, tmp_root)

    assert events[0]["type"] == "status"
    assert any(e["type"] == "assistant" and e["content"] == "Hello!" for e in events)
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_agent_reads_file_and_reports_result(tmp_root: Path) -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({"path": "hello.txt"}),
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The file says world.",
                    }
                }
            ]
        },
    ]
    events = await collect_events("read hello.txt", responses, tmp_root)

    tool_call = next(e for e in events if e["type"] == "tool_call")
    assert tool_call["name"] == "read_file"
    assert tool_call.get("id")
    tool_result = next(e for e in events if e["type"] == "tool_result")
    assert tool_result["content"] == "world\n"
    assert not tool_result["error"]
    assert tool_result.get("id") == tool_call["id"]
    assert any(e["type"] == "assistant" and "world" in e["content"] for e in events)


@pytest.mark.asyncio
async def test_write_requires_confirmation(tmp_root: Path) -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": json.dumps({"path": "x.txt", "content": "x"}),
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I won't write the file without approval.",
                    }
                }
            ]
        },
    ]
    callback = AsyncMock(return_value=False)
    events = await collect_events(
        "write a file", responses, tmp_root, auto_confirm=False, confirm_callback=callback
    )

    assert any(e["type"] == "confirm_required" and e["tool"] == "write_file" for e in events)
    callback.assert_awaited_once()
    tool_result = next((e for e in events if e["type"] == "tool_result"), None)
    assert tool_result is not None
    assert tool_result["error"]
    assert "denied" in tool_result["content"].lower()
    assert not (tmp_root / "x.txt").exists()
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_auto_confirm_allows_write(tmp_root: Path) -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": json.dumps({"path": "x.txt", "content": "x"}),
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Done.",
                    }
                }
            ]
        },
    ]
    events = await collect_events("write a file", responses, tmp_root, auto_confirm=True)

    tool_call = next(e for e in events if e["type"] == "tool_call")
    tool_result = next(e for e in events if e["type"] == "tool_result")
    assert not tool_result["error"]
    assert tool_result.get("id") == tool_call.get("id")
    assert (tmp_root / "x.txt").read_text() == "x"
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_manual_confirmation_allows_write(tmp_root: Path) -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": json.dumps({"path": "y.txt", "content": "y"}),
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Done.",
                    }
                }
            ]
        },
    ]
    callback = AsyncMock(return_value=True)
    events = await collect_events(
        "write a file", responses, tmp_root, auto_confirm=False, confirm_callback=callback
    )

    assert any(e["type"] == "confirm_required" and e["tool"] == "write_file" for e in events)
    callback.assert_awaited_once()
    tool_result = next(e for e in events if e["type"] == "tool_result")
    assert not tool_result["error"]
    assert (tmp_root / "y.txt").read_text() == "y"
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_agent_reads_pdf_and_reports_text(tmp_root: Path) -> None:
    (tmp_root / "doc.pdf").write_bytes(b"%PDF-fake")
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_pdf",
                                    "arguments": json.dumps({"path": "doc.pdf"}),
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {"filename": "doc.pdf", "text": "PDF extracted text"},
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The PDF says extracted text.",
                    }
                }
            ]
        },
    ]
    events = await collect_events("read the pdf", responses, tmp_root)

    tool_call = next(e for e in events if e["type"] == "tool_call")
    assert tool_call["name"] == "read_pdf"
    tool_result = next(e for e in events if e["type"] == "tool_result")
    assert tool_result["content"] == "PDF extracted text"
    assert not tool_result["error"]
    assert any(e["type"] == "assistant" and "extracted text" in e["content"] for e in events)


@pytest.mark.asyncio
async def test_plan_mode_auto_confirm(tmp_root: Path) -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Plan: read file then write result.",
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Done.",
                    }
                }
            ]
        },
    ]
    events = await collect_events(
        "do a thing", responses, tmp_root, plan_mode=True, auto_confirm=True
    )

    plan_event = next(e for e in events if e["type"] == "plan")
    assert "Plan:" in plan_event["content"]
    assert any(e["type"] == "assistant" and e["content"] == "Done." for e in events)
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_plan_mode_manual_approval(tmp_root: Path) -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Plan: read file then write result.",
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Done.",
                    }
                }
            ]
        },
    ]
    plan_callback = AsyncMock(return_value=True)
    events = await collect_events(
        "do a thing", responses, tmp_root, plan_mode=True, plan_callback=plan_callback
    )

    plan_event = next(e for e in events if e["type"] == "plan")
    assert "Plan:" in plan_event["content"]
    plan_callback.assert_awaited_once()
    assert any(e["type"] == "assistant" and e["content"] == "Done." for e in events)
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_plan_mode_manual_deny(tmp_root: Path) -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Plan: read file then write result.",
                    }
                }
            ]
        },
    ]
    plan_callback = AsyncMock(return_value=False)
    events = await collect_events(
        "do a thing", responses, tmp_root, plan_mode=True, plan_callback=plan_callback
    )

    assert any(e["type"] == "plan" for e in events)
    plan_callback.assert_awaited_once()
    assert not any(e["type"] == "tool_call" for e in events)
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_agent_reports_actual_latest_context_usage(tmp_root: Path) -> None:
    response = {"choices": [{"message": {"role": "assistant", "content": "Done"}}],
                "usage": {"prompt_tokens": 1200, "completion_tokens": 37}}
    events = await collect_events("say hi", [response], tmp_root)
    assert [event for event in events if event["type"] == "usage"] == [
        {"type": "usage", "prompt_tokens": 1200, "completion_tokens": 37}]


@pytest.mark.asyncio
async def test_agent_does_not_invent_usage_when_runtime_omits_it(tmp_root: Path) -> None:
    response = {"choices": [{"message": {"role": "assistant", "content": "Done"}}],
                "usage": {"prompt_tokens": -1, "completion_tokens": 37}}
    events = await collect_events("say hi", [response], tmp_root)
    assert not any(event["type"] == "usage" for event in events)


@pytest.mark.parametrize("timings,expected", [
    ({"predicted_n": 100, "predicted_ms": 2000}, 50.0),
    ({"predicted_per_second": 42.5}, 42.5),
    ({"predicted_per_second": -1}, None),
    ({}, None),
])
async def test_agent_reports_only_backend_measured_generation_speed(tmp_root, timings, expected):
    response = {"choices": [{"message": {"role": "assistant", "content": "Synthetic answer"}}],
                "usage": {"prompt_tokens": 4000, "completion_tokens": 100}, "timings": timings}
    events = await collect_events("Synthetic task", [response], tmp_root)
    measured = [event for event in events if event["type"] == "performance"]
    assert measured == ([] if expected is None else [{"type": "performance", "generation_tok_s": expected}])
