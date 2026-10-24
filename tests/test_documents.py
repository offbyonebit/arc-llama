"""Documents work type: office tools, page images, and the vision-only gate."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from helpers import ConfigRouter as FakeRouter

import arc_llama.server as server_mod
from arc_llama.agent import documents
from arc_llama.agent.loop import run_agent
from arc_llama.agent.tools import TOOLS, ToolContext
from arc_llama.config import Config, ModelConfig, ServerConfig
from arc_llama.server import create_app


def _ctx(root: Path) -> ToolContext:
    return ToolContext(root=root, client=MagicMock())


async def _call(name: str, arguments: dict, root: Path):
    return await documents.documents_registry().execute(name, arguments, _ctx(root))


def test_documents_registry_has_no_shell_and_coding_registry_has_no_documents() -> None:
    names = set(documents.documents_registry().list_tools())
    assert {"read_document", "create_document", "create_spreadsheet", "create_presentation", "view_page"} <= names
    assert "run_command" not in names and "write_file" not in names and "apply_patch" not in names
    assert "create_document" not in TOOLS.list_tools()


def test_create_tools_ask_before_writing_and_reading_does_not() -> None:
    registry = documents.documents_registry()
    for name in ("create_document", "create_spreadsheet", "create_presentation"):
        assert registry.requires_confirmation(name)
    assert not registry.requires_confirmation("read_document")
    assert not registry.requires_confirmation("view_page")


@pytest.mark.asyncio
async def test_word_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("docx")
    blocks = [
        {"type": "heading", "text": "Plan", "level": 1},
        {"type": "paragraph", "text": "Hello there."},
        {"type": "bullets", "items": ["one", "two"]},
        {"type": "table", "rows": [["a", "b"], ["1", "2"]]},
    ]
    made = await _call("create_document", {"path": "out/plan.docx", "blocks": blocks}, tmp_path)
    assert not made.error, made.content
    read = await _call("read_document", {"path": "out/plan.docx"}, tmp_path)
    assert not read.error
    assert "# Plan" in read.content and "Hello there." in read.content
    assert "- one" in read.content and "| a | b |" in read.content


@pytest.mark.asyncio
async def test_excel_round_trip_keeps_formulas(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    sheets = [{"name": "Totals", "rows": [["item", "cost"], ["a", 2], ["b", 3], ["sum", "=SUM(B2:B3)"]]}]
    made = await _call("create_spreadsheet", {"path": "t.xlsx", "sheets": sheets}, tmp_path)
    assert not made.error, made.content
    read = await _call("read_document", {"path": "t.xlsx"}, tmp_path)
    assert "## Sheet: Totals" in read.content and "=SUM(B2:B3)" in read.content


@pytest.mark.asyncio
async def test_powerpoint_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("pptx")
    slides = [{"title": "Quarterly review"}, {"title": "Results", "bullets": ["up", "down"], "notes": "say it slowly"}]
    made = await _call("create_presentation", {"path": "deck.pptx", "slides": slides}, tmp_path)
    assert not made.error, made.content
    read = await _call("read_document", {"path": "deck.pptx"}, tmp_path)
    assert "## Slide 2" in read.content and "up" in read.content and "Notes: say it slowly" in read.content


@pytest.mark.asyncio
async def test_paths_cannot_leave_the_folder_or_change_type(tmp_path: Path) -> None:
    outside = await _call("create_document", {"path": "../x.docx", "blocks": [{"type": "paragraph", "text": "x"}]}, tmp_path)
    assert outside.error and not (tmp_path.parent / "x.docx").exists()
    wrong = await _call("create_document", {"path": "x.txt", "blocks": [{"type": "paragraph", "text": "x"}]}, tmp_path)
    assert wrong.error and ".docx" in wrong.content


@pytest.mark.asyncio
async def test_view_page_returns_an_image_for_image_files(tmp_path: Path) -> None:
    (tmp_path / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    result = await _call("view_page", {"path": "pic.png"}, tmp_path)
    assert not result.error and result.images and result.images[0].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_loop_sends_page_images_back_as_a_user_message(tmp_path: Path) -> None:
    (tmp_path / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")

    def reply(data: dict) -> MagicMock:
        r = MagicMock()
        r.json.return_value = data
        r.raise_for_status.return_value = None
        return r

    call = {"id": "c1", "function": {"name": "view_page", "arguments": json.dumps({"path": "pic.png"})}}
    posts = [
        reply({"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [call]}}]}),
        reply({"choices": [{"message": {"role": "assistant", "content": "A picture."}}]}),
    ]
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = AsyncMock(side_effect=posts)
    with patch("arc_llama.agent.loop.httpx.AsyncClient", return_value=client):
        events = [e async for e in run_agent("look", "m", "http://x", tmp_path, work_type="documents")]
    assert events[-1]["type"] == "done"
    second = client.post.await_args_list[1].kwargs["json"]
    image_turns = [
        m for m in second["messages"]
        if m["role"] == "user" and isinstance(m["content"], list) and any(p.get("type") == "image_url" for p in m["content"])
    ]
    assert len(image_turns) == 1
    assert [m["role"] for m in second["messages"][-3:-1]] == ["tool", "user"]
    names = {t["function"]["name"] for t in second["tools"]}
    assert "view_page" in names and "run_command" not in names


# ---- server gate -----------------------------------------------------------


def _app(tmp_path, monkeypatch, models):
    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    cfg = Config(server=ServerConfig(admin_token=""), models=models)
    cfg.paths.state_dir = str(tmp_path / "state")
    return create_app(cfg)


def _model(name: str, vision: bool) -> ModelConfig:
    recipe = {"mmproj": "/m/mmproj.gguf"} if vision else {}
    return ModelConfig(name=name, path=f"/m/{name}.gguf", port=9000, gpu_pci_slot="", recipe=recipe)


def test_documents_need_a_vision_model(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch, [_model("text-only", False), _model("seer", True)])
    with TestClient(app) as client:
        r = client.post("/v1/agent", json={"model": "text-only", "task": "x", "work_type": "documents"})
        assert r.status_code == 400 and "vision" in r.json()["detail"]
        r = client.post("/v1/agent", json={"model": "unknown", "task": "x", "work_type": "documents"})
        assert r.status_code == 400 and "vision" in r.json()["detail"]


def test_documents_need_the_builtin_agent_and_a_known_type(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch, [_model("seer", True)])
    with TestClient(app) as client:
        r = client.post("/v1/agent", json={"model": "seer", "task": "x", "work_type": "documents", "harness": "opencode"})
        assert r.status_code == 400 and "built-in" in r.json()["detail"]
        r = client.post("/v1/agent", json={"model": "seer", "task": "x", "work_type": "nope"})
        assert r.status_code == 400 and "work_type" in r.json()["detail"]


def test_info_lists_work_types_and_vision_models(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    app = _app(tmp_path, monkeypatch, [_model("text-only", False), _model("seer", True)])
    with TestClient(app) as client:
        info = client.get("/v1/agent/info").json()
    assert [t["id"] for t in info["work_types"]] == ["code", "documents"]
    assert info["vision_models"] == ["seer"]


def test_a_projector_given_as_a_launch_flag_counts_as_vision() -> None:
    flagged = ModelConfig(name="g", path="/m/g.gguf", port=9000, gpu_pci_slot="", recipe={"extra_flags": ["--mmproj", "/m/p.gguf"]})
    other = ModelConfig(name="t", path="/m/t.gguf", port=9001, gpu_pci_slot="", recipe={"extra_flags": ["--jinja"]})
    assert "vision" in server_mod.model_capabilities(flagged)
    assert "vision" not in server_mod.model_capabilities(other)


@pytest.mark.asyncio
async def test_view_page_renders_word_files_with_wordcraft_when_present(tmp_path: Path) -> None:
    pytest.importorskip("docx")
    if documents._wordcraft() is None:
        pytest.skip("WordCraft is not installed")
    await _call("create_document", {"path": "a.docx", "blocks": [{"type": "paragraph", "text": "Hello"}]}, tmp_path)
    result = await _call("view_page", {"path": "a.docx"}, tmp_path)
    assert not result.error, result.content
    assert result.images and result.images[0].startswith("data:image/png;base64,")
