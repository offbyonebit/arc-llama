"""Independent regressions for research-derived search, MCP and hub gaps."""
from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_mcp_client import _install_session, _tool
from test_repo_map import _word_index

from arc_llama.agent.mcp_client import MCPClientManager, MCPServerConfig
from arc_llama.agent.repo_map import build_repo_map
from arc_llama.agent.tools import TOOLS
from arc_llama.models import HFModelSpec, download_from_hf


def test_search_rechecks_unselected_equal_stat_edits(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    root = tmp_path / "project"
    root.mkdir()
    source = root / "a.py"
    source.write_text('def alpha():\n    return "aaaaaa"\n')
    (root / "b.py").write_text('def beta():\n    return "bbbbbb"\n')
    before = source.stat()
    index = _word_index(tmp_path, monkeypatch)

    class Embedder:
        def embed(self, texts):
            return [np.array([1., 0.] if "secret" in t else [0., 1.]) for t in texts]

    monkeypatch.setattr(index, "_embedder_instance", lambda: Embedder())
    index.index(root)
    source.write_text('def alpha():\n    return "secret"\n')
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert index.search(root, "secret", top_k=1)[0]["path"] == "a.py"


def test_search_and_repo_map_skip_file_links_outside_root(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def outside_canary():\n    return 1\n")
    try:
        (root / "linked.py").symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks unavailable")
    (root / "inside.py").write_text("def inside():\n    return 2\n")
    index = _word_index(tmp_path, monkeypatch)
    assert "outside_canary" not in build_repo_map(root)
    assert all("outside_canary" not in h["snippet"] for h in index.search(root, "outside_canary"))


@pytest.mark.parametrize("vector", [[float("nan"), 1.], [float("inf"), 0.], [0., 0.]])
def test_bad_embedding_does_not_replace_complete_index(tmp_path, monkeypatch, vector):
    np = pytest.importorskip("numpy")
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.py").write_text("def original():\n    return 1\n")
    index = _word_index(tmp_path, monkeypatch)
    index.index(root)
    before = index._index_path().read_bytes()

    class Invalid:
        def embed(self, texts):
            return [np.array(vector) for _ in texts]

    monkeypatch.setattr(index, "_embedder_instance", lambda: Invalid())
    with pytest.raises(ValueError, match="vectors"):
        index.index(root)
    assert index._index_path().read_bytes() == before


def test_source_swap_to_link_between_scan_and_read_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    path = root / "a.py"
    path.write_text("def inside():\n    return 1\n")
    outside = tmp_path / "outside.py"
    outside.write_text("def private_canary():\n    return 2\n")
    index = _word_index(tmp_path, monkeypatch)
    original = index._indexable_files

    def swapping_scan(project):
        files = original(project)
        path.unlink()
        try:
            path.symlink_to(outside)
        except OSError:
            pytest.skip("file symlinks unavailable")
        return files

    monkeypatch.setattr(index, "_indexable_files", swapping_scan)
    assert index.index(root)["chunks"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["summary only", '{"answer": 42}'])
async def test_structured_value_survives_text_without_duplication(monkeypatch, text):
    result = SimpleNamespace(isError=False, structuredContent={"answer": 42},
                             content=[SimpleNamespace(type="text", text=text)])
    _install_session(monkeypatch, {None: SimpleNamespace(tools=[_tool("s")], nextCursor=None)}, result)
    manager = MCPClientManager([MCPServerConfig(name="both", command="unused")])
    await manager.start()
    try:
        out = await TOOLS.get("mcp_both_s").handler({}, None)
        assert "42" in out.content and text in out.content
        assert out.content.count('"answer"') == 1
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_changed_catalog_replaces_tools_and_invalidates_old_proxy(monkeypatch):
    pages = {None: SimpleNamespace(tools=[_tool("old")], nextCursor=None)}
    _install_session(monkeypatch, pages)
    manager = MCPClientManager([MCPServerConfig(name="refresh", command="unused")])
    await manager.start()
    try:
        old = TOOLS.get("mcp_refresh_old")
        replacement = _tool("new")
        replacement.inputSchema = {"type": "object", "required": ["new_field"]}
        pages[None] = SimpleNamespace(tools=[replacement], nextCursor=None)
        session = manager._sessions[0]
        await session.kwargs["message_handler"](SimpleNamespace(root=SimpleNamespace(method="notifications/tools/list_changed")))
        assert TOOLS.get("mcp_refresh_old") is None
        assert (await old.handler({}, None)).error
        await asyncio.gather(*manager._refresh_tasks.values())
        assert TOOLS.get("mcp_refresh_new").parameters == replacement.inputSchema
        assert manager._registered_tools == ["mcp_refresh_new"]
    finally:
        await manager.stop()
    assert TOOLS.get("mcp_refresh_new") is None


@pytest.mark.asyncio
async def test_shutdown_cancels_pending_catalog_refresh(monkeypatch):
    _install_session(monkeypatch, {None: SimpleNamespace(tools=[_tool("old")], nextCursor=None)})
    manager = MCPClientManager([MCPServerConfig(name="pending", command="unused")])
    await manager.start()
    session = manager._sessions[0]

    async def hanging_list(*args):
        await asyncio.Event().wait()

    session.list_tools = hanging_list
    await session.kwargs["message_handler"](SimpleNamespace(root=SimpleNamespace(method="notifications/tools/list_changed")))
    tasks = list(manager._refresh_tasks.values())
    await asyncio.sleep(0)
    await manager.stop()
    assert all(task.done() for task in tasks)
    assert not manager._refresh_tasks and TOOLS.get("mcp_pending_old") is None


def _hub(monkeypatch, payloads, *, corrupt=False, missing_digest=False):
    calls = []
    siblings = [SimpleNamespace(rfilename=name, size=len(data),
        lfs={"sha256": hashlib.sha256(data).hexdigest()} if not missing_digest else {}, blob_id="b" * 40)
        for name, data in payloads.items()]

    class API:
        def __init__(self, **kwargs):
            pass

        def model_info(self, repo, files_metadata):
            return SimpleNamespace(sha="a" * 40, siblings=siblings)

        def list_repo_files(self, repo, revision):
            assert revision == "a" * 40
            return list(payloads)

    def transfer(**kwargs):
        calls.append(kwargs)
        assert kwargs["revision"] == "a" * 40
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"X" * len(payloads[kwargs["filename"]]) if corrupt else payloads[kwargs["filename"]])
        return str(path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=API, hf_hub_download=transfer))
    return calls


def test_every_model_shard_uses_one_revision(tmp_path, monkeypatch):
    first, second = "model-00001-of-00002.gguf", "model-00002-of-00002.gguf"
    payloads = {first: b"GGUF first shard", second: b"GGUF second shard"}
    calls = _hub(monkeypatch, payloads)
    result = download_from_hf(HFModelSpec(repo="synthetic/model", file=first, quant=None), target_dir=tmp_path)
    assert result == tmp_path / first
    assert len(calls) == 2 and {c["revision"] for c in calls} == {"a" * 40}
    assert all((tmp_path / name).read_bytes() == data for name, data in payloads.items())
    assert not list(tmp_path.glob(".arc-download-*"))


def test_checksum_failure_preserves_existing_model_and_never_publishes(tmp_path, monkeypatch):
    name = "model.gguf"
    (tmp_path / name).write_bytes(b"previous complete model")
    _hub(monkeypatch, {name: b"new complete model"}, corrupt=True)
    with pytest.raises(ValueError, match="checksum"):
        download_from_hf(HFModelSpec(repo="synthetic/model", file=name, quant=None), target_dir=tmp_path)
    assert (tmp_path / name).read_bytes() == b"previous complete model"
    assert not list(tmp_path.glob(".arc-download-*"))


def test_missing_digest_is_refused_before_any_transfer(tmp_path, monkeypatch):
    calls = _hub(monkeypatch, {"model.gguf": b"synthetic"}, missing_digest=True)
    with pytest.raises(ValueError, match="verifiable content digest"):
        download_from_hf(HFModelSpec(repo="synthetic/model", file="model.gguf", quant=None), target_dir=tmp_path)
    assert not calls


def test_missing_companion_shard_is_refused(tmp_path, monkeypatch):
    first = "model-00001-of-00002.gguf"
    calls = _hub(monkeypatch, {first: b"synthetic"})
    with pytest.raises(FileNotFoundError, match="Missing model shard"):
        download_from_hf(HFModelSpec(repo="synthetic/model", file=first, quant=None), target_dir=tmp_path)
    assert not calls


def test_model_discovery_does_not_see_unverified_download_staging(tmp_path):
    from arc_llama.config import Config, PathsConfig
    from arc_llama.models import discover_ggufs

    hidden = tmp_path / ".arc-download-owned"
    hidden.mkdir()
    (hidden / "unverified.gguf").write_bytes(b"not verified")
    visible = tmp_path / "model.gguf"
    visible.write_bytes(b"published fixture")
    found = discover_ggufs(Config(paths=PathsConfig(models_dir=str(tmp_path))))
    assert found == [visible.resolve()]
