"""Synthetic catalogs prove recommendation identities are discovered, not shipped."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from arc_llama.compatibility_io import _fixture
from arc_llama.starter_discovery import (
    candidate,
    check_runtime,
    discover,
    options,
    validate,
)


def info(repo="new-publisher/never-seen-chat", size=2_000_000_000):
    return {
        "id": repo,
        "sha": "a" * 40,
        "private": False,
        "gated": False,
        "pipeline_tag": "text-generation",
        "tags": ["gguf"],
        "cardData": {"license": "mit"},
        "siblings": [{"rfilename": "chat-Q4_K_M.gguf", "size": size}],
    }


@pytest.fixture
def hub(monkeypatch):
    import arc_llama.starter_discovery as module

    rows = [info()]
    calls = []

    async def read(url, limit, **kw):
        calls.append((url, limit, kw))
        assert kw["public"] is True
        if "/resolve/" in url:
            assert "a" * 40 in url
            return _fixture("synthetic_arch")
        if "/api/models?" in url:
            return json.dumps(rows).encode()
        repo = url.split("/api/models/", 1)[1].split("?")[0].split("/revision/")[0]
        return json.dumps(next(row for row in rows if row["id"] == repo)).encode()

    monkeypatch.setattr(module, "_hub_read", read)
    return rows, calls


async def test_unseen_catalog_identity_changes_without_changing_code(hub):
    rows, calls = hub
    first = await discover(12288)
    assert (
        first
        and first.repo == "new-publisher/never-seen-chat"
        and first.license == "mit"
    )
    rows[:] = [info("another-publisher/brand-new-instruct")]
    second = await discover(12288)
    assert second and second.repo == rows[0]["id"] and first.id != second.id
    assert all(limit <= 2097152 for _, limit, _ in calls)
    assert all("search=" not in url for url, _, _ in calls)
    assert any("limit=32" in url and "filter=gguf" in url for url, _, _ in calls)


@pytest.mark.parametrize(
    "defect",
    [
        "unknown",
        "partial",
        "oversize",
        "private",
        "gated",
        "base",
        "embedding",
        "vision",
        "unsafe",
        "unknown_quant",
    ],
)
def test_ineligible_files_are_not_recommended(defect):
    row = info()
    if defect == "unknown":
        row["siblings"][0]["size"] = None
    if defect == "partial":
        row["siblings"][0]["rfilename"] = "chat-Q4_K_M-00001-of-00002.gguf"
    if defect == "oversize":
        row["siblings"][0]["size"] = 30_000_000_000
    if defect == "private":
        row["private"] = True
    if defect == "gated":
        row["gated"] = "auto"
    if defect == "base":
        row["id"] = "publisher/base-model"
    if defect == "embedding":
        row["pipeline_tag"] = "sentence-similarity"
    if defect == "vision":
        row["siblings"].append({"rfilename": "mmproj.gguf", "size": 2})
    if defect == "unsafe":
        row["siblings"][0]["rfilename"] = "../chat-Q4_K_M.gguf"
    if defect == "unknown_quant":
        row["siblings"][0]["rfilename"] = "chat.gguf"
    assert not options(row, 12288)


def test_nested_complete_split_and_rounding():
    row = info()
    row["siblings"] = [
        {"rfilename": f"nested/chat-Q4_K_M-{i:05}-of-00002.gguf", "size": 1_000_000_001}
        for i in (1, 2)
    ]
    result = options(row, 12288)
    assert len(result) == 1 and result[0][1] == 2_000_000_002
    row["siblings"][1]["size"] = None
    assert not options(row, 12288)


async def test_revision_size_license_and_architecture_come_from_publisher(
    hub, monkeypatch
):
    rows, _ = hub
    selected = await candidate(rows[0], 12288)
    assert selected and selected.architecture == "synthetic_arch"
    await validate(selected, 12288)
    rows[0]["sha"] = "b" * 40
    with pytest.raises(ValueError, match="accepted model files"):
        await validate(selected, 12288)
    rows[0]["sha"] = "a" * 40
    rows[0]["siblings"][0]["size"] += 1
    with pytest.raises(ValueError):
        await validate(selected, 12288)
    rows[0]["cardData"] = {}
    assert (await candidate(rows[0], 12288)).license == "Not stated"
    rows[0]["sha"] = "main"
    assert await candidate(rows[0], 12288) is None


async def test_broken_high_rank_does_not_hide_suitable_lower_rank(hub, monkeypatch):
    rows, _ = hub
    rows.insert(0, info("publisher/broken-chat"))
    rows[0]["siblings"] = []
    selected = await discover(12288)
    assert selected and selected.repo == rows[1]["id"]


async def test_runtime_rejection_blocks_weights(hub, monkeypatch):
    selected = await candidate(hub[0][0], 12288)
    monkeypatch.setattr(
        "arc_llama.starter_discovery.runtime_identity",
        lambda command: ("binary", "fingerprint"),
    )
    probe = AsyncMock(return_value="rejected")
    monkeypatch.setattr("arc_llama.starter_discovery.probe_architecture", probe)
    with pytest.raises(ValueError, match="No model weights were downloaded"):
        await check_runtime(selected, "binary")
    probe.return_value = "recognized"
    await check_runtime(selected, "binary")


async def test_empty_catalog_has_no_fallback(hub):
    hub[0].clear()
    assert await discover(12288) is None


async def test_discovery_is_single_flight_cached_retryable_and_discards_changed_gpu(
    tmp_path, monkeypatch
):
    from arc_llama.config import Config, GPUConfig
    from arc_llama.onboarding import FirstAnswerSetup

    rows = info()
    monkeypatch.setattr(
        "arc_llama.starter_discovery._hub_read",
        AsyncMock(return_value=_fixture("fixture")),
    )
    selected = await candidate(rows, 12288)
    gate = asyncio.Event()

    async def lookup(vram):
        await gate.wait()
        return selected

    lookup_mock = AsyncMock(side_effect=lookup)
    monkeypatch.setattr("arc_llama.onboarding.discover", lookup_mock)
    cfg = Config(gpus=[GPUConfig("gpu", 0, "battlemage", vram_mb=12288)])
    manager = FirstAnswerSetup(cfg, tmp_path / "config", None)
    manager.refresh()
    first = manager.discovery_task
    manager.refresh(force=True)
    assert manager.discovery_task is first
    await asyncio.sleep(0)
    cfg.gpus[0].vram_mb = 512
    gate.set()
    await first
    assert manager.public()["starter"] is None
    cfg.gpus[0].vram_mb = 12288
    manager.refresh(force=True)
    await manager.discovery_task
    assert manager.public()["starter"]["repo"] == rows["id"]
    manager.refresh()
    assert lookup_mock.await_count == 2
    lookup_mock.side_effect = OSError("offline")
    manager.refresh(force=True)
    await manager.discovery_task
    assert manager.public()["starter"] is None
    assert manager.discovery_status == "unavailable"
    await manager.shutdown()


async def test_discovery_shutdown_cancels_pending_io(tmp_path, monkeypatch):
    from arc_llama.config import Config, GPUConfig
    from arc_llama.onboarding import FirstAnswerSetup

    gate = asyncio.Event()
    monkeypatch.setattr(
        "arc_llama.onboarding.discover", AsyncMock(side_effect=lambda vram: None)
    )

    async def slow(vram):
        await gate.wait()

    monkeypatch.setattr("arc_llama.onboarding.discover", slow)
    cfg = Config(gpus=[GPUConfig("gpu", 0, "battlemage", vram_mb=12288)])
    manager = FirstAnswerSetup(cfg, tmp_path / "config", None)
    manager.refresh()
    await asyncio.sleep(0)
    await manager.shutdown()
    assert manager.discovery_task.cancelled()


def test_language_tag_does_not_make_a_base_model_chat_ready():
    row = info("publisher/base-model")
    row["tags"] = ["gguf", "it"]
    assert not options(row, 12288)
    row["id"] = "publisher/new-model-it-GGUF"
    assert options(row, 12288)


def test_pathological_catalog_file_sets_stay_bounded():
    row = info()
    row["siblings"][0]["rfilename"] = "chat-Q4_K_M-00001-of-99999.gguf"
    assert not options(row, 12288)
    row["siblings"] = [
        {"rfilename": f"chat-{i}-Q4_K_M.gguf", "size": 123} for i in range(513)
    ]
    assert not options(row, 12288)


async def test_deadline_cancels_inflight_catalog_requests(monkeypatch):
    gate = asyncio.Event()
    cancelled = asyncio.Event()
    monkeypatch.setattr(
        "arc_llama.starter_discovery.catalog",
        AsyncMock(return_value=[{"id": "publisher/slow-chat"}]),
    )

    async def slow(repo):
        try:
            await gate.wait()
        finally:
            cancelled.set()

    monkeypatch.setattr("arc_llama.starter_discovery.metadata", slow)
    with pytest.raises(asyncio.TimeoutError):
        await discover(12288, timeout=0.02)
    assert cancelled.is_set()
