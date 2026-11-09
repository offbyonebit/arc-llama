"""Bounded public catalog discovery: policy, not a built-in model inventory.

Popularity orders the search, not a quality benchmark. Fit is an estimate.
Only public, ungated text chat candidates with complete sizes are suggested.
The accepted commit is carried through to the integrity-checked downloader.
One pinned model is the fallback when the catalog fails or finds nothing,
and the only choice with ARC_LLAMA_STARTER_CATALOG=off, for repeatable runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote, urlencode

from huggingface_hub import constants, hf_hub_url

from arc_llama.compatibility_io import _hub_read
from arc_llama.gguf_meta import gguf_shards, split_info
from arc_llama.model_compatibility import (
    header_architecture,
    probe_architecture,
    runtime_identity,
)
from arc_llama.model_library import fit_badge, group_repo_files
from arc_llama.recipes import KVCacheType

_REPO = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+\Z")
_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_CHAT = re.compile(
    r"(?:^|[^a-z])(instruct(?:ion)?|chat|conversational)(?:$|[^a-z])", re.I
)
# Repo names, tags or files that mark a GGUF llama.cpp will not chat with:
# typed decision models answer only on /v1/systemone, classifiers and
# rerankers return scores, a draft file only feeds speculative decoding.
# GGUF headers mark decision models only after the vocabulary, too far in
# for the bounded header read, so the Hub's names and tags are the check.
_NOT_CHAT = re.compile(r"decision|classifi|rerank|system-?one|draft", re.I)
_LIMIT = 32
# Pinned to the commit used in the Windows 1.1 validation. It goes through the
# same fit, license and architecture checks as a catalog pick.
PINNED_STARTER = ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "9217f5db79a29953eb74d5343926648285ec7e67")


@dataclass(frozen=True)
class Starter:
    id: str
    name: str
    repo: str
    file: str
    size_mb: int
    revision: str
    license: str
    architecture: str
    size_bytes: int


async def catalog() -> list[dict[str, Any]]:
    query = urlencode(
        {
            "filter": "gguf",
            "pipeline_tag": "text-generation",
            "sort": "downloads",
            "direction": -1,
            "limit": _LIMIT,
        }
    )
    data = json.loads(
        await _hub_read(
            f"{constants.ENDPOINT}/api/models?{query}", 2_097_152, public=True
        )
    )
    if not isinstance(data, list):
        raise ValueError("Invalid public model catalog")
    return [row for row in data[:_LIMIT] if isinstance(row, dict)]


async def metadata(repo: str, revision: str | None = None) -> dict[str, Any]:
    if not _REPO.fullmatch(repo) or (
        revision is not None and not _REVISION.fullmatch(revision)
    ):
        raise ValueError("Invalid catalog repository or revision")
    suffix = f"/revision/{revision}" if revision else ""
    data = json.loads(
        await _hub_read(
            f"{constants.ENDPOINT}/api/models/{repo}{suffix}?blobs=true",
            2_097_152,
            public=True,
        )
    )
    if not isinstance(data, dict):
        raise ValueError("Invalid model metadata")
    return data


def options(
    info: dict[str, Any], vram_mb: int, kv_type: KVCacheType = KVCacheType.Q8_0
) -> list[tuple[str, int, str]]:
    """Reject unknown/partial shard sizes rather than underestimating the download."""
    repo = info.get("id", "")
    tags = info.get("tags") or []
    card = info.get("cardData") or {}
    if not isinstance(repo, str) or not _REPO.fullmatch(repo):
        return []
    if info.get("private") or info.get("gated") not in (False, None):
        return []
    chat = _CHAT.search(str([repo, tags])) or re.search(
        r"(?:-|_)it(?:-|_|$)", repo, re.I
    )
    if info.get("pipeline_tag") != "text-generation" or not chat:
        return []
    if _NOT_CHAT.search(str([repo, tags])):
        return []
    if not isinstance(card, dict) or not isinstance(tags, list):
        return []
    siblings = info.get("siblings", [])
    if not isinstance(siblings, list) or len(siblings) > 512:
        return []
    files = {}
    for item in siblings:
        if not isinstance(item, dict):
            continue
        name, size = item.get("rfilename"), item.get("size")
        if not isinstance(name, str) or len(name) > 512:
            continue
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
            continue
        split = split_info(name)
        if split and split[1] > 128:
            continue
        files[name] = (
            size
            if isinstance(size, int) and not isinstance(size, bool) and size > 0
            else None
        )
    grouped, vision = group_repo_files(list(files.items()), vram_mb, kv_type)
    if vision or any(
        "image-text" in str(tag) or "multimodal" in str(tag) for tag in tags
    ):
        return []
    result = []
    for option in grouped:
        if not re.fullmatch(r"Q[4-8](?:_[A-Z0-9]+)*", option.quant):
            continue
        if _NOT_CHAT.search(option.file):
            continue
        shards = gguf_shards(option.file)
        prefix = PurePosixPath(option.file).parent
        names = [str(prefix / shard.name) for shard in shards]
        if not names or any(not files.get(name) for name in names):
            continue
        total = sum(files[name] or 0 for name in names)
        if fit_badge(math.ceil(total / 1_048_576), vram_mb, kv_type) == "fits":
            result.append((option.file, total, option.quant))
    # Standard 4-bit first for a modest first download, then larger fitting
    # variants within that quantization. The repository order comes from the Hub.
    return sorted(result, key=lambda row: (int(row[2][1]), -row[1], row[0]))


async def candidate(
    info: dict[str, Any], vram_mb: int, kv_type: KVCacheType = KVCacheType.Q8_0
) -> Starter | None:
    revision = info.get("sha")
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        return None
    choices = options(info, vram_mb, kv_type)
    if not choices:
        return None
    file, size, _ = choices[0]
    repo = info["id"]
    header = await _hub_read(
        hf_hub_url(repo, file, revision=revision),
        262_144,
        range_header=True,
        public=True,
    )
    architecture = header_architecture(header)
    if architecture is None:
        return None
    card = info.get("cardData") or {}
    license_name = card.get("license")
    if not isinstance(license_name, str):
        license_name = next(
            (
                tag[8:]
                for tag in info.get("tags", [])
                if isinstance(tag, str) and tag.startswith("license:")
            ),
            "Not stated",
        )
    identity = hashlib.sha256(
        repr((repo, file, revision, size, vram_mb)).encode()
    ).hexdigest()[:24]
    return Starter(
        identity,
        repo.split("/")[-1],
        repo,
        file,
        math.ceil(size / 1_048_576),
        revision,
        license_name[:120],
        architecture,
        size,
    )


async def discover(
    vram_mb: int, *, timeout: float = 25, kv_type: KVCacheType = KVCacheType.Q8_0
) -> Starter | None:
    """At most 32 repo checks, four concurrent, under one cancellable deadline."""

    async def search() -> Starter | None:
        rows = await catalog()
        checked = 0
        slots = asyncio.Semaphore(4)

        async def inspect(row: dict[str, Any]) -> Starter | None:
            nonlocal checked
            repo = row.get("id")
            if not isinstance(repo, str) or not _REPO.fullmatch(repo):
                return None
            if row.get("private") or row.get("gated") not in (False, None):
                return None
            async with slots:
                try:
                    info = await metadata(repo)
                    checked += 1
                    return await candidate(info, vram_mb, kv_type)
                except Exception:
                    return None  # One unavailable publisher cannot break the catalog.

        tasks = [asyncio.create_task(inspect(row)) for row in rows]
        try:
            # Consume in popularity order, cancelling unnecessary lower ranks.
            for task in tasks:
                result = await task
                if result:
                    return result
            if rows and not checked:
                raise ValueError("Catalog publishers were unavailable")
            return None
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def pinned() -> Starter | None:
        return await candidate(await metadata(*PINNED_STARTER), vram_mb, kv_type)

    if os.environ.get("ARC_LLAMA_STARTER_CATALOG", "on").lower() == "off":
        return await asyncio.wait_for(pinned(), timeout=timeout)
    try:
        found = await asyncio.wait_for(search(), timeout=timeout)
    except Exception as catalog_error:
        try:
            found = await asyncio.wait_for(pinned(), timeout=12)
        except Exception:
            found = None
        if found is None:
            raise catalog_error
        return found
    if found is None:
        try:
            found = await asyncio.wait_for(pinned(), timeout=12)
        except Exception:
            found = None
    return found


async def validate(
    starter: Starter, vram_mb: int, kv_type: KVCacheType = KVCacheType.Q8_0
) -> None:
    # Exact accepted commit, never current HEAD. Recompute file size/eligibility.
    info = await asyncio.wait_for(metadata(starter.repo, starter.revision), timeout=12)
    if info.get("sha") != starter.revision or not any(
        file == starter.file and size == starter.size_bytes
        for file, size, _ in options(info, vram_mb, kv_type)
    ):
        raise ValueError(
            "The accepted model files cannot be confirmed to fit. Refresh the recommendation or use Discover."
        )


async def check_runtime(starter: Starter, command: str) -> None:
    identity = await asyncio.to_thread(runtime_identity, command)
    if (
        identity is None
        or await probe_architecture(*identity, starter.architecture) != "recognized"
    ):
        raise ValueError(
            "This runtime could not confirm the model architecture. No model weights were downloaded. Choose another model or check your runtime in System."
        )


def source(starter: Starter) -> str:
    return (
        f"https://huggingface.co/{starter.repo}/tree/{quote(starter.revision, safe='')}"
    )
