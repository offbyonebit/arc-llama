"""Discover shelves: a short, hand-picked set of GGUF repos sized for the user's GPU.

First-time users rarely know a model name to search for. The dashboard opens
Discover on these shelves instead of an empty search box. Each entry names the
lab that trained the model; the repo may be a conversion by someone else
(``converter``). Live figures (downloads, license, file sizes) come from the
Hub when the Discover page asks for them, never at startup.

Review this list with each release: repos get renamed and better models ship.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import asdict, dataclass
from typing import Any

from arc_llama.model_library import QuantOption, _api, group_repo_files


@dataclass(frozen=True)
class Featured:
    id: str
    repo: str
    name: str
    author: str
    """Hugging Face namespace of the lab that trained the model (avatar + publisher row)."""
    params: str
    description: str


PUBLISHERS: dict[str, dict[str, str]] = {
    "google": {"name": "Google", "families": "Gemma"},
    "Qwen": {"name": "Qwen", "families": "Qwen"},
    "mistralai": {"name": "Mistral AI", "families": "Ministral, Devstral"},
    "LiquidAI": {"name": "Liquid AI", "families": "LFM"},
}
CONVERTERS = {"unsloth": "Unsloth"}

FEATURED: tuple[Featured, ...] = (
    Featured("gemma-4-12b", "google/gemma-4-12B-it-qat-q4_0-gguf", "Gemma 4 12B", "google", "12B",
             "A dependable first model for questions, explanations and writing. Released by Google as a ready-to-run 4-bit file."),
    Featured("gemma-4-26b", "unsloth/gemma-4-26B-A4B-it-GGUF", "Gemma 4 26B", "google", "26B",
             "Larger Gemma for longer answers and research. Only part of the model runs for each word, so it stays quick."),
    Featured("ministral-3-14b", "mistralai/Ministral-3-14B-Instruct-2512-GGUF", "Ministral 3 14B", "mistralai", "14B",
             "Mistral's mid-size chat model. Good at following instructions and writing in other languages."),
    Featured("qwen-3.8-27b", "unsloth/Qwen3.8-27B-GGUF", "Qwen3.8 27B", "Qwen", "27B",
             "Careful model for coding and multi-step work with tools. Replies more slowly than the chat models."),
    Featured("devstral-small-2", "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF", "Devstral Small 2", "mistralai", "24B",
             "Mistral's coding model, built for working across a whole project with an agent."),
    Featured("gemma-4-e4b", "google/gemma-4-E4B-it-qat-q4_0-gguf", "Gemma 4 E4B", "google", "4B",
             "Small Gemma that downloads in minutes and still handles everyday questions well."),
    Featured("ministral-3-8b", "mistralai/Ministral-3-8B-Instruct-2512-GGUF", "Ministral 3 8B", "mistralai", "8B",
             "Compact Mistral chat model. A good fit for cards with 8 to 12 GB of memory."),
    Featured("lfm-2.5-2.6b", "LiquidAI/LFM2.5-2.6B-GGUF", "LFM2.5 2.6B", "LiquidAI", "2.6B",
             "Very small and very fast. Fine for short questions; weaker on hard problems."),
    Featured("lfm-3-1.7b", "LiquidAI/LFM3-1.7B-GGUF", "LFM3 1.7B", "LiquidAI", "1.7B",
             "Liquid AI's newest small model. Runs on almost any graphics card."),
)

SHELVES: tuple[dict[str, Any], ...] = (
    {"id": "start", "title": "Good first models", "subtitle": "Well-known chat models from major labs",
     "entries": ["gemma-4-12b", "gemma-4-26b", "ministral-3-14b"]},
    {"id": "code", "title": "For coding", "subtitle": "Write, explain and fix code",
     "entries": ["qwen-3.8-27b", "devstral-small-2"]},
    {"id": "small", "title": "Small and fast", "subtitle": "Quick downloads that run on most graphics cards",
     "entries": ["gemma-4-e4b", "ministral-3-8b", "lfm-2.5-2.6b", "lfm-3-1.7b"]},
)

_BITS = re.compile(r"^(?:UD-)?I?Q([1-8])(?:_|$)")


def _bits(quant: str) -> int | None:
    code = quant.upper()
    match = _BITS.match(code)
    if match:
        return int(match.group(1))
    return 4 if code.startswith("MXFP4") else None


def pick_option(options: list[QuantOption]) -> QuantOption | None:
    """The one file a card offers: the largest 4-bit copy that fits comfortably,
    preferring K-quants.

    Without a comfortable 4-bit fit, take the largest smaller copy, then the
    smallest larger one, then the same search among copies that only fit a
    short conversation. Unquantized files are never the default.
    """
    quantized = [o for o in options if o.size_mb > 0 and _bits(o.quant) is not None]
    for fit in ("fits", "tight", "unknown"):
        group = [o for o in quantized if o.fit == fit]
        four = [o for o in group if _bits(o.quant) == 4]
        if four:
            # K-quants beat the older Q4_0/Q4_1 layouts at the same size class.
            return max(four, key=lambda o: ("_K" in o.quant.upper(), o.size_mb, o.file))
        lower = [o for o in group if (_bits(o.quant) or 0) < 4]
        if lower:
            return max(lower, key=lambda o: (o.size_mb, o.file))
        if group:
            return min(group, key=lambda o: (o.size_mb, o.file))
    return min(quantized, key=lambda o: (o.size_mb, o.file)) if quantized else None


def _license(card: Any) -> str | None:
    if card is None:
        return None
    value = card.get("license") if isinstance(card, dict) else getattr(card, "license", None)
    return value if isinstance(value, str) and value else None


def featured_entry(item: Featured, vram_mb: int | None, *, api: Any = None) -> dict[str, Any]:
    """One card: curated text plus live Hub figures and the file to download."""
    owner = item.repo.split("/", 1)[0]
    entry: dict[str, Any] = {
        **asdict(item),
        "author_name": PUBLISHERS[item.author]["name"],
        "converter": CONVERTERS.get(owner) if owner != item.author else None,
    }
    info = _api(api).model_info(item.repo, files_metadata=True)
    files = [(s.rfilename, getattr(s, "size", None)) for s in (info.siblings or [])]
    options, vision = group_repo_files(files, vram_mb)
    pick = pick_option(options)
    modified: Any = getattr(info, "last_modified", None)
    entry.update(
        downloads=getattr(info, "downloads", None),
        likes=getattr(info, "likes", None),
        updated=modified.isoformat() if modified is not None else None,
        license=_license(getattr(info, "card_data", None)),
        vision=vision,
        pick=asdict(pick) if pick else None,
    )
    return entry


class FeaturedCatalog:
    """Builds the shelves once per GPU size and keeps them for a few hours."""

    TTL = 6 * 3600

    def __init__(self, entry_fn: Any = featured_entry) -> None:
        self._entry_fn = entry_fn
        self._cache: dict[int | None, tuple[float, dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def get(self, vram_mb: int | None, gpu_name: str | None) -> dict[str, Any]:
        async with self._lock:
            cached = self._cache.get(vram_mb)
            if cached and time.monotonic() - cached[0] < self.TTL:
                return {**cached[1], "gpu_name": gpu_name}
            results = await asyncio.gather(
                *(asyncio.to_thread(self._entry_fn, item, vram_mb) for item in FEATURED),
                return_exceptions=True,
            )
            entries = []
            for item, result in zip(FEATURED, results, strict=True):
                if isinstance(result, BaseException):
                    entries.append({**asdict(item), "author_name": PUBLISHERS[item.author]["name"],
                                    "error": str(result) or type(result).__name__})
                else:
                    entries.append(result)
            data = {
                "vram_mb": vram_mb,
                "publishers": [{"id": key, **value} for key, value in PUBLISHERS.items()],
                "shelves": list(SHELVES),
                "entries": entries,
                "offline": all("error" in e for e in entries),
            }
            # A partly failed fetch is retried on the next visit, not kept for hours.
            if not any("error" in e for e in entries):
                self._cache[vram_mb] = (time.monotonic(), data)
            return {**data, "gpu_name": gpu_name}
