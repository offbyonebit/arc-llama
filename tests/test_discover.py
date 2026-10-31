from __future__ import annotations

import asyncio
from types import SimpleNamespace

from httpx import ASGITransport, AsyncClient

from arc_llama.discover import (
    FEATURED,
    PUBLISHERS,
    SHELVES,
    FeaturedCatalog,
    featured_entry,
    pick_option,
)
from arc_llama.model_library import (
    FIT_CTX,
    TIGHT_CTX,
    QuantOption,
    group_repo_files,
    memory_need_mb,
)

try:
    from tests.test_model_library import _app
except ModuleNotFoundError:
    from test_model_library import _app

GB = 1024**3


def opt(file: str, quant: str, size_mb: int, fit: str) -> QuantOption:
    return QuantOption(file=file, quant=quant, size_mb=size_mb, fit=fit)


def test_curated_list_is_consistent() -> None:
    ids = {item.id for item in FEATURED}
    assert len(ids) == len(FEATURED)
    assert all(item.author in PUBLISHERS for item in FEATURED)
    assert all(entry in ids for shelf in SHELVES for entry in shelf["entries"])
    assert all("/" in item.repo and item.description for item in FEATURED)


def test_pick_prefers_largest_fitting_four_bit_k_quant() -> None:
    options = [
        opt("m-Q3_K_M.gguf", "Q3_K_M", 4000, "fits"),
        opt("m-Q4_1.gguf", "Q4_1", 5600, "fits"),
        opt("m-Q4_K_M.gguf", "Q4_K_M", 5200, "fits"),
        opt("m-Q4_K_S.gguf", "Q4_K_S", 4900, "fits"),
        opt("m-Q8_0.gguf", "Q8_0", 9000, "fits"),
        opt("m-BF16.gguf", "BF16", 16000, "tight"),
    ]
    assert pick_option(options).file == "m-Q4_K_M.gguf"


def test_pick_falls_back_to_smaller_then_tight_then_smallest() -> None:
    assert pick_option([opt("a-Q3_K_M.gguf", "Q3_K_M", 9, "fits"), opt("a-Q4_K_M.gguf", "Q4_K_M", 12, "tight")]).file == "a-Q3_K_M.gguf"
    assert pick_option([opt("a-Q4_K_M.gguf", "Q4_K_M", 12, "tight"), opt("a-Q8_0.gguf", "Q8_0", 20, "too_big")]).file == "a-Q4_K_M.gguf"
    assert pick_option([opt("a-Q6_K.gguf", "Q6_K", 8, "fits"), opt("a-Q8_0.gguf", "Q8_0", 10, "fits")]).file == "a-Q6_K.gguf"
    assert pick_option([opt("a-Q8_0.gguf", "Q8_0", 30, "too_big"), opt("a-Q4_K_M.gguf", "Q4_K_M", 20, "too_big")]).file == "a-Q4_K_M.gguf"
    assert pick_option([opt("a-F16.gguf", "F16", 30, "fits")]) is None
    assert pick_option([opt("a-Q4_K_M.gguf", "Q4_K_M", 0, "unknown")]) is None


def test_options_carry_memory_need_for_their_badge() -> None:
    options, _ = group_repo_files([("m-Q4_K_M.gguf", 5 * GB), ("m-Q8_0.gguf", 21 * GB)], 24576)
    by_quant = {o.quant: o for o in options}
    assert by_quant["Q4_K_M"].need_mb == memory_need_mb(5120, FIT_CTX)
    assert by_quant["Q8_0"].fit == "tight"
    assert by_quant["Q8_0"].need_mb == memory_need_mb(21 * 1024, TIGHT_CTX)


class FakeApi:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def model_info(self, repo: str, files_metadata: bool = False) -> SimpleNamespace:
        self.calls.append(repo)
        return SimpleNamespace(
            siblings=[SimpleNamespace(rfilename="m-Q4_K_M.gguf", size=5 * GB), SimpleNamespace(rfilename="mmproj-F16.gguf", size=GB)],
            downloads=42,
            likes=3,
            last_modified=None,
            card_data={"license": "apache-2.0"},
        )


def test_featured_entry_combines_curated_text_and_hub_figures() -> None:
    item = next(i for i in FEATURED if i.repo.startswith("unsloth/"))
    entry = featured_entry(item, 24576, api=FakeApi())
    assert entry["author_name"] == PUBLISHERS[item.author]["name"]
    assert entry["converter"] == "Unsloth"
    assert entry["license"] == "apache-2.0" and entry["downloads"] == 42 and entry["vision"]
    assert entry["pick"]["file"] == "m-Q4_K_M.gguf" and entry["pick"]["need_mb"] > entry["pick"]["size_mb"]
    original = next(i for i in FEATURED if i.repo.split("/")[0] == i.author)
    assert featured_entry(original, 24576, api=FakeApi())["converter"] is None


def test_catalog_caches_complete_results_and_retries_failures() -> None:
    calls: list[str] = []
    broken = {FEATURED[0].id}

    def entry_fn(item, vram_mb):
        calls.append(item.id)
        if item.id in broken:
            raise RuntimeError("hub down")
        return {"id": item.id, "pick": None}

    async def run() -> None:
        catalog = FeaturedCatalog(entry_fn)
        first = await catalog.get(24576, "Arc")
        assert not first["offline"] and first["gpu_name"] == "Arc"
        assert next(e for e in first["entries"] if e["id"] == FEATURED[0].id)["error"] == "hub down"
        broken.clear()
        await catalog.get(24576, "Arc")
        assert len(calls) == 2 * len(FEATURED)  # partial failure was not cached
        cached = await catalog.get(24576, "Renamed")
        assert len(calls) == 2 * len(FEATURED) and cached["gpu_name"] == "Renamed"
        await catalog.get(8192, "Small")
        assert len(calls) == 3 * len(FEATURED)  # another card size is its own entry

    asyncio.run(run())


def test_catalog_reports_offline_when_every_fetch_fails() -> None:
    def entry_fn(item, vram_mb):
        raise OSError("no network")

    data = asyncio.run(FeaturedCatalog(entry_fn).get(None, None))
    assert data["offline"] and all(e["error"] == "no network" for e in data["entries"])


async def test_featured_endpoint_requires_admin_and_passes_gpu(monkeypatch, tmp_path) -> None:
    _, app = _app(monkeypatch, tmp_path)
    seen = []

    class Catalog:
        async def get(self, vram_mb, gpu_name):
            seen.append((vram_mb, gpu_name))
            return {"entries": [], "vram_mb": vram_mb, "gpu_name": gpu_name}

    async with app.router.lifespan_context(app):
        app.state.featured = Catalog()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            assert (await client.get("/admin/library/featured")).status_code == 401
            response = await client.get("/admin/library/featured", headers={"Authorization": "Bearer tok"})
            assert response.status_code == 200
    gpu = next(g for g in app.state.cfg.gpus if g.enabled)
    assert seen == [(gpu.vram_mb, gpu.name)]
