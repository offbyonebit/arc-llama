"""Vision projectors, split GGUFs, rerankers, and multi-GPU split recipes."""

from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient

from arc_llama.config import Config, GPUConfig, ModelConfig, ServerConfig
from arc_llama.failures import StartupFailureError
from arc_llama.gguf_meta import (
    gguf_shards,
    gguf_total_bytes,
    is_secondary_shard,
    missing_shards,
    split_info,
)
from arc_llama.launcher import LaunchPlan
from arc_llama.models import (
    HFModelSpec,
    find_mmproj,
    is_auxiliary_gguf,
    is_mmproj_gguf,
    plan_hf_files,
)
from arc_llama.preflight import preflight_launch
from arc_llama.recipes import LaunchRecipe, recipe_to_dict
from arc_llama.router import estimate_model_vram_quick_mb, mmproj_vram_mb
from arc_llama.server import create_app, model_capabilities, request_has_images
from arc_llama.server_caps import _parse_help

# -- split GGUFs -------------------------------------------------------------


def test_split_name_parsing(tmp_path) -> None:
    first = tmp_path / "Big-Q8_0-00001-of-00003.gguf"
    assert split_info(first) == (1, 3)
    assert split_info(tmp_path / "plain.gguf") is None
    assert split_info(tmp_path / "x-00004-of-00003.gguf") is None
    assert not is_secondary_shard(first)
    assert is_secondary_shard(tmp_path / "Big-Q8_0-00002-of-00003.gguf")
    assert [p.name for p in gguf_shards(first)] == [
        "Big-Q8_0-00001-of-00003.gguf",
        "Big-Q8_0-00002-of-00003.gguf",
        "Big-Q8_0-00003-of-00003.gguf",
    ]
    assert gguf_shards(tmp_path / "plain.gguf") == [tmp_path / "plain.gguf"]


def test_split_sizes_and_missing_shards(tmp_path) -> None:
    shards = gguf_shards(tmp_path / "m-00001-of-00002.gguf")
    shards[0].write_bytes(b"a" * 10)
    assert [p.name for p in missing_shards(shards[0])] == ["m-00002-of-00002.gguf"]
    shards[1].write_bytes(b"b" * 7)
    assert missing_shards(shards[0]) == []
    assert gguf_total_bytes(shards[0]) == 17


def test_secondary_shards_are_auxiliary() -> None:
    assert is_auxiliary_gguf("model-00002-of-00004.gguf")
    assert not is_auxiliary_gguf("model-00001-of-00004.gguf")


def _preflight_inputs(tmp_path, model_name: str, extra_argv: list[str] | None = None):
    runtime = tmp_path / ("llama-server.exe" if os.name == "nt" else "llama-server")
    runtime.write_bytes(b"runtime")
    if os.name != "nt":
        runtime.chmod(0o755)
    model_path = tmp_path / model_name
    model_path.write_bytes(b"GGUF")
    gpu = GPUConfig("0000:03:00.0", 0, "battlemage", enabled=True)
    model = ModelConfig("model", str(model_path), 18999, gpu.pci_slot)
    plan = LaunchPlan(
        argv=[str(runtime), "-m", str(model_path), *(extra_argv or [])],
        env={},
        backend_url="http://127.0.0.1:18999",
        health_url="http://127.0.0.1:18999/health",
    )
    return model, gpu, plan


def test_preflight_rejects_incomplete_split_model(tmp_path) -> None:
    model, gpu, plan = _preflight_inputs(tmp_path, "m-00001-of-00002.gguf")
    with pytest.raises(StartupFailureError) as exc:
        preflight_launch(model, gpu, plan, managed_pids=set())
    assert exc.value.category == "shard_missing"
    assert "m-00002-of-00002.gguf" in exc.value.message


def test_preflight_rejects_missing_projector(tmp_path) -> None:
    missing = tmp_path / "mmproj-F16.gguf"
    model, gpu, plan = _preflight_inputs(tmp_path, "m.gguf", ["--mmproj", str(missing)])
    with pytest.raises(StartupFailureError) as exc:
        preflight_launch(model, gpu, plan, managed_pids=set())
    assert exc.value.category == "mmproj_missing"
    assert exc.value.http_status == 404


# -- vision projectors -------------------------------------------------------


def _touch(directory, *names):
    for name in names:
        (directory / name).write_bytes(b"GGUF")


def test_generic_projector_pairs_in_a_single_family_folder(tmp_path) -> None:
    _touch(tmp_path, "gemma-3-27b-it-Q4_K_M.gguf", "gemma-3-27b-it-Q8_0.gguf")
    _touch(tmp_path, "mmproj-F32.gguf", "mmproj-F16.gguf", "mmproj-BF16.gguf")
    assert find_mmproj(tmp_path / "gemma-3-27b-it-Q4_K_M.gguf") == tmp_path / "mmproj-F16.gguf"


def test_generic_projector_is_not_attached_in_a_mixed_folder(tmp_path) -> None:
    _touch(tmp_path, "gemma-3-27b-it-Q4_K_M.gguf", "llama-3.1-8b-Q4_K_M.gguf", "mmproj-F16.gguf")
    assert find_mmproj(tmp_path / "llama-3.1-8b-Q4_K_M.gguf") is None
    assert find_mmproj(tmp_path / "gemma-3-27b-it-Q4_K_M.gguf") is None


def test_named_projector_pairs_with_its_family_only(tmp_path) -> None:
    _touch(
        tmp_path,
        "gemma-3-27b-it-Q4_K_M.gguf",
        "llama-3.1-8b-Q4_K_M.gguf",
        "mmproj-gemma-3-27b-it-f16.gguf",
    )
    assert (
        find_mmproj(tmp_path / "gemma-3-27b-it-Q4_K_M.gguf")
        == tmp_path / "mmproj-gemma-3-27b-it-f16.gguf"
    )
    assert find_mmproj(tmp_path / "llama-3.1-8b-Q4_K_M.gguf") is None


def test_projector_detection() -> None:
    assert is_mmproj_gguf("mmproj-F16.gguf")
    assert is_mmproj_gguf("Qwen2-VL-7B-mmproj-f16.gguf")
    assert not is_mmproj_gguf("gemma-3-27b-it-Q4_K_M.gguf")
    assert not is_mmproj_gguf("mmproj-F16.txt")


def test_registration_pairs_projector(tmp_path) -> None:
    from arc_llama.models import register_discovered

    _touch(tmp_path, "gemma-3-4b-it-Q4_K_M.gguf", "mmproj-F16.gguf")
    cfg = Config(
        gpus=[GPUConfig("0000:03:00.0", 0, "battlemage", vram_mb=24576)],
    )
    cfg.paths.llama_server = str(tmp_path / "missing-llama-server")
    [model] = register_discovered(cfg, [tmp_path / "gemma-3-4b-it-Q4_K_M.gguf"])
    assert model.recipe["mmproj"] == str(tmp_path / "mmproj-F16.gguf")
    assert "vision" in model_capabilities(model)


def test_projector_vram_is_counted(tmp_path) -> None:
    projector = tmp_path / "mmproj-F16.gguf"
    projector.write_bytes(b"\0" * (8 * 1_048_576))
    assert mmproj_vram_mb({"mmproj": str(projector)}) == 10
    assert mmproj_vram_mb({"mmproj": str(projector), "mmproj_offload": False}) == 0
    assert mmproj_vram_mb({}) == 0
    weights = tmp_path / "m.gguf"
    weights.write_bytes(b"\0" * 1_048_576)
    base = ModelConfig("m", str(weights), 1, "x", recipe={"ctx": 4096})
    paired = ModelConfig("m", str(weights), 1, "x", recipe={"ctx": 4096, "mmproj": str(projector)})
    plain_mb = estimate_model_vram_quick_mb(base)
    paired_mb = estimate_model_vram_quick_mb(paired)
    assert plain_mb is not None and paired_mb is not None
    assert paired_mb - plain_mb == 10


# -- Hugging Face file planning ----------------------------------------------


_REPO = [
    "README.md",
    "gemma-3-27b-it-Q4_K_M.gguf",
    "gemma-3-27b-it-Q8_0.gguf",
    "mmproj-F16.gguf",
    "mmproj-F32.gguf",
    "Q8_K_XL/gemma-3-27b-it-UD-Q8_K_XL-00001-of-00002.gguf",
    "Q8_K_XL/gemma-3-27b-it-UD-Q8_K_XL-00002-of-00002.gguf",
]


def test_hf_plan_adds_projector_for_vision_repo() -> None:
    assert plan_hf_files(HFModelSpec("u/r", None, "Q4_K_M"), _REPO) == (
        "gemma-3-27b-it-Q4_K_M.gguf",
        ["mmproj-F16.gguf"],
    )


def test_hf_plan_downloads_every_shard() -> None:
    main, extras = plan_hf_files(HFModelSpec("u/r", None, "Q8_K_XL"), _REPO)
    assert main == "Q8_K_XL/gemma-3-27b-it-UD-Q8_K_XL-00001-of-00002.gguf"
    assert extras == ["Q8_K_XL/gemma-3-27b-it-UD-Q8_K_XL-00002-of-00002.gguf"]


def test_hf_plan_never_picks_projector_or_shard_as_model() -> None:
    files = ["mmproj-F16.gguf", "m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]
    main, extras = plan_hf_files(HFModelSpec("u/r", None, None), files)
    assert main == "m-00001-of-00002.gguf"
    assert "m-00002-of-00002.gguf" in extras


def test_hf_plan_explicit_file_without_listing() -> None:
    assert plan_hf_files(HFModelSpec("u/r", "x.gguf", None), None) == ("x.gguf", [])


# -- launch arguments and capabilities ---------------------------------------


def test_recipe_argv_and_round_trip() -> None:
    recipe = LaunchRecipe(
        mmproj="/m/mmproj.gguf",
        mmproj_offload=False,
        reranking=True,
        tensor_split=[3.0, 1.5],
        split_mode="row",
    )
    argv = recipe.to_argv()
    assert argv[argv.index("--mmproj") + 1] == "/m/mmproj.gguf"
    assert "--no-mmproj-offload" in argv
    assert "--reranking" in argv
    assert argv[argv.index("--tensor-split") + 1] == "3,1.5"
    assert argv[argv.index("--split-mode") + 1] == "row"
    stored = recipe_to_dict(recipe)
    model = ModelConfig("m", "/m.gguf", 1, "x", recipe=stored)
    assert model.launch_recipe().to_argv() == argv


def test_plain_recipe_has_no_new_flags() -> None:
    argv = LaunchRecipe().to_argv()
    for flag in ("--mmproj", "--reranking", "--tensor-split", "--split-mode"):
        assert flag not in argv


def test_help_probe_detects_new_capabilities() -> None:
    caps = _parse_help("--mmproj FILE\n--reranking\n--tensor-split N0,N1\n--flash-attn auto")
    assert caps.supports_mmproj and caps.supports_reranking and caps.supports_tensor_split
    bare = _parse_help("--flash-attn auto")
    assert not (bare.supports_mmproj or bare.supports_reranking or bare.supports_tensor_split)


def test_request_image_detection() -> None:
    assert request_has_images(
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]}
    )
    assert not request_has_images({"messages": [{"role": "user", "content": "hi"}]})
    assert not request_has_images({"messages": "junk"})


def test_capabilities() -> None:
    assert model_capabilities(ModelConfig("a", "/a", 1, "x")) == [
        "chat",
        "completion",
        "embedding",
    ]
    assert model_capabilities(ModelConfig("a", "/a", 1, "x", recipe={"reranking": True})) == [
        "rerank"
    ]


# -- server ------------------------------------------------------------------


def _server_fakes():
    try:
        from tests.test_server import FakeRouter, FakeUpstreamManager
    except ModuleNotFoundError:
        from test_server import FakeRouter, FakeUpstreamManager
    return FakeRouter, FakeUpstreamManager


async def test_text_model_rejects_images_and_rerank(monkeypatch) -> None:
    import arc_llama.server as server_mod

    fake_router, fake_upstream = _server_fakes()
    monkeypatch.setattr(server_mod, "Router", fake_router)
    monkeypatch.setattr(server_mod, "UpstreamManager", fake_upstream)
    cfg = Config(
        server=ServerConfig(admin_token=None),
        models=[ModelConfig("qwen", "/models/qwen.gguf", 18080, "0000:03:00.0")],
    )
    app = create_app(cfg, plugins=[])
    image_body = {
        "model": "qwen",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                ],
            }
        ],
    }
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            response = await client.post("/v1/chat/completions", json=image_body)
            assert response.status_code == 400
            assert "mmproj" in response.json()["detail"]
            rerank = await client.post(
                "/v1/rerank", json={"model": "qwen", "query": "q", "documents": ["a"]}
            )
            assert rerank.status_code == 400
            assert "reranker" in rerank.json()["detail"]
            listing = await client.get("/v1/models")
            [entry] = [m for m in listing.json()["data"] if m["id"] == "qwen"]
            assert entry["metadata"]["capabilities"] == ["chat", "completion", "embedding"]


def test_edit_endpoint_accepts_new_fields(monkeypatch, tmp_path) -> None:
    from fastapi.testclient import TestClient

    import arc_llama.server as server_mod

    try:
        from tests.test_server import FakeRouterWithRebuild, FakeUpstreamManager
    except ModuleNotFoundError:
        from test_server import FakeRouterWithRebuild, FakeUpstreamManager

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(server_mod, "Router", FakeRouterWithRebuild)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    projector = tmp_path / "mmproj-F16.gguf"
    projector.write_bytes(b"GGUF")
    cfg = Config(
        server=ServerConfig(admin_token=None),
        gpus=[GPUConfig("0000:03:00.0", 0, "battlemage", vram_mb=24576)],
        models=[ModelConfig("qwen", "/models/qwen.gguf", 18080, "0000:03:00.0")],
    )
    app = create_app(cfg, config_path=tmp_path / "config.toml", plugins=[])
    with TestClient(app) as client:
        ok = client.post(
            "/admin/models/qwen/edit",
            json={
                "mmproj": str(projector),
                "mmproj_offload": False,
                "tensor_split": [1, 1],
                "split_mode": "layer",
            },
        )
        assert ok.status_code == 200, ok.text
        recipe = ok.json()["recipe"]
        assert recipe["mmproj"] == str(projector)
        assert recipe["mmproj_offload"] is False
        assert recipe["tensor_split"] == [1.0, 1.0]
        for bad in (
            {"mmproj": str(tmp_path / "nope.gguf")},
            {"reranking": "yes"},
            {"tensor_split": [1]},
            {"tensor_split": [0, 0]},
            {"tensor_split": [1, -1]},
            {"split_mode": "diagonal"},
        ):
            assert client.post("/admin/models/qwen/edit", json=bad).status_code == 400, bad
        cleared = client.post("/admin/models/qwen/edit", json={"mmproj": None, "tensor_split": None})
        assert "mmproj" not in cleared.json()["recipe"]
        assert "tensor_split" not in cleared.json()["recipe"]
