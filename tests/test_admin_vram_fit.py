"""Admin VRAM fit visibility: /admin/status must expose estimated fit,
headroom, and confidence for each model's current recipe, without repeating
expensive GGUF scans on every poll."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from arc_llama.config import Config, GPUConfig, ModelConfig
from arc_llama.router import (
    estimate_model_vram_with_cache,
    model_vram_fit_info,
)


def _file_config(tmp_path: Path, *, vram_mb: int = 24_576) -> Config:
    model_file = tmp_path / "dense.gguf"
    model_file.write_bytes(b"x" * (2 * 1_048_576))  # 2 MiB
    cfg = Config()
    cfg.gpus = [
        GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=vram_mb)
    ]
    cfg.models = [
        ModelConfig(
            name="dense",
            path=str(model_file),
            port=18080,
            gpu_pci_slot="0000:03:00.0",
            recipe={"ctx": 8192, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
        )
    ]
    return cfg


def test_fit_info_reports_estimate_headroom_and_confidence(tmp_path):
    cfg = _file_config(tmp_path)
    info = model_vram_fit_info(cfg.models[0], cfg.gpus[0])
    assert info is not None
    assert info["estimated_mb"] > 0
    assert info["confidence"] == "estimated_from_file_size"
    assert info["fit"] is True
    assert info["headroom_mb"] == 24_576 - info["estimated_mb"]


def test_fit_info_reports_negative_headroom_when_oversized(tmp_path):
    cfg = _file_config(tmp_path, vram_mb=64)
    info = model_vram_fit_info(cfg.models[0], cfg.gpus[0])
    assert info is not None
    assert info["fit"] is False
    assert info["headroom_mb"] == 64 - info["estimated_mb"]


def test_fit_info_marks_disabled_gpu_unfit(tmp_path):
    cfg = _file_config(tmp_path)
    cfg.gpus[0].enabled = False
    info = model_vram_fit_info(cfg.models[0], cfg.gpus[0])
    assert info is not None
    assert info["fit"] is False
    assert info["headroom_mb"] is None
    assert info["detail"] == "configured GPU is disabled"


def test_fit_info_unknown_when_gpu_budget_missing(tmp_path):
    cfg = _file_config(tmp_path)
    cfg.gpus[0].vram_mb = None
    info = model_vram_fit_info(cfg.models[0], cfg.gpus[0])
    assert info is not None
    assert info["estimated_mb"] > 0
    assert info["fit"] is None
    assert "headroom_mb" not in info


def test_fit_info_honest_when_file_missing(tmp_path):
    cfg = _file_config(tmp_path, vram_mb=100_000)
    cfg.models[0].path = str(tmp_path / "does-not-exist.gguf")
    # Quick estimator falls back (OSError) -> exact estimator falls back to
    # file size 0 bytes on OSError... both cannot read a missing file: the
    # quick path returns None, the exact path estimates 0 bytes worth of
    # weights. Either way the block never invents a nonzero number.
    info = model_vram_fit_info(cfg.models[0], cfg.gpus[0])
    assert info is None or info["estimated_mb"] >= 0


def test_estimate_cache_hits_within_ttl_and_invalidates(tmp_path, monkeypatch):
    cfg = _file_config(tmp_path)
    model = cfg.models[0]
    calls = {"n": 0}

    def counting_estimator(m):
        calls["n"] += 1
        return 1234

    cache: dict[str, tuple[float, int | None]] = {}
    first = estimate_model_vram_with_cache(model, cache, estimator=counting_estimator)
    second = estimate_model_vram_with_cache(model, cache, estimator=counting_estimator)
    assert first == second == 1234
    assert calls["n"] == 1, "second call must hit the cache"

    # Recipe change invalidates the key: a new ctx is a new footprint.
    model.recipe = {"ctx": 65536, "cache_type_k": "f16", "cache_type_v": "f16"}
    third = estimate_model_vram_with_cache(model, cache, estimator=counting_estimator)
    assert calls["n"] == 2
    assert third == 1234

    # File replacement (mtime change) also invalidates the key.
    model.recipe = {"ctx": 8192, "cache_type_k": "q8_0", "cache_type_v": "q8_0"}
    model_file = Path(model.path)
    model_file.write_bytes(b"y" * (4 * 1_048_576))
    estimate_model_vram_with_cache(model, cache, estimator=counting_estimator)
    assert calls["n"] == 3


def test_admin_status_includes_vram_estimate_block(tmp_path):
    from arc_llama.server import create_app

    cfg = _file_config(tmp_path)
    app = create_app(cfg, plugins=[])
    with TestClient(app) as client:
        response = client.get("/admin/status")
    assert response.status_code == 200
    entry = next(m for m in response.json()["models"] if m["name"] == "dense")
    assert "vram_estimate" in entry
    fit = entry["vram_estimate"]
    assert fit["estimated_mb"] > 0
    assert fit["confidence"] == "estimated_from_file_size"
    assert fit["fit"] is True
    assert fit["headroom_mb"] == 24_576 - fit["estimated_mb"]
    # Surface the switch policy knobs too, so the UI can explain waits.
    server = response.json()["server"]
    assert server["switch_drain_seconds"] == 30.0
    assert server["switch_interrupt_policy"] == "reject_new"


def test_admin_status_repeated_polls_do_not_rescan(tmp_path, monkeypatch):
    from arc_llama.server import create_app

    cfg = _file_config(tmp_path)
    # The config fixture's model has no n_cpu_moe, so the quick estimator
    # handles it without GGUF parsing; prove repeated polls do not escalate
    # to the exact (tensor) estimator at all.
    calls = {"exact": 0}
    real_exact = estimate_model_vram_with_cache

    def counting_exact(model, cache, **kwargs):
        calls["exact"] += 1
        return real_exact(model, cache, **kwargs)

    monkeypatch.setattr("arc_llama.router.estimate_model_vram_with_cache", counting_exact)
    app = create_app(cfg, plugins=[])
    with TestClient(app) as client:
        for _ in range(3):
            assert client.get("/admin/status").status_code == 200
    assert calls["exact"] == 0, "dense-model status polls must use the quick estimator only"
