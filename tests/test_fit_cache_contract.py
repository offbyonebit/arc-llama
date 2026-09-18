"""Fit previews must not reuse another recipe's memory estimate."""
from arc_llama.config import ModelConfig
from arc_llama.router import estimate_model_vram_with_cache


def test_recipe_and_kv_class_invalidate_fit_cache(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"fixture")
    model = ModelConfig(name="model", path=str(path), port=18080, gpu_pci_slot="test", recipe={"ctx": 1024})
    cache = {}
    calls = []

    def estimate(current):
        calls.append((current.recipe["ctx"], current.kv_class))
        return current.recipe["ctx"]

    assert estimate_model_vram_with_cache(model, cache, estimator=estimate) == 1024
    assert estimate_model_vram_with_cache(model, cache, estimator=estimate) == 1024
    assert len(calls) == 1
    model.recipe["ctx"] = 8192
    assert estimate_model_vram_with_cache(model, cache, estimator=estimate) == 8192
    model.kv_class = "large"
    estimate_model_vram_with_cache(model, cache, estimator=estimate)
    assert len(calls) == 3


def test_two_registrations_of_same_file_keep_distinct_fit_estimates(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"fixture")
    first = ModelConfig(name="short", path=str(path), port=18080, gpu_pci_slot="test", recipe={"ctx": 1024})
    second = ModelConfig(name="long", path=str(path), port=18080, gpu_pci_slot="test", recipe={"ctx": 32768})
    cache = {}
    assert estimate_model_vram_with_cache(first, cache, estimator=lambda m: m.recipe["ctx"]) == 1024
    assert estimate_model_vram_with_cache(second, cache, estimator=lambda m: m.recipe["ctx"]) == 32768
