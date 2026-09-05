"""Tests for the community recipe registry (recipe_share)."""
from __future__ import annotations

import json

from arc_llama.recipe_share import (
    REGISTRY_SCHEMA,
    RecipeRegistry,
    _clean_recipe_field,
    _vram_bucket,
    build_pr_body,
    share_fingerprint,
    shared_recipe_edits_to_model_recipe,
    submission_document,
    validate_submission,
)


class TestShareFingerprint:
    def test_stable_across_calls(self):
        a = share_fingerprint("bmg", "sycl", "qwen3_dense", "balanced", 3, 12288)
        b = share_fingerprint("bmg", "sycl", "qwen3_dense", "balanced", 3, 12288)
        assert a == b
        assert len(a) == 64

    def test_differs_on_arch_and_backend(self):
        base = dict(model_class="qwen3_dense", workload_key="balanced",
                    tune_schema_version=3, vram_mb=12288)
        a = share_fingerprint("bmg", "sycl", **base)
        b = share_fingerprint("bmg", "vulkan", **base)
        c = share_fingerprint("dg2", "sycl", **base)
        assert len({a, b, c}) == 3

    def test_vram_bucket_rounds_down(self):
        assert _vram_bucket(12288) == 12 * 1024
        assert _vram_bucket(12000) == 12 * 1024
        assert _vram_bucket(8192) == 8 * 1024
        assert _vram_bucket(24000) == 24 * 1024
        assert _vram_bucket(20000) == 16 * 1024

    def test_bucket_boundary_merges_nearby_cards(self):
        a = share_fingerprint("bmg", "sycl", "x", "balanced", 3, 12000)
        b = share_fingerprint("bmg", "sycl", "x", "balanced", 3, 12300)
        assert a == b


class TestCleanRecipeField:
    def test_drops_unknown_keys(self):
        out = _clean_recipe_field({"kv": "q8_0", "extra_flags": ["--evil"], "threads": 4})
        assert out == {"kv": "q8_0"}

    def test_rejects_invalid_kv_and_fa(self):
        assert _clean_recipe_field({"kv": "q3_k_s"}) == {}
        assert _clean_recipe_field({"fa": "maybe"}) == {}
        assert _clean_recipe_field({"fa": None}) == {}

    def test_ubatch_allowlist_and_batch_ge_ubatch(self):
        assert _clean_recipe_field({"ubatch": 1024, "batch": 2048}) == {"ubatch": 1024, "batch": 2048}
        assert _clean_recipe_field({"ubatch": 777}) == {}
        assert _clean_recipe_field({"ubatch": 2048, "batch": 512}) == {"ubatch": 2048}

    def test_n_cpu_moe_nonneg_int(self):
        assert _clean_recipe_field({"n_cpu_moe": 7}) == {"n_cpu_moe": 7}
        assert _clean_recipe_field({"n_cpu_moe": -1}) == {}
        assert _clean_recipe_field({"n_cpu_moe": "12"}) == {"n_cpu_moe": 12}


class TestValidateSubmission:
    def _doc(self, **over):
        doc = submission_document(
            fingerprint="a" * 64,
            recipe={"kv": "q8_0", "ubatch": 1024},
            prompt_eval_tok_s=400.0,
            generation_tok_s=20.0,
            gpu_name="Arc Pro B60",
            arc_llama_version="0.7.1",
        )
        doc.update(over)
        return doc

    def test_valid_submission_passes(self):
        assert validate_submission(self._doc()) == []

    def test_fingerprint_shape_enforced(self):
        problems = validate_submission(self._doc(fingerprint="XYZ"))
        assert any("fingerprint" in p for p in problems)

    def test_unknown_recipe_keys_rejected(self):
        problems = validate_submission(self._doc(recipe={"kv": "f16", "temp": 0.7}))
        assert any("unknown recipe keys" in p for p in problems)

    def test_extra_flags_never_shareable(self):
        problems = validate_submission(self._doc(recipe={"extra_flags": ["--evil"]}))
        assert problems  # rejected as unknown key

    def test_empty_recipe_rejected(self):
        assert validate_submission(self._doc(recipe={})) != []

    def test_batch_lt_ubatch_rejected(self):
        problems = validate_submission(self._doc(recipe={"ubatch": 2048, "batch": 512}))
        assert any("batch" in p for p in problems)

    def test_nonsense_tok_s_rejected(self):
        for bad in (-5, 0, 10**7, "fast"):
            assert validate_submission(self._doc(prompt_eval_tok_s=bad)) != []

    def test_non_dict_rejected(self):
        assert validate_submission([1, 2]) != []
        assert validate_submission("nope") != []


class TestRecipeRegistry:
    def _registry(self, entries: dict) -> RecipeRegistry:
        return RecipeRegistry({"schema": REGISTRY_SCHEMA, "recipes": entries})

    def test_lookup_hit_returns_sanitised_entry(self):
        fp = "b" * 64
        reg = self._registry({
            fp: {
                "recipe": {"kv": "q8_0", "fa": "auto", "ubatch": 1024, "evil": True},
                "submits": 3,
                "prompt_eval_tok_s": 410.0,
                "generation_tok_s": 19.5,
                "gpu_name": "B60",
            }
        })
        e = reg.lookup(fp)
        assert e is not None
        assert e.submits == 3
        assert e.edits == {"kv": "q8_0", "fa": "auto", "ubatch": 1024}
        assert e.generation_tok_s == 19.5

    def test_lookup_miss_returns_none(self):
        assert self._registry({}).lookup("c" * 64) is None

    def test_lookup_entry_without_valid_fields_is_none(self):
        reg = self._registry({"d" * 64: {"recipe": {"evil": 1}, "submits": 1}})
        assert reg.lookup("d" * 64) is None

    def test_bundled_file_loads(self, monkeypatch, tmp_path):
        # No user override; the shipped data/recipes.json (empty registry) loads.
        monkeypatch.delenv("ARC_LLAMA_RECIPES_PATH", raising=False)
        reg = RecipeRegistry()
        assert reg._data["schema"] == REGISTRY_SCHEMA
        assert reg._data["recipes"] == {}

    def test_user_override_wins(self, monkeypatch, tmp_path):
        override = tmp_path / "recipes.json"
        override.write_text(json.dumps({
            "schema": REGISTRY_SCHEMA,
            "recipes": {"e" * 64: {"recipe": {"kv": "q4_0"}, "submits": 1}},
        }))
        monkeypatch.setenv("ARC_LLAMA_RECIPES_PATH", str(override))
        e = RecipeRegistry().lookup("e" * 64)
        assert e is not None and e.edits == {"kv": "q4_0"}

    def test_future_schema_ignored(self, tmp_path, monkeypatch):
        override = tmp_path / "recipes.json"
        override.write_text(json.dumps({"schema": 999, "recipes": {"f" * 64: {}}}))
        monkeypatch.setenv("ARC_LLAMA_RECIPES_PATH", str(override))
        assert RecipeRegistry().lookup("f" * 64) is None


class TestEditsMapping:
    def test_maps_registry_keys_to_model_recipe_keys(self):
        edits = {"kv": "q8_0", "fa": "auto", "ubatch": 1024, "batch": 2048, "n_cpu_moe": 4}
        out = shared_recipe_edits_to_model_recipe(edits)
        assert out == {
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "flash_attn": "auto",
            "ubatch_size": 1024,
            "batch_size": 2048,
            "n_cpu_moe": 4,
        }


class TestSubmissionDocument:
    def test_document_round_trips_through_validation(self):
        doc = submission_document(
            fingerprint="a" * 64,
            recipe={"kv": "q8_0"},
            prompt_eval_tok_s=1.0,
            generation_tok_s=1.0,
            gpu_name="B580",
            arc_llama_version="0.7.1",
        )
        assert validate_submission(doc) == []
        assert json.loads(json.dumps(doc)) == doc

    def test_pr_body_mentions_fingerprint_and_recipe(self):
        doc = submission_document(
            fingerprint="a" * 64,
            recipe={"kv": "q8_0"},
            prompt_eval_tok_s=300.0,
            generation_tok_s=15.0,
            gpu_name="B580",
            arc_llama_version="0.7.1",
        )
        body = build_pr_body(doc)
        assert "aaaaaaaaaaaaaaaa" in body
        assert "q8_0" in body
