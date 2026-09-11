from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _aggregate_module():
    path = Path(__file__).parent.parent / "registry-repo" / "scripts" / "aggregate.py"
    spec = importlib.util.spec_from_file_location("arc_llama_recipe_aggregate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _doc(
    recipe: dict,
    prompt: float,
    generation: float,
    *,
    provenance: dict | None = None,
    submits: int = 1,
):
    return {
        "fingerprint": "a" * 64,
        "recipe": recipe,
        "prompt_eval_tok_s": prompt,
        "generation_tok_s": generation,
        "gpu_name": "Arc",
        "arc_llama_version": "0.8.0",
        "provenance": provenance or {},
        "submits": submits,
    }


def test_aggregate_does_not_mix_axes_from_different_recipes():
    aggregate = _aggregate_module()
    merged = aggregate.aggregate_documents(
        [
            _doc({"kv": "f16"}, 200.0, 10.0),
            _doc({"kv": "q8_0"}, 100.0, 30.0),
        ]
    )
    winner = merged["a" * 64]
    assert winner["recipe"] == {"kv": "q8_0"}
    assert winner["prompt_eval_tok_s"] == 100.0
    assert winner["generation_tok_s"] == 30.0
    assert winner["candidate_count"] == 2


def test_aggregate_uses_medians_and_counts_only_agreeing_trials():
    aggregate = _aggregate_module()
    provenance = {"llama_server_backend": "sycl"}
    merged = aggregate.aggregate_documents(
        [
            _doc({"kv": "q8_0"}, 100.0, 20.0, provenance=provenance),
            _doc({"kv": "q8_0"}, 1000.0, 22.0, provenance=provenance, submits=2),
            _doc({"kv": "f16"}, 90.0, 19.0),
        ]
    )
    winner = merged["a" * 64]
    assert winner["recipe"] == {"kv": "q8_0"}
    assert winner["prompt_eval_tok_s"] == 550.0
    assert winner["generation_tok_s"] == 21.0
    assert winner["submits"] == 3
    assert winner["sample_count"] == 2


@pytest.mark.parametrize(
    "doc",
    [
        {},
        {"fingerprint": "a" * 64, "recipe": {"kv": "q8_0"}, "submits": "bad"},
        {"fingerprint": "not-a-hash", "recipe": {"kv": "q8_0"}},
    ],
)
def test_aggregate_rejects_malformed_submissions_cleanly(doc):
    aggregate = _aggregate_module()
    with pytest.raises(ValueError, match="submission 0 is invalid"):
        aggregate.aggregate_documents([doc])
