#!/usr/bin/env python3
"""Aggregate recipes/*.json into the single bundled dist/recipes.json.

Source documents are grouped by fingerprint, then by sanitised recipe and
llama-server provenance. Metrics are medians within each agreeing group; the
balanced winner is exposed at the entry root and every candidate group remains
in the bundle for auditing.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "dist" / "recipes.json"


def _metric(doc: dict[str, Any], key: str) -> float | None:
    value = doc.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        return None
    return float(value)


def _median_metric(docs: list[dict[str, Any]], key: str) -> float | None:
    values = [value for doc in docs if (value := _metric(doc, key)) is not None]
    return float(statistics.median(values)) if values else None


def _balanced_score(entry: dict[str, Any]) -> float:
    """Match the tuner's balanced score without importing its runtime stack."""
    prompt = entry.get("prompt_eval_tok_s")
    generation = entry.get("generation_tok_s")
    if isinstance(prompt, (int, float)) and isinstance(generation, (int, float)):
        return math.sqrt(float(prompt) * float(generation))
    if isinstance(prompt, (int, float)):
        return float(prompt)
    if isinstance(generation, (int, float)):
        return float(generation)
    return 0.0


def aggregate_documents(docs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate trials without mixing metrics from different recipes/builds.

    Trials agree only when both their sanitised recipe and provenance match.
    Each group's metrics are medians, not independent best-of-axis values. The
    winning group is selected by the same balanced geometric-mean idea used by
    the tuner, while all grouped candidates remain in the bundle for auditing.
    """
    from arc_llama.recipe_share import (
        _clean_provenance,
        _clean_recipe_field,
        validate_submission,
    )

    by_fingerprint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, doc in enumerate(docs):
        problems = validate_submission(doc)
        if problems:
            raise ValueError(f"submission {index} is invalid: {'; '.join(problems)}")
        by_fingerprint[doc["fingerprint"]].append(doc)

    merged: dict[str, dict[str, Any]] = {}
    for fingerprint, trials in by_fingerprint.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        identities: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for doc in trials:
            recipe = _clean_recipe_field(doc.get("recipe"))
            provenance = _clean_provenance(doc.get("provenance"))
            group_key = json.dumps(
                {"recipe": recipe, "provenance": provenance},
                sort_keys=True,
                separators=(",", ":"),
            )
            grouped[group_key].append(doc)
            identities[group_key] = (recipe, provenance)

        candidates: list[dict[str, Any]] = []
        for group_key, group_trials in grouped.items():
            recipe, provenance = identities[group_key]
            newest = max(
                group_trials,
                key=lambda doc: str(doc.get("updated_at") or ""),
            )
            candidates.append(
                {
                    "recipe": recipe,
                    "prompt_eval_tok_s": _median_metric(group_trials, "prompt_eval_tok_s"),
                    "generation_tok_s": _median_metric(group_trials, "generation_tok_s"),
                    "gpu_name": newest.get("gpu_name", ""),
                    "arc_llama_version": newest.get("arc_llama_version", ""),
                    "provenance": provenance,
                    "submits": sum(max(1, int(doc.get("submits", 1))) for doc in group_trials),
                    "updated_at": newest.get("updated_at"),
                    "sample_count": len(group_trials),
                }
            )

        candidates.sort(
            key=lambda entry: (
                _balanced_score(entry),
                int(entry["submits"]),
                str(entry.get("updated_at") or ""),
            ),
            reverse=True,
        )
        winner = dict(candidates[0])
        winner["candidate_count"] = len(candidates)
        winner["candidates"] = candidates
        merged[fingerprint] = winner
    return merged


def main() -> int:
    try:
        from arc_llama.recipe_share import REGISTRY_SCHEMA
    except ImportError:
        print("arc-llama must be installed to aggregate", file=sys.stderr)
        return 2

    docs: list[dict[str, Any]] = []
    for f in sorted((REPO / "recipes").glob("*.json")):
        docs.append(json.loads(f.read_text()))
    merged = aggregate_documents(docs)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"schema": REGISTRY_SCHEMA, "recipes": merged}, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(merged)} recipes to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
