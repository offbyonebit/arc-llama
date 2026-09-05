#!/usr/bin/env python3
"""Aggregate recipes/*.json into the single bundled dist/recipes.json.

One file per fingerprint is the canonical source; the bundle merges
duplicate submissions for the same fingerprint by keeping the best
measured recipe and summing submit counts, so `submits` reflects how
many independent sweeps agreed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "dist" / "recipes.json"


def main() -> int:
    try:
        from arc_llama.recipe_share import REGISTRY_SCHEMA, _clean_recipe_field
    except ImportError:
        print("arc-llama must be installed to aggregate", file=sys.stderr)
        return 2

    merged: dict[str, dict] = {}
    for f in sorted((REPO / "recipes").glob("*.json")):
        doc = json.loads(f.read_text())
        fp = doc["fingerprint"]
        cur = merged.get(fp)
        if cur is None:
            merged[fp] = {
                "recipe": _clean_recipe_field(doc.get("recipe")),
                "prompt_eval_tok_s": doc.get("prompt_eval_tok_s"),
                "generation_tok_s": doc.get("generation_tok_s"),
                "gpu_name": doc.get("gpu_name", ""),
                "arc_llama_version": doc.get("arc_llama_version", ""),
                "submits": int(doc.get("submits", 1)),
                "updated_at": doc.get("updated_at"),
            }
            continue
        # Duplicate fingerprint: keep the faster measurement, count the submit.
        cur["submits"] += int(doc.get("submits", 1))
        for axis in ("prompt_eval_tok_s", "generation_tok_s"):
            new_v, old_v = doc.get(axis), cur.get(axis)
            if isinstance(new_v, (int, float)) and (not isinstance(old_v, (int, float)) or new_v > old_v):
                cur[axis] = new_v
                cur["recipe"] = _clean_recipe_field(doc.get("recipe"))
                cur["gpu_name"] = doc.get("gpu_name", cur["gpu_name"])
                cur["arc_llama_version"] = doc.get("arc_llama_version", cur["arc_llama_version"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"schema": REGISTRY_SCHEMA, "recipes": merged}, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(merged)} recipes to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
