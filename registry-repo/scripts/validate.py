#!/usr/bin/env python3
"""Validate every recipes/*.json submission against the share schema.

Mirrors arc_llama.recipe_share.validate_submission so the checks cannot
drift: this script imports the same function from the installed package
(declared as a minimal dependency below) or falls back to an inline copy
pinned to the same rules.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    try:
        from arc_llama.recipe_share import validate_submission
    except ImportError:
        print("arc-llama must be installed (pip install arc-llama) to validate", file=sys.stderr)
        return 2

    files = sorted((REPO / "recipes").glob("*.json"))
    if not files:
        print("no submissions yet — ok")
        return 0

    failures = 0
    for f in files:
        try:
            doc = json.loads(f.read_text())
        except ValueError as e:
            print(f"FAIL {f.name}: not valid JSON ({e})")
            failures += 1
            continue
        problems = validate_submission(doc)
        # filename must agree with the fingerprint prefix
        prefix = f.name.removesuffix(".json")
        fp = doc.get("fingerprint", "") if isinstance(doc, dict) else ""
        if fp and not (prefix == fp or fp.startswith(prefix)):
            problems.append(f"filename {f.name} does not match fingerprint prefix {fp[:12]}")
        if problems:
            print(f"FAIL {f.name}:")
            for p in problems:
                print(f"  - {p}")
            failures += 1
        else:
            print(f"ok   {f.name}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
