"""Shipped runtime exclusions; this list is never fetched from the network."""
from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path


def known_bad_reason(tag: str, os_name: str, backend: str) -> str | None:
    match = re.match(r"b(\d+)\b", tag)
    if not match:
        return None
    build = int(match[1])
    entries = json.loads(resources.files("arc_llama").joinpath("data/known_bad_runtimes.json").read_text(encoding="utf-8"))
    for entry in entries:
        if entry["platform"] != os_name.lower() or entry["backend"] != backend:
            continue
        lower = int(entry["min_tag"][1:])
        upper = int(entry["max_tag"][1:]) if entry.get("max_tag") else None
        if build >= lower and (upper is None or build <= upper):
            return str(entry["reason"])
    return None


def installed_runtime_warning(binary: Path, os_name: str, backend: str | None) -> str | None:
    if os_name != "windows":
        return None
    try:
        marker = json.loads((binary.resolve().parent / ".arc-llama-runtime.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        marker = {}
    if not isinstance(marker, dict):
        marker = {}
    tag = str(marker.get("tag") or "")
    backend = marker.get("backend") or backend
    if not tag and backend == "sycl":
        from arc_llama.benchmark import llama_cpp_build

        tag = llama_cpp_build(str(binary)) or ""
    reason = known_bad_reason(tag, os_name, backend or "")
    return f"{tag}: {reason}" if reason else None
