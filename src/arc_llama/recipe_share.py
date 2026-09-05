"""Community tune-recipe registry (the "recipe share" fast path).

The registry is a *cache*, never a dependency. A hit skips a tune sweep the
user was going to run anyway; a miss or a stale entry falls back to exactly
today's behaviour (local sweep via `arc-llama tune` or the background
auto-tuner). Nothing here ever talks to the network implicitly.

Layout
------
The registry ships as a single bundled ``recipes.json`` inside the wheel
(regenerated each release from the community repo by a GitHub Actions job)
and can optionally be refreshed from a newer release asset with
``arc-llama recipes update``.

Schema
------
The top level is ``{"schema": 1, "recipes": {<fingerprint>: <entry>}}``.
A fingerprint here is the *community* fingerprint: sha256 over the stable,
shareable parts of what a recipe depends on — GPU arch, backend, model
architecture/parameter count class, workload profile key, and
``TUNE_SCHEMA_VERSION`` — deliberately *not* the local machine fingerprint
from :mod:`arc_llama.autotune`, which mixes in absolute file paths and
mtimes and would never match across machines.

An entry holds the winning recipe edits and measurement context::

    {
        "kv": "q8_0", "fa": "auto", "ubatch": 1024, "batch": 2048,
        "n_cpu_moe": null,
        "prompt_eval_tok_s": 421.3, "generation_tok_s": 19.8,
        "submits": 7,
        "arc_llama_version": "0.7.1",
        "gpu_name": "Arc Pro B60",
        "updated_at": 1754000000
    }
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

log = logging.getLogger("arc_llama.recipe_share")

REGISTRY_SCHEMA = 1
"""Bump when the bundled JSON shape changes in a backwards-incompatible way."""

DEFAULT_REGISTRY_URL = (
    "https://github.com/offbyonebit/arc-llama-recipes/releases/latest/download/recipes.json"
)
"""Where `recipes update` fetches the fresh bundle from. A plain static file
on github.com — same trust level as installing the package itself."""

# Recipe fields we are willing to accept from the community. Anything outside
# this allowlist in a submitted/loaded entry is dropped: the registry must not
# become a vector for arbitrary llama-server flags (extra_flags is excluded
# on purpose; override_tensor likewise).
_SHAREABLE_RECIPE_KEYS = (
    "kv",
    "fa",
    "ubatch",
    "batch",
    "n_cpu_moe",
)

_ALLOWED_FA = ("on", "off", "auto")
_ALLOWED_KV = ("f16", "f32", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0")

_UBATCH_ALLOWED = (128, 256, 512, 1024, 2048, 4096)


def share_fingerprint(
    gpu_arch: str,
    backend: str,
    model_class: str,
    workload_key: str,
    tune_schema_version: int,
    vram_mb: int,
) -> str:
    """Stable *shareable* fingerprint: what a recipe is valid for.

    Deliberately excludes everything machine-local (paths, mtimes, pci slot,
    llama-server build). Two users with the same card class, backend, model
    class and workload should produce the same fingerprint and be able to
    reuse each other's measurements.
    """
    h = hashlib.sha256()
    h.update(f"arch:{gpu_arch}".encode())
    h.update(f"|backend:{backend}".encode())
    h.update(f"|model_class:{model_class}".encode())
    h.update(f"|workload:{workload_key}".encode())
    h.update(f"|schema:{tune_schema_version}".encode())
    h.update(f"|vram_bucket:{_vram_bucket(vram_mb)}".encode())
    return h.hexdigest()


def _vram_bucket(vram_mb: int) -> int:
    """VRAM bucket in GB, floored to the nearest nominal breakpoint.

    Floor, not round: a recipe tuned on a 12GB card fits on anything bigger,
    but a 24GB recipe must never match a 20GB card. Cards report slightly
    under their nominal size (a "12GB" B580 often reports ~11.7GB), so a
    breakpoint matches within 1GB below nominal. Anything below the first
    breakpoint buckets at 4GB.
    """
    gb = vram_mb / 1024
    bucket = 4
    for bound in (4, 8, 12, 16, 24, 32, 48):
        if gb >= bound - 1:
            bucket = bound
    return bucket * 1024


def _clean_recipe_field(raw: Any) -> dict[str, Any]:
    """Keep only known-safe recipe keys with valid values; drop the rest."""
    out: dict[str, Any] = {}
    if not isinstance(raw, dict):
        return out
    if "kv" in raw:
        kv = str(raw["kv"]).lower()
        if kv in _ALLOWED_KV:
            out["kv"] = kv
    if "fa" in raw and raw["fa"] in _ALLOWED_FA:
        out["fa"] = raw["fa"]
    if "ubatch" in raw:
        try:
            ub = int(raw["ubatch"])
        except (TypeError, ValueError):
            ub = None
        else:
            if ub in _UBATCH_ALLOWED:
                out["ubatch"] = ub
    if "batch" in raw and raw.get("ubatch") is not None:
        try:
            b = int(raw["batch"])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            b = None
        if b is not None and b >= 0 and (out.get("ubatch") is None or b >= out["ubatch"]):
            out["batch"] = b
    if "n_cpu_moe" in raw and raw.get("n_cpu_moe") is not None:
        try:
            moe = int(raw["n_cpu_moe"])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            moe = None
        if moe is not None and moe >= 0:
            out["n_cpu_moe"] = moe
    return out


@dataclass
class SharedRecipe:
    """A recipe entry loaded from the registry, already sanitised."""

    fingerprint: str
    edits: dict[str, Any]
    submits: int
    prompt_eval_tok_s: float | None
    generation_tok_s: float | None
    gpu_name: str
    arc_llama_version: str


def _bundled_path() -> Path | None:
    """Locate the bundled recipes.json inside the installed package."""
    try:
        res = resources.files("arc_llama").joinpath("data/recipes.json")
        if res.is_file():
            return Path(str(res))
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    # Source checkout fallback
    local = Path(__file__).resolve().parent / "data" / "recipes.json"
    return local if local.is_file() else None


def _user_override_path() -> Path:
    """User-supplied registry that wins over the bundled one."""
    return Path(
        os.environ.get(
            "ARC_LLAMA_RECIPES_PATH",
            Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
            / "arc-llama"
            / "recipes.json",
        )
    )


class RecipeRegistry:
    """Read-only lookup over the bundled + user-installed recipe database."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data = data if data is not None else self._load()

    @staticmethod
    def _load() -> dict[str, Any]:
        override = _user_override_path()
        if override.is_file():
            try:
                doc = json.loads(override.read_text())
                if isinstance(doc, dict) and int(doc.get("schema", 0)) <= REGISTRY_SCHEMA:
                    return doc
            except (OSError, ValueError, TypeError):
                log.warning("could not parse recipe registry at %s", override)
        bundled = _bundled_path()
        if bundled is None:
            return {"schema": REGISTRY_SCHEMA, "recipes": {}}
        try:
            doc = json.loads(bundled.read_text())
        except (OSError, ValueError):
            log.warning("could not parse bundled recipe registry")
            return {"schema": REGISTRY_SCHEMA, "recipes": {}}
        if not isinstance(doc, dict) or int(doc.get("schema", 0)) > REGISTRY_SCHEMA:
            return {"schema": REGISTRY_SCHEMA, "recipes": {}}
        return doc

    def lookup(self, fingerprint: str) -> SharedRecipe | None:
        raw = self._data.get("recipes", {}).get(fingerprint)
        if not isinstance(raw, dict):
            return None
        edits = _clean_recipe_field(raw.get("recipe", raw))
        if not edits:
            return None
        return SharedRecipe(
            fingerprint=fingerprint,
            edits=edits,
            submits=int(raw.get("submits", 1) or 1),
            prompt_eval_tok_s=_maybe_float(raw.get("prompt_eval_tok_s")),
            generation_tok_s=_maybe_float(raw.get("generation_tok_s")),
            gpu_name=str(raw.get("gpu_name", "")),
            arc_llama_version=str(raw.get("arc_llama_version", "")),
        )


def _maybe_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


def shared_recipe_edits_to_model_recipe(edits: dict[str, Any]) -> dict[str, Any]:
    """Map registry edit keys onto ModelConfig.recipe keys."""
    out: dict[str, Any] = {}
    if "kv" in edits:
        out["cache_type_k"] = edits["kv"]
        out["cache_type_v"] = edits["kv"]
    if "fa" in edits:
        out["flash_attn"] = edits["fa"]
    if "ubatch" in edits:
        out["ubatch_size"] = edits["ubatch"]
    if "batch" in edits:
        out["batch_size"] = edits["batch"]
    if "n_cpu_moe" in edits:
        out["n_cpu_moe"] = edits["n_cpu_moe"]
    return out


def submission_document(
    fingerprint: str,
    recipe: dict[str, Any],
    prompt_eval_tok_s: float | None,
    generation_tok_s: float | None,
    gpu_name: str,
    arc_llama_version: str,
    submits: int = 1,
) -> dict[str, Any]:
    """The JSON blob a user submits (and CI validates) for one fingerprint."""
    return {
        "fingerprint": fingerprint,
        "recipe": _clean_recipe_field(recipe),
        "prompt_eval_tok_s": prompt_eval_tok_s,
        "generation_tok_s": generation_tok_s,
        "gpu_name": gpu_name,
        "arc_llama_version": arc_llama_version,
        "submits": submits,
        "schema": REGISTRY_SCHEMA,
    }


def validate_submission(doc: Any) -> list[str]:
    """Return a list of problems; empty means the submission is acceptable.

    Used both by `recipes validate` locally and by the registry repo's CI.
    """
    problems: list[str] = []
    if not isinstance(doc, dict):
        return ["submission must be a JSON object"]
    fp = doc.get("fingerprint")
    if not isinstance(fp, str) or len(fp) != 64 or any(c not in "0123456789abcdef" for c in fp):
        problems.append("fingerprint must be a lowercase 64-hex sha256")
    cleaned = _clean_recipe_field(doc.get("recipe"))
    if not cleaned:
        problems.append("recipe has no recognised fields (allowed: kv, fa, ubatch, batch, n_cpu_moe)")
    unknown = set(doc.get("recipe", {}) or {}) - set(_SHAREABLE_RECIPE_KEYS)
    if unknown:
        problems.append(f"unknown recipe keys: {sorted(unknown)}")
    elif "ubatch" in (doc.get("recipe") or {}) and "batch" in (doc.get("recipe") or {}):
        # Check the raw values too: _clean_recipe_field drops an out-of-range
        # batch, which would otherwise make batch<ubatch vanish silently.
        try:
            if int(doc["recipe"]["batch"]) < int(doc["recipe"]["ubatch"]):
                problems.append("batch must be >= ubatch")
        except (TypeError, ValueError):
            pass
    for key in ("prompt_eval_tok_s", "generation_tok_s"):
        v = doc.get(key)
        if v is not None and (not isinstance(v, (int, float)) or v <= 0 or v > 100000):
            problems.append(f"{key} must be a positive tok/s number")
    if not isinstance(doc.get("gpu_name", ""), str):
        problems.append("gpu_name must be a string")
    if not isinstance(doc.get("arc_llama_version", ""), str):
        problems.append("arc_llama_version must be a string")
    return problems


def build_pr_body(doc: dict[str, Any]) -> str:
    """Human-readable PR body for a submitted recipe."""
    r = doc.get("recipe", {})
    lines = [
        "## Shared tune recipe",
        "",
        f"- Fingerprint: `{doc.get('fingerprint', '')[:16]}…`",
        f"- GPU: {doc.get('gpu_name', 'unknown')}",
        f"- arc-llama: {doc.get('arc_llama_version', '?')}",
        f"- Measured: {doc.get('prompt_eval_tok_s', '?')} pp tok/s · "
        f"{doc.get('generation_tok_s', '?')} gen tok/s",
        "",
        "```json",
        json.dumps(r, indent=2, sort_keys=True),
        "```",
        "",
        "_Submitted via `arc-llama tune --share`. CI validates schema and bounds._",
    ]
    return "\n".join(lines)
