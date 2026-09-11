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
import math
import os
import tempfile
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

# Provenance fields that describe where and how a recipe was measured. These
# are *not* recipe edits; they travel in submission/entry metadata and help the
# consumer decide whether a shared recipe is trustworthy for their exact local
# build. Keeping them separate from the shareable recipe keys means old clients
# simply ignore them (the registry stays backwards-compatible).
_SHAREABLE_PROVENANCE_KEYS = (
    "llama_server_version",
    "llama_server_git",
    "llama_server_backend",
)

_ALLOWED_FA = ("on", "off", "auto")
_ALLOWED_KV = ("f16", "f32", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0")

_UBATCH_ALLOWED = (128, 256, 512, 1024, 2048, 4096)

MAX_REGISTRY_BYTES = 16 * 1024 * 1024
"""Largest registry accepted from disk or the network."""

MAX_REGISTRY_RECIPES = 100_000
MAX_SUBMITS = 1_000_000


class RegistryValidationError(ValueError):
    """A downloaded or bundled registry failed structural validation."""


def _empty_registry() -> dict[str, Any]:
    return {"schema": REGISTRY_SCHEMA, "recipes": {}}


def _valid_fingerprint(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _safe_submits(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        submits = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return submits if 1 <= submits <= MAX_SUBMITS else None


def _clean_provenance(raw: Any) -> dict[str, Any]:
    """Return sanitised provenance metadata; drop unknown/harmful fields."""
    out: dict[str, Any] = {}
    if not isinstance(raw, dict):
        return out
    for key in _SHAREABLE_PROVENANCE_KEYS:
        if key in raw and raw[key] is not None:
            value = raw[key]
            if key == "llama_server_backend" and value not in (
                "sycl",
                "vulkan",
                "cuda",
                "kompute",
                "metal",
            ):
                continue
            if not isinstance(value, str):
                continue
            out[key] = value[:256]
    return out


def llama_server_build_identity(binary_path: str | Path) -> dict[str, str]:
    """Inspect *binary_path* for the local llama-server build identity.

    Returns only safe, shareable provenance fields. If the binary is missing,
    unreadable, or back-end detection fails, returns an empty dict. This is
    the small explicit compatibility API that lets ``arc-llama tune --share``
    include local build identity in a community submission without changing the
    stable, shareable fingerprint.
    """
    out: dict[str, str] = {}
    path = Path(binary_path).expanduser()
    if not path.is_file():
        return out
    try:
        from arc_llama.binary import detect_llama_server_backend

        backend = detect_llama_server_backend(path)
        if backend is not None:
            out["llama_server_backend"] = backend.value
    except Exception:  # noqa: BLE001
        pass
    try:
        import subprocess

        proc = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        for line in text.splitlines():
            line_lower = line.lower()
            if line_lower.startswith("version:"):
                out["llama_server_version"] = line.split(":", 1)[1].strip()[:256]
            elif line_lower.startswith("commit:"):
                value = line.split(":", 1)[1].strip()
                # llama.cpp prints the short git hash after the colon.
                out["llama_server_git"] = value[:256]
    except Exception:  # noqa: BLE001
        pass
    return out


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
    # Provenance metadata (may be empty on old registry entries). Consumers can
    # use this to prefer recipes measured with a matching llama-server build.
    provenance: dict[str, str]
    # Normalised confidence signal derived from submits + provenance richness.
    confidence_score: float


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
                return load_registry_file(override)
            except (OSError, RegistryValidationError) as exc:
                log.warning("could not load recipe registry at %s: %s", override, exc)
        bundled = _bundled_path()
        if bundled is None:
            return _empty_registry()
        try:
            return load_registry_file(bundled)
        except (OSError, RegistryValidationError) as exc:
            log.warning("could not load bundled recipe registry: %s", exc)
            return _empty_registry()

    def lookup(self, fingerprint: str) -> SharedRecipe | None:
        recipes = self._data.get("recipes", {}) if isinstance(self._data, dict) else {}
        if not isinstance(recipes, dict):
            return None
        raw = recipes.get(fingerprint)
        if not isinstance(raw, dict):
            return None
        edits = _clean_recipe_field(raw.get("recipe", raw))
        if not edits:
            return None
        provenance = _clean_provenance(raw.get("provenance"))
        submits = _safe_submits(raw.get("submits", 1))
        if submits is None:
            return None
        return SharedRecipe(
            fingerprint=fingerprint,
            edits=edits,
            submits=submits,
            prompt_eval_tok_s=_maybe_float(raw.get("prompt_eval_tok_s")),
            generation_tok_s=_maybe_float(raw.get("generation_tok_s")),
            gpu_name=(raw.get("gpu_name", "") if isinstance(raw.get("gpu_name", ""), str) else "")[
                :256
            ],
            arc_llama_version=(
                raw.get("arc_llama_version", "")
                if isinstance(raw.get("arc_llama_version", ""), str)
                else ""
            )[:64],
            provenance=provenance,
            confidence_score=_confidence_score(submits, provenance),
        )


def _maybe_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and 0 < f <= 100_000 else None


def _confidence_score(submits: int, provenance: dict[str, str]) -> float:
    """Rough 0-1 score indicating how much a consumer should trust the entry.

    More independent submissions increases confidence; provenance metadata
    (especially the exact build identity) increases it further. The formula is
    intentionally simple and additive so old entries without provenance still
    get a sensible default. Max score is capped at 1.0.
    """
    base = min(0.5, submits / 10.0)
    bonus = 0.0
    if provenance:
        bonus += 0.15
    if provenance.get("llama_server_git"):
        bonus += 0.1
    if provenance.get("llama_server_version"):
        bonus += 0.1
    if provenance.get("llama_server_backend"):
        bonus += 0.1
    return min(1.0, round(base + bonus, 3))


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
    provenance: dict[str, str] | None = None,
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
        "provenance": _clean_provenance(provenance),
    }


def validate_submission(doc: Any) -> list[str]:
    """Return a list of problems; empty means the submission is acceptable.

    Used both by `recipes validate` locally and by the registry repo's CI.
    """
    problems: list[str] = []
    if not isinstance(doc, dict):
        return ["submission must be a JSON object"]
    fp = doc.get("fingerprint")
    if not _valid_fingerprint(fp):
        problems.append("fingerprint must be a lowercase 64-hex sha256")
    recipe = doc.get("recipe")
    cleaned = _clean_recipe_field(recipe)
    if not cleaned:
        problems.append(
            "recipe has no recognised fields (allowed: kv, fa, ubatch, batch, n_cpu_moe)"
        )
    if not isinstance(recipe, dict):
        problems.append("recipe must be a JSON object")
        recipe = {}
    unknown = set(recipe) - set(_SHAREABLE_RECIPE_KEYS)
    if unknown:
        problems.append(f"unknown recipe keys: {sorted(unknown)}")
    elif "ubatch" in recipe and "batch" in recipe:
        # Check the raw values too: _clean_recipe_field drops an out-of-range
        # batch, which would otherwise make batch<ubatch vanish silently.
        try:
            if int(recipe["batch"]) < int(recipe["ubatch"]):
                problems.append("batch must be >= ubatch")
        except (TypeError, ValueError):
            pass
    for key in ("prompt_eval_tok_s", "generation_tok_s"):
        v = doc.get(key)
        if v is not None and (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(float(v))
            or v <= 0
            or v > 100000
        ):
            problems.append(f"{key} must be a positive tok/s number")
    if not isinstance(doc.get("gpu_name", ""), str):
        problems.append("gpu_name must be a string")
    if not isinstance(doc.get("arc_llama_version", ""), str):
        problems.append("arc_llama_version must be a string")
    if _safe_submits(doc.get("submits", 1)) is None:
        problems.append(f"submits must be an integer from 1 to {MAX_SUBMITS}")
    prov = doc.get("provenance")
    if prov is not None:
        if not isinstance(prov, dict):
            problems.append("provenance must be a JSON object")
        else:
            unknown_prov = set(prov.keys()) - set(_SHAREABLE_PROVENANCE_KEYS)
            if unknown_prov:
                problems.append(f"unknown provenance keys: {sorted(unknown_prov)}")
            for prov_key in _SHAREABLE_PROVENANCE_KEYS:
                if (
                    prov_key in prov
                    and prov[prov_key] is not None
                    and not isinstance(prov[prov_key], str)
                ):
                    problems.append(f"provenance.{prov_key} must be a string")
            backend = prov.get("llama_server_backend")
            if backend is not None and backend not in (
                "sycl",
                "vulkan",
                "cuda",
                "kompute",
                "metal",
            ):
                problems.append("provenance.llama_server_backend must be a known backend")
    return problems


def validate_registry(doc: Any) -> list[str]:
    """Validate a complete downloaded registry, including every winning entry."""
    if not isinstance(doc, dict):
        return ["registry must be a JSON object"]
    problems: list[str] = []
    schema = doc.get("schema")
    if isinstance(schema, bool) or not isinstance(schema, int):
        problems.append("schema must be an integer")
    elif schema < 1 or schema > REGISTRY_SCHEMA:
        problems.append(f"unsupported registry schema {schema}")
    recipes = doc.get("recipes")
    if not isinstance(recipes, dict):
        problems.append("recipes must be a JSON object")
        return problems
    if len(recipes) > MAX_REGISTRY_RECIPES:
        problems.append(f"registry has more than {MAX_REGISTRY_RECIPES} recipes")
        return problems
    for fingerprint, entry in recipes.items():
        prefix = f"recipes[{fingerprint!r}]"
        if not _valid_fingerprint(fingerprint):
            problems.append(f"{prefix}: key must be a lowercase 64-hex sha256")
            continue
        if not isinstance(entry, dict):
            problems.append(f"{prefix}: entry must be a JSON object")
            continue
        recipe = entry.get("recipe")
        if not isinstance(recipe, dict) or not _clean_recipe_field(recipe):
            problems.append(f"{prefix}: recipe has no valid fields")
        elif set(recipe) - set(_SHAREABLE_RECIPE_KEYS):
            problems.append(f"{prefix}: recipe contains unknown fields")
        if _safe_submits(entry.get("submits", 1)) is None:
            problems.append(f"{prefix}: invalid submits count")
        for metric in ("prompt_eval_tok_s", "generation_tok_s"):
            if entry.get(metric) is not None and _maybe_float(entry.get(metric)) is None:
                problems.append(f"{prefix}: invalid {metric}")
        provenance = entry.get("provenance")
        if provenance is not None and (
            not isinstance(provenance, dict) or _clean_provenance(provenance) != provenance
        ):
            problems.append(f"{prefix}: invalid provenance")
        if len(problems) >= 50:
            problems.append("registry has additional validation errors")
            break
    return problems


def parse_registry_bytes(payload: bytes) -> dict[str, Any]:
    """Decode and validate a registry payload with a strict size bound."""
    if len(payload) > MAX_REGISTRY_BYTES:
        raise RegistryValidationError(
            f"registry exceeds the {MAX_REGISTRY_BYTES // (1024 * 1024)} MiB limit"
        )
    try:
        doc = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegistryValidationError(f"invalid registry JSON: {exc}") from exc
    problems = validate_registry(doc)
    if problems:
        raise RegistryValidationError("; ".join(problems))
    return doc


def load_registry_file(path: Path) -> dict[str, Any]:
    """Load a bounded, structurally valid registry from disk."""
    if path.stat().st_size > MAX_REGISTRY_BYTES:
        raise RegistryValidationError(
            f"registry exceeds the {MAX_REGISTRY_BYTES // (1024 * 1024)} MiB limit"
        )
    return parse_registry_bytes(path.read_bytes())


def write_registry_atomic(doc: dict[str, Any], destination: Path) -> None:
    """Validate and atomically replace the user registry."""
    payload = (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode()
    parse_registry_bytes(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def provenance_matches_local(
    provenance: dict[str, str],
    *,
    llama_server_version: str | None,
    llama_server_git: str | None,
    llama_server_backend: str | None,
) -> bool:
    """True when the shared provenance matches the local build identifiers.

    Missing local identifiers always return False so callers cannot accidentally
    claim a match when they know nothing about the current binary. Missing
    provenance fields from the registry entry also return False: a shared
    recipe without build identity is treated as unverified for this machine.
    """
    if not provenance:
        return False
    if llama_server_version and provenance.get("llama_server_version") != llama_server_version:
        return False
    if llama_server_git and provenance.get("llama_server_git") != llama_server_git:
        return False
    if llama_server_backend and provenance.get("llama_server_backend") != llama_server_backend:
        return False
    # Require at least one local identifier we actually tested, otherwise the
    # match is vacuous and misleading.
    if not any((llama_server_version, llama_server_git, llama_server_backend)):
        return False
    return True


def benchmark_improvement(
    baseline: Any,
    candidate: Any,
    *,
    target: str = "balanced",
    priority: str | None = None,
) -> float | None:
    """Fractional candidate improvement, or None when either result is invalid."""
    from arc_llama.tune import score_result

    baseline_score = score_result(baseline, target, priority)
    candidate_score = score_result(candidate, target, priority)
    if baseline_score is None or candidate_score is None or baseline_score <= 0:
        return None
    return candidate_score / baseline_score - 1.0


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
