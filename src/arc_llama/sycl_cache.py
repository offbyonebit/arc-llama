"""Managed persistent SYCL JIT cache.

Battlemage with libsycl.so.9 from oneAPI 2026.0 reproducibly SIGSEGVs in
`PersistentDeviceCodeCache::getItemFromDisc` when `SYCL_CACHE_PERSISTENT=1`
reads back cache entries written by a *different* stack (older llama-server
build, different driver, interrupted write). The stock workaround — disabling
the cache outright — costs ~20 s of JIT recompilation on every cold start.

This module keeps persistence instead of giving it up, made safe two ways:

1. **Fingerprint isolation.** Every (llama-server binary, kernel GPU driver)
   combination gets its own private cache directory under
   `<state_dir>/sycl-cache/<fingerprint>`. A cache is only ever read by the
   exact stack that wrote it, so the stale-entry crash has nothing to bite on.
   Upgrading llama.cpp or the driver simply lands in a fresh directory; old
   ones are pruned.

2. **Crash guard.** The launcher drops a `warming` marker in the cache dir
   before spawning llama-server and removes it once the health check passes
   (or on a clean stop). If we ever find a leftover marker, the previous run
   died mid-warm-up — we wipe the directory, poison that fingerprint, and fall
   back to `SYCL_CACHE_PERSISTENT=0`. Worst case is exactly the behaviour
   arc-llama shipped with; best case (the common one) cold starts pay the JIT
   cost once per llama-server build instead of every time.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("arc_llama.sycl_cache")

POISON_FILE = "POISONED"
MARKER_FILE = "warming"
KEEP_RECENT_CACHES = 2
"""Old fingerprint dirs kept alongside the active one (rollbacks are cheap)."""

_DRIVER_VERSION_PROBES = (
    Path("/sys/module/xe/srcversion"),
    Path("/sys/module/i915/srcversion"),
)


@dataclass
class JitCachePlan:
    """Outcome of preparing the managed cache for one launch."""
    enabled: bool
    reason: str
    env: dict[str, str] = field(default_factory=dict)
    """Env overrides to layer on top of the arch profile (may re-enable
    SYCL_CACHE_PERSISTENT that the profile disabled)."""
    marker: Path | None = None
    """Warm-up marker the launcher must write before spawn and clear after
    the first successful health check."""
    cache_dir: Path | None = None


def binary_fingerprint(llama_server: str) -> str | None:
    """Fingerprint the llama-server binary + GPU driver combination.

    Uses (resolved path, size, mtime) rather than hashing the multi-hundred-MB
    binary; a rebuild or upgrade always changes at least one of those. Returns
    None if the binary can't be found — callers must fall back to disabled.
    """
    exe = shutil.which(llama_server) or llama_server
    p = Path(exe).expanduser()
    try:
        p = p.resolve()
        st = p.stat()
    except OSError:
        return None
    driver = ""
    for probe in _DRIVER_VERSION_PROBES:
        try:
            driver += probe.read_text().strip()
        except OSError:
            continue
    raw = f"{p}|{st.st_size}|{st.st_mtime_ns}|{driver}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _wipe_cache_entries(cache_dir: Path) -> None:
    """Remove everything in the dir except the poison file."""
    try:
        for entry in cache_dir.iterdir():
            if entry.name == POISON_FILE:
                continue
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
    except OSError:
        pass


def _prune_old_caches(root: Path, current: str) -> None:
    """Delete fingerprint dirs beyond the KEEP_RECENT_CACHES most recent."""
    try:
        siblings = [
            d for d in root.iterdir()
            if d.is_dir() and d.name != current
        ]
    except OSError:
        return
    siblings.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    for stale in siblings[KEEP_RECENT_CACHES:]:
        log.info("pruning stale SYCL JIT cache %s", stale)
        shutil.rmtree(stale, ignore_errors=True)


def prepare_jit_cache(state_dir: str | Path, llama_server: str) -> JitCachePlan:
    """Prepare the managed cache dir for a launch and return env + marker.

    Call once per llama-server spawn. Never raises — any filesystem trouble
    degrades to a disabled plan, which leaves the arch profile's conservative
    `SYCL_CACHE_PERSISTENT=0` in effect.
    """
    fp = binary_fingerprint(llama_server)
    if fp is None:
        return JitCachePlan(
            enabled=False,
            reason=f"llama-server binary not found for fingerprinting: {llama_server}",
        )
    root = Path(state_dir).expanduser() / "sycl-cache"
    cache_dir = root / fp
    poison = cache_dir / POISON_FILE
    marker = cache_dir / MARKER_FILE

    if poison.exists():
        return JitCachePlan(
            enabled=False,
            reason="fingerprint poisoned by an earlier warm-up crash "
                   "(delete the POISONED file to retry)",
            cache_dir=cache_dir,
        )

    if marker.exists():
        # Previous run died between spawn and first health check with this
        # cache active. Assume the persistent cache is implicated: wipe it and
        # never re-enable for this fingerprint. A user Ctrl-C during warm-up
        # also lands here — deliberately conservative, and self-documenting on
        # disk via the poison file.
        log.warning(
            "leftover warm-up marker in %s — previous run crashed during SYCL "
            "JIT warm-up; wiping and poisoning this cache", cache_dir,
        )
        _wipe_cache_entries(cache_dir)
        try:
            poison.write_text(
                f"poisoned {time.strftime('%Y-%m-%dT%H:%M:%S%z')}: previous "
                "llama-server run crashed before its first health check while "
                "this persistent JIT cache was enabled.\n"
            )
        except OSError:
            pass
        return JitCachePlan(
            enabled=False,
            reason="previous run crashed during JIT warm-up; cache wiped and poisoned",
            cache_dir=cache_dir,
        )

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return JitCachePlan(enabled=False, reason=f"cannot create cache dir: {e}")
    _prune_old_caches(root, current=fp)

    return JitCachePlan(
        enabled=True,
        reason=f"managed persistent JIT cache at {cache_dir}",
        env={
            "SYCL_CACHE_PERSISTENT": "1",
            "SYCL_CACHE_DIR": str(cache_dir),
        },
        marker=marker,
        cache_dir=cache_dir,
    )
