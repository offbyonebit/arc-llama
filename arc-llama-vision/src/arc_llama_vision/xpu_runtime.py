"""Intel XPU runtime sanity checks for GPU deployments of torch XPU.

Why this exists: a hard, hard-to-diagnose failure mode on Arc hosts that
also have a system oneAPI install. ``torch`` XPU wheels bundle their own
matched SYCL runtime (``libsycl.so.8``) plus Unified Runtime (UR) loader
and adapters. The UR soname tag (``libur_loader.so.0``) did not change
across oneAPI releases, so an ``LD_LIBRARY_PATH`` exported by a system
oneAPI ``setvars.sh`` silently wins library resolution and binds torch's
SYCL runtime to a newer, mismatched UR loader and adapters. The first
JIT-built device kernel then crashes inside the UR adapter. Observed in
production: SIGSEGV in ``urProgramBuildExp`` -> ``strlen`` while building
the ``index_select`` kernel for ``torch.nn.functional.embedding``, in other
words exactly a GGUF text encoder's token-embedding lookup during the first
prompt encode of Flux2TEModel_ on Intel XPU.

This module is stdlib-only so any component (core, companion, plugin) can
import it without heavy dependencies:

* :func:`detect_ur_runtime_mixing` reads library bindings of a live process
  without touching the GPU and flags a mismatched UR loader;
* :func:`sanitize_launch_env` builds a clean environment for spawning a
  torch XPU process, pinned to the wheel's bundled runtime (the same fix
  the corrected ComfyUI ``start.sh`` applies);
* :func:`describe_binding` returns a human-readable diagnosis for logs.

No function here initializes a device or moves tensors; the module only
reads process mapping state and the environment.
"""

from __future__ import annotations

import os
from typing import Any

# Environment variables that oneAPI's setvars.sh exports, all of which can
# point a torch XPU process at a system oneAPI runtime tree and shadow the
# bundled, matched one. A sanitized launcher strips these before exec.
_ONEAPI_EXPORTED_ENV = (
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "C_INCLUDE_PATH",
    "PKG_CONFIG_PATH",
    "CMAKE_PREFIX_PATH",
)

# Device-layer variables that must be kept: they select and tune the Intel
# Level Zero stack without pinning any runtime library version.
_KEEP_ENV_DEFAULTS = {
    "ONEAPI_DEVICE_SELECTOR": "level_zero:0",
    "ZES_ENABLE_SYSMAN": "1",
    # Persistent SYCL program cache is broken on this host's Xe2 stack;
    # keep the working default unless the caller overrides it.
    "SYCL_CACHE_PERSISTENT": "0",
}

# A same-tree heuristic: the oneAPI compiler tree path appears exactly once
# in both a loader and an adapter shipped from that same tree.
_ONEAPI_COMPILER_MARK = "/oneapi/compiler/"


def _proc_map_paths(pid: int) -> list[str]:
    """Return the mapped file paths of ``pid`` (empty on unreadable state)."""
    try:
        with open(f"/proc/{pid}/maps", encoding="utf-8", errors="replace") as fh:
            return [line.split()[-1] for line in fh if "/" in line]
    except OSError:
        return []


def _runtime_tree(path: str | None) -> str | None:
    """Classify ``path`` into the installation tree it was loaded from.

    Realpaths are resolved first, because wheel-bundled runtimes are reached
    through RPATH indirection like ``.../site-packages/torch/lib/../../../..
    /libsycl.so``; after resolution the file sits in the venv's own ``lib``
    directory next to its matched UR loader. Returns one of:

    * ``"oneapi:<compiler-tree-dir>"`` — a system oneAPI compiler tree;
    * ``"libdir:<containing-dir>"`` — anything else, keyed on the real
      directory holding the library (two different venvs therefore
      classify into different trees, as they should).
    """
    if path is None:
        return None
    resolved = os.path.realpath(path)
    if _ONEAPI_COMPILER_MARK in resolved:
        # Keep the version segment: /oneapi/compiler/<ver>/lib/...
        rest = resolved[resolved.index(_ONEAPI_COMPILER_MARK) + len(_ONEAPI_COMPILER_MARK) :]
        version = rest.split("/")[0]
        root = resolved[: resolved.index(_ONEAPI_COMPILER_MARK)]
        return f"oneapi:{root}{_ONEAPI_COMPILER_MARK.rstrip('/')}/{version}"
    return f"libdir:{os.path.dirname(resolved)}"


def detect_ur_runtime_mixing(pid: int) -> dict[str, Any]:
    """Inspect ``pid`` for a SYCL/UR version mismatch, without GPU access.

    Returns a report dict. ``diagnosis`` is one of:

    * ``"mixed"``: the process's SYCL runtime and UR loader come from
      different installation trees. This is exactly the crash precondition
      observed on Arc hosts: the first JIT-built device kernel can
      segfault inside the UR adapter (observed on the ``index_select``
      kernel that backs ``torch.nn.functional.embedding``).
    * ``"unavailable"``: process map state could not be read (permissions,
      non-Linux host, or the process exited). Callers should skip the
      check, not fail.
    * ``"clean"``: runtime pieces resolve from the same tree, or too few
      pieces are visible to tell.
    """
    paths = _proc_map_paths(pid)
    if not paths:
        return {"diagnosis": "unavailable"}

    sycl = next((p for p in paths if "libsycl.so" in p), None)
    ur_loader = next((p for p in paths if "/libur_loader.so" in p), None)
    ur_adapter = next((p for p in paths if "/libur_adapter_level_zero" in p), None)

    sycl_tree = _runtime_tree(sycl)
    loader_tree = _runtime_tree(ur_loader)

    report: dict[str, Any] = {
        "pid": pid,
        "diagnosis": "clean",
        "sycl_runtime": sycl,
        "ur_loader": ur_loader,
        "ur_adapter": ur_adapter,
        "sycl_tree": sycl_tree,
        "loader_tree": loader_tree,
    }

    if sycl and ur_loader and sycl_tree != loader_tree:
        # Example observed pairing: libsycl.so.8 built with DPC++ 2025.3.2
        # (torch wheel bundled at venv/lib) resolving libur_loader.so.0 and
        # level-zero adapters from a system oneAPI 2026.1 tree because a
        # login shell sourced its setvars.sh. Both trees export the same
        # UR soname, so dlopen binds silently; the adapter then reads the
        # older caller's build-options pointer as a different layout and
        # dereferences garbage (strlen on 0xffffffff) during the first
        # device kernel JIT build.
        report["diagnosis"] = "mixed"
        report["detail"] = (
            "SYCL runtime and Unified Runtime loader come from different "
            f"installation trees: realpath({sycl}) [{sycl_tree}] loaded "
            f"{ur_loader} [{loader_tree}]. This pairing can segfault inside "
            "urProgramBuildExp on the first JIT-built device kernel "
            "(observed: torch.nn.functional.embedding -> index_select on "
            "Intel XPU). Launch the GPU process with a sanitized "
            "environment; see sanitize_launch_env()."
        )
    return report


def sanitize_launch_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return a launch environment with oneAPI library shadows removed.

    Mirrors the runtime pinning in the corrected ComfyUI ``start.sh``:
    every variable a system ``setvars.sh`` exports to point at its own
    runtime tree is dropped, so the torch XPU wheel resolves its bundled
    ``libsycl`` / UR / MKL stack through its own RPATH instead. Device
    selection variables are preserved with working defaults when absent.
    """
    env = dict(base if base is not None else os.environ)
    for var in _ONEAPI_EXPORTED_ENV:
        env.pop(var, None)
    for var, default in _KEEP_ENV_DEFAULTS.items():
        if var not in env or not env[var]:
            env[var] = default
    return env


def describe_binding(pid: int) -> str:
    """One-line human-readable diagnosis for logs, or ``""`` when clean."""
    report = detect_ur_runtime_mixing(pid)
    if report.get("diagnosis") == "mixed":
        return str(report.get("detail"))
    return ""


__all__ = [
    "detect_ur_runtime_mixing",
    "sanitize_launch_env",
    "describe_binding",
]
