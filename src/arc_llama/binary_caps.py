"""Capability detection for the SYCL ``llama-server`` binary.

arc-llama cannot assume every ``llama-server`` was built the same way. A user's
binary may or may not contain the oneDNN XMX SDPA path, may or may not have the
fixed Q8_0 weight reorder kernels, and may be AOT-compiled or JIT. Recipes that
guess wrong actively hurt: recommending ``-fa on`` + f16 KV to a binary without
the oneDNN path costs ~10% decode and buys nothing, and warning about "slow
Q8_0 weights" on a fixed binary is simply wrong advice.

This module answers those questions by inspecting the shipped
``libggml-sycl.so``. It never runs inference, never touches the GPU, and never
launches the server. Results are cached per (resolved library path, mtime).

Every field is **tri-state**: ``True`` / ``False`` / ``None`` for "cannot
determine". Callers must treat ``None`` as "do not make a claim either way" --
on a stripped binary we would rather skip an optimization than emit a confident
falsehood.

Detection methods, and how far each is actually trusted:

* **oneDNN (XMX SDPA)** -- ``ldd`` first. A successful ``ldd`` is authoritative
  in both directions: the linkage either is or is not there. Only if ``ldd`` is
  unavailable do we fall back to scanning for symbol strings, which can
  false-positive on debug text.
* **Q8_0 weight fast path** -- symbol scan. The needles below were verified
  present in a build known to contain the fix (llama.cpp #21517). Note that
  ``ggml_sycl_supports_reorder_mmvq`` is declared ``inline`` upstream and is
  therefore *never* emitted as a symbol -- do not add it as a needle.
* **AOT vs JIT** -- ``spir64_gen`` appears in AOT-compiled device images. This
  is a whole-binary answer, not a per-arch one: the concrete device name passed
  via ``-Xsycl-target-backend "-device <name>"`` is *not* embedded verbatim
  (verified: a binary built with ``-device bmg-g31`` contains ``bmg`` and
  ``spir64_gen`` but not ``bmg-g31``). We therefore do not attempt to report
  which arch a binary was AOT-compiled for.
"""

from __future__ import annotations

import logging
import mmap
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("arc_llama.binary_caps")

#: Shared-library dependency substrings that mean oneDNN is linked.
_ONEDNN_DEPS = ("dnnl", "onednn")

#: Symbol-name substrings that indicate oneDNN, used only when ``ldd`` is
#: unavailable. Deliberately narrow to limit false positives.
_ONEDNN_NEEDLES: tuple[bytes, ...] = (b"libdnnl.so", b"dnnl_sycl", b"dnnl_primitive")

#: Symbols proving the fixed Q8_0 weight MMVQ+reorder path is compiled in.
#: All three verified present in a build carrying the #21517 fix.
_Q8_REORDER_NEEDLES: tuple[bytes, ...] = (
    b"dequantize_mul_mat_vec_q8_0_sycl_reorder",
    b"ggml_sycl_mul_mat_vec_q_id_reorder",
    b"quantize_and_reorder_q8_1_soa",
)

#: Present in AOT-compiled device images.
_AOT_NEEDLES: tuple[bytes, ...] = (b"spir64_gen",)

#: If this is missing the library is almost certainly stripped, and every
#: negative symbol result becomes meaningless.
_SYMBOL_MARKER: tuple[bytes, ...] = (b"ggml_sycl",)

_LDD_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class SyclCaps:
    """What a given ``llama-server``'s SYCL backend can actually do.

    ``None`` on any field means "undetermined" -- callers must not treat it as
    a negative.
    """

    has_onednn_sdpa: bool | None = None
    """oneDNN linked, i.e. the XMX SDPA flash-attention path exists."""

    has_fast_q8_weights: bool | None = None
    """The fixed Q8_0 weight MMVQ+reorder path exists (llama.cpp #21517)."""

    is_aot: bool | None = None
    """Device code is AOT-compiled, so there is no JIT cost to cache away."""

    has_symbols: bool = False
    """False means the library looks stripped; negatives above are unreliable."""

    lib_path: Path | None = None
    """The ``libggml-sycl.so`` that was inspected, if one was found."""

    probed: bool = False
    """True when a library was located and scanned at all."""


def _resolve(path_like: str | Path) -> Path | None:
    """Resolve to a real path, following symlinks. ``None`` if unresolvable."""
    try:
        return Path(path_like).expanduser().resolve(strict=False)
    except OSError:
        return None


def find_sycl_lib(llama_server: str) -> Path | None:
    """Locate the ``libggml-sycl.so`` backing *llama_server*.

    Symlinks are resolved, which matters: a build tree typically ships
    ``libggml-sycl.so -> .so.0 -> .so.0.15.1`` alongside several older
    versioned copies, and only the resolved target is the live one.
    """
    exe_str = shutil.which(llama_server) or llama_server
    exe = _resolve(exe_str)
    if exe is None:
        return None

    # Overwhelmingly the common layout: next to the executable.
    sibling = exe.parent / "libggml-sycl.so"
    if sibling.exists():
        return _resolve(sibling)

    # Otherwise ask the loader where it comes from (handles rpath/LD_LIBRARY_PATH).
    for dep_path in _ldd_paths(exe):
        if "libggml-sycl.so" in dep_path.name:
            return _resolve(dep_path)

    return None


def _ldd_paths(target: Path) -> list[Path]:
    """Resolved shared-library paths reported by ``ldd``, empty on failure."""
    out: list[Path] = []
    try:
        proc = subprocess.run(
            ["ldd", str(target)],
            capture_output=True,
            text=True,
            timeout=_LDD_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return out
    if proc.returncode != 0:
        return out
    for line in proc.stdout.splitlines():
        if "=>" not in line:
            continue
        rhs = line.split("=>", 1)[1].split("(")[0].strip()
        if rhs:
            out.append(Path(rhs))
    return out


def _ldd_has_onednn(target: Path) -> bool | None:
    """Authoritative oneDNN check via ``ldd``.

    Returns ``True``/``False`` when ``ldd`` ran successfully -- a clean run that
    lists no dnnl dependency is a *reliable* negative -- and ``None`` when
    ``ldd`` could not be used at all.
    """
    try:
        proc = subprocess.run(
            ["ldd", str(target)],
            capture_output=True,
            text=True,
            timeout=_LDD_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    blob = (proc.stdout + proc.stderr).lower()
    return any(dep in blob for dep in _ONEDNN_DEPS)


def _scan(path: Path, needle_groups: Iterable[tuple[bytes, ...]]) -> list[bool]:
    """Report, per group, whether *any* needle in that group occurs in *path*.

    The whole file is searched. There is deliberately no size cap: the SYCL
    backend library runs to hundreds of megabytes and a truncated scan produces
    silent false negatives, which is the exact failure this module exists to
    prevent.
    """
    groups = list(needle_groups)
    results = [False] * len(groups)
    try:
        with path.open("rb") as fh:
            try:
                mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            except (OSError, ValueError):
                data = fh.read()
                for i, group in enumerate(groups):
                    results[i] = any(n in data for n in group)
                return results
            try:
                for i, group in enumerate(groups):
                    results[i] = any(mm.find(n) != -1 for n in group)
            finally:
                mm.close()
    except OSError as exc:
        log.debug("could not scan %s: %s", path, exc)
    return results


_cache: dict[tuple[str, float], SyclCaps] = {}


def probe_sycl_caps(llama_server: str) -> SyclCaps:
    """Inspect *llama_server*'s SYCL backend. Cached per (lib path, mtime).

    Never launches the server and never touches the GPU.
    """
    lib = find_sycl_lib(llama_server)
    if lib is None or not lib.exists():
        log.debug("no libggml-sycl.so found for %s", llama_server)
        return SyclCaps()

    try:
        key = (str(lib), lib.stat().st_mtime)
    except OSError:
        return SyclCaps(lib_path=lib)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    has_symbols, onednn_str, q8_fast, aot = _scan(
        lib, (_SYMBOL_MARKER, _ONEDNN_NEEDLES, _Q8_REORDER_NEEDLES, _AOT_NEEDLES)
    )

    # ldd is authoritative in both directions; only fall back to the string
    # scan (which can false-positive on debug text) when ldd is unusable.
    onednn = _ldd_has_onednn(lib)
    if onednn is None:
        onednn = True if onednn_str else (False if has_symbols else None)

    caps = SyclCaps(
        has_onednn_sdpa=onednn,
        # A negative symbol result only means something if symbols exist.
        has_fast_q8_weights=True if q8_fast else (False if has_symbols else None),
        is_aot=True if aot else (False if has_symbols else None),
        has_symbols=has_symbols,
        lib_path=lib,
        probed=True,
    )
    _cache[key] = caps
    log.debug("probed %s -> %s", lib, caps)
    return caps


def clear_cache() -> None:
    """Drop memoized probe results (tests, or after replacing a binary)."""
    _cache.clear()
