"""Tests for SYCL binary capability detection.

All tests use synthetic fake libraries. Nothing here launches llama-server,
touches a GPU, or depends on the host having an Intel card.
"""

from __future__ import annotations

import subprocess

import pytest

from arc_llama import binary_caps
from arc_llama.binary_caps import SyclCaps, find_sycl_lib, probe_sycl_caps


@pytest.fixture(autouse=True)
def _clear_cache():
    binary_caps.clear_cache()
    yield
    binary_caps.clear_cache()


def _make_tree(tmp_path, lib_bytes: bytes, *, versioned: bool = False):
    """Build a fake bin/ dir with llama-server plus a libggml-sycl.so."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "llama-server"
    exe.write_bytes(b"\x7fELF fake server")
    exe.chmod(0o755)

    if versioned:
        # Mirror a real build tree: symlink chain plus stale older copies that
        # must NOT be picked up.
        real = bindir / "libggml-sycl.so.0.15.1"
        real.write_bytes(lib_bytes)
        (bindir / "libggml-sycl.so.0.11.0").write_bytes(b"stale old build")
        (bindir / "libggml-sycl.so.0").symlink_to(real.name)
        (bindir / "libggml-sycl.so").symlink_to("libggml-sycl.so.0")
    else:
        (bindir / "libggml-sycl.so").write_bytes(lib_bytes)
    return exe


# Symbol blobs modelled on what real builds actually contain.
_SYMBOLS = b"ggml_sycl_init ggml_sycl_mul_mat "
_Q8_FIXED = b"_Z34ggml_sycl_mul_mat_vec_q_id_reorder9ggml_type "
_AOT = b"spir64_gen -device bmg "
_ONEDNN = b"libdnnl.so.3 dnnl_sycl_interop "


def _no_ldd(monkeypatch):
    """Force the ldd path to be unavailable, exercising the string fallback."""
    def _boom(*a, **k):
        raise OSError("no ldd")

    monkeypatch.setattr(subprocess, "run", _boom)


def _ldd_reporting(monkeypatch, stdout: str, returncode: int = 0):
    def _fake(*a, **k):
        return subprocess.CompletedProcess(a[0], returncode, stdout, "")

    monkeypatch.setattr(subprocess, "run", _fake)


def test_no_library_found_is_all_unknown(tmp_path):
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"\x7fELF")
    caps = probe_sycl_caps(str(exe))
    assert caps == SyclCaps()
    assert caps.probed is False
    assert caps.has_onednn_sdpa is None


def test_resolves_symlink_chain_and_ignores_stale_versions(tmp_path):
    exe = _make_tree(tmp_path, _SYMBOLS + _Q8_FIXED + _AOT, versioned=True)
    lib = find_sycl_lib(str(exe))
    assert lib is not None
    # Must land on the resolved target, not the symlink or an older copy.
    assert lib.name == "libggml-sycl.so.0.15.1"


def test_detects_fixed_q8_and_aot_without_onednn(tmp_path, monkeypatch):
    """The common real-world case: AOT build, Q8 fix present, no oneDNN."""
    exe = _make_tree(tmp_path, _SYMBOLS + _Q8_FIXED + _AOT)
    _ldd_reporting(monkeypatch, "\tlibsycl.so.8 => /opt/lib/libsycl.so.8 (0x00)\n")

    caps = probe_sycl_caps(str(exe))
    assert caps.probed is True
    assert caps.has_symbols is True
    assert caps.has_fast_q8_weights is True
    assert caps.is_aot is True
    # Clean ldd listing no dnnl is a reliable negative, not "unknown".
    assert caps.has_onednn_sdpa is False


def test_ldd_is_authoritative_for_onednn_present(tmp_path, monkeypatch):
    exe = _make_tree(tmp_path, _SYMBOLS)
    _ldd_reporting(monkeypatch, "\tlibdnnl.so.3 => /opt/lib/libdnnl.so.3 (0x00)\n")
    assert probe_sycl_caps(str(exe)).has_onednn_sdpa is True


def test_string_fallback_detects_onednn_when_ldd_unavailable(tmp_path, monkeypatch):
    exe = _make_tree(tmp_path, _SYMBOLS + _ONEDNN)
    _no_ldd(monkeypatch)
    assert probe_sycl_caps(str(exe)).has_onednn_sdpa is True


def test_stripped_library_yields_unknown_not_false(tmp_path, monkeypatch):
    """A stripped binary must never produce confident negatives.

    This is the case that matters for the portable runtimes `install-runtime`
    downloads, which are frequently stripped.
    """
    exe = _make_tree(tmp_path, b"\x00\x01\x02 no symbols here at all")
    _no_ldd(monkeypatch)

    caps = probe_sycl_caps(str(exe))
    assert caps.has_symbols is False
    assert caps.has_fast_q8_weights is None
    assert caps.is_aot is None
    assert caps.has_onednn_sdpa is None


def test_stripped_library_still_trusts_ldd(tmp_path, monkeypatch):
    """Symbols may be gone, but linkage is still knowable."""
    exe = _make_tree(tmp_path, b"\x00\x01\x02 stripped")
    _ldd_reporting(monkeypatch, "\tlibsycl.so.8 => /opt/lib/libsycl.so.8 (0x00)\n")

    caps = probe_sycl_caps(str(exe))
    assert caps.has_symbols is False
    assert caps.has_onednn_sdpa is False  # ldd still authoritative
    assert caps.has_fast_q8_weights is None  # but symbol scans are not


def test_old_binary_without_q8_fix_reports_false(tmp_path, monkeypatch):
    exe = _make_tree(tmp_path, _SYMBOLS + _AOT)  # symbols present, no reorder
    _ldd_reporting(monkeypatch, "\tlibsycl.so.8 => /opt/lib/libsycl.so.8 (0x00)\n")
    assert probe_sycl_caps(str(exe)).has_fast_q8_weights is False


def test_jit_build_reports_not_aot(tmp_path, monkeypatch):
    exe = _make_tree(tmp_path, _SYMBOLS + _Q8_FIXED + b"spir64 generic target")
    _ldd_reporting(monkeypatch, "")
    assert probe_sycl_caps(str(exe)).is_aot is False


def test_bare_spir64_does_not_false_positive_as_aot(tmp_path, monkeypatch):
    """`spir64` is a prefix of `spir64_gen`; only the latter means AOT."""
    exe = _make_tree(tmp_path, _SYMBOLS + b"-fsycl-targets=spir64 ")
    _ldd_reporting(monkeypatch, "")
    assert probe_sycl_caps(str(exe)).is_aot is False


def test_scan_is_not_truncated_on_large_library(tmp_path, monkeypatch):
    """Needles past any plausible read cap must still be found.

    Real SYCL backends are ~180 MB and the marker can sit deep in the file; a
    capped scan silently reports a false negative.
    """
    padding = b"\x00" * (12 * 1024 * 1024)
    exe = _make_tree(tmp_path, _SYMBOLS + padding + _Q8_FIXED)
    _ldd_reporting(monkeypatch, "")
    assert probe_sycl_caps(str(exe)).has_fast_q8_weights is True


def test_results_are_cached_per_mtime(tmp_path, monkeypatch):
    exe = _make_tree(tmp_path, _SYMBOLS + _Q8_FIXED)
    _ldd_reporting(monkeypatch, "")

    first = probe_sycl_caps(str(exe))
    calls = {"n": 0}
    real_scan = binary_caps._scan

    def _counting(*a, **k):
        calls["n"] += 1
        return real_scan(*a, **k)

    monkeypatch.setattr(binary_caps, "_scan", _counting)
    second = probe_sycl_caps(str(exe))

    assert second == first
    assert calls["n"] == 0, "second probe should hit the cache"
