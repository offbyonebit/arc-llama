"""Tests for arc_llama.sycl_cache — fingerprinting, crash guard, pruning."""
from __future__ import annotations

import time
from pathlib import Path

from arc_llama.sycl_cache import (
    KEEP_RECENT_CACHES,
    MARKER_FILE,
    POISON_FILE,
    binary_fingerprint,
    prepare_jit_cache,
)


def _fake_binary(tmp_path: Path, name: str = "llama-server", content: bytes = b"ELF") -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


class TestFingerprint:
    def test_missing_binary_returns_none(self, tmp_path: Path):
        assert binary_fingerprint(str(tmp_path / "nope")) is None

    def test_stable_for_same_file(self, tmp_path: Path):
        b = _fake_binary(tmp_path)
        assert binary_fingerprint(str(b)) == binary_fingerprint(str(b))

    def test_changes_when_binary_changes(self, tmp_path: Path):
        b = _fake_binary(tmp_path)
        fp1 = binary_fingerprint(str(b))
        # New size + mtime — simulates a llama.cpp upgrade in place.
        b.write_bytes(b"ELF v2 rebuilt")
        fp2 = binary_fingerprint(str(b))
        assert fp1 != fp2


class TestPrepare:
    def test_enables_and_creates_dir(self, tmp_path: Path):
        b = _fake_binary(tmp_path)
        plan = prepare_jit_cache(tmp_path / "state", str(b))
        assert plan.enabled is True
        assert plan.env["SYCL_CACHE_PERSISTENT"] == "1"
        assert plan.cache_dir is not None and plan.cache_dir.is_dir()
        assert plan.env["SYCL_CACHE_DIR"] == str(plan.cache_dir)
        assert plan.marker == plan.cache_dir / MARKER_FILE

    def test_missing_binary_disables(self, tmp_path: Path):
        plan = prepare_jit_cache(tmp_path / "state", str(tmp_path / "nope"))
        assert plan.enabled is False
        assert plan.env == {}

    def test_leftover_marker_wipes_and_poisons(self, tmp_path: Path):
        b = _fake_binary(tmp_path)
        first = prepare_jit_cache(tmp_path / "state", str(b))
        assert first.enabled
        # Simulate a crash during warm-up: marker written, never cleared,
        # plus a cache entry the runtime wrote before dying.
        first.marker.write_text("12345 test\n")
        (first.cache_dir / "some-kernel.bin").write_bytes(b"jit blob")

        second = prepare_jit_cache(tmp_path / "state", str(b))
        assert second.enabled is False
        assert (first.cache_dir / POISON_FILE).exists()
        assert not (first.cache_dir / "some-kernel.bin").exists()
        assert not first.marker.exists()

    def test_poison_is_sticky(self, tmp_path: Path):
        b = _fake_binary(tmp_path)
        first = prepare_jit_cache(tmp_path / "state", str(b))
        first.marker.write_text("crash\n")
        prepare_jit_cache(tmp_path / "state", str(b))   # poisons
        third = prepare_jit_cache(tmp_path / "state", str(b))
        assert third.enabled is False
        assert "poisoned" in third.reason

    def test_clean_lifecycle_not_poisoned(self, tmp_path: Path):
        """Marker written then cleared (healthy run) → next prepare re-enables."""
        b = _fake_binary(tmp_path)
        first = prepare_jit_cache(tmp_path / "state", str(b))
        first.marker.write_text("pid\n")
        first.marker.unlink()
        second = prepare_jit_cache(tmp_path / "state", str(b))
        assert second.enabled is True
        assert second.cache_dir == first.cache_dir

    def test_prunes_old_fingerprints(self, tmp_path: Path):
        state = tmp_path / "state"
        root = state / "sycl-cache"
        root.mkdir(parents=True)
        # Simulate caches left behind by old llama-server builds.
        stale = []
        for i in range(KEEP_RECENT_CACHES + 3):
            d = root / f"oldfp{i:02d}"
            d.mkdir()
            t = time.time() - 1000 + i
            import os
            os.utime(d, (t, t))
            stale.append(d)
        b = _fake_binary(tmp_path)
        plan = prepare_jit_cache(state, str(b))
        assert plan.enabled
        survivors = [d for d in stale if d.exists()]
        assert len(survivors) == KEEP_RECENT_CACHES
        # The newest siblings survive.
        assert survivors == stale[-KEEP_RECENT_CACHES:]
