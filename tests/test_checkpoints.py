"""Tests for the agent checkpoint store."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from arc_llama.agent.checkpoints import CheckpointStore


@pytest.fixture
def store(tmp_path: Path) -> CheckpointStore:
    return CheckpointStore(base_dir=tmp_path / "checkpoints")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (root / "README.md").write_text("# Project\n", encoding="utf-8")
    return root


def test_create_checkpoint(store: CheckpointStore, project: Path) -> None:
    cp = store.create("run-1", project)
    assert cp.run_id == "run-1"
    assert cp.id
    assert "README.md" in cp.files or "src" in cp.files

    cp_dir = store.base_dir / "run-1" / cp.id
    assert (cp_dir / "meta.json").is_file()
    assert (cp_dir / "files" / "README.md").read_text(encoding="utf-8") == "# Project\n"
    assert (cp_dir / "files" / "src" / "main.py").read_text(encoding="utf-8") == "print('hello')\n"


def test_list_checkpoints_sorted(store: CheckpointStore, project: Path) -> None:
    cp1 = store.create("run-1", project)
    cp2 = store.create("run-1", project)
    cps = store.list("run-1")
    assert [c.id for c in cps] == [cp1.id, cp2.id]


def test_restore_checkpoint(store: CheckpointStore, project: Path) -> None:
    cp = store.create("run-1", project)
    (project / "src" / "main.py").write_text("print('changed')\n", encoding="utf-8")
    store.restore("run-1", cp.id, project)
    assert (project / "src" / "main.py").read_text(encoding="utf-8") == "print('hello')\n"


def test_restore_missing_checkpoint(store: CheckpointStore, project: Path) -> None:
    with pytest.raises(FileNotFoundError):
        store.restore("run-1", "no-such-id", project)


def test_eviction_by_per_run_count(store: CheckpointStore, project: Path) -> None:
    """Only the most recent max_per_run checkpoints are kept for a run."""
    store.max_per_run = 2
    cp1 = store.create("run-1", project)
    cp2 = store.create("run-1", project)
    cp3 = store.create("run-1", project)

    cps = store.list("run-1")
    assert [c.id for c in cps] == [cp2.id, cp3.id]
    assert not (store.base_dir / "run-1" / cp1.id).exists()


def test_eviction_by_age(store: CheckpointStore, project: Path) -> None:
    """Checkpoints older than max_age_days are evicted on create."""
    store.max_age_days = 1.0

    now = 1_000_000.0
    with patch("time.time", return_value=now):
        old_cp = store.create("run-1", project)

    # Two days later a new checkpoint should evict the old one.
    later = now + 2 * 24 * 60 * 60
    with patch("time.time", return_value=later):
        new_cp = store.create("run-1", project)

    cps = store.list("run-1")
    assert [c.id for c in cps] == [new_cp.id]
    assert not (store.base_dir / "run-1" / old_cp.id).exists()


def test_eviction_keeps_newest_when_over_count(store: CheckpointStore, project: Path) -> None:
    """When count exceeds max_per_run, the oldest checkpoints are removed."""
    store.max_per_run = 1
    cp1 = store.create("run-1", project)
    cp2 = store.create("run-1", project)

    assert store.list("run-1")[0].id == cp2.id
    assert not (store.base_dir / "run-1" / cp1.id).exists()


def test_eviction_by_global_size_cap(store: CheckpointStore, project: Path) -> None:
    """A global size cap removes oldest checkpoints across runs."""
    cp1 = store.create("run-1", project)
    cp1_size = CheckpointStore._checkpoint_size(store.base_dir / "run-1" / cp1.id)

    # Set a cap that fits one checkpoint but not two.
    store.max_total_size_mb = (cp1_size + 100) / (1024 * 1024)

    cp2 = store.create("run-2", project)

    # The oldest checkpoint across all runs should be evicted.
    assert not (store.base_dir / "run-1" / cp1.id).exists()
    assert (store.base_dir / "run-2" / cp2.id).exists()


def test_cleanup_applies_retention_limits(store: CheckpointStore, project: Path) -> None:
    """cleanup() runs retention rules without creating a checkpoint."""
    store.create("run-1", project)
    cp2 = store.create("run-1", project)
    assert len(store.list("run-1")) == 2

    # Tighten the limit and run cleanup explicitly.
    store.max_per_run = 1
    store.cleanup()

    assert len(store.list("run-1")) == 1
    assert store.list("run-1")[0].id == cp2.id


def test_checkpoint_size_includes_files_and_meta(store: CheckpointStore, project: Path) -> None:
    cp = store.create("run-1", project)
    cp_dir = store.base_dir / "run-1" / cp.id
    size = CheckpointStore._checkpoint_size(cp_dir)
    assert size > 0


def test_empty_run_dir_removed_after_eviction(store: CheckpointStore, project: Path) -> None:
    store.max_per_run = 1
    store.create("run-1", project)
    store.create("run-1", project)
    # After evicting down to one checkpoint, the run dir still has a checkpoint.
    assert (store.base_dir / "run-1").exists()
