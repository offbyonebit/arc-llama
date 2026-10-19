"""Change tracking and undo for agent runs."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from arc_llama.agent import changes as ch
from arc_llama.agent.changes import ChangeTracker


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "a.txt").write_text("one\ntwo\nthree\n")
    (root / "src" / "b.py").write_text("print('b')\n")
    (root / "gone.txt").write_text("bye\n")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "skip.js").write_text("x")
    return root


def _tracker(project: Path, tmp_path: Path) -> ChangeTracker:
    t = ChangeTracker(project, tmp_path / "store")
    t.begin()
    return t


def _bump(path: Path) -> None:
    """Make mtime differ even on coarse filesystems."""
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))


def test_reports_added_modified_deleted_with_stats(project, tmp_path):
    t = _tracker(project, tmp_path)
    (project / "new.txt").write_text("hello\nworld\n")
    (project / "a.txt").write_text("one\nTWO\nthree\nfour\n")
    _bump(project / "a.txt")
    (project / "gone.txt").unlink()
    (project / "node_modules" / "skip.js").write_text("changed")      # ignored directory
    out = t.finish()
    by = {c["path"]: c for c in out["files"]}
    assert set(by) == {"new.txt", "a.txt", "gone.txt"}
    assert by["new.txt"]["status"] == "added" and by["new.txt"]["additions"] == 2
    assert by["a.txt"]["status"] == "modified" and (by["a.txt"]["additions"], by["a.txt"]["deletions"]) == (2, 1)
    assert by["gone.txt"]["status"] == "deleted" and by["gone.txt"]["preview"] == ["bye"]
    assert by["new.txt"]["preview"] == ["hello", "world"] and by["new.txt"]["preview_kind"] == "content"
    # a modified file previews what changed, not the top of the file
    assert by["a.txt"]["preview_kind"] == "diff"
    assert "-two" in by["a.txt"]["preview"] and "+TWO" in by["a.txt"]["preview"] and "+four" in by["a.txt"]["preview"]
    assert not any(line.startswith(("---", "+++", "@@")) for line in by["a.txt"]["preview"])
    assert out["undoable"] is True and out["truncated"] is False


def test_no_changes_reports_nothing(project, tmp_path):
    t = _tracker(project, tmp_path)
    out = t.finish()
    assert out == {"files": [], "undoable": False, "truncated": False}


def test_undo_restores_everything(project, tmp_path):
    t = _tracker(project, tmp_path)
    (project / "new.txt").write_text("hello\n")
    (project / "a.txt").write_text("rewritten\n")
    _bump(project / "a.txt")
    (project / "gone.txt").unlink()
    t.finish()
    result = t.undo()
    assert {r["path"] for r in result["restored"]} == {"new.txt", "a.txt", "gone.txt"} and result["skipped"] == []
    assert not (project / "new.txt").exists()
    assert (project / "a.txt").read_text() == "one\ntwo\nthree\n"
    assert (project / "gone.txt").read_text() == "bye\n"


def test_undo_leaves_files_edited_after_the_run(project, tmp_path):
    t = _tracker(project, tmp_path)
    (project / "a.txt").write_text("agent edit\n")
    _bump(project / "a.txt")
    (project / "new.txt").write_text("agent file\n")
    t.finish()
    (project / "a.txt").write_text("my own edit afterwards\n")       # the user kept working
    result = t.undo()
    assert [s["path"] for s in result["skipped"]] == ["a.txt"] and result["skipped"][0]["reason"] == "changed since the run"
    assert (project / "a.txt").read_text() == "my own edit afterwards\n"
    assert not (project / "new.txt").exists()


def test_large_files_make_the_run_not_undoable_but_still_reported(project, tmp_path, monkeypatch):
    monkeypatch.setattr(ch, "MAX_FILE_BYTES", 8)
    t = _tracker(project, tmp_path)
    (project / "a.txt").write_text("changed content here\n")
    _bump(project / "a.txt")
    out = t.finish()
    assert [c["path"] for c in out["files"]] == ["a.txt"] and out["undoable"] is False
    assert t.undo()["restored"] == []
    assert (project / "a.txt").read_text() == "changed content here\n"      # nothing was touched


def test_too_many_files_is_flagged_truncated(project, tmp_path, monkeypatch):
    monkeypatch.setattr(ch, "MAX_FILES", 2)
    t = _tracker(project, tmp_path)
    (project / "new.txt").write_text("x\n")
    out = t.finish()
    assert out["truncated"] is True and out["undoable"] is False


def test_diff_text_and_binary_handling(project, tmp_path):
    t = _tracker(project, tmp_path)
    (project / "a.txt").write_text("one\nTWO\nthree\n")
    _bump(project / "a.txt")
    (project / "blob.bin").write_bytes(b"\x00\x01\x02")
    out = t.finish()
    diff = t.diff_text("a.txt")
    assert diff and "-two" in diff and "+TWO" in diff
    blob = next(c for c in out["files"] if c["path"] == "blob.bin")
    assert blob["binary"] is True and blob["preview"] == [] and t.diff_text("blob.bin") is None
    assert t.diff_text("missing.txt") is None


def test_symlinks_are_not_followed(project, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n")
    (project / "link.txt").symlink_to(outside)
    t = _tracker(project, tmp_path)
    outside.write_text("changed\n")
    assert t.finish()["files"] == []


def test_discard_removes_backups(project, tmp_path):
    t = _tracker(project, tmp_path)
    assert (tmp_path / "store").exists()
    t.discard()
    assert not (tmp_path / "store").exists()
