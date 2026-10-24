"""Track and undo the file changes made by one agent run.

Harness-agnostic: it looks at the project folder before and after, so it
works for the built-in agent and for external harnesses alike. Before a run
the files that are small enough are copied aside; afterwards the difference
is the "files changed" list, and undo restores from those copies.

Undo never overwrites work done after the run: a file is only restored or
removed if it still has exactly the content the run left behind.
"""
from __future__ import annotations

import difflib
import hashlib
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", ".idea", ".next", "target", "dist", "build",
})
MAX_FILES = 5000
MAX_FILE_BYTES = 1 << 20          # files larger than this are tracked but cannot be restored
MAX_BACKUP_BYTES = 64 << 20       # total budget for the pre-run copies
PREVIEW_BYTES = 64 << 10
PREVIEW_LINES = 12


def _sha(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 16), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _is_text(sample: bytes) -> bool:
    return b"\0" not in sample


@dataclass
class _Entry:
    size: int
    mtime_ns: int
    backed_up: bool = False


@dataclass
class ChangeTracker:
    root: Path
    store: Path
    before: dict[str, _Entry] = field(default_factory=dict)
    complete: bool = True          # False when the tree was too large to track fully
    undoable: bool = True          # False when some pre-run file could not be copied
    after_hash: dict[str, str | None] = field(default_factory=dict)
    changes: list[dict[str, Any]] = field(default_factory=list)
    created: float = field(default_factory=time.time)

    def _walk(self) -> tuple[dict[str, _Entry], bool]:
        found: dict[str, _Entry] = {}
        root = self.root
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not (Path(dirpath) / d).is_symlink())
            for name in sorted(filenames):
                path = Path(dirpath) / name
                try:
                    st = path.lstat()
                except OSError:
                    continue
                if not path.is_file() or path.is_symlink():
                    continue
                if len(found) >= MAX_FILES:
                    return found, False
                found[path.relative_to(root).as_posix()] = _Entry(st.st_size, st.st_mtime_ns)
        return found, True

    def begin(self) -> None:
        """Record the tree and copy small files aside. Blocking; call from a thread."""
        self.before, self.complete = self._walk()
        spent = 0
        self.store.mkdir(parents=True, exist_ok=True)
        for rel, entry in self.before.items():
            if entry.size > MAX_FILE_BYTES or spent + entry.size > MAX_BACKUP_BYTES:
                self.undoable = False
                continue
            dest = self.store / rel
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.root / rel, dest)
                entry.backed_up = True
                spent += entry.size
            except OSError:
                self.undoable = False
        if not self.complete:
            self.undoable = False

    def finish(self) -> dict[str, Any]:
        """Compare with the pre-run tree. Blocking; call from a thread."""
        after, complete = self._walk()
        self.complete = self.complete and complete
        changes: list[dict[str, Any]] = []
        for rel, now in after.items():
            old = self.before.get(rel)
            if old is None:
                changes.append(self._describe(rel, "added", None))
            elif (old.size, old.mtime_ns) != (now.size, now.mtime_ns):
                changes.append(self._describe(rel, "modified", old))
        for rel, old in self.before.items():
            if rel not in after:
                changes.append(self._describe(rel, "deleted", old))
        changes.sort(key=lambda c: c["path"])
        self.changes = changes
        for change in changes:
            if change["status"] != "deleted":
                self.after_hash[change["path"]] = _sha(self.root / change["path"])
            else:
                self.after_hash[change["path"]] = None
        # A change can only be undone if its original is on hand (added files need none).
        undoable = self.complete and all(
            c["status"] == "added" or self.before[c["path"]].backed_up for c in changes
        )
        self.undoable = bool(changes) and undoable
        return {"files": changes, "undoable": self.undoable, "truncated": not self.complete}

    def _describe(self, rel: str, status: str, old: _Entry | None) -> dict[str, Any]:
        new_lines = self._lines(self.root / rel) if status != "deleted" else []
        old_lines = self._lines(self.store / rel) if old is not None and old.backed_up else []
        if status == "added":
            additions, deletions = len(new_lines or []), 0
        elif status == "deleted":
            additions, deletions = 0, len(old_lines or [])
        else:
            additions = deletions = 0
            if new_lines is not None and old_lines is not None:
                for tag, a1, a2, b1, b2 in difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes():
                    if tag in ("replace", "delete"):
                        deletions += a2 - a1
                    if tag in ("replace", "insert"):
                        additions += b2 - b1
        preview_kind = "content"
        if status == "modified" and new_lines is not None and old_lines is not None:
            # Show what actually changed, not the top of the file.
            hunks = list(difflib.unified_diff(old_lines, new_lines, lineterm="", n=1))[2:]
            preview, preview_kind = [h for h in hunks if not h.startswith("@@")][:PREVIEW_LINES * 2], "diff"
        elif status != "deleted":
            preview = (new_lines or [])[:PREVIEW_LINES]
        else:
            preview = (old_lines or [])[:PREVIEW_LINES]
        return {
            "path": rel, "status": status, "additions": additions, "deletions": deletions,
            "preview": preview, "preview_kind": preview_kind,
            "binary": new_lines is None and status != "deleted",
        }

    @staticmethod
    def _lines(path: Path) -> list[str] | None:
        """Text lines of a small file, or None when it is binary/unreadable/too big."""
        try:
            with path.open("rb") as fh:
                data = fh.read(PREVIEW_BYTES + 1)
        except OSError:
            return None
        if len(data) > PREVIEW_BYTES or not _is_text(data):
            return None
        return data.decode("utf-8", errors="replace").splitlines()

    def diff_text(self, rel: str) -> str | None:
        """Unified diff for one changed file, for the Review view."""
        change = next((c for c in self.changes if c["path"] == rel), None)
        if change is None:
            return None
        old = self._lines(self.store / rel) if rel in self.before and self.before[rel].backed_up else []
        new = self._lines(self.root / rel) if change["status"] != "deleted" else []
        if old is None or new is None:
            return None
        return "\n".join(difflib.unified_diff(old, new, f"a/{rel}", f"b/{rel}", lineterm=""))

    def undo(self) -> dict[str, list[dict[str, str]]]:
        """Restore the pre-run state, leaving anything touched since the run alone."""
        restored: list[dict[str, str]] = []
        skipped: list[dict[str, str]] = []
        if not self.undoable:
            return {"restored": [], "skipped": [{"path": c["path"], "reason": "not undoable"} for c in self.changes]}
        root = self.root.resolve()
        for change in self.changes:
            rel, status = change["path"], change["status"]
            target = (self.root / rel)
            try:
                inside = target.resolve().is_relative_to(root)
            except OSError:
                inside = False
            if not inside:
                skipped.append({"path": rel, "reason": "outside project"})
                continue
            current = _sha(target) if target.exists() else None
            if current != self.after_hash.get(rel):
                skipped.append({"path": rel, "reason": "changed since the run"})
                continue
            try:
                if status == "added":
                    target.unlink()
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(self.store / rel, target)
                restored.append({"path": rel, "status": status})
            except OSError as exc:
                skipped.append({"path": rel, "reason": str(exc)})
        return {"restored": restored, "skipped": skipped}

    def discard(self) -> None:
        shutil.rmtree(self.store, ignore_errors=True)
