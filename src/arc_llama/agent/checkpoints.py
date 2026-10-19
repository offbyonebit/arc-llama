"""Checkpoint / rollback support for the agent.

A checkpoint is a snapshot of the project root taken before the first
destructive operation in a run. It lets the user revert the agent's file
changes if something goes wrong.

Checkpoints are retained with bounded disk usage: per-run count and age
limits evict old snapshots automatically, and an optional global size cap
can trim the oldest checkpoints across all runs.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("arc_llama.agent.checkpoints")


@dataclass
class Checkpoint:
    """Metadata for a single checkpoint."""

    id: str
    run_id: str
    created_at: float
    files: list[str]


class CheckpointStore:
    """Stores per-run file checkpoints on disk with retention limits."""

    IGNORE_PATTERNS = (".git", "node_modules", "__pycache__", ".venv", "venv", "build", "dist")

    def __init__(
        self,
        base_dir: Path | None = None,
        *,
        max_per_run: int = 10,
        max_age_days: float = 7.0,
        max_total_size_mb: float | None = None,
    ) -> None:
        self.base_dir = Path(base_dir or ".arc_llama_checkpoints")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_per_run = max(1, int(max_per_run))
        self.max_age_days = max(0.0, float(max_age_days))
        self.max_total_size_mb = max_total_size_mb

    @property
    def max_age_seconds(self) -> float:
        return self.max_age_days * 24 * 60 * 60

    @property
    def max_total_size_bytes(self) -> int | None:
        if self.max_total_size_mb is None:
            return None
        return max(0, int(self.max_total_size_mb * 1024 * 1024))

    def _run_dir(self, run_id: str) -> Path:
        return self.base_dir / run_id

    @staticmethod
    def _checkpoint_size(cp_dir: Path) -> int:
        """Return the total byte size of a checkpoint directory."""
        total = 0
        if not cp_dir.exists():
            return 0
        try:
            for path in cp_dir.rglob("*"):
                if path.is_file():
                    try:
                        total += path.stat().st_size
                    except OSError:
                        pass
        except OSError:
            pass
        return total

    def _delete_checkpoint(self, run_id: str, checkpoint_id: str) -> None:
        """Remove a checkpoint directory and log the eviction."""
        cp_dir = self._run_dir(run_id) / checkpoint_id
        if not cp_dir.exists():
            return
        try:
            shutil.rmtree(cp_dir)
            log.info("Evicted checkpoint %s for run %s", checkpoint_id, run_id)
        except OSError as e:
            log.warning("Failed to evict checkpoint %s/%s: %s", run_id, checkpoint_id, e)

        # Clean up empty run directories so list() stays tidy.
        run_dir = self._run_dir(run_id)
        try:
            if run_dir.exists() and not any(run_dir.iterdir()):
                run_dir.rmdir()
        except OSError:
            pass

    def _all_checkpoints(self) -> list[tuple[str, str, float, Path]]:
        """Return (run_id, checkpoint_id, created_at, cp_dir) for all checkpoints."""
        result: list[tuple[str, str, float, Path]] = []
        if not self.base_dir.exists():
            return result
        for run_dir in self.base_dir.iterdir():
            if not run_dir.is_dir():
                continue
            for cp_dir in run_dir.iterdir():
                if not cp_dir.is_dir():
                    continue
                meta_path = cp_dir / "meta.json"
                created_at: float = 0.0
                if meta_path.is_file():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        created_at = float(meta.get("created_at", 0))
                    except (json.JSONDecodeError, OSError):
                        pass
                result.append((run_dir.name, cp_dir.name, created_at, cp_dir))
        return result

    def _evict_run(self, run_id: str, now: float) -> None:
        """Apply per-run count and age limits for a single run."""
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            return

        checkpoints = self._all_checkpoints_for_run(run_id)
        if not checkpoints:
            return

        # Sort newest first so eviction always removes the oldest entries.
        checkpoints.sort(key=lambda x: x[2], reverse=True)

        kept: list[tuple[str, float]] = []
        for cp_id, created_at, _ in checkpoints:
            if self.max_age_seconds and (now - created_at) > self.max_age_seconds:
                self._delete_checkpoint(run_id, cp_id)
                continue
            kept.append((cp_id, created_at))

        # Enforce per-run count, removing the oldest first.
        while len(kept) > self.max_per_run:
            oldest_id, _ = kept.pop()
            self._delete_checkpoint(run_id, oldest_id)

    def _all_checkpoints_for_run(self, run_id: str) -> list[tuple[str, float, Path]]:
        """Return (checkpoint_id, created_at, cp_dir) for a run."""
        run_dir = self._run_dir(run_id)
        result: list[tuple[str, float, Path]] = []
        if not run_dir.exists():
            return result
        for cp_dir in run_dir.iterdir():
            if not cp_dir.is_dir():
                continue
            meta_path = cp_dir / "meta.json"
            created_at = 0.0
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    created_at = float(meta.get("created_at", 0))
                except (json.JSONDecodeError, OSError):
                    pass
            result.append((cp_dir.name, created_at, cp_dir))
        return result

    def _evict_global(self, now: float) -> None:
        """Apply the global size cap by removing oldest checkpoints first."""
        if self.max_total_size_bytes is None:
            return

        checkpoints = self._all_checkpoints()
        total = sum(self._checkpoint_size(cp_dir) for _, _, _, cp_dir in checkpoints)
        if total <= self.max_total_size_bytes:
            return

        # Oldest first across all runs.
        checkpoints.sort(key=lambda x: x[2])
        for run_id, cp_id, _, cp_dir in checkpoints:
            if total <= self.max_total_size_bytes:
                break
            size = self._checkpoint_size(cp_dir)
            self._delete_checkpoint(run_id, cp_id)
            total -= size

    def create(self, run_id: str, root: Path) -> Checkpoint:
        """Snapshot the whole project root before any changes are made."""
        root = root.resolve()
        ts = time.time()
        checkpoint_id = str(int(ts * 1000))
        cp_dir = self._run_dir(run_id) / checkpoint_id
        # Sub-millisecond checkpoints can collide; bump the id until free.
        while cp_dir.exists():
            checkpoint_id = str(int(checkpoint_id) + 1)
            cp_dir = self._run_dir(run_id) / checkpoint_id
        files_dir = cp_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        if root.exists():
            shutil.copytree(
                root,
                files_dir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(*self.IGNORE_PATTERNS),
            )

        # Record which top-level entries were captured.
        captured: list[str] = []
        if files_dir.exists():
            captured = sorted(p.name for p in files_dir.iterdir())

        meta = {"created_at": ts, "files": captured}
        (cp_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        log.info("Created checkpoint %s for run %s", checkpoint_id, run_id)

        # Evict old checkpoints so disk usage stays bounded.
        self._evict_run(run_id, ts)
        self._evict_global(ts)

        return Checkpoint(
            id=checkpoint_id,
            run_id=run_id,
            created_at=ts,
            files=captured,
        )

    def list(self, run_id: str) -> list[Checkpoint]:
        """Return all checkpoints for a run, oldest first."""
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            return []

        checkpoints: list[Checkpoint] = []
        for cp_dir in sorted(run_dir.iterdir(), key=lambda p: int(p.name)):
            meta_path = cp_dir / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            checkpoints.append(
                Checkpoint(
                    id=cp_dir.name,
                    run_id=run_id,
                    created_at=float(meta.get("created_at", 0)),
                    files=list(meta.get("files", [])),
                )
            )
        return checkpoints

    def restore(self, run_id: str, checkpoint_id: str, root: Path) -> None:
        """Restore the project root from a checkpoint."""
        cp_files_dir = self._run_dir(run_id) / checkpoint_id / "files"
        if not cp_files_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)

        # Remove existing top-level entries that were captured so the restore
        # is faithful, then copy the checkpoint contents back.
        meta_path = self._run_dir(run_id) / checkpoint_id / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}

        for name in meta.get("files", []):
            target = root / name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()

        for src in cp_files_dir.iterdir():
            dst = root / src.name
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

        log.info("Restored checkpoint %s for run %s", checkpoint_id, run_id)

    def cleanup(self) -> None:
        """Explicitly apply retention limits to all stored checkpoints."""
        now = time.time()
        for run_id, _, _, _ in self._all_checkpoints():
            self._evict_run(run_id, now)
        self._evict_global(now)
