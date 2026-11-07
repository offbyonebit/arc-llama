"""Repo map and optional semantic code search for the agent.

The repo map needs no extra dependencies. Semantic search is opt-in via the
``semantic`` extra (fastembed + numpy) so the core package stays light.
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import json
import logging
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("arc_llama.agent.repo_map")


IGNORE_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv*",
    "venv",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".eggs",
    "*.egg-info",
}

MAX_FILE_SIZE = 512 * 1024  # 512 KiB

# Language -> regexes for symbol extraction.
SYMBOL_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    ".py": [
        re.compile(r"^(?:async\s+)?def\s+(\w+)\s*\("),
        re.compile(r"^class\s+(\w+)\s*[\(:]"),
    ],
    ".js": [
        re.compile(r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+(\w+)"),
        re.compile(r"^class\s+(\w+)"),
        re.compile(r"^const\s+(\w+)\s*=.*=>"),
    ],
    ".ts": [
        re.compile(r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+(\w+)"),
        re.compile(r"^class\s+(\w+)"),
        re.compile(r"^interface\s+(\w+)"),
        re.compile(r"^const\s+(\w+)\s*=.*=>"),
    ],
    ".go": [
        re.compile(r"^func\s+(?:\(.*\)\s+)?(\w+)"),
        re.compile(r"^type\s+(\w+)\s+"),
    ],
    ".rs": [
        re.compile(r"^(?:pub\s+)?fn\s+(\w+)"),
        re.compile(r"^(?:pub\s+)?struct\s+(\w+)"),
        re.compile(r"^(?:pub\s+)?enum\s+(\w+)"),
    ],
    ".java": [
        re.compile(
            r"^(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:final\s+)?[\w<>,\s]+\s+(\w+)\s*\("
        ),
        re.compile(r"^(?:public\s+|private\s+|protected\s+)?class\s+(\w+)"),
    ],
    ".c": [re.compile(r"^[\w\s*]+\s+(\w+)\s*\([^)]*\)\s*\{")],
    ".cpp": [
        re.compile(r"^[\w\s*:<>,]+\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?\{"),
        re.compile(r"^class\s+(\w+)"),
    ],
}
SYMBOL_PATTERNS[".jsx"] = SYMBOL_PATTERNS[".js"]
SYMBOL_PATTERNS[".tsx"] = SYMBOL_PATTERNS[".ts"]
SYMBOL_PATTERNS[".h"] = SYMBOL_PATTERNS[".c"]
SYMBOL_PATTERNS[".hpp"] = SYMBOL_PATTERNS[".cpp"]

TEXT_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".md",
    ".txt",
    ".html",
    ".css",
}


def _is_ignored(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    for part in rel.parts:
        if part in IGNORE_DIRS or any(fnmatch.fnmatch(part, pat) for pat in IGNORE_DIRS):
            return True
    return False


def _project_files(root: Path) -> list[Path]:
    """Prune ignored directories before traversal and preserve sorted output."""
    paths: list[Path] = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        dirs[:] = [
            name
            for name in dirs
            if not _is_ignored(base / name, root) and not (base / name).is_symlink()
        ]
        for name in files:
            path = base / name
            if _is_ignored(path, root):
                continue
            try:
                if stat.S_ISLNK(os.lstat(path).st_mode):
                    continue
            except OSError:
                continue
            paths.append(path)
    return sorted(paths)


def _read_project_bytes(path: Path, root: Path) -> bytes:
    """Read one bounded regular file without following project symlinks.

    On POSIX, directory-relative opens pin every parent and refuse links even
    if an entry changes after traversal. Other platforms also reject links and
    require the resolved path to remain in the supplied root.
    """
    relative = path.relative_to(root)
    if any(part in ("", ".", "..") for part in relative.parts):
        raise OSError("invalid project path")
    parents: list[int] = []
    try:
        if os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW"):
            flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_DIRECTORY", 0)
            parent = os.open(root, flags)
            parents.append(parent)
            for part in relative.parts[:-1]:
                parent = os.open(part, flags, dir_fd=parent)
                parents.append(parent)
            fd = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0), dir_fd=parent)
        else:
            if any((root / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)):
                raise OSError("project symlink refused")
            if not path.resolve(strict=True).is_relative_to(root):
                raise OSError("path outside project root")
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_SIZE:
                raise OSError("project source must be a bounded regular file")
            data = stream.read(MAX_FILE_SIZE + 1)
            after = os.fstat(stream.fileno())
            if len(data) > MAX_FILE_SIZE or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise OSError("project file changed while reading")
            return data
    finally:
        for parent in reversed(parents):
            os.close(parent)


def _is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    try:
        with path.open("rb") as f:
            chunk = f.read(1024)
        return b"\0" not in chunk
    except OSError:
        return False


def _extract_symbols(path: Path) -> list[str]:
    ext = path.suffix.lower()
    patterns = SYMBOL_PATTERNS.get(ext)
    if not patterns:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    symbols: list[str] = []
    for line in text.splitlines():
        for pat in patterns:
            m = pat.match(line)
            if m:
                symbols.append(m.group(1))
                break
    return symbols


@dataclass
class CodeChunk:
    path: str
    text: str
    start_line: int


def _chunk_file(path: Path, root: Path, text: str | None = None) -> list[CodeChunk]:
    rel = path.relative_to(root).as_posix()
    ext = path.suffix.lower()
    try:
        if text is None:
            text = _read_project_bytes(path, root).decode("utf-8", errors="ignore")
        lines = text.splitlines()
    except OSError:
        return []

    patterns = SYMBOL_PATTERNS.get(ext)
    if not patterns:
        # Generic chunked fallback.
        chunks: list[CodeChunk] = []
        for i in range(0, len(lines), 40):
            chunk_text = "\n".join(lines[i : i + 80])
            chunks.append(CodeChunk(rel, chunk_text, i + 1))
        return chunks

    # AST-naive chunking: each definition starts a new chunk.
    chunks = []
    current: list[tuple[int, str]] = []
    current_start = 1

    def flush() -> None:
        nonlocal current, current_start
        if current:
            text = "\n".join(line for _, line in current)
            chunks.append(CodeChunk(rel, text, current_start))
            current = []

    for i, line in enumerate(lines, start=1):
        is_boundary = any(pat.match(line) for pat in patterns)
        if is_boundary and current:
            flush()
            current_start = i
        current.append((i, line))
    flush()
    return chunks


def build_repo_map(root: Path, max_entries: int = 500) -> str:
    """Return a concise symbol/file tree map of the project."""
    root = root.resolve()
    if not root.exists():
        return "(project root does not exist)"

    lines: list[str] = []
    entries = 0
    for path in _project_files(root):
        if entries >= max_entries:
            lines.append("... (truncated)")
            break
        if not path.is_file():
            continue
        if not _is_text_file(path):
            continue
        try:
            if path.stat().st_size > MAX_FILE_SIZE:
                continue
        except OSError:
            continue
        symbols = _extract_symbols(path)
        rel = path.relative_to(root).as_posix()
        if symbols:
            lines.append(f"{rel}: {', '.join(symbols[:10])}")
        else:
            lines.append(f"{rel}")
        entries += 1

    return "\n".join(lines) if lines else "(empty project)"


class SemanticIndex:
    """Optional local semantic search over the project codebase.

    The whole index (embeddings, chunks and the file manifest) is one
    ``index.npz`` written to a temporary name and renamed into place, so a
    query never pairs chunks from one indexing run with vectors from another,
    even if indexing was killed partway. Each search re-lists the project, so
    added and deleted files trigger a re-index, and every returned chunk's file
    is checked against its recorded SHA-256 so an edit that kept the same size
    and timestamp is never cited from stale text.
    """

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._embedder: Any | None = None
        self._enabled: bool | None = None
        self._loaded: tuple[Any, list[dict[str, Any]], dict[str, Any]] | None = None
        self._loaded_stamp: tuple[int, int] | None = None

    def _check_enabled(self) -> bool:
        if self._enabled is None:
            self._enabled = importlib.util.find_spec("fastembed") is not None
        return self._enabled

    def _embedder_instance(self) -> Any:
        if self._embedder is None:
            from fastembed import TextEmbedding  # type: ignore[import-not-found]

            self._embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return self._embedder

    def _index_path(self) -> Path:
        return self.index_dir / "index.npz"

    def _clear_embedding_cache(self) -> None:
        self._loaded = None
        self._loaded_stamp = None

    def _load(self) -> tuple[Any, list[dict[str, Any]], dict[str, Any]] | None:
        """(unit-normalized embeddings, chunks, manifest), cached per on-disk index version."""
        import numpy as np

        path = self._index_path()
        try:
            stat = path.stat()
        except OSError:
            return None
        stamp = (stat.st_mtime_ns, stat.st_size)
        if self._loaded is not None and stamp == self._loaded_stamp:
            return self._loaded
        try:
            with np.load(path, allow_pickle=False) as data:
                matrix = data["embeddings"]
                meta = json.loads(str(data["meta"]))
        except (OSError, ValueError, KeyError) as exc:
            log.warning("Semantic index %s is unreadable; rebuilding: %s", path, exc)
            return None
        if not isinstance(meta, dict):
            return None
        chunks, manifest = meta.get("chunks", []), meta.get("manifest", {})
        if not isinstance(chunks, list) or not isinstance(manifest, dict):
            return None
        if matrix.ndim != 2 or not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all() or len(chunks) != len(matrix):
            log.warning("Semantic index %s is inconsistent; rebuilding", path)
            return None
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True) if len(matrix) else matrix
        if len(matrix) and (not np.isfinite(norms).all() or (norms == 0).any()):
            return None
        unit_matrix = matrix / np.where(norms == 0, 1, norms) if len(matrix) else matrix
        self._loaded = (unit_matrix, chunks, manifest)
        self._loaded_stamp = stamp
        return self._loaded

    @staticmethod
    def _indexable_files(root: Path) -> list[tuple[Path, os.stat_result]]:
        found = []
        for path in _project_files(root):
            try:
                file_stat = path.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(file_stat.st_mode) or not _is_text_file(path):
                continue
            if file_stat.st_size <= MAX_FILE_SIZE:
                found.append((path, file_stat))
        return found

    def _is_stale(self, root: Path, manifest: dict[str, Any]) -> bool:
        if manifest.get("root") != str(root):
            return True
        recorded = {entry["path"]: entry for entry in manifest.get("files", [])}
        current = self._indexable_files(root)
        if len(current) != len(recorded):
            return True
        for path, file_stat in current:
            entry = recorded.get(path.relative_to(root).as_posix())
            if (
                entry is None
                or entry.get("mtime_ns") != file_stat.st_mtime_ns
                or entry.get("size") != file_stat.st_size
            ):
                return True
            try:
                if hashlib.sha256(_read_project_bytes(path, root)).hexdigest() != entry.get("sha256"):
                    return True
            except OSError:
                return True
        return False

    def index(self, root: Path) -> dict[str, Any]:
        """(Re-)index the project and return a status dict."""
        if not self._check_enabled():
            raise RuntimeError(
                "Semantic search requires the 'semantic' extra. "
                "Install with: pip install 'arc-llama[semantic]'"
            )
        import numpy as np

        root = root.resolve()
        chunks: list[CodeChunk] = []
        files: list[dict[str, Any]] = []
        for path, file_stat in self._indexable_files(root):
            try:
                data = _read_project_bytes(path, root)
                digest = hashlib.sha256(data).hexdigest()
            except OSError:
                continue
            chunks.extend(_chunk_file(path, root, data.decode("utf-8", errors="ignore")))
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "mtime_ns": file_stat.st_mtime_ns,
                    "size": file_stat.st_size,
                    "sha256": digest,
                }
            )

        if chunks:
            embedder = self._embedder_instance()
            matrix = np.vstack(list(embedder.embed([c.text for c in chunks]))).astype(np.float32)
            if matrix.ndim != 2 or len(matrix) != len(chunks) or not matrix.shape[1] or not np.isfinite(matrix).all():
                raise ValueError("Embedder returned invalid or non-finite vectors")
            if (np.linalg.norm(matrix.astype(np.float64), axis=1) == 0).any():
                raise ValueError("Embedder returned zero vectors")
        else:
            matrix = np.zeros((0, 0), dtype=np.float32)
        meta = {
            "manifest": {"root": str(root), "files": files, "chunk_count": len(chunks)},
            "chunks": [{"path": c.path, "start_line": c.start_line, "text": c.text[:500]} for c in chunks],
        }
        fd, name = tempfile.mkstemp(prefix=".index-", suffix=".tmp.npz", dir=self.index_dir)
        tmp = Path(name)
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, embeddings=matrix, meta=np.array(json.dumps(meta)))
            os.replace(tmp, self._index_path())
        finally:
            tmp.unlink(missing_ok=True)
        self._clear_embedding_cache()
        # Files from the earlier three-part layout are no longer read.
        for old in ("embeddings.npy", "chunks.json", "manifest.json"):
            (self.index_dir / old).unlink(missing_ok=True)
        return {"indexed_files": len(files), "chunks": len(chunks)}

    def search(self, root: Path, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k most semantically similar code chunks."""
        if not self._check_enabled():
            raise RuntimeError(
                "Semantic search requires the 'semantic' extra. "
                "Install with: pip install 'arc-llama[semantic]'"
            )
        root = root.resolve()
        for attempt in range(2):
            loaded = self._load()
            if loaded is None or self._is_stale(root, loaded[2]):
                self.index(root)
                loaded = self._load()
            if loaded is None:
                return []
            results = self._rank(loaded, query, top_k)
            if self._cites_changed_file(root, loaded[2], results):
                # Same size and timestamp, different bytes: rebuild once and rank again.
                if attempt == 0:
                    self.index(root)
                    continue
                raise RuntimeError("Project files kept changing during search; retry")
            return results
        return results

    def _rank(
        self, loaded: tuple[Any, list[dict[str, Any]], dict[str, Any]], query: str, top_k: int
    ) -> list[dict[str, Any]]:
        import numpy as np

        unit_matrix, chunks, _ = loaded
        if not chunks:
            return []
        query_embedding = list(self._embedder_instance().embed([query]))[0]
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim != 1 or query_embedding.shape[0] != unit_matrix.shape[1] or not np.isfinite(query_embedding).all():
            raise ValueError("Embedder returned an invalid query vector")
        q_norm = np.linalg.norm(query_embedding.astype(np.float64))
        if not np.isfinite(q_norm) or q_norm == 0:
            raise ValueError("Embedder returned an invalid query vector")
        q_unit = query_embedding / q_norm if q_norm else query_embedding
        scores = unit_matrix @ q_unit
        results = []
        for idx in np.argsort(scores)[::-1][:top_k]:
            chunk = chunks[int(idx)]
            results.append(
                {
                    "path": chunk["path"],
                    "start_line": chunk["start_line"],
                    "score": round(float(scores[int(idx)]), 4),
                    "snippet": chunk["text"],
                }
            )
        return results

    @staticmethod
    def _cites_changed_file(root: Path, manifest: dict[str, Any], results: list[dict[str, Any]]) -> bool:
        digests = {entry["path"]: entry.get("sha256") for entry in manifest.get("files", [])}
        for path in {r["path"] for r in results}:
            try:
                if hashlib.sha256(_read_project_bytes(root / path, root)).hexdigest() != digests.get(path):
                    return True
            except OSError:
                return True
        return False
