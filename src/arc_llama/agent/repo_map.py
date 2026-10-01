"""Repo map and optional semantic code search for the agent.

The repo map needs no extra dependencies. Semantic search is opt-in via the
``semantic`` extra (fastembed + numpy) so the core package stays light.
"""
from __future__ import annotations

import fnmatch
import importlib.util
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("arc_llama.agent.repo_map")


IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv*", "venv", "build", "dist",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".eggs", "*.egg-info",
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
        re.compile(r"^(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:final\s+)?[\w<>,\s]+\s+(\w+)\s*\("),
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
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".c", ".cpp",
    ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt", ".scala", ".sh",
    ".yaml", ".yml", ".json", ".toml", ".md", ".txt", ".html", ".css",
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
        dirs[:] = [name for name in dirs
                   if not _is_ignored(base / name, root) and not (base / name).is_symlink()]
        paths.extend(base / name for name in files if not _is_ignored(base / name, root))
    return sorted(paths)


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


def _chunk_file(path: Path, root: Path) -> list[CodeChunk]:
    rel = path.relative_to(root).as_posix()
    ext = path.suffix.lower()
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []

    patterns = SYMBOL_PATTERNS.get(ext)
    if not patterns:
        # Generic chunked fallback.
        chunks: list[CodeChunk] = []
        for i in range(0, len(lines), 40):
            chunk_text = "\n".join(lines[i:i + 80])
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
    """Optional local semantic search over the project codebase."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._embedder: Any | None = None
        self._enabled: bool | None = None
        self._normalized_embeddings: Any | None = None
        self._normalized_embeddings_stamp: tuple[int, int] | None = None

    def _check_enabled(self) -> bool:
        if self._enabled is None:
            self._enabled = importlib.util.find_spec("fastembed") is not None
        return self._enabled

    def _embedder_instance(self) -> Any:
        if self._embedder is None:
            from fastembed import TextEmbedding  # type: ignore[import-not-found]
            self._embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return self._embedder

    def _manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    def _embeddings_path(self) -> Path:
        return self.index_dir / "embeddings.npy"

    def _chunks_path(self) -> Path:
        return self.index_dir / "chunks.json"

    def _clear_embedding_cache(self) -> None:
        self._normalized_embeddings = None
        self._normalized_embeddings_stamp = None

    def _load_normalized_embeddings(self) -> Any:
        """Load and normalize embeddings once per on-disk index version."""
        import numpy as np

        path = self._embeddings_path()
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if self._normalized_embeddings is not None and stamp == self._normalized_embeddings_stamp:
            return self._normalized_embeddings

        matrix = np.load(path)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        unit_matrix = matrix / np.where(norms == 0, 1, norms)
        self._normalized_embeddings = unit_matrix
        self._normalized_embeddings_stamp = stamp
        return unit_matrix

    def _is_stale(self, root: Path, manifest: dict[str, Any]) -> bool:
        for entry in manifest.get("files", []):
            path = root / entry["path"]
            try:
                file_stat = path.stat()
                if file_stat.st_mtime != entry["mtime"] or file_stat.st_size != entry["size"]:
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

        root = root.resolve()
        chunks: list[CodeChunk] = []
        files: list[dict[str, Any]] = []

        for path in _project_files(root):
            if not path.is_file():
                continue
            if not _is_text_file(path):
                continue
            try:
                file_stat = path.stat()
            except OSError:
                continue
            if file_stat.st_size > MAX_FILE_SIZE:
                continue
            file_chunks = _chunk_file(path, root)
            chunks.extend(file_chunks)
            files.append({
                "path": path.relative_to(root).as_posix(),
                "mtime": file_stat.st_mtime,
                "size": file_stat.st_size,
            })

        if not chunks:
            manifest = {"files": files, "chunk_count": 0}
            self._manifest_path().write_text(json.dumps(manifest), encoding="utf-8")
            self._chunks_path().write_text(json.dumps([]), encoding="utf-8")
            self._clear_embedding_cache()
            if self._embeddings_path().exists():
                self._embeddings_path().unlink()
            return {"indexed_files": len(files), "chunks": 0}

        embedder = self._embedder_instance()
        embeddings = list(embedder.embed([c.text for c in chunks]))

        import numpy as np
        matrix = np.vstack(embeddings).astype(np.float32)
        self._clear_embedding_cache()
        np.save(self._embeddings_path(), matrix)

        manifest = {"files": files, "chunk_count": len(chunks)}
        self._manifest_path().write_text(json.dumps(manifest), encoding="utf-8")
        self._chunks_path().write_text(
            json.dumps([
                {"path": c.path, "start_line": c.start_line, "text": c.text[:500]}
                for c in chunks
            ]),
            encoding="utf-8",
        )
        return {"indexed_files": len(files), "chunks": len(chunks)}

    def search(self, root: Path, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k most semantically similar code chunks."""
        if not self._check_enabled():
            raise RuntimeError(
                "Semantic search requires the 'semantic' extra. "
                "Install with: pip install 'arc-llama[semantic]'"
            )

        root = root.resolve()
        manifest: dict[str, Any] = {"files": []}
        if self._manifest_path().exists():
            try:
                manifest = json.loads(self._manifest_path().read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        if not self._embeddings_path().exists() or self._is_stale(root, manifest):
            self.index(root)

        if not self._chunks_path().exists():
            return []

        chunks = json.loads(self._chunks_path().read_text(encoding="utf-8"))
        if not chunks:
            return []

        embedder = self._embedder_instance()
        query_embedding = list(embedder.embed([query]))[0]

        import numpy as np
        # Cosine similarity on normalized vectors. The corpus matrix is
        # unchanged across queries, so keep its normalized form in memory.
        unit_matrix = self._load_normalized_embeddings()
        q_norm = np.linalg.norm(query_embedding)
        q_unit = query_embedding / q_norm if q_norm else query_embedding
        scores = unit_matrix @ q_unit

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            chunk = chunks[int(idx)]
            results.append({
                "path": chunk["path"],
                "start_line": chunk["start_line"],
                "score": round(float(scores[int(idx)]), 4),
                "snippet": chunk["text"],
            })
        return results
