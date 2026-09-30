"""Tests for repo map and semantic search helpers."""
from __future__ import annotations

from pathlib import Path

import pytest

from arc_llama.agent.repo_map import SemanticIndex, build_repo_map


def test_build_repo_map(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def hello():\n    pass\n\nclass Greeter:\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")

    text = build_repo_map(tmp_path, max_entries=50)
    assert "src/main.py" in text
    assert "hello" in text
    assert "Greeter" in text
    assert "README.md" in text


def test_semantic_search_requires_optional_dependency(tmp_path: Path) -> None:
    index = SemanticIndex(tmp_path / "idx")
    # If fastembed is installed this will index; if not it raises RuntimeError.
    try:
        import fastembed  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="semantic"):
            index.index(tmp_path)
        return

    # When the optional dep is present, exercise the full flow.
    (tmp_path / "main.py").write_text(
        "def authenticate():\n    pass\n\ndef login():\n    pass\n",
        encoding="utf-8",
    )
    stats = index.index(tmp_path)
    assert stats["indexed_files"] == 1
    results = index.search(tmp_path, "authentication logic", top_k=2)
    assert len(results) <= 2
    assert any("authenticate" in r["path"] or "authenticate" in r["snippet"] for r in results)


def test_semantic_search_reuses_normalized_matrix_and_refreshes_after_reindex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    np = pytest.importorskip("numpy")
    root = tmp_path / "project"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def first():\n    pass\n", encoding="utf-8")

    class FakeEmbedder:
        vector = (1.0, 0.0)

        def embed(self, texts: list[str]):
            return [
                np.asarray((1.0, 0.0) if text == "query" else self.vector, dtype=np.float32)
                for text in texts
            ]

    embedder = FakeEmbedder()
    index = SemanticIndex(tmp_path / "idx")
    monkeypatch.setattr(index, "_check_enabled", lambda: True)
    monkeypatch.setattr(index, "_embedder_instance", lambda: embedder)

    stat_calls = 0
    original_stat = Path.stat

    def counted_stat(path: Path, *args, **kwargs):
        nonlocal stat_calls
        if path == source:
            stat_calls += 1
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", counted_stat)
    index.index(root)
    # One stat backs is_file(); the explicit metadata capture is shared by
    # the size check and manifest construction.
    assert stat_calls == 2

    loads = 0
    original_load = np.load

    def counted_load(*args, **kwargs):
        nonlocal loads
        loads += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counted_load)
    first = index.search(root, "query")
    second = index.search(root, "query")
    assert first == second
    assert loads == 1

    embedder.vector = (0.0, 1.0)
    index.index(root)
    refreshed = index.search(root, "query")
    assert refreshed and all(result["score"] == 0.0 for result in refreshed)
    assert loads == 2
