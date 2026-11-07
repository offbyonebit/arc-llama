"""Tests for repo map and semantic search helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

import arc_llama.agent.repo_map as repo_map_mod
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


def test_semantic_search_requires_optional_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Hide fastembed whether or not the extra is installed, so this checks the
    # missing-dependency error instead of depending on the environment.
    real_find_spec = repo_map_mod.importlib.util.find_spec
    monkeypatch.setattr(
        repo_map_mod.importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: (
            None if name == "fastembed" else real_find_spec(name, *args, **kwargs)
        ),
    )
    index = SemanticIndex(tmp_path / "idx")
    with pytest.raises(RuntimeError, match="semantic"):
        index.index(tmp_path)


def test_semantic_search_indexes_and_ranks_with_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    np = pytest.importorskip("numpy")
    root = tmp_path / "project"
    root.mkdir()
    (root / "auth.py").write_text("def authenticate():\n    pass\n", encoding="utf-8")
    (root / "other.py").write_text("def render_chart():\n    pass\n", encoding="utf-8")

    class FakeEmbedder:
        """Deterministic stand-in so no embedding model is downloaded."""

        def embed(self, texts: list[str]):
            return [
                np.asarray(
                    (1.0, 0.0) if "authent" in text.lower() else (0.0, 1.0),
                    dtype=np.float32,
                )
                for text in texts
            ]

    index = SemanticIndex(tmp_path / "idx")
    monkeypatch.setattr(index, "_check_enabled", lambda: True)
    monkeypatch.setattr(index, "_embedder_instance", lambda: FakeEmbedder())

    stats = index.index(root)
    assert stats["indexed_files"] == 2
    results = index.search(root, "authentication logic", top_k=2)
    assert len(results) == 2
    assert "authenticate" in results[0]["snippet"]
    assert results[0]["score"] > results[1]["score"]


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
    original_stat = Path.lstat

    def counted_stat(path: Path, *args, **kwargs):
        nonlocal stat_calls
        if path == source:
            stat_calls += 1
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", counted_stat)
    index.index(root)
    # The explicit metadata capture is shared by size and manifest.
    # Observe the explicit lstat boundary; Path.stat is no longer called by
    # Path.lstat on Python 3.14.
    assert 1 <= stat_calls <= 2

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


def test_repo_map_prunes_ignored_trees_before_traversal(tmp_path, monkeypatch):
    import os

    root = tmp_path / "project"
    root.mkdir()
    (root / "visible.py").write_text("def visible():\n    pass\n")
    for name in ["node_modules", ".venv-310", "package.egg-info"]:
        directory = root / name / "nested"
        directory.mkdir(parents=True)
        (directory / "ignored.py").write_text("def ignored():\n    pass\n")
    visited = []
    original_walk = os.walk

    def counted_walk(*args, **kwargs):
        for item in original_walk(*args, **kwargs):
            visited.append(Path(item[0]))
            yield item

    monkeypatch.setattr("arc_llama.agent.repo_map.os.walk", counted_walk)
    assert build_repo_map(root) == "visible.py: visible"
    assert visited == [root]


def test_repo_map_handles_file_disappearing_during_scan(tmp_path, monkeypatch):
    source = tmp_path / "deleted.py"
    source.write_text("def removed():\n    pass\n")
    original_is_text = repo_map_mod._is_text_file

    def removing_is_text(path):
        result = original_is_text(path)
        path.unlink()
        return result

    monkeypatch.setattr("arc_llama.agent.repo_map._is_text_file", removing_is_text)
    assert build_repo_map(tmp_path) == "(empty project)"


def _word_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SemanticIndex:
    """Index with a deterministic bag-of-words embedder (no model download)."""
    np = pytest.importorskip("numpy")
    import hashlib

    class WordEmbedder:
        def embed(self, texts: list[str]):
            for text in texts:
                vector = np.zeros(64, dtype=np.float32)
                for word in text.replace("(", " ").replace(")", " ").split():
                    vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
                yield vector

    index = SemanticIndex(tmp_path / "idx")
    monkeypatch.setattr(index, "_check_enabled", lambda: True)
    monkeypatch.setattr(index, "_embedder_instance", lambda: WordEmbedder())
    return index


def test_semantic_search_finds_files_added_after_indexing(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.py").write_text("def alpha_helper():\n    return 1\n")
    index = _word_index(tmp_path, monkeypatch)
    index.index(root)
    (root / "b.py").write_text("def brand_new_function():\n    return 2\n")
    assert index.search(root, "brand_new_function", top_k=1)[0]["path"] == "b.py"
    (root / "b.py").unlink()
    assert all(hit["path"] != "b.py" for hit in index.search(root, "brand_new_function", top_k=3))


def test_semantic_search_never_cites_text_from_an_equal_size_edit(tmp_path, monkeypatch):
    import os

    root = tmp_path / "project"
    root.mkdir()
    source = root / "c.py"
    source.write_text("def compute_total():\n    return 1111\n")
    before = source.stat()
    index = _word_index(tmp_path, monkeypatch)
    index.index(root)
    source.write_text("def compute_total():\n    return 2222\n")
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    snippet = index.search(root, "compute_total", top_k=1)[0]["snippet"]
    assert "2222" in snippet and "1111" not in snippet


def test_killed_reindex_keeps_the_previous_index_consistent(tmp_path, monkeypatch):
    import arc_llama.agent.repo_map as repo_map

    root = tmp_path / "project"
    root.mkdir()
    (root / "d.py").write_text("def delta():\n    return 1\n")
    index = _word_index(tmp_path, monkeypatch)
    index.index(root)
    for i in range(3):
        (root / f"e{i}.py").write_text("".join(f"def extra_{i}_{j}():\n    return {j}\n\n" for j in range(20)))

    def killed(src, dst):
        raise SystemExit("killed before publication")

    with monkeypatch.context() as patch:
        patch.setattr(repo_map.os, "replace", killed)
        with pytest.raises(SystemExit):
            index.index(root)
    fresh = _word_index(tmp_path, monkeypatch)  # a new process: no in-memory cache
    hits = fresh.search(root, "extra_1_5", top_k=2)
    assert hits and hits[0]["path"] == "e1.py"
    assert not list((tmp_path / "idx").glob(".index-*.tmp.npz"))


def test_semantic_index_shared_between_projects_never_mixes_them(tmp_path, monkeypatch):
    first, second = tmp_path / "one", tmp_path / "two"
    for root, word in ((first, "first_project_only"), (second, "second_project_only")):
        root.mkdir()
        (root / "same.py").write_text(f"def {word}():\n    return 1\n")
    index = _word_index(tmp_path, monkeypatch)
    assert "first_project_only" in index.search(first, "first_project_only", top_k=1)[0]["snippet"]
    assert "second_project_only" in index.search(second, "first_project_only", top_k=1)[0]["snippet"]
