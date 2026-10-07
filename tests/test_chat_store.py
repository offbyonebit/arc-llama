"""Tests for the chat-history store and server endpoints."""
from __future__ import annotations

from pathlib import Path

import pytest

from arc_llama.chat_store import ChatMessage, ChatStore


@pytest.fixture
def store(tmp_path: Path) -> ChatStore:
    return ChatStore(tmp_path / "chats")


def test_create_and_get(store: ChatStore) -> None:
    chat = store.create("chat-1", "First chat")
    assert chat.id == "chat-1"
    assert chat.title == "First chat"
    assert chat.messages == []

    loaded = store.get("chat-1")
    assert loaded is not None
    assert loaded.title == "First chat"


def test_create_duplicate_raises(store: ChatStore) -> None:
    store.create("chat-1", "First chat")
    with pytest.raises(FileExistsError):
        store.create("chat-1", "Duplicate")


def test_save_appends_messages(store: ChatStore) -> None:
    chat = store.create("chat-1", "First chat")
    chat.messages.append(ChatMessage(role="user", content="hello"))
    chat.messages.append(ChatMessage(role="assistant", content="hi"))
    store.save(chat)

    loaded = store.get("chat-1")
    assert loaded is not None
    assert len(loaded.messages) == 2
    assert loaded.messages[0].role == "user"
    assert loaded.messages[0].content == "hello"
    assert loaded.messages[1].role == "assistant"
    assert loaded.updated_at >= chat.created_at


def test_list_chats_sorted_by_updated(store: ChatStore) -> None:
    chat_a = store.create("a", "A")
    store.create("b", "B")
    chat_a.messages.append(ChatMessage(role="user", content="update"))
    store.save(chat_a)

    chats = store.list_chats()
    assert [c.id for c in chats] == ["a", "b"]


def test_delete(store: ChatStore) -> None:
    store.create("chat-1", "First chat")
    assert store.delete("chat-1") is True
    assert store.get("chat-1") is None
    assert store.delete("chat-1") is False


def test_search_matches_title(store: ChatStore) -> None:
    store.create("project-x", "Project X planning")
    store.create("notes", "Random notes")
    results = store.search("project x")
    assert len(results) == 1
    assert results[0][0].id == "project-x"
    assert results[0][1] == [-1]


def test_search_matches_message_content(store: ChatStore) -> None:
    chat = store.create("chat-1", "Untitled chat")
    chat.messages.append(ChatMessage(role="user", content="I love rust"))
    chat.messages.append(ChatMessage(role="assistant", content="Rust is great"))
    store.save(chat)

    results = store.search("rust")
    assert len(results) == 1
    assert results[0][1] == [0, 1]


def test_summary(store: ChatStore) -> None:
    chat = store.create("chat-1", "Summary test")
    chat.messages.append(ChatMessage(role="user", content="hi"))
    summary = chat.summary()
    assert summary["id"] == "chat-1"
    assert summary["title"] == "Summary test"
    assert summary["message_count"] == 1
    assert "created_at" in summary
    assert "updated_at" in summary


def test_create_with_folder(store: ChatStore) -> None:
    chat = store.create("chat-1", "Work chat", folder="work")
    assert chat.folder == "work"
    assert (store.directory / "work" / "chat-1.json").exists()


def test_list_chats_by_folder(store: ChatStore) -> None:
    store.create("a", "Root chat")
    store.create("b", "Work chat", folder="work")
    store.create("c", "Personal chat", folder="personal")

    root_chats = store.list_chats(folder="")
    assert [c.id for c in root_chats] == ["a"]

    work_chats = store.list_chats(folder="work")
    assert [c.id for c in work_chats] == ["b"]

    all_chats = store.list_chats()
    assert {c.id for c in all_chats} == {"a", "b", "c"}


def test_move_chat_between_folders(store: ChatStore) -> None:
    store.create("chat-1", "Work chat", folder="work")
    moved = store.move("chat-1", "personal")
    assert moved.folder == "personal"
    assert (store.directory / "personal" / "chat-1.json").exists()
    assert not (store.directory / "work" / "chat-1.json").exists()


def test_list_folders(store: ChatStore) -> None:
    store.create("a", "Root chat")
    store.create("b", "Work chat", folder="work")
    store.create("c", "Another work chat", folder="work")

    folders = store.list_folders()
    by_name = {f["name"]: f["count"] for f in folders}
    assert by_name.get("") == 1
    assert by_name.get("work") == 2


def test_folder_duplicate_id_global(store: ChatStore) -> None:
    store.create("chat-1", "Root chat")
    with pytest.raises(FileExistsError):
        store.create("chat-1", "Work chat", folder="work")


def test_chat_path_cache_avoids_walk_and_tracks_move_and_delete(
    store: ChatStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.create("chat-1", "Work chat", folder="work")

    walks = 0
    original_rglob = Path.rglob

    def counted_rglob(path: Path, pattern: str):
        nonlocal walks
        if path == store.directory:
            walks += 1
        return original_rglob(path, pattern)

    monkeypatch.setattr(Path, "rglob", counted_rglob)

    assert store.get("chat-1") is not None
    assert store.get("chat-1") is not None
    assert walks == 1

    store.move("chat-1", "personal")
    assert store.get("chat-1") is not None
    assert walks == 1

    assert store.delete("chat-1")
    assert store.get("chat-1") is None
    assert walks == 2



def test_failed_atomic_replace_preserves_existing_chat(store, monkeypatch):
    chat = store.create("chat-1", "Original")
    chat.title = "Changed"
    original_replace = Path.replace

    def fail_chat_replace(path, target):
        if Path(target).name == "chat-1.json":
            raise OSError("disk full")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_chat_replace)
    with pytest.raises(OSError, match="disk full"):
        store.save(chat)
    assert store.get("chat-1").title == "Original"
    assert list(store.directory.iterdir()) == [store.directory / "chat-1.json"]


def test_import_overwrite_moves_existing_chat_without_duplicates(store):
    store.create("chat-1", "Original", folder="work")
    result = store.import_chats([{ "id": "chat-1", "title": "Imported", "folder": "personal"}], overwrite=True)
    assert result["imported"] == 1
    assert len(store.list_chats()) == 1
    assert store.get("chat-1").title == "Imported"
    assert not (store.directory / "work" / "chat-1.json").exists()


@pytest.mark.parametrize("payload", ['[]', '{"messages": [null]}', '{"updated_at": "invalid"}'])
def test_malformed_chat_is_skipped_without_breaking_other_chats(store, payload):
    store.create("good", "Good")
    (store.directory / "bad.json").write_text(payload)
    assert store.get("bad") is None
    assert [chat.id for chat in store.list_chats()] == ["good"]


@pytest.mark.parametrize("folder", [".", ".."])
def test_reserved_folder_cannot_escape_or_alias_store_root(store, folder):
    with pytest.raises(ValueError):
        store.create("unsafe", "Unsafe", folder=folder)
    assert not (store.directory.parent / "unsafe.json").exists()


def test_summary_cache_reuses_reads_and_refreshes_after_external_edit(store, monkeypatch):
    import json
    chat = store.create('a', 'Original', [ChatMessage('user', 'message')])
    reads = []
    original_read = Path.read_text
    def counted_read(path, *args, **kwargs):
        reads.append(path)
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', counted_read)
    first = store.list_summaries()
    first[0]['title'] = 'Cannot poison cache'
    assert store.list_summaries()[0]['title'] == 'Original'
    assert store.list_folders() == [{'name': '', 'count': 1}]
    assert len(reads) == 1
    data = chat.to_dict()
    data['title'] = 'External edit with different size'
    (store.directory / 'a.json').write_text(json.dumps(data))
    assert store.list_summaries()[0]['title'] == data['title']
    assert len(reads) == 2
    store.move('a', 'work')
    assert store.list_summaries()[0]['folder'] == 'work'
    store.delete('a')
    assert store.list_summaries() == []
    assert store._summary_cache == {}


def test_summary_cache_tracks_save_folder_filters_and_corrupt_files(store):
    chat = store.create('a', 'Original')
    store.create('b', 'Work', folder='work')
    assert len(store.list_summaries()) == 2
    chat.title = 'Saved'
    store.save(chat)
    assert store.list_summaries(folder='')[0]['title'] == 'Saved'
    assert store.list_summaries(folder='work')[0]['id'] == 'b'
    (store.directory / 'a.json').write_text('[]')
    assert [s['id'] for s in store.list_summaries()] == ['b']


def test_failed_flush_preserves_saved_history(tmp_path, monkeypatch):
    import os

    import pytest
    store = ChatStore(tmp_path / 'chats')
    chat = store.create('flush-failure', 'Original')
    original = (store.directory / 'flush-failure.json').read_bytes()
    chat.title = 'Replacement'

    def failed_fsync(fd):
        raise OSError('disk full')

    monkeypatch.setattr(os, 'fsync', failed_fsync)
    with pytest.raises(OSError, match='disk full'):
        store.save(chat)
    assert (store.directory / 'flush-failure.json').read_bytes() == original
    assert not list(store.directory.glob('*.tmp'))


def test_legacy_null_folder_remains_readable(tmp_path):
    import json
    store = ChatStore(tmp_path / 'chats')
    (store.directory / 'legacy.json').write_text(json.dumps({'id': 'legacy', 'folder': None}))
    assert store.get('legacy').folder == ''
    assert store.list_summaries()[0]['folder'] == ''


def test_corrupt_unrepresentable_timestamp_is_skipped(tmp_path):
    import json
    store = ChatStore(tmp_path / 'chats')
    (store.directory / 'corrupt.json').write_text(json.dumps({'id': 'corrupt', 'updated_at': 10 ** 1000}))
    assert store.get('corrupt') is None
    assert store.list_summaries() == []
