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

