"""Native history uses synthetic databases only; never inspect personal chats."""
from __future__ import annotations

import json
import sqlite3

import pytest

from arc_llama.agent import harness_history as history


@pytest.fixture
def databases(tmp_path, monkeypatch):
    paths = {"opencode": tmp_path / "opencode.db", "hermes": tmp_path / "hermes.db"}
    monkeypatch.setattr(history, "history_paths", lambda: paths)
    with sqlite3.connect(paths["opencode"]) as connection:
        connection.executescript("""
            CREATE TABLE session (id TEXT, title TEXT, directory TEXT, parent_id TEXT,
                time_updated INTEGER, time_archived INTEGER);
            CREATE TABLE message (id TEXT, session_id TEXT, data TEXT, time_created INTEGER);
            CREATE TABLE part (id TEXT, message_id TEXT, session_id TEXT, data TEXT, time_created INTEGER);
            INSERT INTO session VALUES ('ses_fixture', 'Synthetic OpenCode chat', '/synthetic/project', NULL, 2000, NULL);
            INSERT INTO session VALUES ('ses_child', 'Child', '/synthetic/project', 'ses_fixture', 3000, NULL);
            INSERT INTO session VALUES ('ses_archived', 'Archived', '/synthetic/project', NULL, 4000, 4000);
        """)
        for index, (role, text) in enumerate([
            ("system", "Do not import system policy"), ("user", "Create a synthetic file"),
            ("assistant", "The synthetic file is ready"), ("user", "Explain the synthetic file"),
            ("assistant", "It contains a fixture"),
        ]):
            connection.execute("INSERT INTO message VALUES (?, 'ses_fixture', ?, ?)",
                               (f"m{index}", json.dumps({"role": role}), 100 + index))
            connection.execute("INSERT INTO part VALUES (?, ?, 'ses_fixture', ?, ?)",
                               (f"p{index}", f"m{index}", json.dumps({"type": "text", "text": text}), 100 + index))
        connection.execute("INSERT INTO part VALUES ('tool', 'm2', 'ses_fixture', ?, 105)",
                           (json.dumps({"type": "tool", "state": {"input": {"command": "DO NOT REPLAY"}}}),))
    with sqlite3.connect(paths["hermes"]) as connection:
        connection.executescript("""
            CREATE TABLE sessions (id TEXT, title TEXT, cwd TEXT, parent_session_id TEXT,
                started_at REAL, ended_at REAL, archived INTEGER);
            CREATE TABLE messages (id INTEGER, session_id TEXT, role TEXT, content TEXT, timestamp REAL, active INTEGER);
            INSERT INTO sessions VALUES ('fixture_hermes', 'Synthetic Hermes chat', '/synthetic/hermes', NULL, 1, 3, 0);
            INSERT INTO messages VALUES (1, 'fixture_hermes', 'user', 'Synthetic question', 1, 1);
            INSERT INTO messages VALUES (2, 'fixture_hermes', 'assistant', 'Synthetic answer', 2, 1);
            INSERT INTO messages VALUES (3, 'fixture_hermes', 'assistant', 'Rewound answer', 3, 0);
            INSERT INTO messages VALUES (4, 'fixture_hermes', 'tool', 'Do not replay tool output', 3, 1);
        """)
    return paths


def test_lists_top_level_unarchived_sessions_without_message_contents(databases):
    sessions, warnings = history.list_native_sessions()
    assert not warnings
    assert [row["id"] for row in sessions] == ["hermes-fixture_hermes", "oc-ses_fixture"]
    assert sessions[1]["tasks"] == 2
    assert sessions[1]["folder"] == "/synthetic/project"
    assert all(row["native"] for row in sessions)
    assert "Synthetic answer" not in json.dumps(sessions)


def test_opencode_imports_text_in_order_and_never_replays_tools_or_system_policy(databases):
    before = databases["opencode"].read_bytes()
    record = history.get_native_session("oc-ses_fixture")
    assert record["harness"] == record["origin"] == "opencode"
    assert record["model"] == ""  # a native cloud model is never selected implicitly
    assert [run["task"] for run in record["runs"]] == ["Create a synthetic file", "Explain the synthetic file"]
    assert record["runs"][0]["answer"] == ["The synthetic file is ready"]
    assert record["runs"][1]["answer"] == ["It contains a fixture"]
    assert all(run["steps"] == [] and run["status"] == "done" and run["imported"] for run in record["runs"])
    assert "DO NOT REPLAY" not in json.dumps(record)
    assert "system policy" not in json.dumps(record)
    assert databases["opencode"].read_bytes() == before


def test_hermes_imports_text_and_respects_rewind(databases):
    record = history.get_native_session("hermes-fixture_hermes")
    assert record["runs"][0]["answer"] == ["Synthetic answer"]
    assert record["folder"] == "/synthetic/hermes"
    assert "Rewound" not in json.dumps(record)
    assert "tool output" not in json.dumps(record)


def test_missing_databases_are_not_created(tmp_path, monkeypatch):
    missing = tmp_path / "missing.db"
    monkeypatch.setattr(history, "history_paths", lambda: {"opencode": missing, "hermes": missing})
    assert history.list_native_sessions() == ([], [])
    assert history.get_native_session("oc-ses_fixture") is None
    assert not missing.exists()


def test_incompatible_database_is_reported_without_hiding_other_history(databases):
    with sqlite3.connect(databases["opencode"]) as connection:
        connection.execute("DROP TABLE session")
    sessions, warnings = history.list_native_sessions()
    assert len(sessions) == len(warnings) == 1
    assert sessions[0]["harness"] == "hermes"
    with pytest.raises(history.NativeHistoryError, match="could not be read"):
        history.get_native_session("oc-ses_fixture")


@pytest.mark.parametrize("session_id", ["../opencode.db", "oc-' OR 1=1 --", "oc-ses_missing", "session-fixture", "oc-ses_child", "oc-ses_archived"])
def test_unknown_child_archived_or_invalid_ids_do_not_open(databases, session_id):
    assert history.get_native_session(session_id) is None


def test_database_connection_is_read_only(databases):
    with history._connect(databases["opencode"]) as connection:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM session")


def test_live_wal_history_is_visible(databases):
    with sqlite3.connect(databases["opencode"]) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("UPDATE session SET title = 'Synthetic WAL update' WHERE id='ses_fixture'")
        writer.commit()
        assert history.get_native_session("oc-ses_fixture")["title"] == "Synthetic WAL update"


def test_oversized_and_corrupt_history_is_explicit(databases, monkeypatch):
    monkeypatch.setattr(history, "MAX_SESSION_BYTES", 20)
    with pytest.raises(history.NativeHistoryError, match="too large"):
        history.get_native_session("oc-ses_fixture")
    monkeypatch.setattr(history, "MAX_SESSION_BYTES", 4 * 1024 * 1024)
    with sqlite3.connect(databases["opencode"]) as connection:
        connection.execute("UPDATE part SET data='not JSON' WHERE id='p2'")
    with pytest.raises(history.NativeHistoryError, match="could not be read"):
        history.get_native_session("oc-ses_fixture")


def test_message_limit_does_not_silently_truncate(databases, monkeypatch):
    monkeypatch.setattr(history, "MAX_MESSAGES", 2)
    with pytest.raises(history.NativeHistoryError, match="too many messages"):
        history.get_native_session("oc-ses_fixture")


def test_paths_follow_harness_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    assert history.history_paths() == {
        "opencode": tmp_path / "data" / "opencode" / "opencode.db",
        "hermes": tmp_path / "hermes" / "state.db",
    }


def test_hermes_text_blocks_exclude_attachments(databases):
    blocks = [{"type": "text", "text": "Synthetic block"}, {"type": "image_url", "image_url": {"url": "private"}}]
    with sqlite3.connect(databases["hermes"]) as connection:
        connection.execute("UPDATE messages SET content=? WHERE id=2", (json.dumps(blocks),))
    record = history.get_native_session("hermes-fixture_hermes")
    assert record["runs"][0]["answer"] == ["Synthetic block"]
    assert "private" not in json.dumps(record)
