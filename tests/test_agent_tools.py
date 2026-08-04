"""Tests for the agent tool sandbox."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from arc_llama.agent.tools import (
    apply_patch,
    list_directory,
    read_file,
    read_pdf,
    run_command,
    search_files,
    write_file,
)


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')\n")
    (tmp_path / "README.md").write_text("# Project\n")
    (tmp_path / "nested" / "deep").mkdir(parents=True)
    (tmp_path / "nested" / "deep" / "file.txt").write_text("deep content\n")
    return tmp_path


def test_read_file_relative(tmp_root: Path) -> None:
    res = read_file("src/main.py", tmp_root)
    assert not res.error
    assert res.content == "print('hello')\n"


def test_read_file_absolute_inside_root(tmp_root: Path) -> None:
    res = read_file(str(tmp_root / "README.md"), tmp_root)
    assert not res.error
    assert res.content == "# Project\n"


def test_read_file_outside_root(tmp_root: Path) -> None:
    res = read_file("../outside.txt", tmp_root)
    assert res.error
    assert "escapes project root" in res.content


def test_write_file_and_read_back(tmp_root: Path) -> None:
    res = write_file("new/file.txt", "new content", tmp_root)
    assert not res.error
    assert (tmp_root / "new" / "file.txt").read_text() == "new content"


def test_write_file_outside_root(tmp_root: Path) -> None:
    res = write_file("../../evil.txt", "bad", tmp_root)
    assert res.error
    assert "escapes project root" in res.content


def test_list_directory(tmp_root: Path) -> None:
    res = list_directory(".", tmp_root)
    assert not res.error
    lines = res.content.split("\n")
    assert any("D README.md" not in line and "README.md" in line for line in lines)
    assert any("D nested" in line for line in lines)
    assert any("D src" in line for line in lines)


def test_list_directory_nested(tmp_root: Path) -> None:
    res = list_directory("nested/deep", tmp_root)
    assert not res.error
    assert "file.txt" in res.content


def test_run_command_echo(tmp_root: Path) -> None:
    res = run_command("echo hello", tmp_root)
    assert not res.error
    assert "hello" in res.content


def test_run_command_cwd_is_root(tmp_root: Path) -> None:
    # Use the interpreter's own cwd report rather than a shell builtin like
    # `pwd` — on the Windows CI runner that resolves to Git Bash's pwd.exe,
    # which prints an MSYS-style path ("/c/Users/...") instead of a native
    # Windows one, even though the actual cwd is correct.
    res = run_command(f'"{sys.executable}" -c "import os; print(os.getcwd())"', tmp_root)
    assert not res.error
    assert str(tmp_root) in res.content


def test_run_command_blocks_git_commit(tmp_root: Path) -> None:
    res = run_command("git commit -m test", tmp_root)
    assert res.error
    assert "git history-mutating" in res.content


def test_run_command_blocks_git_push(tmp_root: Path) -> None:
    res = run_command("git push origin main", tmp_root)
    assert res.error
    assert "git history-mutating" in res.content


def test_run_command_allows_git_status(tmp_root: Path) -> None:
    res = run_command("git status", tmp_root)
    # git status is allowed, but may fail because no repo; that's a normal error
    assert "git history-mutating" not in res.content


# --- Guard bypass regressions ------------------------------------------------
# The git-mutation denylist is best-effort, not a sandbox. These tests pin the
# common bypasses that the guard is expected to catch: extra whitespace, a
# leading env-var assignment, shell separators (`;`, `&&`, `||`, `|`), and a
# wrapper prefix (`sudo`, `env`, ...). They assert the command is blocked
# before it ever reaches subprocess.run, so no git repo is needed on the host.


def test_run_command_blocks_git_commit_extra_whitespace(tmp_root: Path) -> None:
    res = run_command("git\tcommit   -m test", tmp_root)
    assert res.error
    assert "git history-mutating" in res.content


def test_run_command_blocks_git_commit_leading_env_assignment(tmp_root: Path) -> None:
    res = run_command("FOO=1 git commit -m test", tmp_root)
    assert res.error
    assert "git history-mutating" in res.content


def test_run_command_blocks_git_push_after_semicolon(tmp_root: Path) -> None:
    res = run_command("echo hi; git push", tmp_root)
    assert res.error
    assert "git history-mutating" in res.content


def test_run_command_blocks_git_push_after_and(tmp_root: Path) -> None:
    res = run_command("make && git push", tmp_root)
    assert res.error
    assert "git history-mutating" in res.content


def test_run_command_blocks_git_push_after_or(tmp_root: Path) -> None:
    res = run_command("false || git push", tmp_root)
    assert res.error
    assert "git history-mutating" in res.content


def test_run_command_blocks_git_push_after_pipe(tmp_root: Path) -> None:
    res = run_command("echo hi | git push", tmp_root)
    assert res.error
    assert "git history-mutating" in res.content


def test_run_command_blocks_sudo_git_push(tmp_root: Path) -> None:
    res = run_command("sudo git push", tmp_root)
    assert res.error
    assert "git history-mutating" in res.content


def test_run_command_blocks_env_git_commit(tmp_root: Path) -> None:
    res = run_command("env git commit -m test", tmp_root)
    assert res.error
    assert "git history-mutating" in res.content


def test_run_command_blocks_env_with_assignment_git_commit(tmp_root: Path) -> None:
    res = run_command("env FOO=1 git commit", tmp_root)
    assert res.error
    assert "git history-mutating" in res.content


def test_run_command_blocks_nested_wrapper_git_push(tmp_root: Path) -> None:
    # Multiple wrappers should be unwrapped recursively.
    res = run_command("sudo env git push", tmp_root)
    assert res.error
    assert "git history-mutating" in res.content


def test_run_command_allows_legitimate_wrapped_command(tmp_root: Path) -> None:
    # A benign command that happens to be wrapped should still run.
    res = run_command("env echo hello", tmp_root)
    assert not res.error
    assert "hello" in res.content


def test_run_command_allows_chained_benign_commands(tmp_root: Path) -> None:
    res = run_command("echo a && echo b", tmp_root)
    assert not res.error
    assert "a" in res.content
    assert "b" in res.content


def test_search_files(tmp_root: Path) -> None:
    res = search_files("hello", tmp_root)
    assert not res.error
    assert "src/main.py" in res.content


def test_search_files_with_glob(tmp_root: Path) -> None:
    res = search_files("deep", tmp_root, path_glob="*.txt")
    assert not res.error
    assert "nested/deep/file.txt" in res.content
    assert "src/main.py" not in res.content


@pytest.mark.asyncio
async def test_read_pdf_extracts_text(tmp_root: Path) -> None:
    (tmp_root / "doc.pdf").write_bytes(b"%PDF-fake")
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"filename": "doc.pdf", "text": "extracted text"}
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    res = await read_pdf("doc.pdf", tmp_root, mock_client)
    assert not res.error
    assert res.content == "extracted text"
    mock_client.post.assert_awaited_once()
    call_args = mock_client.post.call_args
    assert call_args.args[0] == "/admin/parse-pdf"
    assert "file" in call_args.kwargs.get("files", {})


@pytest.mark.asyncio
async def test_read_pdf_rejects_non_pdf(tmp_root: Path) -> None:
    (tmp_root / "doc.txt").write_text("not a pdf")
    mock_client = MagicMock()
    mock_client.post = AsyncMock()

    res = await read_pdf("doc.txt", tmp_root, mock_client)
    assert res.error
    assert "only PDF files are supported" in res.content
    mock_client.post.assert_not_awaited()


def test_apply_patch_single_occurrence(tmp_root: Path) -> None:
    (tmp_root / "src" / "main.py").write_text("print('hello')\n", encoding="utf-8")
    res = apply_patch("src/main.py", "print('hello')", "print('world')", tmp_root)
    assert not res.error
    assert (tmp_root / "src" / "main.py").read_text(encoding="utf-8") == "print('world')\n"


def test_apply_patch_replace_all(tmp_root: Path) -> None:
    (tmp_root / "src" / "main.py").write_text("aaa\nbbb\naaa\n", encoding="utf-8")
    res = apply_patch("src/main.py", "aaa", "xxx", tmp_root, replace_all=True)
    assert not res.error
    assert (tmp_root / "src" / "main.py").read_text(encoding="utf-8") == "xxx\nbbb\nxxx\n"


def test_apply_patch_ambiguous_without_replace_all(tmp_root: Path) -> None:
    (tmp_root / "src" / "main.py").write_text("aaa\nbbb\naaa\n", encoding="utf-8")
    res = apply_patch("src/main.py", "aaa", "xxx", tmp_root, replace_all=False)
    assert res.error
    assert "occurs 2 times" in res.content


def test_apply_patch_missing_old_string(tmp_root: Path) -> None:
    (tmp_root / "src" / "main.py").write_text("hello\n", encoding="utf-8")
    res = apply_patch("src/main.py", "notfound", "xxx", tmp_root)
    assert res.error
    assert "old_string not found" in res.content
