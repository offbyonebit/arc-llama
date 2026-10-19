from __future__ import annotations

import asyncio
import json

import pytest

from arc_llama.workbench import files, sandbox
from arc_llama.workbench.runner import WorkManager
from arc_llama.workbench.store import WorkStore


def tool(name, arguments, identity="model-id"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": identity,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


async def wait_job(manager, identity, state):
    for _ in range(300):
        job = manager.store.get("job", identity)
        if job["state"] == state:
            return job
        if job["state"] == "failed" and state != "failed":
            pytest.fail(job["message"])
        await asyncio.sleep(0.01)
    pytest.fail(f"Job did not reach {state}: {job['state']}")


@pytest.fixture
async def work(tmp_path):
    source = tmp_path / "project"
    source.mkdir()
    (source / "hello.txt").write_text("original\n")
    manager = WorkManager(tmp_path / "state", "http://unused")
    project = manager.workspace(
        {
            "root": str(source),
            "model": "local",
            "name": "Project",
            "instructions": "Keep changes small",
            "verify": False,
        }
    )
    await manager.start()
    yield manager, project, source
    await manager.shutdown()


def responses(manager, monkeypatch, messages):
    requests = []

    async def generate(job):
        requests.append(json.loads(json.dumps(job["messages"])))
        return messages.pop(0)

    monkeypatch.setattr(manager, "generate", generate)
    return requests


async def changed_job(work, monkeypatch):
    manager, project, _ = work
    responses(
        manager,
        monkeypatch,
        [
            tool("write_file", {"path": "hello.txt", "content": "changed\n"}),
            tool("write_file", {"path": "new.txt", "content": "new"}),
            {"role": "assistant", "content": "Finished edits, tests were not run."},
        ],
    )
    job = await manager.create_job(project["id"], "Change two files")
    return await wait_job(manager, job["id"], "completed")


async def test_work_isolation_review_apply_undo_and_conflicts(work, monkeypatch):
    manager, _, source = work
    job = await changed_job(work, monkeypatch)
    root = manager.job_dir(job["id"])
    assert (source / "hello.txt").read_text() == "original\n"
    receipt, diff = files.changes(root / "baseline", root / "tree")
    assert "changed" in diff and "new.txt" in receipt
    with pytest.raises(ValueError, match="Review"):
        await manager.action(job["id"], "apply", {})
    body = {"review_revision": files.revision(receipt)}
    (source / "hello.txt").write_text("independent change")
    with pytest.raises(ValueError, match="independently"):
        await manager.action(job["id"], "apply", body)
    assert not (source / "new.txt").exists()
    assert manager.store.get("job", job["id"])["apply_state"] == "none"
    (source / "hello.txt").write_text("original\n")
    await manager.action(job["id"], "apply", body)
    assert (source / "hello.txt").read_text() == "changed\n"
    assert (source / "new.txt").read_text() == "new"
    (source / "hello.txt").write_text("later edit")
    with pytest.raises(ValueError, match="independently"):
        await manager.action(job["id"], "undo", {})
    assert (source / "new.txt").exists()
    (source / "hello.txt").write_text("changed\n")
    await manager.action(job["id"], "undo", {})
    assert (source / "hello.txt").read_text() == "original\n"
    assert not (source / "new.txt").exists()
    await manager.action(job["id"], "discard", {})
    assert manager.store.get("job", job["id"])["state"] == "discarded"
    assert not root.exists()


async def test_review_revision_rejects_changed_copy(work, monkeypatch):
    manager, _, source = work
    job = await changed_job(work, monkeypatch)
    root = manager.job_dir(job["id"])
    receipt, _ = files.changes(root / "baseline", root / "tree")
    (root / "tree" / "hello.txt").write_text("changed after review")
    with pytest.raises(ValueError, match="Review"):
        await manager.action(job["id"], "apply", {"review_revision": files.revision(receipt)})
    assert (source / "hello.txt").read_text() == "original\n"


async def test_resume_keeps_receipts_without_repeating_completed_tools(work, monkeypatch):
    manager, project, _ = work
    entered = asyncio.Event()

    async def generate(job):
        if job["turns"] == 0:
            return tool("write_file", {"path": "hello.txt", "content": "only once"})
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(manager, "generate", generate)
    job = await manager.create_job(project["id"], "Persist progress")
    await asyncio.wait_for(entered.wait(), 3)
    await manager.shutdown()
    restarted = WorkManager(manager.store.root, "http://unused")
    seen = responses(
        restarted, monkeypatch, [{"role": "assistant", "content": "Resumed successfully"}]
    )
    await restarted.start()
    try:
        assert restarted.store.get("job", job["id"])["state"] == "paused"
        await restarted.action(job["id"], "resume", {})
        done = await wait_job(restarted, job["id"], "completed")
        assert len([m for m in seen[0] if m["role"] == "tool"]) == 1
        assert len([e for e in restarted.store.events(job["id"]) if e["type"] == "tool_call"]) == 1
        assert done["turns"] == 2
    finally:
        await restarted.shutdown()


async def test_pause_between_steps_preserves_queued_tool(work, monkeypatch):
    manager, project, source = work
    entered, release = asyncio.Event(), asyncio.Event()

    async def generate(job):
        entered.set()
        await release.wait()
        return (
            tool("write_file", {"path": "hello.txt", "content": "after resume"})
            if job["turns"] == 0
            else {"role": "assistant", "content": "Done"}
        )

    monkeypatch.setattr(manager, "generate", generate)
    job = await manager.create_job(project["id"], "Pause this task")
    await asyncio.wait_for(entered.wait(), 3)
    await manager.action(job["id"], "pause", {})
    release.set()
    paused = await wait_job(manager, job["id"], "paused")
    assert paused["pending"][0]["state"] == "pending"
    assert (manager.job_dir(job["id"]) / "tree" / "hello.txt").read_text() == "original\n"
    await manager.action(job["id"], "resume", {})
    await wait_job(manager, job["id"], "completed")
    assert (source / "hello.txt").read_text() == "original\n"


async def test_uncertain_operation_requires_review_and_is_never_replayed(work, monkeypatch):
    manager, project, _ = work
    entered = asyncio.Event()

    async def generate(job):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(manager, "generate", generate)
    job = await manager.create_job(project["id"], "Interrupted operation")
    await entered.wait()
    job = manager.store.get("job", job["id"])
    job["pending"] = [
        {
            "id": "uncertain-call",
            "name": "write_file",
            "arguments": {"path": "do-not-repeat", "content": "x"},
            "state": "executing",
        }
    ]
    manager.store.save("job", job)
    await manager.shutdown()
    restarted = WorkManager(manager.store.root, "http://unused")
    responses(
        restarted, monkeypatch, [{"role": "assistant", "content": "Reviewed uncertain outcome"}]
    )
    await restarted.start()
    try:
        with pytest.raises(ValueError, match="acknowledge"):
            await restarted.action(job["id"], "resume", {})
        await restarted.action(job["id"], "resume", {"acknowledge_uncertain": True})
        done = await wait_job(restarted, job["id"], "completed")
        assert "uncertain" in done["messages"][-2]["content"]
        assert not (restarted.job_dir(job["id"]) / "tree" / "do-not-repeat").exists()
    finally:
        await restarted.shutdown()


async def test_approvals_survive_restart_and_do_not_approve_later_calls(work, monkeypatch):
    manager, project, _ = work
    monkeypatch.setattr(sandbox, "available", lambda: True)
    manager.workspace({"allow_commands": True}, project["id"])
    responses(manager, monkeypatch, [tool("run_command", {"command": "first"})])
    job = await manager.create_job(project["id"], "Approval test")
    first = await wait_job(manager, job["id"], "waiting_approval")
    first_id = first["pending"][0]["id"]
    await manager.shutdown()
    restarted = WorkManager(manager.store.root, "http://unused")
    responses(
        restarted,
        monkeypatch,
        [tool("run_command", {"command": "second"}), {"role": "assistant", "content": "Done"}],
    )
    executed = []

    async def command(root, text):
        executed.append(text)
        return "ok", False

    monkeypatch.setattr(sandbox, "run", command)
    await restarted.start()
    try:
        with pytest.raises(ValueError, match="boolean"):
            await restarted.action(job["id"], "approve", {"call_id": first_id, "approved": "true"})
        await restarted.action(job["id"], "approve", {"call_id": first_id, "approved": False})
        for _ in range(300):
            second = restarted.store.get("job", job["id"])
            if second["state"] == "waiting_approval" and second["pending"][0]["id"] != first_id:
                break
            await asyncio.sleep(0.01)
        with pytest.raises(ValueError, match="match"):
            await restarted.action(job["id"], "approve", {"call_id": first_id, "approved": True})
        await restarted.action(
            job["id"], "approve", {"call_id": second["pending"][0]["id"], "approved": True}
        )
        await wait_job(restarted, job["id"], "completed")
        assert executed == ["second"]
    finally:
        await restarted.shutdown()


async def test_scope_and_readonly_permissions(work, monkeypatch):
    manager, project, source = work
    responses(manager, monkeypatch, [{"role": "assistant", "content": "Ready"}])
    job = await manager.create_job(project["id"], "Permissions")
    job = await wait_job(manager, job["id"], "completed")
    for path in ("../escape", ".git/config", str(source / "hello.txt")):
        _, error = await manager.execute(
            job, {"name": "write_file", "arguments": {"path": path, "content": "bad"}}
        )
        assert error
    job["workspace"]["allow_edits"] = False
    _, error = await manager.execute(
        job, {"name": "write_file", "arguments": {"path": "hello.txt", "content": "bad"}}
    )
    assert error
    assert (source / "hello.txt").read_text() == "original\n"


def test_snapshots_skip_links_metadata_and_do_not_share_file_inodes(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "copy"
    source.mkdir()
    (source / "file").write_text("original")
    (source / ".git").mkdir()
    (source / ".git" / "config").write_text("private git config")
    try:
        (source / "link").symlink_to(source / "file")
    except OSError:
        pass
    manifest = files.copy_project(source, dest)
    assert manifest.keys() == {"file"}
    (dest / "file").write_text("copy edited")
    assert (source / "file").read_text() == "original"


def test_work_store_commits_events_with_state_and_survives_reopen(tmp_path):
    store = WorkStore(tmp_path)
    value = store.create("job", state="queued")
    value["state"] = "paused"
    store.save("job", value, {"type": "pause"})
    reopened = WorkStore(tmp_path)
    assert reopened.get("job", value["id"])["state"] == "paused"
    assert reopened.events(value["id"])[0]["type"] == "pause"
    assert not reopened.events(value["id"], after=1)


def test_command_sandbox_has_only_project_write_access_and_aggregate_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "available", lambda: True)
    args = sandbox.command_argv(tmp_path, "echo test")
    assert "--unshare-all" in args and "--clearenv" in args
    assert "MemoryMax=1G" in args and "TasksMax=64" in args
    assert args[args.index("--bind") + 1 : args.index("--bind") + 3] == [
        str(tmp_path),
        "/workspace",
    ]
    assert "/home" not in args


async def test_failed_apply_rolls_back_and_keeps_explicit_undo_journal(work, monkeypatch):
    manager, _, source = work
    job = await changed_job(work, monkeypatch)
    root = manager.job_dir(job["id"])
    receipt, _ = files.changes(root / "baseline", root / "tree")
    original = files.atomic_file
    failed = False

    def fail_second(path, data, mode=0o644):
        nonlocal failed
        if path.name == "new.txt" and data is not None and not failed:
            failed = True
            raise OSError("injected disk failure")
        return original(path, data, mode)

    monkeypatch.setattr(files, "atomic_file", fail_second)
    with pytest.raises(OSError, match="disk failure"):
        await manager.action(job["id"], "apply", {"review_revision": files.revision(receipt)})
    assert (source / "hello.txt").read_text() == "original\n"
    assert not (source / "new.txt").exists()
    assert manager.store.get("job", job["id"])["apply_state"] == "applying"
    await manager.action(job["id"], "undo", {})
    assert manager.store.get("job", job["id"])["apply_state"] == "none"


async def test_work_queue_has_one_owner(work):
    from arc_llama.workbench.store import WorkOwnerError

    manager, _, _ = work
    duplicate = WorkManager(manager.store.root, "http://unused")
    with pytest.raises(WorkOwnerError, match="Another server"):
        await duplicate.start()
    await manager.shutdown()
    await duplicate.start()
    await duplicate.shutdown()


async def test_transient_model_error_pauses_and_can_resume(work, monkeypatch):
    manager, project, _ = work

    async def failed_generate(job):
        raise OSError("model temporarily unavailable")

    monkeypatch.setattr(manager, "generate", failed_generate)
    job = await manager.create_job(project["id"], "Retry inference later")
    paused = await wait_job(manager, job["id"], "paused")
    assert "temporarily unavailable" in paused["message"]
    responses(manager, monkeypatch, [{"role": "assistant", "content": "Recovered"}])
    await manager.action(job["id"], "resume", {})
    await wait_job(manager, job["id"], "completed")


async def test_snapshot_limits_fail_without_modifying_source(work, monkeypatch):
    manager, project, source = work
    monkeypatch.setattr(files, "MAX_BYTES", 1)
    job = await manager.create_job(project["id"], "Oversized project")
    failed = await wait_job(manager, job["id"], "failed")
    assert "limit" in failed["message"]
    assert (source / "hello.txt").read_text() == "original\n"
    await manager.action(job["id"], "discard", {})


async def test_search_is_literal_and_reads_only_project_copy(work, monkeypatch):
    manager, project, source = work
    responses(manager, monkeypatch, [{"role": "assistant", "content": "Ready"}])
    job = await manager.create_job(project["id"], "Search copied files")
    job = await wait_job(manager, job["id"], "completed")
    result, error = await manager.execute(
        job, {"name": "search_files", "arguments": {"pattern": "original"}}
    )
    assert not error and "hello.txt:1" in result
    result, error = await manager.execute(
        job, {"name": "search_files", "arguments": {"pattern": "(a+)+$"}}
    )
    assert not error and result == "No matches"
    assert (source / "hello.txt").read_text() == "original\n"


async def test_structural_source_conflict_does_not_leave_apply_pending(work, monkeypatch):
    manager, _, source = work
    job = await changed_job(work, monkeypatch)
    root = manager.job_dir(job["id"])
    receipt, _ = files.changes(root / "baseline", root / "tree")
    (source / "hello.txt").unlink()
    (source / "hello.txt").mkdir()
    with pytest.raises(ValueError, match="directory"):
        await manager.action(job["id"], "apply", {"review_revision": files.revision(receipt)})
    assert manager.store.get("job", job["id"])["apply_state"] == "none"
    assert (source / "hello.txt").is_dir()
    assert not (source / "new.txt").exists()


async def test_shutdown_keeps_queue_ownership_until_snapshot_finishes(work, monkeypatch):
    import threading

    from arc_llama.workbench.store import WorkOwnerError

    manager, project, source = work
    started, release = threading.Event(), threading.Event()
    original = files.copy_project

    def slow_copy(*args):
        started.set()
        if not release.wait(5):
            raise RuntimeError("Test did not release the copy thread")
        return original(*args)

    monkeypatch.setattr(files, "copy_project", slow_copy)
    job = await manager.create_job(project["id"], "Copy and then stop")
    assert await asyncio.to_thread(started.wait, 2)
    stopping = asyncio.create_task(manager.shutdown())
    duplicate = WorkManager(manager.store.root, "http://unused")
    try:
        await asyncio.sleep(0.02)
        assert not stopping.done()
        with pytest.raises(WorkOwnerError):
            await duplicate.start()
    finally:
        release.set()
        await stopping
    await duplicate.start()
    try:
        assert duplicate.store.get("job", job["id"])["state"] == "failed"
        assert (source / "hello.txt").read_text() == "original\n"
    finally:
        await duplicate.shutdown()
