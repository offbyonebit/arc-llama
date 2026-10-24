from __future__ import annotations

import pytest
from test_work import tool, wait_job
from test_work_verify import script

from arc_llama.workbench import attempts
from arc_llama.workbench.runner import WorkManager


def done(text="Done."):
    return {"role": "assistant", "content": text}


@pytest.fixture
async def manager(tmp_path):
    source = tmp_path / "project"
    source.mkdir()
    (source / "keep.txt").write_text("keep")
    work = WorkManager(tmp_path / "state", "http://unused")
    project = work.workspace({"root": str(source), "model": "local", "max_attempts": 3})
    await work.start()
    yield work, project
    await work.shutdown()


def test_workspace_attempt_settings_are_validated(tmp_path):
    source = tmp_path / "p"
    source.mkdir()
    work = WorkManager(tmp_path / "state", "http://unused")
    base = {"root": str(source), "model": "local"}
    with pytest.raises(ValueError, match="max_attempts"):
        work.workspace({**base, "max_attempts": 9})
    with pytest.raises(ValueError, match="time_budget_minutes"):
        work.workspace({**base, "time_budget_minutes": 0})
    with pytest.raises(ValueError, match="checking"):
        work.workspace({**base, "max_attempts": 2, "verify": False})
    assert work.workspace({**base, "verify": False})["max_attempts"] == 1


def test_budget_blocks_an_attempt_that_would_not_finish_in_time():
    job = {
        "workspace": {"max_attempts": 3, "time_budget_minutes": 10, "verify": True},
        "attempts": [{"seconds": 400.0}],
        "started_at": 1000.0,
    }
    entry = {"verdict": "failed"}
    assert attempts.may_start_another(job, entry, now=1100.0)
    assert not attempts.may_start_another(job, entry, now=1300.0)
    assert not attempts.may_start_another(job, {"verdict": "verified"}, now=1100.0)


async def test_failed_attempt_is_replaced_by_a_verified_one_on_a_fresh_copy(manager, monkeypatch):
    work, project = manager
    script(
        work,
        monkeypatch,
        [done(), done(), done(), tool("write_file", {"path": "out.txt", "content": "ok"}), done()],
        {"criteria": [{"type": "exists", "path": "out.txt"}]},
    )
    job = await work.create_job(project["id"], "Make out.txt")
    final = await wait_job(work, job["id"], "completed")
    assert [a["verdict"] for a in final["attempts"]] == ["failed", "verified"]
    assert final["chosen_attempt"] == 2
    assert final["verification"]["verdict"] == "verified"
    root = work.job_dir(job["id"])
    assert (root / "tree" / "out.txt").read_text() == "ok"
    assert not (root / "best").exists() and not (root / "tree.new").exists()
    assert final["message"].startswith("Kept attempt 2 of 2")


async def test_best_earlier_attempt_is_restored_when_later_ones_are_worse(manager, monkeypatch):
    work, project = manager
    script(
        work,
        monkeypatch,
        [tool("write_file", {"path": "a.txt", "content": "a"})]
        + [done()] * 3
        + [done()] * 3
        + [done()] * 3,
        {
            "criteria": [
                {"type": "exists", "path": "a.txt"},
                {"type": "exists", "path": "b.txt"},
            ]
        },
    )
    job = await work.create_job(project["id"], "Make a.txt and b.txt")
    final = await wait_job(work, job["id"], "completed")
    assert len(final["attempts"]) == 3
    assert final["chosen_attempt"] == 1
    assert (work.job_dir(job["id"]) / "tree" / "a.txt").exists()
    assert final["verification"]["verdict"] == "failed"


async def test_single_attempt_default_behaves_as_before(tmp_path, monkeypatch):
    source = tmp_path / "project"
    source.mkdir()
    work = WorkManager(tmp_path / "state", "http://unused")
    project = work.workspace({"root": str(source), "model": "local"})
    await work.start()
    try:
        script(
            work, monkeypatch, [done()] * 3, {"criteria": [{"type": "exists", "path": "nope.txt"}]}
        )
        job = await work.create_job(project["id"], "Do it")
        final = await wait_job(work, job["id"], "completed")
        assert len(final["attempts"]) == 1 and final["repairs"] == 2
    finally:
        await work.shutdown()


async def test_command_approval_cannot_cross_attempts(manager, monkeypatch):
    from arc_llama.workbench import sandbox

    work, project = manager
    monkeypatch.setattr(sandbox, "available", lambda: True)
    work.workspace({"allow_commands": True, "max_attempts": 2}, project["id"])
    script(
        work,
        monkeypatch,
        [tool("run_command", {"command": "first"})]
        + [done()] * 3
        + [tool("run_command", {"command": "second"}), done()],
        {"criteria": [{"type": "exists", "path": "out.txt"}]},
    )
    executed = []

    async def command(copy, text):
        executed.append(text)
        (copy / "out.txt").write_text("ok")
        return "ok", False

    monkeypatch.setattr(sandbox, "run", command)
    job = await work.create_job(project["id"], "Make out.txt")
    first = await wait_job(work, job["id"], "waiting_approval")
    first_id = first["pending"][0]["id"]
    await work.action(job["id"], "approve", {"call_id": first_id, "approved": False})
    second = await wait_job(work, job["id"], "waiting_approval")
    second_id = second["pending"][0]["id"]
    assert second["attempt"] == 2
    assert second_id != first_id
    with pytest.raises(ValueError, match="match"):
        await work.action(job["id"], "approve", {"call_id": first_id, "approved": True})
    assert executed == []
    await work.action(job["id"], "approve", {"call_id": second_id, "approved": True})
    result = await wait_job(work, job["id"], "completed")
    assert result["chosen_attempt"] == 2
    assert executed == ["second"]


@pytest.mark.parametrize(
    "kind,keep,phase",
    [
        ("rotate", keep, phase)
        for keep in (True, False)
        for phase in ("prepare", "retire", "install")
    ]
    + [("adopt", False, phase) for phase in ("retire", "install")]
    + [("cleanup", False, "cleanup")],
)
async def test_restart_finishes_copy_transition_without_mixing_results(
    manager, monkeypatch, kind, keep, phase
):
    work, project = manager
    script(work, monkeypatch, [done()], {"criteria": [{"type": "exists", "path": "keep.txt"}]})
    job = await work.create_job(project["id"], "Keep the file")
    job = await wait_job(work, job["id"], "completed")
    root = work.job_dir(job["id"])
    (root / "tree" / "keep.txt").write_text("current attempt")
    (root / "best").mkdir()
    (root / "best" / "keep.txt").write_text("earlier best")
    job["state"] = "running" if kind == "rotate" else "completed"
    job["attempt"] = 2 if kind == "rotate" else 1
    original = WorkManager._copy_step

    def stop_after_step(path, transition):
        original(path, transition)
        if transition["phase"] == phase:
            raise RuntimeError("simulated server stop")

    monkeypatch.setattr(WorkManager, "_copy_step", staticmethod(stop_after_step))
    with pytest.raises(RuntimeError, match="simulated"):
        await work._transition_copy(job, kind, keep=keep)
    saved = work.store.get("job", job["id"])
    assert saved["state"] == "settling"
    for action in ("apply", "discard", "undo"):
        with pytest.raises(ValueError, match="Pause or finish"):
            await work.action(job["id"], action, {})
    await work.shutdown()
    monkeypatch.setattr(WorkManager, "_copy_step", staticmethod(original))
    restarted = WorkManager(work.store.root, "http://unused")
    await restarted.start()
    try:
        recovered = restarted.store.get("job", job["id"])
        assert "copy_transition" not in recovered
        assert recovered["state"] == ("paused" if kind == "rotate" else "completed")
        expected = (
            "keep" if kind == "rotate" else "earlier best" if kind == "adopt" else "current attempt"
        )
        assert (root / "tree" / "keep.txt").read_text() == expected
        if kind == "rotate":
            assert (root / "best" / "keep.txt").read_text() == (
                "current attempt" if keep else "earlier best"
            )
        else:
            assert not (root / "best").exists()
        assert not (root / "tree.new").exists()
    finally:
        await restarted.shutdown()


async def test_legacy_job_can_resume_without_attempt_metadata(manager, monkeypatch):
    work, project = manager
    script(work, monkeypatch, [done()], {"criteria": [{"type": "exists", "path": "keep.txt"}]})
    job = await work.create_job(project["id"], "Keep the file")
    job = await wait_job(work, job["id"], "completed")
    for field in ("attempt", "attempts", "chosen_attempt", "verification_by_attempt"):
        job.pop(field, None)
    job["state"] = "paused"
    work.store.save("job", job)
    script(work, monkeypatch, [done()], {})
    await work.action(job["id"], "resume", {})
    final = await wait_job(work, job["id"], "completed")
    assert final["chosen_attempt"] == 1
    assert len(final["attempts"]) == 1


async def test_pause_during_acceptance_planning_is_preserved(manager, monkeypatch):
    import asyncio

    work, project = manager
    started, release = asyncio.Event(), asyncio.Event()

    async def plan(job):
        started.set()
        await release.wait()
        return []

    async def should_not_run(job):
        pytest.fail("Generation started despite a pause during planning")

    monkeypatch.setattr(work, "plan_criteria", plan)
    monkeypatch.setattr(work, "generate", should_not_run)
    job = await work.create_job(project["id"], "Plan then pause")
    await asyncio.wait_for(started.wait(), 3)
    await work.action(job["id"], "pause", {})
    release.set()
    paused = await wait_job(work, job["id"], "paused")
    assert paused["turns"] == 0
