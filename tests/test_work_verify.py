from __future__ import annotations

import asyncio
import json

import pytest
from test_work import tool, wait_job

from arc_llama.workbench import verify
from arc_llama.workbench.runner import WorkManager


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "Acceptance criteria:\n{}"])
@pytest.mark.parametrize("as_object", [True, False])
def test_planner_json_preserves_every_criterion(wrapper, as_object):
    # The pinned real Qwen model returned this array for a two-file task.
    criteria = [
        {"type": "exists", "path": "A.txt"},
        {"type": "exists", "path": "B.txt"},
        {"type": "contains", "path": "A.txt", "value": "alpha"},
        {"type": "contains", "path": "B.txt", "value": "beta"},
    ]
    reply = json.dumps({"criteria": criteria} if as_object else criteria)
    parsed = verify.parse_json(wrapper.format(reply))
    normalised = verify.normalise(parsed)
    assert [(c["type"], c["path"]) for c in normalised] == [
        ("exists", "A.txt"),
        ("exists", "B.txt"),
        ("contains", "A.txt"),
        ("contains", "B.txt"),
    ]


def test_invalid_planner_json_does_not_invent_criteria():
    assert verify.parse_json("No criteria available") is None
    assert verify.parse_json("```json\n{broken\n```") is None
    assert verify.normalise(verify.parse_json("[]")) == []


def test_normalise_drops_unsafe_and_malformed_criteria():
    raw = {
        "criteria": [
            {"type": "exists", "path": "notes/a.md"},
            {"type": "exists", "path": "../escape"},
            {"type": "exists", "path": "/etc/passwd"},
            {"type": "contains", "path": "a.txt"},
            {"type": "file_count", "glob": "*.md"},
            {"type": "file_count", "glob": "*.md", "min": 1},
            {"type": "bogus", "text": "x"},
            {"type": "judge", "text": "Summary is readable"},
            {"type": "judge", "text": " "},
        ]
    }
    kinds = [(c["id"], c["type"]) for c in verify.normalise(raw)]
    assert kinds == [(1, "exists"), (2, "file_count"), (3, "judge")]
    assert verify.normalise("not a list") == []


def test_judge_pass_without_real_evidence_is_downgraded():
    criteria = [{"id": 1, "type": "judge", "text": "A"}, {"id": 2, "type": "judge", "text": "B"}]
    reply = json.dumps(
        {
            "results": [
                {"id": 1, "status": "pass", "evidence": "Looks great to me overall"},
                {"id": 2, "status": "pass", "evidence": "report.md has the summary"},
            ]
        }
    )
    results = verify.grade_judged(criteria, reply, "files: report.md")
    assert [r["status"] for r in results] == ["unknown", "pass"]
    assert verify.verdict(results) == "partial"
    assert verify.verdict(results[1:]) == "judged"


def test_verdict_requires_machine_evidence_to_be_verified():
    safe = {"kind": "safety", "status": "pass"}
    assert verify.verdict([safe]) == "unverified"
    assert verify.verdict([safe, {"kind": "machine", "status": "pass"}]) == "verified"
    assert verify.verdict([safe, {"kind": "machine", "status": "fail"}]) == "failed"


@pytest.fixture
async def manager(tmp_path):
    source = tmp_path / "project"
    (source / "inbox").mkdir(parents=True)
    (source / "inbox" / "a.txt").write_text("alpha")
    (source / "inbox" / "b.txt").write_text("beta")
    work = WorkManager(tmp_path / "state", "http://unused")
    project = work.workspace({"root": str(source), "model": "local"})
    await work.start()
    yield work, project, source
    await work.shutdown()


def script(work, monkeypatch, agent, criteria, judge=None):
    asked = []

    async def generate(job):
        return agent.pop(0)

    async def ask(job, system, user, max_tokens=1500):
        asked.append(system)
        return json.dumps(criteria) if system == verify.PLAN_PROMPT else judge or "{}"

    monkeypatch.setattr(work, "generate", generate)
    monkeypatch.setattr(work, "ask", ask)
    return asked


async def test_reorganising_without_loss_is_verified_by_code(manager, monkeypatch):
    work, project, source = manager
    script(
        work,
        monkeypatch,
        [
            tool("write_file", {"path": "sorted/a.txt", "content": "alpha"}),
            tool("write_file", {"path": "sorted/b.txt", "content": "beta"}),
            {"role": "assistant", "content": "Copied both."},
        ],
        {
            "criteria": [
                {"type": "exists", "path": "sorted/a.txt"},
                {"type": "file_count", "glob": "sorted/*.txt", "min": 2, "max": 2},
            ]
        },
    )
    job = await work.create_job(project["id"], "Sort the inbox")
    done = await wait_job(work, job["id"], "completed")
    report = done["verification"]
    assert report["verdict"] == "verified"
    assert report["counts"]["machine"] == 2
    assert all(r["status"] == "pass" for r in report["results"])
    assert (source / "inbox" / "a.txt").exists()


async def test_failed_verification_feeds_back_and_agent_repairs(manager, monkeypatch):
    work, project, _ = manager
    script(
        work,
        monkeypatch,
        [
            tool("write_file", {"path": "sorted/a.txt", "content": "alpha"}),
            {"role": "assistant", "content": "All done."},
            tool("write_file", {"path": "sorted/b.txt", "content": "beta"}),
            {"role": "assistant", "content": "Now both are there."},
        ],
        {"criteria": [{"type": "exists", "path": "sorted/b.txt"}]},
    )
    job = await work.create_job(project["id"], "Sort the inbox")
    done = await wait_job(work, job["id"], "completed")
    assert done["repairs"] == 1
    assert done["verification"]["verdict"] == "verified"
    feedback = [
        m for m in done["messages"] if m["role"] == "user" and "independent check" in m["content"]
    ]
    assert len(feedback) == 1 and "sorted/b.txt" in feedback[0]["content"]


async def test_deleting_unplanned_content_fails_even_without_criteria(manager, monkeypatch):
    work, project, source = manager
    script(
        work,
        monkeypatch,
        [
            tool("write_file", {"path": "inbox/a.txt", "content": ""}),
            {"role": "assistant", "content": "Cleaned."},
        ]
        * 3,
        {"criteria": []},
    )
    job = await work.create_job(project["id"], "Tidy a.txt")
    done = await wait_job(work, job["id"], "completed")
    failed = [r["id"] for r in done["verification"]["results"] if r["status"] == "fail"]
    assert failed == ["no-emptied"]
    assert done["verification"]["verdict"] == "failed"
    assert done["repairs"] == 2
    assert (source / "inbox" / "a.txt").read_text() == "alpha"


async def test_unavailable_reviewer_does_not_hide_the_job(manager, monkeypatch):
    work, project, _ = manager
    script(
        work,
        monkeypatch,
        [{"role": "assistant", "content": "Nothing to do."}],
        {"criteria": [{"type": "judge", "text": "b.txt was handled"}]},
    )

    async def broken(job, system, user, max_tokens=1500):
        if system == verify.PLAN_PROMPT:
            return json.dumps({"criteria": [{"type": "judge", "text": "b.txt was handled"}]})
        raise ValueError("model offline")

    monkeypatch.setattr(work, "ask", broken)
    job = await work.create_job(project["id"], "Check")
    done = await wait_job(work, job["id"], "completed")
    judged = [r for r in done["verification"]["results"] if r["kind"] == "judge"]
    assert [r["status"] for r in judged] == ["unknown"]
    assert done["verification"]["verdict"] == "partial"


async def test_restart_during_verification_never_claims_verified(manager, monkeypatch):
    work, project, _ = manager
    started = asyncio.Event()

    async def generate(job):
        return {"role": "assistant", "content": "Done."}

    async def ask(job, system, user, max_tokens=1500):
        if system == verify.PLAN_PROMPT:
            return json.dumps({"criteria": [{"type": "judge", "text": "ok"}]})
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(work, "generate", generate)
    monkeypatch.setattr(work, "ask", ask)
    job = await work.create_job(project["id"], "Slow check")
    await asyncio.wait_for(started.wait(), 3)
    await work.shutdown()
    restarted = WorkManager(work.store.root, "http://unused")
    await restarted.start()
    try:
        saved = restarted.store.get("job", job["id"])
        assert saved["state"] == "completed"
        assert saved["verification"]["verdict"] == "unverified"
    finally:
        await restarted.shutdown()


@pytest.mark.parametrize("value", ["/etc/passwd", r"\etc\passwd", r"C:outside.txt", r"C:\outside.txt", r"\\server\share\file"])
def test_windows_rooted_criteria_cannot_escape_the_project(monkeypatch, value):
    from pathlib import PureWindowsPath

    monkeypatch.setattr(verify, "Path", PureWindowsPath)
    assert verify.normalise([{"type": "exists", "path": value}]) == []
