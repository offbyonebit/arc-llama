"""Sequential attempts: retry on a fresh copy, keep the best verified result.

Attempts never overlap, so this works with one resident model on one GPU. The time budget
decides whether another attempt may start; it never interrupts one that is running.
"""

from __future__ import annotations

from typing import Any

MAX_ATTEMPTS = 5
MAX_BUDGET_MINUTES = 720
RANK = {"verified": 4, "judged": 3, "partial": 2, "unverified": 1, "failed": 0}


def settings(workspace: dict[str, Any]) -> tuple[int, int]:
    attempts = workspace.get("max_attempts", 1)
    budget = workspace.get("time_budget_minutes", 30)
    if not workspace.get("verify", True):
        attempts = 1
    return attempts, budget


def score(entry: dict[str, Any]) -> tuple[int, int, int, int]:
    """Higher is better; between equal results prefer the complete and then the earlier attempt."""
    return (
        0 if entry.get("incomplete") else 1,
        RANK.get(entry["verdict"], 0),
        entry.get("passed", 0),
        -entry["n"],
    )


def best(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return max(entries, key=score)


def may_start_another(job: dict[str, Any], entry: dict[str, Any], now: float) -> bool:
    attempts, budget = settings(job["workspace"])
    done = job["attempts"]
    if len(done) >= attempts or entry["verdict"] == "verified" and not entry.get("incomplete"):
        return False
    elapsed = now - job["started_at"]
    average = sum(e["seconds"] for e in done) / len(done)
    return elapsed + average <= budget * 60


def entry_for(
    job: dict[str, Any], report: dict[str, Any], now: float, incomplete: bool
) -> dict[str, Any]:
    results = report.get("results", [])
    return {
        "n": job["attempt"],
        "verdict": report["verdict"],
        "label": report["label"],
        "reason": report.get("reason", ""),
        "passed": sum(r["status"] == "pass" for r in results),
        "failed": [r["text"] for r in results if r["status"] == "fail"][:6],
        "turns": job["turns"],
        "repairs": job.get("repairs", 0),
        "seconds": max(0.0, now - job["attempt_started_at"]),
        "incomplete": incomplete,
    }
