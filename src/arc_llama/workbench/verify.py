"""Acceptance criteria written before work starts and graded after it ends.

Most criteria are machine-checkable (a file exists, text appears, nothing was lost) and are
graded by code, so a weak model cannot talk its way past them. Only criteria that need
judgement go to a model, and a verdict that rests on judgement alone is labelled as such.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any

from arc_llama.workbench import files

MAX_CRITERIA = 8
MACHINE = {"exists", "not_exists", "contains", "not_contains", "file_count"}
POLICY = {"may_delete"}
KINDS = MACHINE | POLICY | {"judge"}

PLAN_PROMPT = """You write acceptance criteria for a task that an AI agent will perform on a folder of files.
Reply with JSON only: {"criteria": [...]}. Write at most 8 criteria that a stranger could check afterwards.
Each criterion is an object with a "type" and a short "text" (plain words for the user):
- {"type":"exists","path":"relative/file"}: the file must exist afterwards.
- {"type":"not_exists","path":"relative/file"}: the file must be gone afterwards.
- {"type":"contains","path":"relative/file","value":"text"}: the file must contain this text.
- {"type":"not_contains","path":"relative/file","value":"text"}: the file must not contain this text.
- {"type":"file_count","glob":"docs/*.md","min":1,"max":50}: how many files match the pattern (min and max optional).
- {"type":"may_delete","glob":"*.tmp"}: files matching this pattern may legitimately disappear. Every file that
  disappears without being moved or edited fails the check unless you list it here, so declare deletions the task requires.
- {"type":"judge","text":"..."}: only when nothing above can express it; a reviewer will check it by reading the result.
Prefer the checkable types. Paths are relative to the project folder. Do not invent file contents you cannot know."""

JUDGE_PROMPT = """You are an independent reviewer. You did not do this work and you must not assume it was done well.
For each numbered criterion decide "pass", "fail" or "unknown". Pass only when you can point to specific evidence in the
material below: quote or name a real file path. If the material does not let you tell, answer "unknown".
Reply with JSON only: {"results":[{"id":1,"status":"pass","evidence":"short quote or path"}]}."""


def parse_json(text: str) -> Any:
    # Local models may return a bare criteria array, fenced JSON, or a short
    # preamble. Decode a whole value before normalising its schema; slicing from
    # the first to last brace loses arrays containing several criteria objects.
    decoder = json.JSONDecoder()
    for offset, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text, offset)
        except ValueError:
            continue
        return value
    return None


def _relative(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip() or len(value) > 500:
        return None
    path = Path(value.strip())
    if path.is_absolute() or ".." in path.parts or any(files.excluded(p) for p in path.parts):
        return None
    return path.as_posix()


def normalise(raw: Any) -> list[dict[str, Any]]:
    """Keep only well-formed criteria; unknown or unsafe ones become judged text or are dropped."""
    items = raw.get("criteria") if isinstance(raw, dict) else raw
    result: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or item.get("type") not in KINDS:
            continue
        kind = item["type"]
        criterion: dict[str, Any] = {"type": kind}
        text = item.get("text")
        if kind in ("exists", "not_exists", "contains", "not_contains"):
            path = _relative(item.get("path"))
            if path is None:
                continue
            criterion["path"] = path
            if kind.endswith("contains"):
                value = item.get("value")
                if not isinstance(value, str) or not value or len(value) > 2000:
                    continue
                criterion["value"] = value
        elif kind in ("file_count", "may_delete"):
            glob = item.get("glob")
            if not isinstance(glob, str) or not glob.strip() or len(glob) > 300:
                continue
            if _relative(glob.replace("*", "x").replace("?", "x")) is None:
                continue
            criterion["glob"] = glob.strip()
            if kind == "file_count":
                bounds = {k: item[k] for k in ("min", "max") if type(item.get(k)) is int}
                if not bounds or any(v < 0 for v in bounds.values()):
                    continue
                criterion.update(bounds)
        elif not isinstance(text, str) or not text.strip():
            continue
        criterion["text"] = (
            text if isinstance(text, str) and text.strip() else describe(criterion)
        )[:300]
        result.append({"id": len(result) + 1, **criterion})
        if len(result) >= MAX_CRITERIA:
            break
    return result


def describe(criterion: dict[str, Any]) -> str:
    kind = criterion["type"]
    if kind == "exists":
        return f"{criterion['path']} exists"
    if kind == "not_exists":
        return f"{criterion['path']} is gone"
    if kind == "contains":
        return f"{criterion['path']} contains the expected text"
    if kind == "not_contains":
        return f"{criterion['path']} no longer contains the unwanted text"
    if kind == "file_count":
        return f"The number of files matching {criterion['glob']} is in range"
    if kind == "may_delete":
        return f"Files matching {criterion['glob']} may be deleted"
    return criterion.get("text", "")


def _read(tree: Path, relative: str) -> str | None:
    try:
        target = files.safe_path(tree, relative)
        if not target.is_file() or target.stat().st_size > files.MAX_FILE:
            return None
        return target.read_text(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        return None


def _result(criterion: dict[str, Any], status: str, evidence: str, kind: str) -> dict[str, Any]:
    return {
        "id": criterion["id"],
        "kind": kind,
        "text": criterion["text"],
        "status": status,
        "evidence": evidence[:500],
    }


def grade_machine(criterion: dict[str, Any], after: dict[str, str], tree: Path) -> dict[str, Any]:
    kind = criterion["type"]
    if kind in ("exists", "not_exists"):
        present = criterion["path"] in after
        ok = present if kind == "exists" else not present
        return _result(
            criterion,
            "pass" if ok else "fail",
            f"{criterion['path']} is {'present' if present else 'absent'} in the result",
            "machine",
        )
    if kind in ("contains", "not_contains"):
        content = _read(tree, criterion["path"])
        if content is None:
            status = "fail" if kind == "contains" else "pass"
            return _result(
                criterion, status, f"{criterion['path']} is missing or unreadable", "machine"
            )
        found = criterion["value"] in content
        ok = found if kind == "contains" else not found
        return _result(
            criterion,
            "pass" if ok else "fail",
            f"text {'found' if found else 'not found'} in {criterion['path']}",
            "machine",
        )
    count = sum(fnmatch.fnmatch(name, criterion["glob"]) for name in after)
    low, high = criterion.get("min", 0), criterion.get("max")
    ok = count >= low and (high is None or count <= high)
    return _result(
        criterion, "pass" if ok else "fail", f"{count} file(s) match {criterion['glob']}", "machine"
    )


def safety_checks(
    criteria: list[dict[str, Any]],
    before: dict[str, str],
    after: dict[str, str],
    base: Path,
    tree: Path,
) -> list[dict[str, Any]]:
    """Checks that always run, whatever the plan said."""
    allowed = [c["glob"] for c in criteria if c["type"] == "may_delete"]
    surviving = set(after.values())
    lost = sorted(
        name
        for name, digest in before.items()
        if name not in after
        and digest not in surviving
        and not any(fnmatch.fnmatch(name, g) for g in allowed)
    )
    checks = [
        {
            "id": "no-loss",
            "kind": "safety",
            "text": "No file was deleted unless the plan allowed it (moved and edited files do not count)",
            "status": "fail" if lost else "pass",
            "evidence": (
                f"{len(lost)} file(s) vanished: "
                + ", ".join(lost[:8])
                + (" ..." if len(lost) > 8 else "")
                if lost
                else "Every original file is still present, edited, or moved"
            ),
        }
    ]
    emptied = sorted(
        name
        for name, digest in after.items()
        if digest != before.get(name)
        and name in before
        and (tree / name).stat().st_size == 0
        and (base / name).stat().st_size > 0
    )
    checks.append(
        {
            "id": "no-emptied",
            "kind": "safety",
            "text": "No file with content was emptied",
            "status": "fail" if emptied else "pass",
            "evidence": ", ".join(emptied[:8]) if emptied else "No file was truncated to nothing",
        }
    )
    return checks


def grade_judged(
    criteria: list[dict[str, Any]], reply: str | None, evidence_pool: str
) -> list[dict[str, Any]]:
    """Grade judge criteria from a reviewer reply, distrusting any pass it cannot support."""
    parsed = parse_json(reply or "")
    answers = {}
    for item in (parsed or {}).get("results", []) if isinstance(parsed, dict) else []:
        if isinstance(item, dict) and type(item.get("id")) is int:
            answers[item["id"]] = item
    results = []
    for criterion in criteria:
        answer = answers.get(criterion["id"], {})
        status = str(answer.get("status"))
        if status not in ("pass", "fail"):
            status = "unknown"
        evidence = answer.get("evidence")
        evidence = evidence if isinstance(evidence, str) else ""
        if status == "pass" and (
            len(evidence.strip()) < 8 or not _supported(evidence, evidence_pool)
        ):
            status, evidence = (
                "unknown",
                "The reviewer's pass cited nothing that exists in the result",
            )
        results.append(_result(criterion, status, evidence or "No evidence given", "judge"))
    return results


def _supported(evidence: str, pool: str) -> bool:
    quoted = [
        part for part in evidence.replace("`", "'").replace('"', "'").split("'") if len(part) >= 4
    ]
    tokens = [token.strip(".,;:()[]") for token in evidence.split() if "/" in token or "." in token]
    return any(piece in pool for piece in quoted + tokens if len(piece) >= 4)


def verdict(results: list[dict[str, Any]]) -> str:
    if any(r["status"] == "fail" for r in results):
        return "failed"
    graded = [r for r in results if r["kind"] != "safety"]
    if not graded:
        return "unverified"
    if any(r["status"] == "unknown" for r in graded):
        return "partial"
    if any(r["kind"] == "machine" for r in graded):
        return "verified"
    return "judged"


LABELS = {
    "verified": "Verified: checked by code, all criteria passed",
    "judged": "Judged only: a reviewer model passed it, nothing was checked by code",
    "partial": "Partly verified: some criteria could not be confirmed",
    "failed": "Failed verification",
    "unverified": "Not verified: no checkable criteria",
}


def failures(results: list[dict[str, Any]]) -> str:
    return "\n".join(f"- {r['text']}: {r['evidence']}" for r in results if r["status"] == "fail")
