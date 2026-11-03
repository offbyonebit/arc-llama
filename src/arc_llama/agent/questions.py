"""Questions a model puts to the user, with choices to click instead of a reply to type.

OpenCode's ``question`` tool and the built-in agent's ``ask_user`` share one shape, so the Work view
draws both the same way."""
from __future__ import annotations

from typing import Any

ASK_USER = "ask_user"


def clean_questions(raw: Any) -> list[dict[str, Any]]:
    """The questions of an OpenCode question request, in the shape the Work view draws.

    Each is {"question", "header", "options": [{"label", "description"}], "multiple", "custom"}; anything
    malformed is dropped so a confused model cannot break the page."""
    questions: list[dict[str, Any]] = []
    for q in raw if isinstance(raw, list) else []:
        if not isinstance(q, dict) or not isinstance(q.get("question"), str) or not q["question"].strip():
            continue
        options = []
        raw_options = q.get("options")
        for o in raw_options if isinstance(raw_options, list) else []:
            if isinstance(o, str) and o.strip():
                options.append({"label": o.strip()[:120], "description": ""})
            elif isinstance(o, dict) and isinstance(o.get("label"), str) and o["label"].strip():
                desc = o.get("description")
                options.append({"label": o["label"].strip()[:120], "description": desc.strip()[:400] if isinstance(desc, str) else ""})
        header = q.get("header")
        questions.append({
            "question": q["question"].strip()[:2000],
            "header": header.strip()[:30] if isinstance(header, str) else "",
            "options": options[:12],
            "multiple": q.get("multiple") is True,
            "custom": q.get("custom") is not False or not options,
        })
    return questions[:8]


def format_answers(questions: list[dict[str, Any]], answers: list[list[str]] | None) -> str:
    """What the model is told after the user answered (or dismissed) its questions."""
    if answers is None:
        return "The user dismissed the question without answering. Continue with your best judgement or stop and explain."
    lines = []
    for i, q in enumerate(questions):
        picked = answers[i] if i < len(answers) else []
        lines.append(f"{q['question']}\nAnswer: {', '.join(picked) if picked else '(no answer)'}")
    return "The user answered:\n\n" + "\n\n".join(lines)


def clean_answers(raw: Any, count: int) -> list[list[str]] | None:
    """Validate answers from the browser: one list of strings per question. None means invalid."""
    if not isinstance(raw, list) or len(raw) != count:
        return None
    out: list[list[str]] = []
    for item in raw:
        if not isinstance(item, list) or not all(isinstance(v, str) for v in item) or len(item) > 20:
            return None
        out.append([v.strip()[:2000] for v in item if v.strip()])
    return out
