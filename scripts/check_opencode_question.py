"""Prove the question round trip against the real OpenCode installed on this machine.

A fake model calls OpenCode's ``question`` tool; the check answers through ``question_callback`` and
fails unless the question reached arc-llama in the shape the Work view draws and the answer came back
to the model as the tool's result. Needs OpenCode, so it is not part of the unit tests; run it with
`scripts/check.sh --live`."""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from arc_llama.agent.harness.opencode import find_binary, run_opencode

QUESTION = {"question": "Which database should I use?", "header": "Database",
            "options": [{"label": "SQLite", "description": "One file"}, {"label": "Postgres", "description": "A server"}]}
seen: dict[str, Any] = {"asked": False, "tool_result": None, "offered": set()}


class Model(BaseHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if not self.path.endswith("/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return
        names = {t["function"]["name"] for t in body.get("tools") or [] if isinstance(t, dict) and "function" in t}
        seen["offered"] |= names
        results = [m for m in body.get("messages") or [] if m.get("role") == "tool"]
        if results:
            seen["tool_result"] = results[-1].get("content")
            delta: dict[str, Any] = {"role": "assistant", "content": "Using Postgres."}
        elif "question" in names and not seen["asked"]:
            seen["asked"] = True
            delta = {"role": "assistant", "tool_calls": [{"index": 0, "id": "call_q", "type": "function",
                     "function": {"name": "question", "arguments": json.dumps({"questions": [QUESTION]})}}]}
        else:
            delta = {"role": "assistant", "content": "ok"}
        finish = "tool_calls" if "tool_calls" in delta else "stop"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in ({"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                      {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                       "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}):
            self.wfile.write(f"data: {json.dumps({'id': 'q', 'object': 'chat.completion.chunk', 'model': body.get('model'), **chunk})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


async def main() -> int:
    if find_binary() is None:
        print("opencode not found: skipping")
        return 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    asked: list[list[dict[str, Any]]] = []

    async def answer(_question_id: str, questions: list[dict[str, Any]]) -> list[list[str]]:
        asked.append(questions)
        return [["Postgres"]]

    problems: list[str] = []
    events: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as root:
        async for event in run_opencode("Set up storage", "m", f"http://127.0.0.1:{server.server_address[1]}", Path(root),
                                        auto_confirm=True, question_callback=answer, timeout=120):
            events.append(event)
        # Without a callback nobody would see the question, so the tool must stay off.
        offered_with = set(seen["offered"])
        seen["offered"] = set()
        async for _ in run_opencode("Say ok", "m", f"http://127.0.0.1:{server.server_address[1]}", Path(root),
                                    auto_confirm=True, timeout=120):
            pass
        if "question" in seen["offered"]:
            problems.append("OpenCode offered the question tool with nobody to answer it")
        seen["offered"] = offered_with
    server.shutdown()
    kinds = [e["type"] for e in events]
    if "question" not in seen["offered"]:
        problems.append("OpenCode did not offer the question tool to the model")
    if not asked or asked[0][0]["question"] != QUESTION["question"] or [o["label"] for o in asked[0][0]["options"]] != ["SQLite", "Postgres"]:
        problems.append(f"the question did not reach arc-llama intact: {asked}")
    if not any(e["type"] == "question_required" and e.get("call_id") for e in events):
        problems.append("question_required carried no call id to place it beside its step")
    if "Postgres" not in str(seen["tool_result"] or ""):
        problems.append(f"the answer did not come back to the model: {seen['tool_result']!r}")
    if kinds[-1:] != ["done"] or "error" in kinds:
        problems.append(f"the run did not finish cleanly: {[e for e in events if e['type'] == 'error'] or kinds[-3:]}")
    for message in problems:
        print("PROBLEM:", message)
    if not problems:
        print(f"Question round trip works; the model was told: {seen['tool_result']!r}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
