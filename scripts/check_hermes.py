"""Exercise the installed Hermes ACP with fake local inference, never a model.

Checks real ACP startup, stream translation, token accounting, file approval and
denial, and cancellation. Run with scripts/check.sh --live.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from arc_llama.agent.harness.hermes import find_hermes_binary, run_hermes


class ProbeAPI(BaseHTTPRequestHandler):
    calls: list[dict[str, Any]] = []

    def log_message(self, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        self.calls.append({"path": self.path, "model": body.get("model")})
        if self.path == "/v1/chat/budget":
            data = json.dumps(
                {"prompt_tokens": 3200, "context_tokens": 8192, "exact": True}
            ).encode()
            content_type = "application/json"
        elif self.path == "/v1/chat/completions":
            has_result = any(m.get("role") == "tool" for m in body.get("messages", []))
            if has_result:
                delta = {"role": "assistant", "content": "Hermes completed the approval probe."}
                finish = "stop"
            else:
                delta = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "probe-write",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": json.dumps(
                                    {"path": "approved.txt", "content": "Hermes approval probe\n"}
                                ),
                            },
                        }
                    ],
                }
                finish = "tool_calls"
            frames = [
                {
                    "id": "probe",
                    "object": "chat.completion.chunk",
                    "model": "arc-hermes-probe",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                },
                {
                    "id": "probe",
                    "object": "chat.completion.chunk",
                    "model": "arc-hermes-probe",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 3200, "completion_tokens": 12, "total_tokens": 3212},
                },
            ]
            data = (
                "".join("data: " + json.dumps(frame) + "\n\n" for frame in frames)
                + "data: [DONE]\n\n"
            ).encode()
            content_type = "text/event-stream"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


async def main() -> int:
    if find_hermes_binary() is None:
        print("Hermes not installed: skipping real ACP check")
        return 0
    api = ThreadingHTTPServer(("127.0.0.1", 0), ProbeAPI)
    thread = threading.Thread(target=api.serve_forever, daemon=True)
    thread.start()
    try:
        for approved in (False, True):
            asked = []

            async def confirm(*args: Any, collected=asked, allowed=approved) -> bool:
                collected.append(args)
                return allowed

            with tempfile.TemporaryDirectory(prefix="arc-hermes-probe-project-") as folder:
                async with asyncio.timeout(90):
                    events = [
                        e
                        async for e in run_hermes(
                            task="Write approved.txt in the project folder.",
                            model="arc-hermes-probe",
                            base_url=f"http://127.0.0.1:{api.server_port}",
                            root=Path(folder),
                            confirm_callback=confirm,
                        )
                    ]
                errors = [e for e in events if e["type"] == "error"]
                assert not errors, errors
                assert len(asked) == 1 and asked[0][1] == "write_file", asked
                assert (Path(folder) / "approved.txt").exists() is approved
                assert any(e["type"] == "context" and e["prompt_tokens"] == 3200 for e in events)
                assert any(e["type"] == "assistant_delta" for e in events)
                assert events[-1]["type"] == "done"
                print(
                    "Hermes real ACP: edit "
                    + ("approved" if approved else "denied")
                    + ", streaming and context counts passed",
                    flush=True,
                )
        with tempfile.TemporaryDirectory(prefix="arc-hermes-cancel-project-") as folder:
            reached = asyncio.Event()

            async def wait_for_user(*args: Any) -> bool:
                reached.set()
                await asyncio.Event().wait()
                return False

            async def consume() -> None:
                async for _ in run_hermes(
                    task="Write approved.txt.",
                    model="arc-hermes-probe",
                    base_url=f"http://127.0.0.1:{api.server_port}",
                    root=Path(folder),
                    confirm_callback=wait_for_user,
                ):
                    pass

            task = asyncio.create_task(consume())
            try:
                await asyncio.wait_for(reached.wait(), 60)
            finally:
                task.cancel()
                try:
                    await asyncio.wait_for(task, 10)
                except asyncio.CancelledError:
                    pass
            assert not (Path(folder) / "approved.txt").exists()
            print("Hermes real ACP: cancelled approval stopped without writing", flush=True)
        assert ProbeAPI.calls and all(c["model"] == "arc-hermes-probe" for c in ProbeAPI.calls)
        return 0
    finally:
        api.shutdown()
        api.server_close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
