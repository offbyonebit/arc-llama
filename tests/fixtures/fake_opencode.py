#!/usr/bin/env python3
"""Minimal stand-in for `opencode serve`, for adapter tests.

Replays tests/fixtures/opencode_events_bash.json when a prompt arrives. With
FAKE_OPENCODE_PERMISSION=1 it first raises a permission request and only
continues after a reply, which it records in FAKE_OPENCODE_LOG."""
import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

EVENTS = json.loads((Path(__file__).parent / "opencode_events_bash.json").read_text())
SID = next(e["properties"]["sessionID"] for e in EVENTS if e["type"] == "session.created")
mode = os.environ.get("FAKE_OPENCODE_PERMISSION")
want_permission = mode in ("1", "event")
list_endpoint_broken = mode == "event"        # mirrors OpenCode 1.19.0: GET /permission answers 400 for a web fetch
log_path = os.environ.get("FAKE_OPENCODE_LOG")
subscribers: list[queue.Queue] = []
state = {"permission": None, "replied": threading.Event(), "seen_env": {}}


def log(entry):
    if log_path:
        with open(log_path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")


def replay():
    if want_permission:
        state["permission"] = {"id": "per_1", "sessionID": SID, "permission": "bash",
                               "patterns": ["rm -rf build"], "metadata": {}, "always": []}
        if list_endpoint_broken:
            for q in list(subscribers):
                q.put({"type": "permission.asked", "properties": state["permission"]})
        state["replied"].wait(10)
    for e in EVENTS:
        if e["type"] in ("session.created",):
            continue
        for q in list(subscribers):
            q.put(e)
        time.sleep(0.01)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/event"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            q = queue.Queue()
            subscribers.append(q)
            self.wfile.write(b'data: {"type":"server.connected","properties":{}}\n\n')
            self.wfile.flush()
            try:
                while True:
                    self.wfile.write(f"data: {json.dumps(q.get())}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
        if self.path.startswith("/command"):
            return self._json([
                {"name": "review", "description": "review changes", "source": "command", "template": "x", "hints": ["$ARGUMENTS"]},
                {"name": "quota", "description": "Show quota", "source": "command", "template": "/quota", "hints": []},
                {"name": "pdf", "description": "PDF skill", "source": "skill", "template": "x", "hints": []},
            ])
        if self.path.startswith("/permission"):
            if list_endpoint_broken and state["permission"] and not state["replied"].is_set():
                return self._json({"name": "BadRequest", "data": {"message": "Expected JSON value, got undefined"}}, 400)
            return self._json([state["permission"]] if state["permission"] and not state["replied"].is_set() else [])
        self._json({"error": "not found"}, 404)

    def do_DELETE(self):
        log({"path": self.path.split("?")[0], "method": "DELETE"})
        self._json(True)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        log({"path": self.path.split("?")[0], "body": body, "auth": bool(self.headers.get("Authorization"))})
        if self.path == "/__exit":                      # test-only: stop without signals
            self._json({"ok": True})
            threading.Timer(0.05, lambda: os._exit(0)).start()
            return
        if self.path.startswith("/session") and self.path.split("?")[0] == "/session":
            return self._json({"id": SID})
        if self.path.split("?")[0].endswith("/command"):
            if os.environ.get("FAKE_OPENCODE_COMMAND_FAILS"):
                return self._json({"error": "nope"}, 400)
            threading.Thread(target=replay, daemon=True).start()
            return self._json({})
        if "/prompt_async" in self.path and os.environ.get("FAKE_OPENCODE_SEND_TOOLS"):
            # Like the real thing, make one chat request to the configured endpoint that carries the tool list.
            import urllib.request
            cfg = json.loads(os.environ["OPENCODE_CONFIG_CONTENT"])
            base = cfg["provider"]["arc-llama"]["options"]["baseURL"]
            tools = [{"type": "function", "function": {"name": n, "parameters": {}}} for n in os.environ["FAKE_OPENCODE_SEND_TOOLS"].split(",")]
            data = json.dumps({"model": "m", "stream": True, "tools": tools, "messages": []}).encode()
            def call():
                with urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", data=data, headers={"Content-Type": "application/json"}), timeout=10) as r:
                    r.read()
            threading.Thread(target=call, daemon=True).start()
            return self._json({})
        if "/prompt_async" in self.path:
            threading.Thread(target=replay, daemon=True).start()
            return self._json({})
        if "/permission/" in self.path:
            state["replied"].set()
            return self._json(True)
        self._json({})


def main():
    log({"argv": sys.argv[1:], "env": {k: os.environ.get(k) for k in (
        "OPENCODE_CONFIG_CONTENT", "OPENCODE_DISABLE_AUTOUPDATE", "OPENCODE_SERVER_PASSWORD")}, "cwd": os.getcwd()})
    threading.Timer(90, lambda: os._exit(0)).start()   # never outlive a crashed test run
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    print(f"opencode server listening on http://127.0.0.1:{srv.server_address[1]}", flush=True)
    srv.serve_forever()


main()
