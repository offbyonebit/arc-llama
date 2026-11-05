#!/usr/bin/env python3
"""Minimal stand-in for OpenCode 2's `opencode serve` (the /api surface), for adapter tests.

The event shapes come from a real OpenCode 2.2.5 run against a scripted model with synthetic data.
A prompt replays a shell tool call and an answer. The switches match fake_opencode.py:
FAKE_OPENCODE_PERMISSION ("1": only through the list endpoint, "event": only on the stream) raises a
permission request first; FAKE_OPENCODE_QUESTION does the same with a question form
(FAKE_OPENCODE_QUESTION_MULTI adds a multi-select question). FAKE_OPENCODE_LOG records requests."""
import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

SID = "ses_fake2"
mode = os.environ.get("FAKE_OPENCODE_PERMISSION")
question_mode = os.environ.get("FAKE_OPENCODE_QUESTION")
log_path = os.environ.get("FAKE_OPENCODE_LOG")
subscribers: list[queue.Queue] = []
state = {"permission": None, "replied": threading.Event(), "form": None, "answered": threading.Event(), "command_lists": 0}


def log(entry):
    if log_path:
        with open(log_path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")


def publish(kind, data):
    event = {"id": f"evt_{time.time_ns()}", "created": int(time.time() * 1000), "type": kind, "data": data}
    for q in list(subscribers):
        q.put(event)
    time.sleep(0.01)


def replay():
    publish("session.execution.started", {"sessionID": SID})
    publish("session.step.started", {"sessionID": SID, "assistantMessageID": "msg_a", "agent": "build"})
    publish("session.text.delta", {"sessionID": "ses_other", "assistantMessageID": "msg_x", "ordinal": 0, "delta": "not ours"})
    publish("session.reasoning.started", {"sessionID": SID, "assistantMessageID": "msg_a", "ordinal": 0})
    publish("session.reasoning.delta", {"sessionID": SID, "assistantMessageID": "msg_a", "ordinal": 0, "delta": "Thinking."})
    publish("session.reasoning.ended", {"sessionID": SID, "assistantMessageID": "msg_a", "ordinal": 0, "text": "Thinking."})
    publish("session.tool.input.started", {"sessionID": SID, "assistantMessageID": "msg_a", "id": "call_1", "name": "shell"})
    publish("session.tool.called", {"sessionID": SID, "assistantMessageID": "msg_a", "id": "call_1",
                                    "input": {"command": "rm -rf build"}, "executed": False})
    if mode in ("1", "event"):
        state["permission"] = {"id": "per_1", "sessionID": SID, "action": "shell", "resources": ["rm -rf build"],
                               "save": ["rm *"], "source": {"type": "tool", "messageID": "msg_a", "id": "call_1"}}
        if mode == "event":
            publish("permission.asked", state["permission"])
        state["replied"].wait(10)
    if question_mode:
        fields = [{"key": "q0", "title": "Database", "description": "Which database?", "type": "string", "custom": True,
                   "options": [{"value": "SQLite", "label": "SQLite", "description": "One file"},
                               {"value": "Postgres", "label": "Postgres", "description": "A server"}]}]
        if os.environ.get("FAKE_OPENCODE_QUESTION_MULTI"):
            fields.append({"key": "q1", "title": "Extras", "description": "Which extras?", "type": "multiselect", "custom": True,
                           "options": [{"value": "Cache", "label": "Cache", "description": "c"},
                                       {"value": "Search", "label": "Search", "description": "s"}]})
        state["form"] = {"id": "frm_1", "sessionID": SID, "title": "Questions",
                         "metadata": {"kind": "question", "tool": {"messageID": "msg_a", "id": "call_q"}}, "fields": fields}
        if question_mode == "event":
            publish("form.created", {"form": state["form"]})
        state["answered"].wait(10)
    publish("session.tool.success", {"sessionID": SID, "assistantMessageID": "msg_a", "id": "call_1",
                                     "content": [{"type": "text", "text": "(no output)"}], "metadata": {"exit": 0}})
    publish("session.step.ended", {"sessionID": SID, "assistantMessageID": "msg_a", "finish": "tool-calls",
                                   "tokens": {"input": 100, "output": 7, "reasoning": 0, "cache": {"read": 20, "write": 0}}})
    publish("session.retry.scheduled", {"sessionID": SID, "assistantMessageID": "msg_b", "attempt": 2,
                                        "error": {"type": "provider.transport", "message": "ECONNRESET"}})
    publish("session.step.started", {"sessionID": SID, "assistantMessageID": "msg_b", "agent": "build"})
    publish("session.text.started", {"sessionID": SID, "assistantMessageID": "msg_b", "ordinal": 0})
    for delta in ("Done. ", "All good."):
        publish("session.text.delta", {"sessionID": SID, "assistantMessageID": "msg_b", "ordinal": 0, "delta": delta})
    publish("session.text.ended", {"sessionID": SID, "assistantMessageID": "msg_b", "ordinal": 0, "text": "Done. All good."})
    publish("session.step.ended", {"sessionID": SID, "assistantMessageID": "msg_b", "finish": "stop",
                                   "tokens": {"input": 130, "output": 4, "reasoning": 0, "cache": {"read": 0, "write": 0}}})
    if os.environ.get("FAKE_OPENCODE_FAIL"):
        publish("session.execution.failed", {"sessionID": SID, "error": {"type": "provider.invalid-request", "message": "context length exceeded"}})
    else:
        publish("session.execution.succeeded", {"sessionID": SID})


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

    def _empty(self):
        self.send_response(204)
        self.end_headers()

    def _location(self):
        return parse_qs(urlparse(self.path).query).get("location[directory]", [None])[0]

    def do_GET(self):
        path = urlparse(self.path).path
        log({"path": path, "method": "GET", "location": self._location()})
        if path == "/api/info":
            return self._json({"version": "2.2.5", "pid": os.getpid()})
        if path == "/api/event":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            q = queue.Queue()
            subscribers.append(q)
            self.wfile.write(b'event: server.connected\ndata: {"type":"server.connected","data":{}}\n\n')
            self.wfile.flush()
            try:
                while True:
                    event = q.get()
                    self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
        if path == "/api/command":
            # Like the real thing, the list is empty until the server has loaded its commands.
            state["command_lists"] += 1
            data = [] if state["command_lists"] == 1 else [{"name": "review", "description": "review changes"}]
            return self._json({"location": {"directory": self._location()}, "data": data})
        if path == "/api/skill":
            return self._json({"location": {"directory": self._location()}, "data": [
                {"id": "pdf", "name": "PDF", "description": "PDF skill", "path": "/x/SKILL.md", "content": "x"}]})
        if path == "/api/provider":
            # Policies from the user's own config can remove the arc-llama provider before Work sees it.
            ids = ["openai"] if os.environ.get("FAKE_OPENCODE_PROVIDER_BLOCKED") else ["arc-llama", "openai"]
            return self._json({"location": {"directory": self._location()}, "data": [{"id": i, "name": i} for i in ids]})
        if path == "/api/permission/saved":
            return self._json({"data": json.loads(os.environ.get("FAKE_OPENCODE_SAVED") or "[]")})
        if path == f"/api/session/{SID}/permission":
            if mode == "event":
                return self._json({"_tag": "InvalidRequestError", "message": "nope"}, 400)
            pending = state["permission"] if mode == "1" and state["permission"] and not state["replied"].is_set() else None
            return self._json({"data": [pending] if pending else []})
        if path == f"/api/session/{SID}/form":
            pending = state["form"] if question_mode == "1" and state["form"] and not state["answered"].is_set() else None
            # A malformed entry first: the poller must skip it and still see the real form.
            return self._json({"data": [{"sessionID": SID}, pending] if pending else []})
        self._json({"_tag": "NotFound"}, 404)

    def do_DELETE(self):
        path = urlparse(self.path).path
        log({"path": path, "method": "DELETE"})
        if path.startswith(f"/api/session/{SID}/form/"):
            state["answered"].set()
        self._empty()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        path = urlparse(self.path).path
        log({"path": path, "body": body, "auth": bool(self.headers.get("Authorization")), "location": self._location()})
        if path == "/__exit":                      # test-only: stop without signals
            self._json({"ok": True})
            threading.Timer(0.05, lambda: os._exit(0)).start()
            return
        if path == "/api/session":
            return self._json({"data": {"id": SID, "location": body.get("location"), "model": body.get("model")}})
        if path == f"/api/session/{SID}/command":
            if os.environ.get("FAKE_OPENCODE_COMMAND_FAILS"):
                return self._json({"_tag": "CommandExecutionError", "command": body.get("name"), "message": "nope"}, 500)
            if body.get("name") == "pdf":
                return self._json({"_tag": "CommandNotFoundError", "command": "pdf", "message": "Command not found"}, 404)
            threading.Thread(target=replay, daemon=True).start()
            return self._empty()
        if path == f"/api/session/{SID}/prompt" and os.environ.get("FAKE_OPENCODE_SEND_TOOLS"):
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
            return self._json({"data": {"id": "msg_u", "sessionID": SID, "type": "user"}})
        if path == f"/api/session/{SID}/prompt":
            threading.Thread(target=replay, daemon=True).start()
            return self._json({"data": {"id": "msg_u", "sessionID": SID, "type": "user"}})
        if path.startswith(f"/api/session/{SID}/form/"):
            state["answered"].set()
            return self._empty()
        if path.startswith(f"/api/session/{SID}/permission/"):
            state["replied"].set()
            return self._empty()
        if path == f"/api/session/{SID}/interrupt":
            return self._json({"interrupted": False})
        self._json({"_tag": "NotFound"}, 404)


def main():
    log({"argv": sys.argv[1:], "env": {k: os.environ.get(k) for k in (
        "OPENCODE_CONFIG_CONTENT", "OPENCODE_DISABLE_AUTOUPDATE", "OPENCODE_SERVER_PASSWORD",
        "OPENCODE_ENABLE_QUESTION_TOOL")}, "cwd": os.getcwd()})
    threading.Timer(90, lambda: os._exit(0)).start()   # never outlive a crashed test run
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    print(f"Server listening on http://127.0.0.1:{srv.server_address[1]}", flush=True)
    srv.serve_forever()


main()
