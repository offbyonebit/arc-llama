"""Tiny ACP peer for testing Arc Llama's Hermes adapter, never real inference."""

import json
import os
import sys
from pathlib import Path

root = Path.cwd()
scenario = (root / ".scenario").read_text() if (root / ".scenario").exists() else "success"
home = Path(os.environ["HERMES_HOME"])
config = json.loads((home / "config.yaml").read_text())
(root / "observed.json").write_text(
    json.dumps({"home": str(home), "pid": os.getpid(), "env": dict(os.environ), "config": config})
)


def send(**message):
    print(json.dumps({"jsonrpc": "2.0", **message}), flush=True)


def update(**content):
    send(method="session/update", params={"sessionId": "session-1", "update": content})


for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    request_id = message.get("id")
    if method == "initialize":
        if scenario == "startup_timeout":
            sys.stdin.read()
            sys.exit(0)
        send(
            id=request_id,
            result={"protocolVersion": 1, "agentInfo": {"name": "hermes-agent", "version": "test"}},
        )
    elif method == "session/new":
        assert message["params"]["cwd"] == str(root)
        send(id=request_id, result={"sessionId": "session-1"})
    elif method == "session/set_mode":
        assert message["params"]["modeId"] == "default"
        send(id=request_id, result={})
    elif method == "session/prompt":
        if scenario == "malformed":
            print("{broken", flush=True)
            break
        if scenario == "error":
            send(id=request_id, error={"code": -32000, "message": "fake"})
            break
        if scenario == "early_exit":
            break
        if scenario == "hang":
            update(sessionUpdate="agent_thought_chunk", content={"type": "text", "text": "working"})
            sys.stdin.read()
            sys.exit(0)
        if scenario == "unsupported":
            send(
                id="unknown",
                method="fs/write_text_file",
                params={"path": "outside.txt", "content": "unsafe"},
            )
            reply = json.loads(next(sys.stdin))
            (root / "unsupported.json").write_text(json.dumps(reply))
        if scenario in ("approve", "deny", "outside", "persistent_only"):
            target = str(root.parent / "outside.txt") if scenario == "outside" else "answer.txt"
            args = {"path": target, "content": "allowed write\n"}
            update(
                sessionUpdate="tool_call",
                toolCallId="write-1",
                title="write: " + target,
                kind="edit",
                rawInput=args,
                status="in_progress",
            )
            options = (
                [{"optionId": "session", "kind": "allow_always", "name": "Always"}]
                if scenario == "persistent_only"
                else [
                    {"optionId": "one", "kind": "allow_once", "name": "Once"},
                    {"optionId": "no", "kind": "reject_once", "name": "Deny"},
                ]
            )
            send(
                id="permission-1",
                method="session/request_permission",
                params={
                    "sessionId": "session-1",
                    "toolCall": {
                        "toolCallId": "edit-1",
                        "kind": "edit",
                        "rawInput": {"tool": "write_file", "arguments": args},
                    },
                    "options": options,
                },
            )
            reply = json.loads(next(sys.stdin))
            (root / "permission.json").write_text(json.dumps(reply))
            approved = reply.get("result", {}).get("outcome", {}).get("optionId") == "one"
            if approved:
                (root / target).write_text(args["content"])
            update(
                sessionUpdate="tool_call_update",
                toolCallId="write-1",
                status="completed" if approved else "failed",
                rawOutput="written" if approved else "denied",
            )
        update(sessionUpdate="agent_thought_chunk", content={"type": "text", "text": "planning"})
        for text in ("Hello ", "from Hermes."):
            update(sessionUpdate="agent_message_chunk", content={"type": "text", "text": text})
        send(
            id=request_id,
            result={"stopReason": "max_turn_requests" if scenario == "incomplete" else "end_turn"},
        )
    elif method == "session/cancel":
        break
