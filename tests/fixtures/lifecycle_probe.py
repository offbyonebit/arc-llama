"""Native Linux probe; all processes, ports and files belong to this fixture."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

from arc_llama.launcher import LaunchPlan, LlamaServer

CHILD = r'''
import json, os, signal, socket, subprocess, sys, time
from pathlib import Path
signal.alarm(10)  # A broken supervisor cannot leave a permanent fixture process.
ready, mode, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
if mode == "crash":
    print("synthetic startup crash", flush=True)
    sys.exit(23)
if mode == "startup":
    time.sleep(5)
nested = None
if mode in ("graceful", "concurrent"):
    nested = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.alarm(10); time.sleep(9)"])
def shutdown(*_):
    if nested is not None:
        nested.wait(timeout=2)
    sys.exit(0)
signal.signal(signal.SIGTERM, signal.SIG_IGN if mode in ("force", "cancel") else shutdown)
listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", port))
listener.listen()
Path(ready).write_text(json.dumps({"port": listener.getsockname()[1], "nested": nested.pid if nested else None}))
print("synthetic server ready", flush=True)
while True: time.sleep(0.01)
'''


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("synthetic lifecycle probe exceeded readiness deadline")


async def cancel_stop(server):
    task = asyncio.create_task(server.astop(drain_seconds=0.1))
    await asyncio.sleep(0.02)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def run(root, mode):
    root.mkdir(parents=True, exist_ok=True)
    port = 0
    for iteration in range(3):
        ready = root / f"ready-{iteration}.json"
        plan = LaunchPlan(
            argv=[sys.executable, "-u", "-c", CHILD, str(ready), mode, str(port)],
            env=dict(os.environ), resident_lock_path=root / "resident.lock",
        )
        server = LlamaServer(plan, name="synthetic-lifecycle")
        process = None
        try:
            server.start(log_dir=root / "logs")
            process = server.process
            assert process is not None
            if mode == "crash":
                process.wait(timeout=5)
                assert process.returncode == 23
            elif mode != "startup":
                wait_for(ready.exists)
                info = json.loads(ready.read_text())
                port = info["port"]
            if mode == "cancel":
                asyncio.run(cancel_stop(server))
            elif mode == "concurrent":
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                    list(pool.map(lambda _, server=server: server.stop(drain_seconds=2), range(3)))
            else:
                server.stop(drain_seconds=0.1 if mode in ("force", "startup") else 2)
            assert process.poll() is not None
            if mode == "force":
                assert process.returncode == -signal.SIGKILL
            assert server.process is None
            assert server._resident_lock is None
            assert server._log_file is None
            assert not server.ready
            if mode in ("graceful", "concurrent"):
                assert not Path(f"/proc/{info['nested']}").exists(), "nested fixture was not reaped"
            if port:
                with socket.socket() as rebound:
                    rebound.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    rebound.bind(("127.0.0.1", port))
            log = server.log_path
            assert log is not None and log.exists()
            if mode != "startup":
                assert "synthetic" in log.read_text()
        finally:
            # The process handle/group was created above; never discover or signal
            # unrelated host processes. Reap even after a failed assertion.
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            server.stop(drain_seconds=0.1)
    print(json.dumps({"scenario": mode, "iterations": 3, "port_released": True}))


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2])
