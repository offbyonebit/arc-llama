"""Open the local UI only after this server has bound its listening socket."""

from __future__ import annotations

import logging
import threading
import time
import webbrowser
from typing import Any

log = logging.getLogger(__name__)


def wait_and_open(
    server: Any, url: str, stopped: threading.Event, timeout: float = 60
) -> None:
    deadline = time.monotonic() + timeout
    while not stopped.is_set() and time.monotonic() < deadline:
        if server.started:
            try:
                if not webbrowser.open(url, new=2):
                    log.warning("Could not open a browser. Open %s manually.", url)
            except Exception:
                log.warning("Could not open a browser. Open %s manually.", url)
            return
        if server.should_exit:
            return
        stopped.wait(0.1)


def serve_in_browser(app: Any, *, host: str, port: int, path: str) -> None:
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info"))
    stopped = threading.Event()
    opener = threading.Thread(
        target=wait_and_open,
        args=(server, f"http://127.0.0.1:{port}{path}", stopped),
        daemon=True,
    )
    opener.start()
    try:
        server.run()
    finally:
        stopped.set()
        opener.join(timeout=0.5)
