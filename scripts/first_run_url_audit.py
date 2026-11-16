"""Copied to an isolated child's sitecustomize.py by first_run_check.py."""

from __future__ import annotations

import importlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request


def record(url: object, status: int, error: str = "") -> None:
    parts = urllib.parse.urlsplit(str(url))
    if parts.scheme not in {"http", "https"} or parts.hostname in {"localhost", "127.0.0.1", "::1"}:
        return
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query = [(k, "REDACTED" if k.lower().startswith(("x-amz-", "x-goog-")) or
              k.lower() in {"token", "signature", "sig", "key", "api_key", "policy", "key-pair-id"} else v)
             for k, v in query]
    safe_url = urllib.parse.urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path,
                                     urllib.parse.urlencode(query), ""))
    entry = {"time": time.time(), "url": safe_url, "status": status, "error": error}
    fd = os.open(os.environ["ARC_FIRST_RUN_URL_LOG"], os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, (json.dumps(entry) + "\n").encode())
    finally:
        os.close(fd)


def responses(response: object) -> None:
    for item in [*getattr(response, "history", []), response]:
        record(item.url, item.status_code)


_open = urllib.request.OpenerDirector.open
_redirect = urllib.request.HTTPRedirectHandler.redirect_request


def audited_redirect(self, request, fp, code, msg, headers, newurl):
    record(request.full_url, code)
    return _redirect(self, request, fp, code, msg, headers, newurl)


def audited_open(self, fullurl, *args, **kwargs):
    url = getattr(fullurl, "full_url", fullurl)
    try:
        result = _open(self, fullurl, *args, **kwargs)
    except urllib.error.HTTPError as exc:
        record(exc.url, exc.code)
        raise
    except Exception as exc:
        record(url, 0, type(exc).__name__)
        raise
    record(result.url, result.status)
    return result


urllib.request.OpenerDirector.open = audited_open
urllib.request.HTTPRedirectHandler.redirect_request = audited_redirect

# These packages belong to the installed application, not the runner. The
# stdlib hook also loads while a new venv is being created, before they exist.
try:
    import httpx
except ImportError:
    pass
else:
    _send = httpx.Client.send
    _async_send = httpx.AsyncClient.send

    def audited_send(self, request, *args, **kwargs):
        try:
            result = _send(self, request, *args, **kwargs)
        except Exception as exc:
            record(request.url, 0, type(exc).__name__)
            raise
        responses(result)
        return result

    async def audited_async_send(self, request, *args, **kwargs):
        try:
            result = await _async_send(self, request, *args, **kwargs)
        except Exception as exc:
            record(request.url, 0, type(exc).__name__)
            raise
        responses(result)
        return result

    httpx.Client.send = audited_send
    httpx.AsyncClient.send = audited_async_send

def instrument_requests(module_name: str) -> None:
    try:
        requests = importlib.import_module(module_name)
    except ImportError:
        return
    original_send = requests.Session.send

    def audited_requests_send(self, request, *args, **kwargs):
        try:
            result = original_send(self, request, *args, **kwargs)
        except Exception as exc:
            record(request.url, 0, type(exc).__name__)
            raise
        responses(result)
        return result

    requests.Session.send = audited_requests_send


instrument_requests("requests")
instrument_requests("pip._vendor.requests")
