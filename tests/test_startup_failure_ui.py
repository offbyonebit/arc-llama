"""Startup failure payloads: readable structured errors in chat.

The server sends {category, message, action, diagnostics_id, details}; the
bundled chat UI must render that as a readable card with a retry affordance
and expandable diagnostics, be honest about cold-start waiting stages, and
never show a stuck empty assistant bubble. This file covers payload bounding
and redaction; rendered failure cards, loading stages, and retry behavior
are covered in browser-tests/.
"""

from __future__ import annotations

import json

from arc_llama.failures import StartupFailureError, bounded_details


def test_bounded_details_truncates_long_strings():
    details = {"log_tail": "x" * 20_000}
    bounded = bounded_details(details, limit=1_000)
    # The whole payload exceeded the encoded budget, so a bounded preview is
    # returned instead.
    assert bounded["truncated"] is True
    assert len(bounded["preview"]) < 1_000
    # Within-budget strings pass through untouched.
    ok = bounded_details({"path": "/models/qwen.gguf", "exit_code": 3}, limit=1_000)
    assert ok == {"path": "/models/qwen.gguf", "exit_code": 3}


def test_bounded_details_caps_depth():
    deep = {"a": {"b": {"c": {"d": {"e": {"f": {"g": "way down"}}}}}}}
    bounded = bounded_details(deep, limit=1_000)
    assert bounded["a"]["b"]["c"]["d"]["e"]["f"]["g"] == "[truncated]"
    assert "way down" not in json.dumps(bounded)


def test_bounded_details_caps_node_count():
    bounded = bounded_details({"items": list(range(1_000))}, limit=20_000)
    assert len(bounded["items"]) <= 128
    assert bounded["items"][-1] == "[truncated]"
    assert 999 not in bounded["items"]


def test_bounded_details_rejects_non_mapping_input():
    assert bounded_details(["not", "a", "dict"]) == {}
    assert bounded_details(None) == {}


def test_to_dict_includes_details_only_when_requested():
    failure = StartupFailureError(
        "startup_timeout",
        "llama-server timed out while loading qwen.",
        "Open the retained model log, correct the reported problem, and retry.",
        details={"log_tail": "abc"},
    )
    assert "details" not in failure.to_dict()["error"]
    assert failure.to_dict(include_details=True)["error"]["details"] == {"log_tail": "abc"}


def test_ollama_compat_wraps_structured_failure_body():
    """Ollama-compat clients see one plain error string with the readable
    structured message, never nested JSON."""
    from fastapi import Response

    import arc_llama.server as server_mod

    structured = Response(
        content=json.dumps(
            {
                "error": {
                    "category": "model_missing",
                    "message": "Model file not found: /models/missing.gguf.",
                    "action": "Update the model path or remove this registration.",
                    "diagnostics_id": "model_missing-x",
                }
            }
        ),
        status_code=404,
    )
    wrapped = server_mod._openai_response_as_ollama(structured, "gemma", generate=False)
    assert wrapped.status_code == 404
    payload = json.loads(bytes(wrapped.body))
    assert payload == {"error": "Model file not found: /models/missing.gguf."}

    # Flat detail errors keep their existing contract.
    flat = Response(content=json.dumps({"detail": "broken pipe"}), status_code=502)
    wrapped_flat = server_mod._openai_response_as_ollama(flat, "gemma", generate=False)
    assert json.loads(bytes(wrapped_flat.body)) == {"error": "broken pipe"}


def test_public_inference_errors_do_not_expose_admin_diagnostics(monkeypatch):
    from fastapi.testclient import TestClient
    from test_server import FakeRouter, FakeUpstreamManager

    import arc_llama.server as server_mod
    from arc_llama.config import Config

    class FailingRouter(FakeRouter):
        async def ensure_active(self, query, *, acquire=False):
            raise StartupFailureError(
                "process_exited", "Model could not start.", "Retry or inspect admin diagnostics.",
                details={"log_tail": "private local backend log", "argv": ["/private/runtime"]},
            )

    monkeypatch.setattr(server_mod, "Router", FailingRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    app = server_mod.create_app(Config(), plugins=[])
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "qwen", "messages": []})
        assert response.status_code == 503
        assert "details" not in response.json()["error"]
        assert "private local" not in response.text
        assert response.json()["error"]["diagnostics_id"]


def test_admin_stop_timeout_returns_diagnostics_instead_of_success(monkeypatch):
    from fastapi.testclient import TestClient
    from test_server import FakeRouter, FakeUpstreamManager

    import arc_llama.server as server_mod
    from arc_llama.config import Config

    class StuckRouter(FakeRouter):
        async def stop_one(self, name):
            raise StartupFailureError(
                "gpu_unavailable", "Child has not exited", "Wait for child exit",
                details={"pid": 12345, "reason": "shutdown_timeout"},
            )

        async def stop_all(self):
            return await self.stop_one("qwen")

    monkeypatch.setattr(server_mod, "Router", StuckRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    app = server_mod.create_app(Config(), plugins=[])
    with TestClient(app) as client:
        for endpoint in ("/admin/stop/qwen", "/admin/stop-all"):
            response = client.post(endpoint)
            assert response.status_code == 503
            assert response.json()["error"]["details"]["reason"] == "shutdown_timeout"
            assert "has not exited" in response.json()["error"]["message"]
            assert "loaded" not in response.json()
