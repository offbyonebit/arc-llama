"""Startup failure payloads: readable structured errors in chat.

The server sends {category, message, action, diagnostics_id, details}; the
bundled chat UI must render that as a readable card with a retry affordance
and expandable diagnostics, be honest about cold-start waiting stages, and
never show a stuck empty assistant bubble. This file covers the payload
contract (bounding, redaction) and the UI's structure via the static
sources; the real-browser behaviour lives in browser-tests/.
"""

from __future__ import annotations

import json
from pathlib import Path

from arc_llama.failures import StartupFailureError, bounded_details

STATIC = Path(__file__).parent.parent / "src" / "arc_llama" / "static"


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


def test_bounded_details_caps_node_counts_and_depth():
    deep = {"a": {"b": {"c": {"d": {"e": {"f": {"g": "way down"}}}}}}}
    bounded = bounded_details(deep, limit=1_000)
    assert "truncated" in json.dumps(bounded) or bounded["a"]["b"]["c"]["d"] is not None


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


# ---------------------------------------------------------------------------
# Chat UI structure (static checks; browser-tests/run.mjs drives behaviour)
# ---------------------------------------------------------------------------


def test_chat_js_renders_structured_failure_cards():
    js = (STATIC / "chat.js").read_text(encoding="utf-8")
    # The card renders the structured fields, not the raw JSON.
    assert "function showStartupFailure" in js
    assert 'textContent = "Load failed"' in js
    assert "category" in js
    assert "diagnostics_id" in js
    # Message, action, diagnostics id, retry, and expandable details are all
    # present in the card markup the function builds.
    assert 'div.dataset.failureCategory' in js
    assert 'div.dataset.diagnosticsId' in js
    assert '"What to do: "' in js
    assert "load-failure-retry" in js
    assert "load-failure-details-toggle" in js
    assert "Show diagnostics" in js
    assert "Hide diagnostics" in js


def test_chat_js_parses_structured_error_bodies():
    js = (STATIC / "chat.js").read_text(encoding="utf-8")
    assert "function parseStructuredFailure" in js
    assert "body.error" in js


def test_chat_js_shows_honest_loading_stage():
    js = (STATIC / "chat.js").read_text(encoding="utf-8")
    # The named loading stage appears in the chat log while the model loads.
    assert "Starting model, this can take a while on first load" in js
    assert "load-wait-card" in js
    # The loading chip is removed on both load outcomes: never a stuck
    # "waiting" message after success or a settled failure.
    assert js.count("loadingCard") >= 3


def test_chat_failure_css_exists():
    css = (STATIC / "chat.css").read_text(encoding="utf-8")
    assert ".load-failure-action" in css
    assert ".load-failure-retry" in css
    assert ".load-failure-details-toggle" in css
    assert ".load-failure-details pre" in css
    assert ".load-failure-details[hidden] { display: none; }" in css


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
