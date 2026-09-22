from __future__ import annotations

from arc_llama.failures import StartupFailureError, redact_details


def test_startup_failure_has_stable_safe_public_shape():
    failure = StartupFailureError(
        "model_missing",
        "Model file not found.",
        "Update the model path.",
        details={"path": "/models/missing.gguf", "api_token": "secret"},
    )

    assert failure.http_status == 404
    assert failure.diagnostics_id.startswith("model_missing-")
    assert failure.details["api_token"] == "[REDACTED]"
    assert failure.to_dict() == {
        "error": {
            "category": "model_missing",
            "message": "Model file not found.",
            "action": "Update the model path.",
            "diagnostics_id": failure.diagnostics_id,
        }
    }


def test_diagnostic_redaction_covers_nested_keys_and_command_flags():
    safe = redact_details(
        {
            "authorization": "Bearer private",
            "nested": {"password": "private", "ordinary": "kept"},
            "argv": [
                "tool",
                "--token",
                "private",
                "--api-key=private",
                "--model",
                "kept",
            ],
        }
    )

    assert safe["authorization"] == "[REDACTED]"
    assert safe["nested"] == {"password": "[REDACTED]", "ordinary": "kept"}
    assert safe["argv"] == [
        "tool",
        "--token",
        "[REDACTED]",
        "--api-key=[REDACTED]",
        "--model",
        "kept",
    ]
