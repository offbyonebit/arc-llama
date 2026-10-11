from __future__ import annotations

import asyncio
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from arc_llama import model_compatibility as compat
from arc_llama.config import Config, ModelConfig


def header(architecture: str) -> bytes:
    def string(value):
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    return (
        b"GGUF"
        + struct.pack("<IQQ", 3, 0, 1)
        + string("general.architecture")
        + struct.pack("<I", 8)
        + string(architecture)
    )


@pytest.mark.parametrize(
    "data", [b"", b"GGUF", header("llama")[:-2], header("../llama"), b"NOT!" + header("llama")[4:]]
)
def test_unusable_metadata_stays_unknown(data):
    assert compat.header_architecture(data) is None


def test_architecture_comes_from_metadata_not_filename():
    assert compat.header_architecture(header("lfm2")) == "lfm2"


@pytest.mark.parametrize(
    "body",
    [
        {"repo": "../bad", "file": "m.gguf"},
        {"repo": "u/r", "file": "../m.gguf"},
        {"repo": "u/r", "file": "/m.gguf"},
        {"repo": "u/r", "file": "m\\x.gguf"},
        {"repo": "u/r", "file": "m.txt"},
        {"name": 42},
    ],
)
def test_untrusted_request_paths_rejected(body):
    with pytest.raises(ValueError):
        compat.validate_request(body)


def test_runtime_identity_changes_with_adjacent_library(tmp_path):
    runtime = tmp_path / "llama-server"
    runtime.write_bytes(b"fake")
    library = tmp_path / "libllama.so"
    library.write_bytes(b"one")
    first = compat.runtime_identity(str(runtime))
    library.write_bytes(b"two-updated")
    assert first != compat.runtime_identity(str(runtime))
    runtime.unlink()
    assert compat.runtime_identity(str(runtime)) is None


@pytest.mark.parametrize(
    ("diagnostic", "expected"),
    [
        ("unknown model architecture: 'newarch'", "rejected"),
        ("key not found in model: newarch.context_length", "recognized"),
        ("unknown model architecture: otherarch", "unknown"),
        ("device not available", "unknown"),
        ("key not found in model: tokenizer.ggml.tokens", "unknown"),
    ],
)
async def test_probe_requires_architecture_specific_evidence(monkeypatch, diagnostic, expected):
    compat._PROBE_CACHE.clear()
    monkeypatch.setattr(compat, "probe_output", AsyncMock(return_value=(1, diagnostic)))
    assert await compat.probe_architecture("runtime", "identity", "newarch") == expected
    compat._PROBE_CACHE.clear()


async def test_remote_assessment_pins_selected_file_and_does_not_claim_inference(monkeypatch):
    cfg = Config()
    cfg.paths.llama_server = "test-runtime"
    sha = "a" * 40
    monkeypatch.setattr(compat, "runtime_identity", lambda _: ("/runtime", "fingerprint"))
    monkeypatch.setattr(
        compat,
        "remote_metadata",
        AsyncMock(return_value={"sha": sha, "siblings": [{"rfilename": "m.gguf"}]}),
    )
    reads = []

    async def read(repo, file, revision):
        reads.append((repo, file, revision))
        return header("llama")

    monkeypatch.setattr(compat, "remote_header", read)
    monkeypatch.setattr(compat, "probe_architecture", AsyncMock(return_value="recognized"))
    result = await compat.assess_compatibility(cfg, {"repo": "u/r", "file": "m.gguf"})
    assert reads == [("u/r", "m.gguf", sha)]
    assert result["status"] == "recognized" and "unverified" in result["scope"]
    assert result["revision"] == sha
    with pytest.raises(ValueError):
        await compat.assess_compatibility(cfg, {"repo": "u/r", "file": "wrong.gguf"})
    assert len(reads) == 1


async def test_local_rejection_and_changed_runtime_remain_distinct(tmp_path, monkeypatch):
    cfg = Config()
    file = tmp_path / "looks-like-llama.gguf"
    file.write_bytes(header("newarch"))
    cfg.models = [ModelConfig("local", str(file), 18080, "gpu")]
    monkeypatch.setattr(compat, "runtime_identity", lambda _: ("/runtime", "old"))
    monkeypatch.setattr(compat, "probe_architecture", AsyncMock(return_value="rejected"))
    result = await compat.assess_compatibility(cfg, {"name": "local"})
    assert result["status"] == "incompatible" and result["architecture"] == "newarch"
    identities = iter([("/runtime", "old"), ("/runtime", "new")])
    monkeypatch.setattr(compat, "runtime_identity", lambda _: next(identities))
    assert (await compat.assess_compatibility(cfg, {"name": "local"}))["status"] == "unknown"


async def test_compatibility_endpoint_auth_and_validation(monkeypatch, tmp_path):
    from httpx import ASGITransport, AsyncClient
    from test_model_library import _app

    _, app = _app(monkeypatch, tmp_path)
    calls = []

    async def assess(cfg, body):
        calls.append(body)
        return {"status": "unknown"}

    monkeypatch.setattr(compat, "assess_compatibility", assess)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            assert (
                await client.post(
                    "/admin/library/compatibility", json={"repo": "u/r", "file": "m.gguf"}
                )
            ).status_code == 401
            auth = {"Authorization": "Bearer tok"}
            assert (
                await client.post(
                    "/admin/library/compatibility",
                    json={"repo": "u/r", "file": "../m.gguf"},
                    headers=auth,
                )
            ).status_code == 400
            response = await client.post(
                "/admin/library/compatibility", json={"repo": "u/r", "file": "m.gguf"}, headers=auth
            )
            assert response.json() == {"status": "unknown"}
    assert calls == [{"repo": "u/r", "file": "m.gguf"}]


async def test_unknown_probe_is_retryable(monkeypatch):
    compat._PROBE_CACHE.clear()
    probe = AsyncMock(
        side_effect=[
            (1, "device not available"),
            (1, "key not found in model: llama.context_length"),
        ]
    )
    monkeypatch.setattr(compat, "probe_output", probe)
    assert await compat.probe_architecture("runtime", "identity", "llama") == "unknown"
    assert await compat.probe_architecture("runtime", "identity", "llama") == "recognized"
    assert await compat.probe_architecture("runtime", "identity", "llama") == "recognized"
    assert probe.await_count == 2
    compat._PROBE_CACHE.clear()


def test_registered_file_identity_changes_when_file_is_replaced(tmp_path):
    from arc_llama.model_library import file_readiness

    path = tmp_path / "m.gguf"
    path.write_bytes(header("llama"))
    model = ModelConfig("local", str(path), 18080, "gpu")
    before = file_readiness(model)["identity"]
    path.write_bytes(header("qwen3"))
    assert before != file_readiness(model)["identity"]


async def test_deadline_includes_queue_wait_and_recovers_capacity(monkeypatch):
    started = asyncio.Event()
    cleaned = []

    async def blocked(*args):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.append(True)

    monkeypatch.setattr(compat, "assess_compatibility", blocked)
    checks = compat.CompatibilityChecks(timeout=1, concurrency=1)
    first = asyncio.create_task(checks.assess(Config(), {}))
    await started.wait()
    checks.timeout = 0.02
    result = await checks.assess(Config(), {})
    assert result["status"] == "unknown" and "timed out" in result["detail"]
    assert not cleaned  # Queued check never started external IO.
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert cleaned == [True]
    monkeypatch.setattr(
        compat, "assess_compatibility", AsyncMock(return_value={"status": "recognized"})
    )
    assert (await checks.assess(Config(), {}))["status"] == "recognized"


async def test_whole_check_deadline_cancels_active_work(monkeypatch):
    cleaned = []

    async def blocked(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.append(True)

    monkeypatch.setattr(compat, "assess_compatibility", blocked)
    checks = compat.CompatibilityChecks(timeout=0.02)
    result = await checks.assess(Config(), {})
    assert result["status"] == "unknown" and "timed out" in result["detail"]
    assert cleaned == [True]


async def test_shutdown_cancels_checks_and_stops_new_work(monkeypatch):
    started = asyncio.Event()
    cleaned = []

    async def blocked(*args):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.append(True)

    monkeypatch.setattr(compat, "assess_compatibility", blocked)
    checks = compat.CompatibilityChecks()
    pending = asyncio.create_task(checks.assess(Config(), {}))
    await started.wait()
    await checks.shutdown()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert cleaned == [True]
    assert "shutting down" in (await checks.assess(Config(), {}))["detail"]


async def test_disconnect_cancels_assessment_before_returning(monkeypatch):
    from fastapi import HTTPException

    from arc_llama.api.library import _compatibility_until_disconnect

    started = asyncio.Event()
    cleaned = []

    async def assess(*args):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.append(True)

    async def disconnected():
        await started.wait()
        return {"type": "http.disconnect"}

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(compatibility_checks=SimpleNamespace(assess=assess), cfg=Config())
        ),
        receive=disconnected,
    )
    with pytest.raises(HTTPException) as error:
        await _compatibility_until_disconnect(request, {})
    assert error.value.status_code == 499 and cleaned == [True]


@pytest.mark.parametrize(
    ("phase", "exception", "detail", "expected_log"),
    [
        (
            "metadata",
            __import__("httpx").ConnectError("secret-token-url"),
            "metadata",
            "ConnectError",
        ),
        ("metadata", ValueError("private-token"), "metadata", "ValueError"),
        ("probe", OSError("private/path/token"), "runtime probe", "OSError"),
    ],
)
async def test_failures_log_phase_and_cause_without_secrets(
    monkeypatch, caplog, phase, exception, detail, expected_log
):
    monkeypatch.setattr(compat, "runtime_identity", lambda _: ("runtime", "identity"))
    metadata = {"sha": "a" * 40, "siblings": [{"rfilename": "m.gguf"}]}
    monkeypatch.setattr(
        compat,
        "remote_metadata",
        AsyncMock(side_effect=exception)
        if phase == "metadata"
        else AsyncMock(return_value=metadata),
    )
    monkeypatch.setattr(compat, "remote_header", AsyncMock(return_value=header("llama")))
    monkeypatch.setattr(compat, "probe_architecture", AsyncMock(side_effect=exception))
    result = await compat.assess_compatibility(Config(), {"repo": "u/r", "file": "m.gguf"})
    assert result["status"] == "unknown" and detail in result["detail"]
    assert expected_log in caplog.text
    assert str(exception) not in caplog.text and str(exception) not in str(result)
