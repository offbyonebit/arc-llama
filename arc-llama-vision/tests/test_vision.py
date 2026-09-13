"""Focused end-to-end tests for the vision companion scaffold.

The tests drive the real FastAPI app in-process through TestClient (which
exercises startup/shutdown lifecycle hooks) plus one bounded boot of the
actual uvicorn server over a real TCP socket. The deterministic fake backend
supplies image bytes, so no model weights, network access, or ML
dependencies are required. No POSIX-only mechanisms are used, keeping the
suite runnable on Windows.

Run from the repo root:

    .venv/bin/python -m pytest arc-llama-vision/tests -q
"""

from __future__ import annotations

import base64
import importlib
import struct
import subprocess
import sys
import threading
import time
import zlib
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

# Make the companion's src importable regardless of install state, mirroring
# how tests/test_example_plugin.py preprends the example plugin's src path.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from arc_llama_vision.app import create_app  # noqa: E402
from arc_llama_vision.backend import (  # noqa: E402
    BACKEND_REGISTRY,
    BackendUnavailableError,
    CapabilityError,
    FakeImageBackend,
    ModelInfo,
    UnavailableBackend,
    create_backend,
    validate_size,
)
from arc_llama_vision.config import VisionConfig  # noqa: E402


@pytest.fixture
def cfg() -> VisionConfig:
    return VisionConfig()


@pytest.fixture
def client(cfg: VisionConfig) -> Iterator[TestClient]:
    app = create_app(cfg, FakeImageBackend())
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


def test_health_ok_with_ready_backend(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["backend"] == "fake"


def test_health_reports_degraded_after_backend_startup_failure(cfg: VisionConfig) -> None:
    class BrokenStartupBackend(FakeImageBackend):
        async def startup(self) -> None:
            raise RuntimeError("no GPU today")

    app = create_app(cfg, BrokenStartupBackend())
    with TestClient(app) as c:
        resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"


# ---------------------------------------------------------------------------
# model listing
# ---------------------------------------------------------------------------


def test_models_list_includes_image_modality_metadata(client: TestClient) -> None:
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert "arc-vision-diffusion" in ids
    assert "arc-vision-thumbnail" in ids
    for m in body["data"]:
        assert m["object"] == "model"
        assert m["owned_by"] == "vision-companion"
        assert m["metadata"]["modality"] == "image->image"
        assert m["metadata"]["backend"] == "fake"
    # The companion contract requires unique model ids.
    assert len(ids) == len(set(ids))


def test_models_list_empty_for_modelless_backend(cfg: VisionConfig) -> None:
    class ModellessBackend(FakeImageBackend):
        models = []

    app = create_app(cfg, ModellessBackend())
    with TestClient(app) as client:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert resp.json()["data"] == []


def test_model_ids_distinct_from_arc_llama_text_models(client: TestClient) -> None:
    """Image model ids must not collide with local GGUF model names used by the core."""
    body = client.get("/v1/models").json()
    for m in body["data"]:
        assert not m["id"].endswith(".gguf")


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def _decode_png(b64: str) -> bytes:
    return base64.b64decode(b64, validate=True)


def test_generation_returns_openai_shape_and_valid_png(client: TestClient) -> None:
    resp = client.post(
        "/v1/images/generations",
        json={"model": "arc-vision-diffusion", "prompt": "a red cube", "size": "64x64"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["created"], int)
    assert isinstance(body["data"], list) and len(body["data"]) == 1
    img = body["data"][0]
    assert set(img) == {"b64_json", "metadata"}
    png = _decode_png(img["b64_json"])
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert img["metadata"]["backend"] == "fake"
    assert img["metadata"]["format"] == "png"
    assert img["metadata"]["size"] == "64x64"
    import zlib as _zlib

    assert _zlib.crc32(png) is not None  # fully-formed payload


def test_generation_png_dimensions_match_request(client: TestClient) -> None:
    resp = client.post(
        "/v1/images/generations",
        json={"model": "arc-vision-diffusion", "prompt": "sizing", "size": "128x96"},
    )
    assert resp.status_code == 200
    png = _decode_png(resp.json()["data"][0]["b64_json"])
    width, height = struct.unpack(">II", png[16:24])
    assert (width, height) == (128, 96)


def test_generation_is_deterministic_for_same_request(cfg: VisionConfig) -> None:
    app = create_app(cfg.apply_overrides(max_batch_images=2), FakeImageBackend())
    with TestClient(app) as client:
        payload = {"model": "arc-vision-diffusion", "prompt": "same prompt", "n": 2}
        first = client.post("/v1/images/generations", json=payload)
        second = client.post("/v1/images/generations", json=payload)
        assert first.status_code == second.status_code == 200
        assert first.json()["data"] == second.json()["data"]
        # Batched entries must differ per index.
        imgs = first.json()["data"]
        assert imgs[0]["b64_json"] != imgs[1]["b64_json"]


def test_generation_distinct_prompts_produce_distinct_images(client: TestClient) -> None:
    a = client.post(
        "/v1/images/generations",
        json={"model": "arc-vision-diffusion", "prompt": "apple"},
    ).json()["data"][0]["b64_json"]
    b = client.post(
        "/v1/images/generations",
        json={"model": "arc-vision-diffusion", "prompt": "banana"},
    ).json()["data"][0]["b64_json"]
    assert a != b


def test_generation_fake_backend_seed_matches_prompt_crc(client: TestClient) -> None:
    prompt = "deterministic color check"
    img = client.post(
        "/v1/images/generations",
        json={"model": "arc-vision-diffusion", "prompt": prompt},
    ).json()["data"][0]
    seed = img["metadata"]["seed"]
    expected_crc = zlib.crc32(prompt.encode("utf-8")) & 0xFFFFFFFF
    assert seed == expected_crc


def test_generation_accepts_optional_fields(client: TestClient) -> None:
    resp = client.post(
        "/v1/images/generations",
        json={
            "model": "arc-vision-diffusion",
            "prompt": "a still life",
            "n": 1,
            "size": "128x128",
            "quality": "standard",
            "style": "natural",
            "response_format": "b64_json",
            "user": "someone",
        },
    )
    assert resp.status_code == 200
    assert "b64_json" in resp.json()["data"][0]


def test_generation_rejects_url_response_format(client: TestClient) -> None:
    """url output needs a persistent image store; the companion is stateless."""
    calls = []

    class RecordingBackend(FakeImageBackend):
        async def generate(self, prompt, model, *, n=1, size=None, **kwargs):
            calls.append(size)
            return await super().generate(prompt, model, n=n, size=size, **kwargs)

    app = create_app(VisionConfig(), RecordingBackend())
    with TestClient(app) as c:
        resp = c.post(
            "/v1/images/generations",
            json={
                "model": "arc-vision-diffusion",
                "prompt": "x",
                "response_format": "url",
            },
        )
        assert resp.status_code == 400
        err = resp.json()["detail"]["error"]
        assert err["type"] == "invalid_request_error"
        assert "b64_json" in err["message"]
    # Rejected before any adapter work, per the fail-fast guard order.
    assert calls == []


# ---------------------------------------------------------------------------
# invalid model / invalid input
# ---------------------------------------------------------------------------


def test_generation_unknown_model_returns_404(client: TestClient) -> None:
    resp = client.post(
        "/v1/images/generations",
        json={"model": "no-such-model", "prompt": "anything"},
    )
    assert resp.status_code == 404
    err = resp.json()["detail"]["error"]
    assert err["type"] == "invalid_request_error"
    assert "no-such-model" in err["message"]


def test_generation_missing_prompt_returns_400(client: TestClient) -> None:
    resp = client.post("/v1/images/generations", json={"model": "arc-vision-diffusion"})
    # Validation errors are remapped to OpenAI-style 400s.
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"]["type"] == "invalid_request_error"


def test_generation_empty_prompt_returns_400(client: TestClient) -> None:
    resp = client.post(
        "/v1/images/generations",
        json={"model": "arc-vision-diffusion", "prompt": ""},
    )
    assert resp.status_code == 400


def test_generation_invalid_size_returns_400(client: TestClient) -> None:
    resp = client.post(
        "/v1/images/generations",
        json={"model": "arc-vision-diffusion", "prompt": "x", "size": "not-a-size"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"]["type"] == "invalid_request_error"


def test_generation_rejects_unknown_size_values_without_backend_call(client: TestClient) -> None:
    """Unsupported sizes must be rejected before any adapter work happens."""
    calls = []

    class RecordingBackend(FakeImageBackend):
        async def generate(self, prompt, model, *, n=1, size=None, **kwargs):
            calls.append(size)
            return await super().generate(prompt, model, n=n, size=size, **kwargs)

    app = create_app(VisionConfig(), RecordingBackend())
    with TestClient(app) as c:
        bad = c.post(
            "/v1/images/generations",
            json={"model": "arc-vision-diffusion", "prompt": "x", "size": "10000x1000"},
        )
        assert bad.status_code == 400
        good = c.post(
            "/v1/images/generations",
            json={"model": "arc-vision-diffusion", "prompt": "x", "size": "256x256"},
        )
        assert good.status_code == 200
    assert calls == ["256x256"]


def test_generation_over_limit_batch_and_prompt_return_400(client: TestClient) -> None:
    ok = client.post(
        "/v1/images/generations",
        json={"model": "arc-vision-diffusion", "prompt": "x", "n": 1},
    )
    assert ok.status_code == 200
    # Defaults: max_batch_images=1, max_prompt_chars=8192.
    too_many = client.post(
        "/v1/images/generations",
        json={"model": "arc-vision-diffusion", "prompt": "x", "n": 2},
    )
    assert too_many.status_code == 400
    too_long = client.post(
        "/v1/images/generations",
        json={"model": "arc-vision-diffusion", "prompt": "x" * 9000},
    )
    assert too_long.status_code == 400
    # Configured limits raise the boundary accordingly.
    app = create_app(VisionConfig(max_batch_images=3, max_prompt_chars=32), FakeImageBackend())
    with TestClient(app) as c:
        three = c.post(
            "/v1/images/generations",
            json={"model": "arc-vision-diffusion", "prompt": "x", "n": 3},
        )
        assert three.status_code == 200
        assert len(three.json()["data"]) == 3
        long_ok = c.post(
            "/v1/images/generations",
            json={"model": "arc-vision-diffusion", "prompt": "y" * 40},
        )
        assert long_ok.status_code == 400


# ---------------------------------------------------------------------------
# backend-unavailable behavior
# ---------------------------------------------------------------------------


def test_generation_backend_unavailable_returns_503(cfg: VisionConfig) -> None:
    app = create_app(cfg, UnavailableBackend())
    with TestClient(app) as client:
        resp = client.post(
            "/v1/images/generations",
            json={"model": "arc-vision-diffusion", "prompt": "anything"},
        )
        assert resp.status_code == 503
        err = resp.json()["detail"]["error"]
        assert err["type"] == "server_error"
        assert "unavailable" in err["message"].lower()


def test_generation_after_backend_startup_failure_returns_503(cfg: VisionConfig) -> None:
    class BrokenStartupBackend(FakeImageBackend):
        # Models advertised but the backend process never came up.

        async def startup(self) -> None:
            raise RuntimeError("backend process failed to launch")

    app = create_app(cfg, BrokenStartupBackend())
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.json()["status"] == "degraded"
        resp = client.post(
            "/v1/images/generations",
            json={"model": "arc-vision-diffusion", "prompt": "anything"},
        )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# adapter seam: factories and size validation
# ---------------------------------------------------------------------------


def test_create_backend_known_and_unknown() -> None:
    assert isinstance(create_backend("fake"), FakeImageBackend)
    assert isinstance(create_backend("unavailable"), UnavailableBackend)
    with pytest.raises(CapabilityError):
        create_backend("nope")


def test_backend_registry_covers_declared_backends() -> None:
    assert set(BACKEND_REGISTRY) == {"fake", "unavailable"}


def test_validate_size_accepts_and_rejects() -> None:
    assert validate_size("64x64") == (64, 64)
    assert validate_size("1024x1024") == (1024, 1024)
    assert validate_size(" 64x64 ") == (64, 64)  # whitespace tolerated
    assert validate_size(None) is None
    for bad in ("", "64", "ax64", "64x", "0x64", "4097x1024", "65536x64"):
        with pytest.raises(CapabilityError):
            validate_size(bad)


async def test_unavailable_backend_generate_raises() -> None:
    """Adapters raise HTTP-free exceptions; the endpoint owns status mapping.

    Verifies both failure flavors cross the seam as plain exceptions.

    BackendUnavailableError: unusable backend process (-> 503 upstream).
    CapabilityError: unsupported size/format at the adapter layer (-> 400).
    """
    backend = UnavailableBackend()
    with pytest.raises(BackendUnavailableError):
        await backend.generate("x", ModelInfo(id="m", backend="unavailable"))

    fake = FakeImageBackend()
    with pytest.raises(CapabilityError):
        await fake.generate("x", fake.models[0], size="10000x1000")


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    defaults = VisionConfig()
    assert defaults.host == "127.0.0.1"
    assert defaults.port == 11440
    assert defaults.backend == "fake"
    assert defaults.max_batch_images == 1
    assert defaults.max_prompt_chars == 8192


def test_config_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("ARV_HOST", "192.0.2.1")
    monkeypatch.setenv("ARV_PORT", "9111")
    monkeypatch.setenv("ARV_BACKEND", "unavailable")
    monkeypatch.setenv("ARV_MAX_BATCH_IMAGES", "4")
    monkeypatch.setenv("ARV_BACKEND_OPTIONS", '{"api_base": "http://10.0.0.5:1234"}')

    cfg = VisionConfig.from_env()
    assert cfg.host == "192.0.2.1"
    assert cfg.port == 9111
    assert cfg.backend == "unavailable"
    assert cfg.max_batch_images == 4
    assert cfg.backend_options == {"api_base": "http://10.0.0.5:1234"}


def test_config_invalid_env_values_fall_back_to_defaults(monkeypatch) -> None:
    monkeypatch.setenv("ARV_PORT", "not-a-number")
    monkeypatch.setenv("ARV_MAX_PROMPT_CHARS", "also-not-a-number")
    monkeypatch.setenv("ARV_BACKEND_OPTIONS", "{not json")
    cfg = VisionConfig.from_env()
    assert cfg.port == 11440
    assert cfg.max_prompt_chars == 8192
    assert cfg.backend_options == {}


def test_config_apply_overrides_give_cli_precedence(monkeypatch) -> None:
    monkeypatch.setenv("ARV_PORT", "9999")
    base = VisionConfig.from_env()
    assert base.port == 9999
    assert base.apply_overrides(port=None).port == 9999  # None = untouched
    assert base.apply_overrides(port=11440).port == 11440


# ---------------------------------------------------------------------------
# end-to-end over a real socket
# ---------------------------------------------------------------------------


def test_companion_serves_real_traffic_end_to_end(cfg: VisionConfig) -> None:
    """Start the real companion (uvicorn) and exercise the HTTP surface.

    Bounded: a single OS-assigned port on the loopback interface, threads
    instead of POSIX process tools, and explicit shutdown. This validates
    that the package is a usable standalone server, not just a WSGI-ish object.
    """
    import socket

    app = create_app(cfg, FakeImageBackend())

    # Reserve an ephemeral port to avoid racing on a hard-coded number.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": None}, daemon=True)
    # uvicorn signal handlers must not run in a non-main thread.
    server.install_signal_handlers = (  # type: ignore[method-assign, attr-defined]
        lambda: None
    )
    thread.start()

    base = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(base_url=base, timeout=5.0) as http:
            deadline = time.time() + 10
            up = False
            while time.time() < deadline:
                try:
                    if http.get("/health").status_code == 200:
                        up = True
                        break
                except httpx.HTTPError:
                    time.sleep(0.05)
            assert up, "companion did not start serving within 10s"

            health = http.get("/health").json()
            assert health == {"status": "ok", "backend": "fake"}

            models = http.get("/v1/models").json()
            assert models["object"] == "list"
            assert all(m["metadata"]["modality"] == "image->image" for m in models["data"])

            gen = http.post(
                "/v1/images/generations",
                json={
                    "model": "arc-vision-diffusion",
                    "prompt": "end-to-end smoke",
                    "size": "64x64",
                },
            )
            assert gen.status_code == 200
            assert base64.b64decode(gen.json()["data"][0]["b64_json"]).startswith(b"\x89PNG")

            unknown = http.post(
                "/v1/images/generations",
                json={"model": "missing-model", "prompt": "x"},
            )
            assert unknown.status_code == 404
    finally:
        server.should_exit = True
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# package hygiene: bounded dependencies, clean imports
# ---------------------------------------------------------------------------


def test_no_heavy_ml_dependencies_at_import() -> None:
    """Importing the companion must not pull torch/diffusers/transformers/numpy."""
    code = (
        f"import sys; sys.path.insert(0, {str(_SRC)!r}); "
        "import arc_llama_vision; "
        "bad = sorted(m.split('.')[0] for m in sys.modules "
        "if m.split('.')[0] in ('torch', 'diffusers', 'transformers', 'numpy')); "
        "print(','.join(bad))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"heavy imports leaked: {result.stdout}"


def test_config_module_reloads_cleanly() -> None:
    import arc_llama_vision.config as config_mod

    importlib.reload(config_mod)
    assert config_mod.VisionConfig().port == 11440
