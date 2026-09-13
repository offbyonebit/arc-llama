"""Focused tests for the ComfyUI adapter (no server, no weights, no network).

The ComfyUI HTTP surface is simulated by monkeypatching the adapter module's
three stdlib HTTP thread-targets (``_http_json``, ``_http_bytes``,
``_http_json_bytes``), so the tests exercise the full submit → poll → fetch
lifecycle — workflow construction, response parsing, timeout handling, and
error mapping — without any server process or ML dependency.

The fake-backend end-to-end suite in ``test_vision.py`` is retained as-is;
this file only covers the ``comfyui`` adapter.

Run from the repo root:

    .venv/bin/python -m pytest arc-llama-vision/tests -q
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import urllib.error
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

# Make src importable regardless of install state, mirroring test_vision.py.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import arc_llama_vision.comfyui as comfy_mod  # noqa: E402
from arc_llama_vision.backend import (  # noqa: E402
    BackendUnavailableError,
    CapabilityError,
    GeneratedImage,
    create_backend,
)
from arc_llama_vision.comfyui import ComfyUIBackend, ComfyUIConfig  # noqa: E402

MODEL_ID = "arc-vision-flux2-klein-9b-uncensored"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40 + b"IEND\xaeB`\x82"


# ---------------------------------------------------------------------------
# helpers: a tiny fake ComfyUI server installed over the HTTP targets
# ---------------------------------------------------------------------------

_OUTPUT_IMAGE = {
    "filename": "arc-vision_00001_.png",
    "subfolder": "",
    "type": "output",
}


class FakeComfy:
    """In-memory simulacrum of the three ComfyUI endpoints the adapter uses.

    Records every request for assertions and answers with ComfyUI-shaped
    JSON: ``{"prompt_id"}`` on submit, a history mapping on
    ``/history/{id}``, and image bytes on ``/view``.
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, Any]] = []  # (method, path, payload)
        self.history: dict[str, Any] = {}
        self.submitted: dict[str, Any] = {}
        self.images: dict[str, bytes] = {}
        self.next_prompt_id = 0
        # What /history answers until overridden: prompt finished with one
        # saved output image (the SaveImage node's descriptor).
        self.status_str = "success"
        self.outputs: dict[str, Any] = {"37": {"images": [_OUTPUT_IMAGE]}}

    # -- routing ----------------------------------------------------------

    def _answer(self, url: str, method: str, payload: Any) -> Any:
        if method == "GET" and "/system_stats" in url:
            self.requests.append((method, "/system_stats", None))
            return {"system": {"comfyui_version": "0.3"}}
        if method == "GET" and "/history/" in url:
            self.requests.append((method, "/history", None))
            pid = url.rsplit("/", 1)[-1]
            entry = {"status": {"status_str": self.status_str, "completed": True}}
            if self.outputs is not None:
                entry["outputs"] = self.outputs
            return {pid: entry}
        if method == "GET" and "/view" in url:
            self.requests.append((method, "/view", None))
            return self.images.get("default", PNG_BYTES)
        if method == "POST" and "/prompt" in url:
            body = json.loads(payload.decode("utf-8"))
            self.requests.append((method, "/prompt", body))
            self.next_prompt_id += 1
            pid = f"fake-prompt-{self.next_prompt_id}"
            self.submitted[pid] = body
            return {"prompt_id": pid}
        if method == "GET" and "/prompt" in url:  # queue overview / probe fallback
            self.requests.append((method, "/prompt", None))
            return {}
        raise AssertionError(f"unexpected request: {method} {url!r}")

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_json(url: str, *, timeout: float) -> Any:
            return self._answer(url, "GET", None)

        def fake_json_post(url: str, *, data: bytes, timeout: float) -> Any:
            return self._answer(url, "POST", data)

        def fake_bytes(url: str, *, timeout: float) -> bytes:
            out = self._answer(url, "GET", None)
            assert isinstance(out, bytes)
            return out

        monkeypatch.setattr(comfy_mod, "_http_json", fake_json)
        monkeypatch.setattr(comfy_mod, "_http_json_bytes", fake_json_post)
        monkeypatch.setattr(comfy_mod, "_http_bytes", fake_bytes)


@pytest.fixture
def make_comfy(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeComfy]:
    """Factory installing a (subclass of) fake server on this test's patcher.

    Using the ``monkeypatch`` fixture (rather than ad-hoc instances) makes
    teardown automatic, so no patch can leak past a failing test.
    """

    def _make(cls: type[FakeComfy] = FakeComfy) -> FakeComfy:
        server = cls()
        server.install(monkeypatch)
        return server

    return _make


@pytest.fixture
def fake_comfy(make_comfy: Callable[..., FakeComfy]) -> Iterator[FakeComfy]:
    yield make_comfy()


@pytest.fixture
def backend(fake_comfy: FakeComfy) -> ComfyUIBackend:
    return ComfyUIBackend(options={"poll_interval": 0.01})


# ---------------------------------------------------------------------------
# configuration: defaults, ARV_BACKEND_OPTIONS parsing, validation
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = ComfyUIConfig.from_options({})
    assert cfg.base_url == "http://127.0.0.1:8190"
    assert cfg.model_id == MODEL_ID
    assert (
        cfg.unet_gguf
        and cfg.clip_gguf
        and cfg.vae
        and cfg.sampler
        and cfg.filename_prefix
    )
    assert cfg.sampler == "euler"
    assert cfg.steps == 20
    assert cfg.guidance == 5.0
    assert cfg.filename_prefix == "arc-vision"
    assert cfg.submit_timeout == 60.0
    assert cfg.poll_timeout == 1200.0
    assert cfg.poll_interval == 1.0


def test_config_options_override_defaults() -> None:
    cfg = ComfyUIConfig.from_options(
        {
            "base_url": "http://10.0.0.5:8188/",
            "model": "my-flux",
            "unet_gguf": "klein-q4.gguf",
            "clip_gguf": "qwen3-8b-q2.gguf",
            "vae": "flux-2-vae.safetensors",
            "sampler": "dpmpp_2m",
            "steps": 8,
            "guidance": 3.5,
            "filename_prefix": "test-out",
            "poll_timeout": 300,
            "poll_interval": 0.5,
            "submit_timeout": 5,
            "future_unknown_option": "ignored",
        }
    )
    assert cfg.base_url == "http://10.0.0.5:8188"  # trailing slash stripped
    assert cfg.model_id == "my-flux"
    assert cfg.unet_gguf == "klein-q4.gguf"
    assert cfg.clip_gguf == "qwen3-8b-q2.gguf"
    assert cfg.vae == "flux-2-vae.safetensors"
    assert cfg.sampler == "dpmpp_2m"
    assert cfg.steps == 8
    assert cfg.guidance == 3.5
    assert cfg.filename_prefix == "test-out"
    assert cfg.poll_timeout == 300.0
    assert cfg.poll_interval == 0.5
    assert cfg.submit_timeout == 5.0  # unknown key ignored silently


def test_config_rejects_bad_values() -> None:
    for key, bad in [
        ("base_url", 123),
        ("model", ""),
        ("unet_gguf", None),
        ("steps", "20"),
        ("steps", True),
        ("steps", 0),
        ("guidance", "high"),
        ("poll_timeout", -1),
    ]:
        with pytest.raises(CapabilityError):
            ComfyUIConfig.from_options({key: bad})
    with pytest.raises(CapabilityError):
        ComfyUIConfig.from_options({"base_url": "ftp://weird"})


def test_backend_resolves_through_create_backend() -> None:
    inst = create_backend("comfyui", {"steps": 42})
    assert isinstance(inst, ComfyUIBackend)
    assert inst.backend_name == "comfyui"
    assert inst.config.steps == 42
    assert [m.id for m in inst.models] == [inst.config.model_id]


# ---------------------------------------------------------------------------
# workflow construction: minimal Flux2 Klein text-to-image graph
# ---------------------------------------------------------------------------


def _workflow(backend: ComfyUIBackend) -> dict[str, Any]:
    return backend.build_workflow(
        "a test prompt", width=512, height=512, seed=12345, filename_prefix="arc-vision"
    )


def test_workflow_is_valid_api_format_graph(backend: ComfyUIBackend) -> None:
    wf = _workflow(backend)
    assert set(wf) == {
        "10", "11", "12", "20", "21", "30", "31", "32", "33", "34", "35", "36", "37"
    }
    for node_id, node in wf.items():
        assert isinstance(node_id, str)
        assert isinstance(node["class_type"], str) and node["class_type"]
        assert isinstance(node["inputs"], dict)


def test_workflow_uses_exactly_the_flux2_klein_nodes(backend: ComfyUIBackend) -> None:
    wf = _workflow(backend)
    classes = [n["class_type"] for n in wf.values()]
    assert sorted(classes) == sorted(
        [
            "UnetLoaderGGUF",
            "CLIPLoaderGGUF",
            "VAELoader",
            "CLIPTextEncode",
            "CLIPTextEncode",  # positive + empty negative each
            "EmptyFlux2LatentImage",
            "KSamplerSelect",
            "RandomNoise",
            "Flux2Scheduler",
            "CFGGuider",
            "SamplerCustomAdvanced",
            "VAEDecode",
            "SaveImage",
        ]
    )
    encode_nodes = [n for n in wf.values() if n["class_type"] == "CLIPTextEncode"]
    assert len(encode_nodes) == 2
    assert {n["inputs"]["text"] for n in encode_nodes} == {"", "a test prompt"}


def test_workflow_inputs_mirror_upstream_node_signatures(backend: ComfyUIBackend) -> None:
    """Node classes and input names must match upstream ComfyUI exactly."""
    wf = _workflow(backend)
    assert wf["10"] == {
        "class_type": "UnetLoaderGGUF",
        "inputs": {"unet_name": backend.config.unet_gguf},
    }
    assert wf["11"]["class_type"] == "CLIPLoaderGGUF"
    assert wf["11"]["inputs"] == {"clip_name": backend.config.clip_gguf, "type": "flux2"}
    assert wf["12"] == {
        "class_type": "VAELoader",
        "inputs": {"vae_name": backend.config.vae},
    }
    assert wf["30"]["class_type"] == "EmptyFlux2LatentImage"
    assert wf["30"]["inputs"] == {"width": 512, "height": 512, "batch_size": 1}
    assert wf["31"]["inputs"] == {"noise_seed": 12345}
    assert wf["32"]["inputs"] == {"sampler_name": backend.config.sampler}
    assert wf["33"]["inputs"] == {
        "steps": backend.config.steps,
        "width": 512,
        "height": 512,
    }
    guider = wf["34"]["inputs"]
    assert guider["model"] == ["10", 0]
    assert guider["positive"] == ["20", 0]
    assert guider["negative"] == ["21", 0]
    assert guider["cfg"] == backend.config.guidance
    sca = wf["35"]["inputs"]
    assert sca["noise"] == ["31", 0]
    assert sca["guider"] == ["34", 0]
    assert sca["sampler"] == ["32", 0]
    assert sca["sigmas"] == ["33", 0]
    assert sca["latent_image"] == ["30", 0]
    assert wf["36"]["inputs"] == {"samples": ["35", 0], "vae": ["12", 0]}
    assert wf["37"]["inputs"] == {"images": ["36", 0], "filename_prefix": "arc-vision"}
    # Every link reference points at an existing node id.
    for node in wf.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in wf


def test_workflow_link_topology_matches_klein_samplercustom_graph(backend: ComfyUIBackend) -> None:
    """The link set must mirror ComfyUI's official Flux.2 Klein template.

    Loader nodes feed the guider/decode/save chain: 13 total input links.
    """
    wf = _workflow(backend)
    links: set[tuple[str, str, int]] = set()  # (source node, target node, slot)
    for node_id, node in wf.items():
        for value in node["inputs"].values():
            if isinstance(value, list):
                links.add((value[0], node_id, value[1]))
    assert len(links) == 13
    # Every class participates, none dangles.
    sources = {src for src, _, _ in links}
    targets = {tgt for _, tgt, _ in links}
    assert sources | targets == set(wf)
    assert "37" in targets  # SaveImage consumes; nothing consumes SaveImage.


def test_generate_rejects_sizes_not_divisible_by_16(backend: ComfyUIBackend) -> None:
    with pytest.raises(CapabilityError):
        asyncio.run(backend.generate("x", backend.models[0], size="513x512"))


def test_generate_rejects_n_above_one(backend: ComfyUIBackend) -> None:
    with pytest.raises(CapabilityError):
        asyncio.run(backend.generate("x", backend.models[0], n=2))


# ---------------------------------------------------------------------------
# mocked submit / history / view success path
# ---------------------------------------------------------------------------


def test_generate_end_to_end_with_mocked_comfyui(fake_comfy: FakeComfy) -> None:
    backend = ComfyUIBackend(options={"poll_interval": 0.01})
    images = asyncio.run(backend.generate("a red cube", backend.models[0], size="512x512"))
    assert len(images) == 1
    img = images[0]
    assert isinstance(img, GeneratedImage)
    assert img.b64_json == base64.b64encode(PNG_BYTES).decode("ascii")
    assert img.metadata["backend"] == "comfyui"
    assert img.metadata["format"] == "png"
    assert img.metadata["size"] == "512x512"
    assert 0 <= img.metadata["seed"] < 2**63
    assert img.metadata["comfyui"]["filename"] == "arc-vision_00001_.png"
    assert img.metadata["comfyui"]["subfolder"] == ""
    assert img.metadata["comfyui"]["prompt_id"].startswith("fake-prompt-")

    # Submit carried the workflow with the prompt and dimensions...
    submits = [r for r in fake_comfy.requests if r[1] == "/prompt" and r[0] == "POST"]
    assert len(submits) == 1
    body = submits[0][2]
    assert body["prompt"]["20"]["inputs"]["text"] == "a red cube"
    assert body["prompt"]["21"]["inputs"]["text"] == ""
    assert body["prompt"]["30"]["inputs"] == {
        "width": 512,
        "height": 512,
        "batch_size": 1,
    }
    assert body["prompt"]["37"]["inputs"]["filename_prefix"] == "arc-vision"
    # .../view was fetched last: submit → poll → view, in that order.
    order = [r[1] for r in fake_comfy.requests]
    assert order.index("/prompt") < order.index("/history") < order.index("/view")


def test_generate_uses_default_size_1024_when_unspecified(fake_comfy: FakeComfy) -> None:
    backend = ComfyUIBackend(options={"poll_interval": 0.01})
    asyncio.run(backend.generate("anything", backend.models[0]))
    submit = next(r for r in fake_comfy.requests if r[1] == "/prompt" and r[0] == "POST")[2]
    assert submit["prompt"]["30"]["inputs"]["width"] == 1024
    assert submit["prompt"]["30"]["inputs"]["height"] == 1024


def test_generate_without_output_images_is_capability_error(make_comfy) -> None:
    server = make_comfy()
    server.outputs = {}
    backend = ComfyUIBackend(options={"poll_interval": 0.01})
    with pytest.raises(CapabilityError):
        asyncio.run(backend.generate("x", backend.models[0]))


def test_generate_with_execution_error_status_is_capability_error(make_comfy) -> None:
    server = make_comfy()
    server.status_str = "error"
    backend = ComfyUIBackend(options={"poll_interval": 0.01})
    with pytest.raises(CapabilityError):
        asyncio.run(backend.generate("x", backend.models[0]))


def test_generate_with_non_png_view_response_is_capability_error(fake_comfy: FakeComfy) -> None:
    fake_comfy.images["default"] = b"not-an-image"
    backend = ComfyUIBackend(options={"poll_interval": 0.01})
    with pytest.raises(CapabilityError):
        asyncio.run(backend.generate("x", backend.models[0]))


# ---------------------------------------------------------------------------
# timeout and backend failure paths
# ---------------------------------------------------------------------------


class _NeverFinishes(FakeComfy):
    def _answer(self, url: str, method: str, payload: Any) -> Any:
        if method == "GET" and "/history/" in url:
            self.requests.append((method, "/history", None))
            return {}  # the prompt never lands in history
        return super()._answer(url, method, payload)


class _DiesAfterSubmit(FakeComfy):
    def _answer(self, url: str, method: str, payload: Any) -> Any:
        if method == "GET" and "/history/" in url:
            raise urllib.error.URLError("server went away")
        return super()._answer(url, method, payload)


class _RejectsSubmit400(FakeComfy):
    def _answer(self, url: str, method: str, payload: Any) -> Any:
        if method == "POST" and "/prompt" in url:
            raise urllib.error.HTTPError(
                url, 400, "invalid prompt", hdrs=None, fp=None  # type: ignore[arg-type]
            )
        return super()._answer(url, method, payload)


class _ViewUnreachable(FakeComfy):
    def _answer(self, url: str, method: str, payload: Any) -> Any:
        if method == "GET" and "/view" in url:
            raise urllib.error.URLError("gone")
        return super()._answer(url, method, payload)


def test_generate_poll_timeout_is_backend_unavailable(make_comfy) -> None:
    make_comfy(_NeverFinishes)
    backend = ComfyUIBackend(options={"poll_timeout": 0.05, "poll_interval": 0.01})
    with pytest.raises(BackendUnavailableError):
        asyncio.run(backend.generate("x", backend.models[0]))


def test_generate_server_gone_mid_poll_is_backend_unavailable(make_comfy) -> None:
    make_comfy(_DiesAfterSubmit)
    backend = ComfyUIBackend(options={"poll_interval": 0.01})
    with pytest.raises(BackendUnavailableError):
        asyncio.run(backend.generate("x", backend.models[0]))


def test_generate_submit_rejected_400_is_capability_error(make_comfy) -> None:
    make_comfy(_RejectsSubmit400)
    backend = ComfyUIBackend(options={"poll_interval": 0.01})
    with pytest.raises(CapabilityError):
        asyncio.run(backend.generate("x", backend.models[0]))


def test_generate_view_unreachable_is_backend_unavailable(make_comfy) -> None:
    make_comfy(_ViewUnreachable)
    backend = ComfyUIBackend(options={"poll_interval": 0.01})
    with pytest.raises(BackendUnavailableError):
        asyncio.run(backend.generate("x", backend.models[0]))


# ---------------------------------------------------------------------------
# startup health probe (system_stats / prompt fallback)
# ---------------------------------------------------------------------------


def test_startup_probe_success_and_models_listed(fake_comfy: FakeComfy) -> None:
    backend = ComfyUIBackend()
    asyncio.run(backend.startup())  # no exception = healthy
    assert any(r[1] == "/system_stats" for r in fake_comfy.requests)
    assert [m.id for m in backend.models] == [MODEL_ID]


class _DeadComfy(FakeComfy):
    def _answer(self, url: str, method: str, payload: Any) -> Any:
        raise urllib.error.URLError("refused")


def test_startup_unreachable_is_backend_unavailable(make_comfy) -> None:
    make_comfy(_DeadComfy)
    backend = ComfyUIBackend()
    with pytest.raises(BackendUnavailableError):
        asyncio.run(backend.startup())


class _HiddenSystemStats(FakeComfy):
    def _answer(self, url: str, method: str, payload: Any) -> Any:
        if method == "GET" and "/system_stats" in url:
            raise urllib.error.URLError("hidden")
        return super()._answer(url, method, payload)


def test_startup_uses_prompt_fallback_when_system_stats_hidden(make_comfy) -> None:
    make_comfy(_HiddenSystemStats)
    backend = ComfyUIBackend()
    asyncio.run(backend.startup())  # /prompt probe fallback answered


def test_unreachable_comfyui_degrades_health_and_returns_503(make_comfy) -> None:
    """Boot the real app with a dead ComfyUI: /health degraded, generation 503."""
    from arc_llama_vision.app import create_app
    from arc_llama_vision.config import VisionConfig
    from fastapi.testclient import TestClient

    make_comfy(_DeadComfy)
    backend = ComfyUIBackend()
    app = create_app(VisionConfig(), backend)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "degraded"
        resp = client.post(
            "/v1/images/generations",
            json={"model": MODEL_ID, "prompt": "x"},
        )
        assert resp.status_code == 503


def test_reachable_comfyui_serves_generation_end_to_end_through_the_app(
    fake_comfy: FakeComfy,
) -> None:
    """The wired-in adapter drives the real HTTP surface over the fake server."""
    from arc_llama_vision.app import create_app
    from arc_llama_vision.config import VisionConfig
    from fastapi.testclient import TestClient

    backend = ComfyUIBackend(options={"poll_interval": 0.01})
    app = create_app(VisionConfig(), backend)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "backend": "comfyui"}
        models = client.get("/v1/models").json()["data"]
        assert [m["id"] for m in models] == [MODEL_ID]
        assert models[0]["metadata"]["backend"] == "comfyui"
        resp = client.post(
            "/v1/images/generations",
            json={"model": MODEL_ID, "prompt": "a red cube", "size": "512x512"},
        )
        assert resp.status_code == 200
        img = resp.json()["data"][0]
        assert img["b64_json"] == base64.b64encode(PNG_BYTES).decode("ascii")
        assert img["metadata"]["backend"] == "comfyui"


# ---------------------------------------------------------------------------
# stdlib-only constraint (mirrors test_vision.py's hygiene test)
# ---------------------------------------------------------------------------


def test_comfyui_adapter_module_imports_no_heavy_ml_dependencies() -> None:
    code = (
        f"import sys, importlib; sys.path.insert(0, {str(_SRC)!r}); "
        "importlib.import_module('arc_llama_vision.comfyui'); "
        "bad = sorted(m.split('.')[0] for m in sys.modules "
        "if m.split('.')[0] in ('torch', 'diffusers', 'transformers', 'numpy')); "
        "print(','.join(bad))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"heavy imports leaked: {result.stdout}"
