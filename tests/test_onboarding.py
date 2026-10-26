from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from arc_llama.api.onboarding import register_onboarding_routes
from arc_llama.config import Config, GPUConfig, ModelConfig, ServerConfig
from arc_llama.onboarding import FirstAnswerSetup
from arc_llama.server import _read_json_body, _require_admin
from arc_llama.starter_discovery import Starter

STARTERS = (
    Starter(
        "catalog-large",
        "Synthetic Large Chat",
        "fixture/large-chat",
        "large-Q4_K_M.gguf",
        4465,
        "a" * 40,
        "mit",
        "fixture",
        4465 * 1048576,
    ),
    Starter(
        "catalog-small",
        "Synthetic Small Chat",
        "fixture/small-chat",
        "small-Q4_K_M.gguf",
        1068,
        "b" * 40,
        "apache-2.0",
        "fixture",
        1068 * 1048576,
    ),
)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    import arc_llama.models as models
    import arc_llama.onboarding as onboarding
    import arc_llama.runtime as runtime

    cfg = Config(
        gpus=[
            GPUConfig(
                "gpu", 0, "battlemage", vram_mb=12288, name="Test Arc", backend="vulkan"
            )
        ]
    )
    cfg.paths.models_dir = str(tmp_path / "models")
    cfg.paths.state_dir = str(tmp_path / "state")
    cfg.paths.llama_server = str(tmp_path / "llama-server")
    cfg.server = ServerConfig(admin_token="test-token")
    router = SimpleNamespace(
        running_models=Mock(return_value=[]),
        _build_servers=Mock(),
        ensure_active=AsyncMock(
            return_value=(None, SimpleNamespace(is_running=True, ready=True))
        ),
    )

    def install(**kw):
        kw["on_progress"](50, 100)
        Path(cfg.paths.llama_server).write_bytes(b"runtime")
        cfg.save(tmp_path / "config.toml")

    installer = Mock(side_effect=install)
    monkeypatch.setattr(runtime, "install_runtime", installer)
    from arc_llama.arch import Backend

    monkeypatch.setattr(
        onboarding, "detect_backends", lambda path: {Backend(cfg.gpus[0].backend)}
    )

    def options(repo, vram):
        s = next(s for s in STARTERS if s.repo == repo)
        return {"options": [{"file": s.file, "size_mb": s.size_mb, "fit": "fits"}]}

    metadata = Mock(side_effect=options)

    async def validate(s, vram):
        result = metadata(s.repo, vram)
        option = next((o for o in result["options"] if o["file"] == s.file), None)
        if (
            not option
            or not option["size_mb"]
            or option["fit"] != "fits"
            or option["size_mb"] != s.size_mb
        ):
            raise ValueError("Current metadata cannot confirm fit")

    monkeypatch.setattr(onboarding, "validate", validate)
    monkeypatch.setattr(onboarding, "check_runtime", AsyncMock())

    async def discover(vram):
        from arc_llama.model_library import fit_badge

        return next((s for s in STARTERS if fit_badge(s.size_mb, vram) == "fits"), None)

    monkeypatch.setattr(onboarding, "discover", discover)

    def download(spec, *, target_dir, progress, revision):
        assert revision == next(s.revision for s in STARTERS if s.repo == spec.repo)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / spec.file
        path.write_bytes(b"synthetic gguf")
        return path

    downloader = Mock(side_effect=download)
    monkeypatch.setattr(models, "download_from_hf", downloader)

    def register(c, paths, *, gpu_pci_slot):
        model = ModelConfig(
            "starter", str(paths[0]), 18080, "gpu", recipe={"ctx": 32768}
        )
        c.models.append(model)
        return [model]

    monkeypatch.setattr(models, "register_discovered", register)
    manager = FirstAnswerSetup(cfg, tmp_path / "config.toml", router)
    manager.suggestion = STARTERS[0]
    manager.discovery_context = manager._context()
    manager.discovery_status = "ready"
    manager.discovery_at = time.monotonic()
    return manager, installer, downloader, metadata


@pytest.mark.parametrize(
    "memory,expected",
    [
        (0, None),
        (512, None),
        (4096, "catalog-small"),
        (12288, "catalog-large"),
        (24576, "catalog-large"),
    ],
)
async def test_recommendation_leaves_headroom_and_unknown_memory_is_not_a_fit(
    setup, memory, expected
):
    manager, installer, download, metadata = setup
    manager.cfg.gpus[0].vram_mb = memory
    manager.refresh(force=True)
    if manager.discovery_task:
        await manager.discovery_task
    suggested = manager.public()["starter"]
    assert (suggested["id"] if suggested else None) == expected
    manager.public()
    installer.assert_not_called()
    download.assert_not_called()
    metadata.assert_not_called()


async def finish(manager):
    await asyncio.wait_for(manager.task, timeout=3)


async def test_fresh_explicit_setup_prepares_registers_and_loads_exact_model(setup):
    manager, installer, download, _ = setup
    manager.start("catalog-large")
    task = manager.task
    manager.start("catalog-large")
    assert manager.task is task
    await finish(manager)
    assert manager.phase == "ready", manager.error
    installer.assert_called_once()
    download.assert_called_once()
    manager.router.ensure_active.assert_awaited_once_with(
        "starter", allow_eviction=False
    )
    assert manager.cfg.models[0].recipe == {"ctx": 4096, "parallel": 1}
    assert manager.public()["chat_url"] == "/chat?model=starter&welcome=1"
    assert manager.download.bytes_done == manager.download.bytes_total
    await manager.shutdown()


async def test_existing_runtime_preserved(setup):
    manager, installer, _, _ = setup
    binary = Path(manager.cfg.paths.llama_server)
    binary.write_bytes(b"existing SYCL")
    manager.cfg.gpus[0].backend = "sycl"
    manager.refresh(force=True)
    await manager.discovery_task
    manager.start("catalog-large")
    await finish(manager)
    assert manager.phase == "ready"
    installer.assert_not_called()
    assert (
        binary.read_bytes() == b"existing SYCL"
        and manager.cfg.gpus[0].backend == "sycl"
    )


@pytest.mark.parametrize("invalid", ["../bad", "not-listed", "catalog-small"])
async def test_only_current_catalog_recommendation_can_start(setup, invalid):
    manager, installer, download, _ = setup
    with pytest.raises(ValueError, match="recommendation changed"):
        manager.start(invalid)
    assert manager.task is None
    installer.assert_not_called()
    download.assert_not_called()


async def test_existing_registered_or_running_models_are_not_changed(setup):
    manager, installer, download, _ = setup
    manager.cfg.models = [ModelConfig("existing", "/synthetic.gguf", 1, "gpu")]
    with pytest.raises(ValueError, match="already registered"):
        manager.start("catalog-large")
    manager.cfg.models = []
    manager.router.running_models.return_value = ["someone-else"]
    with pytest.raises(ValueError, match="Another model"):
        manager.start("catalog-large")
    installer.assert_not_called()
    download.assert_not_called()


@pytest.mark.parametrize("defect", ["missing", "unknown_size", "too_big", "grew"])
async def test_current_metadata_is_checked_before_install_or_download(setup, defect):
    manager, installer, download, metadata = setup
    options = [{"file": STARTERS[0].file, "size_mb": 4465, "fit": "fits"}]
    if defect == "missing":
        options = []
    elif defect == "unknown_size":
        options[0]["size_mb"] = 0
    elif defect == "too_big":
        options[0]["fit"] = "too_big"
    else:
        options[0]["size_mb"] = 9000
    metadata.side_effect = None
    metadata.return_value = {"options": options}
    manager.start("catalog-large")
    await finish(manager)
    assert manager.phase == "error"
    installer.assert_not_called()
    download.assert_not_called()


async def test_pause_after_download_keeps_verified_files_and_resume_does_not_redownload(
    setup,
):
    manager, installer, download, _ = setup
    original = download.side_effect

    def pause_at_download(*args, **kw):
        manager.pause()
        return original(*args, **kw)

    download.side_effect = pause_at_download
    manager.start("catalog-large")
    await finish(manager)
    assert manager.phase == "paused" and manager.model == "starter"
    manager.router.ensure_active.assert_not_awaited()
    manager.start("catalog-large")
    await finish(manager)
    assert manager.phase == "ready"
    installer.assert_called_once()
    download.assert_called_once()
    manager.router.ensure_active.assert_awaited_once()


async def test_pause_during_check_never_starts_download(setup):
    manager, installer, download, _ = setup
    manager.start("catalog-large")
    manager.pause()
    await finish(manager)
    assert manager.phase == "paused"
    installer.assert_not_called()
    download.assert_not_called()


async def test_download_failure_can_retry_without_losing_runtime(setup):
    manager, installer, download, _ = setup
    original = download.side_effect
    download.side_effect = OSError("Synthetic network interruption")
    manager.start("catalog-large")
    await finish(manager)
    assert manager.phase == "error" and "interruption" in manager.error
    manager.router.ensure_active.assert_not_awaited()
    download.side_effect = original
    manager.start("catalog-large")
    await finish(manager)
    assert manager.phase == "ready"
    installer.assert_called_once()
    assert download.call_count == 2


async def test_load_failure_can_retry_without_redownloading(setup):
    manager, installer, download, _ = setup
    manager.router.ensure_active.side_effect = RuntimeError("Synthetic driver failure")
    manager.start("catalog-large")
    await finish(manager)
    assert manager.phase == "error" and "driver" in manager.error
    manager.router.ensure_active.side_effect = None
    manager.start("catalog-large")
    await finish(manager)
    assert manager.phase == "ready"
    installer.assert_called_once()
    download.assert_called_once()


async def test_foreign_model_started_during_download_is_not_evicted(setup):
    manager, _, download, _ = setup
    original = download.side_effect

    def foreign(*args, **kw):
        manager.router.running_models.return_value = ["someone-else"]
        return original(*args, **kw)

    download.side_effect = foreign
    manager.start("catalog-large")
    await finish(manager)
    assert manager.phase == "error" and "Another model is now running" in manager.error
    manager.router.ensure_active.assert_not_awaited()


async def test_registry_save_failure_rolls_back_and_can_retry(setup, monkeypatch):
    manager, _, download, _ = setup
    # An existing runtime avoids involving its separate config save.
    Path(manager.cfg.paths.llama_server).write_bytes(b"existing")
    real_save = manager.cfg.save
    monkeypatch.setattr(
        manager.cfg, "save", Mock(side_effect=OSError("synthetic full disk"))
    )
    manager.start("catalog-large")
    await finish(manager)
    assert manager.phase == "error" and not manager.cfg.models
    manager.router.ensure_active.assert_not_awaited()
    manager.router._build_servers.assert_not_called()
    monkeypatch.setattr(manager.cfg, "save", real_save)
    manager.start("catalog-large")
    await finish(manager)
    assert manager.phase == "ready" and download.call_count == 2


async def test_routes_require_admin_and_read_does_no_work(setup, monkeypatch):
    manager, installer, download, metadata = setup
    app = FastAPI()
    app.state.cfg = manager.cfg
    app.state.router = manager.router
    app.state.first_answer = manager
    register_onboarding_routes(
        app,
        require_admin=_require_admin,
        read_json_body=_read_json_body,
        config_path=manager.config_path,
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/admin/onboarding")).status_code == 401
        assert (
            await client.post("/admin/onboarding", json={"starter": "catalog-large"})
        ).status_code == 401
        auth = {"Authorization": "Bearer test-token"}
        response = await client.get("/admin/onboarding", headers=auth)
        assert response.json()["phase"] == "idle"
        installer.assert_not_called()
        download.assert_not_called()
        metadata.assert_not_called()
        for body in ({"starter": {}}, {"action": "arbitrary"}, {"starter": "x"}):
            assert (
                await client.post("/admin/onboarding", headers=auth, json=body)
            ).status_code == 400
        assert (
            await client.post(
                "/admin/onboarding", headers=auth, json={"starter": "catalog-large"}
            )
        ).status_code == 200
        await finish(manager)
        assert (await client.get("/admin/onboarding", headers=auth)).json()[
            "phase"
        ] == "ready"


async def test_changed_gpu_after_metadata_does_not_download(setup):
    manager, installer, download, metadata = setup
    original = metadata.side_effect

    def changed(*args):
        result = original(*args)
        manager.cfg.gpus[0].vram_mb = 512
        return result

    metadata.side_effect = changed
    manager.start("catalog-large")
    await finish(manager)
    assert manager.phase == "error" and "capacity changed" in manager.error
    installer.assert_not_called()
    download.assert_not_called()


async def test_unrecognized_existing_runtime_is_not_replaced(setup, monkeypatch):
    manager, installer, download, _ = setup
    binary = Path(manager.cfg.paths.llama_server)
    binary.write_bytes(b"existing unknown runtime")
    monkeypatch.setattr("arc_llama.onboarding.detect_backends", lambda path: set())
    manager.start("catalog-large")
    await finish(manager)
    assert manager.phase == "error" and "no recognized" in manager.error
    assert binary.read_bytes() == b"existing unknown runtime"
    installer.assert_not_called()
    download.assert_not_called()


async def test_router_checks_no_eviction_after_waiting_for_swap_lock(
    tmp_path, monkeypatch
):
    from conftest import make_config
    from helpers import FakeServer

    import arc_llama.router as router_mod

    FakeServer.starts = []
    FakeServer.stops = []
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)
    router = router_mod.Router(make_config(tmp_path))
    await router._lock.acquire()
    attempt = asyncio.create_task(router.ensure_active("gemma", allow_eviction=False))
    await asyncio.sleep(0)
    # Another load became resident while setup waited to make its swap decision.
    router._servers["qwen"].running = True
    router._servers["qwen"].ready = True
    router._lock.release()
    with pytest.raises(RuntimeError, match="Another model is running"):
        await attempt
    assert router._servers["qwen"].is_running
    assert not router._servers["gemma"].is_running
    assert not FakeServer.stops and not FakeServer.starts
    # Ordinary user-requested swapping retains its existing behavior.
    await router.ensure_active("gemma")
    assert FakeServer.stops == ["qwen"] and FakeServer.starts == ["gemma"]
    await router.shutdown()


async def test_full_app_setup_routes_register_into_real_router_without_a_prompt(
    setup, monkeypatch
):
    from helpers import FakeServer

    from arc_llama.server import create_app

    manager, _, download, _ = setup
    cfg = manager.cfg
    cfg.gpus[0].vram_mb = 4096
    cfg.tune.auto = False
    FakeServer.starts = []
    FakeServer.stops = []
    monkeypatch.setattr("arc_llama.router.LlamaServer", FakeServer)
    app = create_app(cfg, config_path=manager.config_path, plugins=[])
    auth = {"Authorization": "Bearer test-token"}
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            initial = await client.get("/admin/onboarding", headers=auth)
            await app.state.first_answer.discovery_task
            initial = await client.get("/admin/onboarding", headers=auth)
            assert (
                initial.status_code == 200
                and initial.json()["starter"]["id"] == "catalog-small"
            )
            assert FakeServer.starts == []
            assert (await client.get("/")).status_code == 200
            response = await client.post(
                "/admin/onboarding", headers=auth, json={"starter": "catalog-small"}
            )
            assert response.status_code == 200
            actual = app.state.first_answer
            await finish(actual)
            assert actual.phase == "ready", actual.error
            status = (await client.get("/admin/status", headers=auth)).json()
            assert status["models"][0]["name"] == actual.model
            assert status["models"][0]["loaded"] is True
            assert status["gpus"][0]["backend"] == "vulkan"
            assert FakeServer.starts == [actual.model]
            download.assert_called_once()


async def test_setup_retains_actionable_startup_failure_remedy(setup):
    from arc_llama.failures import StartupFailureError

    manager, _, _, _ = setup
    manager.router.ensure_active.side_effect = StartupFailureError(
        "gpu_unavailable",
        "Synthetic GPU is unavailable.",
        "Run arc-llama doctor to check device permissions.",
    )
    manager.start("catalog-large")
    await finish(manager)
    assert manager.phase == "error"
    assert "Run arc-llama doctor" in manager.error


async def test_runtime_architecture_failure_prevents_model_download(setup, monkeypatch):
    manager, installer, download, _ = setup
    monkeypatch.setattr("arc_llama.onboarding.check_runtime", AsyncMock(side_effect=ValueError("Architecture rejected before weights")))
    manager.start("catalog-large")
    await finish(manager)
    assert manager.phase == "error" and "Architecture rejected" in manager.error
    installer.assert_called_once()
    download.assert_not_called()
    manager.router.ensure_active.assert_not_awaited()


async def test_expired_recommendation_cannot_start(setup):
    manager, installer, download, _ = setup
    manager.discovery_at -= 301
    with pytest.raises(ValueError, match="recommendation changed"):
        manager.start("catalog-large")
    installer.assert_not_called()
    download.assert_not_called()
