from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from arc_llama.config import Config, GPUConfig, ModelConfig, ServerConfig
from arc_llama.model_library import (
    DownloadManager,
    deletable_files,
    disk_report,
    fit_badge,
    group_repo_files,
    quant_label,
    repo_options,
    search_repos,
)

GB = 1024**3


def test_quant_labels() -> None:
    assert quant_label("Qwen3-8B-Q4_K_M.gguf") == "Q4_K_M"
    assert quant_label("x-UD-Q8_K_XL-00001-of-00002.gguf") == "UD-Q8_K_XL"
    assert quant_label("model-IQ3_XXS.gguf") == "IQ3_XXS"
    assert quant_label("model-BF16.gguf") == "BF16"
    assert quant_label("model.gguf") == "unknown"


def test_fit_badges() -> None:
    assert fit_badge(5000, 24576) == "fits"
    assert fit_badge(22500, 24576) == "tight"
    assert fit_badge(30000, 24576) == "too_big"
    assert fit_badge(5000, None) == "unknown"


def test_group_repo_files_folds_shards_and_flags_vision() -> None:
    files = [
        ("README.md", 10),
        ("m-Q4_K_M.gguf", 5 * GB),
        ("mmproj-F16.gguf", GB // 2),
        ("Q8/m-Q8_0-00001-of-00002.gguf", 10 * GB),
        ("Q8/m-Q8_0-00002-of-00002.gguf", 10 * GB),
        ("m-F32.gguf", None),
    ]
    options, vision = group_repo_files(files, 24576)
    assert vision
    by_file = {o.file: o for o in options}
    assert set(by_file) == {"m-Q4_K_M.gguf", "Q8/m-Q8_0-00001-of-00002.gguf", "m-F32.gguf"}
    split = by_file["Q8/m-Q8_0-00001-of-00002.gguf"]
    assert split.shards == 2 and split.size_mb == 20 * 1024 and split.fit == "fits"
    assert by_file["m-Q4_K_M.gguf"].fit == "fits"
    assert by_file["m-F32.gguf"].fit == "unknown"
    assert [o.file for o in options][0] == "m-F32.gguf"  # sorted by size, unknown first


class FakeApi:
    def list_models(self, **kwargs):
        assert kwargs["filter"] == "gguf"
        return [
            SimpleNamespace(
                id="u/qwen-GGUF",
                downloads=100,
                likes=5,
                last_modified=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
            ),
            SimpleNamespace(id="u/other", downloads=None, likes=None, last_modified=None),
        ]

    def model_info(self, repo, files_metadata=False):
        assert files_metadata
        return SimpleNamespace(
            siblings=[SimpleNamespace(rfilename="qwen-Q4_K_M.gguf", size=4 * GB)]
        )


def test_search_and_repo_options() -> None:
    results = search_repos("qwen", api=FakeApi())
    assert results[0] == {
        "repo": "u/qwen-GGUF",
        "downloads": 100,
        "likes": 5,
        "updated": "2026-01-01T00:00:00+00:00",
    }
    assert results[1]["updated"] is None
    info = repo_options("u/qwen-GGUF", 24576, api=FakeApi())
    assert info["options"][0]["fit"] == "fits"
    with pytest.raises(ValueError):
        repo_options("../etc", 1, api=FakeApi())


def _cfg(tmp_path: Path) -> Config:
    cfg = Config(gpus=[GPUConfig("gpu", 0, "battlemage", vram_mb=24576)])
    cfg.paths.models_dir = str(tmp_path / "models")
    cfg.paths.state_dir = str(tmp_path / "state")
    return cfg


async def test_download_manager_runs_and_registers(tmp_path, monkeypatch) -> None:
    import arc_llama.models as models_mod

    cfg = _cfg(tmp_path)

    def fake_download(spec, *, target_dir, progress):
        target_dir.mkdir(parents=True, exist_ok=True)
        out = target_dir / spec.file
        out.write_bytes(b"x" * 1000)
        return out

    monkeypatch.setattr(models_mod, "download_from_hf", fake_download)
    registered: list[Path] = []

    async def register(path: Path) -> list[str]:
        registered.append(path)
        return ["qwen-q4"]

    manager = DownloadManager(cfg, register)
    job = manager.submit("u/qwen-GGUF", "qwen-Q4_K_M.gguf", 1000)
    with pytest.raises(ValueError):
        manager.submit("u/qwen-GGUF", "qwen-Q4_K_M.gguf", 1000)
    for bad in (("bad repo", "x.gguf"), ("u/r", "x.bin"), ("u/r", "../x.gguf")):
        with pytest.raises(ValueError):
            manager.submit(*bad, None)
    for _ in range(100):
        if job.status in ("done", "error"):
            break
        await asyncio.sleep(0.02)
    assert job.status == "done", job.error
    assert job.registered == ["qwen-q4"]
    assert registered[0].name == "qwen-Q4_K_M.gguf"
    assert Path(job.target_dir) == tmp_path / "models" / "qwen-GGUF"


async def test_download_failure_is_reported(tmp_path, monkeypatch) -> None:
    import arc_llama.models as models_mod

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(models_mod, "download_from_hf", boom)

    async def register(_path):
        raise AssertionError("must not register")

    manager = DownloadManager(_cfg(tmp_path), register)
    job = manager.submit("u/r", "x.gguf", None)
    for _ in range(100):
        if job.status == "error":
            break
        await asyncio.sleep(0.02)
    assert job.status == "error" and "disk full" in (job.error or "")


def test_deletable_files_stay_inside_models_dir_and_skip_shared(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    folder = tmp_path / "models" / "gemma"
    folder.mkdir(parents=True)
    projector = folder / "mmproj-F16.gguf"
    projector.write_bytes(b"p")
    q4 = folder / "gemma-Q4_K_M.gguf"
    q8 = folder / "gemma-Q8_0.gguf"
    q4.write_bytes(b"a")
    q8.write_bytes(b"b")
    outside = tmp_path / "elsewhere.gguf"
    outside.write_bytes(b"c")
    cfg.models = [
        ModelConfig("q4", str(q4), 1, "gpu", recipe={"mmproj": str(projector)}),
        ModelConfig("q8", str(q8), 2, "gpu", recipe={"mmproj": str(projector)}),
        ModelConfig("mine", str(outside), 3, "gpu"),
    ]
    assert deletable_files(cfg, cfg.models[0]) == [q4]
    assert deletable_files(cfg, cfg.models[2]) == []
    cfg.models = cfg.models[:1]
    assert set(deletable_files(cfg, cfg.models[0])) == {q4, projector}
    report = disk_report(cfg, {"q4": 123.0})
    [row] = report["models"]
    assert row["managed"] and row["last_used"] == 123.0 and row["files"] == 2


def _app(monkeypatch, tmp_path):
    import arc_llama.server as server_mod

    try:
        from tests.test_server import FakeRouter, FakeUpstreamManager
    except ModuleNotFoundError:
        from test_server import FakeRouter, FakeUpstreamManager

    class Router(FakeRouter):
        async def stop_one(self, name):
            self.stopped = name
            return True

    monkeypatch.setattr(server_mod, "Router", Router)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    cfg = _cfg(tmp_path)
    cfg.server = ServerConfig(admin_token="tok")
    model_file = tmp_path / "models" / "q.gguf"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"x" * 2048)
    cfg.models = [ModelConfig("qwen", str(model_file), 18080, "gpu")]
    return server_mod, server_mod.create_app(cfg, config_path=tmp_path / "c.toml", plugins=[])


async def test_library_endpoints(monkeypatch, tmp_path) -> None:
    server_mod, app = _app(monkeypatch, tmp_path)
    monkeypatch.setattr(server_mod, "search_repos", lambda q, limit: [{"repo": "u/r"}])

    def fake_options(repo, vram):
        if repo == "u/broken":
            raise RuntimeError("hub down")
        return {"repo": repo, "vram_mb": vram, "options": [], "vision": False}

    monkeypatch.setattr(server_mod, "repo_options", fake_options)
    auth = {"Authorization": "Bearer tok"}
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            assert (await client.get("/admin/library/search?q=qw")).status_code == 401
            found = await client.get("/admin/library/search?q=qwen", headers=auth)
            assert found.json() == {"results": [{"repo": "u/r"}]}
            repo = await client.get("/admin/library/repo?repo=u/r", headers=auth)
            assert repo.json()["vram_mb"] == 24576
            broken = await client.get("/admin/library/repo?repo=u/broken", headers=auth)
            assert broken.status_code == 502
            bad = await client.post(
                "/admin/library/download", json={"repo": 1, "file": "x"}, headers=auth
            )
            assert bad.status_code == 400
            assert (await client.get("/admin/library/jobs", headers=auth)).json() == {"jobs": []}
            disk = (await client.get("/admin/library/disk", headers=auth)).json()
            assert disk["models"][0]["name"] == "qwen"
            missing = await client.delete("/admin/library/models/nope", headers=auth)
            assert missing.status_code == 404
            removed = await client.delete(
                "/admin/library/models/qwen?delete_files=true", headers=auth
            )
            assert removed.json()["deleted_files"] == 1
            assert app.state.cfg.models == []
            assert app.state.router.stopped == "qwen"
    assert not (tmp_path / "models" / "q.gguf").exists()
