"""Admin routes for Hugging Face downloads and local model-library storage."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request

from arc_llama.config import Config
from arc_llama.perf_history import PerfHistory
from arc_llama.router import Router


def register_library_routes(
    app: FastAPI,
    *,
    require_admin: Callable[..., Any],
    read_json_body: Callable[[Request], Awaitable[dict[str, Any]]],
    config_path: Path | None,
    search_repos_fn: Callable[..., Any],
    repo_options_fn: Callable[..., Any],
    disk_report_fn: Callable[..., Any],
    deletable_files_fn: Callable[..., Any],
    download_manager_factory: Callable[..., Any],
) -> None:
    """Register model-library endpoints using server-owned auth and helpers."""
    # ------------------------------------------------------------------
    # Model library: Hugging Face search, downloads, disk usage
    # ------------------------------------------------------------------

    def _library_vram(c: Config) -> int | None:
        gpu = next((g for g in c.gpus if g.enabled), None)
        return gpu.vram_mb if gpu is not None else None

    def _downloads(request: Request) -> Any:
        manager = getattr(request.app.state, "downloads", None)
        if manager is None:
            c: Config = request.app.state.cfg
            rt: Router = request.app.state.router

            async def register(path: Path) -> list[str]:
                from arc_llama.config import default_config_path
                from arc_llama.models import register_discovered

                added = register_discovered(c, [path])
                if added:
                    c.save(config_path or default_config_path())
                    rt._build_servers()  # type: ignore[attr-defined]
                return [m.name for m in added]

            manager = download_manager_factory(c, register)
            request.app.state.downloads = manager
        return manager

    @app.get("/admin/library/search")
    async def library_search(
        request: Request,
        q: str = Query(..., min_length=2, max_length=100),
        limit: int = Query(20, ge=1, le=50),
        _auth: None = Depends(require_admin),
    ) -> dict[str, Any]:
        try:
            results = await asyncio.to_thread(search_repos_fn, q, limit=limit)
        except Exception as e:  # noqa: BLE001 - network or hub errors
            raise HTTPException(status_code=502, detail=f"Hugging Face search failed: {e}") from e
        return {"results": results}

    @app.get("/admin/library/repo")
    async def library_repo(
        request: Request,
        repo: str = Query(..., max_length=200),
        _auth: None = Depends(require_admin),
    ) -> dict[str, Any]:
        c: Config = request.app.state.cfg
        try:
            return await asyncio.to_thread(repo_options_fn, repo, _library_vram(c))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Could not read {repo}: {e}") from e

    @app.post("/admin/library/compatibility")
    async def library_compatibility(
        request: Request, _auth: None = Depends(require_admin)
    ) -> dict[str, Any]:
        from arc_llama.model_compatibility import assess_compatibility, validate_request

        body = await read_json_body(request)
        try:
            validate_request(body)
            return await asyncio.to_thread(assess_compatibility, request.app.state.cfg, body)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/admin/library/download")
    async def library_download(
        request: Request, _auth: None = Depends(require_admin)
    ) -> dict[str, Any]:
        body = await read_json_body(request)
        repo, file = body.get("repo"), body.get("file")
        size_mb = body.get("size_mb")
        if not isinstance(repo, str) or not isinstance(file, str):
            raise HTTPException(status_code=400, detail="repo and file must be strings")
        total = int(size_mb) * 1_048_576 if isinstance(size_mb, int) and size_mb > 0 else None
        try:
            job = _downloads(request).submit(repo, file, total)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return job.public()

    @app.get("/admin/library/jobs")
    async def library_jobs(
        request: Request, _auth: None = Depends(require_admin)
    ) -> dict[str, Any]:
        manager = getattr(request.app.state, "downloads", None)
        jobs = list(manager.jobs.values()) if manager is not None else []
        return {"jobs": [j.public() for j in sorted(jobs, key=lambda j: -j.started_at)]}

    @app.get("/admin/library/disk")
    async def library_disk(
        request: Request, _auth: None = Depends(require_admin)
    ) -> dict[str, Any]:
        c: Config = request.app.state.cfg
        history: PerfHistory | None = getattr(request.app.state, "perf_history", None)
        last_used: dict[str, float] = {}
        if history is not None:
            for point in history.query(days=90):
                last_used[point["model"]] = max(last_used.get(point["model"], 0), point["t"])
        return await asyncio.to_thread(disk_report_fn, c, last_used)

    @app.delete("/admin/library/models/{name}")
    async def library_remove(
        name: str,
        request: Request,
        delete_files: bool = False,
        _auth: None = Depends(require_admin),
    ) -> dict[str, Any]:
        """Unregister a model, optionally deleting its files.

        Files are deleted only inside ``paths.models_dir`` and only when no
        other registered model uses them.
        """
        from arc_llama.config import default_config_path

        c: Config = request.app.state.cfg
        rt: Router = request.app.state.router
        model = next((m for m in c.models if m.name == name), None)
        if model is None:
            raise HTTPException(status_code=404, detail=f"Unknown model: {name!r}")
        doomed = deletable_files_fn(c, model) if delete_files else []
        await rt.stop_one(name)
        previous = list(c.models)
        c.models = [m for m in c.models if m.name != name]
        try:
            c.save(config_path or default_config_path())
        except OSError as e:
            c.models = previous
            raise HTTPException(
                status_code=500, detail=f"Could not persist config; nothing removed: {e}"
            ) from e
        rt._servers.pop(name, None)
        freed = 0
        failed: list[str] = []
        for path in doomed:
            try:
                size = path.stat().st_size
                path.unlink()
                freed += size
            except OSError:
                failed.append(str(path))
        return {
            "removed": name,
            "deleted_files": len(doomed) - len(failed),
            "freed_mb": freed // 1_048_576,
            "failed": failed,
        }
