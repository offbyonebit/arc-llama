"""Admin endpoints for status, load/stop, tuning and model editing."""
from __future__ import annotations

import io
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, UploadFile

from arc_llama.config import Config
from arc_llama.server_routers.common import require_admin

log = logging.getLogger("arc_llama.server")
router = APIRouter()


@router.get("/admin/status", dependencies=[require_admin])
async def admin_status(request: Request) -> dict:
    rt = request.app.state.router
    c: Config = request.app.state.cfg
    mgr = request.app.state.upstream_mgr
    models = []
    for m in rt.all_models():
        srv = rt._servers.get(m.name)
        r = m.recipe or {}
        running = bool(srv and srv.is_running)
        models.append({
            "name": m.name,
            "display_name": m.display_name,
            "path": m.path,
            "gpu_pci_slot": m.gpu_pci_slot,
            "port": m.port,
            "loaded": running,
            "pid": getattr(getattr(srv, "process", None), "pid", None) if running else None,
            "ctx": r.get("ctx"),
            "cache_type_k": r.get("cache_type_k"),
            "cache_type_v": r.get("cache_type_v"),
            "flash_attn": r.get("flash_attn"),
            "ubatch_size": r.get("ubatch_size"),
            "batch_size": r.get("batch_size"),
            "kv_class": m.kv_class,
            "aliases": list(m.aliases),
            "tune_state": m.tune_state,
            "tuned_at": m.tuned_at,
            "tune_error": m.tune_error,
            "tune_fingerprint": m.tune_fingerprint,
        })
    gpus = [{
        "pci_slot": g.pci_slot,
        "sycl_index": g.sycl_index,
        "arch": g.arch,
        "vram_mb": g.vram_mb,
        "name": g.name,
        "enabled": g.enabled,
    } for g in c.gpus]
    return {
        "server": {
            "host": c.server.host,
            "port": c.server.port,
            "single_resident": c.server.single_resident,
            "auto_tune": c.tune.auto,
        },
        "gpus": gpus,
        "models": models,
        "upstreams": mgr.upstreams_status(),
    }


@router.post("/admin/load/{name}", dependencies=[require_admin])
async def admin_load(name: str, request: Request) -> dict:
    rt = request.app.state.router
    mgr = request.app.state.upstream_mgr
    if mgr.find_model(name) is not None:
        raise HTTPException(status_code=400, detail=f"Upstream model cannot be loaded locally: {name!r}")
    try:
        model, srv = await rt.ensure_active(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown model: {name!r}") from None
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"name": model.name, "loaded": srv.is_running}


@router.post("/admin/stop/{name}", dependencies=[require_admin])
async def admin_stop(name: str, request: Request) -> dict:
    rt = request.app.state.router
    mgr = request.app.state.upstream_mgr
    if mgr.find_model(name) is not None:
        raise HTTPException(status_code=400, detail=f"Upstream model cannot be stopped locally: {name!r}")
    if name not in {m.name for m in rt.all_models()}:
        raise HTTPException(status_code=404, detail=f"Unknown model: {name!r}")
    was_running = await rt.stop_one(name)
    return {"name": name, "was_running": was_running, "loaded": False}


@router.post("/admin/stop-all", dependencies=[require_admin])
async def admin_stop_all(request: Request) -> dict:
    rt = request.app.state.router
    stopped = await rt.stop_all()
    return {"stopped": stopped}


@router.get("/admin/tune/status", dependencies=[require_admin])
async def admin_tune_status(request: Request) -> dict[str, Any]:
    """Return the current auto-tuner state and per-model tune status."""
    rt = request.app.state.router
    c: Config = request.app.state.cfg
    tuner = getattr(request.app.state, "tuner", None)
    return {
        "auto_tune": c.tune.auto,
        "idle_seconds": c.tune.idle_seconds,
        "sweep_running": bool(tuner and tuner.is_sweep_running),
        "sweep_model": getattr(tuner, "running_model", None),
        "sweep_stage": getattr(tuner, "running_stage", None),
        "models": [
            {
                "name": m.name,
                "tune_state": m.tune_state,
                "tuned_at": m.tuned_at,
                "tune_error": m.tune_error,
                "tune_fingerprint": m.tune_fingerprint,
            }
            for m in rt.all_models()
        ],
    }


@router.post("/admin/tune/{name}", dependencies=[require_admin])
async def admin_tune_queue(name: str, request: Request) -> dict[str, Any]:
    """Queue a model for immediate tuning when the next idle window arrives."""
    c: Config = request.app.state.cfg
    m = c.find_model(name)
    if m is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {name!r}")
    tuner = getattr(request.app.state, "tuner", None)
    if tuner is None:
        raise HTTPException(status_code=503, detail="Auto-tuner is not running")
    ok = tuner.queue_now(name)
    return {"queued": ok, "name": name}


@router.delete("/admin/tune", dependencies=[require_admin])
async def admin_tune_abort(request: Request) -> dict[str, bool]:
    """Abort the currently running sweep."""
    tuner = getattr(request.app.state, "tuner", None)
    if tuner is None:
        return {"aborted": False}
    aborted = tuner.abort_sweep()
    return {"aborted": aborted}


@router.post("/admin/parse-pdf", dependencies=[require_admin])
async def admin_parse_pdf(
    request: Request,
    file: UploadFile,
) -> dict:
    """Extract text from an uploaded PDF using pypdf."""
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as e:
        raise HTTPException(
            status_code=501,
            detail="PDF parsing is not available; install pypdf: pip install pypdf",
        ) from e

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="PDF must be under 50 MB")
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse PDF: {e}") from e
    finally:
        await file.close()

    return {"filename": file.filename, "text": text}


@router.post("/admin/models/{name}/edit", dependencies=[require_admin])
async def admin_edit_model(name: str, request: Request) -> dict:
    """Update a model's recipe in-place."""
    from arc_llama.config import default_config_path
    from arc_llama.gguf_meta import validate_override_patterns, weight_tensor_table
    from arc_llama.recipes import FLASH_ATTN_VALUES, KVCacheType

    c: Config = request.app.state.cfg
    rt = request.app.state.router
    mgr = request.app.state.upstream_mgr
    if mgr.find_model(name) is not None:
        raise HTTPException(status_code=400, detail=f"Upstream model cannot be edited locally: {name!r}")
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    model = next((m for m in c.models if m.name == name), None)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {name!r}")
    valid_kv = {kv.value for kv in KVCacheType}
    valid_classes = {
        "default", "moe_a3b", "qwen3_dense", "qwen3_27b_dense",
        "qwen2_5", "gemma_swa", "phi4", "llama3", "deepseek_r1_distill",
    }
    recipe = dict(model.recipe or {})
    changed: list[str] = []
    if "ctx" in body:
        try:
            ctx = int(body["ctx"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="ctx must be an integer") from None
        if not (256 <= ctx <= 1_048_576):
            raise HTTPException(status_code=400, detail="ctx must be 256..1048576")
        recipe["ctx"] = ctx
        changed.append("ctx")
    for fld in ("cache_type_k", "cache_type_v"):
        if fld in body:
            v = str(body[fld])
            if v not in valid_kv:
                raise HTTPException(
                    status_code=400,
                    detail=f"{fld} must be one of {sorted(valid_kv)}",
                )
            recipe[fld] = v
            changed.append(fld)
    if "parallel" in body:
        try:
            par = int(body["parallel"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="parallel must be an integer") from None
        if not (1 <= par <= 32):
            raise HTTPException(status_code=400, detail="parallel must be 1..32")
        recipe["parallel"] = par
        changed.append("parallel")
    if "kv_class" in body:
        v = str(body["kv_class"])
        if v not in valid_classes:
            raise HTTPException(
                status_code=400,
                detail=f"kv_class must be one of {sorted(valid_classes)}",
            )
        model.kv_class = v
        changed.append("kv_class")
    if "spec_type" in body:
        v = str(body["spec_type"])
        recipe["spec_type"] = v
        changed.append("spec_type")
    if "ubatch_size" in body:
        try:
            ub = int(body["ubatch_size"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="ubatch_size must be an integer") from None
        if not (1 <= ub <= 4096):
            raise HTTPException(status_code=400, detail="ubatch_size must be 1..4096")
        recipe["ubatch_size"] = ub
        changed.append("ubatch_size")
    if "batch_size" in body:
        try:
            b = int(body["batch_size"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="batch_size must be an integer") from None
        if not (1 <= b <= 8192):
            raise HTTPException(status_code=400, detail="batch_size must be 1..8192")
        recipe["batch_size"] = b
        changed.append("batch_size")
    if "flash_attn" in body:
        v = body["flash_attn"]
        if v is None:
            recipe.pop("flash_attn", None)
        elif str(v) in FLASH_ATTN_VALUES:
            recipe["flash_attn"] = str(v)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"flash_attn must be one of {list(FLASH_ATTN_VALUES)} or null",
            )
        changed.append("flash_attn")
    if "n_cpu_moe" in body:
        v = body["n_cpu_moe"]
        if v is None:
            recipe.pop("n_cpu_moe", None)
        else:
            try:
                n = int(v)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400, detail="n_cpu_moe must be an integer or null"
                ) from None
            if not (0 <= n <= 1024):
                raise HTTPException(
                    status_code=400, detail="n_cpu_moe must be 0..1024 (layers)"
                )
            if n == 0:
                recipe.pop("n_cpu_moe", None)
            else:
                recipe["n_cpu_moe"] = n
                recipe.pop("override_tensor", None)
        changed.append("n_cpu_moe")
    if "override_tensor" in body:
        v = body["override_tensor"]
        if v is None:
            recipe.pop("override_tensor", None)
        else:
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise HTTPException(
                    status_code=400,
                    detail="override_tensor must be a list of strings or null",
                )
            table = weight_tensor_table(model.path)
            ok, err = validate_override_patterns(table, v)
            if not ok:
                raise HTTPException(status_code=400, detail=err)
            recipe["override_tensor"] = list(v)
            recipe.pop("n_cpu_moe", None)
        changed.append("override_tensor")
    if not changed:
        raise HTTPException(status_code=400, detail="no recognised fields to edit")
    model.recipe = recipe
    config_path = getattr(request.app.state, "config_path", None)
    try:
        c.save(config_path or default_config_path())
    except OSError as e:
        log.warning("edit %s: persist failed: %s", name, e)
    rebuilt, was_running = await rt.rebuild_model(name)
    return {
        "name": name,
        "changed": changed,
        "recipe": recipe,
        "kv_class": model.kv_class,
        "stopped_running_instance": was_running,
        "rebuilt": rebuilt,
    }


@router.post("/admin/scan", dependencies=[require_admin])
async def admin_scan(request: Request) -> dict:
    """Re-walk scan paths for new GGUFs and auto-register them."""
    from arc_llama.config import default_config_path
    from arc_llama.models import discover_ggufs, register_discovered

    c: Config = request.app.state.cfg
    rt = request.app.state.router
    try:
        found = discover_ggufs(c)
        added = register_discovered(c, found)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if added:
        try:
            c.save(getattr(request.app.state, "config_path", None) or default_config_path())
        except OSError as e:
            log.warning("scan: persist failed: %s", e)
        rt._build_servers()  # type: ignore[attr-defined]
    return {
        "found": len(found),
        "added": [m.name for m in added],
    }
