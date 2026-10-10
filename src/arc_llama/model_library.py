"""Model library: Hugging Face search with fit badges, downloads, disk usage.

Backs the dashboard's library panel. Network access goes through
``huggingface_hub`` only when the user searches or downloads; nothing here
runs at startup.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import stat
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from arc_llama.config import Config, ModelConfig
from arc_llama.gguf_meta import gguf_shards, split_info
from arc_llama.models import is_auxiliary_gguf, is_mmproj_gguf
from arc_llama.recipes import KVCacheType, estimate_kv_bytes

log = logging.getLogger("arc_llama.model_library")

_QUANT_RE = re.compile(r"(UD-)?(IQ\d[A-Z0-9_]*|Q\d[A-Z0-9_]*|BF16|F16|F32)", re.IGNORECASE)
_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
# Allowances matching the router's admission estimate.
_COMPUTE_MB = 768
_SAFETY_MB = 256
# Context sizes behind the badges: "fits" leaves room for a 16k chat,
# "tight" only for a 4k one.
FIT_CTX = 16384
TIGHT_CTX = 4096


def quant_label(filename: str) -> str:
    match = _QUANT_RE.search(Path(filename).name)
    return match.group(0).upper() if match else "unknown"


def fit_badge(size_mb: int, vram_mb: int | None) -> str:
    """``fits``, ``tight``, ``too_big``, or ``unknown`` for a model of ``size_mb``.

    Uses the generic KV-per-token constant: the exact figure needs the GGUF
    header, which is not downloaded yet. Registration recomputes the context
    from the real metadata, so the badge is a shopping guide, not a promise.
    """
    if not vram_mb:
        return "unknown"

    def need(ctx: int) -> int:
        kv = estimate_kv_bytes(ctx, KVCacheType.Q8_0) // 1_048_576
        return size_mb + kv + _COMPUTE_MB + _SAFETY_MB

    # "fits" keeps 10% of the card free for the driver, desktop, and
    # estimate error; "tight" only promises a short context.
    if need(FIT_CTX) <= vram_mb * 0.9:
        return "fits"
    if need(TIGHT_CTX) <= vram_mb:
        return "tight"
    return "too_big"


@dataclass
class QuantOption:
    file: str
    """Path to pass to the download (the first shard of a split model)."""
    quant: str
    size_mb: int
    shards: int = 1
    fit: str = "unknown"


def group_repo_files(files: list[tuple[str, int | None]], vram_mb: int | None) -> tuple[list[QuantOption], bool]:
    """Turn a repo's ``(path, size)`` list into downloadable options.

    Split shards fold into their first shard with the summed size; projectors
    and other auxiliary GGUFs are not options but mark the repo as vision.
    Returns ``(options sorted by size, has_vision)``.
    """
    sizes = {name: size or 0 for name, size in files if name.lower().endswith(".gguf")}
    vision = any(is_mmproj_gguf(name) for name in sizes)
    options: list[QuantOption] = []
    for name in sorted(sizes):
        if is_auxiliary_gguf(Path(name).name):
            continue
        shards = [name]
        info = split_info(name)
        if info is not None:
            folder = str(Path(name).parent).replace("\\", "/")
            prefix = "" if folder == "." else folder + "/"
            shards = [prefix + p.name for p in gguf_shards(name)]
        size_mb = sum(sizes.get(s, 0) for s in shards) // 1_048_576
        options.append(
            QuantOption(file=name, quant=quant_label(name), size_mb=size_mb, shards=len(shards))
        )
    for option in options:
        option.fit = fit_badge(option.size_mb, vram_mb) if option.size_mb else "unknown"
    options.sort(key=lambda o: (o.size_mb, o.file))
    return options, vision


def _api(api: Any = None) -> Any:
    if api is not None:
        return api
    from huggingface_hub import HfApi

    return HfApi()


def search_repos(query: str, *, limit: int = 20, api: Any = None) -> list[dict[str, Any]]:
    """GGUF repositories matching ``query``, most downloaded first."""
    results = _api(api).list_models(
        search=query, filter="gguf", sort="downloads", limit=limit
    )
    out = []
    for item in results:
        modified: Any = getattr(item, "last_modified", None)
        out.append(
            {
                "repo": item.id,
                "downloads": getattr(item, "downloads", None),
                "likes": getattr(item, "likes", None),
                "updated": modified.isoformat() if modified is not None else None,
            }
        )
    return out


def repo_options(repo: str, vram_mb: int | None, *, api: Any = None) -> dict[str, Any]:
    if not _REPO_RE.match(repo):
        raise ValueError("repo must look like owner/name")
    info = _api(api).model_info(repo, files_metadata=True)
    files = [(s.rfilename, getattr(s, "size", None)) for s in (info.siblings or [])]
    options, vision = group_repo_files(files, vram_mb)
    return {
        "repo": repo,
        "vision": vision,
        "vram_mb": vram_mb,
        "options": [asdict(o) for o in options],
    }


# -- downloads -----------------------------------------------------------


@dataclass
class DownloadJob:
    id: str
    repo: str
    file: str
    target_dir: str
    status: str = "queued"
    """queued, downloading, registering, done, or error."""
    bytes_done: int = 0
    bytes_total: int | None = None
    error: str | None = None
    registered: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def public(self) -> dict[str, Any]:
        return asdict(self)


def _dir_bytes(path: Path) -> int:
    total = 0
    try:
        for item in path.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


class DownloadManager:
    """Runs Hugging Face downloads one at a time and registers the results.

    One at a time keeps disk and network use predictable and stops two jobs
    racing on the same repo folder. Progress is the folder's size against
    the expected total, polled while the blocking download runs in a thread.
    """

    def __init__(self, cfg: Config, on_registered: Any) -> None:
        self.cfg = cfg
        self.on_registered = on_registered
        self.jobs: dict[str, DownloadJob] = {}
        self._queue: asyncio.Queue[DownloadJob] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    def submit(self, repo: str, file: str, bytes_total: int | None) -> DownloadJob:
        if not _REPO_RE.match(repo) or not file.lower().endswith(".gguf") or ".." in file:
            raise ValueError("invalid repo or file")
        if any(j.repo == repo and j.file == file and j.status not in ("done", "error") for j in self.jobs.values()):
            raise ValueError("that file is already downloading")
        target = Path(self.cfg.paths.models_dir).expanduser() / repo.split("/")[-1]
        job = DownloadJob(
            id=uuid.uuid4().hex[:12],
            repo=repo,
            file=file,
            target_dir=str(target),
            bytes_total=bytes_total,
        )
        self.jobs[job.id] = job
        self._queue.put_nowait(job)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run())
        return job

    async def _run(self) -> None:
        while not self._queue.empty():
            job = self._queue.get_nowait()
            await self._process(job)

    async def _process(self, job: DownloadJob) -> None:
        from arc_llama.models import HFModelSpec, download_from_hf

        target = Path(job.target_dir)
        baseline = _dir_bytes(target)
        job.status = "downloading"
        spec = HFModelSpec(repo=job.repo, file=job.file, quant=None)
        task = asyncio.create_task(
            asyncio.to_thread(download_from_hf, spec, target_dir=target, progress=False)
        )
        try:
            while not task.done():
                job.bytes_done = max(0, _dir_bytes(target) - baseline)
                await asyncio.wait({task}, timeout=1.0)
            path = task.result()
            job.bytes_done = job.bytes_total or job.bytes_done
            job.status = "registering"
            job.registered = await self.on_registered(Path(path))
            job.status = "done"
        except Exception as exc:  # noqa: BLE001 - surface every failure on the job
            job.status = "error"
            job.error = str(exc)[:500]
            log.warning("download %s/%s failed: %s", job.repo, job.file, exc)
        finally:
            job.finished_at = time.time()

    async def shutdown(self) -> None:
        if self._worker is not None and not self._worker.done():
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


# -- disk usage and removal ------------------------------------------------


def model_files(model: ModelConfig) -> list[Path]:
    """Every file a model owns on disk: its shards and its projector."""
    paths = [p for p in gguf_shards(Path(model.path).expanduser()) if p.exists()]
    mmproj = (model.recipe or {}).get("mmproj")
    if mmproj and Path(mmproj).expanduser().exists():
        paths.append(Path(mmproj).expanduser())
    return paths


def disk_report(cfg: Config, last_used: dict[str, float] | None = None) -> dict[str, Any]:
    models_dir = Path(cfg.paths.models_dir).expanduser()
    rows: list[dict[str, Any]] = []
    for model in cfg.models:
        files = model_files(model)
        size = 0
        for path in files:
            try:
                size += path.stat().st_size
            except OSError:
                continue
        rows.append(
            {
                "name": model.name,
                "size_mb": size // 1_048_576,
                "files": len(files),
                "missing": not Path(model.path).expanduser().exists(),
                "managed": _inside(Path(model.path).expanduser(), models_dir),
                "last_used": (last_used or {}).get(model.name),
            }
        )
    rows.sort(key=lambda r: -r["size_mb"])
    free = total = None
    try:
        usage = shutil.disk_usage(models_dir if models_dir.exists() else models_dir.parent)
        free, total = usage.free // 1_048_576, usage.total // 1_048_576
    except OSError:
        pass
    return {"models_dir": str(models_dir), "free_mb": free, "total_mb": total, "models": rows}


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def deletable_files(cfg: Config, model: ModelConfig) -> list[Path]:
    """Files of ``model`` that are safe to delete.

    Only files inside ``paths.models_dir`` (so a registration pointing at a
    user's own folder is never touched) and not shared with another
    registered model (a projector can serve several quants).
    """
    root = Path(cfg.paths.models_dir).expanduser()
    shared: set[Path] = set()
    for other in cfg.models:
        if other.name != model.name:
            shared.update(p.resolve() for p in model_files(other))
    return [
        p for p in model_files(model)
        if _inside(p, root) and p.resolve() not in shared
    ]


def file_readiness(model: ModelConfig) -> dict[str, Any]:
    """Check that configured model files are regular, readable, and non-empty.

    Reads at most one byte from each file. This does not validate GGUF
    contents or test inference.
    """
    primary = Path(model.path).expanduser()
    try:
        split = split_info(primary)
        files = gguf_shards(primary) if split is not None else [primary]
    except (OSError, ValueError):
        files = [primary]
    projector_raw = (model.recipe or {}).get("mmproj")
    if projector_raw:
        files.append(Path(projector_raw).expanduser())

    for path in files:
        try:
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                return {"status": "not_file", "available": False, "detail": f"Not a regular file: {path}"}
            if info.st_size == 0:
                return {"status": "empty", "available": False, "detail": f"File is empty: {path}"}
            with path.open("rb") as source:
                if not source.read(1):
                    return {"status": "empty", "available": False, "detail": f"File is empty: {path}"}
        except FileNotFoundError:
            return {"status": "missing", "available": False, "detail": f"Missing file: {path}"}
        except OSError:
            return {"status": "inaccessible", "available": False, "detail": f"Cannot read file: {path}"}
    return {"status": "available", "available": True, "detail": "Files are present and readable; GGUF contents and inference have not been checked."}
