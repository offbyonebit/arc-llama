"""Safe llama.cpp runtime updates: install beside, verify, then switch.

``arc-llama install-runtime`` replaces the configured binary as soon as it is
downloaded. When upstream llama.cpp renames or removes a flag (0.9.1 shipped
because ``--no-mmap`` disappeared), that leaves every registered model
unable to start. ``update_runtime`` instead:

1. installs the candidate next to the current runtime without touching the
   config;
2. checks that the candidate's ``--help`` lists every capability the
   registered recipes rely on;
3. optionally runs a canary: starts one registered model on the candidate,
   asks for a few tokens, and stops it;
4. switches ``paths.llama_server`` only when every check passes, recording
   the previous binary so ``rollback_runtime`` can restore it.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from arc_llama.config import Config, ModelConfig
from arc_llama.server_caps import ServerCaps, probe_server_caps

log = logging.getLogger("arc_llama.runtime_update")

HISTORY_FILE = "history.json"
_HISTORY_LIMIT = 10
CANARY_PROMPT = "Reply with the single word OK."
CANARY_TIMEOUT_SECONDS = 180.0


@dataclass
class CanaryResult:
    ok: bool
    detail: str
    model: str | None = None
    seconds: float | None = None


@dataclass
class UpdateResult:
    status: str
    """``switched``, ``up_to_date``, ``rejected``, or ``dry_run``."""
    previous: str
    candidate: str | None = None
    tag: str | None = None
    problems: list[str] = field(default_factory=list)
    canary: CanaryResult | None = None

    @property
    def switched(self) -> bool:
        return self.status == "switched"


def _history_path(cfg: Config) -> Path:
    return Path(cfg.paths.state_dir).expanduser() / "runtime" / HISTORY_FILE


def read_history(cfg: Config) -> list[dict[str, Any]]:
    try:
        data = json.loads(_history_path(cfg).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [entry for entry in data if isinstance(entry, dict)] if isinstance(data, list) else []


def _write_history(cfg: Config, history: list[dict[str, Any]]) -> None:
    path = _history_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(history[-_HISTORY_LIMIT:], indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def required_capabilities(cfg: Config) -> dict[str, list[str]]:
    """Map each capability the registered recipes need to the models needing it."""
    needs: dict[str, list[str]] = {}

    def need(cap: str, model: ModelConfig) -> None:
        needs.setdefault(cap, []).append(model.name)

    for model in cfg.models:
        recipe = model.recipe or {}
        if recipe.get("mmproj"):
            need("supports_mmproj", model)
        if recipe.get("reranking"):
            need("supports_reranking", model)
        if recipe.get("tensor_split"):
            need("supports_tensor_split", model)
        if recipe.get("spec_type"):
            need("supports_speculative", model)
        if recipe.get("spec_draft_model") or recipe.get("spec_draft_name"):
            need("supports_draft_model", model)
        if recipe.get("flash_attn") in ("on", "off", "auto"):
            need("supports_flash_attn", model)
    return needs


_CAPABILITY_LABELS = {
    "supports_mmproj": "--mmproj (vision projectors)",
    "supports_reranking": "--reranking",
    "supports_tensor_split": "--tensor-split",
    "supports_speculative": "--spec-type (speculative decoding)",
    "supports_draft_model": "--spec-draft-model",
    "supports_flash_attn": "--flash-attn",
}


def capability_problems(caps: ServerCaps, cfg: Config) -> list[str]:
    """Human-readable reasons the candidate cannot serve the current recipes."""
    if not caps.probed:
        return ["the candidate's --help did not run, so its flags could not be checked"]
    problems = []
    for cap, models in sorted(required_capabilities(cfg).items()):
        if not getattr(caps, cap, False):
            shown = ", ".join(sorted(models)[:4]) + (" ..." if len(models) > 4 else "")
            problems.append(f"missing {_CAPABILITY_LABELS.get(cap, cap)}, used by {shown}")
    return problems


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def pick_canary_model(cfg: Config, name: str | None = None) -> ModelConfig | None:
    """The named model, or the smallest registered model whose file exists."""
    if name:
        return cfg.find_model(name)
    present = []
    for model in cfg.models:
        try:
            present.append((Path(model.path).stat().st_size, model.name, model))
        except OSError:
            continue
    if not present:
        return None
    return min(present, key=lambda item: (item[0], item[1]))[2]


async def run_canary(
    cfg: Config, binary: Path, model: ModelConfig, *, timeout: float = CANARY_TIMEOUT_SECONDS
) -> CanaryResult:
    """Start ``model`` on ``binary`` at a scratch port and request a few tokens.

    The canary never uses the model's registered port and never touches the
    running service. It honours the single-resident lock, so it refuses to
    start (rather than competing for VRAM) while ``arc-llama serve`` has a
    model loaded.
    """
    from arc_llama.launcher import LlamaServer, build_plan

    gpu = cfg.find_gpu(model.gpu_pci_slot)
    if gpu is None:
        return CanaryResult(False, f"model {model.name} references an unknown GPU", model.name)
    trial_cfg = copy.deepcopy(cfg)
    trial_cfg.paths.llama_server = str(binary)
    trial_model = copy.deepcopy(model)
    trial_model.port = _free_port()
    # A short context keeps the canary quick and small in VRAM.
    trial_model.recipe = {**(trial_model.recipe or {}), "ctx": 2048, "parallel": 1}
    plan = build_plan(trial_cfg, trial_model, gpu, host="127.0.0.1")
    server = LlamaServer(plan, name=f"canary-{model.name}")
    started = time.monotonic()
    log_dir = Path(cfg.paths.state_dir).expanduser() / "runtime" / "canary-logs"
    try:
        try:
            await asyncio.to_thread(server.start, log_dir)
        except OSError as exc:
            return CanaryResult(False, f"could not start: {exc}", model.name)
        if not await server.wait_ready(timeout=timeout):
            tail = server.tail_log(15).strip().splitlines()
            last = tail[-1] if tail else "no log output"
            return CanaryResult(False, f"did not become healthy: {last}", model.name)
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{plan.backend_url}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": CANARY_PROMPT}],
                    "max_tokens": 16,
                    "temperature": 0,
                },
            )
        if response.status_code != 200:
            return CanaryResult(
                False, f"chat request returned HTTP {response.status_code}", model.name
            )
        choice = (response.json().get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = (message.get("content") or message.get("reasoning_content") or "").strip()
        if not text:
            return CanaryResult(False, "chat request returned no text", model.name)
        return CanaryResult(True, f"answered {text[:40]!r}", model.name, time.monotonic() - started)
    except Exception as exc:  # noqa: BLE001 - any failure rejects the candidate
        return CanaryResult(False, f"canary failed: {exc}", model.name)
    finally:
        await server.astop()


Installer = Callable[..., Any]
Canary = Callable[[Config, Path, ModelConfig], Any]


def update_runtime(
    cfg: Config,
    config_path: Path | None,
    *,
    backend: str | None = None,
    version: str = "latest",
    canary_model: str | None = None,
    canary: bool = True,
    dry_run: bool = False,
    installer: Installer | None = None,
    canary_runner: Canary | None = None,
    caps_probe: Callable[[str], ServerCaps] = probe_server_caps,
) -> UpdateResult:
    """Install a candidate runtime beside the current one and switch if it passes."""
    from arc_llama.runtime import install_runtime

    previous = cfg.paths.llama_server
    enabled = [g.backend for g in cfg.gpus if g.enabled]
    previous_backend = enabled[0] if enabled else None
    if backend is None:
        backend = previous_backend or "vulkan"
    install = installer or install_runtime
    installed = install(
        backend=backend,
        version=version,
        cfg=cfg,
        set_default=False,
        config_path=config_path,
    )
    candidate = Path(installed.binary_path)
    result = UpdateResult(
        status="rejected", previous=previous, candidate=str(candidate), tag=installed.tag
    )
    try:
        same = candidate.resolve() == Path(previous).expanduser().resolve()
    except OSError:
        same = False
    if same:
        result.status = "up_to_date"
        return result

    result.problems = capability_problems(caps_probe(str(candidate)), cfg)
    if result.problems:
        return result

    if canary:
        model = pick_canary_model(cfg, canary_model)
        if model is None:
            result.problems.append(
                "no registered model is available for the canary; register one or "
                "pass --no-canary"
            )
            return result
        runner = canary_runner or run_canary
        outcome = runner(cfg, candidate, model)
        if asyncio.iscoroutine(outcome):
            outcome = asyncio.run(outcome)
        result.canary = outcome
        if not outcome.ok:
            result.problems.append(f"canary on {model.name}: {outcome.detail}")
            return result

    if dry_run:
        result.status = "dry_run"
        return result

    history = read_history(cfg)
    history.append(
        {
            "previous": previous,
            "previous_backend": previous_backend,
            "installed": str(candidate),
            "tag": installed.tag,
            "at": time.time(),
        }
    )
    _write_history(cfg, history)
    cfg.paths.llama_server = str(candidate)
    for gpu in cfg.gpus:
        if gpu.enabled:
            gpu.backend = backend
    cfg.save(config_path)
    result.status = "switched"
    return result


def rollback_runtime(cfg: Config, config_path: Path | None) -> tuple[str, str] | None:
    """Restore the runtime that the most recent update replaced.

    Returns ``(from, to)`` or None when there is nothing to roll back to or the
    previous binary no longer exists.
    """
    history = read_history(cfg)
    while history:
        entry = history.pop()
        previous = str(entry.get("previous") or "")
        if not previous:
            continue
        resolved = Path(previous).expanduser()
        if previous != "llama-server" and not resolved.exists():
            log.warning("previous runtime %s no longer exists; skipping", previous)
            continue
        current = cfg.paths.llama_server
        cfg.paths.llama_server = previous
        previous_backend = entry.get("previous_backend")
        if previous_backend in ("vulkan", "sycl"):
            for gpu in cfg.gpus:
                if gpu.enabled:
                    gpu.backend = previous_backend
        cfg.save(config_path)
        _write_history(cfg, history)
        return current, previous
    _write_history(cfg, history)
    return None
