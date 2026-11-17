"""Small, once-per-change inference checks; failures warn and never stop a model."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

import httpx

from arc_llama.config import Config, ModelConfig
from arc_llama.model_compatibility import runtime_identity

log = logging.getLogger(__name__)
PROMPTS = (("What is 2 + 2? Reply with only the digit.", "4"),
           ("Repeat only this word: lantern", "lantern"))


def _path(cfg: Config) -> Path:
    return Path(cfg.paths.state_dir).expanduser() / "runtime" / "correctness.json"


def read_checks(cfg: Config) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(_path(cfg).read_text(encoding="utf-8"))
        models = data.get("models", {}) if isinstance(data, dict) else {}
        return {name: value for name, value in models.items() if isinstance(value, dict)} if isinstance(models, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(cfg: Config, records: dict[str, dict[str, Any]]) -> None:
    try:
        path = _path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps({"schema": 1, "models": records}, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)
    except OSError:
        log.warning("Could not save runtime correctness checks", exc_info=True)


def mark_runtime_changed(cfg: Config) -> None:
    records = read_checks(cfg)
    for model in cfg.models:
        records[model.name] = {**records.get(model.name, {}), "pending": True, "status": "pending"}
    _save(cfg, records)


def _binary(cfg: Config, model: ModelConfig) -> str:
    command = str(Path(cfg.model_llama_server(model)).expanduser())
    return shutil.which(command, path=model.env.get("PATH")) or command


def fingerprint(cfg: Config, model: ModelConfig) -> str | None:
    identity = runtime_identity(_binary(cfg, model), env=model.env or None)
    if identity is None:
        return None
    gpu = cfg.find_gpu(model.gpu_pci_slot)
    values = (identity, model.gpu_pci_slot, gpu.backend if gpu else None)
    return hashlib.sha256(repr(values).encode()).hexdigest()


def _build(cfg: Config, model: ModelConfig) -> str:
    from arc_llama.benchmark import llama_cpp_build

    binary = _binary(cfg, model)
    try:
        marker = json.loads((Path(binary).expanduser().resolve().parent / ".arc-llama-runtime.json").read_text(encoding="utf-8"))
        if isinstance(marker, dict) and isinstance(marker.get("tag"), str):
            return str(marker["tag"])
    except (OSError, ValueError):
        pass
    return llama_cpp_build(binary) or "build not reported"


class RuntimeCorrectness:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.records = read_checks(cfg)
        # Existing installations get an observed baseline, not an invented passing result.
        self.initial = {m.name: fingerprint(cfg, m) for m in cfg.models} if cfg.server.runtime_self_check else {}

    def warning(self, name: str) -> dict[str, Any] | None:
        record = self.records.get(name, {})
        return record if record.get("status") == "warning" else None

    async def after_load(self, model: ModelConfig, backend_url: str) -> None:
        recipe = model.recipe or {}
        if not self.cfg.server.runtime_self_check or recipe.get("embedding") or recipe.get("reranking"):
            return
        current = await asyncio.to_thread(fingerprint, self.cfg, model)
        if current is None:
            return
        self.records.update(read_checks(self.cfg))
        previous = self.records.get(model.name, {})
        old = previous.get("fingerprint") or self.initial.get(model.name)
        if not previous.get("pending") and (old is None or old == current):
            if not previous:
                self.records[model.name] = {"fingerprint": current, "status": "baseline"}
                _save(self.cfg, self.records)
            return
        build = "build not reported"
        try:
            build = await asyncio.wait_for(asyncio.to_thread(_build, self.cfg, model), 5)
            failures = await asyncio.wait_for(self._check(model, backend_url), 12)
            detail = "; ".join(failures)
        except Exception as exc:
            failures = ["check could not complete"]
            detail = f"Check could not complete ({type(exc).__name__})."
        self.records[model.name] = {"fingerprint": current, "pending": False,
                                    "status": "warning" if failures else "passed", "build": build,
                                    "detail": detail, "checked_at": time.time()}
        _save(self.cfg, self.records)
        if failures:
            log.warning("Correctness check for %s on %s: %s. Model remains running; consider arc-llama runtime rollback.", model.name, build, detail)

    async def _check(self, model: ModelConfig, backend_url: str) -> list[str]:
        failures = []
        async with httpx.AsyncClient(timeout=5, follow_redirects=False) as client:
            for prompt, expected in PROMPTS:
                response = await client.post(backend_url + "/v1/chat/completions", json={
                    "model": model.name, "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0, "max_tokens": 32, "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                })
                response.raise_for_status()
                content = response.json()["choices"][0]["message"].get("content") or ""
                text = re.sub(r"<think>.*?(?:</think>|$)", "", content, flags=re.S | re.I)
                if not re.search(r"\b" + re.escape(expected) + r"\b", text, flags=re.I):
                    failures.append(f"Expected {expected!r} in a short answer")
        return failures
