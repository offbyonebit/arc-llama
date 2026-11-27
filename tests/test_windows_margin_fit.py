"""A model added on a Windows card must pass the router's own fit guard.

The router budgets a 768 MiB safety margin on Windows. If the add-time
context picker still budgets 256, a freshly added model can be refused with
out_of_memory on its very first launch. The margins are fixed at import
from sys.platform, so each platform is checked in a fresh interpreter.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")

_PROBE = r"""
import asyncio, json, subprocess, signal, sys, shutil, tempfile, ctypes, socket, threading, platform, importlib.metadata, logging, getpass, mimetypes, uuid, locale, selectors
import gguf  # pulls in numpy, which runs OS checks at import, so load it before faking
sys.platform = PLATFORM  # only module-level margin checks read this
import arc_llama.recipes as recipes
import arc_llama.router as router
from arc_llama.arch import Arch
from arc_llama.config import ModelConfig

weight_mb, bpt = 5 * 1024, 64 * 1024
router.estimate_weight_vram_bytes = lambda path, **k: weight_mb * 1_048_576
router.kv_bytes_per_token_f16 = lambda path: bpt
router.gguf_total_bytes = lambda path: weight_mb * 1_048_576
out = []
for vram in (8 * 1024, 10 * 1024, 12 * 1024):
    r = recipes.default_recipe(next(iter(Arch)), vram, weight_mb, f16_bytes_per_token=bpt)
    m = ModelConfig("m", MODEL, 18080, "gpu", recipe={"ctx": r.ctx, "cache_type_k": r.cache_type_k.value})
    need = router._estimate_model_vram_mb(m)
    out.append({"vram": vram, "ctx": r.ctx, "need": need})
print(json.dumps({"margin": router._VRAM_SAFETY_MARGIN_MB, "rows": out}))
"""


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_fresh_recipe_fits_router_guard(platform, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    code = _PROBE.replace("PLATFORM", repr(platform)).replace("MODEL", repr(str(model)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([SRC, os.environ.get("PYTHONPATH", "")]))
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stderr
    data = json.loads(res.stdout.strip().splitlines()[-1])
    assert data["margin"] == (768 if platform == "win32" else 256)
    for row in data["rows"]:
        assert row["need"] <= row["vram"], row
