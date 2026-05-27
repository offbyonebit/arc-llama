"""Tests for arc_llama.recipes — VRAM math, recipe generation, KV sizing."""
from __future__ import annotations

from arc_llama.arch import Arch
from arc_llama.recipes import (
    DEFAULT_CTX_CAP,
    KVCacheType,
    LaunchRecipe,
    default_recipe,
    estimate_kv_bytes,
    suggest_ctx,
)


class TestEstimateKvBytes:
    def test_default_f16(self):
        # 70 KiB/token at f16 for default class
        assert estimate_kv_bytes(1024, KVCacheType.F16, "default") == 1024 * 70 * 1024

    def test_moe_a3b_q8(self):
        # 20 KiB/token f16, q8_0 halves it
        assert estimate_kv_bytes(4096, KVCacheType.Q8_0, "moe_a3b") == 4096 * 20 * 1024 * 0.5

    def test_gemma_swa_f16(self):
        assert estimate_kv_bytes(8192, KVCacheType.F16, "gemma_swa") == 8192 * 16 * 1024

    def test_unknown_class_fallback(self):
        # Unknown kv_class falls back to default f16 per-token
        assert estimate_kv_bytes(1000, KVCacheType.F16, "no_such_class") == 1000 * 70 * 1024


class TestSuggestCtx:
    def test_basic_fit(self):
        # 24 GB VRAM, 4 GB model, q8_0 KV
        ctx = suggest_ctx(
            vram_mb=24 * 1024,
            model_file_mb=4 * 1024,
            kv_type=KVCacheType.Q8_0,
        )
        assert ctx >= 4096
        assert ctx <= DEFAULT_CTX_CAP
        assert ctx % 4096 == 0

    def test_oom_returns_minimum(self):
        # Model bigger than VRAM
        ctx = suggest_ctx(
            vram_mb=1024,
            model_file_mb=2048,
            kv_type=KVCacheType.F16,
        )
        assert ctx == 4096

    def test_ctx_cap_clamps(self):
        # Ridiculous VRAM should still be capped
        ctx = suggest_ctx(
            vram_mb=1024 * 1024,  # 1 TB
            model_file_mb=1,
            kv_type=KVCacheType.Q8_0,
            ctx_cap=131072,
        )
        assert ctx == 131072

    def test_f32_doubles_bytes(self):
        # Tight VRAM so the difference is visible below the cap
        f16_ctx = suggest_ctx(
            vram_mb=8 * 1024,
            model_file_mb=6 * 1024,
            kv_type=KVCacheType.F16,
            ctx_cap=1_000_000,
        )
        f32_ctx = suggest_ctx(
            vram_mb=8 * 1024,
            model_file_mb=6 * 1024,
            kv_type=KVCacheType.F32,
            ctx_cap=1_000_000,
        )
        assert f32_ctx < f16_ctx

    def test_q4_saves_more_than_q8(self):
        # Tight VRAM so the difference is visible
        q8_ctx = suggest_ctx(
            vram_mb=8 * 1024,
            model_file_mb=6 * 1024,
            kv_type=KVCacheType.Q8_0,
            ctx_cap=1_000_000,
        )
        q4_ctx = suggest_ctx(
            vram_mb=8 * 1024,
            model_file_mb=6 * 1024,
            kv_type=KVCacheType.Q4_0,
            ctx_cap=1_000_000,
        )
        assert q4_ctx > q8_ctx


class TestDefaultRecipe:
    def test_battlemage_prefers_q8(self):
        r = default_recipe(
            Arch.BATTLEMAGE,
            vram_mb=24 * 1024,
            model_file_mb=4 * 1024,
        )
        assert r.cache_type_k == KVCacheType.Q8_0
        assert r.cache_type_v == KVCacheType.Q8_0
        assert r.n_gpu_layers == 999

    def test_unknown_arch_is_conservative(self):
        r = default_recipe(
            Arch.UNKNOWN,
            vram_mb=8 * 1024,
            model_file_mb=4 * 1024,
        )
        assert r.ctx >= 4096
        assert r.cache_type_k == KVCacheType.Q8_0

    def test_no_prefer_q8_gives_f16(self):
        r = default_recipe(
            Arch.BATTLEMAGE,
            vram_mb=24 * 1024,
            model_file_mb=4 * 1024,
            prefer_q8_kv=False,
        )
        assert r.cache_type_k == KVCacheType.F16
        assert r.cache_type_v == KVCacheType.F16

    def test_moe_class_increases_ctx(self):
        # MoE has smaller per-token KV, so same VRAM → larger ctx
        dense = default_recipe(
            Arch.BATTLEMAGE,
            vram_mb=8 * 1024,
            model_file_mb=6 * 1024,
            kv_class="default",
            prefer_q8_kv=False,
        )
        moe = default_recipe(
            Arch.BATTLEMAGE,
            vram_mb=8 * 1024,
            model_file_mb=6 * 1024,
            kv_class="moe_a3b",
            prefer_q8_kv=False,
        )
        assert moe.ctx > dense.ctx


class TestLaunchRecipeArgv:
    def test_all_fields_present(self):
        r = LaunchRecipe(
            n_gpu_layers=999,
            ctx=32768,
            parallel=2,
            cache_type_k=KVCacheType.Q8_0,
            cache_type_v=KVCacheType.Q5_1,
            threads=8,
            temp=0.7,
            top_p=0.9,
            top_k=40,
            extra_flags=["--reasoning", "off"],
        )
        argv = r.to_argv()
        assert argv == [
            "-ngl", "999",
            "-c", "32768",
            "--parallel", "2",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q5_1",
            "-t", "8",
            "--temp", "0.7",
            "--top-p", "0.9",
            "--top-k", "40",
            "--reasoning", "off",
        ]

    def test_optional_fields_omitted(self):
        r = LaunchRecipe()
        argv = r.to_argv()
        assert "-t" not in argv
        assert "--temp" not in argv
        assert "--top-p" not in argv
        assert "--top-k" not in argv
        assert "--spec-type" not in argv
        assert "-ub" not in argv

    def test_spec_type_and_ubatch_size(self):
        r = LaunchRecipe(
            spec_type="draft-mtp",
            ubatch_size=8,
        )
        argv = r.to_argv()
        assert "--spec-type" in argv
        assert argv[argv.index("--spec-type") + 1] == "draft-mtp"
        assert "-ub" in argv
        assert argv[argv.index("-ub") + 1] == "8"
