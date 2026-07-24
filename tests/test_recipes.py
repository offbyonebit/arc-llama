"""Tests for arc_llama.recipes — VRAM math, recipe generation, KV sizing."""
from __future__ import annotations

from unittest.mock import patch

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

    def test_sycl_q8_does_not_force_flash_attn(self):
        from arc_llama.arch import Backend

        # Production SYCL: q8 V without --flash-attn is fine.
        r = default_recipe(
            Arch.BATTLEMAGE,
            vram_mb=24 * 1024,
            model_file_mb=4 * 1024,
            backend=Backend.SYCL,
        )
        assert r.cache_type_k == KVCacheType.Q8_0
        assert r.cache_type_v == KVCacheType.Q8_0
        assert "--flash-attn" not in r.extra_flags

    def test_vulkan_q8_includes_flash_attn_on(self):
        from arc_llama.arch import Backend

        r = default_recipe(
            Arch.BATTLEMAGE,
            vram_mb=24 * 1024,
            model_file_mb=4 * 1024,
            backend=Backend.VULKAN,
        )
        assert r.cache_type_k == KVCacheType.Q8_0
        assert r.cache_type_v == KVCacheType.Q8_0
        assert "--flash-attn" in r.extra_flags
        assert "on" in r.extra_flags

    def test_big_card_bumps_ubatch(self):
        # >=16 GB VRAM: default up from llama.cpp's stock 512 for prompt speed.
        r = default_recipe(
            Arch.BATTLEMAGE,
            vram_mb=24 * 1024,
            model_file_mb=4 * 1024,
        )
        assert r.ubatch_size == 1024

    def test_small_card_keeps_default_ubatch(self):
        # 8 GB card: leave ubatch unset so a fitting model doesn't OOM on the
        # bigger compute buffer.
        r = default_recipe(
            Arch.BATTLEMAGE,
            vram_mb=8 * 1024,
            model_file_mb=4 * 1024,
        )
        assert r.ubatch_size is None

    def test_vulkan_f16_when_profile_disallows_q8(self):
        from arc_llama.arch import ArchProfile, Backend

        profile = ArchProfile(
            arch=Arch.BATTLEMAGE,
            display_name="Test",
            sycl_env={},
            safe_kv_q8=True,
            safe_kv_q8_vulkan=False,
        )
        with patch("arc_llama.recipes.profile_for", return_value=profile):
            r = default_recipe(
                Arch.BATTLEMAGE,
                vram_mb=24 * 1024,
                model_file_mb=4 * 1024,
                backend=Backend.VULKAN,
            )
        assert r.cache_type_k == KVCacheType.F16
        assert r.cache_type_v == KVCacheType.F16
        assert "--flash-attn" not in r.extra_flags


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

    def test_spec_draft_model_and_ngl_emitted(self):
        r = LaunchRecipe(
            spec_type="draft-mtp",
            spec_draft_model="/models/mtp-gemma.gguf",
            spec_draft_ngl=999,
            spec_draft_n_max=3,
        )
        argv = r.to_argv()
        assert argv[argv.index("--spec-draft-model") + 1] == "/models/mtp-gemma.gguf"
        assert argv[argv.index("--spec-draft-ngl") + 1] == "999"
        assert argv[argv.index("--spec-draft-n-max") + 1] == "3"

    def test_spec_draft_model_omitted_when_none(self):
        r = LaunchRecipe(spec_type="draft-mtp")
        argv = r.to_argv()
        assert "--spec-draft-model" not in argv
        assert "--spec-draft-ngl" not in argv

    def test_n_cpu_moe_emitted_when_set(self):
        r = LaunchRecipe(n_cpu_moe=4)
        argv = r.to_argv()
        assert "--n-cpu-moe" in argv
        assert argv[argv.index("--n-cpu-moe") + 1] == "4"

    def test_n_cpu_moe_omitted_when_none(self):
        r = LaunchRecipe()
        argv = r.to_argv()
        assert "--n-cpu-moe" not in argv


class TestLaunchRecipePerfArgv:
    def test_batch_size_emitted(self):
        argv = LaunchRecipe(batch_size=2048).to_argv()
        assert argv[argv.index("-b") + 1] == "2048"

    def test_flash_attn_new_style_takes_value(self):
        for v in ("on", "off", "auto"):
            argv = LaunchRecipe(flash_attn=v).to_argv(fa_takes_value=True)
            assert argv[argv.index("-fa") + 1] == v

    def test_flash_attn_old_style_boolean(self):
        # Old builds: -fa is a boolean; only 'on' emits it, bare.
        argv = LaunchRecipe(flash_attn="on").to_argv(fa_takes_value=False)
        assert "-fa" in argv
        idx = argv.index("-fa")
        assert idx == len(argv) - 1 or argv[idx + 1].startswith("-")
        for v in ("off", "auto"):
            assert "-fa" not in LaunchRecipe(flash_attn=v).to_argv(fa_takes_value=False)

    def test_flash_attn_none_omitted(self):
        assert "-fa" not in LaunchRecipe().to_argv()

    def test_invalid_flash_attn_value_omitted(self):
        # A typo'd value in a hand-edited config must not produce a bad argv.
        assert "-fa" not in LaunchRecipe(flash_attn="yes").to_argv()

    def test_mmap_and_mlock(self):
        argv = LaunchRecipe(no_mmap=True, mlock=True).to_argv()
        assert "--no-mmap" in argv
        assert "--mlock" in argv
        argv = LaunchRecipe().to_argv()
        assert "--no-mmap" not in argv
        assert "--mlock" not in argv


class TestPerfDefaults:
    def test_big_card_gets_perf_batching(self):
        from arc_llama.recipes import PERF_BATCH, PERF_UBATCH

        r = default_recipe(Arch.BATTLEMAGE, vram_mb=24 * 1024, model_file_mb=4 * 1024)
        assert r.ubatch_size == PERF_UBATCH
        assert r.batch_size == PERF_BATCH
        assert r.flash_attn == "auto"

    def test_small_card_keeps_stock_batching(self):
        r = default_recipe(Arch.ALCHEMIST, vram_mb=8 * 1024, model_file_mb=4 * 1024)
        assert r.ubatch_size is None
        assert r.batch_size is None
        assert r.flash_attn == "auto"

    def test_perf_batching_reserves_bigger_compute_buffer(self):
        # Same card either side of the threshold: the perf recipe must budget
        # more compute buffer, i.e. never suggest a LARGER ctx than stock.
        big = default_recipe(Arch.BATTLEMAGE, vram_mb=16 * 1024, model_file_mb=14 * 1024)
        small = default_recipe(Arch.BATTLEMAGE, vram_mb=16 * 1024 - 1, model_file_mb=14 * 1024)
        assert big.ctx <= small.ctx


class TestRecipeToDict:
    def test_round_trips_through_model_config(self):
        from arc_llama.config import ModelConfig
        from arc_llama.recipes import recipe_to_dict

        r = default_recipe(Arch.BATTLEMAGE, vram_mb=24 * 1024, model_file_mb=4 * 1024)
        mc = ModelConfig(
            name="m", path="/x.gguf", port=1, gpu_pci_slot="0000:03:00.0",
            recipe=recipe_to_dict(r),
        )
        back = mc.launch_recipe()
        assert back.ctx == r.ctx
        assert back.cache_type_k == r.cache_type_k
        assert back.flash_attn == r.flash_attn
        assert back.ubatch_size == r.ubatch_size
        assert back.batch_size == r.batch_size

    def test_unset_optionals_not_serialised(self):
        from arc_llama.recipes import recipe_to_dict

        d = recipe_to_dict(LaunchRecipe())
        assert "flash_attn" not in d
        assert "ubatch_size" not in d
        assert "batch_size" not in d
        assert "spec_type" not in d
        assert "no_mmap" not in d
        assert "mlock" not in d


class TestXmxSdpaGating:
    """The oneDNN/XMX SDPA path must only be selected when provably available.

    Measured on an Arc Pro B60 whose llama.cpp had GGML_SYCL_DNN=ON but where
    find_package(DNNL) failed (path compiled out): forcing `-fa on` cost ~10-11%
    decode at shallow context and gained nothing on prefill. So anything short
    of a positive detection must keep the previous defaults.
    """

    B60 = dict(arch=Arch.BATTLEMAGE, vram_mb=24 * 1024, model_file_mb=4 * 1024)

    @staticmethod
    def _caps(**kw):
        from arc_llama.binary_caps import SyclCaps

        return SyclCaps(**kw)

    def test_no_binary_path_keeps_previous_defaults(self):
        """Callers that don't pass llama_server must see zero behaviour change."""
        r = default_recipe(**self.B60)
        assert r.cache_type_k is KVCacheType.Q8_0
        assert r.flash_attn == "auto"

    def test_onednn_present_selects_f16_and_fa_on(self):
        with patch(
            "arc_llama.binary_caps.probe_sycl_caps",
            return_value=self._caps(has_onednn_sdpa=True, has_symbols=True, probed=True),
        ):
            r = default_recipe(**self.B60, llama_server="/fake/llama-server")
        assert r.cache_type_k is KVCacheType.F16
        assert r.cache_type_v is KVCacheType.F16
        assert r.flash_attn == "on"

    def test_onednn_absent_keeps_q8_and_does_not_force_fa(self):
        with patch(
            "arc_llama.binary_caps.probe_sycl_caps",
            return_value=self._caps(has_onednn_sdpa=False, has_symbols=True, probed=True),
        ):
            r = default_recipe(**self.B60, llama_server="/fake/llama-server")
        assert r.cache_type_k is KVCacheType.Q8_0
        assert r.flash_attn == "auto"

    def test_unknown_onednn_is_treated_as_absent(self):
        """A stripped binary reports None; guessing 'present' would regress it."""
        with patch(
            "arc_llama.binary_caps.probe_sycl_caps",
            return_value=self._caps(has_onednn_sdpa=None, has_symbols=False, probed=True),
        ):
            r = default_recipe(**self.B60, llama_server="/fake/llama-server")
        assert r.cache_type_k is KVCacheType.Q8_0
        assert r.flash_attn == "auto"

    def test_tight_vram_keeps_q8_even_with_onednn(self):
        """f16 KV doubles KV bytes; not worth it if context collapses."""
        with patch(
            "arc_llama.binary_caps.probe_sycl_caps",
            return_value=self._caps(has_onednn_sdpa=True, has_symbols=True, probed=True),
        ):
            r = default_recipe(
                arch=Arch.BATTLEMAGE,
                vram_mb=8 * 1024,
                model_file_mb=7 * 1024,
                llama_server="/fake/llama-server",
            )
        assert r.cache_type_k is KVCacheType.Q8_0

    def test_alchemist_never_takes_the_xmx_path(self):
        """Xe-HPG has no XMX SDPA measurements; do not extrapolate from Xe2."""
        with patch(
            "arc_llama.binary_caps.probe_sycl_caps",
            return_value=self._caps(has_onednn_sdpa=True, has_symbols=True, probed=True),
        ):
            r = default_recipe(
                arch=Arch.ALCHEMIST,
                vram_mb=16 * 1024,
                model_file_mb=4 * 1024,
                llama_server="/fake/llama-server",
            )
        assert r.cache_type_k is KVCacheType.Q8_0
        assert r.flash_attn == "auto"

    def test_unknown_arch_never_takes_the_xmx_path(self):
        with patch(
            "arc_llama.binary_caps.probe_sycl_caps",
            return_value=self._caps(has_onednn_sdpa=True, has_symbols=True, probed=True),
        ):
            r = default_recipe(
                arch=Arch.UNKNOWN,
                vram_mb=24 * 1024,
                model_file_mb=4 * 1024,
                llama_server="/fake/llama-server",
            )
        assert r.cache_type_k is KVCacheType.Q8_0

    def test_probe_failure_does_not_break_registration(self):
        """A broken/missing binary must never make model registration explode."""
        with patch(
            "arc_llama.binary_caps.probe_sycl_caps",
            side_effect=OSError("boom"),
        ):
            r = default_recipe(**self.B60, llama_server="/fake/llama-server")
        assert r.cache_type_k is KVCacheType.Q8_0
        assert r.flash_attn == "auto"
