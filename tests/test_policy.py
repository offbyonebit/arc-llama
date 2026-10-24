"""Tests for launch policy (verified adjustments only)."""

from __future__ import annotations

from arc_llama.arch import Arch, Backend
from arc_llama.policy import apply_launch_policy, needs_flash_attn
from arc_llama.recipes import KVCacheType, LaunchRecipe


class TestNeedsFlashAttn:
    def test_vulkan_q8_needs_fa(self):
        r = LaunchRecipe(cache_type_v=KVCacheType.Q8_0)
        assert needs_flash_attn(Backend.VULKAN, r)

    def test_sycl_q8_does_not_need_fa_inject(self):
        # Production SYCL serves q8 V with no --flash-attn flag.
        r = LaunchRecipe(cache_type_v=KVCacheType.Q8_0)
        assert not needs_flash_attn(Backend.SYCL, r)

    def test_f16_not_required(self):
        r = LaunchRecipe(cache_type_v=KVCacheType.F16)
        assert not needs_flash_attn(Backend.VULKAN, r)
        assert not needs_flash_attn(Backend.SYCL, r)


class TestApplyLaunchPolicy:
    def test_sycl_q8_does_not_inject_flash_attn(self):
        recipe = LaunchRecipe(
            cache_type_k=KVCacheType.Q8_0,
            cache_type_v=KVCacheType.Q8_0,
        )
        out = apply_launch_policy(
            recipe,
            arch=Arch.BATTLEMAGE,
            backend=Backend.SYCL,
            model_path="/m.gguf",
        )
        assert "--flash-attn" not in out.extra_flags
        assert "-fa" not in out.extra_flags

    def test_injects_flash_attn_for_vulkan_q8(self):
        recipe = LaunchRecipe(cache_type_v=KVCacheType.Q8_0)
        out = apply_launch_policy(
            recipe,
            arch=Arch.BATTLEMAGE,
            backend=Backend.VULKAN,
            model_path="/m.gguf",
        )
        assert "--flash-attn" in out.extra_flags
        idx = out.extra_flags.index("--flash-attn")
        assert out.extra_flags[idx + 1] == "on"

    def test_overrides_explicit_fa_off_on_vulkan_q8(self):
        recipe = LaunchRecipe(
            cache_type_v=KVCacheType.Q8_0,
            extra_flags=["--flash-attn", "off"],
        )
        out = apply_launch_policy(
            recipe,
            arch=Arch.BATTLEMAGE,
            backend=Backend.VULKAN,
            model_path="/m.gguf",
        )
        idx = out.extra_flags.index("--flash-attn")
        assert out.extra_flags[idx + 1] == "on"

    def test_sycl_leaves_explicit_fa_off_alone(self):
        # SYCL does not force FA; leave user flags alone.
        recipe = LaunchRecipe(
            cache_type_v=KVCacheType.Q8_0,
            extra_flags=["--flash-attn", "off"],
        )
        out = apply_launch_policy(
            recipe,
            arch=Arch.BATTLEMAGE,
            backend=Backend.SYCL,
            model_path="/m.gguf",
        )
        assert out.extra_flags == ["--flash-attn", "off"]

    def test_does_not_strip_draft_mtp(self):
        recipe = LaunchRecipe(spec_type="draft-mtp")
        out = apply_launch_policy(
            recipe,
            arch=Arch.BATTLEMAGE,
            backend=Backend.SYCL,
            model_path="/m.gguf",
            model_name="hybrid-mtp",
        )
        assert out.spec_type == "draft-mtp"

    def test_f16_leaves_flags_alone(self):
        recipe = LaunchRecipe(
            cache_type_k=KVCacheType.F16,
            cache_type_v=KVCacheType.F16,
        )
        out = apply_launch_policy(
            recipe,
            arch=Arch.BATTLEMAGE,
            backend=Backend.SYCL,
            model_path="/m.gguf",
        )
        assert "--flash-attn" not in out.extra_flags

    def test_warns_high_draft_n_max(self, caplog):
        import logging

        recipe = LaunchRecipe(spec_type="draft-mtp", spec_draft_n_max=6)
        with caplog.at_level(logging.WARNING):
            apply_launch_policy(
                recipe,
                arch=Arch.BATTLEMAGE,
                backend=Backend.SYCL,
                model_path="/m.gguf",
                model_name="m",
            )
        assert "spec_draft_n_max=6" in caplog.text
