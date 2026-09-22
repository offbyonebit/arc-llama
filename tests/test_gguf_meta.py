"""Tests for arc_llama.gguf_meta — MTP head detection from GGUF metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from arc_llama.gguf_meta import (
    estimate_weight_vram_bytes,
    expert_count,
    expert_tensor_bytes_by_layer,
    gguf,
    has_mtp_heads,
    is_hybrid_ssm,
    is_moe,
    kv_bytes_per_token_f16,
    mtp_info,
    read_gguf_meta,
    scan_weight_tensors,
    tokenizer_fingerprint,
    trained_context_length,
)

# Real GGUFs on the host's storage, discovered during exploration.
_BASE_QWEN = Path("/mnt/storage/models/qwen3.6-27b/Qwen_Qwen3.6-27B-Q4_K_M.gguf")
_MTP_QWEN = Path("/mnt/storage/models/qwen3.6-27b/Qwen3.6-27B-MTP-UD-Q4_K_XL.gguf")
_GEMMA_MOE = Path("/mnt/storage/models/gemma-4-26b-a4b/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf")
_QWEN_CODER_MOE = Path(
    "/mnt/storage/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"
)


def _have_fixtures() -> bool:
    return _BASE_QWEN.exists() and _MTP_QWEN.exists()


def _have_moe_fixtures() -> bool:
    return _GEMMA_MOE.exists() and _QWEN_CODER_MOE.exists()


class _FakeField:
    def __init__(self, value: Any):
        self._value = value

    def contents(self) -> Any:
        return self._value


# Distinct from the tensor-table _FakeReader defined further down: this one
# answers get_field() lookups, that one answers .tensors. Same name would
# shadow, and the later definition would silently win.
class _FakeFieldReader:
    def __init__(self, fields: dict[str, Any]):
        self._fields = fields

    def get_field(self, key: str):
        if key not in self._fields:
            return None
        return _FakeField(self._fields[key])


class TestReadGgufMetaExpertCountKeys:
    def _patch_reader(self, monkeypatch: pytest.MonkeyPatch, fields: dict[str, Any]) -> None:
        monkeypatch.setattr(
            "arc_llama.gguf_meta.gguf.GGUFReader",
            lambda _path: _FakeFieldReader(fields),
        )

    def test_reads_arch_prefixed_expert_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        p = tmp_path / "model.gguf"
        p.write_text("")
        self._patch_reader(
            monkeypatch,
            {
                gguf.Keys.General.ARCHITECTURE: "gemma4",
                "gemma4.expert_count": 128,
            },
        )
        meta = read_gguf_meta(p)
        assert meta["expert_count"] == 128

    def test_reads_bare_alternative_expert_count_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        p = tmp_path / "model.gguf"
        p.write_text("")
        self._patch_reader(
            monkeypatch,
            {
                gguf.Keys.General.ARCHITECTURE: "gemma4",
                "num_experts": 64,
            },
        )
        meta = read_gguf_meta(p)
        assert meta["expert_count"] == 64

    def test_reads_arch_prefixed_alternative_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        p = tmp_path / "model.gguf"
        p.write_text("")
        self._patch_reader(
            monkeypatch,
            {
                gguf.Keys.General.ARCHITECTURE: "gemma4",
                "gemma4.moe_expert_count": 32,
            },
        )
        meta = read_gguf_meta(p)
        assert meta["expert_count"] == 32

    def test_skips_zero_and_uses_next_positive_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        p = tmp_path / "model.gguf"
        p.write_text("")
        self._patch_reader(
            monkeypatch,
            {
                gguf.Keys.General.ARCHITECTURE: "gemma4",
                "expert_count": 0,
                "num_experts": 8,
            },
        )
        meta = read_gguf_meta(p)
        assert meta["expert_count"] == 8

    def test_no_expert_count_when_no_keys(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        p = tmp_path / "model.gguf"
        p.write_text("")
        self._patch_reader(
            monkeypatch,
            {
                gguf.Keys.General.ARCHITECTURE: "llama",
            },
        )
        meta = read_gguf_meta(p)
        assert "expert_count" not in meta


class TestTokenizerFingerprint:
    def _fingerprint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fields: dict[str, Any],
    ) -> str | None:
        path = tmp_path / "model.gguf"
        path.write_bytes(b"GGUF")
        monkeypatch.setattr(
            "arc_llama.gguf_meta.gguf.GGUFReader",
            lambda _path: _FakeFieldReader(fields),
        )
        return tokenizer_fingerprint(path)

    def test_same_tokenizer_is_stable(self, tmp_path, monkeypatch):
        fields = {
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.tokens": ["a", "b", "c"],
            "tokenizer.ggml.bos_token_id": 1,
        }
        first = self._fingerprint(tmp_path, monkeypatch, fields)
        second = self._fingerprint(tmp_path, monkeypatch, fields)
        assert first is not None
        assert first == second

    def test_token_id_change_changes_fingerprint(self, tmp_path, monkeypatch):
        first = self._fingerprint(
            tmp_path,
            monkeypatch,
            {"tokenizer.ggml.tokens": ["a", "b"]},
        )
        second = self._fingerprint(
            tmp_path,
            monkeypatch,
            {"tokenizer.ggml.tokens": ["b", "a"]},
        )
        assert first is not None
        assert first != second

    def test_requires_vocabulary_metadata(self, tmp_path, monkeypatch):
        assert (
            self._fingerprint(
                tmp_path,
                monkeypatch,
                {"tokenizer.ggml.model": "gpt2"},
            )
            is None
        )


class TestReadGgufMeta:
    @pytest.mark.skipif(not _have_fixtures(), reason="fixture GGUFs not on disk")
    def test_reads_architecture(self):
        meta = read_gguf_meta(_BASE_QWEN)
        assert meta["architecture"] == "qwen35"

    @pytest.mark.skipif(not _have_fixtures(), reason="fixture GGUFs not on disk")
    def test_reads_nextn_and_layers(self):
        meta = read_gguf_meta(_MTP_QWEN)
        assert meta["nextn_predict_layers"] == 1
        assert meta["block_count"] == 65

    def test_missing_file_returns_empty(self):
        meta = read_gguf_meta("/nonexistent/file.gguf")
        assert meta == {}


class TestKvBytesPerToken:
    def _estimate(self, tmp_path, monkeypatch, meta):
        path = tmp_path / "kv-model.gguf"
        path.write_bytes(b"GGUF")
        monkeypatch.setattr("arc_llama.gguf_meta.read_gguf_meta", lambda _path: meta)
        return kv_bytes_per_token_f16(path)

    def test_scalar_transformer_geometry(self, tmp_path, monkeypatch):
        # 32 layers × 8 KV heads × (128 K + 128 V) × 2-byte f16.
        result = self._estimate(
            tmp_path,
            monkeypatch,
            {
                "block_count": 32,
                "embedding_length": 4096,
                "attention.head_count": 32,
                "attention.head_count_kv": 8,
            },
        )
        assert result == 32 * 8 * (128 + 128) * 2

    def test_hybrid_per_layer_geometry_skips_recurrent_layers(self, tmp_path, monkeypatch):
        result = self._estimate(
            tmp_path,
            monkeypatch,
            {
                "block_count": 4,
                "embedding_length": 2048,
                "attention.head_count": 32,
                "attention.head_count_kv": [0, 8, 0, 8],
            },
        )
        assert result == 2 * 8 * (64 + 64) * 2

    def test_explicit_key_and_value_lengths_win(self, tmp_path, monkeypatch):
        result = self._estimate(
            tmp_path,
            monkeypatch,
            {
                "block_count": 2,
                "attention.head_count_kv": 4,
                "attention.key_length": 192,
                "attention.value_length": 128,
            },
        )
        assert result == 2 * 4 * (192 + 128) * 2

    def test_sliding_window_uses_family_fallback(self, tmp_path, monkeypatch):
        result = self._estimate(
            tmp_path,
            monkeypatch,
            {
                "block_count": 2,
                "embedding_length": 2048,
                "attention.head_count": 32,
                "attention.head_count_kv": 8,
                "attention.sliding_window": 4096,
            },
        )
        assert result is None

    def test_qwen35_scalar_heads_use_hybrid_fallback(self, tmp_path, monkeypatch):
        result = self._estimate(
            tmp_path,
            monkeypatch,
            {
                "architecture": "qwen35",
                "block_count": 65,
                "embedding_length": 5120,
                "attention.head_count": 24,
                "attention.head_count_kv": 4,
                "attention.key_length": 256,
                "attention.value_length": 256,
            },
        )
        assert result is None

    def test_incomplete_geometry_uses_family_fallback(self, tmp_path, monkeypatch):
        assert self._estimate(tmp_path, monkeypatch, {"block_count": 32}) is None


class TestMtpDetection:
    @pytest.mark.skipif(not _have_fixtures(), reason="fixture GGUFs not on disk")
    def test_base_qwen_has_no_mtp(self):
        assert has_mtp_heads(_BASE_QWEN) is False

    @pytest.mark.skipif(not _have_fixtures(), reason="fixture GGUFs not on disk")
    def test_mtp_qwen_has_mtp(self):
        assert has_mtp_heads(_MTP_QWEN) is True

    def test_missing_file_is_false(self):
        assert has_mtp_heads("/nonexistent.gguf") is False


class TestHybridSsmDetection:
    @pytest.mark.skipif(not _have_fixtures(), reason="fixture GGUFs not on disk")
    def test_qwen_is_hybrid_ssm(self):
        assert is_hybrid_ssm(_BASE_QWEN) is True
        assert is_hybrid_ssm(_MTP_QWEN) is True

    def test_missing_file_is_false(self):
        assert is_hybrid_ssm("/nonexistent.gguf") is False


class TestMtpInfo:
    @pytest.mark.skipif(not _have_fixtures(), reason="fixture GGUFs not on disk")
    def test_summary_keys(self):
        info = mtp_info(_MTP_QWEN)
        assert info["has_mtp_heads"] is True
        assert info["is_hybrid_ssm"] is True
        assert info["nextn_predict_layers"] == 1


class TestTrainedContextLength:
    def test_missing_file_returns_none(self):
        assert trained_context_length("/nonexistent.gguf") is None

    def test_reads_arch_specific_context_length(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "arc_llama.gguf_meta.read_gguf_meta",
            lambda _path: {"architecture": "qwen2", "qwen2.context_length": 32768},
        )
        assert trained_context_length("/fake.gguf") == 32768

    def test_falls_back_to_bare_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "arc_llama.gguf_meta.read_gguf_meta",
            lambda _path: {"architecture": "qwen2", "context_length": 32768},
        )
        assert trained_context_length("/fake.gguf") == 32768

    def test_returns_none_when_unknown(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "arc_llama.gguf_meta.read_gguf_meta",
            lambda _path: {"architecture": "qwen2"},
        )
        assert trained_context_length("/fake.gguf") is None

    def test_ignores_non_positive_values(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "arc_llama.gguf_meta.read_gguf_meta",
            lambda _path: {"architecture": "qwen2", "qwen2.context_length": 0},
        )
        assert trained_context_length("/fake.gguf") is None


class TestMoeDetection:
    def test_missing_file_is_not_moe(self):
        assert is_moe("/nonexistent.gguf") is False
        assert expert_count("/nonexistent.gguf") is None

    def test_detects_moe_from_architecture_prefix(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "arc_llama.gguf_meta.read_gguf_meta",
            lambda _path: {"architecture": "qwen2moe"},
        )
        assert is_moe("/fake.gguf") is True

    def test_expert_count_reads_arch_specific_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "arc_llama.gguf_meta.read_gguf_meta",
            lambda _path: {"architecture": "qwen2moe", "qwen2moe.expert_count": 64},
        )
        assert expert_count("/fake.gguf") == 64

    def test_expert_count_falls_back_to_generic_keys(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "arc_llama.gguf_meta.read_gguf_meta",
            lambda _path: {"architecture": "llama4", "expert_count": 16},
        )
        assert expert_count("/fake.gguf") == 16

    def test_expert_count_returns_none_when_unknown(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "arc_llama.gguf_meta.read_gguf_meta",
            lambda _path: {"architecture": "llama"},
        )
        assert expert_count("/fake.gguf") is None

    @pytest.mark.skipif(not _have_fixtures(), reason="fixture GGUFs not on disk")
    def test_dense_qwen_is_not_moe(self):
        # Regression guard: qwen35 (dense) must not be misdetected as MoE.
        assert is_moe(_BASE_QWEN) is False
        assert expert_count(_BASE_QWEN) is None

    @pytest.mark.skipif(not _have_moe_fixtures(), reason="MoE fixture GGUFs not on disk")
    def test_gemma_moe_detected_via_expert_count_not_arch_prefix(self):
        # gemma4 is deliberately excluded from _MOE_ARCH_PREFIXES since dense
        # and MoE Gemma-4 GGUFs share the same architecture string -- this
        # must be detected via the expert_count metadata field instead.
        meta = read_gguf_meta(_GEMMA_MOE)
        assert meta["architecture"] == "gemma4"
        assert is_moe(_GEMMA_MOE) is True
        assert expert_count(_GEMMA_MOE) == 128

    @pytest.mark.skipif(not _have_moe_fixtures(), reason="MoE fixture GGUFs not on disk")
    def test_qwen3_coder_moe_detected_via_arch_prefix(self):
        meta = read_gguf_meta(_QWEN_CODER_MOE)
        assert meta["architecture"] == "qwen3moe"
        assert is_moe(_QWEN_CODER_MOE) is True
        assert expert_count(_QWEN_CODER_MOE) == 128


# ---------------------------------------------------------------------------
# MoE expert offload accounting (--n-cpu-moe N = N layers of expert tensors)
# ---------------------------------------------------------------------------


class _FakeTensor:
    def __init__(self, name: str, n_bytes: int):
        self.name = name
        self.n_bytes = n_bytes


class _FakeField:
    def __init__(self, value: str):
        self._value = value

    def contents(self) -> str:
        return self._value


class _FakeReader:
    """Stands in for gguf.GGUFReader: a tensor table plus an arch field."""

    def __init__(self, tensors: list[_FakeTensor], arch: str):
        self._tensors = tensors
        self._arch = arch

    @property
    def tensors(self) -> list[_FakeTensor]:
        return self._tensors

    def get_field(self, key: str):
        if key == "general.architecture":
            return _FakeField(self._arch)
        return None


def _patch_reader(monkeypatch: pytest.MonkeyPatch, tensors, arch: str = "qwen3moe"):
    monkeypatch.setattr(
        "arc_llama.gguf_meta.gguf.GGUFReader",
        lambda _path: _FakeReader(tensors, arch),
    )


def _moe_tensors() -> list[_FakeTensor]:
    return [
        _FakeTensor("token_embd.weight", 1000),
        _FakeTensor("blk.0.attn_q.weight", 500),
        _FakeTensor("blk.0.ffn_up_exps.weight", 300),
        _FakeTensor("blk.0.ffn_down_exps.weight", 300),
        _FakeTensor("blk.0.ffn_up_shexp.weight", 200),  # shared expert: stays on GPU
        _FakeTensor("blk.0.ffn_gate_inp.weight", 50),  # router gate: stays on GPU
        _FakeTensor("blk.1.ffn_gate_exps.weight", 400),
        _FakeTensor("blk.1.ffn_up_exps.weight", 400),
        _FakeTensor("blk.1.ffn_down_exps.weight", 400),
    ]


def _fused_moe_tensors() -> list[_FakeTensor]:
    """Real tensor names from gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf.

    Gemma fuses the gate and up projections into a single `ffn_gate_up_exps`
    tensor. An enumerated {gate,up,down}_exps pattern misses it and counts
    only the down projection, which under-reported expert bytes by ~2.7x on
    real hardware.
    """
    return [
        _FakeTensor("token_embd.weight", 1000),
        _FakeTensor("blk.0.attn_q.weight", 100),
        _FakeTensor("blk.0.ffn_gate_up_exps.weight", 800),  # fused: must count
        _FakeTensor("blk.0.ffn_down_exps.weight", 400),
        _FakeTensor("blk.0.ffn_down_exps.scale", 20),  # quant metadata: counts
        _FakeTensor("blk.0.ffn_gate_inp.weight", 30),  # router: excluded
        _FakeTensor("blk.0.ffn_up_shexp.weight", 60),  # shared expert: excluded
    ]


class TestFusedExpertProjections:
    def test_fused_gate_up_tensor_is_counted(self, tmp_path, monkeypatch):
        f = tmp_path / "gemma.gguf"
        f.write_bytes(b"x")
        _patch_reader(monkeypatch, _fused_moe_tensors(), arch="gemma3moe")
        _total, by_layer = scan_weight_tensors(f)
        # 800 (fused gate+up) + 400 (down) + 20 (scale); router and shexp excluded.
        assert by_layer == {0: 1220}

    def test_unfused_layout_still_counted(self, tmp_path, monkeypatch):
        f = tmp_path / "qwen.gguf"
        f.write_bytes(b"x")
        _patch_reader(monkeypatch, _moe_tensors())
        _total, by_layer = scan_weight_tensors(f)
        assert by_layer == {0: 600, 1: 1200}

    def test_implausibly_small_expert_share_warns(self, tmp_path, monkeypatch, caplog):
        """The guard that would have caught the fused-projection bug."""
        f = tmp_path / "odd.gguf"
        f.write_bytes(b"x")
        tensors = [
            _FakeTensor("token_embd.weight", 9000),
            _FakeTensor("blk.0.ffn_down_exps.weight", 100),
            _FakeTensor("blk.0.ffn_someNewFusion.weight", 900),
        ]
        _patch_reader(monkeypatch, tensors)
        with caplog.at_level("WARNING"):
            scan_weight_tensors(f)
        assert "implausibly low" in caplog.text

    def test_healthy_moe_does_not_warn(self, tmp_path, monkeypatch, caplog):
        f = tmp_path / "gemma.gguf"
        f.write_bytes(b"x")
        _patch_reader(monkeypatch, _fused_moe_tensors(), arch="gemma3moe")
        with caplog.at_level("WARNING"):
            scan_weight_tensors(f)
        assert "implausibly low" not in caplog.text


class TestOffloadAccounting:
    def test_scan_returns_total_and_per_layer_experts(self, tmp_path, monkeypatch):
        f = tmp_path / "m.gguf"
        f.write_bytes(b"x")
        _patch_reader(monkeypatch, _moe_tensors())
        total, by_layer = scan_weight_tensors(f)
        assert total == 1000 + 500 + 300 + 300 + 200 + 50 + 400 * 3
        assert by_layer == {0: 600, 1: 1200}

    def test_estimate_subtracts_offloaded_layers(self, tmp_path, monkeypatch):
        f = tmp_path / "m.gguf"
        f.write_bytes(b"x")
        _patch_reader(monkeypatch, _moe_tensors())
        total = 1000 + 500 + 300 + 300 + 200 + 50 + 400 * 3
        assert estimate_weight_vram_bytes(f) == total
        assert estimate_weight_vram_bytes(f, n_cpu_moe=0) == total
        # N=1 offloads layer 0's routed experts only (600), not the shared
        # expert (200) and not the router gate (50).
        assert estimate_weight_vram_bytes(f, n_cpu_moe=1) == total - 600
        assert estimate_weight_vram_bytes(f, n_cpu_moe=2) == total - 600 - 1200
        # N beyond the layer count offloads everything MoE.
        assert estimate_weight_vram_bytes(f, n_cpu_moe=99) == total - 600 - 1200

    def test_dense_model_with_offload_flag_is_unchanged(self, tmp_path, monkeypatch):
        """A non-MoE model has no expert tensors; the flag is inert and the
        full-weight count is correct (byte-for-byte unchanged)."""
        f = tmp_path / "m.gguf"
        f.write_bytes(b"x")
        _patch_reader(
            monkeypatch,
            [_FakeTensor("token_embd.weight", 1000), _FakeTensor("blk.0.ffn_up.weight", 500)],
            arch="llama",
        )
        assert estimate_weight_vram_bytes(f, n_cpu_moe=4) == 1500

    def test_moe_without_recognisable_expert_tensors_returns_none(self, tmp_path, monkeypatch):
        """MoE arch but tensor names we don't recognise: the offloaded bytes
        are unknown, so the estimate must be None — callers must not fall
        back to counting full weights and refusing the load."""
        f = tmp_path / "m.gguf"
        f.write_bytes(b"x")
        _patch_reader(
            monkeypatch,
            [_FakeTensor("token_embd.weight", 1000), _FakeTensor("blk.0.moe_stuff.weight", 500)],
            arch="qwen3moe",
        )
        assert estimate_weight_vram_bytes(f, n_cpu_moe=4) is None
        # Without offload requested the full count is still fine.
        assert estimate_weight_vram_bytes(f) == 1500

    def test_unreadable_file_returns_none(self):
        assert estimate_weight_vram_bytes("/nonexistent.gguf", n_cpu_moe=4) is None
        assert scan_weight_tensors("/nonexistent.gguf") is None
        assert expert_tensor_bytes_by_layer("/nonexistent.gguf") is None
