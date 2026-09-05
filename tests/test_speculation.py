from __future__ import annotations

from arc_llama.config import Config, GPUConfig, ModelConfig
from arc_llama.speculation import discover_drafts


def _model(name: str, path: str) -> ModelConfig:
    return ModelConfig(name=name, path=path, port=18080, gpu_pci_slot="gpu")


def test_discovers_smaller_same_family_draft(tmp_path):
    target_path = tmp_path / "Qwen3-30B-Q4.gguf"
    draft_path = tmp_path / "Qwen3-4B-Q4.gguf"
    other_path = tmp_path / "Llama-3-8B-Q4.gguf"
    target_path.write_bytes(b"x" * (3 * 1024 * 1024))
    draft_path.write_bytes(b"x" * (1024 * 1024))
    other_path.write_bytes(b"x" * 1_000)
    target = _model("qwen3-30b", str(target_path))
    cfg = Config(
        gpus=[GPUConfig(pci_slot="gpu", sycl_index=0, arch="battlemage", vram_mb=24 * 1024)],
        models=[target, _model("qwen3-4b", str(draft_path)), _model("llama-8b", str(other_path))],
    )
    assert [c.name for c in discover_drafts(cfg, target)] == ["qwen3-4b"]


def test_rejects_larger_draft(tmp_path):
    target_path = tmp_path / "Qwen3-4B-Q4.gguf"
    draft_path = tmp_path / "Qwen3-30B-Q4.gguf"
    target_path.write_bytes(b"x" * 1_000)
    draft_path.write_bytes(b"x" * 2_000)
    target = _model("qwen3-4b", str(target_path))
    cfg = Config(models=[target, _model("qwen3-30b", str(draft_path))])
    assert discover_drafts(cfg, target) == []
