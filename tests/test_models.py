from __future__ import annotations

import pytest

from arc_llama.config import Config, GPUConfig, PathsConfig
from arc_llama.models import (
    discover_ggufs,
    infer_kv_class,
    parse_hf_spec,
    register_discovered,
)


@pytest.mark.parametrize(
    ("spec", "repo", "file", "quant"),
    [
        ("unsloth/gemma-3-1b-it-GGUF", "unsloth/gemma-3-1b-it-GGUF", None, None),
        (
            "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
            "unsloth/gemma-3-1b-it-GGUF",
            "gemma-3-1b-it-Q4_K_M.gguf",
            None,
        ),
        ("unsloth/gemma-3-1b-it-GGUF:Q4_K_M", "unsloth/gemma-3-1b-it-GGUF", None, "Q4_K_M"),
    ],
)
def test_parse_hf_spec_accepts_repo_file_and_quant_forms(spec, repo, file, quant):
    parsed = parse_hf_spec(spec)

    assert parsed.repo == repo
    assert parsed.file == file
    assert parsed.quant == quant


def test_parse_hf_spec_rejects_invalid_input():
    with pytest.raises(ValueError, match="Invalid HF spec"):
        parse_hf_spec("not-a-repo")


@pytest.mark.parametrize(
    ("filename", "kv_class"),
    [
        ("gemma-3-4b-it-Q4_K_M.gguf", "gemma_swa"),
        ("Qwen3-27B-Q4_K_M.gguf", "qwen3_27b_dense"),
        ("qwen3-30b-a3b-q4_k_m.gguf", "moe_a3b"),
        ("plain-7b-q4_k_m.gguf", "default"),
    ],
)
def test_infer_kv_class_from_common_model_names(filename, kv_class):
    assert infer_kv_class(filename) == kv_class


def test_discover_and_register_ggufs_skips_hidden_symlink_and_existing_files(tmp_path):
    models_dir = tmp_path / "models"
    nested = models_dir / "vendor" / "model"
    nested.mkdir(parents=True)
    first = nested / "Qwen3-27B-Q4_K_M.gguf"
    first.write_bytes(b"fake gguf")
    (nested / "notes.txt").write_text("not a model")
    hidden = models_dir / ".hidden"
    hidden.mkdir()
    (hidden / "ignored.gguf").write_bytes(b"hidden")
    (models_dir / "link.gguf").symlink_to(first)

    cfg = Config(
        paths=PathsConfig(models_dir=str(models_dir)),
        gpus=[
            GPUConfig(
                pci_slot="0000:03:00.0",
                sycl_index=0,
                arch="battlemage",
                vram_mb=24576,
                name="Arc Pro B60",
            )
        ],
    )

    discovered = discover_ggufs(cfg)
    added = register_discovered(cfg, discovered)
    added_again = register_discovered(cfg, discovered)

    assert discovered == [first.resolve()]
    assert [model.name for model in added] == ["qwen3-27b-q4_k_m"]
    assert added[0].kv_class == "qwen3_27b_dense"
    assert added_again == []
    assert len(cfg.models) == 1
