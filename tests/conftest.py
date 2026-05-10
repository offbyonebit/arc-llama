from __future__ import annotations

from pathlib import Path

from arc_llama.config import Config, GPUConfig, ModelConfig


def make_config(tmp_path: Path, *, single_resident: bool = True) -> Config:
    cfg = Config()
    cfg.server.single_resident = single_resident
    cfg.paths.llama_server = "/usr/bin/llama-server"
    cfg.paths.models_dir = str(tmp_path / "models")
    cfg.gpus = [
        GPUConfig(
            pci_slot="0000:03:00.0",
            sycl_index=0,
            arch="battlemage",
            vram_mb=24576,
            name="Arc Pro B60",
        ),
        GPUConfig(
            pci_slot="0000:04:00.0",
            sycl_index=1,
            arch="alchemist",
            vram_mb=16384,
            name="Arc A770",
        ),
    ]
    cfg.models = [
        ModelConfig(
            name="qwen",
            display_name="Qwen 3",
            path=str(tmp_path / "models" / "Qwen3-7B-Q4_K_M.gguf"),
            port=18080,
            gpu_pci_slot="0000:03:00.0",
            aliases=["qwen.gguf"],
            recipe={
                "ctx": 8192,
                "n_gpu_layers": 999,
                "parallel": 1,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
            },
        ),
        ModelConfig(
            name="gemma",
            display_name="Gemma",
            path=str(tmp_path / "models" / "gemma-3-4b-Q4_K_M.gguf"),
            port=18081,
            gpu_pci_slot="0000:04:00.0",
            aliases=["gemma.gguf"],
            recipe={
                "ctx": 8192,
                "n_gpu_layers": 999,
                "parallel": 1,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
            },
        ),
    ]
    return cfg
