from __future__ import annotations

from arc_llama.config import Config, GPUConfig, ModelConfig, load_config


def test_config_round_trips_models_gpus_and_paths(tmp_path):
    path = tmp_path / "config.toml"
    cfg = Config()
    cfg.paths.llama_server = "/opt/llama.cpp/llama-server"
    cfg.paths.scan_paths = [str(tmp_path / "extra-models")]
    cfg.gpus = [
        GPUConfig(
            pci_slot="0000:03:00.0",
            sycl_index=0,
            arch="battlemage",
            vram_mb=24576,
            name="Arc Pro B60",
        )
    ]
    cfg.models = [
        ModelConfig(
            name="qwen",
            path=str(tmp_path / "Qwen3-7B-Q4_K_M.gguf"),
            port=18080,
            gpu_pci_slot="0000:03:00.0",
            display_name="Qwen 3 7B",
            aliases=["Qwen3-7B-Q4_K_M.gguf"],
            recipe={"ctx": 32768, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
        )
    ]

    cfg.save(path)
    loaded = load_config(path)

    assert loaded.paths.llama_server == "/opt/llama.cpp/llama-server"
    assert loaded.paths.scan_paths == [str(tmp_path / "extra-models")]
    assert loaded.gpus[0].name == "Arc Pro B60"
    assert loaded.models[0].recipe["ctx"] == 32768


def test_find_model_matches_name_alias_display_name_and_filename(tmp_path):
    cfg = Config(
        models=[
            ModelConfig(
                name="qwen",
                path=str(tmp_path / "Qwen3-7B-Q4_K_M.gguf"),
                port=18080,
                gpu_pci_slot="0000:03:00.0",
                display_name="Qwen 3 7B",
                aliases=["chat-default"],
            )
        ]
    )

    assert cfg.find_model("qwen").name == "qwen"
    assert cfg.find_model("chat-default").name == "qwen"
    assert cfg.find_model("qwen 3").name == "qwen"
    assert cfg.find_model("Q4_K_M").name == "qwen"
    assert cfg.find_model("missing") is None
