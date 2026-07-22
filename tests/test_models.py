from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arc_llama.config import Config, GPUConfig, ModelConfig, PathsConfig
from arc_llama.models import (
    HFModelSpec,
    add_local_model,
    discover_ggufs,
    download_from_hf,
    parse_hf_spec,
    register_discovered,
    short_name_from_path,
)

# ===========================================================================
# parse_hf_spec
# ===========================================================================


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
        # Quant hints — various forms
        ("org/repo:Q8_0", "org/repo", None, "Q8_0"),
        ("org/repo:IQ2_XXS", "org/repo", None, "IQ2_XXS"),
        ("org/repo:UD-Q4_K_M", "org/repo", None, "UD-Q4_K_M"),
        ("org/repo:IQ3_M", "org/repo", None, "IQ3_M"),
        # Exact filename with .gguf
        ("org/repo:model-Q4_K_M.gguf", "org/repo", "model-Q4_K_M.gguf", None),
        # Filename that looks like quant but has .gguf → treated as filename
        ("org/repo:Q4_K_M.gguf", "org/repo", "Q4_K_M.gguf", None),
        # Weird but valid repo names
        ("my-org/repo-name-GGUF", "my-org/repo-name-GGUF", None, None),
        ("org123/repo_456", "org123/repo_456", None, None),
    ],
)
def test_parse_hf_spec_accepts_repo_file_and_quant_forms(spec, repo, file, quant):
    parsed = parse_hf_spec(spec)
    assert parsed.repo == repo
    assert parsed.file == file
    assert parsed.quant == quant


@pytest.mark.parametrize(
    "bad_spec",
    [
        "not-a-repo",  # missing slash
        "",  # empty string
        "org",  # no slash
        "org/",  # empty repo name after slash
        "/repo",  # empty org
        "org/repo@branch",  # @ not allowed
        "org repo",  # space not allowed
    ],
)
def test_parse_hf_spec_rejects_invalid_input(bad_spec):
    with pytest.raises(ValueError, match="Invalid HF spec"):
        parse_hf_spec(bad_spec)


def test_parse_hf_spec_empty_after_colon():
    """Trailing colon with nothing after it is rejected by the regex."""
    with pytest.raises(ValueError, match="Invalid HF spec"):
        parse_hf_spec("org/repo:")


# ===========================================================================
# download_from_hf
# ===========================================================================


@pytest.fixture
def mock_hf_module():
    """Return a mock huggingface_hub module with HfApi and hf_hub_download."""
    mock = MagicMock()
    mock_api_instance = MagicMock()
    mock.HfApi.return_value = mock_api_instance
    mock.hf_hub_download = MagicMock(return_value="/mock/download/model.gguf")
    return mock, mock_api_instance


def _make_download_from_hf_test_env(mock_hf_module):
    """Patch sys.modules so huggingface_hub resolves to our mock."""
    mock_module, _ = mock_hf_module
    return patch.dict(sys.modules, {"huggingface_hub": mock_module})


def test_download_from_hf_explicit_filename(mock_hf_module, tmp_path):
    mock_module, mock_api = mock_hf_module
    spec = HFModelSpec(repo="org/repo", file="model-Q4_K_M.gguf", quant=None)

    with _make_download_from_hf_test_env(mock_hf_module):
        result = download_from_hf(spec, target_dir=tmp_path)

    assert result == Path("/mock/download/model.gguf")
    mock_api.list_repo_files.assert_not_called()
    mock_module.hf_hub_download.assert_called_once_with(
        repo_id="org/repo",
        filename="model-Q4_K_M.gguf",
        local_dir=str(tmp_path),
        token=None,
    )


def test_download_from_hf_quant_hint_single_match(mock_hf_module, tmp_path):
    mock_module, mock_api = mock_hf_module
    mock_api.list_repo_files.return_value = [
        "README.md",
        "model-Q4_K_M.gguf",
        "model-Q8_0.gguf",
    ]
    spec = HFModelSpec(repo="org/repo", file=None, quant="Q4_K_M")

    with _make_download_from_hf_test_env(mock_hf_module):
        result = download_from_hf(spec, target_dir=tmp_path)

    assert result == Path("/mock/download/model.gguf")
    mock_api.list_repo_files.assert_called_once_with("org/repo")
    mock_module.hf_hub_download.assert_called_once_with(
        repo_id="org/repo",
        filename="model-Q4_K_M.gguf",
        local_dir=str(tmp_path),
        token=None,
    )


def test_download_from_hf_quant_hint_prefers_uniform(mock_hf_module, tmp_path):
    """When multiple files match the quant hint, prefer non-UD/non-XL variants."""
    mock_module, mock_api = mock_hf_module
    mock_api.list_repo_files.return_value = [
        "model-UD-Q4_K_M.gguf",
        "model-Q4_K_M.gguf",
        "model-Q4_K_M_XL.gguf",
    ]
    spec = HFModelSpec(repo="org/repo", file=None, quant="Q4_K_M")

    with _make_download_from_hf_test_env(mock_hf_module):
        result = download_from_hf(spec, target_dir=tmp_path)

    # Should pick the uniform quant (not UD- or _XL)
    assert result == Path("/mock/download/model.gguf")
    mock_module.hf_hub_download.assert_called_once_with(
        repo_id="org/repo",
        filename="model-Q4_K_M.gguf",
        local_dir=str(tmp_path),
        token=None,
    )


def test_download_from_hf_quant_hint_no_matches(mock_hf_module, tmp_path):
    mock_module, mock_api = mock_hf_module
    mock_api.list_repo_files.return_value = [
        "model-Q5_K_M.gguf",
        "model-Q8_0.gguf",
    ]
    spec = HFModelSpec(repo="org/repo", file=None, quant="Q4_K_M")

    with _make_download_from_hf_test_env(mock_hf_module):
        with pytest.raises(FileNotFoundError, match="No GGUF in org/repo matched quant hint"):
            download_from_hf(spec, target_dir=tmp_path)


def test_download_from_hf_multiple_ggufs_no_hint(mock_hf_module, tmp_path):
    mock_module, mock_api = mock_hf_module
    mock_api.list_repo_files.return_value = [
        "model-Q4_K_M.gguf",
        "model-Q8_0.gguf",
    ]
    spec = HFModelSpec(repo="org/repo", file=None, quant=None)

    with _make_download_from_hf_test_env(mock_hf_module):
        with pytest.raises(ValueError, match="has 2 GGUF files; specify one"):
            download_from_hf(spec, target_dir=tmp_path)


def test_download_from_hf_single_gguf_auto_picks(mock_hf_module, tmp_path):
    mock_module, mock_api = mock_hf_module
    mock_api.list_repo_files.return_value = [
        "README.md",
        "model-Q4_K_M.gguf",
    ]
    spec = HFModelSpec(repo="org/repo", file=None, quant=None)

    with _make_download_from_hf_test_env(mock_hf_module):
        result = download_from_hf(spec, target_dir=tmp_path)

    assert result == Path("/mock/download/model.gguf")
    mock_module.hf_hub_download.assert_called_once_with(
        repo_id="org/repo",
        filename="model-Q4_K_M.gguf",
        local_dir=str(tmp_path),
        token=None,
    )


def test_download_from_hf_missing_huggingface_hub(tmp_path, monkeypatch):
    """When huggingface_hub is not installed, raise RuntimeError with helpful message."""
    import builtins

    spec = HFModelSpec(repo="org/repo", file="model.gguf", quant=None)

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(RuntimeError, match="huggingface-hub is required for downloads"):
        download_from_hf(spec, target_dir=tmp_path)


def test_download_from_hf_returns_correct_path(mock_hf_module, tmp_path):
    """Verify the returned path matches what hf_hub_download returns."""
    mock_module, _ = mock_hf_module
    custom_path = "/custom/path/to/model.gguf"
    mock_module.hf_hub_download.return_value = custom_path
    spec = HFModelSpec(repo="org/repo", file="model.gguf", quant=None)

    with _make_download_from_hf_test_env(mock_hf_module):
        result = download_from_hf(spec, target_dir=tmp_path)

    assert result == Path(custom_path)


def test_download_from_hf_passes_token(mock_hf_module, tmp_path):
    mock_module, mock_api = mock_hf_module
    mock_api.list_repo_files.return_value = ["model.gguf"]
    spec = HFModelSpec(repo="org/repo", file=None, quant=None)

    with _make_download_from_hf_test_env(mock_hf_module):
        download_from_hf(spec, target_dir=tmp_path, token="hf_12345")

    mock_api.list_repo_files.assert_called_once_with("org/repo")
    mock_module.hf_hub_download.assert_called_once_with(
        repo_id="org/repo",
        filename="model.gguf",
        local_dir=str(tmp_path),
        token="hf_12345",
    )


# ===========================================================================
# add_local_model
# ===========================================================================


@pytest.fixture
def mock_recipe_and_mtp():
    """Patch default_recipe and has_mtp_heads for add_local_model/register_discovered."""
    from arc_llama.recipes import KVCacheType, LaunchRecipe

    def _recipe(*, arch, vram_mb, model_file_mb, kv_class):
        return LaunchRecipe(
            n_gpu_layers=999,
            ctx=8192,
            parallel=1,
            cache_type_k=KVCacheType.Q8_0,
            cache_type_v=KVCacheType.Q8_0,
        )

    with (
        patch("arc_llama.models.default_recipe", side_effect=_recipe) as mock_recipe,
        patch("arc_llama.models.has_mtp_heads", return_value=False) as mock_mtp,
    ):
        yield mock_recipe, mock_mtp


def _make_config_with_gpu(tmp_path):
    cfg = Config(
        paths=PathsConfig(models_dir=str(tmp_path)),
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
    return cfg


@pytest.mark.parametrize(
    "bad_name",
    [
        ".hidden",  # starts with dot
        "has space",  # contains space
        "has/slash",  # contains slash
        "has@at",  # contains @
        "",  # empty
        "-starts-with-dash",  # starts with non-alphanumeric
    ],
)
def test_add_local_model_rejects_invalid_name(bad_name, tmp_path, mock_recipe_and_mtp):
    cfg = _make_config_with_gpu(tmp_path)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake")

    with pytest.raises(ValueError, match="Model name .* must match"):
        add_local_model(
            cfg,
            name=bad_name,
            path=str(model_file),
            gpu_pci_slot="0000:03:00.0",
        )


def test_add_local_model_missing_file(tmp_path, mock_recipe_and_mtp):
    cfg = _make_config_with_gpu(tmp_path)

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        add_local_model(
            cfg,
            name="test-model",
            path=str(tmp_path / "nonexistent.gguf"),
            gpu_pci_slot="0000:03:00.0",
        )


def test_add_local_model_missing_gpu(tmp_path, mock_recipe_and_mtp):
    cfg = _make_config_with_gpu(tmp_path)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake")

    with pytest.raises(ValueError, match="GPU 0000:99:00.0 not in config"):
        add_local_model(
            cfg,
            name="test-model",
            path=str(model_file),
            gpu_pci_slot="0000:99:00.0",
        )


def test_add_local_model_duplicate_name(tmp_path, mock_recipe_and_mtp):
    cfg = _make_config_with_gpu(tmp_path)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake")

    add_local_model(
        cfg,
        name="test-model",
        path=str(model_file),
        gpu_pci_slot="0000:03:00.0",
    )

    model_file2 = tmp_path / "model2.gguf"
    model_file2.write_bytes(b"fake2")

    with pytest.raises(ValueError, match="Model name 'test-model' already registered"):
        add_local_model(
            cfg,
            name="test-model",
            path=str(model_file2),
            gpu_pci_slot="0000:03:00.0",
        )


def test_add_local_model_duplicate_port(tmp_path, mock_recipe_and_mtp):
    cfg = _make_config_with_gpu(tmp_path)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake")

    add_local_model(
        cfg,
        name="test-model",
        path=str(model_file),
        gpu_pci_slot="0000:03:00.0",
        port=18080,
    )

    model_file2 = tmp_path / "model2.gguf"
    model_file2.write_bytes(b"fake2")

    with pytest.raises(ValueError, match="Port 18080 already in use"):
        add_local_model(
            cfg,
            name="test-model2",
            path=str(model_file2),
            gpu_pci_slot="0000:03:00.0",
            port=18080,
        )


def test_add_local_model_recipe_overrides_applied(tmp_path, mock_recipe_and_mtp):
    cfg = _make_config_with_gpu(tmp_path)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake")

    mc = add_local_model(
        cfg,
        name="test-model",
        path=str(model_file),
        gpu_pci_slot="0000:03:00.0",
        recipe_overrides={"ctx": 12345, "parallel": 4},
    )

    assert mc.recipe["ctx"] == 12345
    assert mc.recipe["parallel"] == 4
    assert mc.recipe["n_gpu_layers"] == 999  # from default_recipe


def test_add_local_model_auto_mtp_heads(tmp_path):
    """When has_mtp_heads returns True, recipe should include draft-mtp settings."""
    cfg = _make_config_with_gpu(tmp_path)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake")

    with (
        patch("arc_llama.models.default_recipe") as mock_recipe,
        patch("arc_llama.models.has_mtp_heads", return_value=True),
    ):
        from arc_llama.recipes import KVCacheType, LaunchRecipe

        mock_recipe.return_value = LaunchRecipe(
            n_gpu_layers=999,
            ctx=8192,
            parallel=1,
            cache_type_k=KVCacheType.Q8_0,
            cache_type_v=KVCacheType.Q8_0,
        )

        mc = add_local_model(
            cfg,
            name="test-model",
            path=str(model_file),
            gpu_pci_slot="0000:03:00.0",
        )

    assert mc.recipe["spec_type"] == "draft-mtp"
    assert mc.recipe["ubatch_size"] == 8


def test_add_local_model_auto_port(tmp_path, mock_recipe_and_mtp):
    """When port is not specified, it should auto-assign the next free port."""
    cfg = _make_config_with_gpu(tmp_path)
    # Pre-populate with a model on port 18080
    cfg.models.append(
        ModelConfig(
            name="existing",
            path=str(tmp_path / "existing.gguf"),
            port=18080,
            gpu_pci_slot="0000:03:00.0",
        )
    )
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake")

    mc = add_local_model(
        cfg,
        name="test-model",
        path=str(model_file),
        gpu_pci_slot="0000:03:00.0",
    )

    assert mc.port == 18081  # next free after 18080


# ===========================================================================
# short_name_from_path
# ===========================================================================


def test_short_name_from_path_basic():
    path = Path("/models/Qwen3-27B-Q4_K_M.gguf")
    assert short_name_from_path(path, set()) == "qwen3-27b-q4_k_m"


def test_short_name_from_path_unique_on_collision():
    used = set()
    path1 = Path("/models/Qwen3-27B-Q4_K_M.gguf")
    name1 = short_name_from_path(path1, used)
    assert name1 == "qwen3-27b-q4_k_m"
    used.add(name1)

    # Same filename in different directory — should get -2 suffix
    path2 = Path("/other/Qwen3-27B-Q4_K_M.gguf")
    name2 = short_name_from_path(path2, used)
    assert name2 == "qwen3-27b-q4_k_m-2"
    used.add(name2)

    # Third collision
    name3 = short_name_from_path(path1, used)
    assert name3 == "qwen3-27b-q4_k_m-3"


def test_short_name_from_path_strips_gguf_suffix():
    path = Path("/models/model.gguf")
    assert short_name_from_path(path, set()) == "model"


def test_short_name_from_path_strips_imatrix_suffix():
    path = Path("/models/model.imatrix.gguf")
    assert short_name_from_path(path, set()) == "model"


# ===========================================================================
# register_discovered
# ===========================================================================


def test_register_discovered_skips_already_registered(tmp_path, mock_recipe_and_mtp):
    cfg = _make_config_with_gpu(tmp_path)
    model_file = tmp_path / "Qwen3-27B-Q4_K_M.gguf"
    model_file.write_bytes(b"fake")

    # First registration
    added1 = register_discovered(cfg, [model_file])
    assert len(added1) == 1

    # Second registration should skip
    added2 = register_discovered(cfg, [model_file])
    assert added2 == []


def test_register_discovered_skips_nonexistent_files(tmp_path, mock_recipe_and_mtp):
    cfg = _make_config_with_gpu(tmp_path)
    existing = tmp_path / "existing.gguf"
    existing.write_bytes(b"fake")
    missing = tmp_path / "missing.gguf"

    added = register_discovered(cfg, [existing, missing])
    assert len(added) == 1
    assert added[0].path == str(existing.resolve())


def test_register_discovered_correct_recipe_assignment(tmp_path, mock_recipe_and_mtp):
    cfg = _make_config_with_gpu(tmp_path)
    model_file = tmp_path / "Qwen3-27B-Q4_K_M.gguf"
    model_file.write_bytes(b"fake")

    added = register_discovered(cfg, [model_file])
    assert len(added) == 1
    assert added[0].kv_class == "qwen3_27b_dense"
    assert added[0].recipe["n_gpu_layers"] == 999


def test_register_discovered_no_gpus_raises(tmp_path, mock_recipe_and_mtp):
    cfg = Config(paths=PathsConfig(models_dir=str(tmp_path)), gpus=[])
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake")

    with pytest.raises(ValueError, match="No GPUs in config"):
        register_discovered(cfg, [model_file])


def test_register_discovered_uses_first_gpu_when_none_enabled(tmp_path, mock_recipe_and_mtp):
    cfg = Config(
        paths=PathsConfig(models_dir=str(tmp_path)),
        gpus=[
            GPUConfig(
                pci_slot="0000:03:00.0",
                sycl_index=0,
                arch="battlemage",
                vram_mb=24576,
                enabled=False,
                name="Arc Pro B60",
            )
        ],
    )
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake")

    added = register_discovered(cfg, [model_file])
    assert len(added) == 1
    assert added[0].gpu_pci_slot == "0000:03:00.0"


def test_register_discovered_auto_mtp(tmp_path):
    """Discovered models with MTP heads get draft-mtp in recipe."""
    cfg = _make_config_with_gpu(tmp_path)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake")

    with (
        patch("arc_llama.models.default_recipe") as mock_recipe,
        patch("arc_llama.models.has_mtp_heads", return_value=True),
    ):
        from arc_llama.recipes import KVCacheType, LaunchRecipe

        mock_recipe.return_value = LaunchRecipe(
            n_gpu_layers=999,
            ctx=8192,
            parallel=1,
            cache_type_k=KVCacheType.Q8_0,
            cache_type_v=KVCacheType.Q8_0,
        )

        added = register_discovered(cfg, [model_file])

    assert len(added) == 1
    assert added[0].recipe["spec_type"] == "draft-mtp"
    assert added[0].recipe["ubatch_size"] == 8


# ===========================================================================
# _slugify_for_name (from cli.py)
# ===========================================================================


def test_slugify_for_name_basic():
    from arc_llama.cli import _slugify_for_name

    assert _slugify_for_name("repo-name", "model-Q4_K_M.gguf") == "repo-name-q4_k_m"


def test_slugify_for_name_strips_special_chars():
    from arc_llama.cli import _slugify_for_name

    # Underscore is allowed in the character class, slash becomes hyphen
    assert _slugify_for_name("my_org/repo", "file.gguf") == "my_org-repo"


def test_slugify_for_name_extracts_quant():
    from arc_llama.cli import _slugify_for_name

    assert _slugify_for_name("repo", "model-IQ2_XXS.gguf") == "repo-iq2_xxs"
    assert _slugify_for_name("repo", "model-UD-Q4_K_M.gguf") == "repo-ud-q4_k_m"
    assert _slugify_for_name("repo", "model-Q8_0.gguf") == "repo-q8_0"


def test_slugify_for_name_empty_parent():
    from arc_llama.cli import _slugify_for_name

    assert _slugify_for_name("", "model-Q4_K_M.gguf") == "model-q4_k_m"


# ===========================================================================
# Existing discovery test
# ===========================================================================


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


# ===========================================================================
# slow weight-quant warning + new recipe field persistence
# ===========================================================================


class TestSlowWeightQuantWarning:
    def test_q8_0_on_battlemage_warns(self):
        from arc_llama.arch import Arch
        from arc_llama.models import warn_slow_weight_quant
        msg = warn_slow_weight_quant(Arch.BATTLEMAGE, "Qwen3.6-27B-Q8_0.gguf")
        assert msg is not None
        assert "Q8_0" in msg

    def test_q4_k_m_on_battlemage_silent(self):
        from arc_llama.arch import Arch
        from arc_llama.models import warn_slow_weight_quant
        assert warn_slow_weight_quant(Arch.BATTLEMAGE, "Qwen3.6-27B-Q4_K_M.gguf") is None

    def test_q8_0_on_alchemist_silent(self):
        # No measured slow path on Alchemist — don't cry wolf.
        from arc_llama.arch import Arch
        from arc_llama.models import warn_slow_weight_quant
        assert warn_slow_weight_quant(Arch.ALCHEMIST, "model-Q8_0.gguf") is None


def test_add_local_model_persists_xmx_fields(tmp_path):
    """Real default_recipe on a roomy Battlemage → FA fields land in config."""
    with patch("arc_llama.models.has_mtp_heads", return_value=False):
        cfg = _make_config_with_gpu(tmp_path)
        gguf = tmp_path / "model-Q4_K_M.gguf"
        gguf.write_bytes(b"\x00" * 1024)
        mc = add_local_model(
            cfg, name="m", path=str(gguf), gpu_pci_slot="0000:03:00.0",
        )
        assert mc.recipe["cache_type_k"] == "f16"
        assert mc.recipe["flash_attn"] == "on"
        assert mc.recipe["ubatch_size"] == 1024
        assert mc.recipe["batch_size"] == 2048
        assert mc.recipe["cache_reuse"] == 256
