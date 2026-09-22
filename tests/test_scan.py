"""Focused tests for the model scan improvements: auxiliary-file skipping,
recursive GGUF discovery, prune/stale accounting, and `arc-llama scan` output.

These complement test_models.py's discovery tests; they focus on the
`scan_models` / ScanResult layer and the `scan` CLI command.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from arc_llama.cli import cli
from arc_llama.config import Config, GPUConfig, PathsConfig
from arc_llama.models import (
    ScanResult,
    discover_ggufs,
    find_stale_models,
    is_auxiliary_gguf,
    is_partial_download,
    is_skipped_dir,
    prune_missing_models,
    scan_models,
)


def _make_cfg(tmp_path: Path, models_dir: Path | None = None) -> Config:
    return Config(
        paths=PathsConfig(models_dir=str(models_dir or tmp_path)),
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


# ===========================================================================
# Auxiliary-file classification
# ===========================================================================


@pytest.mark.parametrize(
    "name",
    [
        "mmproj-model.gguf",
        "mmproj-Qwen3-VL-8B.gguf",
        "model.mmproj.gguf",
        "model-projector.gguf",
        "clip-vision.gguf",
        "model.vision.gguf",
        "model-vision_proj.gguf",
        "model.mtp.gguf",
        "model-MTP.gguf",
        # draft-prefixed projection sidecars
        "mtp-mmproj-model.gguf",
        # MTP draft sidecar with the marker mid-name (nextn style)
        "model.IQ4_XS-00001-of-00002.mtp.gguf",
    ],
)
def test_is_auxiliary_gguf_matches(name):
    assert is_auxiliary_gguf(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "Qwen3.6-27B-MTP-UD-Q4_K_XL.gguf",  # embedded MTP is a full model
        "gemma-4-26B-A4B-it.gguf",
        "model-q4_k_m.gguf",
        "deepseek-r1-distill-qwen-7b.gguf",
        "mtp-gemma-4-26B-A4B-it.gguf",  # draft sidecar handled by find_draft_model
        "model.gguf",
        "project-alpha.gguf",  # 'project' but not a projector marker
        "clipper-8b.gguf",  # 'clip' prefix must be token-bounded
    ],
)
def test_is_auxiliary_gguf_does_not_overmatch(name):
    assert is_auxiliary_gguf(name) is False


def test_is_auxiliary_gguf_tolerates_missing_extension():
    assert is_auxiliary_gguf("mmproj-model") is True
    assert is_auxiliary_gguf("model.mtp") is True


def test_is_partial_download_matches_locks_and_partials():
    assert is_partial_download(Path("/m/model.gguf.lock")) is True
    assert is_partial_download(Path("/m/model.gguf.partial")) is True
    assert is_partial_download(Path("/m/model.gguf.incomplete")) is True
    assert is_partial_download(Path("/m/download-model.gguf")) is False
    assert is_partial_download(Path("/m/model.gguf")) is False
    assert is_partial_download(Path("/m/model-lock.gguf")) is False


def test_is_skipped_dir():
    assert is_skipped_dir("__pycache__") is True
    assert is_skipped_dir(".cache") is True
    assert is_skipped_dir("models") is False
    assert is_skipped_dir("vendor") is False


# ===========================================================================
# Recursive discovery: deep nesting + aux skipping
# ===========================================================================


def _write(p: Path, size: int = 4096) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0" * size)
    return p


def _layout(tmp_path: Path) -> tuple[Path, list[Path]]:
    """models_dir with a real model tree and every auxiliary kind."""
    models_dir = tmp_path / "models"
    real = [
        _write(models_dir / "Qwen3.6-27B-Q4_K_M.gguf"),
        _write(models_dir / "vendor" / "deep" / "deepseek-r1-distill-qwen-7b.gguf"),
    ]
    aux = [
        _write(models_dir / "Qwen3.6-VL" / "mmproj-Qwen3.6-VL-8B.gguf", 128),
        _write(models_dir / "Qwen3.6-27B.mtp.gguf", 128),
        _write(models_dir / "Qwen3.6-27B-Q4_K_M.gguf.partial", 128),
        _write(models_dir / "Qwen3.6-27B-Q4_K_M.gguf.lock", 0),
    ]
    # cache directory with a real-looking model inside — must not be walked
    _write(models_dir / "__pycache__" / "hidden-model.gguf")
    _write(models_dir / ".cache" / "hf" / "other-model.gguf")
    return models_dir, real + aux


def test_discover_ggufs_is_recursive_and_skips_aux(tmp_path):
    models_dir, real = _layout(tmp_path)
    found = discover_ggufs(_make_cfg(tmp_path, models_dir))
    assert Path(real[0]).resolve() in [p.resolve() for p in found]
    assert Path(real[1]).resolve() in [p.resolve() for p in found]
    for name in (
        "mmproj-Qwen3.6-VL-8B.gguf",
        "Qwen3.6-27B.mtp.gguf",
        "Qwen3.6-27B-Q4_K_M.gguf.partial",
        "Qwen3.6-27B-Q4_K_M.gguf.lock",
        "hidden-model.gguf",
        "other-model.gguf",
    ):
        assert all(p.name != name for p in found), name


@pytest.mark.skipif(sys.platform == "win32", reason="depth 4 is shallow; fine on both")
def test_discover_ggufs_respects_max_depth(tmp_path):
    models_dir = tmp_path / "models"
    _write(models_dir / "a" / "b" / "c" / "d" / "e" / "deep.gguf")
    _write(models_dir / "a" / "b" / "shallow.gguf")
    found = discover_ggufs(_make_cfg(tmp_path, models_dir), max_depth=4)
    names = [p.name for p in found]
    assert "shallow.gguf" in names
    assert "deep.gguf" not in names


def test_discover_ggufs_walks_configured_scan_paths(tmp_path):
    models_dir = tmp_path / "models"
    _write(models_dir / "in-models-dir.gguf")
    extra_root = tmp_path / "extra"
    extra_file = _write(extra_root / "nested" / "in-extra-scan-path.gguf")
    _write(tmp_path / "not-scanned" / "ignored.gguf")

    cfg = _make_cfg(tmp_path, models_dir)
    cfg.paths.scan_paths = [str(extra_root)]
    found = [p.resolve() for p in discover_ggufs(cfg)]
    assert extra_file.resolve() in found
    assert all("ignored" not in p.name for p in found)


# ===========================================================================
# scan_models: counts + prune
# ===========================================================================


@contextmanager
def _no_meta_patches():
    """Neutralize GGUF-metadata probes so fake byte payloads pass through."""
    from contextlib import ExitStack

    from arc_llama.recipes import KVCacheType, LaunchRecipe

    recipe = LaunchRecipe(
        n_gpu_layers=999,
        ctx=8192,
        parallel=1,
        cache_type_k=KVCacheType.Q8_0,
        cache_type_v=KVCacheType.Q8_0,
    )
    with ExitStack() as stack:
        patches = [
            stack.enter_context(p)
            for p in (
                patch("arc_llama.models.default_recipe", return_value=recipe),
                patch("arc_llama.models.has_mtp_heads", return_value=False),
                patch("arc_llama.models.is_moe", return_value=False),
            )
        ]
        yield tuple(patches)


def test_scan_models_counts_new_unchanged_skipped(tmp_path):
    models_dir, real = _layout(tmp_path)
    cfg = _make_cfg(tmp_path, models_dir)

    with _no_meta_patches():
        res = scan_models(cfg)
        assert isinstance(res, ScanResult)
        assert len(res.new) == 2
        assert res.new[0].path == str(real[0].resolve())
        # First pass: nothing stale, nothing pruned, nothing unchanged
        assert res.unchanged == []
        assert res.stale == []
        assert res.pruned == []
        # The two *.gguf-shaped aux files are counted as skipped; the
        # .partial/.lock scraps never pass the walker's *.gguf filter, and
        # the cache-dir models are never traversed at all.
        assert len(res.skipped_aux) == 2
        assert res.counts_line() == "2 new, 0 unchanged, 2 skipped, 0 stale"

        # Second pass: both models now unchanged; aux files are still skipped
        res2 = scan_models(cfg)
        assert res2.new == []
        assert len(res2.unchanged) == 2
        assert len(res2.skipped_aux) == 2
        assert res2.counts_line() == "0 new, 2 unchanged, 2 skipped, 0 stale"


def test_scan_models_reports_stale_without_prune(tmp_path):
    models_dir = tmp_path / "models"
    model = _write(models_dir / "Qwen3.6-27B-Q4_K_M.gguf")
    cfg = _make_cfg(tmp_path, models_dir)

    with _no_meta_patches():
        res = scan_models(cfg)
        assert len(res.new) == 1

    # File goes missing: default scan reports stale but keeps the registry.
    model.unlink()
    res2 = scan_models(cfg)
    assert res2.new == []
    assert len(res2.stale) == 1
    assert res2.stale[0].name == "qwen3.6-27b-q4_k_m"
    assert res2.pruned == []
    assert len(cfg.models) == 1  # non-destructive by default
    assert res2.counts_line() == "0 new, 0 unchanged, 0 skipped, 1 stale"


def test_scan_models_prune_removes_stale(tmp_path):
    models_dir = tmp_path / "models"
    model = _write(models_dir / "Qwen3.6-27B-Q4_K_M.gguf")
    cfg = _make_cfg(tmp_path, models_dir)

    with _no_meta_patches():
        scan_models(cfg)
    model.unlink()

    res = scan_models(cfg, prune=True)
    assert res.stale == []
    assert len(res.pruned) == 1
    assert cfg.models == []
    assert res.counts_line() == "0 new, 0 unchanged, 0 skipped, 0 stale, 1 pruned"


def test_prune_missing_models_and_find_stale(tmp_path):
    models_dir = tmp_path / "models"
    present = _write(models_dir / "present.gguf")
    cfg = _make_cfg(tmp_path, models_dir)
    from arc_llama.config import ModelConfig

    cfg.models = [
        ModelConfig(
            name="present",
            path=str(present),
            port=18080,
            gpu_pci_slot="0000:03:00.0",
        ),
        ModelConfig(
            name="gone",
            path=str(models_dir / "gone.gguf"),
            port=18081,
            gpu_pci_slot="0000:03:00.0",
        ),
    ]

    assert [m.name for m in find_stale_models(cfg)] == ["gone"]
    removed = prune_missing_models(cfg)
    assert [m.name for m in removed] == ["gone"]
    assert [m.name for m in cfg.models] == ["present"]

    # Idempotent: a second pass is a no-op.
    assert prune_missing_models(cfg) == []


# ===========================================================================
# Draft sidecar still skipped (register-discovered path)
# ===========================================================================


def test_scan_models_skips_pairable_draft_sidecar(tmp_path):
    models_dir = tmp_path / "models" / "gemma"
    main = _write(models_dir / "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf", 4000)
    draft = _write(models_dir / "mtp-gemma-4-26B-A4B-it.gguf", 200)
    cfg = _make_cfg(tmp_path, models_dir.parent)

    with _no_meta_patches():
        res = scan_models(cfg)

    assert len(res.new) == 1
    assert res.new[0].path == str(main.resolve())
    assert Path(res.new[0].recipe["spec_draft_model"]).resolve() == draft.resolve()
    assert draft.resolve() in [p.resolve() for p in res.skipped_aux]


# ===========================================================================
# CLI: arc-llama scan
# ===========================================================================


@pytest.fixture
def runner():
    return CliRunner()


def _run_scan(runner: CliRunner, cfg_path: Path, *args: str) -> object:
    return runner.invoke(cli, ["--config", str(cfg_path), "scan", *args])


def _write_config(cfg: Config, tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    cfg.save(path)
    return path


def test_cli_scan_default_non_destructive_reports_stale(runner, tmp_path):
    models_dir = tmp_path / "models"
    model = _write(models_dir / "Qwen3.6-27B-Q4_K_M.gguf")
    cfg = _make_cfg(tmp_path, models_dir)
    cfg_path = _write_config(cfg, tmp_path)

    with _no_meta_patches():
        result = _run_scan(runner, cfg_path)
        assert result.exit_code == 0, result.output
        assert "1 new" in result.output
        assert "0 unchanged" in result.output
        assert "0 stale" in result.output
        assert "Registered 1 new model(s)" in result.output

        model.unlink()
        result = _run_scan(runner, cfg_path)
        assert result.exit_code == 0, result.output
        assert "1 stale" in result.output
        assert "--prune" in result.output  # hint to remove
        assert "qwen3.6-27b-q4_k_m" in result.output
        # Not removed without the flag
        from arc_llama.config import load_config

        assert load_config(cfg_path).models  # still registered


def test_cli_scan_prune_removes_stale(runner, tmp_path):
    models_dir = tmp_path / "models"
    model = _write(models_dir / "Qwen3.6-27B-Q4_K_M.gguf")
    cfg = _make_cfg(tmp_path, models_dir)
    cfg_path = _write_config(cfg, tmp_path)

    with _no_meta_patches():
        assert _run_scan(runner, cfg_path).exit_code == 0
        model.unlink()
        result = _run_scan(runner, cfg_path, "--prune")
        assert result.exit_code == 0, result.output
        assert "1 pruned" in result.output
        assert "Pruned 1 stale model(s): qwen3.6-27b-q4_k_m" in result.output

    from arc_llama.config import load_config

    assert load_config(cfg_path).models == []


def test_cli_scan_persists_explicit_roots(runner, tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    explicit = tmp_path / "srv-models"
    _write(explicit / "DeepSeek-R1-Distill-Qwen-7B.gguf")
    cfg = _make_cfg(tmp_path, models_dir)
    cfg_path = _write_config(cfg, tmp_path)

    with _no_meta_patches():
        result = _run_scan(runner, cfg_path, str(explicit))
        assert result.exit_code == 0, result.output
        assert "1 new" in result.output
        assert "paths.scan_paths" in result.output

    from arc_llama.config import load_config

    loaded = load_config(cfg_path)
    assert [str(Path(p).resolve()) for p in loaded.paths.scan_paths] == [str(explicit.resolve())]
    assert [m.path for m in loaded.models] == [
        str((explicit / "DeepSeek-R1-Distill-Qwen-7B.gguf").resolve())
    ]

    # A repeated scan with the same root must not duplicate the entry.
    with _no_meta_patches():
        assert _run_scan(runner, cfg_path, str(explicit)).exit_code == 0
    assert len(load_config(cfg_path).paths.scan_paths) == 1


def test_cli_scan_no_persist_leaves_config_alone(runner, tmp_path):
    models_dir = tmp_path / "models"
    _write(models_dir / "Qwen3.6-27B-Q4_K_M.gguf")
    cfg = _make_cfg(tmp_path, models_dir)
    cfg_path = _write_config(cfg, tmp_path)

    with _no_meta_patches():
        result = _run_scan(runner, cfg_path, "--no-persist")
        assert result.exit_code == 0, result.output
        assert "config NOT saved" in result.output

    from arc_llama.config import load_config

    assert load_config(cfg_path).models == []


def test_cli_scan_skips_aux_files_with_counts(runner, tmp_path):
    models_dir, _ = _layout(tmp_path)
    cfg = _make_cfg(tmp_path, models_dir)
    cfg_path = _write_config(cfg, tmp_path)

    with _no_meta_patches():
        result = _run_scan(runner, cfg_path)
        assert result.exit_code == 0, result.output
        assert "2 new" in result.output
        assert "2 skipped" in result.output
        assert "mmproj" not in "\n".join(
            line for line in result.output.splitlines() if "Registered" in line
        )


def test_cli_scan_requires_gpus(runner, tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    cfg = Config(paths=PathsConfig(models_dir=str(models_dir)), gpus=[])
    cfg_path = _write_config(cfg, tmp_path)
    result = _run_scan(runner, cfg_path)
    assert result.exit_code == 1
    assert "No GPUs" in result.output
