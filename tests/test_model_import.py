from __future__ import annotations

import errno
import json
from pathlib import Path

import gguf
import pytest
from click.testing import CliRunner

from arc_llama.config import Config, GPUConfig, PathsConfig, load_config
from arc_llama.model_import import ImportCandidate, default_roots, discover, import_models
from arc_llama.model_library import deletable_files, disk_report


def model_file(path: Path, split_count: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_context_length(4096)
    if split_count > 1:
        writer.add_uint16("split.count", split_count)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()
    return path


def manifest(root: Path, name: str, digest: str, **kwargs) -> Path:
    path = root / "manifests" / "registry.ollama.ai" / "library" / name / "latest"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": digest}], **kwargs}))
    return path


def config(tmp_path: Path) -> Config:
    return Config(gpus=[GPUConfig("gpu", 0, "battlemage", vram_mb=24576)],
                  paths=PathsConfig(models_dir=str(tmp_path / "arc-models"), state_dir=str(tmp_path / "state")))


def test_ollama_manifests_map_names_and_skip_unsafe_or_multipart_models(tmp_path):
    model_file(tmp_path / "blobs" / ("sha256-" + "a" * 64))
    model_file(tmp_path / "blobs" / ("sha256-" + "b" * 64), split_count=2)
    (tmp_path / "blobs" / ("sha256-" + "c" * 64)).write_bytes(b"not GGUF")
    manifest(tmp_path, "tiny", "sha256:" + "a" * 64)
    manifest(tmp_path, "split", "sha256:" + "b" * 64)
    manifest(tmp_path, "other-format", "sha256:" + "c" * 64)
    manifest(tmp_path, "unsafe", "sha256:../../outside")
    manifest(tmp_path, "multi-layer", "unused", layers=[
        {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:" + "a" * 64},
        {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:" + "b" * 64},
    ])
    found, skipped = discover("ollama", [tmp_path])
    assert [(item.name, item.path.name) for item in found] == [("tiny:latest", "sha256-" + "a" * 64)]
    assert len(skipped) == 4
    assert any("multi-part" in reason for reason in skipped)
    assert any("not a GGUF" in reason for reason in skipped)
    assert any("invalid blob digest" in reason for reason in skipped)


def test_ollama_aliases_reuse_the_same_file_and_preserve_projector(tmp_path):
    root = tmp_path / "ollama"
    blob = model_file(root / "blobs" / ("sha256-" + "a" * 64))
    projector = model_file(root / "blobs" / ("sha256-" + "d" * 64))
    for name in ("tiny", "alias"):
        manifest(root, name, "unused", layers=[
            {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:" + "a" * 64},
            {"mediaType": "application/vnd.ollama.image.projector", "digest": "sha256:" + "d" * 64},
        ])
    found, skipped = discover("ollama", [root])
    assert not skipped
    cfg = config(tmp_path)
    added = import_models(cfg, "ollama", found)
    assert len(added) == 1 and len(cfg.models) == 1
    assert set(cfg.models[0].aliases) == {"tiny:latest", "alias:latest"}
    assert cfg.models[0].recipe["mmproj"] == str(projector.resolve())
    assert Path(cfg.models[0].path).samefile(blob)


def test_lmstudio_scan_skips_shards_auxiliary_and_non_gguf(tmp_path):
    main = model_file(tmp_path / "owner" / "repo" / "tiny-Q4_K_M.gguf")
    model_file(tmp_path / "split-00001-of-00002.gguf")
    model_file(tmp_path / "split-00002-of-00002.gguf")
    model_file(tmp_path / "mmproj-other.gguf")
    (tmp_path / "bad.gguf").write_bytes(b"not GGUF")
    (tmp_path / "weights.safetensors").write_bytes(b"other format")
    found, skipped = discover("lmstudio", [tmp_path])
    assert [item.path for item in found] == [main.resolve()]
    assert len(skipped) == 5
    assert any("non-GGUF" in reason for reason in skipped)
    assert sum("multi-part" in reason for reason in skipped) == 2


def test_reference_import_is_idempotent_persistent_and_protected(tmp_path, monkeypatch):
    import arc_llama.model_import as importer

    monkeypatch.setattr(importer, "_is_windows", lambda: False)
    original = model_file(tmp_path / "source" / "tiny.gguf")
    before = original.read_bytes(), original.stat().st_mtime_ns
    cfg = config(tmp_path)
    # Even overlapping Arc/other-app directories must never make sources deletable.
    cfg.paths.models_dir = str(original.parent)
    candidates = [ImportCandidate("tiny", original)]
    added = import_models(cfg, "lmstudio", candidates)
    assert len(added) == 1 and Path(added[0].path) == original.resolve()
    assert import_models(cfg, "lmstudio", candidates) == []
    assert deletable_files(cfg, added[0]) == []
    assert disk_report(cfg)["models"][0]["managed"] is False
    cfg.save(tmp_path / "config.toml")
    loaded = load_config(tmp_path / "config.toml")
    assert loaded.models[0].import_source == "lmstudio"
    assert deletable_files(loaded, loaded.models[0]) == []
    assert (original.read_bytes(), original.stat().st_mtime_ns) == before
    assert list(original.parent.iterdir()) == [original]


def test_windows_same_volume_uses_hardlink_without_symlink(tmp_path, monkeypatch):
    import arc_llama.model_import as importer

    monkeypatch.setattr(importer, "_is_windows", lambda: True)
    original = model_file(tmp_path / "source" / "tiny.gguf")
    cfg = config(tmp_path)
    added = import_models(cfg, "lmstudio", [ImportCandidate("tiny", original)])
    linked = Path(added[0].path)
    assert linked != original and linked.samefile(original)
    assert not linked.is_symlink()
    assert added[0].import_source == "lmstudio"
    assert added[0].import_link_path == str(linked)
    assert deletable_files(cfg, added[0]) == [linked]
    assert disk_report(cfg)["models"][0]["managed"] is True
    assert len(import_models(cfg, "lmstudio", [ImportCandidate("tiny", original)])) == 0
    cfg.save(tmp_path / "config.toml")
    loaded = load_config(tmp_path / "config.toml")
    assert deletable_files(loaded, loaded.models[0]) == [linked]


def test_source_under_imports_is_not_owned_and_shared_link_is_protected(tmp_path, monkeypatch):
    from dataclasses import replace

    import arc_llama.model_import as importer

    cfg = config(tmp_path)
    original = model_file(Path(cfg.paths.models_dir) / "imports" / "lmstudio" / "source.gguf")
    monkeypatch.setattr(importer, "_is_windows", lambda: False)
    model = import_models(cfg, "lmstudio", [ImportCandidate("source", original)])[0]
    assert deletable_files(cfg, model) == []
    assert not disk_report(cfg)["models"][0]["managed"]
    cfg.models.clear()
    monkeypatch.setattr(importer, "_is_windows", lambda: True)
    model = import_models(cfg, "lmstudio", [ImportCandidate("source", original)])[0]
    link = Path(model.path)
    other = replace(model, name="other")
    cfg.models.append(other)
    assert deletable_files(cfg, model) == []
    cfg.models.remove(other)
    # Same inode is insufficient: pointing at the source must never delete it.
    model.path = str(original)
    assert deletable_files(cfg, model) == []
    model.path = str(link)
    assert deletable_files(cfg, model) == [link]


def test_windows_cross_volume_or_denied_link_falls_back_to_reference(tmp_path, monkeypatch):
    import arc_llama.model_import as importer

    monkeypatch.setattr(importer, "_is_windows", lambda: True)
    def cross_volume(*args):
        raise OSError(errno.EXDEV, "different volumes")
    monkeypatch.setattr(importer.os, "link", cross_volume)
    original = model_file(tmp_path / "source" / "tiny.gguf")
    added = import_models(config(tmp_path), "lmstudio", [ImportCandidate("tiny", original)])
    assert Path(added[0].path) == original.resolve()
    assert original.is_file()


@pytest.mark.parametrize("windows", [False, True])
def test_platform_default_roots_and_ollama_override(tmp_path, monkeypatch, windows):
    import arc_llama.model_import as importer

    monkeypatch.setattr(importer, "_is_windows", lambda: windows)
    monkeypatch.setattr(importer.Path, "home", lambda: tmp_path / "linux-home")
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "windows-profile"))
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    home = tmp_path / ("windows-profile" if windows else "linux-home")
    assert default_roots("ollama") == [home / ".ollama" / "models"]
    assert default_roots("lmstudio") == [home / ".lmstudio" / "models", home / ".cache" / "lm-studio" / "models"]
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "other-drive"))
    assert default_roots("ollama") == [tmp_path / "other-drive"]


def test_cli_dry_run_and_real_import_do_not_copy_source(tmp_path, monkeypatch):
    import arc_llama.model_import as importer
    from arc_llama.cli import cli

    monkeypatch.setattr(importer, "_is_windows", lambda: False)
    source = tmp_path / "source"
    original = model_file(source / "tiny.gguf")
    before = original.read_bytes()
    cfg_path = tmp_path / "arc.toml"
    runner = CliRunner()
    args = ["-c", str(cfg_path), "import", "--from", "lmstudio", "--root", str(source)]
    result = runner.invoke(cli, [*args, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Would import: tiny" in result.output
    assert not cfg_path.exists()
    config(tmp_path).save(cfg_path)
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    loaded = load_config(cfg_path)
    assert len(loaded.models) == 1 and Path(loaded.models[0].path) == original.resolve()
    assert original.read_bytes() == before
    assert not (tmp_path / "arc-models").exists()


def test_cli_ollama_dry_run_lists_manifest_name(tmp_path):
    from arc_llama.cli import cli

    model_file(tmp_path / "blobs" / ("sha256-" + "a" * 64))
    manifest(tmp_path, "tiny", "sha256:" + "a" * 64)
    result = CliRunner().invoke(cli, ["-c", str(tmp_path / "absent.toml"), "import", "--from", "ollama", "--root", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Would import: tiny:latest" in result.output
    assert not (tmp_path / "absent.toml").exists()


def test_library_delete_only_unregisters_imported_source_files(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from helpers import ConfigRouter

    import arc_llama.model_import as importer
    import arc_llama.server as server
    from arc_llama.config import ServerConfig

    class Router(ConfigRouter):
        async def stop_one(self, name):
            return False
    monkeypatch.setattr(importer, "_is_windows", lambda: False)
    monkeypatch.setattr(server, "Router", Router)
    original = model_file(tmp_path / "source" / "tiny.gguf")
    cfg = config(tmp_path)
    cfg.server = ServerConfig(admin_token="test-import-token")
    cfg.tune.auto = False
    cfg.paths.models_dir = str(original.parent)
    model = import_models(cfg, "lmstudio", [ImportCandidate("tiny", original)])[0]
    path = tmp_path / "arc.toml"
    cfg.save(path)
    with TestClient(server.create_app(cfg, config_path=path, plugins=[])) as client:
        response = client.delete(f"/admin/library/models/{model.name}?delete_files=true",
                                 headers={"Authorization": "Bearer test-import-token"})
        assert response.status_code == 200
        assert response.json()["deleted_files"] == 0
    assert original.is_file()
    assert load_config(path).models == []


@pytest.mark.parametrize("remove_original", [False, True])
def test_library_delete_removes_only_arc_created_hardlink(tmp_path, monkeypatch, remove_original):
    from fastapi.testclient import TestClient
    from helpers import ConfigRouter

    import arc_llama.model_import as importer
    import arc_llama.server as server
    from arc_llama.config import ServerConfig

    class Router(ConfigRouter):
        async def stop_one(self, name):
            return False
    monkeypatch.setattr(importer, "_is_windows", lambda: True)
    monkeypatch.setattr(server, "Router", Router)
    original = model_file(tmp_path / "source" / "tiny.gguf")
    before = original.read_bytes(), original.stat().st_mtime_ns
    cfg = config(tmp_path)
    cfg.server = ServerConfig(admin_token="test-import-token")
    cfg.tune.auto = False
    model = import_models(cfg, "lmstudio", [ImportCandidate("tiny", original)])[0]
    link = Path(model.path)
    assert link != original and link.samefile(original)
    if remove_original:
        original.unlink()  # The other app removes its own directory entry.
    config_path = tmp_path / "arc.toml"
    cfg.save(config_path)
    with TestClient(server.create_app(cfg, config_path=config_path, plugins=[])) as client:
        response = client.delete(f"/admin/library/models/{model.name}?delete_files=true",
                                 headers={"Authorization": "Bearer test-import-token"})
        assert response.status_code == 200
        assert response.json()["deleted_files"] == 1
        assert response.json()["failed"] == []
    assert not link.exists()
    if not remove_original:
        assert (original.read_bytes(), original.stat().st_mtime_ns) == before
        assert response.json()["freed_mb"] == 0
    assert load_config(config_path).models == []


def test_ollama_manifest_list_selects_installed_gguf_variant(tmp_path):
    blob = model_file(tmp_path / "blobs" / ("sha256-" + "a" * 64))
    child = {"layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:" + "a" * 64}]}
    (tmp_path / "blobs" / ("sha256-" + "e" * 64)).write_text(json.dumps(child))
    manifest(tmp_path, "modern", "unused", manifests=[
        {"runner": "mlx", "format": "safetensors", "digest": "sha256:" + "f" * 64},
        {"runner": "llamacpp", "format": "gguf", "digest": "sha256:" + "e" * 64},
    ])
    found, skipped = discover("ollama", [tmp_path])
    assert not skipped
    assert [(item.name, item.path) for item in found] == [("modern:latest", blob.resolve())]


def test_cyclic_or_non_gguf_manifest_lists_are_skipped(tmp_path):
    children = [{"runner": "llamacpp", "format": "gguf", "digest": "sha256:" + "e" * 64}]
    (tmp_path / "blobs").mkdir()
    (tmp_path / "blobs" / ("sha256-" + "e" * 64)).write_text(json.dumps({"manifests": children}))
    manifest(tmp_path, "cycle", "unused", manifests=children)
    manifest(tmp_path, "safetensors", "unused", manifests=[{"runner": "mlx", "format": "safetensors"}])
    found, skipped = discover("ollama", [tmp_path])
    assert not found and len(skipped) == 2
    assert all("no installed single-file GGUF" in reason for reason in skipped)
