"""Arc Llama Runtime install: signature and checksum gates, hardware gate, tuned per-model settings."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from arc_llama import optimized_runtime as opt
from arc_llama.cli import cli
from arc_llama.config import Config, GPUConfig, ModelConfig, load_config
from arc_llama.launcher import build_model_env

pytestmark = pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="needs OpenSSH ssh-keygen")

NAME = "arc-llama-runtime-0.1.0+b11624-linux-x64-sycl-bmg"
MANIFEST = {
    "schema": 1,
    "version": "0.1.0",
    "model_defaults": [
        {"architecture": "gemma4", "moe": True, "env": {"ARC_RT_C": "128", "ARC_RT_A": "0"}},
        {"architecture": "qwen35", "moe": True, "env": {"ARC_RT_N": "1"}},
        {"architecture": "qwen3moe", "env": {"ARC_RT_C": "64", "LD_PRELOAD": "/evil.so"}},
    ],
}


def _key(tmp_path: Path, name: str) -> tuple[Path, Path]:
    key = tmp_path / name
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    signers = tmp_path / f"{name}.allowed_signers"
    signers.write_text(f"{opt.SIGNER_ID} {key.with_suffix('.pub').read_text()}")
    return key, signers


def _release(tmp_path: Path, key: Path, *, manifest: dict | None = MANIFEST, tamper: bool = False) -> Path:
    """Build a release folder: archive, SHA256SUMS and its signature."""
    src = tmp_path / "src" / NAME
    src.mkdir(parents=True)
    server = src / "llama-server"
    server.write_text("#!/bin/sh\necho llama-server\n")
    server.chmod(0o755)
    if manifest is not None:
        (src / opt.MANIFEST_NAME).write_text(json.dumps(manifest))
    out = tmp_path / "dist"
    out.mkdir()
    archive = out / f"{NAME}.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(src, arcname=NAME)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out / opt.SUMS_NAME).write_text(f"{digest}  {archive.name}\n")
    subprocess.run(["ssh-keygen", "-Y", "sign", "-q", "-f", str(key), "-n", opt.SIGNING_NAMESPACE,
                    str(out / opt.SUMS_NAME)], check=True)
    if tamper:
        with open(archive, "ab") as f:
            f.write(b"tampered")
    return archive


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch) -> Config:
    monkeypatch.setattr("arc_llama.launcher.oneapi_runtime_env_needed", lambda: False)
    cfg = Config()
    cfg.tune.auto = False
    cfg.paths.state_dir = str(tmp_path / "state")
    cfg.paths.llama_server = str(tmp_path / "stock" / "llama-server")
    cfg.gpus = [GPUConfig(pci_slot="test-gpu", sycl_index=0, arch="battlemage", backend="vulkan")]
    return cfg


def test_verified_archive_installs_and_becomes_the_default(tmp_path, cfg):
    key, signers = _key(tmp_path, "release")
    archive = _release(tmp_path, key)
    config_path = tmp_path / "config.toml"
    result = opt.install_optimized_runtime(cfg=cfg, config_path=config_path, archive=archive,
                                           allowed_signers=signers, check_support=False)
    assert result.binary_path.name == "llama-server" and result.binary_path.is_file()
    assert result.install_dir.name == "arc-llama-runtime-0.1.0+b11624" and result.tag == "b11624"
    marker = json.loads((result.install_dir / opt.MARKER_NAME).read_text())
    assert marker["source"] == "arc-llama-runtime" and marker["version"] == "0.1.0"
    saved = load_config(config_path)
    assert saved.paths.llama_server == str(result.binary_path)
    assert all(g.backend == "sycl" for g in saved.gpus)
    assert opt.is_optimized_runtime(result.binary_path)


def test_reinstall_reuses_the_verified_install(tmp_path, cfg):
    key, signers = _key(tmp_path, "release")
    archive = _release(tmp_path, key)
    first = opt.install_optimized_runtime(cfg=cfg, config_path=tmp_path / "c.toml", archive=archive,
                                          allowed_signers=signers, check_support=False)
    (first.binary_path).write_text("#!/bin/sh\necho kept\n")
    again = opt.install_optimized_runtime(cfg=cfg, config_path=tmp_path / "c.toml", archive=archive,
                                          allowed_signers=signers, check_support=False)
    assert again.binary_path == first.binary_path and "kept" in again.binary_path.read_text()


@pytest.mark.parametrize("problem", ["tampered", "wrong-key", "unsigned", "unlisted", "no-manifest"])
def test_nothing_is_installed_unless_signature_checksum_and_manifest_check_out(tmp_path, cfg, problem):
    key, signers = _key(tmp_path, "release")
    if problem == "wrong-key":
        other, _ = _key(tmp_path, "attacker")
        archive = _release(tmp_path, other)
    else:
        archive = _release(tmp_path, key, tamper=problem == "tampered",
                           manifest=None if problem == "no-manifest" else MANIFEST)
    if problem == "unsigned":
        (archive.parent / opt.SIG_NAME).unlink()
    if problem == "unlisted":
        renamed = archive.with_name("arc-llama-runtime-9.9.9+b1-linux-x64-sycl-bmg.tar.gz")
        archive.rename(renamed)
        archive = renamed
    config_path = tmp_path / "config.toml"
    with pytest.raises(opt.OptimizedRuntimeError):
        opt.install_optimized_runtime(cfg=cfg, config_path=config_path, archive=archive,
                                      allowed_signers=signers, check_support=False)
    runtime_dir = Path(cfg.paths.state_dir) / "runtime"
    assert not list(runtime_dir.glob("arc-llama-runtime-*")), "nothing may be published"
    assert not config_path.exists(), "the config must not change"


def test_bundled_release_key_is_a_valid_allowed_signers_entry():
    line = opt.ALLOWED_SIGNERS.read_text().strip()
    principal, keytype, blob = line.split()[:3]
    assert principal == opt.SIGNER_ID and keytype == "ssh-ed25519" and len(blob) > 60


def test_support_gate(cfg, monkeypatch):
    monkeypatch.setattr(opt.platform, "system", lambda: "Linux")
    monkeypatch.setattr(opt.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr("arc_llama.platform_checks.oneapi_setvars_path", lambda: Path("/opt/intel/oneapi/setvars.sh"))
    assert opt.support_problems(cfg) == []
    cfg.gpus[0].arch = "alchemist"
    assert any("Battlemage" in p for p in opt.support_problems(cfg))
    cfg.gpus[0].arch = "battlemage"
    monkeypatch.setattr(opt.platform, "system", lambda: "Windows")
    assert any("Linux" in p for p in opt.support_problems(cfg))
    monkeypatch.setattr(opt.platform, "system", lambda: "Linux")
    monkeypatch.setattr("arc_llama.platform_checks.oneapi_setvars_path", lambda: None)
    cfg.paths.oneapi_setvars = None
    assert any("oneAPI" in p for p in opt.support_problems(cfg))


def test_install_refuses_unsupported_hosts_before_touching_anything(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(opt, "support_problems", lambda c: ["it needs an Intel Arc B-series (Battlemage) GPU"])
    with pytest.raises(opt.OptimizedRuntimeError, match="Battlemage"):
        opt.install_optimized_runtime(cfg=cfg, config_path=tmp_path / "c.toml", archive=tmp_path / "x.tar.gz")
    assert not (Path(cfg.paths.state_dir) / "runtime").exists()


def _runtime_with_manifest(tmp_path: Path, manifest: dict) -> Path:
    d = tmp_path / "rt"
    d.mkdir()
    binary = d / "llama-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    (d / opt.MANIFEST_NAME).write_text(json.dumps(manifest))
    return binary


def test_model_defaults_match_architecture_and_moe_and_only_arc_rt_keys(tmp_path, metadata_ggufs):
    binary = _runtime_with_manifest(tmp_path, MANIFEST)
    assert opt.model_runtime_defaults(binary, metadata_ggufs["gemma_moe"]) == {"ARC_RT_C": "128", "ARC_RT_A": "0"}
    assert opt.model_runtime_defaults(binary, metadata_ggufs["base"]) == {}  # qwen35 but not MoE
    assert opt.model_runtime_defaults(binary, metadata_ggufs["qwen_moe"]) == {"ARC_RT_C": "64"}  # LD_PRELOAD dropped
    assert opt.model_runtime_defaults(tmp_path / "stock-llama-server", metadata_ggufs["gemma_moe"]) == {}
    assert opt.model_runtime_defaults(binary, tmp_path / "missing.gguf") == {}


def test_launch_env_gets_runtime_defaults_and_model_env_wins(tmp_path, cfg, metadata_ggufs):
    binary = _runtime_with_manifest(tmp_path, MANIFEST)
    tuned = ModelConfig(name="gemma", path=str(metadata_ggufs["gemma_moe"]), port=18200, gpu_pci_slot="test-gpu",
                        llama_server=str(binary), env={"ARC_RT_C": "96"})
    stock = ModelConfig(name="stock", path=str(metadata_ggufs["gemma_moe"]), port=18201, gpu_pci_slot="test-gpu")
    env = build_model_env(cfg, tuned, cfg.gpus[0])
    assert env["ARC_RT_A"] == "0" and env["ARC_RT_C"] == "96"
    assert not any(k.startswith("ARC_RT_") for k in build_model_env(cfg, stock, cfg.gpus[0]))


def test_cli_rejects_archive_without_optimized_and_points_back_to_stock(tmp_path, cfg, monkeypatch):
    config_path = tmp_path / "config.toml"
    cfg.save(config_path)
    archive = tmp_path / "x.tar.gz"
    archive.write_bytes(b"")
    runner = CliRunner()
    r = runner.invoke(cli, ["--config", str(config_path), "install-runtime", "--archive", str(archive)])
    assert r.exit_code == 1 and "--archive needs --optimized" in r.output
    monkeypatch.setattr(opt, "support_problems", lambda c: ["it is built for Linux x86-64 only"])
    r = runner.invoke(cli, ["--config", str(config_path), "install-runtime", "--optimized", "--archive", str(archive)])
    assert r.exit_code == 1 and "Linux x86-64" in r.output and "Stock llama.cpp is unchanged" in r.output
