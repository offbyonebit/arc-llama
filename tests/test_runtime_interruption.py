"""A runtime installer killed mid-publication must leave a recoverable install.

Each case runs a synthetic installer in a child interpreter that calls
os._exit (no except/finally blocks run) or pauses at a publication boundary.
These are process-exit tests in temporary directories, not power-loss tests.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import arc_llama.runtime as rt
from arc_llama.config import Config, PathsConfig

ROOT = Path(__file__).resolve().parents[1]
CHILD = ROOT / "tests/fixtures/runtime_install_child.py"
OLD = {"llama-server": b"previous verified executable\n", "libggml.so": b"previous library\n",
       ".arc-llama-runtime.json": b'{"previous": true}\n'}
NAME = "llama-b12345-vulkan"


def tree(path: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in path.iterdir()}


def leftovers(dest: Path) -> list[str]:
    return sorted(p.name for p in dest.iterdir() if p.name != NAME)


@pytest.fixture
def env(tmp_path):
    dest = tmp_path / "runtime"
    install = dest / NAME
    install.mkdir(parents=True)
    for name, data in OLD.items():
        (install / name).write_bytes(data)
    config_path = tmp_path / "config.toml"
    Config(paths=PathsConfig(state_dir=str(tmp_path), llama_server=str(install / "llama-server"))).save(
        config_path
    )
    return dest, install, config_path


def child(dest, config_path, point="none", action="exit", network="online", force="force",
          text="replacement executable\n", wait=True):
    child_env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "ARC_LLAMA_ADMIN_TOKEN": "synthetic"}
    args = [sys.executable, str(CHILD), str(dest), str(config_path), point, action, network, force, text]
    proc = subprocess.Popen(args, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if not wait:
        return proc
    out, err = proc.communicate(timeout=60)
    return proc.returncode, out, err


def result(out: str) -> dict:
    return json.loads(out.strip().splitlines()[-1])


def configured(config_path: Path) -> Path:
    from arc_llama.config import load_config

    return Path(load_config(config_path).paths.llama_server)


NEW = {"llama-server": b"replacement executable\n", "libggml.so": b"synthetic library\n"}


@pytest.mark.parametrize("point", ["backup_renamed", "published", "trash_renamed", "trash_partial"])
def test_killed_replacement_is_recovered_by_next_run_even_offline(env, point):
    dest, install, config_path = env
    code, _out, err = child(dest, config_path, point)
    assert code == 77, err
    if point == "backup_renamed":
        # The reproduced gap: the configured runtime is gone, the old install
        # survives only as a hidden backup, and a complete staging tree sits
        # beside it that recovery must not promote.
        assert not install.exists()
        assert not configured(config_path).exists()
        [backup] = [p for p in dest.iterdir() if ".backup-" in p.name]
        assert tree(backup) == OLD
        assert any(".staging-" in p.name for p in dest.iterdir())
        expected = OLD
    else:
        expected = {**NEW, ".arc-llama-runtime.json": (install / ".arc-llama-runtime.json").read_bytes()}

    code, out, err = child(dest, config_path, network="offline", force="reuse")
    assert code == 0, err
    assert result(out) == {"ok": False, "error": "ConnectError"}
    assert tree(install) == expected
    assert configured(config_path).read_bytes() == expected["llama-server"]
    assert not [n for n in leftovers(dest) if ".backup-" in n or ".trash-" in n]

    code, out, err = child(dest, config_path, text="retried executable\n")
    assert code == 0, err
    assert result(out)["ok"]
    assert configured(config_path).read_bytes() == b"retried executable\n"
    assert (install / "libggml.so").read_bytes() == NEW["libggml.so"]
    # The killed run's own download and staging copy are not reclaimed.
    assert all(".staging-" in n or n.startswith("tmp") for n in leftovers(dest))


def backup_beside(dest: Path, files: dict[str, bytes]) -> Path:
    backup = dest / f".{NAME}.backup-abc123"
    backup.mkdir(parents=True)
    for name, data in files.items():
        (backup / name).write_bytes(data)
    return backup


def test_failed_restore_keeps_the_only_complete_install(tmp_path, monkeypatch):
    dest = tmp_path / "runtime"
    backup = backup_beside(dest, OLD)
    real_replace = rt.os.replace

    def refuse(src, dst):
        if Path(src) == backup:
            raise PermissionError("synthetic restore failure")
        return real_replace(src, dst)

    monkeypatch.setattr(rt.os, "replace", refuse)
    assert rt._recover_interrupted_publish(dest) == []
    assert tree(backup) == OLD and not (dest / NAME).exists()
    monkeypatch.setattr(rt.os, "replace", real_replace)
    assert rt._recover_interrupted_publish(dest) == [dest / NAME]
    assert tree(dest / NAME) == OLD and leftovers(dest) == []


def test_recovery_never_promotes_staging_incomplete_backup_or_symlink(tmp_path):
    dest = tmp_path / "runtime"
    incomplete = backup_beside(dest, {"libggml.so": b"no executable\n"})
    staging = dest / f".{NAME}.staging-xyz789"
    staging.mkdir()
    (staging / "llama-server").write_bytes(b"unverified\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "llama-server").write_bytes(b"not an install\n")
    link = dest / ".llama-b99999-vulkan.backup-link00"
    try:
        link.symlink_to(elsewhere, target_is_directory=True)
    except OSError:
        link = None  # Windows without symlink privilege
    assert rt._recover_interrupted_publish(dest) == []
    assert not (dest / NAME).exists() and not (dest / "llama-b99999-vulkan").exists()
    assert tree(incomplete) == {"libggml.so": b"no executable\n"}
    assert tree(staging) == {"llama-server": b"unverified\n"}
    assert link is None or link.is_symlink()


def test_failed_cleanup_keeps_backup_and_install(tmp_path, monkeypatch):
    dest = tmp_path / "runtime"
    install = dest / NAME
    install.mkdir(parents=True)
    (install / "llama-server").write_bytes(b"current\n")
    backup = backup_beside(dest, OLD)
    real_replace = rt.os.replace

    def refuse(src, dst):
        if ".trash-" in Path(dst).name:
            raise PermissionError("synthetic cleanup failure")
        return real_replace(src, dst)

    monkeypatch.setattr(rt.os, "replace", refuse)
    rt._recover_interrupted_publish(dest)
    assert tree(install) == {"llama-server": b"current\n"}
    assert tree(backup) == OLD


def wait_reached(dest: Path, proc) -> None:
    deadline = time.monotonic() + 30
    while not (dest.parent / "reached").exists():
        assert proc.poll() is None, proc.communicate()
        assert time.monotonic() < deadline
        time.sleep(0.02)


def test_overlap_second_installer_while_first_has_install_moved_aside(env):
    dest, install, config_path = env
    first = child(dest, config_path, "backup_renamed", "pause", text="first\n", wait=False)
    wait_reached(dest, first)
    code, out, err = child(dest, config_path, text="second\n")
    assert code == 0 and result(out)["ok"], err
    (dest.parent / "go").write_text("")
    out, err = first.communicate(timeout=60)
    # The first installer finds the install back in place and fails rather
    # than replacing it; the second's complete install is what remains.
    # Linux reports ENOTEMPTY (OSError); Windows refusal types are not verified.
    assert first.returncode == 0, err
    assert result(out) in ({"ok": False, "error": e} for e in ("OSError", "PermissionError", "FileExistsError"))
    assert tree(install)["llama-server"] == b"second\n"
    assert configured(config_path).read_bytes() == b"second\n"
    assert not [n for n in leftovers(dest) if ".backup-" in n or ".trash-" in n]


def test_overlap_second_installer_while_first_is_retiring_backup(env):
    dest, install, config_path = env
    first = child(dest, config_path, "published", "pause", text="first\n", wait=False)
    wait_reached(dest, first)
    code, out, err = child(dest, config_path, network="offline", force="reuse")
    assert code == 0 and result(out) == {"ok": False, "error": "ConnectError"}, err
    (dest.parent / "go").write_text("")
    out, err = first.communicate(timeout=60)
    assert first.returncode == 0 and result(out)["ok"], err
    assert tree(install)["llama-server"] == b"first\n"
    assert configured(config_path).read_bytes() == b"first\n"
    assert leftovers(dest) == []
