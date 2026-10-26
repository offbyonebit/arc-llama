"""In-app OpenCode/Hermes installs: synthetic archives, mock GitHub, fake subprocesses."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import threading
import zipfile
from pathlib import Path

import httpx
import pytest

import arc_llama.harness_install as hi
from arc_llama.config import Config, PathsConfig, load_config


def tar_bytes(files: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(data))
    return out.getvalue()


def zip_bytes(files: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return out.getvalue()


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_ADMIN_TOKEN", "synthetic")
    config = Config(paths=PathsConfig(state_dir=str(tmp_path / "state")))
    path = tmp_path / "config.toml"
    config.save(path)
    return config, path


class FakeProc:
    """Popen stand-in: records the command, prints a line, exits with *code*."""

    def __init__(self, args, code=0, out="ok\n", effect=None, **kwargs):
        self.args, self.kwargs, self.returncode = args, kwargs, None
        self._code = code
        self.stdout = io.StringIO(out)
        if effect:
            effect(args, kwargs)

    def poll(self):
        self.returncode = self._code
        return self.returncode

    def wait(self):
        return self.poll()

    def kill(self):
        self._code = -9


def runner_for(calls, *, fail_on=None, effect=None):
    def run(args, **kwargs):
        calls.append(list(args))
        code = 1 if fail_on and fail_on in " ".join(map(str, args)) else 0
        return FakeProc(args, code=code, out="synthetic failure\n" if code else "ok\n", effect=effect, **kwargs)

    return run


def opencode_release(tag: str, payload: bytes, name: str, digest: str | None = "auto") -> dict:
    if digest == "auto":
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    return {"tag_name": tag, "assets": [{
        "name": name, "browser_download_url": f"https://synthetic.invalid/{name}",
        "size": len(payload), "digest": digest,
    }]}


def mock_client(routes: dict[str, httpx.Response]):
    def handler(request: httpx.Request) -> httpx.Response:
        for prefix, response in routes.items():
            if str(request.url).startswith(prefix):
                return response
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    "os_name,arch,musl,avx2,expected",
    [
        ("linux", "x64", False, True, "opencode-linux-x64.tar.gz"),
        ("linux", "x64", False, False, "opencode-linux-x64-baseline.tar.gz"),
        ("linux", "x64", True, True, "opencode-linux-x64-musl.tar.gz"),
        ("linux", "x64", True, False, "opencode-linux-x64-baseline-musl.tar.gz"),
        ("linux", "arm64", False, False, "opencode-linux-arm64.tar.gz"),
        ("windows", "x64", False, True, "opencode-windows-x64.zip"),
        ("windows", "x64", False, False, "opencode-windows-x64-baseline.zip"),
    ],
)
def test_opencode_asset_selection(os_name, arch, musl, avx2, expected):
    assert hi.opencode_asset_name(os_name, arch, musl=musl, avx2=avx2) == expected


def test_opencode_asset_selection_rejects_unsupported():
    with pytest.raises(hi.HarnessInstallError):
        hi.opencode_asset_name("macos", "arm64", musl=False, avx2=True)


def _linux(monkeypatch):
    monkeypatch.setattr(hi, "host_platform", lambda: ("linux", "x64"))
    monkeypatch.setattr(hi, "detect_musl", lambda: False)
    monkeypatch.setattr(hi, "detect_avx2", lambda: True)
    monkeypatch.setattr(hi.sys, "platform", "linux")


def test_opencode_install_verifies_and_selects(cfg, monkeypatch):
    config, path = cfg
    _linux(monkeypatch)
    payload = tar_bytes({"opencode": b"#!/bin/sh\necho 1.19.4\n"})
    name = "opencode-linux-x64.tar.gz"
    release = opencode_release(hi.OPENCODE_PIN, payload, name)
    calls: list = []
    with mock_client({
        f"{hi.GITHUB_API}/repos/{hi.OPENCODE_REPO}/releases/tags/{hi.OPENCODE_PIN}": httpx.Response(200, json=release),
        f"https://synthetic.invalid/{name}": httpx.Response(200, content=payload),
    }) as client:
        result = hi.install_harness(config, "opencode", config_path=path, client=client, runner=runner_for(calls))
    assert result.tested and result.version == hi.OPENCODE_PIN
    assert result.binary.read_bytes() == b"#!/bin/sh\necho 1.19.4\n"
    # The smoke test ran on the staged copy, before publication.
    [[exe, flag]] = calls
    assert flag == "--version" and Path(exe).name == "opencode" and ".staging-" in exe
    assert load_config(path).agent.opencode_bin == str(result.binary)
    marker = json.loads((result.install_dir / hi.MARKER).read_text())
    assert marker["sha256"] == hashlib.sha256(payload).hexdigest() and marker["license"] == "MIT"
    [item] = hi.managed_installs(config, "opencode")
    assert item["version"] == hi.OPENCODE_PIN and hi.is_managed(config, item["path"])
    leftovers = [p.name for p in result.install_dir.parent.iterdir() if p != result.install_dir]
    assert leftovers == []


@pytest.mark.parametrize("problem", ["digest_mismatch", "no_digest", "unsafe_member", "no_executable", "smoke_fails"])
def test_opencode_install_failures_leave_nothing_selected(cfg, monkeypatch, problem):
    config, path = cfg
    _linux(monkeypatch)
    files = {"opencode": b"binary"}
    if problem == "unsafe_member":
        files = {"../escape": b"x", "opencode": b"binary"}
    if problem == "no_executable":
        files = {"README": b"nothing here"}
    payload = tar_bytes(files)
    name = "opencode-linux-x64.tar.gz"
    digest = {"digest_mismatch": "sha256:" + "0" * 64, "no_digest": None}.get(problem, "auto")
    release = opencode_release(hi.OPENCODE_PIN, payload, name, digest=digest)
    calls: list = []
    with mock_client({
        f"{hi.GITHUB_API}/repos/{hi.OPENCODE_REPO}/releases/tags/": httpx.Response(200, json=release),
        f"https://synthetic.invalid/{name}": httpx.Response(200, content=payload),
    }) as client, pytest.raises(hi.HarnessInstallError):
        hi.install_harness(config, "opencode", config_path=path, client=client,
                           runner=runner_for(calls, fail_on="--version" if problem == "smoke_fails" else None))
    assert load_config(path).agent.opencode_bin is None
    assert hi.managed_installs(config, "opencode") == []
    root = hi.harness_root(config) / "opencode"
    assert not root.exists() or list(root.iterdir()) == []
    assert not (Path(config.paths.state_dir).parent / "escape").exists()


def test_opencode_windows_zip_and_latest_is_untested(cfg, monkeypatch):
    config, path = cfg
    monkeypatch.setattr(hi, "host_platform", lambda: ("windows", "x64"))
    monkeypatch.setattr(hi, "detect_musl", lambda: False)
    monkeypatch.setattr(hi, "detect_avx2", lambda: True)
    monkeypatch.setattr(hi.sys, "platform", "win32")
    payload = zip_bytes({"opencode.exe": b"MZ synthetic"})
    name = "opencode-windows-x64.zip"
    release = opencode_release("v9.9.9", payload, name)
    with mock_client({
        f"{hi.GITHUB_API}/repos/{hi.OPENCODE_REPO}/releases/latest": httpx.Response(200, json=release),
        f"https://synthetic.invalid/{name}": httpx.Response(200, content=payload),
    }) as client:
        result = hi.install_harness(config, "opencode", version="latest", config_path=path, client=client,
                                    runner=runner_for([]))
    assert result.version == "v9.9.9" and result.tested is False
    assert result.binary.name == "opencode.exe" and result.install_dir.name == "opencode-v9.9.9-windows-x64"


def test_opencode_killed_mid_publish_is_recovered_on_next_install(cfg, monkeypatch):
    config, path = cfg
    _linux(monkeypatch)
    root = hi.harness_root(config) / "opencode"
    old = root / f"opencode-{hi.OPENCODE_PIN}-linux-x64"
    backup = root / f".{old.name}.backup-abc123"
    backup.mkdir(parents=True)
    (backup / "opencode").write_bytes(b"previous")
    (backup / hi.MARKER).write_text(json.dumps({"name": "opencode", "version": hi.OPENCODE_PIN,
                                                "executable": "opencode", "installed_at": 1}))
    payload = tar_bytes({"opencode": b"new"})
    name = "opencode-linux-x64.tar.gz"
    with mock_client({
        f"{hi.GITHUB_API}/repos/{hi.OPENCODE_REPO}/releases/tags/": httpx.Response(200, json=opencode_release(hi.OPENCODE_PIN, payload, name)),
        f"https://synthetic.invalid/{name}": httpx.Response(503),
    }) as client, pytest.raises(hi.HarnessInstallError):
        hi.install_harness(config, "opencode", config_path=path, client=client, runner=runner_for([]))
    # The download failed, but the backup a killed run left behind was restored first.
    assert (old / "opencode").read_bytes() == b"previous"
    assert [p.name for p in root.iterdir()] == [old.name]


# ---------------------------------------------------------------------------
# Hermes
# ---------------------------------------------------------------------------


def hermes_source(tag: str) -> bytes:
    top = f"hermes-agent-{tag.lstrip('v')}"
    return tar_bytes({f"{top}/pyproject.toml": b"[project]\nname='hermes-agent'\n",
                      f"{top}/uv.lock": b"version = 1\n", f"{top}/acp_adapter/entry.py": b"\n"})


def hermes_routes(payload: bytes, tag: str) -> dict[str, httpx.Response]:
    return {
        f"https://github.com/{hi.HERMES_REPO}/archive/refs/tags/{tag}.tar.gz": httpx.Response(200, content=payload),
        f"{hi.GITHUB_API}/repos/{hi.HERMES_REPO}/releases/latest": httpx.Response(200, json={"tag_name": tag}),
    }


def fake_uv_sync(args, kwargs):
    """uv sync creates the venv's hermes command, as the real one does."""
    if len(args) > 1 and args[1] == "sync":
        venv = Path(kwargs["env"]["UV_PROJECT_ENVIRONMENT"])
        exe = venv / ("Scripts/hermes.exe" if sys.platform == "win32" else "bin/hermes")
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_text("#!/bin/sh\n")


def _pin_hermes(monkeypatch, payload: bytes):
    monkeypatch.setattr(hi, "HERMES_PIN", hi.SourcePin("v0.22.5", "c" * 40, hashlib.sha256(payload).hexdigest(), len(payload)))


def test_hermes_install_uses_locked_sync_and_isolated_dirs(cfg, monkeypatch, tmp_path):
    config, path = cfg
    _linux(monkeypatch)
    payload = hermes_source("v0.22.5")
    _pin_hermes(monkeypatch, payload)
    monkeypatch.setattr(hi.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    calls: list = []
    seen_env: dict = {}

    def effect(args, kwargs):
        fake_uv_sync(args, kwargs)
        if args[1] == "sync":
            seen_env.update(kwargs["env"])
            assert Path(kwargs["cwd"]).name == "src" and (Path(kwargs["cwd"]) / "uv.lock").is_file()

    with mock_client(hermes_routes(payload, "v0.22.5")) as client:
        result = hi.install_harness(config, "hermes", config_path=path, client=client,
                                    runner=runner_for(calls, effect=effect))
    assert calls[0][:3] == ["/usr/bin/uv", "sync", "--frozen"] and "--extra" in calls[0] and "acp" in calls[0]
    assert "--no-dev" in calls[0] and calls[0][-2:] == ["--python", hi.HERMES_PYTHON]
    assert calls[1][1:] == ["acp", "--check"]
    root = hi.harness_root(config)
    assert seen_env["UV_PROJECT_ENVIRONMENT"] == str(result.install_dir / "venv")
    assert seen_env["UV_PYTHON_INSTALL_DIR"].startswith(str(root)) and seen_env["UV_CACHE_DIR"].startswith(str(root))
    assert seen_env["UV_NO_CONFIG"] == "1" and "VIRTUAL_ENV" not in seen_env
    assert result.tested and load_config(path).agent.hermes_bin == str(result.binary)
    assert (result.install_dir / "src" / "uv.lock").is_file() and (result.install_dir / hi.MARKER).is_file()
    assert not any(p.name.startswith(".hermes-src-") for p in result.install_dir.parent.iterdir())


@pytest.mark.parametrize("fail_on", ["sync", "--check"])
def test_hermes_failed_step_removes_folder_and_keeps_config(cfg, monkeypatch, fail_on):
    config, path = cfg
    _linux(monkeypatch)
    payload = hermes_source("v0.22.5")
    _pin_hermes(monkeypatch, payload)
    monkeypatch.setattr(hi.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    with mock_client(hermes_routes(payload, "v0.22.5")) as client, pytest.raises(hi.HarnessInstallError, match="failed"):
        hi.install_harness(config, "hermes", config_path=path, client=client,
                           runner=runner_for([], fail_on=fail_on, effect=fake_uv_sync))
    assert load_config(path).agent.hermes_bin is None
    assert list((hi.harness_root(config) / "hermes").iterdir()) == []


def test_hermes_rejects_tampered_source(cfg, monkeypatch):
    config, path = cfg
    _linux(monkeypatch)
    _pin_hermes(monkeypatch, hermes_source("v0.22.5"))
    tampered = hermes_source("v0.22.5") + b"\0"
    monkeypatch.setattr(hi.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    calls: list = []
    with mock_client(hermes_routes(tampered, "v0.22.5")) as client, pytest.raises(hi.HarnessInstallError):
        hi.install_harness(config, "hermes", config_path=path, client=client, runner=runner_for(calls))
    assert calls == [] and load_config(path).agent.hermes_bin is None


def test_hermes_downloads_pinned_uv_when_missing(cfg, monkeypatch):
    config, path = cfg
    _linux(monkeypatch)
    payload = hermes_source("v0.22.5")
    _pin_hermes(monkeypatch, payload)
    monkeypatch.setattr(hi.shutil, "which", lambda name: None)
    uv_archive = tar_bytes({"uv-x86_64-unknown-linux-musl/uv": b"#!/bin/sh\n"})
    name = "uv-x86_64-unknown-linux-musl.tar.gz"
    monkeypatch.setitem(hi.UV_ASSETS, ("linux", "x64"), (name, hashlib.sha256(uv_archive).hexdigest(), len(uv_archive)))
    routes = {**hermes_routes(payload, "v0.22.5"),
              f"https://github.com/astral-sh/uv/releases/download/{hi.UV_VERSION}/{name}": httpx.Response(200, content=uv_archive)}
    calls: list = []
    with mock_client(routes) as client:
        hi.install_harness(config, "hermes", config_path=path, client=client, runner=runner_for(calls, effect=fake_uv_sync))
    uv = Path(calls[0][0])
    assert uv.name == "uv" and uv.is_relative_to(hi.harness_root(config) / "tools")


def test_hermes_latest_is_untested_and_needs_no_pin(cfg, monkeypatch):
    config, path = cfg
    _linux(monkeypatch)
    payload = hermes_source("v0.99.0")
    monkeypatch.setattr(hi.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    with mock_client(hermes_routes(payload, "v0.99.0")) as client:
        result = hi.install_harness(config, "hermes", version="latest", config_path=path, client=client,
                                    runner=runner_for([], effect=fake_uv_sync))
    assert result.version == "v0.99.0" and result.tested is False


def test_remove_clears_selection_and_only_touches_managed(cfg, monkeypatch):
    config, path = cfg
    install = hi.harness_root(config) / "opencode" / "opencode-v1-linux-x64"
    install.mkdir(parents=True)
    (install / "opencode").write_bytes(b"x")
    (install / hi.MARKER).write_text(json.dumps({"name": "opencode", "version": "v1", "executable": "opencode"}))
    config.agent.opencode_bin = str(install / "opencode")
    config.save(path)
    assert hi.remove_harness(config, "opencode", "v0", config_path=path) is False
    assert hi.remove_harness(config, "opencode", "v1", config_path=path) is True
    assert not install.exists() and load_config(path).agent.opencode_bin is None


def test_cancel_stops_a_running_command(cfg):
    class Hanging(FakeProc):
        killed = False

        def poll(self):
            return self._code if self.killed else None

        def kill(self):
            self.killed = True

    procs: list = []
    cancel = threading.Event()

    def run(args, **kw):
        procs.append(Hanging(args, code=-9, out="", **kw))
        return procs[-1]

    ctx = hi._Context(progress=lambda *_: None, log_line=lambda _l: None, cancel=cancel, runner=run)
    timer = threading.Timer(0.3, cancel.set)
    timer.start()
    with pytest.raises(hi.InstallCancelledError):
        hi._run(ctx, ["uv", "sync"], step="sync")
    assert procs[0].killed


def test_manager_runs_one_install_at_a_time_and_reports(cfg):
    config, path = cfg
    started, release = threading.Event(), threading.Event()

    def installer(cfg_, name, **kw):
        kw["progress"]("Downloading", 5, 10)
        kw["log_line"]("line")
        started.set()
        release.wait(5)
        return hi.InstallResult(name, "v1", Path("/x"), Path("/"), True)

    manager = hi.HarnessInstallManager(config, path, installer=installer)
    job = manager.submit("opencode")
    assert started.wait(5)
    with pytest.raises(ValueError, match="already running"):
        manager.submit("hermes")
    assert job.public()["step"] == "Downloading" and job.public()["log"] == ["line"]
    release.set()
    for _ in range(100):
        if job.status == "done":
            break
        threading.Event().wait(0.05)
    assert job.status == "done" and job.installed_version == "v1" and job.tested
    with pytest.raises(ValueError):
        manager.submit("opencode", "v1.2.3")


def test_removing_last_hermes_also_removes_its_tools(cfg):
    config, path = cfg
    root = hi.harness_root(config)
    for version in ("v1", "v2"):
        install = root / "hermes" / f"hermes-{version}-linux-x64"
        (install / "venv" / "bin").mkdir(parents=True)
        (install / "venv" / "bin" / "hermes").write_text("")
        (install / hi.MARKER).write_text(json.dumps({"name": "hermes", "version": version, "executable": "venv/bin/hermes"}))
    (root / "tools" / "python").mkdir(parents=True)
    assert hi.remove_harness(config, "hermes", "v1", config_path=path)
    assert (root / "tools").is_dir(), "tools removed while another Hermes still needs them"
    assert hi.remove_harness(config, "hermes", "v2", config_path=path)
    assert not (root / "tools").exists()
