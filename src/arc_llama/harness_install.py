"""Install the external Work harnesses (OpenCode, Hermes Agent) from inside Arc Llama.

Each harness is installed per version under ``<state_dir>/harnesses/<name>/``
and selected by recording its executable in ``agent.opencode_bin`` or
``agent.hermes_bin``. The user's own installations, ``~/.hermes`` and
OpenCode's data are never touched.

* OpenCode publishes single-file CLI archives on GitHub releases. The archive
  is checked against GitHub's SHA-256 asset digest, extracted safely, and
  published with the same rename-and-recover steps as llama.cpp runtimes.
* Hermes publishes no binaries. Its tagged source is checked against a pinned
  SHA-256 and kept in the install folder: Hermes refuses wheel builds and
  supports only an editable install. ``uv sync --frozen`` then builds a private
  environment from the project's hash-locked ``uv.lock``, and uv supplies the
  Python it needs. A venv cannot be moved after creation, so it is built in its
  final folder and only counts as installed once ``hermes acp --check`` passes
  and the marker exists.

Pinned versions are the ones Arc Llama was verified against. ``latest`` is
allowed and recorded as untested.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from arc_llama.config import Config
from arc_llama.runtime import (
    GITHUB_API,
    RuntimeAsset,
    RuntimeInstallError,
    _auth_headers,
    _normalise_sha256_digest,
    _publish_install,
    _recover_interrupted_publish,
    _runtime_ssl_context,
    download_asset,
    extract_archive_files,
    host_platform,
)

log = logging.getLogger("arc_llama.harness_install")

HARNESSES = ("opencode", "hermes")
LABELS = {"opencode": "OpenCode", "hermes": "Hermes"}
MARKER = ".arc-llama-harness.json"
LICENSE = "MIT"

OPENCODE_REPO = "anomalyco/opencode"
OPENCODE_PIN = "v1.19.4"

HERMES_REPO = "NousResearch/hermes-agent"


@dataclass(frozen=True)
class SourcePin:
    tag: str
    commit: str
    sha256: str
    size: int


HERMES_PIN = SourcePin(
    tag="v0.22.5",
    commit="56b6a31fededdf7364e1a0a79548d144e7fd81a3",
    sha256="a8f67e5ec14fea681acef6cc7ff1553f002bb6101fd2df22bfdd4b7eecc54eb2",
    size=87135743,
)
HERMES_PYTHON = "3.14"
"""Hermes's uv.lock only declares Python 3.14+ environments."""

# uv is only downloaded when it is not already on PATH. These are the
# statically linked builds, so one Linux asset works on glibc and musl.
UV_VERSION = "0.11.8"
UV_ASSETS: dict[tuple[str, str], tuple[str, str, int]] = {
    ("linux", "x64"): (
        "uv-x86_64-unknown-linux-musl.tar.gz",
        "de82507d12e31cfc86c1c776238f7c248e48e40d996dedc812d64fdd31c6ed12",
        24320043,
    ),
    ("linux", "arm64"): (
        "uv-aarch64-unknown-linux-musl.tar.gz",
        "29418befb64f926a2dba3473e8e69acd00b36fb845d85344ef11321a993ad8f5",
        22462932,
    ),
    ("windows", "x64"): (
        "uv-x86_64-pc-windows-msvc.zip",
        "c84629a56e0706b69a47ea35862208af827cb6fbfa1d0ca763c52c67594637e8",
        23530194,
    ),
    ("windows", "arm64"): (
        "uv-aarch64-pc-windows-msvc.zip",
        "bb48716e74e4998993f15bc57a55e4d0d73ccbd27a66d7cbed37605f7c67d747",
        22106633,
    ),
}

HERMES_WINDOWS_EXPERIMENTAL = True
"""Hermes on native Windows has not been validated through Arc Llama yet."""

COMMAND_TIMEOUT = 1800.0
"""Upper bound for one install subprocess (uv sync downloads ~100 packages)."""


class HarnessInstallError(RuntimeError):
    """An install step failed; the message says which and what to do."""


class InstallCancelledError(HarnessInstallError):
    """The user cancelled the install."""


Progress = Callable[[str, int | None, int | None], None]
"""progress(step, bytes_done, bytes_total)"""


@dataclass
class InstallResult:
    name: str
    version: str
    binary: Path
    install_dir: Path
    tested: bool


@dataclass
class _Context:
    progress: Progress
    log_line: Callable[[str], None]
    cancel: threading.Event
    runner: Callable[..., subprocess.Popen]

    def check(self) -> None:
        if self.cancel.is_set():
            raise InstallCancelledError("Install cancelled.")


def harness_root(cfg: Config) -> Path:
    return Path(cfg.paths.state_dir).expanduser() / "harnesses"


def _platform_tag(os_name: str, arch: str) -> str:
    return f"{os_name}-{arch}"


# ---------------------------------------------------------------------------
# platform detection
# ---------------------------------------------------------------------------


def detect_musl() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    libc, _ = platform.libc_ver()
    if libc == "glibc":
        return False
    return any(Path("/lib").glob("ld-musl-*")) or libc == "musl"


def detect_avx2() -> bool:
    """True unless the CPU is known to lack AVX2 (then use OpenCode's baseline build)."""
    if sys.platform.startswith("linux"):
        try:
            flags = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return " avx2" in flags
    if sys.platform == "win32":
        try:
            import ctypes

            return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            return False
    return True


def opencode_asset_name(os_name: str, arch: str, *, musl: bool, avx2: bool) -> str:
    if os_name not in ("linux", "windows") or arch not in ("x64", "arm64"):
        raise HarnessInstallError(f"OpenCode has no Arc Llama install for {os_name}/{arch}.")
    name = f"opencode-{os_name}-{arch}"
    if arch == "x64" and not avx2:
        name += "-baseline"
    if os_name == "linux" and musl:
        name += "-musl"
    return name + (".zip" if os_name == "windows" else ".tar.gz")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _client() -> httpx.Client:
    return httpx.Client(
        follow_redirects=True, timeout=300.0, headers=_auth_headers(), verify=_runtime_ssl_context()
    )


def _release(client: httpx.Client, repo: str, tag: str | None) -> dict[str, Any]:
    url = f"{GITHUB_API}/repos/{repo}/releases/" + (f"tags/{tag}" if tag else "latest")
    response = client.get(url)
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    return data


def _write_marker(install_dir: Path, data: dict[str, Any]) -> None:
    (install_dir / MARKER).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_marker(install_dir: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((install_dir / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _run(ctx: _Context, args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
         step: str) -> None:
    """Run one install command, streaming its output to the job log and honouring cancel."""
    ctx.log_line("$ " + " ".join(args))
    try:
        proc = ctx.runner(
            args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
        )
    except OSError as exc:
        raise HarnessInstallError(f"{step} could not start: {exc}") from exc
    lines: list[str] = []

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line.rstrip())
            ctx.log_line(line.rstrip())

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    deadline = time.monotonic() + COMMAND_TIMEOUT
    while proc.poll() is None:
        if ctx.cancel.is_set() or time.monotonic() > deadline:
            proc.kill()
            proc.wait()
            reader.join(5)
            if ctx.cancel.is_set():
                raise InstallCancelledError("Install cancelled.")
            raise HarnessInstallError(f"{step} took longer than {int(COMMAND_TIMEOUT)} s and was stopped.")
        time.sleep(0.2)
    reader.join(5)
    if proc.returncode != 0:
        tail = next((line for line in reversed(lines) if line.strip()), "no output")
        raise HarnessInstallError(f"{step} failed (exit {proc.returncode}): {tail[:300]}")


def _download(ctx: _Context, client: httpx.Client, asset: RuntimeAsset, dest: Path, step: str) -> Path:
    def on_progress(done: int, total: int) -> None:
        ctx.progress(step, done, total or None)
        ctx.check()

    ctx.progress(step, 0, asset.size or None)
    try:
        return download_asset(client, asset, dest, on_progress=on_progress)
    except RuntimeInstallError as exc:
        raise HarnessInstallError(str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HarnessInstallError(f"Download of {asset.name} failed: {exc}") from exc


def _find_executable(root: Path, name: str) -> Path | None:
    candidates = sorted(
        (p for p in root.rglob(name) if p.is_file() and not p.is_symlink()), key=lambda p: len(p.parts)
    )
    return candidates[0] if candidates else None


def _opencode_exe(root: Path) -> Path | None:
    return _find_executable(root, "opencode.exe" if sys.platform == "win32" else "opencode")


def _hermes_exe(install_dir: Path) -> Path:
    if sys.platform == "win32":
        return install_dir / "venv" / "Scripts" / "hermes.exe"
    return install_dir / "venv" / "bin" / "hermes"


def _check_space(path: Path, need: int) -> None:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < need:
        raise HarnessInstallError(
            f"Not enough disk space in {probe}: need about {need / 2**20:.0f} MiB, {free / 2**20:.0f} MiB free."
        )


def _default_runner(*args: Any, **kwargs: Any) -> subprocess.Popen:
    if sys.platform == "win32":
        kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)  # type: ignore[attr-defined]
    return subprocess.Popen(*args, **kwargs)


# ---------------------------------------------------------------------------
# OpenCode
# ---------------------------------------------------------------------------


def _install_opencode(cfg: Config, version: str, client: httpx.Client, ctx: _Context) -> InstallResult:
    os_name, arch = host_platform()
    name = opencode_asset_name(os_name, arch, musl=detect_musl(), avx2=detect_avx2())
    tag = OPENCODE_PIN if version == "pinned" else (None if version == "latest" else version)
    ctx.progress("Finding release", None, None)
    try:
        release = _release(client, OPENCODE_REPO, tag)
    except httpx.HTTPError as exc:
        raise HarnessInstallError(f"Could not read OpenCode release {tag or 'latest'}: {exc}") from exc
    tag = str(release.get("tag_name") or tag)
    meta = next((a for a in release.get("assets", []) if a.get("name") == name), None)
    if meta is None:
        raise HarnessInstallError(f"OpenCode {tag} has no {name} download.")
    digest = _normalise_sha256_digest(meta.get("digest"))
    if digest is None:
        raise HarnessInstallError(f"OpenCode {tag} publishes no SHA-256 for {name}; refusing to install it.")
    asset = RuntimeAsset(name=name, url=meta["browser_download_url"], size=int(meta.get("size") or 0),
                         tag=tag, digest=digest)
    dest = harness_root(cfg) / "opencode"
    dest.mkdir(parents=True, exist_ok=True)
    _recover_interrupted_publish(dest, _opencode_exe)
    _check_space(dest, 3 * asset.size + 64 * 2**20)
    install_dir = dest / f"opencode-{tag}-{_platform_tag(os_name, arch)}"
    work = Path(tempfile.mkdtemp(prefix=".download-", dir=dest))
    staging = Path(tempfile.mkdtemp(prefix=f".{install_dir.name}.staging-", dir=dest))
    try:
        archive = _download(ctx, client, asset, work / name, "Downloading OpenCode")
        ctx.check()
        ctx.progress("Extracting", None, None)
        try:
            extract_archive_files(archive, staging)
        except RuntimeInstallError as exc:
            raise HarnessInstallError(str(exc)) from exc
        exe = _opencode_exe(staging)
        if exe is None:
            raise HarnessInstallError(f"{name} contains no opencode executable.")
        if os.name == "posix":
            os.chmod(exe, 0o755)
        ctx.progress("Checking OpenCode", None, None)
        _run(ctx, [str(exe), "--version"], env={**os.environ, "OPENCODE_DISABLE_AUTOUPDATE": "1"},
             step="opencode --version")
        relative = exe.relative_to(staging)
        tested = tag == OPENCODE_PIN
        _write_marker(staging, {
            "schema": 1, "name": "opencode", "version": tag, "source": asset.url, "sha256": digest,
            "license": LICENSE, "tested": tested, "executable": relative.as_posix(),
            "installed_at": time.time(),
        })
        ctx.check()
        _publish_install(staging, install_dir)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
    return InstallResult("opencode", tag, install_dir / relative, install_dir, tested)


# ---------------------------------------------------------------------------
# Hermes
# ---------------------------------------------------------------------------


def _uv(cfg: Config, client: httpx.Client, ctx: _Context) -> Path:
    found = shutil.which("uv")
    if found:
        return Path(found)
    os_name, arch = host_platform()
    entry = UV_ASSETS.get((os_name, arch))
    if entry is None:
        raise HarnessInstallError(f"No uv download is pinned for {os_name}/{arch}; install uv and retry.")
    name, digest, size = entry
    tools = harness_root(cfg) / "tools"
    exe_name = "uv.exe" if os_name == "windows" else "uv"
    target = tools / f"uv-{UV_VERSION}"
    existing = _find_executable(target, exe_name) if target.is_dir() else None
    if existing is not None:
        return existing
    tools.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".uv-", dir=tools))
    try:
        url = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{name}"
        archive = _download(ctx, client, RuntimeAsset(name, url, size, UV_VERSION, digest), work / name,
                            "Downloading uv")
        staging = work / "out"
        try:
            extract_archive_files(archive, staging)
        except RuntimeInstallError as exc:
            raise HarnessInstallError(str(exc)) from exc
        if _find_executable(staging, exe_name) is None:
            raise HarnessInstallError(f"{name} contains no {exe_name}.")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(staging, target)
        except OSError:
            if not target.is_dir():
                raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    exe = _find_executable(target, exe_name)
    if exe is None:
        raise HarnessInstallError("uv was downloaded but its executable is missing.")
    if os.name == "posix":
        os.chmod(exe, 0o755)
    return exe


def _hermes_source(client: httpx.Client, version: str) -> tuple[str, str, str | None, int]:
    """(tag, tarball URL, pinned sha256 or None, size or 0)."""
    if version == "pinned":
        pin = HERMES_PIN
        return pin.tag, f"https://github.com/{HERMES_REPO}/archive/refs/tags/{pin.tag}.tar.gz", pin.sha256, pin.size
    try:
        release = _release(client, HERMES_REPO, None if version == "latest" else version)
    except httpx.HTTPError as exc:
        raise HarnessInstallError(f"Could not read Hermes release {version}: {exc}") from exc
    tag = str(release["tag_name"])
    return tag, f"https://github.com/{HERMES_REPO}/archive/refs/tags/{tag}.tar.gz", None, 0


def _install_hermes(cfg: Config, version: str, client: httpx.Client, ctx: _Context) -> InstallResult:
    os_name, arch = host_platform()
    if os_name not in ("linux", "windows"):
        raise HarnessInstallError(f"Hermes has no Arc Llama install for {os_name}.")
    tag, url, sha256, size = _hermes_source(client, version)
    dest = harness_root(cfg) / "hermes"
    dest.mkdir(parents=True, exist_ok=True)
    _check_space(dest, 1024 * 2**20)
    install_dir = dest / f"hermes-{tag}-{_platform_tag(os_name, arch)}"
    if read_marker(install_dir) is not None and _hermes_exe(install_dir).is_file():
        tested = tag == HERMES_PIN.tag
        return InstallResult("hermes", tag, _hermes_exe(install_dir), install_dir, tested)
    uv = _uv(cfg, client, ctx)
    ctx.check()
    tools = harness_root(cfg) / "tools"
    work = Path(tempfile.mkdtemp(prefix=".hermes-src-", dir=dest))
    try:
        name = f"hermes-agent-{tag}.tar.gz"
        archive = _download(ctx, client, RuntimeAsset(name, url, size, tag, sha256), work / name,
                            "Downloading Hermes source")
        ctx.check()
        # An unmarked folder is a failed or interrupted earlier attempt.
        if install_dir.exists():
            shutil.rmtree(install_dir)
        install_dir.mkdir(parents=True)
        ctx.progress("Extracting", None, None)
        try:
            extract_archive_files(archive, install_dir / ".src-extract")
        except RuntimeInstallError as exc:
            raise HarnessInstallError(str(exc)) from exc
        projects = [p.parent for p in (install_dir / ".src-extract").glob("*/pyproject.toml")]
        if len(projects) != 1 or not (projects[0] / "uv.lock").is_file():
            raise HarnessInstallError("The Hermes source has no single project with a uv.lock.")
        source = install_dir / "src"
        os.replace(projects[0], source)
        shutil.rmtree(install_dir / ".src-extract", ignore_errors=True)
        env = {
            **os.environ,
            "UV_PROJECT_ENVIRONMENT": str(install_dir / "venv"),
            "UV_CACHE_DIR": str(work / "uv-cache"),
            "UV_PYTHON_INSTALL_DIR": str(tools / "python"),
            "UV_PYTHON_PREFERENCE": "only-managed",
            "UV_NO_CONFIG": "1",
        }
        for key in ("VIRTUAL_ENV", "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "PIP_INDEX_URL"):
            env.pop(key, None)
        ctx.progress("Installing Hermes and its dependencies", None, None)
        _run(ctx, [str(uv), "sync", "--frozen", "--no-dev", "--extra", "acp", "--python", HERMES_PYTHON],
             cwd=source, env=env, step="Installing Hermes")
        exe = _hermes_exe(install_dir)
        if not exe.is_file():
            raise HarnessInstallError("Hermes installed without its hermes command.")
        ctx.progress("Checking Hermes ACP", None, None)
        home = work / "hermes-home"
        home.mkdir()
        _run(ctx, [str(exe), "acp", "--check"], env={**os.environ, "HERMES_HOME": str(home)},
             step="hermes acp --check")
        tested = tag == HERMES_PIN.tag
        _write_marker(install_dir, {
            "schema": 1, "name": "hermes", "version": tag, "source": url, "sha256": sha256,
            "license": LICENSE, "tested": tested, "executable": exe.relative_to(install_dir).as_posix(),
            "installed_at": time.time(),
        })
    except BaseException:
        if read_marker(install_dir) is None:
            shutil.rmtree(install_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return InstallResult("hermes", tag, exe, install_dir, tested)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def install_harness(
    cfg: Config,
    name: str,
    *,
    version: str = "pinned",
    config_path: Path | None = None,
    set_default: bool = True,
    client: httpx.Client | None = None,
    progress: Progress | None = None,
    log_line: Callable[[str], None] | None = None,
    cancel: threading.Event | None = None,
    runner: Callable[..., subprocess.Popen] = _default_runner,
) -> InstallResult:
    """Install *name* (``opencode`` or ``hermes``) and select it for Work."""
    if name not in HARNESSES:
        raise HarnessInstallError(f"Unknown harness {name!r}. Choose opencode or hermes.")
    ctx = _Context(
        progress=progress or (lambda *_: None),
        log_line=log_line or (lambda _line: None),
        cancel=cancel or threading.Event(),
        runner=runner,
    )
    own = client is None
    client = client or _client()
    try:
        installer = _install_opencode if name == "opencode" else _install_hermes
        result = installer(cfg, version, client, ctx)
    finally:
        if own:
            client.close()
    if set_default:
        setattr(cfg.agent, f"{name}_bin", str(result.binary))
        cfg.save(config_path)
    ctx.progress("Done", None, None)
    return result


def managed_installs(cfg: Config, name: str) -> list[dict[str, Any]]:
    """Complete managed installs of *name*, newest first."""
    root = harness_root(cfg) / name
    found: list[dict[str, Any]] = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return []
    for entry in entries:
        if entry.name.startswith(".") or not entry.is_dir() or entry.is_symlink():
            continue
        marker = read_marker(entry)
        if not marker or marker.get("name") != name:
            continue
        exe = entry / str(marker.get("executable", ""))
        if not exe.is_file():
            continue
        found.append({**marker, "path": str(exe), "dir": str(entry)})
    found.sort(key=lambda m: float(m.get("installed_at") or 0), reverse=True)
    return found


def is_managed(cfg: Config, binary: str | None) -> bool:
    if not binary:
        return False
    try:
        Path(binary).resolve().relative_to(harness_root(cfg).resolve())
    except (OSError, ValueError):
        return False
    return True


def remove_harness(cfg: Config, name: str, version: str, *, config_path: Path | None = None) -> bool:
    """Delete one managed install; clear the selection if it pointed there."""
    for item in managed_installs(cfg, name):
        if item.get("version") != version:
            continue
        install_dir = Path(item["dir"])
        current = getattr(cfg.agent, f"{name}_bin", None)
        if current and Path(current).resolve().is_relative_to(install_dir.resolve()):
            setattr(cfg.agent, f"{name}_bin", None)
            cfg.save(config_path)
        trash = install_dir.with_name(f".{install_dir.name}.trash-{os.getpid()}")
        os.replace(install_dir, trash)
        shutil.rmtree(trash, ignore_errors=True)
        if name == "hermes" and not managed_installs(cfg, "hermes"):
            # The Python and uv downloaded for Hermes serve nothing else.
            shutil.rmtree(harness_root(cfg) / "tools", ignore_errors=True)
        return True
    return False


def offer(cfg: Config, name: str) -> dict[str, Any]:
    """What the UI shows before installing *name*."""
    os_name, arch = host_platform()
    supported = os_name in ("linux", "windows") and (name == "hermes" or arch in ("x64", "arm64"))
    if name == "opencode":
        version, repo, size_note = OPENCODE_PIN, OPENCODE_REPO, "about 80 MB download, ~200 MB installed"
    else:
        version, repo, size_note = HERMES_PIN.tag, HERMES_REPO, "about 90 MB download, ~600 MB installed"
    return {
        "installable": supported,
        "version": version,
        "license": LICENSE,
        "source": f"https://github.com/{repo}",
        "size_note": size_note,
        "location": str(harness_root(cfg) / name),
        "experimental": name == "hermes" and os_name == "windows" and HERMES_WINDOWS_EXPERIMENTAL,
    }


# ---------------------------------------------------------------------------
# background jobs for the web UI
# ---------------------------------------------------------------------------


@dataclass
class InstallJob:
    id: str
    harness: str
    version: str
    status: str = "queued"
    """queued, running, done, error or cancelled."""
    step: str = ""
    bytes_done: int | None = None
    bytes_total: int | None = None
    error: str | None = None
    installed_version: str | None = None
    tested: bool | None = None
    log: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["log"] = self.log[-40:]
        return data


class HarnessInstallManager:
    """Runs one harness install at a time in a worker thread."""

    def __init__(self, cfg: Config, config_path: Path | None = None,
                 installer: Callable[..., InstallResult] = install_harness) -> None:
        self.cfg = cfg
        self.config_path = config_path
        self.installer = installer
        self.jobs: dict[str, InstallJob] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def active(self) -> InstallJob | None:
        return next((j for j in self.jobs.values() if j.status in ("queued", "running")), None)

    def submit(self, name: str, version: str = "pinned") -> InstallJob:
        if name not in HARNESSES:
            raise ValueError(f"Unknown harness {name!r}.")
        if version not in ("pinned", "latest"):
            raise ValueError("version must be 'pinned' or 'latest'.")
        with self._lock:
            if self.active() is not None:
                raise ValueError("Another agent install is already running.")
            import uuid

            job = InstallJob(id=uuid.uuid4().hex[:12], harness=name, version=version)
            self.jobs[job.id] = job
            self._cancel[job.id] = threading.Event()
        threading.Thread(target=self._run, args=(job,), name=f"harness-install-{name}", daemon=True).start()
        return job

    def cancel(self, job_id: str) -> bool:
        event = self._cancel.get(job_id)
        job = self.jobs.get(job_id)
        if event is None or job is None or job.status not in ("queued", "running"):
            return False
        event.set()
        return True

    def _run(self, job: InstallJob) -> None:
        job.status = "running"

        def progress(step: str, done: int | None, total: int | None) -> None:
            job.step, job.bytes_done, job.bytes_total = step, done, total

        def log_line(line: str) -> None:
            job.log.append(line[:500])
            del job.log[:-200]

        try:
            result = self.installer(
                self.cfg, job.harness, version=job.version, config_path=self.config_path,
                progress=progress, log_line=log_line, cancel=self._cancel[job.id],
            )
        except InstallCancelledError:
            job.status, job.error = "cancelled", "Install cancelled."
        except Exception as exc:  # noqa: BLE001 - every failure is reported on the job
            job.status, job.error = "error", str(exc)[:500]
            log.warning("harness install %s failed: %s", job.harness, exc)
        else:
            job.status, job.installed_version, job.tested = "done", result.version, result.tested
        finally:
            job.finished_at = time.time()
