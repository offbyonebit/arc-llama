"""Fetch a portable prebuilt llama-server binary from official ggml-org/llama.cpp
GitHub releases, so a fresh Intel Arc user can skip installing oneAPI or building
llama.cpp from source. Vulkan is the default backend because the Vulkan build is
fully portable on Arc (no oneAPI runtime needed). SYCL is offered for max speed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import platform
import re
import shutil
import tarfile
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from arc_llama.arch import Backend
from arc_llama.binary import detect_llama_server_backend
from arc_llama.config import Config, default_state_dir, load_config

log = logging.getLogger("arc_llama.runtime")

LLAMA_CPP_REPO = "ggml-org/llama.cpp"
GITHUB_API = "https://api.github.com"


class RuntimeInstallError(RuntimeError):
    """Raised for any unrecoverable problem installing a runtime."""


@dataclass
class RuntimeAsset:
    name: str
    url: str
    size: int
    tag: str
    digest: str | None = None


@dataclass
class RuntimeInstallResult:
    binary_path: Path
    backend: Backend | None
    requested_backend: str
    tag: str
    install_dir: Path
    set_as_default: bool


def host_platform() -> tuple[str, str]:
    """Return (os_name, arch) lowercased and normalised."""
    system = platform.system()
    machine = platform.machine()
    os_map = {"Linux": "linux", "Windows": "windows", "Darwin": "macos"}
    os_name = os_map.get(system, system.lower())
    if machine in ("x86_64", "AMD64"):
        arch = "x64"
    elif machine in ("aarch64", "arm64"):
        arch = "arm64"
    else:
        arch = machine.lower()
    return (os_name, arch)


def asset_suffix(os_name: str, arch: str, backend: str) -> str:
    """Return the exact release-asset name suffix for the given combo."""
    if backend not in ("vulkan", "sycl"):
        raise RuntimeInstallError(f"Unsupported backend '{backend}'. Choose 'vulkan' or 'sycl'.")
    table = {
        ("linux", "x64", "vulkan"): "bin-ubuntu-vulkan-x64.tar.gz",
        ("linux", "x64", "sycl"): "bin-ubuntu-sycl-fp16-x64.tar.gz",
        ("windows", "x64", "vulkan"): "bin-win-vulkan-x64.zip",
        ("windows", "x64", "sycl"): "bin-win-sycl-x64.zip",
    }
    key = (os_name, arch, backend)
    if key not in table:
        raise RuntimeInstallError(
            f"No prebuilt llama-server for {os_name}/{arch}/{backend}. "
            "Only linux/x64 and windows/x64 are supported."
        )
    return table[key]


def select_asset(release_json: dict, os_name: str, arch: str, backend: str) -> RuntimeAsset:
    """Find the matching asset in a GitHub release JSON payload."""
    tag = release_json["tag_name"]
    suffix = asset_suffix(os_name, arch, backend)
    for asset in release_json.get("assets", []):
        name = asset.get("name", "")
        if name.endswith(suffix):
            return RuntimeAsset(
                name=name,
                url=asset["browser_download_url"],
                size=asset.get("size", 0),
                tag=tag,
                digest=_normalise_sha256_digest(asset.get("digest")),
            )
    raise RuntimeInstallError(
        f"No asset matching '{suffix}' in release {tag}. "
        "The release may not ship a binary for this platform/backend."
    )


def resolve_release(client: httpx.Client, version: str) -> dict:
    """Fetch the release JSON for *version* ('latest' or a tag like 'b10092')."""
    if version in ("latest", "", None):
        url = f"{GITHUB_API}/repos/{LLAMA_CPP_REPO}/releases/latest"
    else:
        url = f"{GITHUB_API}/repos/{LLAMA_CPP_REPO}/releases/tags/{version}"
    r = client.get(url)
    r.raise_for_status()
    return r.json()


def _normalise_sha256_digest(value: object) -> str | None:
    """Return a lowercase SHA-256 hex digest from GitHub's asset field.

    GitHub returns release-asset digests as ``sha256:<hex>``. Older releases
    may have no digest, so absence remains supported, but malformed values are
    ignored rather than accidentally treated as trusted checksums.
    """
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"sha256:([0-9a-fA-F]{64})", value.strip())
    return match.group(1).lower() if match else None


def _resolve_release_with_asset(
    client: httpx.Client, os_name: str, arch: str, backend: str
) -> dict:
    """Find the newest rolling release that actually ships a runtime asset.

    GitHub's ``/releases/latest`` may point at a lightweight release with no
    binary assets. Prebuilt binaries continue to be published on the rolling
    ``bNNNNN`` releases, so ``latest`` needs to search those releases on every
    supported platform.
    """
    url = f"{GITHUB_API}/repos/{LLAMA_CPP_REPO}/releases?per_page=30"
    response = client.get(url)
    response.raise_for_status()
    for release in response.json():
        try:
            select_asset(release, os_name, arch, backend)
        except RuntimeInstallError:
            continue
        return release
    raise RuntimeInstallError(
        f"No {os_name}/{arch} {backend} runtime asset found in recent llama.cpp releases."
    )


def _resolve_windows_release_with_asset(client: httpx.Client, arch: str, backend: str) -> dict:
    """Backward-compatible wrapper for the original Windows-only helper."""
    return _resolve_release_with_asset(client, "windows", arch, backend)


def download_asset(
    client: httpx.Client,
    asset: RuntimeAsset,
    dest_file: Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Stream *asset* to *dest_file* and verify size/digest when available."""
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    bytes_so_far = 0
    sha256 = hashlib.sha256()
    try:
        with client.stream("GET", asset.url) as r:
            r.raise_for_status()
            with open(dest_file, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
                    sha256.update(chunk)
                    bytes_so_far += len(chunk)
                    if on_progress is not None:
                        on_progress(bytes_so_far, asset.size)
        if asset.size > 0 and bytes_so_far != asset.size:
            raise RuntimeInstallError(
                f"Downloaded size mismatch for {asset.name}: expected {asset.size} bytes, "
                f"got {bytes_so_far}."
            )
        actual_digest = sha256.hexdigest()
        if asset.digest is not None and actual_digest != asset.digest:
            raise RuntimeInstallError(
                f"SHA-256 mismatch for {asset.name}: expected {asset.digest}, got {actual_digest}."
            )
    except BaseException:
        # Never leave a corrupt file that a caller could mistake for a complete
        # download. install_runtime also cleans its temporary file, but this
        # function is public and is tested/usable independently.
        dest_file.unlink(missing_ok=True)
        raise
    return dest_file


def _find_llama_server(root: Path) -> Path | None:
    """Recursively search *root* for llama-server / llama-server.exe (shallowest first)."""
    candidates: list[Path] = []
    for pattern in ("llama-server", "llama-server.exe"):
        candidates.extend(root.rglob(pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]


def _validated_archive_path(dest_dir: Path, member_name: str) -> Path:
    """Resolve an archive member beneath *dest_dir* or reject it.

    Both separators are normalised because ZIP members created on Windows can
    contain backslashes. Drive-qualified paths, absolute paths, and ``..`` are
    rejected before extraction on every supported Python version.
    """
    normalised = member_name.replace("\\", "/")
    if not normalised or normalised.startswith("/") or re.match(r"^[A-Za-z]:", normalised):
        raise RuntimeInstallError(f"Unsafe archive member path: {member_name!r}")
    parts = [part for part in normalised.split("/") if part not in ("", ".")]
    if not parts or ".." in parts:
        raise RuntimeInstallError(f"Unsafe archive member path: {member_name!r}")
    root = dest_dir.resolve()
    target = (root / Path(*parts)).resolve()
    try:
        common = Path(os.path.commonpath([root, target]))
    except ValueError as exc:
        raise RuntimeInstallError(f"Unsafe archive member path: {member_name!r}") from exc
    if common != root:
        raise RuntimeInstallError(f"Unsafe archive member path: {member_name!r}")
    return target


def _validate_tar_members(tf: tarfile.TarFile, dest_dir: Path) -> list[tarfile.TarInfo]:
    members = tf.getmembers()
    for member in members:
        _validated_archive_path(dest_dir, member.name)
        if member.ischr() or member.isblk() or member.isfifo():
            raise RuntimeInstallError(f"Unsafe special file in archive: {member.name!r}")
        if member.issym():
            link_name = str(Path(member.name).parent / member.linkname)
            _validated_archive_path(dest_dir, link_name)
        elif member.islnk():
            # Tar hard-link targets are archive-root-relative.
            _validated_archive_path(dest_dir, member.linkname)
    return members


def extract_archive(archive: Path, dest_dir: Path) -> Path:
    """Safely extract *archive* and return its llama-server binary."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive, "r:gz") as tf:
            members = _validate_tar_members(tf, dest_dir)
            # Python 3.12+'s data filter adds another defence layer. Our own
            # validation above provides the equivalent path/link protection on
            # supported Python 3.10 and 3.11 installations.
            supports_filter = "filter" in inspect.signature(tf.extractall).parameters
            if supports_filter:
                # Security-maintained 3.10/3.11 releases backported the filter
                # argument. `fully_trusted` remains usable if a downstream
                # interpreter exposes the API without `data_filter`; our strict
                # member validation above is still authoritative in that case.
                filter_name: Literal["data", "fully_trusted"] = (
                    "data" if hasattr(tarfile, "data_filter") else "fully_trusted"
                )
                tf.extractall(dest_dir, members=members, filter=filter_name)
            else:  # pragma: no cover - old Python 3.10/3.11 maintenance releases
                tf.extractall(dest_dir, members=members)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                _validated_archive_path(dest_dir, member.filename)
            zf.extractall(dest_dir)
    else:
        raise RuntimeInstallError(f"Unknown archive type: {archive.name}")
    found = _find_llama_server(dest_dir)
    if found is None:
        raise RuntimeInstallError(f"no llama-server binary found in {archive.name}")
    if os.name == "posix":
        os.chmod(found, 0o755)
    return found


def _publish_install(staging_dir: Path, install_dir: Path) -> None:
    """Atomically publish a staged runtime, restoring the old one on failure."""
    backup_dir: Path | None = None
    if install_dir.exists():
        backup_dir = Path(
            tempfile.mkdtemp(prefix=f".{install_dir.name}.backup-", dir=install_dir.parent)
        )
        backup_dir.rmdir()
        os.replace(install_dir, backup_dir)
    try:
        os.replace(staging_dir, install_dir)
    except BaseException:
        if backup_dir is not None and backup_dir.exists() and not install_dir.exists():
            os.replace(backup_dir, install_dir)
        raise
    if backup_dir is not None:
        shutil.rmtree(backup_dir, ignore_errors=True)


def _auth_headers() -> dict[str, str]:
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        return {"Authorization": f"Bearer {tok}"}
    return {}


def install_runtime(
    *,
    backend: str = "vulkan",
    version: str = "latest",
    dest: Path | None = None,
    cfg: Config | None = None,
    set_default: bool = True,
    config_path: Path | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    client: httpx.Client | None = None,
    force: bool = False,
) -> RuntimeInstallResult:
    """Download, extract, and verify a prebuilt llama-server binary.

    Parameters
    ----------
    backend:
        'vulkan' (portable, no oneAPI) or 'sycl' (faster, needs oneAPI on Linux).
    version:
        llama.cpp release tag, e.g. 'b10092', or 'latest'.
    dest:
        Root install directory. Defaults to <state_dir>/runtime.
    cfg:
        Existing Config to read/update. Loaded from *config_path* if None.
    set_default:
        Write the binary path into cfg.paths.llama_server and save.
    config_path:
        Path to the TOML config file (passed to cfg.save).
    on_progress:
        Optional callback(bytes_so_far, total) called during download.
    client:
        Pre-built httpx.Client. A transient one is created if None.
    force:
        Re-download even if this version is already installed.
    """
    if backend not in ("vulkan", "sycl"):
        raise RuntimeInstallError(f"Unsupported backend '{backend}'. Choose 'vulkan' or 'sycl'.")

    os_name, arch = host_platform()

    if dest is None:
        if cfg is not None:
            base = Path(cfg.paths.state_dir).expanduser()
        else:
            base = default_state_dir()
        dest = base / "runtime"

    own_client = client is None
    if own_client:
        client = httpx.Client(follow_redirects=True, timeout=300.0, headers=_auth_headers())
    assert client is not None
    try:
        release = resolve_release(client, version)
        try:
            asset = select_asset(release, os_name, arch, backend)
        except RuntimeInstallError:
            if version in ("latest", "", None):
                release = _resolve_release_with_asset(client, os_name, arch, backend)
                asset = select_asset(release, os_name, arch, backend)
            else:
                raise
        install_dir = dest / f"llama-{asset.tag}-{backend}"

        # Short-circuit: reuse existing install when not forced.
        if not force and install_dir.exists():
            existing = _find_llama_server(install_dir)
            if existing is not None:
                log.info("Reusing existing runtime at %s", existing)
                detected = detect_llama_server_backend(existing)
                if set_default:
                    if cfg is None:
                        cfg = load_config(config_path)
                    cfg.paths.llama_server = str(existing)
                    for gpu_cfg in cfg.gpus:
                        gpu_cfg.backend = backend
                    cfg.save(config_path)
                return RuntimeInstallResult(
                    binary_path=existing,
                    backend=detected,
                    requested_backend=backend,
                    tag=asset.tag,
                    install_dir=install_dir,
                    set_as_default=set_default,
                )

        dest.mkdir(parents=True, exist_ok=True)

        # Give the temp file the right extension so extract_archive can detect type.
        if asset.name.endswith(".tar.gz"):
            tmp_suffix = ".tar.gz"
        elif asset.name.endswith(".zip"):
            tmp_suffix = ".zip"
        else:
            tmp_suffix = ".download"
        fd, tmp_name = tempfile.mkstemp(dir=dest, suffix=tmp_suffix)
        os.close(fd)
        tmp_path = Path(tmp_name)
        staging_dir = Path(tempfile.mkdtemp(prefix=f".{install_dir.name}.staging-", dir=dest))
        try:
            download_asset(client, asset, tmp_path, on_progress=on_progress)
            staged_binary = extract_archive(tmp_path, staging_dir)
            binary_relative = staged_binary.relative_to(staging_dir)
            detected = detect_llama_server_backend(staged_binary)
            if detected is not None and detected.value != backend:
                raise RuntimeInstallError(
                    f"Downloaded runtime backend '{detected.value}' does not match "
                    f"requested backend '{backend}' for {asset.name}."
                )

            # Write the marker before publication so callers only ever see the
            # previous complete install or this complete install.
            marker = {
                "schema": 1,
                "asset": asset.name,
                "tag": asset.tag,
                "backend": backend,
                "sha256": asset.digest,
                "size": asset.size,
            }
            (staging_dir / ".arc-llama-runtime.json").write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _publish_install(staging_dir, install_dir)
            binary = install_dir / binary_relative
        finally:
            tmp_path.unlink(missing_ok=True)
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

        if detected is None:
            log.warning("Could not detect backend of %s", binary)

        if set_default:
            if cfg is None:
                cfg = load_config(config_path)
            cfg.paths.llama_server = str(binary)
            for gpu_cfg in cfg.gpus:
                gpu_cfg.backend = backend
            cfg.save(config_path)

        return RuntimeInstallResult(
            binary_path=binary,
            backend=detected,
            requested_backend=backend,
            tag=asset.tag,
            install_dir=install_dir,
            set_as_default=set_default,
        )
    finally:
        if own_client and client is not None:
            client.close()
