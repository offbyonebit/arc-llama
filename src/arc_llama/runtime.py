"""Fetch a portable prebuilt llama-server binary from official ggml-org/llama.cpp
GitHub releases, so a fresh Intel Arc user can skip installing oneAPI or building
llama.cpp from source. Vulkan is the default backend because the Vulkan build is
fully portable on Arc (no oneAPI runtime needed). SYCL is offered for max speed.
"""
from __future__ import annotations

import logging
import os
import platform
import tarfile
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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
        raise RuntimeInstallError(
            f"Unsupported backend '{backend}'. Choose 'vulkan' or 'sycl'."
        )
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


def select_asset(
    release_json: dict, os_name: str, arch: str, backend: str
) -> RuntimeAsset:
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


def _resolve_windows_release_with_asset(
    client: httpx.Client, arch: str, backend: str
) -> dict:
    """Find the newest Windows release that actually ships a runtime asset.

    GitHub's ``/releases/latest`` currently points at a lightweight v0.4.0
    release with no binary assets.  Windows prebuilt binaries continue to be
    published on the rolling ``bNNNNN`` releases, so ``latest`` needs this
    Windows-only fallback.  Linux resolution is intentionally unchanged.
    """
    url = f"{GITHUB_API}/repos/{LLAMA_CPP_REPO}/releases?per_page=30"
    response = client.get(url)
    response.raise_for_status()
    for release in response.json():
        try:
            select_asset(release, "windows", arch, backend)
        except RuntimeInstallError:
            continue
        return release
    raise RuntimeInstallError(
        f"No Windows {backend} runtime asset found in the recent llama.cpp releases."
    )


def download_asset(
    client: httpx.Client,
    asset: RuntimeAsset,
    dest_file: Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Stream *asset* to *dest_file*, optionally reporting progress."""
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    bytes_so_far = 0
    with client.stream("GET", asset.url) as r:
        r.raise_for_status()
        with open(dest_file, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
                bytes_so_far += len(chunk)
                if on_progress is not None:
                    on_progress(bytes_so_far, asset.size)
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


def extract_archive(archive: Path, dest_dir: Path) -> Path:
    """Extract *archive* into *dest_dir* and return the path to the llama-server binary."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive, "r:gz") as tf:
            try:
                tf.extractall(dest_dir, filter="data")
            except TypeError:
                tf.extractall(dest_dir)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
    else:
        raise RuntimeInstallError(f"Unknown archive type: {archive.name}")
    found = _find_llama_server(dest_dir)
    if found is None:
        raise RuntimeInstallError(f"no llama-server binary found in {archive.name}")
    if os.name == "posix":
        os.chmod(found, 0o755)
    return found


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
        raise RuntimeInstallError(
            f"Unsupported backend '{backend}'. Choose 'vulkan' or 'sycl'."
        )

    os_name, arch = host_platform()

    if dest is None:
        if cfg is not None:
            base = Path(cfg.paths.state_dir).expanduser()
        else:
            base = default_state_dir()
        dest = base / "runtime"

    own_client = client is None
    if own_client:
        client = httpx.Client(
            follow_redirects=True, timeout=300.0, headers=_auth_headers()
        )
    assert client is not None
    try:
        release = resolve_release(client, version)
        try:
            asset = select_asset(release, os_name, arch, backend)
        except RuntimeInstallError:
            if os_name == "windows" and version in ("latest", "", None):
                release = _resolve_windows_release_with_asset(client, arch, backend)
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
        try:
            download_asset(client, asset, tmp_path, on_progress=on_progress)
            binary = extract_archive(tmp_path, install_dir)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        detected = detect_llama_server_backend(binary)
        if detected is None:
            log.warning("Could not detect backend of %s", binary)
        elif detected.value != backend:
            log.warning(
                "Detected backend '%s' does not match requested '%s' for %s",
                detected.value,
                backend,
                binary,
            )

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
