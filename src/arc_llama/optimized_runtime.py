"""Arc Llama Runtime: the optional optimized llama.cpp build for Intel Arc B-series GPUs.

It is distributed as signed, closed binaries (MIT licensed) from the arc-llama-runtime repository. Installing it
verifies an Ed25519 signature over SHA256SUMS with the public key shipped in this package (``ssh-keygen -Y verify``),
then the archive's checksum, before anything is extracted. It is offered only on Linux x86-64 with a Battlemage GPU
and the Intel oneAPI runtime; everything else keeps stock llama.cpp.

A runtime install carries ``arc-llama-runtime.json`` next to ``llama-server``. Its ``model_defaults`` give per-model
environment settings tuned for that build (matched on GGUF architecture, optionally MoE only). They apply only to
models launched with that runtime, and a model's own ``env`` always wins.
"""

from __future__ import annotations

import functools
import json
import logging
import platform
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from arc_llama.config import Config, default_state_dir, load_config
from arc_llama.runtime import (
    GITHUB_API,
    RuntimeAsset,
    RuntimeInstallError,
    RuntimeInstallResult,
    _auth_headers,
    _find_llama_server,
    _publish_install,
    _recover_interrupted_publish,
    _runtime_ssl_context,
    download_asset,
    extract_archive,
)

log = logging.getLogger("arc_llama.optimized_runtime")

RUNTIME_REPO = "offbyonebit/arc-llama-runtime"
SIGNER_ID = "arc-llama-runtime"
SIGNING_NAMESPACE = "arc-llama-runtime"
MANIFEST_NAME = "arc-llama-runtime.json"
MARKER_NAME = ".arc-llama-runtime.json"
ALLOWED_SIGNERS = Path(__file__).parent / "data" / "arc-llama-runtime.allowed_signers"
SUMS_NAME = "SHA256SUMS"
SIG_NAME = "SHA256SUMS.sig"
ASSET_RE = re.compile(r"^arc-llama-runtime-(?P<version>[0-9A-Za-z.\-]+)\+(?P<tag>[0-9A-Za-z.]+)-linux-x64-sycl-bmg\.tar\.gz$")


class OptimizedRuntimeError(RuntimeInstallError):
    """The optimized runtime cannot be installed or failed verification."""


def support_problems(cfg: Config) -> list[str]:
    """Reasons this host cannot use the optimized runtime; empty when it can."""
    from arc_llama.platform_checks import oneapi_setvars_path

    problems = []
    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        problems.append("it is built for Linux x86-64 only")
    gpus = [g for g in cfg.gpus if g.enabled] or cfg.gpus
    if not any((g.arch or "").lower() == "battlemage" for g in gpus):
        problems.append("it needs an Intel Arc B-series (Battlemage) GPU")
    setvars = getattr(cfg.paths, "oneapi_setvars", None)
    if not ((setvars and Path(setvars).expanduser().exists()) or oneapi_setvars_path()):
        problems.append("it needs the Intel oneAPI runtime (setvars.sh not found)")
    if shutil.which("ssh-keygen") is None:
        problems.append("signature verification needs ssh-keygen (OpenSSH)")
    return problems


def verify_signature(sums: Path, sig: Path, allowed_signers: Path = ALLOWED_SIGNERS) -> None:
    """Raise unless *sig* is a valid signature of *sums* by the release key."""
    if not sig.is_file():
        raise OptimizedRuntimeError(f"Missing signature {sig.name}; refusing to install an unsigned runtime.")
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        raise OptimizedRuntimeError("ssh-keygen (OpenSSH) is required to verify the runtime signature.")
    with open(sums, "rb") as data:
        result = subprocess.run(
            [ssh_keygen, "-Y", "verify", "-f", str(allowed_signers), "-I", SIGNER_ID,
             "-n", SIGNING_NAMESPACE, "-s", str(sig)],
            stdin=data, capture_output=True, text=True, timeout=30,
        )
    if result.returncode != 0:
        raise OptimizedRuntimeError("Runtime signature verification failed; the download was not installed.")


def parse_sums(text: str) -> dict[str, str]:
    """``sha256sum`` output -> {file name: lowercase hex digest}."""
    sums = {}
    for line in text.splitlines():
        m = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line.strip())
        if m:
            sums[m.group(2)] = m.group(1).lower()
    return sums


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _release(client: httpx.Client, version: str) -> dict:
    if version in ("", "latest"):
        url = f"{GITHUB_API}/repos/{RUNTIME_REPO}/releases/latest"
    else:
        url = f"{GITHUB_API}/repos/{RUNTIME_REPO}/releases/tags/{version}"
    r = client.get(url, headers={"Accept": "application/vnd.github+json"})
    if r.status_code == 404:
        raise OptimizedRuntimeError(f"No Arc Llama Runtime release '{version}' found.")
    r.raise_for_status()
    return r.json()


def _release_asset(release: dict, name: str) -> RuntimeAsset:
    for asset in release.get("assets", []):
        if asset.get("name") == name:
            return RuntimeAsset(
                name=name, url=asset["browser_download_url"], size=asset.get("size", 0),
                tag=release.get("tag_name", ""), digest=None,
            )
    raise OptimizedRuntimeError(f"Release {release.get('tag_name', '?')} has no {name}.")


def install_optimized_runtime(
    *,
    cfg: Config | None = None,
    config_path: Path | None = None,
    version: str = "latest",
    archive: Path | None = None,
    dest: Path | None = None,
    set_default: bool = True,
    force: bool = False,
    client: httpx.Client | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    allowed_signers: Path = ALLOWED_SIGNERS,
    check_support: bool = True,
) -> RuntimeInstallResult:
    """Verify and install an Arc Llama Runtime release, from GitHub or a local *archive*.

    A local archive must sit next to its ``SHA256SUMS`` and ``SHA256SUMS.sig``. Nothing is extracted until the
    signature and the archive checksum both verify.
    """
    if cfg is None:
        cfg = load_config(config_path)
    if check_support:
        problems = support_problems(cfg)
        if problems:
            raise OptimizedRuntimeError("Arc Llama Runtime is not available here: " + "; ".join(problems) + ".")
    if dest is None:
        dest = Path(cfg.paths.state_dir).expanduser() / "runtime" if cfg.paths.state_dir else default_state_dir() / "runtime"
    dest.mkdir(parents=True, exist_ok=True)
    _recover_interrupted_publish(dest)

    work = Path(tempfile.mkdtemp(prefix=".arc-llama-runtime-dl-", dir=dest))
    own_client = False
    try:
        if archive is not None:
            archive = archive.expanduser().resolve()
            sums_path, sig_path, archive_path = archive.parent / SUMS_NAME, archive.parent / SIG_NAME, archive
            if not sums_path.is_file():
                raise OptimizedRuntimeError(f"{SUMS_NAME} not found next to {archive.name}.")
            release = None
        else:
            if client is None:
                client = httpx.Client(follow_redirects=True, timeout=300.0, headers=_auth_headers(),
                                      verify=_runtime_ssl_context())
                own_client = True
            release = _release(client, version)
            sums_path, sig_path = work / SUMS_NAME, work / SIG_NAME
            download_asset(client, _release_asset(release, SUMS_NAME), sums_path)
            download_asset(client, _release_asset(release, SIG_NAME), sig_path)
            archive_path = None

        verify_signature(sums_path, sig_path, allowed_signers)
        sums = parse_sums(sums_path.read_text(encoding="utf-8"))
        names = [n for n in sums if ASSET_RE.match(n)]
        if archive_path is not None:
            if archive_path.name not in sums:
                raise OptimizedRuntimeError(f"{archive_path.name} is not listed in the signed {SUMS_NAME}.")
            name = archive_path.name
        elif len(names) == 1:
            name = names[0]
        else:
            raise OptimizedRuntimeError(f"Expected one runtime archive in {SUMS_NAME}, found {len(names)}.")
        m = ASSET_RE.match(name)
        if m is None:
            raise OptimizedRuntimeError(f"Unexpected runtime archive name {name}.")
        install_dir = dest / f"arc-llama-runtime-{m.group('version')}+{m.group('tag')}"

        existing = _find_llama_server(install_dir) if install_dir.exists() else None
        if existing is not None and not force:
            binary = existing
        else:
            if archive_path is None:
                assert release is not None and client is not None
                archive_path = work / name
                download_asset(client, _release_asset(release, name), archive_path, on_progress=on_progress)
            if _sha256(archive_path) != sums[name]:
                raise OptimizedRuntimeError(f"Checksum mismatch for {name}; the download was not installed.")
            staging = Path(tempfile.mkdtemp(prefix=f".{install_dir.name}.staging-", dir=dest))
            try:
                staged = extract_archive(archive_path, staging)
                if not (staged.parent / MANIFEST_NAME).is_file():
                    raise OptimizedRuntimeError(f"{name} has no {MANIFEST_NAME}; not an Arc Llama Runtime.")
                (staging / MARKER_NAME).write_text(json.dumps({
                    "schema": 1, "source": "arc-llama-runtime", "asset": name,
                    "version": m.group("version"), "tag": m.group("tag"), "sha256": sums[name],
                }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                relative = staged.relative_to(staging)
                _publish_install(staging, install_dir)
                binary = install_dir / relative
            finally:
                shutil.rmtree(staging, ignore_errors=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if own_client and client is not None:
            client.close()

    if set_default:
        cfg.paths.llama_server = str(binary)
        for gpu in cfg.gpus:
            gpu.backend = "sycl"
        cfg.save(config_path)
    from arc_llama.arch import Backend

    return RuntimeInstallResult(
        binary_path=binary, backend=Backend.SYCL, requested_backend="sycl",
        tag=m.group("tag"), install_dir=install_dir, set_as_default=set_default,
    )


@functools.lru_cache(maxsize=32)
def _manifest_cached(path: str, mtime_ns: int) -> dict[str, Any] | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("Ignoring unreadable %s", path)
        return None
    return data if isinstance(data, dict) else None


def runtime_manifest(binary: str | Path) -> dict[str, Any] | None:
    """The Arc Llama Runtime manifest next to *binary*, or None for any other llama-server."""
    try:
        resolved = shutil.which(str(binary)) or str(binary)
        path = Path(resolved).expanduser().resolve().parent / MANIFEST_NAME
        st = path.stat()
    except OSError:
        return None
    return _manifest_cached(str(path), st.st_mtime_ns)


def model_runtime_defaults(binary: str | Path, model_path: str | Path) -> dict[str, str]:
    """Environment defaults the runtime at *binary* recommends for the model at *model_path*."""
    manifest = runtime_manifest(binary)
    entries = manifest.get("model_defaults") if manifest else None
    if not isinstance(entries, list) or not entries:
        return {}
    from arc_llama.gguf_meta import read_gguf_meta
    from arc_llama.model_compatibility import header_architecture

    try:
        with open(Path(model_path).expanduser(), "rb") as f:
            arch = header_architecture(f.read(1 << 20))
        is_moe = int(read_gguf_meta(model_path).get("expert_count") or 0) > 0
    except (OSError, ValueError, TypeError):
        return {}
    env: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("architecture") != arch:
            continue
        if entry.get("moe") is True and not is_moe:
            continue
        for key, value in (entry.get("env") or {}).items():
            if isinstance(key, str) and re.fullmatch(r"ARC_RT_[A-Z0-9_]+", key):
                env[key] = str(value)
    return env


def is_optimized_runtime(binary: str | Path) -> bool:
    return runtime_manifest(binary) is not None

