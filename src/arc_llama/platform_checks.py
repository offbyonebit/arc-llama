"""Host / driver checks that matter for Arc inference performance.

Used by ``arc-llama doctor``. Detection is best-effort and never requires
root. Unknown results return None so the UI can say "could not determine".
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# PCI resource flag bits (linux/ioport.h)
_IORESOURCE_MEM = 0x00000200

# Without ReBAR, Arc typically exposes a 256 MiB aperture. Treat < 512 MiB as
# "ReBAR off / not usable"; >= 1 GiB as "ReBAR likely on".
_REBAR_OFF_MAX = 512 * 1024 * 1024
_REBAR_ON_MIN = 1024 * 1024 * 1024


@dataclass
class CheckResult:
    """One doctor line item."""
    name: str
    ok: bool | None  # True pass, False fail, None unknown/skip
    detail: str
    severity: str = "info"  # info | warn | fail
    hint: str = ""


@dataclass
class DoctorReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.ok is False and c.severity == "fail"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.ok is False and c.severity == "warn"]

    def add(
        self,
        name: str,
        ok: bool | None,
        detail: str,
        *,
        severity: str = "info",
        hint: str = "",
    ) -> None:
        if ok is False and severity == "info":
            severity = "warn"
        self.checks.append(
            CheckResult(name=name, ok=ok, detail=detail, severity=severity, hint=hint)
        )


def max_memory_bar_bytes(sysfs_path: str | Path) -> int | None:
    """Largest MMIO BAR size (bytes) for a PCI device via sysfs ``resource``."""
    path = Path(sysfs_path) / "resource"
    if not path.is_file():
        return None
    try:
        text = path.read_text()
    except OSError:
        return None
    largest = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            start = int(parts[0], 16)
            end = int(parts[1], 16)
            flags = int(parts[2], 16)
        except ValueError:
            continue
        if start == 0 and end == 0:
            continue
        if not (flags & _IORESOURCE_MEM):
            continue
        size = end - start + 1
        if size > largest:
            largest = size
    return largest or None


def rebar_likely_enabled(
    sysfs_path: str | Path,
    vram_mb: int | None = None,
) -> bool | None:
    """Heuristic: is Resizable BAR exposing a large VRAM aperture?

    Returns True / False / None (unknown).
    """
    bar = max_memory_bar_bytes(sysfs_path)
    if bar is None:
        return None
    if bar < _REBAR_OFF_MAX:
        return False
    if vram_mb and vram_mb > 0:
        # Half of device VRAM mapped is a strong ReBAR signal.
        if bar >= (vram_mb * 1024 * 1024) // 2:
            return True
    if bar >= _REBAR_ON_MIN:
        return True
    # Between 512 MiB and 1 GiB — ambiguous (some partial BAR configs).
    return None


def format_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GiB"
    if n >= 1024**2:
        return f"{n / 1024**2:.0f} MiB"
    return f"{n} B"


def user_in_groups(*needed: str) -> dict[str, bool]:
    """Return membership for each group name (Linux). Empty on Windows."""
    if sys.platform == "win32":
        return {g: False for g in needed}
    try:
        out = subprocess.run(
            ["id", "-nG"], capture_output=True, text=True, timeout=2, check=False,
        )
        groups = set(out.stdout.split())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        groups = set()
    return {g: g in groups for g in needed}


def level_zero_loader_present() -> tuple[bool, str]:
    """Look for Level Zero loader library on common paths / ldconfig (Linux) or PATH / oneAPI roots (Windows)."""
    if sys.platform == "win32":
        names = (
            "libze_loader.dll",
            "ze_loader.dll",
        )
        search_dirs: list[Path] = []
        program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        search_dirs.append(Path(program_files_x86) / "Intel" / "oneAPI")
        oneapi_root = os.environ.get("ONEAPI_ROOT") or os.environ.get("CMPLR_ROOT")
        if oneapi_root:
            p = Path(oneapi_root)
            search_dirs.extend([p, p / "lib", p / "bin"])
        # Also check directories on PATH.
        for entry in os.environ.get("PATH", "").split(os.pathsep):
            entry = entry.strip()
            if entry:
                search_dirs.append(Path(entry))

        for d in search_dirs:
            if not d.is_dir():
                continue
            for name in names:
                candidate = d / name
                if candidate.exists():
                    return True, str(candidate)
                try:
                    for child in d.iterdir():
                        if child.is_dir():
                            c2 = child / name
                            if c2.exists():
                                return True, str(c2)
                except OSError:
                    pass
        return False, ""

    names = (
        "libze_loader.so.1",
        "libze_loader.so",
        "libze_intel_gpu.so.1",
        "libze_intel_gpu.so",
    )
    search_dirs = [
        Path("/usr/lib"),
        Path("/usr/lib64"),
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/local/lib"),
        Path("/opt/intel/oneapi/lib"),
        Path("/opt/intel/oneapi/lib/intel64"),
    ]
    # Also walk ONEAPI_ROOT if set.
    oneapi_root = os.environ.get("ONEAPI_ROOT") or os.environ.get("CMPLR_ROOT")
    if oneapi_root:
        search_dirs.append(Path(oneapi_root))
        search_dirs.append(Path(oneapi_root) / "lib")
        search_dirs.append(Path(oneapi_root) / "lib" / "intel64")

    for d in search_dirs:
        if not d.is_dir():
            continue
        for name in names:
            # Direct child
            candidate = d / name
            if candidate.exists():
                return True, str(candidate)
            # One level of subdirs (e.g. lib/intel64)
            try:
                for child in d.iterdir():
                    if child.is_dir():
                        c2 = child / name
                        if c2.exists():
                            return True, str(c2)
            except OSError:
                pass

    # ldconfig -p is authoritative when present.
    ldconfig = shutil.which("ldconfig")
    if ldconfig:
        try:
            out = subprocess.run(
                [ldconfig, "-p"], capture_output=True, text=True, timeout=5, check=False,
            )
            for name in names:
                if name in out.stdout:
                    m = re.search(rf"{re.escape(name)}[^\n]*=>\s*(\S+)", out.stdout)
                    return True, m.group(1) if m else name
        except (OSError, subprocess.TimeoutExpired):
            pass
    return False, ""


def oneapi_setvars_path() -> Path | None:
    """Return the path to Intel oneAPI's setvars script, if found.

    Searches, in order:
      - ``$ONEAPI_ROOT/setvars.sh`` (or ``.bat`` on Windows)
      - ``$CMPLR_ROOT/../setvars.sh`` (common when only the compiler module is active)
      - ``/opt/intel/oneapi/setvars.sh`` (standard apt install)
      - ``/usr/local/intel/oneapi/setvars.sh`` (common tarball/custom prefix)
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        oneapi_root = os.environ.get("ONEAPI_ROOT")
        if oneapi_root:
            p = Path(oneapi_root) / "setvars.bat"
            if p.exists():
                return p
        cmplr_root = os.environ.get("CMPLR_ROOT")
        if cmplr_root:
            # CMPLR_ROOT points at e.g. ...\oneapi\compiler\2026.1; setvars.bat is two
            # levels above, at the oneapi root.
            p = Path(cmplr_root).parent.parent / "setvars.bat"
            if p.exists():
                return p
        p = base / "Intel" / "oneAPI" / "setvars.bat"
        return p if p.exists() else None

    candidates: list[Path] = []
    oneapi_root = os.environ.get("ONEAPI_ROOT")
    if oneapi_root:
        candidates.append(Path(oneapi_root) / "setvars.sh")
    cmplr_root = os.environ.get("CMPLR_ROOT")
    if cmplr_root:
        # CMPLR_ROOT points at e.g. .../oneapi/compiler/2026.1; setvars.sh is two
        # levels above, at the oneapi root.
        candidates.append(Path(cmplr_root).parent.parent / "setvars.sh")
    candidates.extend([
        Path("/opt/intel/oneapi/setvars.sh"),
        Path("/usr/local/intel/oneapi/setvars.sh"),
    ])
    for p in candidates:
        if p.exists():
            return p
    return None


def oneapi_runtime_env_needed() -> bool:
    """Heuristic: does the current process environment lack oneAPI runtime libs?

    Returns True when neither the Level Zero loader nor the SVML/compiler runtime
    can be found via ldconfig or common oneAPI paths. In that state a SYCL
    llama-server binary is likely to fail at startup with missing-library errors
    or "No device of requested type available".
    """
    if sys.platform == "win32":
        names = (
            "libsvml.dll",
            "libze_loader.dll",
            "ze_loader.dll",
        )
        search_dirs: list[Path] = []
        program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        search_dirs.append(Path(program_files_x86) / "Intel" / "oneAPI")
        oneapi_root = os.environ.get("ONEAPI_ROOT") or os.environ.get("CMPLR_ROOT")
        if oneapi_root:
            p = Path(oneapi_root)
            search_dirs.extend([p, p / "lib", p / "bin"])
        for entry in os.environ.get("PATH", "").split(os.pathsep):
            entry = entry.strip()
            if entry:
                search_dirs.append(Path(entry))

        for d in search_dirs:
            if not d.is_dir():
                continue
            for name in names:
                if (d / name).exists():
                    return False
                try:
                    for child in d.iterdir():
                        if child.is_dir() and (child / name).exists():
                            return False
                except OSError:
                    pass
        return True

    names = (
        "libsvml.so",
        "libsvml.so.2",
        "libze_loader.so.1",
        "libze_loader.so",
    )
    search_dirs = [
        Path("/opt/intel/oneapi/lib"),
        Path("/opt/intel/oneapi/lib/intel64"),
        Path("/usr/local/intel/oneapi/lib"),
        Path("/usr/local/intel/oneapi/lib/intel64"),
    ]
    oneapi_root = os.environ.get("ONEAPI_ROOT") or os.environ.get("CMPLR_ROOT")
    if oneapi_root:
        p = Path(oneapi_root)
        search_dirs.extend([p, p / "lib", p / "lib" / "intel64"])

    # If ldconfig can resolve any of the runtime libs, the env/system is fine.
    ldconfig = shutil.which("ldconfig")
    if ldconfig:
        try:
            out = subprocess.run(
                [ldconfig, "-p"], capture_output=True, text=True, timeout=5, check=False,
            )
            for name in names:
                if name in out.stdout:
                    return False
        except (OSError, subprocess.TimeoutExpired):
            pass

    # Check common oneAPI prefixes directly.
    for d in search_dirs:
        if not d.is_dir():
            continue
        for name in names:
            if (d / name).exists():
                return False
            try:
                for child in d.iterdir():
                    if child.is_dir() and (child / name).exists():
                        return False
            except OSError:
                pass

    return True


def source_setvars(setvars_path: Path | str) -> dict[str, str]:
    """Source a oneAPI setvars script and return the resulting environment diff.

    On Linux, runs the script in a bash subprocess and parses null-delimited
    env output. On Windows, runs the equivalent ``setvars.bat`` in ``cmd.exe``
    and parses the ``set`` output. If the script does not exist or fails,
    returns an empty dict.
    """
    setvars_path = Path(setvars_path)
    if not setvars_path.exists():
        return {}

    if sys.platform == "win32":
        # setvars.bat must be run from its own directory or it complains.
        # We run a tiny wrapper batch that calls setvars.bat (using `call` so
        # control returns to the wrapper) then dumps the environment.
        import tempfile

        wrapper_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".bat", delete=False, encoding="utf-8"
            ) as wrapper:
                wrapper_path = Path(wrapper.name)
                wrapper.write("@echo off\n")
                wrapper.write(f'cd /d "{setvars_path.parent}"\n')
                # `call` is required so cmd resumes the wrapper after the batch.
                wrapper.write(f'call "{setvars_path}"\n')
                wrapper.write("set\n")
            out = subprocess.run(
                ["cmd", "/C", str(wrapper_path)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return {}
        finally:
            if wrapper_path is not None:
                try:
                    os.unlink(wrapper_path)
                except OSError:
                    pass
        if out.returncode != 0:
            return {}
        sourced: dict[str, str] = {}
        for line in out.stdout.splitlines():
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            sourced[key] = value
        return sourced

    # Use bash to source the script and dump the environment. setvars.sh is
    # a bash script, so this is the most reliable way to capture its effects.
    script = (
        f"source '{setvars_path}' > /dev/null 2>&1; "
        "env -0 | sort -z"
    )
    try:
        out = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=False,
            timeout=60,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return {}

    if out.returncode != 0:
        return {}

    # Parse null-delimited env output.
    sourced = {}
    for item in out.stdout.split(b"\x00"):
        if b"=" not in item:
            continue
        key, _, value = item.partition(b"=")
        key_str = key.decode(errors="replace")
        # Skip bash-only bookkeeping variables.
        if key_str in ("PWD", "SHLVL", "_", "SHELLOPTS"):
            continue
        sourced[key_str] = value.decode(errors="replace")

    return sourced


def kernel_module_loaded(name: str) -> bool:
    if sys.platform == "win32":
        return False
    return Path(f"/sys/module/{name}").exists()


def parse_kernel_version(release: str | None = None) -> tuple[int, int] | None:
    """Return (major, minor) from ``uname -r`` style string."""
    if release is None:
        release = os.uname().release if hasattr(os, "uname") else ""
    m = re.match(r"(\d+)\.(\d+)", release or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))
