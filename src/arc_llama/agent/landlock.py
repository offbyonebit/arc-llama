"""Confine the built-in agent's shell commands to writing inside the project (Linux Landlock).

Landlock (Linux 5.13+) lets an unprivileged process give up filesystem rights
for itself and its children. Commands start through a tiny launcher that
restricts *writes* (create, delete, rename, modify, link, special files) to
the project folder, the temp folders and ``/dev``, then execs the shell.
Reading files, running programs and network access are unchanged, so builds,
tests and package installs into the project still work.

The launcher is a fresh ``python -I -S`` process rather than a ``preexec_fn``:
the server is multi-threaded, and running ctypes code between fork and exec
there is not safe.

This is not a VM boundary: a command can still read files the user can read and
use the network. See docs/work.md.
"""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
from pathlib import Path

_SYS_CREATE_RULESET, _SYS_ADD_RULE, _SYS_RESTRICT_SELF = 444, 445, 446
_CREATE_RULESET_VERSION = 1

_abi_cache: int | None = None


def abi_version() -> int:
    """The kernel's Landlock ABI version, or 0 when Landlock is unavailable."""
    global _abi_cache
    if _abi_cache is None:
        _abi_cache = 0
        if sys.platform.startswith("linux"):
            try:
                libc = ctypes.CDLL(None, use_errno=True)
                version = libc.syscall(_SYS_CREATE_RULESET, None, ctypes.c_size_t(0), _CREATE_RULESET_VERSION)
                _abi_cache = max(int(version), 0)
            except (OSError, AttributeError):
                _abi_cache = 0
    return _abi_cache


def enabled() -> bool:
    return os.environ.get("ARC_LLAMA_COMMAND_SANDBOX", "landlock").lower() != "off" and abi_version() > 0


def writable_roots(root: Path) -> list[str]:
    paths = [str(root), tempfile.gettempdir(), "/tmp", "/var/tmp", "/dev"]
    return list(dict.fromkeys(p for p in paths if os.path.isdir(p)))


# Runs in the launcher process: stdlib only, no Arc Llama imports.
_LAUNCHER = r"""
import ctypes, os, sys
WRITE = (1 << 1) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11) | (1 << 12)
abi = int(sys.argv[1]); sep = sys.argv.index("--")
paths, argv = sys.argv[2:sep], sys.argv[sep + 1:]
handled = WRITE | (1 << 13 if abi >= 2 else 0) | (1 << 14 if abi >= 3 else 0) | (1 << 15 if abi >= 5 else 0)
libc = ctypes.CDLL(None, use_errno=True)
class Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]
class PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]
def fail(what):
    err = ctypes.get_errno()
    sys.stderr.write("arc-llama sandbox: %s failed: %s\n" % (what, os.strerror(err)))
    os._exit(126)
attr = Ruleset(handled)
fd = libc.syscall(444, ctypes.byref(attr), ctypes.c_size_t(ctypes.sizeof(attr)), 0)
if fd < 0:
    fail("landlock_create_ruleset")
for path in paths:
    try:
        pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    except OSError:
        continue
    rule = PathBeneath(handled, pfd)
    if libc.syscall(445, fd, 1, ctypes.byref(rule), 0) != 0:
        fail("landlock_add_rule " + path)
    os.close(pfd)
if libc.prctl(38, 1, 0, 0, 0) != 0:
    fail("no_new_privs")
if libc.syscall(446, fd, 0) != 0:
    fail("landlock_restrict_self")
os.close(fd)
os.execv(argv[0], argv)
"""


def wrap(argv: list[str], root: Path) -> list[str]:
    """Return an argv that runs *argv* with writes confined to *root* and temp folders."""
    return [sys.executable, "-I", "-S", "-c", _LAUNCHER, str(abi_version()), *writable_roots(root), "--", *argv]
