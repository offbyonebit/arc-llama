"""Probe a llama-server binary for CLI capabilities.

llama.cpp's flag surface moves fast and arc-llama can't assume the user's
binary matches the Docker image's pin. The one incompatibility that actually
breaks launches today is `--flash-attn`:

  * old builds (pre ~b6300, Sep 2025): boolean flag, default *off* —
    `-fa on` is a parse error.
  * new builds: `-fa {on,off,auto}`, default *auto* — bare `-fa` is a
    parse error.

We run `llama-server --help` once (no GPU touched, returns immediately) and
sniff the help text. Results are cached per (path, mtime) so config reloads
and repeated launches don't re-exec.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("arc_llama.server_caps")

_HELP_TIMEOUT_S = 10


@dataclass(frozen=True)
class ServerCaps:
    """What the probed llama-server binary understands."""
    supports_flash_attn: bool = True
    flash_attn_takes_value: bool = True
    """True: `-fa {on,off,auto}` (new style). False: boolean `-fa` (old style)."""
    probed: bool = False
    """False when the probe failed and these are optimistic defaults."""


#: Assumed when the probe can't run (missing binary, timeout). Modern syntax
#: is the safe guess: `-fa auto` is also that style's default, so worst case
#: on an unprobeable old binary we only emit flags the user explicitly set.
DEFAULT_CAPS = ServerCaps()

_cache: dict[tuple[str, float], ServerCaps] = {}


def _parse_help(help_text: str) -> ServerCaps:
    idx = help_text.find("--flash-attn")
    if idx < 0:
        return ServerCaps(supports_flash_attn=False, flash_attn_takes_value=False, probed=True)
    # New-style help reads: "-fa, --flash-attn FA  set Flash Attention use
    # ('on', 'off', or 'auto', default: 'auto')". Old-style: "-fa, --flash-attn
    # enable Flash Attention (default: disabled)". 'auto' in the option's help
    # window is the discriminator.
    window = help_text[idx : idx + 240]
    takes_value = "auto" in window
    return ServerCaps(supports_flash_attn=True, flash_attn_takes_value=takes_value, probed=True)


def probe_server_caps(llama_server: str) -> ServerCaps:
    """Return the capabilities of the given llama-server binary, cached."""
    try:
        mtime = Path(llama_server).stat().st_mtime
    except OSError:
        # PATH-relative binary or missing file — probe uncached each call is
        # wasteful, so cache under mtime 0.
        mtime = 0.0
    key = (llama_server, mtime)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    try:
        proc = subprocess.run(
            [llama_server, "--help"],
            capture_output=True,
            text=True,
            timeout=_HELP_TIMEOUT_S,
        )
        caps = _parse_help(proc.stdout + proc.stderr)
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("could not probe %s (%s); assuming modern flag syntax", llama_server, e)
        caps = DEFAULT_CAPS
    _cache[key] = caps
    return caps
