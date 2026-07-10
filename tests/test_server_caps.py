"""Tests for arc_llama.server_caps — llama-server --help sniffing."""
from __future__ import annotations

import os
import stat

from arc_llama.server_caps import DEFAULT_CAPS, _parse_help, probe_server_caps

NEW_STYLE_HELP = """\
-fa,   --flash-attn FA                  set Flash Attention use ('on', 'off', or 'auto', default: 'auto')
       (env: LLAMA_ARG_FLASH_ATTN)
-b,    --batch-size N                   logical maximum batch size (default: 2048)
"""

OLD_STYLE_HELP = """\
-fa, --flash-attn                       enable Flash Attention (default: disabled)
-b,  --batch-size N                     logical maximum batch size (default: 2048)
"""

NO_FA_HELP = """\
-b,  --batch-size N                     logical maximum batch size (default: 2048)
"""


class TestParseHelp:
    def test_new_style(self):
        caps = _parse_help(NEW_STYLE_HELP)
        assert caps.supports_flash_attn
        assert caps.flash_attn_takes_value
        assert caps.probed

    def test_old_style(self):
        caps = _parse_help(OLD_STYLE_HELP)
        assert caps.supports_flash_attn
        assert not caps.flash_attn_takes_value

    def test_no_flash_attn(self):
        caps = _parse_help(NO_FA_HELP)
        assert not caps.supports_flash_attn


class TestProbe:
    def _fake_server(self, tmp_path, help_text: str) -> str:
        script = tmp_path / "llama-server"
        script.write_text(f"#!/bin/sh\ncat <<'EOF'\n{help_text}\nEOF\n")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        return str(script)

    def test_probe_new_style_binary(self, tmp_path):
        caps = probe_server_caps(self._fake_server(tmp_path, NEW_STYLE_HELP))
        assert caps.probed
        assert caps.flash_attn_takes_value

    def test_probe_old_style_binary(self, tmp_path):
        caps = probe_server_caps(self._fake_server(tmp_path, OLD_STYLE_HELP))
        assert caps.probed
        assert caps.supports_flash_attn
        assert not caps.flash_attn_takes_value

    def test_missing_binary_gives_optimistic_defaults(self, tmp_path):
        caps = probe_server_caps(str(tmp_path / "nope"))
        assert caps == DEFAULT_CAPS
        assert not caps.probed

    def test_cache_invalidated_on_mtime_change(self, tmp_path):
        path = self._fake_server(tmp_path, OLD_STYLE_HELP)
        assert not probe_server_caps(path).flash_attn_takes_value
        # "Rebuild" the binary with new-style help and a newer mtime.
        with open(path, "w") as f:
            f.write(f"#!/bin/sh\ncat <<'EOF'\n{NEW_STYLE_HELP}\nEOF\n")
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
        os.utime(path, (os.stat(path).st_atime, os.stat(path).st_mtime + 10))
        assert probe_server_caps(path).flash_attn_takes_value
