"""Local interface addresses for sharing the chat URL; no connectivity probes."""
from __future__ import annotations

import ipaddress
import json
import socket
import struct
import subprocess
import sys
from pathlib import Path

from arc_llama.config import ServerConfig


def local_addresses() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if sys.platform == "win32":
        # Get-NetIPAddress reads the local adapter configuration, including
        # Tailscale's adapter, without DNS, route probes or network requests.
        try:
            result = subprocess.run(
                ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
                 "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); "
                 "@(Get-NetIPAddress | Select-Object InterfaceAlias,IPAddress) | ConvertTo-Json -Compress"],
                capture_output=True, timeout=3, check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            rows = json.loads(result.stdout.decode("utf-8-sig"))
            if isinstance(rows, dict):
                rows = [rows]
            if isinstance(rows, list):
                found = [(row["InterfaceAlias"], row["IPAddress"]) for row in rows
                         if isinstance(row, dict) and isinstance(row.get("InterfaceAlias"), str)
                         and isinstance(row.get("IPAddress"), str)]
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    elif sys.platform.startswith("linux"):
        import fcntl

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                for _, name in socket.if_nameindex():
                    try:
                        value = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", name.encode()[:15]))
                        found.append((name, socket.inet_ntoa(value[20:24])))
                    except OSError:
                        continue
        except OSError:
            pass
        try:
            for line in Path("/proc/net/if_inet6").read_text().splitlines():
                address, _, _, _, _, name = line.split()
                found.append((name, str(ipaddress.IPv6Address(int(address, 16)))))
        except (OSError, ValueError):
            pass
    return found


def network_payload(server: ServerConfig, addresses: list[tuple[str, str]]) -> dict:
    try:
        bound = ipaddress.ip_address(server.host.strip("[]"))
    except ValueError:
        return {"enabled": False, "addresses": [], "hint": "Run arc-llama serve --lan to enable phone access."}
    enabled = not bound.is_loopback
    urls = []
    seen = set()
    if enabled:
        for interface, text in addresses:
            try:
                address = ipaddress.ip_address(text.split("%", 1)[0])
            except ValueError:
                continue
            if (address.is_loopback or address.is_unspecified or address.is_link_local
                    or address.is_multicast or address.version != bound.version):
                continue
            if not bound.is_unspecified and address != bound:
                continue
            host = f"[{address}]" if address.version == 6 else str(address)
            url = f"http://{host}:{server.port}/chat"
            if url in seen:
                continue
            seen.add(url)
            tailscale = "tailscale" in interface.casefold()
            urls.append({"url": url, "interface": interface, "kind": "tailscale" if tailscale else "lan"})
    return {"enabled": enabled, "addresses": urls, "hint": (
        "Connect your phone to this LAN or Tailscale, then scan an address. Enter the existing LAN API key in Chat if asked."
        if urls else "No address was found for this bind interface. Check its local adapter configuration."
        if enabled else "Run arc-llama serve --lan to enable phone access."
    )}
