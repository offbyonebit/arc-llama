"""Registration for host diagnostics and support-bundle commands."""
from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from arc_llama.arch import Arch, Backend, aot_arch_for, hardware_support
from arc_llama.binary import detect_backends, detect_llama_server_backend
from arc_llama.config import Config
from arc_llama.detect import lspci_intel_gpus
from arc_llama.platform_checks import (
    DoctorReport,
    format_bytes,
    kernel_module_loaded,
    level_zero_loader_present,
    max_memory_bar_bytes,
    oneapi_setvars_path,
    parse_kernel_version,
    rebar_likely_enabled,
    user_in_groups,
)


def register_diagnostic_commands(
    cli: click.Group, *, console: Console, load_config: Any, detect_gpus: Any,
    is_windows: Any,
) -> dict[str, Any]:
    """Register system diagnostics and support-bundle commands."""
    # ===========================================================================
    # doctor
    # ===========================================================================


    def _doctor_marker(ok: bool | None, severity: str = "info") -> str:
        if ok is True:
            return "[green]ok[/green]"
        if ok is None:
            return "[dim]?[/dim]"
        if severity == "fail":
            return "[red]FAIL[/red]"
        return "[yellow]warn[/yellow]"


    @cli.command()
    @click.pass_context
    def doctor(ctx: click.Context) -> None:
        """Diagnose the local environment for competitive Arc inference."""
        config_path: Path = ctx.obj["config_path"]
        console.print("[bold]arc-llama doctor[/bold]\n")
        report = DoctorReport()

        cfg: Config | None = None
        if config_path.exists():
            try:
                cfg = load_config(config_path)
            except Exception:
                console.print("[yellow]  warning: could not read config[/yellow]")

        # Kernel + driver (Linux-only diagnostics)
        if is_windows():
            console.print(f"  platform:      Windows {platform.release()}")
            console.print("  [dim]Kernel/driver checks are not available on Windows.[/dim]")
        else:
            uname = platform.uname()
            console.print(f"  kernel:        {uname.release}")
            kv = parse_kernel_version(uname.release)
            has_xe = kernel_module_loaded("xe")
            has_i915 = kernel_module_loaded("i915")
            console.print(f"  xe driver:     {'loaded' if has_xe else 'not loaded'}")
            console.print(f"  i915 driver:   {'loaded' if has_i915 else 'not loaded'}")
            if not has_xe and not has_i915:
                report.add(
                    "gpu_driver",
                    False,
                    "neither xe nor i915 module loaded",
                    severity="fail",
                    hint="Install/load the Intel GPU kernel driver for your generation.",
                )
            else:
                report.add(
                    "gpu_driver",
                    True,
                    f"xe={'yes' if has_xe else 'no'} i915={'yes' if has_i915 else 'no'}",
                )

        # GPU detection (enrich=True so clinfo populates VRAM where xe doesn't via sysfs)
        gpus = detect_gpus(enrich=True)
        if gpus:
            console.print(f"\n  detected {len(gpus)} Intel GPU(s):")
            for g in gpus:
                vram = f"{g.vram_gb} GB" if g.vram_gb else "VRAM unknown"
                console.print(
                    f"    - {g.name} @ {g.pci_slot}  ({g.arch.value}, driver={g.driver or '—'}, {vram})"
                )
                support = hardware_support(g.device_id, g.arch, Backend.SYCL)
                support_marker = "[green]validated[/green]" if support.status == "validated" else "[yellow]detected[/yellow]"
                console.print(f"        support: {support_marker}  {support.summary}")
                if g.notes:
                    for n in g.notes:
                        console.print(f"        note: {n}")
                # ReBAR aperture — critical for Arc llama.cpp performance
                if not is_windows() and g.sysfs_path:
                    bar = max_memory_bar_bytes(g.sysfs_path)
                    rebar = rebar_likely_enabled(g.sysfs_path, g.vram_mb)
                    bar_txt = format_bytes(bar) if bar is not None else "unknown"
                    if rebar is True:
                        marker = _doctor_marker(True)
                        console.print(f"        ReBAR:   {marker}  largest BAR {bar_txt}")
                        report.add("rebar", True, f"{g.pci_slot} BAR {bar_txt}")
                    elif rebar is False:
                        marker = _doctor_marker(False, "fail")
                        console.print(
                            f"        ReBAR:   {marker}  largest BAR {bar_txt} "
                            f"(need BIOS Resizable BAR / Above 4G Decoding)"
                        )
                        report.add(
                            "rebar",
                            False,
                            f"{g.pci_slot} BAR {bar_txt} — ReBAR looks off",
                            severity="fail",
                            hint="Enable Resizable BAR / Above 4G Decoding in BIOS. "
                            "Without it llama.cpp falls back to slow paths on Arc.",
                        )
                    else:
                        console.print(f"        ReBAR:   {_doctor_marker(None)}  largest BAR {bar_txt}")
                        report.add("rebar", None, f"{g.pci_slot} BAR {bar_txt}")
                # Battlemage wants a recent kernel
                if g.arch == Arch.BATTLEMAGE and not is_windows():
                    kv = parse_kernel_version()
                    if kv is not None and (kv[0], kv[1]) < (6, 14):
                        report.add(
                            "kernel_bmg",
                            False,
                            f"kernel {kv[0]}.{kv[1]} < 6.14 for Battlemage",
                            severity="warn",
                            hint="Kernel 6.14+ recommended for stable xe on Battlemage.",
                        )
                # AOT build guidance — eliminates the ~20s SYCL JIT cold start
                # that every model swap pays on Battlemage (where the JIT cache is
                # disabled to dodge a SIGSEGV). Compute the ocloc -device string
                # from the user's actual detected device ID rather than a static
                # hint, so an A770 owner sees `acm-g10` and a B580 owner sees
                # `bmg-g21`.
                aot = aot_arch_for(g.device_id)
                if aot is not None:
                    console.print(
                        f"        AOT:      [dim]build with "
                        f"-DGGML_SYCL_DEVICE_ARCH={aot} "
                        f"(ocloc -device {aot}) to skip the ~20s JIT cold start[/dim]"
                    )
            report.add("gpus", True, f"{len(gpus)} Intel GPU(s)")
        else:
            console.print("\n  [red]no Intel GPUs detected via sysfs[/red]")
            report.add("gpus", False, "no Intel GPUs via sysfs", severity="fail")
            raw = lspci_intel_gpus()
            if raw:
                console.print("\n  raw lspci output for Intel display devices:")
                for line in raw.splitlines():
                    console.print(f"    {line}")
            else:
                console.print("    lspci shows no Intel display devices either.")

        # External tools
        console.print("\n  external tools:")
        for tool in ("clinfo", "sycl-ls", "vulkaninfo", "intel_gpu_top", "nvtop", "lspci"):
            path = shutil.which(tool)
            console.print(f"    {tool:<14} {path or '— missing —'}")

        # Level Zero loader (required for SYCL)
        console.print("\n  Level Zero:")
        lz_ok, lz_path = level_zero_loader_present()
        if lz_ok:
            console.print(f"    loader:      {_doctor_marker(True)}  {lz_path}")
            report.add("level_zero", True, lz_path)
        else:
            console.print(
                f"    loader:      {_doctor_marker(False, 'warn')}  not found "
                f"(install intel-level-zero-gpu / compute-runtime for SYCL)"
            )
            report.add(
                "level_zero",
                False,
                "Level Zero loader not found",
                severity="warn",
                hint="Install intel-level-zero-gpu / intel-compute-runtime packages.",
            )

        # Permissions (Linux-only)
        if is_windows():
            console.print("\n  user groups:")
            console.print("    [dim]Group checks are not available on Windows.[/dim]")
        else:
            console.print("\n  user groups:")
            membership = user_in_groups("render", "video")
            for needed, ok in membership.items():
                console.print(f"    {needed:<14} {_doctor_marker(ok, 'warn' if not ok else 'info')}")
                report.add(
                    f"group_{needed}",
                    ok,
                    needed,
                    severity="warn" if not ok else "info",
                    hint=("sudo usermod -aG render,video $USER && re-login" if not ok else ""),
                )
            if not all(membership.values()):
                console.print(
                    "    [yellow]→ add yourself with `sudo usermod -aG render,video $USER` "
                    "and re-login.[/yellow]"
                )

        # oneAPI
        console.print("\n  oneAPI:")
        setvars = oneapi_setvars_path()
        if setvars is not None:
            console.print(f"    setvars:     {_doctor_marker(True)}  {setvars}")
            report.add("oneapi_setvars", True, str(setvars))
        else:
            console.print(
                f"    setvars:     {_doctor_marker(False, 'warn')}  not found — "
                f"install Intel oneAPI Base Toolkit if building llama.cpp from source"
            )
            report.add(
                "oneapi_setvars",
                False,
                "setvars not found",
                severity="warn",
                hint="Install Intel oneAPI Base Toolkit to build a SYCL llama-server.",
            )

        # llama-server binary
        console.print("\n  llama-server binary:")
        if cfg is not None:
            llama_server = Path(cfg.paths.llama_server).expanduser()
            if llama_server.exists():
                backends = detect_backends(llama_server)
                primary = detect_llama_server_backend(llama_server)
                backend_list = ", ".join(sorted(b.value for b in backends)) if backends else "unknown"
                console.print(f"    path:        {llama_server}")
                console.print(f"    backends:    {backend_list}")
                if primary is None:
                    report.add(
                        "llama_server",
                        False,
                        "binary present but no SYCL/Vulkan markers",
                        severity="fail",
                        hint="Rebuild llama-server with GGML_SYCL=ON (or Vulkan).",
                    )
                else:
                    report.add("llama_server", True, backend_list)
                if cfg.gpus:
                    for gpu_cfg in cfg.gpus:
                        if not gpu_cfg.enabled:
                            continue
                        want = gpu_cfg.backend
                        have = {b.value for b in backends}
                        if have and want not in have:
                            console.print(
                                f"    [yellow]→ GPU {gpu_cfg.pci_slot} wants "
                                f"'{want}' but binary has [{backend_list}].[/yellow]"
                            )
                            report.add(
                                f"backend_match_{gpu_cfg.pci_slot}",
                                False,
                                f"config={want} binary=[{backend_list}]",
                                severity="warn",
                            )
            else:
                console.print(f"    [yellow]not found[/yellow] at {llama_server}")
                console.print(
                    "    [yellow]→ run [bold]arc-llama install-runtime[/bold] to download "
                    "a portable Vulkan build.[/yellow]"
                )
                report.add(
                    "llama_server",
                    False,
                    f"missing at {llama_server}",
                    severity="fail",
                    hint="Run arc-llama install-runtime, or point paths.llama_server at a build.",
                )
        else:
            console.print("    [dim]no config loaded[/dim]")

        # Config
        console.print("\n  config:")
        if cfg is not None:
            console.print(f"    [green]found[/green] at {config_path}")
            if cfg.gpus:
                console.print("    GPUs in config:")
                for gpu_cfg in cfg.gpus:
                    status = "enabled" if gpu_cfg.enabled else "disabled"
                    console.print(
                        f"      - {gpu_cfg.pci_slot}  {gpu_cfg.name or gpu_cfg.arch}  "
                        f"backend={gpu_cfg.backend}  [{status}]"
                    )
        else:
            console.print(
                f"    [yellow]missing[/yellow] at {config_path} — run [bold]arc-llama init[/bold]."
            )
            report.add(
                "config",
                False,
                f"missing at {config_path}",
                severity="warn",
                hint="Run arc-llama init.",
            )

        # Plugins: import each without registering routes, so a broken add-on is
        # reported here instead of surfacing only in the serve log.
        from arc_llama.plugin_api import PLUGIN_API_VERSION
        from arc_llama.plugins import plugin_health

        console.print(f"\n  plugins (API {PLUGIN_API_VERSION}):")
        health = plugin_health()
        if not health:
            console.print("    [dim]none installed[/dim]")
        for entry in health:
            ok = entry["status"] in ("loaded", "disabled")
            version = f" v{entry['version']}" if entry.get("version") else ""
            console.print(
                f"    {entry['name']:<14} {_doctor_marker(ok, 'warn')}  {entry['status']}{version}"
            )
            if entry.get("error"):
                console.print(f"        {entry['error']}")
            if not ok:
                report.add(
                    f"plugin_{entry['name']}",
                    False,
                    f"{entry['status']}: {entry.get('error', '')}",
                    severity="warn",
                    hint="Update or uninstall the plugin, or exclude it with ARC_LLAMA_PLUGINS.",
                )

        # Summary of competitive-inference gates
        fails = report.failures
        warns = report.warnings
        console.print("\n  [bold]competitive-inference gates[/bold]")
        if not fails and not warns:
            console.print("    [green]all checked gates look good[/green]")
        else:
            for c in fails + warns:
                console.print(f"    {_doctor_marker(c.ok, c.severity)}  {c.name}: {c.detail}")
                if c.hint:
                    console.print(f"        → {c.hint}")
        if fails:
            # Non-zero so scripts/CI can gate on a healthy Arc host.
            sys.exit(2)


    @cli.command("support-bundle")
    @click.option(
        "--output",
        type=click.Path(path_type=Path, dir_okay=False),
        default=None,
        help="Output zip path (default: ./arc-llama-support-bundle.zip).",
    )
    @click.pass_context
    def support_bundle_cmd(ctx: click.Context, output: Path | None) -> None:
        """Write a redacted diagnostic bundle for troubleshooting."""
        from arc_llama.support_bundle import create_support_bundle

        config_path: Path = ctx.obj["config_path"]
        try:
            cfg = load_config(config_path) if config_path.exists() else None
        except Exception:
            cfg = None
        try:
            gpus = detect_gpus(enrich=True)
        except Exception:
            gpus = []
        destination = output or Path.cwd() / "arc-llama-support-bundle.zip"
        try:
            bundle = create_support_bundle(
                destination,
                config_path=config_path,
                cfg=cfg,
                gpus=gpus,
            )
        except FileExistsError as exc:
            raise click.ClickException(f"Output already exists: {destination}") from exc
        except OSError as exc:
            raise click.ClickException(f"Could not write support bundle: {exc}") from exc
        console.print(f"[green]Wrote support bundle[/green] {bundle}")
        console.print("[dim]Review the archive before sharing it.[/dim]")



    return {"_doctor_marker": _doctor_marker, "doctor": doctor, "support_bundle_cmd": support_bundle_cmd}
