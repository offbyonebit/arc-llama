"""CLI command: doctor."""
from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path

import click

from arc_llama.arch import aot_arch_for
from arc_llama.binary import detect_backends, detect_llama_server_backend
from arc_llama.cli_bindings import load_config
from arc_llama.detect import detect_gpus, lspci_intel_gpus
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

from .common import console


def _doctor_marker(ok: bool | None, severity: str = "info") -> str:
    if ok is True:
        return "[green]ok[/green]"
    if ok is None:
        return "[dim]?[/dim]"
    if severity == "fail":
        return "[red]FAIL[/red]"
    return "[yellow]warn[/yellow]"


@click.command()
@click.pass_context
def doctor_cmd(ctx: click.Context) -> None:
    """Diagnose the local environment for competitive Arc inference."""
    from arc_llama.arch import Arch
    from arc_llama.cli_commands.common import _IS_WINDOWS

    config_path: Path = ctx.obj["config_path"]
    console.print("[bold]arc-llama doctor[/bold]\n")
    report = DoctorReport()

    cfg = None
    if config_path.exists():
        try:
            cfg = load_config(config_path)
        except Exception:
            console.print("[yellow]  warning: could not read config[/yellow]")

    if _IS_WINDOWS:
        console.print(f"  platform:      Windows {platform.release()}")
        console.print(
            "  [dim]Kernel/driver checks are not available on Windows.[/dim]"
        )
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

    gpus = detect_gpus(enrich=True)
    if gpus:
        console.print(f"\n  detected {len(gpus)} Intel GPU(s):")
        for g in gpus:
            vram = f"{g.vram_gb} GB" if g.vram_gb else "VRAM unknown"
            console.print(
                f"    - {g.name} @ {g.pci_slot}  ({g.arch.value}, "
                f"driver={g.driver or '—'}, {vram})"
            )
            if g.notes:
                for n in g.notes:
                    console.print(f"        note: {n}")
            if not _IS_WINDOWS and g.sysfs_path:
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
                    console.print(
                        f"        ReBAR:   {_doctor_marker(None)}  largest BAR {bar_txt}"
                    )
                    report.add("rebar", None, f"{g.pci_slot} BAR {bar_txt}")
            if g.arch == Arch.BATTLEMAGE and not _IS_WINDOWS:
                kv = parse_kernel_version()
                if kv is not None and (kv[0], kv[1]) < (6, 14):
                    report.add(
                        "kernel_bmg",
                        False,
                        f"kernel {kv[0]}.{kv[1]} < 6.14 for Battlemage",
                        severity="warn",
                        hint="Kernel 6.14+ recommended for stable xe on Battlemage.",
                    )
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

    console.print("\n  external tools:")
    for tool in ("clinfo", "sycl-ls", "vulkaninfo", "intel_gpu_top", "nvtop", "lspci"):
        path = shutil.which(tool)
        console.print(f"    {tool:<14} {path or '— missing —'}")

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

    if _IS_WINDOWS:
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
                hint=(
                    "sudo usermod -aG render,video $USER && re-login"
                    if not ok
                    else ""
                ),
            )
        if not all(membership.values()):
            console.print(
                "    [yellow]→ add yourself with `sudo usermod -aG render,video $USER` "
                "and re-login.[/yellow]"
            )

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

    console.print("\n  llama-server binary:")
    if cfg is not None:
        llama_server = Path(cfg.paths.llama_server).expanduser()
        if llama_server.exists():
            backends = detect_backends(llama_server)
            primary = detect_llama_server_backend(llama_server)
            backend_list = (
                ", ".join(sorted(b.value for b in backends)) if backends else "unknown"
            )
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
            f"    [yellow]missing[/yellow] at {config_path} — run "
            f"[bold]arc-llama init[/bold]."
        )
        report.add(
            "config",
            False,
            f"missing at {config_path}",
            severity="warn",
            hint="Run arc-llama init.",
        )

    fails = report.failures
    warns = report.warnings
    console.print("\n  [bold]competitive-inference gates[/bold]")
    if not fails and not warns:
        console.print("    [green]all checked gates look good[/green]")
    else:
        for c in fails + warns:
            console.print(
                f"    {_doctor_marker(c.ok, c.severity)}  {c.name}: {c.detail}"
            )
            if c.hint:
                console.print(f"        → {c.hint}")
    if fails:
        sys.exit(2)


__all__ = ["doctor_cmd"]
