"""CLI commands: benchmark, tune."""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict

import click

from arc_llama import benchmark as benchmark_mod
from arc_llama.autotune import compute_fingerprint, set_tuned_state
from arc_llama.cli_bindings import load_config
from arc_llama.tune import print_multi_summary, print_report, tune_all, tune_model

from .common import console, print_tune_status_table, server_url_from_ctx


@click.command("benchmark")
@click.argument("model")
@click.option(
    "--server", "server_url",
    default=None,
    help="Base URL of a running `arc-llama serve` (default: http://HOST:PORT from config).",
)
@click.option(
    "--prompt-tokens", "prompt_tokens",
    type=int, default=benchmark_mod.DEFAULT_PROMPT_TOKENS, show_default=True,
    help="Approximate prompt length to benchmark.",
)
@click.option(
    "--gen-tokens", "gen_tokens",
    type=int, default=benchmark_mod.DEFAULT_GEN_TOKENS, show_default=True,
    help="Number of tokens to generate.",
)
@click.option(
    "--sweep-ctx", "sweep_ctx",
    default="",
    help="Comma-separated ctx values for a sweep (e.g. 4096,8192,16384).",
)
@click.option(
    "--sweep-kv", "sweep_kv",
    default="",
    help="Comma-separated KV types for a sweep (e.g. f16,q8_0,q4_0).",
)
@click.option(
    "--kv", "kv_types", multiple=True,
    type=click.Choice(["f16", "q8_0", "q5_1", "q4_0"]),
    help="KV cache type(s) for --sweep-ctx (repeatable; default: f16 q8_0).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON instead of tables.")
@click.pass_context
def benchmark_cmd(
    ctx: click.Context,
    model: str,
    server_url: str | None,
    prompt_tokens: int,
    gen_tokens: int,
    sweep_ctx: str,
    sweep_kv: str,
    kv_types: tuple[str, ...],
    as_json: bool,
) -> None:
    """Measure prompt-eval and generation tok/s for MODEL."""

    cfg = load_config(ctx.obj["config_path"])
    if cfg.find_model(model) is None:
        console.print(f"[red]Model '{model}' is not registered in the config.[/red]")
        sys.exit(1)
    url = server_url_from_ctx(ctx, server_url)

    ctx_values = [int(x.strip()) for x in sweep_ctx.split(",") if x.strip()] if sweep_ctx else []
    kv_values = [x.strip() for x in sweep_kv.split(",") if x.strip()] if sweep_kv else []
    if kv_types:
        kv_values = list(kv_types)

    model_cfg = cfg.find_model(model)
    assert model_cfg is not None

    async def _run() -> int:
        if ctx_values or kv_values:
            recipe = model_cfg.recipe or {}
            results = await benchmark_mod.benchmark_sweep(
                url, model,
                ctx_values=ctx_values or [recipe.get("ctx", 4096)],
                kv_types=kv_values or [recipe.get("cache_type_k", "f16")],
                prompt_tokens=prompt_tokens,
                gen_tokens=gen_tokens,
                cfg=cfg,
            )
            if as_json:
                click.echo(json.dumps([r.to_dict() for r in results], indent=2))
            else:
                benchmark_mod.print_sweep_table(results)
            return 1 if all(r.error for r in results) else 0
        result = await benchmark_mod.benchmark_model(
            url, model,
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            cfg=cfg,
        )
        if as_json:
            click.echo(json.dumps(result.to_dict(), indent=2))
        else:
            benchmark_mod.print_result(result)
        return 1 if result.error else 0

    try:
        sys.exit(asyncio.run(_run()))
    except KeyboardInterrupt:
        console.print("[yellow]Benchmark interrupted.[/yellow]")
        sys.exit(130)


@click.command("tune")
@click.argument("model", required=False)
@click.option(
    "--all", "all_models", is_flag=True,
    help="Tune every registered model sequentially.",
)
@click.option(
    "--server", "server_url", default=None,
    help="Base URL of a running `arc-llama serve` (default: http://HOST:PORT from config).",
)
@click.option(
    "--target", type=click.Choice(["balanced", "generation", "prompt"]),
    default="balanced", show_default=True,
    help="What to optimise: generation tok/s, prompt-eval tok/s, or both.",
)
@click.option("--prompt-tokens", type=int, default=1024, show_default=True)
@click.option("--gen-tokens", type=int, default=128, show_default=True)
@click.option(
    "--apply/--dry-run", "apply_", default=True,
    help="Write the winning config into the model's recipe (default) or restore the original.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON instead of tables.")
@click.option(
    "--status", "status_only", is_flag=True,
    help="Print the per-model tune state table and exit without measuring.",
)
@click.pass_context
def tune_cmd(
    ctx: click.Context,
    model: str | None,
    all_models: bool,
    server_url: str | None,
    target: str,
    prompt_tokens: int,
    gen_tokens: int,
    apply_: bool,
    as_json: bool,
    status_only: bool,
) -> None:
    """Find the fastest recipe for MODEL by measuring, then persist it."""
    from arc_llama import __version__, workload

    cfg = load_config(ctx.obj["config_path"])

    if status_only:
        print_tune_status_table(cfg)
        sys.exit(0)

    if all_models and model:
        console.print("[red]Pass either MODEL or --all, not both.[/red]")
        sys.exit(1)
    if not all_models and not model:
        console.print("[red]Specify a MODEL to tune, or --all for every registered model.[/red]")
        sys.exit(1)

    url = server_url_from_ctx(ctx, server_url)

    try:
        if all_models:
            model_names = [m.name for m in cfg.models]
            if not model_names:
                console.print("[yellow]No models registered.[/yellow]")
                sys.exit(0)

            def on_start(name: str, i: int, total: int) -> None:
                console.print(f"[bold]\\[{i}/{total}] tuning {name}[/bold]")

            reports = asyncio.run(tune_all(
                url, model_names,
                target=target, prompt_tokens=prompt_tokens, gen_tokens=gen_tokens,
                apply=apply_, cfg=cfg, on_start=on_start,
            ))
            for r in reports:
                if not r.error and not r.aborted:
                    m = cfg.find_model(r.model)
                    if m is not None:
                        gpu = cfg.find_gpu(m.gpu_pci_slot)
                        fp = compute_fingerprint(
                            m, cfg.paths.llama_server, gpu, __version__,
                            workload.fingerprint_key(cfg.workload),
                        )
                        set_tuned_state(cfg, m, fp)
            try:
                cfg.save(ctx.obj["config_path"])
            except OSError as e:
                console.print(f"[yellow]Warning: failed to save tune state: {e}[/yellow]")
            if as_json:
                click.echo(json.dumps([asdict(r) for r in reports], indent=2, default=str))
            else:
                print_multi_summary(reports)
            sys.exit(1 if any(r.error for r in reports) else 0)

        if cfg.find_model(model) is None:
            console.print(f"[red]Model '{model}' is not registered in the config.[/red]")
            sys.exit(1)
        report = asyncio.run(tune_model(
            url, model,
            target=target, prompt_tokens=prompt_tokens, gen_tokens=gen_tokens,
            apply=apply_, cfg=cfg,
        ))
    except KeyboardInterrupt:
        console.print("[yellow]Tune interrupted.[/yellow]")
        sys.exit(130)

    if not report.error and not report.aborted:
        m = cfg.find_model(report.model)
        if m is not None:
            gpu = cfg.find_gpu(m.gpu_pci_slot)
            fp = compute_fingerprint(
                m, cfg.paths.llama_server, gpu, __version__,
                workload.fingerprint_key(cfg.workload),
            )
            set_tuned_state(cfg, m, fp)
            try:
                cfg.save(ctx.obj["config_path"])
            except OSError as e:
                console.print(f"[yellow]Warning: failed to save tune state: {e}[/yellow]")

    if as_json:
        click.echo(json.dumps(asdict(report), indent=2, default=str))
    else:
        print_report(report)
    sys.exit(1 if report.error else 0)


__all__ = ["benchmark_cmd", "tune_cmd"]
