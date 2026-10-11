"""Registration for benchmark, tuning, and speculative decoding commands."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import click


def register_performance_commands(
    cli: click.Group, *, console: Any, benchmark_mod: Any, httpx: Any,
    load_config: Any, server_url_from: Any, print_tune_status_table: Any,
    emit_recipe_submission: Any, save_or_die: Any,
) -> dict[str, Any]:
    """Register commands that measure or select model execution recipes."""
    @cli.command("benchmark")
    @click.argument("model")
    @click.option(
        "--server",
        "server_url",
        default=None,
        help="Base URL of a running `arc-llama serve` (default: http://HOST:PORT from config).",
    )
    @click.option(
        "--prompt-tokens",
        "prompt_tokens",
        type=int,
        default=benchmark_mod.DEFAULT_PROMPT_TOKENS,
        show_default=True,
        help="Approximate prompt length to benchmark.",
    )
    @click.option(
        "--gen-tokens",
        "gen_tokens",
        type=int,
        default=benchmark_mod.DEFAULT_GEN_TOKENS,
        show_default=True,
        help="Number of tokens to generate.",
    )
    @click.option(
        "--sweep-ctx",
        "sweep_ctx",
        default="",
        help="Comma-separated ctx values for a sweep (e.g. 4096,8192,16384).",
    )
    @click.option(
        "--sweep-kv",
        "sweep_kv",
        default="",
        help="Comma-separated KV types for a sweep (e.g. f16,q8_0,q4_0).",
    )
    @click.option(
        "--kv",
        "kv_types",
        multiple=True,
        type=click.Choice(["f16", "q8_0", "q5_1", "q4_0"]),
        help="KV cache type(s) for --sweep-ctx (repeatable; default: f16 q8_0).",
    )
    @click.option(
        "--depths",
        default=None,
        help="Measure decode speed after these prefill sizes instead "
        "(e.g. 0,4096,16384,32768). Use 'default' for that set.",
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
        depths: str | None,
        as_json: bool,
    ) -> None:
        """Measure prompt-eval and generation tok/s for MODEL.

        Requires a running `arc-llama serve` — measurements go through the
        router so they use the exact SYCL env and recipe your requests get.
        """
        cfg = load_config(ctx.obj["config_path"])
        if cfg.find_model(model) is None:
            console.print(f"[red]Model '{model}' is not registered in the config.[/red]")
            sys.exit(1)
        url = server_url_from(ctx, server_url)

        ctx_values = [int(x.strip()) for x in sweep_ctx.split(",") if x.strip()] if sweep_ctx else []
        kv_values = [x.strip() for x in sweep_kv.split(",") if x.strip()] if sweep_kv else []
        if kv_types:
            kv_values = list(kv_types)

        model_cfg = cfg.find_model(model)
        assert model_cfg is not None

        depth_values: list[int] = []
        if depths:
            if depths.strip() == "default":
                depth_values = list(benchmark_mod.DEFAULT_DEPTHS)
            else:
                try:
                    depth_values = [int(x) for x in depths.split(",") if x.strip()]
                except ValueError as exc:
                    raise click.BadParameter("use comma-separated integers", param_hint="--depths") from exc
            if not depth_values or min(depth_values) < 0:
                raise click.BadParameter("depths must be non-negative", param_hint="--depths")

        async def _run() -> int:
            if depth_values:
                ctx_size = (model_cfg.recipe or {}).get("ctx")
                runnable, skipped = benchmark_mod.usable_depths(depth_values, ctx_size, gen_tokens)
                if not runnable:
                    console.print(f"[red]No depth fits in {model}'s context ({ctx_size}).[/red]")
                    return 1
                rows = await benchmark_mod.benchmark_depths(url, model, runnable, gen_tokens=gen_tokens)
                if as_json:
                    click.echo(
                        json.dumps(
                            {"results": [r.to_dict() for r in rows], "skipped": skipped}, indent=2
                        )
                    )
                else:
                    benchmark_mod.print_depth_table(model, rows, skipped)
                return 1 if all(r.error for r in rows) else 0
            if ctx_values or kv_values:
                recipe = model_cfg.recipe or {}
                results = await benchmark_mod.benchmark_sweep(
                    url,
                    model,
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
                url,
                model,
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


    @cli.command("tune")
    @click.argument("model", required=False)
    @click.option(
        "--all",
        "all_models",
        is_flag=True,
        help="Tune every registered model sequentially.",
    )
    @click.option(
        "--server",
        "server_url",
        default=None,
        help="Base URL of a running `arc-llama serve` (default: http://HOST:PORT from config).",
    )
    @click.option(
        "--target",
        type=click.Choice(["balanced", "generation", "prompt"]),
        default="balanced",
        show_default=True,
        help="What to optimise: generation tok/s, prompt-eval tok/s, or both.",
    )
    @click.option("--prompt-tokens", type=int, default=1024, show_default=True)
    @click.option("--gen-tokens", type=int, default=128, show_default=True)
    @click.option(
        "--apply/--dry-run",
        "apply_",
        default=True,
        help="Write the winning config into the model's recipe (default) or restore the original.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Emit raw JSON instead of tables.")
    @click.option(
        "--share",
        "share",
        is_flag=True,
        help="After a successful tune, emit a community-registry submission "
        "(JSON file + PR link). Opt-in; nothing is uploaded automatically.",
    )
    @click.option(
        "--status",
        "status_only",
        is_flag=True,
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
        share: bool,
        status_only: bool,
    ) -> None:
        """Find the fastest recipe for MODEL by measuring, then persist it.

        Staged sweep over KV cache type, ubatch size, and flash attention —
        roughly 6–9 measured configs, each paying one model reload. Expect
        ~10 minutes on a Battlemage-class card. Pass `--all` to sweep every
        registered model in one run. Requires a running `arc-llama serve`.
        """
        from dataclasses import asdict

        from arc_llama.autotune import (
            compute_fingerprint,
            set_tuned_state,
        )
        from arc_llama.tune import print_multi_summary, print_report, tune_all, tune_model

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

        url = server_url_from(ctx, server_url)

        try:
            if all_models:
                model_names = [m.name for m in cfg.models]
                if not model_names:
                    console.print("[yellow]No models registered.[/yellow]")
                    sys.exit(0)

                def on_start(name: str, i: int, total: int) -> None:
                    console.print(f"[bold]\\[{i}/{total}] tuning {name}[/bold]")

                reports = asyncio.run(
                    tune_all(
                        url,
                        model_names,
                        target=target,
                        prompt_tokens=prompt_tokens,
                        gen_tokens=gen_tokens,
                        apply=apply_,
                        cfg=cfg,
                        on_start=on_start,
                    )
                )
                # Dry-run must leave tune state untouched: recording "tuned" with a
                # matching fingerprint makes background auto-tune skip the model
                # forever, turning a look-don't-touch run into a permanent opt-out.
                if apply_:
                    for r in reports:
                        if not r.error and not r.aborted:
                            m = cfg.find_model(r.model)
                            if m is not None:
                                gpu = cfg.find_gpu(m.gpu_pci_slot)
                                from arc_llama import __version__, workload

                                fp = compute_fingerprint(
                                    m,
                                    cfg.paths.llama_server,
                                    gpu,
                                    __version__,
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

            assert model is not None
            if cfg.find_model(model) is None:
                console.print(f"[red]Model '{model}' is not registered in the config.[/red]")
                sys.exit(1)
            report = asyncio.run(
                tune_model(
                    url,
                    model,
                    target=target,
                    prompt_tokens=prompt_tokens,
                    gen_tokens=gen_tokens,
                    apply=apply_,
                    cfg=cfg,
                )
            )
        except KeyboardInterrupt:
            console.print("[yellow]Tune interrupted.[/yellow]")
            sys.exit(130)

        # Same dry-run guard as the --all branch above.
        if apply_ and not report.error and not report.aborted:
            m = cfg.find_model(report.model)
            if m is not None:
                gpu = cfg.find_gpu(m.gpu_pci_slot)
                from arc_llama import __version__, workload

                fp = compute_fingerprint(
                    m,
                    cfg.paths.llama_server,
                    gpu,
                    __version__,
                    workload.fingerprint_key(cfg.workload),
                )
                set_tuned_state(cfg, m, fp)
                try:
                    cfg.save(ctx.obj["config_path"])
                except OSError as e:
                    console.print(f"[yellow]Warning: failed to save tune state: {e}[/yellow]")

        if share and apply_ and not report.error and not report.aborted:
            emit_recipe_submission(ctx, cfg, report)

        if as_json:
            click.echo(json.dumps(asdict(report), indent=2, default=str))
        else:
            print_report(report)
        sys.exit(1 if report.error else 0)



    # ===========================================================================
    # speculative
    # ===========================================================================


    @cli.command("speculative")
    @click.argument("model")
    @click.option("--status", "show_status", is_flag=True, help="Show support and saved recipe.")
    @click.option("--dry-run", is_flag=True, help="Show safe candidates without changing config.")
    @click.option("--off", "turn_off", is_flag=True, help="Disable speculative decoding.")
    @click.option(
        "--auto",
        "auto_select",
        is_flag=True,
        help="Pick a draft automatically; with --verify, measure every fitting draft "
        "and n-gram, then keep the fastest.",
    )
    @click.option("--draft", "draft_name", default=None, help="Registered model name to use as draft.")
    @click.option(
        "--ngram", "use_ngram", is_flag=True, help="Use llama.cpp n-gram speculation when supported."
    )
    @click.option("--draft-tokens", default=4, show_default=True, type=click.IntRange(1, 16))
    @click.option(
        "--verify/--no-verify",
        default=True,
        help="A/B benchmark target-only versus speculation and roll back unless it is faster.",
    )
    @click.option(
        "--min-speedup",
        type=click.FloatRange(min=0.0),
        default=0.02,
        show_default=True,
        help="Minimum generation-speed improvement required by --verify.",
    )
    @click.pass_context
    def speculative_cmd(
        ctx: click.Context,
        model: str,
        show_status: bool,
        dry_run: bool,
        turn_off: bool,
        auto_select: bool,
        draft_name: str | None,
        use_ngram: bool,
        draft_tokens: int,
        verify: bool,
        min_speedup: float,
    ) -> None:
        """Configure native llama.cpp speculation and prove the speedup locally."""
        from arc_llama.recipe_share import benchmark_improvement
        from arc_llama.server_caps import format_speculation_capability, probe_server_caps
        from arc_llama.speculation import discover_drafts
        from arc_llama.tune import _apply_edits, _restore_final_state

        cfg_path: Path = ctx.obj["config_path"]
        cfg = load_config(cfg_path)
        target = cfg.find_model(model)
        if target is None:
            raise click.ClickException(f"unknown model {model!r}")
        caps = probe_server_caps(cfg.paths.llama_server)
        candidates = discover_drafts(cfg, target)
        recipe = target.recipe

        if show_status or dry_run or not any((turn_off, auto_select, draft_name, use_ngram)):
            console.print(f"[bold]{target.name}[/bold]")
            console.print(f"  llama-server: {cfg.paths.llama_server}")
            console.print(
                f"  llama-server speculation: {format_speculation_capability(caps)} "
                f"(probed={'yes' if caps.probed else 'no'})"
            )
            console.print(f"  configured: {recipe.get('spec_type', 'off')}")
            if recipe.get("spec_draft_name"):
                console.print(f"  draft: {recipe['spec_draft_name']}")
            if recipe.get("speculation_result"):
                console.print(f"  verification: {recipe['speculation_result']}")
            if candidates:
                console.print("  draft candidates:")
                for c in candidates:
                    marker = "fit" if c.fits else "does not fit"
                    console.print(f"    {c.name}: ~{c.estimated_mb} MiB ({marker}; {c.reason})")
            else:
                console.print(
                    "  draft candidates: none (only smaller same-family registered models qualify)"
                )
            if dry_run or show_status or not any((turn_off, auto_select, draft_name, use_ngram)):
                return

        if turn_off:
            for key in (
                "spec_type",
                "spec_draft_name",
                "spec_draft_model",
                "spec_draft_ngl",
                "spec_draft_n_max",
            ):
                recipe.pop(key, None)
            recipe["speculation_result"] = "disabled by user"
            save_or_die(cfg, cfg_path)
            console.print(f"[green]Disabled speculation for {target.name}.[/green]")
            return

        # Each proposal is (recipe edits, description). --auto with --verify
        # measures every fitting draft (the three smallest) and n-gram when the
        # binary has it, against one target-only baseline, and keeps the winner.
        proposals: list[tuple[dict[str, Any], str]] = []

        def draft_proposal(name: str) -> tuple[dict[str, Any], str]:
            return (
                {"spec_type": "draft-simple", "spec_draft_name": name, "spec_draft_n_max": draft_tokens},
                f"draft {name}/{draft_tokens}",
            )

        ngram_proposal: tuple[dict[str, Any], str] = (
            {"spec_type": "ngram-simple", "spec_draft_name": None, "spec_draft_n_max": draft_tokens},
            f"n-gram/{draft_tokens}",
        )
        if use_ngram:
            if not caps.supports_ngram:
                raise click.ClickException(
                    "installed llama-server does not advertise n-gram speculation"
                )
            proposals.append(ngram_proposal)
        elif draft_name:
            candidate = next((c for c in candidates if c.name == draft_name and c.fits), None)
            if candidate is None:
                raise click.ClickException(f"{draft_name!r} is not a fitting draft for {target.name}")
            if not caps.supports_draft_model:
                raise click.ClickException(
                    "installed llama-server does not advertise --spec-draft-model"
                )
            proposals.append(draft_proposal(candidate.name))
        else:
            fitting = [c for c in candidates if c.fits] if caps.supports_draft_model else []
            for c in fitting[: 3 if verify else 1]:
                proposals.append(draft_proposal(c.name))
            if verify and caps.supports_ngram:
                proposals.append(ngram_proposal)
            if not proposals:
                raise click.ClickException(
                    "no fitting registered draft candidate; add a smaller same-family model or use --ngram"
                )

        def save_choice(proposed: dict[str, Any], result: str) -> None:
            for key, value in proposed.items():
                if value is None:
                    recipe.pop(key, None)
                else:
                    recipe[key] = value
            recipe.pop("spec_draft_model", None)
            recipe["speculation_result"] = result
            save_or_die(cfg, cfg_path)

        if not verify:
            proposed, description = proposals[0]
            save_choice(proposed, f"{description} selected without A/B verification")
            console.print(f"[green]Saved speculation recipe for {target.name}.[/green]")
            return

        url = server_url_from(ctx, None)
        headers = (
            {"Authorization": f"Bearer {cfg.server.admin_token}"} if cfg.server.admin_token else {}
        )
        original = dict(recipe)
        restore = {
            "spec_type": original.get("spec_type"),
            "spec_draft_name": original.get("spec_draft_name"),
            "spec_draft_n_max": original.get("spec_draft_n_max"),
            "speculation_result": original.get("speculation_result"),
        }
        target_only = {
            "spec_type": None,
            "spec_draft_name": None,
            "spec_draft_n_max": None,
            "speculation_result": None,
        }

        async def _measure() -> Any:
            return await benchmark_mod.benchmark_model(
                url,
                target.name,
                prompt_tokens=cfg.tune.prompt_tokens,
                gen_tokens=cfg.tune.gen_tokens,
                cfg=cfg,
            )

        async def _verify() -> tuple[int | None, list[tuple[str, float | None, str | None]]]:
            """Return (index of the winning proposal or None, per-proposal outcomes)."""
            outcomes: list[tuple[str, float | None, str | None]] = []
            winner: int | None = None
            changed = False
            async with httpx.AsyncClient(base_url=url, timeout=600.0, headers=headers) as client:
                try:
                    error = await _apply_edits(client, target.name, target_only)
                    if error:
                        return None, [("target-only", None, error)]
                    changed = True
                    baseline = await _measure()
                    if baseline.error:
                        return None, [("target-only", None, f"benchmark failed: {baseline.error}")]
                    best_gain: float | None = None
                    for index, (proposed, description) in enumerate(proposals):
                        error = await _apply_edits(client, target.name, proposed)
                        if error:
                            outcomes.append((description, None, error))
                            continue
                        result = await _measure()
                        if result.error:
                            outcomes.append((description, None, f"benchmark failed: {result.error}"))
                            continue
                        gain = benchmark_improvement(baseline, result, target="generation")
                        outcomes.append((description, gain, None if gain is not None else "unscored"))
                        if gain is not None and gain >= min_speedup and (
                            best_gain is None or gain > best_gain
                        ):
                            winner, best_gain = index, gain
                    if winner is not None and winner != len(proposals) - 1:
                        # The server holds the last proposal tried; switch to the winner.
                        error = await _apply_edits(client, target.name, proposals[winner][0])
                        if error:
                            raise RuntimeError(f"could not apply the winning speculation: {error}")
                    return winner, outcomes
                finally:
                    if changed and winner is None:
                        restore_error = await _restore_final_state(
                            client, target.name, restore, cfg=None
                        )
                        if restore_error:
                            raise RuntimeError(
                                f"speculation rejected but rollback failed: {restore_error}"
                            )

        try:
            winner, outcomes = asyncio.run(_verify())
        except KeyboardInterrupt:
            raise click.ClickException("speculation verification interrupted") from None
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc
        for description, gain, problem in outcomes:
            shown = f"{gain:+.1%}" if gain is not None else problem
            console.print(f"  {description:<28} {shown}")
        if winner is None:
            best = max((g for _, g, _ in outcomes if g is not None), default=None)
            detail = (
                f"best generation change {best:+.1%}; required {min_speedup:.1%}"
                if best is not None
                else (outcomes[0][2] if outcomes else "no proposal could be measured")
            )
            raise click.ClickException(f"{detail}; original speculation recipe restored")
        proposed, description = proposals[winner]
        gain = next(g for d, g, _ in outcomes if d == description)
        assert gain is not None
        save_choice(proposed, f"{description} verified at {gain:.1%} generation speedup")
        console.print(
            f"[green]Saved verified speculation for {target.name}: {description}, "
            f"{gain:.1%} faster generation.[/green]"
        )



    return {
        "benchmark_cmd": benchmark_cmd,
        "tune_cmd": tune_cmd,
        "speculative_cmd": speculative_cmd,
    }
