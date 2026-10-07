"""CLI registration for community tune-recipe commands."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console


def register_recipe_commands(
    cli: click.Group,
    *,
    console: Console,
    benchmark_mod: Any,
    httpx: Any,
    load_config: Any,
    server_url_from: Any,
) -> None:
    """Register community recipe commands on the main CLI group."""
    # ===========================================================================
    # recipes (community registry)
    # ===========================================================================


    @cli.group("recipes")
    def recipes_group() -> None:
        """Community tune-recipe registry (lookup, update, validate)."""


    @recipes_group.command("lookup")
    @click.argument("model")
    @click.pass_context
    def recipes_lookup(ctx: click.Context, model: str) -> None:
        """Show the community recipe registered for MODEL's fingerprint, if any."""
        from arc_llama import workload as workload_mod
        from arc_llama.recipe_share import RecipeRegistry, share_fingerprint

        cfg = load_config(ctx.obj["config_path"])
        m = cfg.find_model(model)
        if m is None:
            console.print(f"[red]Model '{model}' is not registered.[/red]")
            sys.exit(1)
        gpu = cfg.find_gpu(m.gpu_pci_slot)
        fp = share_fingerprint(
            gpu_arch=(gpu.arch if gpu else "unknown"),
            backend=(gpu.backend if gpu else "sycl"),
            model_class=(m.kv_class or "default"),
            workload_key=workload_mod.fingerprint_key(cfg.workload),
            tune_schema_version=3,
            vram_mb=(gpu.vram_mb if gpu is not None and gpu.vram_mb is not None else 0),
        )
        entry = RecipeRegistry().lookup(fp)
        if entry is None:
            console.print(
                f"[yellow]No community recipe for {fp[:16]}… — run `arc-llama tune` to measure one.[/yellow]"
            )
            sys.exit(1)
        console.print(f"[bold]{entry.submits} measurement(s)[/bold] for {fp[:16]}…")
        console.print(f"  confidence: {entry.confidence_score:.0%}")
        if entry.gpu_name:
            console.print(f"  gpu: {entry.gpu_name}")
        if entry.provenance:
            build = entry.provenance.get("llama_server_git") or entry.provenance.get(
                "llama_server_version", "unknown"
            )
            console.print(
                f"  provenance: {entry.provenance.get('llama_server_backend', '?')} / {build}"
            )
        if entry.prompt_eval_tok_s or entry.generation_tok_s:
            console.print(
                f"  measured: {entry.prompt_eval_tok_s or '?'} pp tok/s · "
                f"{entry.generation_tok_s or '?'} gen tok/s"
            )
        console.print(f"  recipe: [dim]{json.dumps(entry.edits, sort_keys=True)}[/dim]")


    @recipes_group.command("apply")
    @click.argument("model")
    @click.option(
        "--server",
        "server_url",
        default=None,
        help="Base URL of a running arc-llama server.",
    )
    @click.option(
        "--verify/--no-verify",
        default=True,
        help="A/B benchmark the current and shared recipes, rolling back unless the shared recipe wins.",
    )
    @click.option(
        "--min-improvement",
        type=click.FloatRange(min=0.0),
        default=0.01,
        show_default=True,
        help="Minimum fractional A/B score improvement required to keep the recipe.",
    )
    @click.option(
        "--min-confidence",
        type=click.FloatRange(min=0.0, max=1.0),
        default=0.5,
        show_default=True,
    )
    @click.option(
        "--allow-unverified",
        is_flag=True,
        help="Allow missing or mismatched llama-server provenance; A/B verification is still recommended.",
    )
    @click.option(
        "--dry-run", is_flag=True, help="Show the decision and edits without changing anything."
    )
    @click.pass_context
    def recipes_apply(
        ctx: click.Context,
        model: str,
        server_url: str | None,
        verify: bool,
        min_improvement: float,
        min_confidence: float,
        allow_unverified: bool,
        dry_run: bool,
    ) -> None:
        """Safely apply a community recipe, optionally proving it locally first."""
        from arc_llama import workload as workload_mod
        from arc_llama.recipe_share import (
            RecipeRegistry,
            benchmark_improvement,
            llama_server_build_identity,
            provenance_matches_local,
            share_fingerprint,
            shared_recipe_edits_to_model_recipe,
        )
        from arc_llama.tune import _apply_edits, _restore_edits, _restore_final_state

        cfg = load_config(ctx.obj["config_path"])
        m = cfg.find_model(model)
        if m is None:
            raise click.ClickException(f"model {model!r} is not registered")
        gpu = cfg.find_gpu(m.gpu_pci_slot)
        fp = share_fingerprint(
            gpu_arch=(gpu.arch if gpu else "unknown"),
            backend=(gpu.backend if gpu else "sycl"),
            model_class=(m.kv_class or "default"),
            workload_key=workload_mod.fingerprint_key(cfg.workload),
            tune_schema_version=3,
            vram_mb=(gpu.vram_mb if gpu and gpu.vram_mb else 0),
        )
        entry = RecipeRegistry().lookup(fp)
        if entry is None:
            raise click.ClickException(f"no community recipe for {fp[:16]}…")
        if entry.confidence_score < min_confidence:
            raise click.ClickException(
                f"recipe confidence {entry.confidence_score:.0%} is below "
                f"the required {min_confidence:.0%}"
            )

        local = llama_server_build_identity(cfg.paths.llama_server)
        provenance_ok = provenance_matches_local(
            entry.provenance,
            llama_server_version=local.get("llama_server_version"),
            llama_server_git=local.get("llama_server_git"),
            llama_server_backend=local.get("llama_server_backend"),
        )
        if not provenance_ok and not allow_unverified:
            raise click.ClickException(
                "shared recipe provenance does not match this llama-server build; "
                "use --allow-unverified to rely on local A/B verification"
            )

        edits = shared_recipe_edits_to_model_recipe(entry.edits)
        console.print(
            f"[bold]Community recipe[/bold] {fp[:16]}… "
            f"(confidence {entry.confidence_score:.0%}, "
            f"provenance {'matched' if provenance_ok else 'unverified'})"
        )
        console.print(f"  edits: [dim]{json.dumps(edits, sort_keys=True)}[/dim]")
        if dry_run:
            return

        url = server_url_from(ctx, server_url)
        headers = (
            {"Authorization": f"Bearer {cfg.server.admin_token}"} if cfg.server.admin_token else {}
        )

        async def _run() -> tuple[bool, float | None, str | None]:
            touched = set(edits)
            restore = _restore_edits(dict(m.recipe or {}), touched)
            accepted = False
            candidate_applied = False
            failure: str | None = None
            gain: float | None = None
            baseline = None
            if verify:
                baseline = await benchmark_mod.benchmark_model(
                    url,
                    model,
                    prompt_tokens=cfg.tune.prompt_tokens,
                    gen_tokens=cfg.tune.gen_tokens,
                    cfg=cfg,
                )
                if baseline.error:
                    return False, None, f"baseline benchmark failed: {baseline.error}"

            async with httpx.AsyncClient(base_url=url, timeout=600.0, headers=headers) as client:
                try:
                    failure = await _apply_edits(client, model, edits)
                    if failure:
                        return False, None, failure
                    candidate_applied = True
                    if not verify:
                        accepted = True
                        return True, None, None
                    candidate = await benchmark_mod.benchmark_model(
                        url,
                        model,
                        prompt_tokens=cfg.tune.prompt_tokens,
                        gen_tokens=cfg.tune.gen_tokens,
                        cfg=cfg,
                    )
                    if candidate.error:
                        failure = f"candidate benchmark failed: {candidate.error}"
                        return False, None, failure
                    gain = benchmark_improvement(
                        baseline,
                        candidate,
                        target=workload_mod.tune_target(cfg),
                        priority=workload_mod.score_priority(cfg),
                    )
                    if gain is None:
                        failure = "could not score the baseline and candidate benchmarks"
                        return False, None, failure
                    accepted = gain >= min_improvement
                    if not accepted:
                        failure = (
                            f"shared recipe improved the workload score by {gain:.1%}; "
                            f"required {min_improvement:.1%}"
                        )
                    return accepted, gain, failure
                finally:
                    if candidate_applied and not accepted:
                        restore_error = await _restore_final_state(client, model, restore, cfg=None)
                        if restore_error:
                            raise RuntimeError(
                                f"recipe was rejected but rollback failed: {restore_error}"
                            )

        try:
            accepted, gain, failure = asyncio.run(_run())
        except KeyboardInterrupt:
            raise click.ClickException("recipe verification interrupted") from None
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc
        if not accepted:
            raise click.ClickException(f"{failure}; original recipe restored")
        if gain is None:
            console.print(f"[green]Applied shared recipe to {model}.[/green]")
        else:
            console.print(
                f"[green]Applied shared recipe to {model}; local A/B score improved {gain:.1%}.[/green]"
            )


    @recipes_group.command("update")
    @click.option(
        "--url",
        default=None,
        help="Fetch the registry from this URL instead of the default release asset.",
    )
    @click.pass_context
    def recipes_update(ctx: click.Context, url: str | None) -> None:
        """Refresh the local registry from the community release asset."""
        import httpx as _httpx

        from arc_llama.recipe_share import (
            DEFAULT_REGISTRY_URL,
            MAX_REGISTRY_BYTES,
            RegistryValidationError,
            _user_override_path,
            parse_registry_bytes,
            write_registry_atomic,
        )

        src = url or DEFAULT_REGISTRY_URL
        dest = _user_override_path()
        console.print(f"Fetching {src} …")
        try:
            payload = bytearray()
            with _httpx.stream("GET", src, follow_redirects=True, timeout=30) as resp:
                resp.raise_for_status()
                content_length = resp.headers.get("content-length")
                if content_length is not None and int(content_length) > MAX_REGISTRY_BYTES:
                    raise RegistryValidationError("downloaded registry exceeds the 16 MiB limit")
                for chunk in resp.iter_bytes():
                    payload.extend(chunk)
                    if len(payload) > MAX_REGISTRY_BYTES:
                        raise RegistryValidationError("downloaded registry exceeds the 16 MiB limit")
            doc = parse_registry_bytes(bytes(payload))
            write_registry_atomic(doc, dest)
        except (OSError, ValueError, _httpx.HTTPError, RegistryValidationError) as e:
            console.print(f"[red]Download failed: {e}[/red]")
            sys.exit(1)
        n = len(doc["recipes"])
        console.print(f"[green]Saved {n} recipe(s) to {dest}[/green]")


    @recipes_group.command("validate")
    @click.argument("path", type=click.Path(exists=True, path_type=Path))
    def recipes_validate(path: Path) -> None:
        """Validate a submission JSON file (same checks registry CI runs)."""
        from arc_llama.recipe_share import validate_submission

        problems = validate_submission(json.loads(path.read_text()))
        if problems:
            for p in problems:
                console.print(f"[red]- {p}[/red]")
            sys.exit(1)
        console.print("[green]OK[/green]")
