"""CLI commands: agent, code, agent-tui."""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import click
import httpx

from arc_llama.agent import run_agent
from arc_llama.agent.checkpoints import CheckpointStore
from arc_llama.agent.interactive import InteractiveAgent
from arc_llama.agent.mcp_client import MCPClientManager
from arc_llama.chat_store import ChatMessage, ChatStore
from arc_llama.cli_bindings import load_config
from arc_llama.config import Config
from arc_llama.skills import load_skills

from .common import console, state_dir_from_config


@asynccontextmanager
async def _agent_tool_context(cfg: Config, profile: str | None):
    """Load skills and start the active profile's MCP servers for a CLI agent run."""
    load_skills(cfg.paths.skills_dir)
    manager = MCPClientManager(cfg.active_mcp_servers(profile))
    try:
        await manager.start()
        yield
    finally:
        await manager.stop()


async def _prompt_yes_no(prompt: str) -> bool:
    """Prompt the user for a yes/no answer from an async context."""
    loop = asyncio.get_running_loop()
    while True:
        answer = await loop.run_in_executor(None, input, prompt)
        cleaned = answer.strip().lower()
        if cleaned in ("y", "yes"):
            return True
        if cleaned in ("n", "no"):
            return False
        console.print("[dim]Please answer y or n.[/dim]")


def _render_agent_event(event: dict) -> None:
    t = event.get("type")
    if t == "status":
        console.print(f"[dim]# {event.get('message', '')}[/dim]")
    elif t == "plan":
        console.print("[bold cyan]Proposed plan:[/bold cyan]")
        console.print(event.get("content", ""))
    elif t == "assistant":
        content = event.get("content", "")
        if content:
            console.print(content)
    elif t == "tool_call":
        name = event.get("name", "tool")
        args = event.get("arguments", {})
        console.print(f"[bold yellow]▶ {name}[/bold yellow]")
        console.print(f"[dim]{json.dumps(args, indent=2, ensure_ascii=False)}[/dim]")
    elif t == "tool_result":
        name = event.get("name", "tool")
        content = event.get("content", "")
        if event.get("error"):
            console.print(f"[red]✗ {name} failed[/red]")
        else:
            console.print(f"[green]✓ {name} done[/green]")
        console.print(f"[dim]{content}[/dim]")
    elif t == "confirm_required":
        console.print(f"[yellow]⚠ Confirmation required for {event.get('tool', 'tool')}[/yellow]")
    elif t == "checkpoint":
        console.print(f"[dim]Checkpoint saved: {event.get('id', '')}[/dim]")
    elif t == "error":
        console.print(f"[red]Error: {event.get('message', '')}[/red]")
    elif t == "done":
        console.print("[green]Agent finished.[/green]")


def _agent_base_setup(ctx: click.Context, base_url: str | None) -> tuple[Config, str]:
    cfg = load_config(ctx.obj["config_path"])
    if base_url is None:
        base_url = f"http://{cfg.server.host}:{cfg.server.port}"
    try:
        health = httpx.get(f"{base_url.rstrip('/')}/health", timeout=5.0)
        health.raise_for_status()
    except Exception as e:
        console.print(f"[red]Cannot reach arc-llama server at {base_url}: {e}[/red]")
        console.print("[dim]Start one with:[/dim] arc-llama serve")
        sys.exit(1)
    return cfg, base_url


@click.command("agent")
@click.argument("task")
@click.option("--model", "-m", required=True, help="Model id to use.")
@click.option("--root", "-r", default=None, help="Project root (default: agent.root from config).")
@click.option("--auto-confirm", is_flag=True, help="Do not prompt for tool confirmation.")
@click.option("--plan-mode", is_flag=True, help="Generate a plan first and ask for approval.")
@click.option("--max-turns", type=int, default=30, help="Maximum agent turns (default: 30).")
@click.option("--folder", "-f", default="", help="Folder to save the agent transcript chat.")
@click.option(
    "--profile",
    default=None,
    help="MCP profile name (overrides agent.profile in config).",
)
@click.option(
    "--base-url",
    default=None,
    help="arc-llama server base URL (default: http://HOST:PORT from config).",
)
@click.pass_context
def agent_cmd(
    ctx: click.Context,
    task: str,
    model: str,
    root: str | None,
    auto_confirm: bool,
    plan_mode: bool,
    max_turns: int,
    folder: str,
    profile: str | None,
    base_url: str | None,
) -> None:
    """Run the local coding agent from the terminal."""
    cfg, base_url = _agent_base_setup(ctx, base_url)

    root_path = Path(root or cfg.agent.root).expanduser().resolve()
    state_dir = state_dir_from_config(cfg)
    chat_store = ChatStore(
        state_dir / "chats" if state_dir else Path(".arc_llama_chats")
    )
    checkpoint_store = CheckpointStore(
        state_dir / "checkpoints" if state_dir else Path(".arc_llama_checkpoints")
    )

    title = task.strip().split("\n")[0][:80] or "Agent task"
    agent_chat = chat_store.create(str(uuid.uuid4()), title, folder=folder)
    run_id = str(uuid.uuid4())
    transcript: list[ChatMessage] = [ChatMessage(role="user", content=task)]

    async def confirm_callback(call_id: str, tool: str, arguments: dict) -> bool:
        summary = json.dumps(arguments, ensure_ascii=False)[:200]
        return await _prompt_yes_no(f"Allow [bold]{tool}[/bold] {summary}? [y/n] ")

    async def plan_callback(plan_text: str) -> bool:
        return await _prompt_yes_no("Approve plan? [y/n] ")

    async def run() -> None:
        async with _agent_tool_context(cfg, profile):
            async for event in run_agent(
                task=task,
                model=model,
                base_url=base_url,
                root=root_path,
                auto_confirm=auto_confirm,
                confirm_callback=confirm_callback,
                plan_mode=plan_mode,
                plan_callback=plan_callback,
                run_id=run_id,
                checkpoint_store=checkpoint_store,
                max_turns=max_turns,
                chat_store=chat_store,
            ):
                _render_agent_event(event)
                if event.get("type") == "assistant" and event.get("content"):
                    transcript.append(ChatMessage(role="assistant", content=event["content"]))
                elif event.get("type") == "tool_result":
                    name = event.get("name", "tool")
                    content = event.get("content", "")
                    transcript.append(ChatMessage(role="tool", content=f"{name}:\n{content}"))

            chat = chat_store.get(agent_chat.id)
            if chat is not None:
                chat.messages.extend(transcript)
                chat_store.save(chat)
                console.print(f"[dim]Transcript saved: chat {agent_chat.id} in folder '{folder or 'default'}'[/dim]")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("[yellow]Agent run interrupted.[/yellow]")


@click.command("code")
@click.option("--model", "-m", required=True, help="Model id to use.")
@click.option("--root", "-r", default=None, help="Project root (default: agent.root from config).")
@click.option("--auto-confirm", is_flag=True, help="Do not prompt for tool confirmation.")
@click.option("--plan-mode", is_flag=True, help="Generate a plan first and ask for approval each turn.")
@click.option("--max-turns", type=int, default=30, help="Maximum agent turns per user message (default: 30).")
@click.option("--folder", "-f", default="", help="Folder to save the session transcript chat.")
@click.option(
    "--profile",
    default=None,
    help="MCP profile name (overrides agent.profile in config).",
)
@click.option(
    "--base-url",
    default=None,
    help="arc-llama server base URL (default: http://HOST:PORT from config).",
)
@click.pass_context
def code_cmd(
    ctx: click.Context,
    model: str,
    root: str | None,
    auto_confirm: bool,
    plan_mode: bool,
    max_turns: int,
    folder: str,
    profile: str | None,
    base_url: str | None,
) -> None:
    """Start an interactive coding agent REPL."""
    cfg, base_url = _agent_base_setup(ctx, base_url)

    root_path = Path(root or cfg.agent.root).expanduser().resolve()
    state_dir = state_dir_from_config(cfg)
    chat_store = ChatStore(
        state_dir / "chats" if state_dir else Path(".arc_llama_chats")
    )
    checkpoint_store = CheckpointStore(
        state_dir / "checkpoints" if state_dir else Path(".arc_llama_checkpoints")
    )

    session_chat = chat_store.create(str(uuid.uuid4()), "CLI session", folder=folder)
    run_id = str(uuid.uuid4())

    agent = InteractiveAgent(
        model=model,
        base_url=base_url,
        root=root_path,
        auto_confirm=auto_confirm,
        plan_mode=plan_mode,
        max_turns=max_turns,
        chat_store=chat_store,
        checkpoint_store=checkpoint_store,
        run_id=run_id,
    )

    settings = {
        "model": model,
        "root": str(root_path),
        "folder": folder or "default",
        "auto_confirm": auto_confirm,
        "plan_mode": plan_mode,
        "max_turns": max_turns,
    }

    console.print("[bold green]arc-llama code[/bold green] — interactive agent")
    for key, value in settings.items():
        console.print(f"  [dim]{key}:[/dim] {value}")
    console.print("[dim]Type /help for commands, /quit to exit.[/dim]\n")

    async def confirm_callback(call_id: str, tool: str, arguments: dict) -> bool:
        summary = json.dumps(arguments, ensure_ascii=False)[:200]
        return await _prompt_yes_no(f"Allow [bold]{tool}[/bold] {summary}? [y/n] ")

    async def plan_callback(plan_text: str) -> bool:
        return await _prompt_yes_no("Approve plan? [y/n] ")

    async def save_transcript(messages: list[ChatMessage]) -> None:
        if not messages:
            return
        chat = chat_store.get(session_chat.id)
        if chat is None:
            return
        chat.messages.extend(messages)
        chat_store.save(chat)

    async def repl() -> None:
        nonlocal session_chat, agent
        async with _agent_tool_context(cfg, profile):
            loop = asyncio.get_running_loop()
            while True:
                try:
                    user_input = await loop.run_in_executor(None, input, ">>> ")
                except EOFError:
                    console.print("\n[yellow]Exiting.[/yellow]")
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    command = user_input[1:].strip()
                    if command in ("quit", "exit"):
                        console.print("[yellow]Goodbye.[/yellow]")
                        break
                    if command == "help":
                        console.print(
                            "[bold]Commands:[/bold]\n"
                            "  /help              show this message\n"
                            "  /quit, /exit       leave the REPL\n"
                            "  /auto              toggle auto-confirm\n"
                            "  /plan              toggle plan mode\n"
                            "  /model <id>        change model\n"
                            "  /root <path>       change project root\n"
                            "  /folder <name>     move transcript to folder\n"
                            "  /max-turns <n>     change max turns per message\n"
                            "  /clear             start a new session chat"
                        )
                        continue
                    if command == "auto":
                        agent.auto_confirm = not agent.auto_confirm
                        console.print(f"[dim]auto_confirm = {agent.auto_confirm}[/dim]")
                        continue
                    if command == "plan":
                        agent.plan_mode = not agent.plan_mode
                        console.print(f"[dim]plan_mode = {agent.plan_mode}[/dim]")
                        continue
                    if command == "clear":
                        await agent.close()
                        session_chat = chat_store.create(str(uuid.uuid4()), "CLI session", folder=folder)
                        agent = InteractiveAgent(
                            model=agent.model,
                            base_url=base_url,
                            root=agent.root,
                            auto_confirm=agent.auto_confirm,
                            plan_mode=agent.plan_mode,
                            max_turns=agent.max_turns,
                            chat_store=chat_store,
                            checkpoint_store=checkpoint_store,
                            run_id=str(uuid.uuid4()),
                        )
                        console.print("[dim]Started a new session chat.[/dim]")
                        continue
                    if command.startswith("model "):
                        agent.model = command[6:].strip() or agent.model
                        console.print(f"[dim]model = {agent.model}[/dim]")
                        continue
                    if command.startswith("root "):
                        new_root = Path(command[5:].strip()).expanduser().resolve()
                        agent.root = new_root
                        console.print(f"[dim]root = {agent.root}[/dim]")
                        continue
                    if command.startswith("folder "):
                        new_folder = command[7:].strip()
                        chat = chat_store.get(session_chat.id)
                        if chat is not None:
                            chat.folder = new_folder
                            chat_store.save(chat)
                        console.print(f"[dim]folder = {new_folder}[/dim]")
                        continue
                    if command.startswith("max-turns "):
                        try:
                            agent.max_turns = int(command[10:].strip())
                            console.print(f"[dim]max_turns = {agent.max_turns}[/dim]")
                        except ValueError:
                            console.print("[red]max-turns requires an integer[/red]")
                        continue
                    console.print(f"[red]Unknown command: /{command}[/red]")
                    continue

                turn_messages: list[ChatMessage] = [ChatMessage(role="user", content=user_input)]
                async for event in agent.chat(
                    user_input,
                    confirm_callback=confirm_callback,
                    plan_callback=plan_callback,
                ):
                    _render_agent_event(event)
                    if event.get("type") == "assistant" and event.get("content"):
                        turn_messages.append(ChatMessage(role="assistant", content=event["content"]))
                    elif event.get("type") == "tool_result":
                        name = event.get("name", "tool")
                        content = event.get("content", "")
                        turn_messages.append(ChatMessage(role="tool", content=f"{name}:\n{content}"))

                await save_transcript(turn_messages)

            await agent.close()

    try:
        asyncio.run(repl())
    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted.[/yellow]")


@click.command("agent-tui")
@click.option("--model", "-m", default=None, help="Model id to use (default: first available).")
@click.option("--root", "-r", default=None, help="Project root (default: current directory).")
@click.option("--folder", "-f", default="", help="Folder to save the session transcript chat.")
@click.option(
    "--profile",
    default=None,
    help="MCP profile name (overrides agent.profile in config).",
)
@click.option(
    "--base-url",
    default=None,
    help="arc-llama server base URL (default: http://HOST:PORT from config).",
)
@click.pass_context
def agent_tui_cmd(
    ctx: click.Context,
    model: str | None,
    root: str | None,
    folder: str,
    profile: str | None,
    base_url: str | None,
) -> None:
    """Launch the interactive arcllama agent TUI."""
    cfg = load_config(ctx.obj["config_path"])
    try:
        from arc_llama.agent_tui import run_agent_tui

        run_agent_tui(
            base_url=base_url,
            model=model,
            root=root,
            folder=folder,
            profile=profile,
            config=cfg,
        )
    except SystemExit as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


__all__ = [
    "agent_cmd",
    "code_cmd",
    "agent_tui_cmd",
]
