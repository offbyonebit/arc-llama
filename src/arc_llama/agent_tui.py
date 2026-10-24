"""arcllama — an interactive terminal agent UI.

Run anywhere and the agent opens for the current directory:

    arcllama

Requires a running `arc-llama serve` instance and the `[tui]` extra:

    pip install 'arc-llama[tui]'
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import httpx

try:
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.message import Message
    from textual.screen import ModalScreen
    from textual.widgets import Button, Checkbox, Input, Label, RichLog, Static
except ImportError as e:  # pragma: no cover
    raise SystemExit("arcllama requires textual. Install with: pip install 'arc-llama[tui]'") from e

from arc_llama.agent.checkpoints import CheckpointStore
from arc_llama.agent.interactive import InteractiveAgent
from arc_llama.agent.mcp_client import MCPClientManager
from arc_llama.chat_store import ChatMessage, ChatStore
from arc_llama.config import Config
from arc_llama.skills import load_skills


class AgentEvent(Message):
    """Message carrying one agent loop event."""

    def __init__(self, event: dict[str, Any]) -> None:
        super().__init__()
        self.event = event


class ConfirmToolScreen(ModalScreen[bool]):
    """Modal asking the user to approve a destructive tool call."""

    DEFAULT_CSS = """
    ConfirmToolScreen {
        align: center middle;
    }
    ConfirmToolScreen > Vertical {
        background: $surface;
        border: solid $primary;
        padding: 1 2;
        width: 80;
        height: auto;
    }
    ConfirmToolScreen Label.title {
        text-style: bold;
        margin-bottom: 1;
    }
    ConfirmToolScreen Static.args {
        color: $text-muted;
        margin: 1 0;
    }
    ConfirmToolScreen Horizontal { height: auto; margin-top: 1; }
    ConfirmToolScreen Button { margin-right: 1; }
    """

    def __init__(self, tool: str, arguments: dict[str, Any]) -> None:
        super().__init__()
        self.tool = tool
        self.arguments = arguments

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"Allow tool call: {self.tool}", classes="title")
            yield Static(
                json.dumps(self.arguments, indent=2, ensure_ascii=False),
                classes="args",
            )
            with Horizontal():
                yield Button("Approve", id="approve", variant="primary")
                yield Button("Deny", id="deny")

    @on(Button.Pressed, "#approve")
    def approve(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#deny")
    def deny(self) -> None:
        self.dismiss(False)


class ApprovePlanScreen(ModalScreen[bool]):
    """Modal asking the user to approve a plan."""

    DEFAULT_CSS = """
    ApprovePlanScreen {
        align: center middle;
    }
    ApprovePlanScreen > Vertical {
        background: $surface;
        border: solid $primary;
        padding: 1 2;
        width: 90;
        height: auto;
    }
    ApprovePlanScreen Label.title {
        text-style: bold;
        margin-bottom: 1;
    }
    ApprovePlanScreen Static.plan {
        margin: 1 0;
    }
    ApprovePlanScreen Horizontal { height: auto; margin-top: 1; }
    ApprovePlanScreen Button { margin-right: 1; }
    """

    def __init__(self, plan_text: str) -> None:
        super().__init__()
        self.plan_text = plan_text

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Approve plan", classes="title")
            yield Static(self.plan_text, classes="plan")
            with Horizontal():
                yield Button("Approve", id="approve", variant="primary")
                yield Button("Deny", id="deny")

    @on(Button.Pressed, "#approve")
    def approve(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#deny")
    def deny(self) -> None:
        self.dismiss(False)


class AgentApp(App):
    """Textual app for the arcllama interactive agent."""

    CSS = """
    #header {
        height: auto;
        padding: 0 1;
        background: $surface-darken-1;
    }
    #header Static {
        padding: 0 2 0 0;
        color: $text-muted;
    }
    #chat-log {
        border: solid $primary-darken-2;
        padding: 0 1;
    }
    #input-row {
        height: auto;
        padding: 0 1 1 1;
    }
    #chat-input {
        width: 1fr;
    }
    #send-btn {
        width: auto;
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+n", "new_session", "New session", show=True),
    ]

    def __init__(
        self,
        base_url: str,
        model: str,
        root: Path,
        folder: str,
        chat_store: ChatStore,
        checkpoint_store: CheckpointStore,
    ) -> None:
        super().__init__()
        self.base_url = base_url
        self.model = model
        self.root = root
        self.folder = folder
        self.chat_store = chat_store
        self.checkpoint_store = checkpoint_store
        self.session_chat = chat_store.create(str(uuid.uuid4()), "TUI session", folder=folder)
        self.agent = InteractiveAgent(
            model=model,
            base_url=base_url,
            root=root,
            chat_store=chat_store,
            checkpoint_store=checkpoint_store,
            run_id=str(uuid.uuid4()),
        )
        self.chat_log: RichLog | None = None
        self.chat_input: Input | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(id="header"):
                yield Static(f"Model: {self.model}", id="model-label")
                yield Static(f"Root: {self.root}", id="root-label")
                yield Static(f"Folder: {self.folder or 'default'}", id="folder-label")
                yield Checkbox("Auto-confirm", id="auto-confirm", value=False)
                yield Checkbox("Plan mode", id="plan-mode", value=False)
            yield RichLog(id="chat-log", wrap=True, highlight=True)
            with Horizontal(id="input-row"):
                yield Input(placeholder="Type a task and press Enter...", id="chat-input")
                yield Button("Send", id="send-btn", variant="primary")

    def on_mount(self) -> None:
        self.chat_log = self.query_one("#chat-log", RichLog)
        self.chat_input = self.query_one("#chat-input", Input)
        self.chat_input.focus()
        self._write(
            "[b]arcllama[/b] — type a task below. Press Ctrl+C to quit, Ctrl+N for a new session."
        )
        self._write(
            "[yellow]Draft interface:[/yellow] still working on ergonomics and style — subject to change."
        )

    def _write(self, text: str) -> None:
        if self.chat_log is not None:
            self.chat_log.write(text)

    @on(Input.Submitted, "#chat-input")
    def on_input_submitted(self) -> None:
        self._submit()

    @on(Button.Pressed, "#send-btn")
    def on_send_pressed(self) -> None:
        self._submit()

    def _submit(self) -> None:
        if self.chat_input is None:
            return
        text = self.chat_input.value.strip()
        if not text:
            return
        self.chat_input.value = ""
        self._write(f"[b]You:[/b] {text}")
        self.run_agent_turn(text)

    @on(Checkbox.Changed, "#auto-confirm")
    def on_auto_confirm_changed(self, event: Checkbox.Changed) -> None:
        self.agent.auto_confirm = event.value
        self._write(f"[dim]auto_confirm = {event.value}[/dim]")

    @on(Checkbox.Changed, "#plan-mode")
    def on_plan_mode_changed(self, event: Checkbox.Changed) -> None:
        self.agent.plan_mode = event.value
        self._write(f"[dim]plan_mode = {event.value}[/dim]")

    @on(AgentEvent)
    def on_agent_event(self, message: AgentEvent) -> None:
        event = message.event
        t = event.get("type")
        if t == "status":
            self._write(f"[dim]# {event.get('message', '')}[/dim]")
        elif t == "plan":
            self._write("[b cyan]Plan[/b cyan]")
            self._write(event.get("content", ""))
        elif t == "assistant":
            content = event.get("content", "")
            if content:
                self._write(f"[b]Agent:[/b] {content}")
        elif t == "tool_call":
            name = event.get("name", "tool")
            args = event.get("arguments", {})
            self._write(f"[b yellow]▶ {name}[/b yellow]")
            self._write(f"[dim]{json.dumps(args, indent=2, ensure_ascii=False)}[/dim]")
        elif t == "tool_result":
            name = event.get("name", "tool")
            if event.get("error"):
                self._write(f"[b red]✗ {name} failed[/b red]")
            else:
                self._write(f"[b green]✓ {name} done[/b green]")
            self._write(f"[dim]{event.get('content', '')}[/dim]")
        elif t == "checkpoint":
            self._write(f"[dim]Checkpoint saved: {event.get('id', '')}[/dim]")
        elif t == "error":
            self._write(f"[b red]Error: {event.get('message', '')}[/b red]")
        elif t == "done":
            self._write("[dim]Done.[/dim]")

    @work(group="agent", exclusive=True)
    async def run_agent_turn(self, text: str) -> None:
        async def confirm_callback(call_id: str, tool: str, arguments: dict[str, Any]) -> bool:
            return await self.push_screen_wait(ConfirmToolScreen(tool, arguments))

        async def plan_callback(plan_text: str) -> bool:
            return await self.push_screen_wait(ApprovePlanScreen(plan_text))

        turn_messages: list[ChatMessage] = [ChatMessage(role="user", content=text)]
        try:
            async for event in self.agent.chat(
                text,
                confirm_callback=confirm_callback,
                plan_callback=plan_callback,
            ):
                self.post_message(AgentEvent(event))
                if event.get("type") == "assistant" and event.get("content"):
                    turn_messages.append(ChatMessage(role="assistant", content=event["content"]))
                elif event.get("type") == "tool_result":
                    name = event.get("name", "tool")
                    content = event.get("content", "")
                    turn_messages.append(ChatMessage(role="tool", content=f"{name}:\n{content}"))
        except Exception as e:
            self.post_message(AgentEvent({"type": "error", "message": str(e)}))
        finally:
            chat = self.chat_store.get(self.session_chat.id)
            if chat is not None:
                chat.messages.extend(turn_messages)
                self.chat_store.save(chat)

    def action_new_session(self) -> None:
        self.session_chat = self.chat_store.create(
            str(uuid.uuid4()), "TUI session", folder=self.folder
        )
        self.agent = InteractiveAgent(
            model=self.model,
            base_url=self.base_url,
            root=self.root,
            auto_confirm=self.agent.auto_confirm,
            plan_mode=self.agent.plan_mode,
            chat_store=self.chat_store,
            checkpoint_store=self.checkpoint_store,
            run_id=str(uuid.uuid4()),
        )
        if self.chat_log is not None:
            self.chat_log.clear()
        self._write("[b]New session started.[/b]")


def run_agent_tui(
    base_url: str | None = None,
    model: str | None = None,
    root: Path | str | None = None,
    folder: str = "",
    profile: str | None = None,
    config: Config | None = None,
) -> None:
    """Launch the arcllama agent TUI."""

    async def _main() -> None:
        cfg = config
        if cfg is None:
            from arc_llama.config import load_config

            cfg = load_config()

        nonlocal base_url, model
        if base_url is None:
            base_url = f"http://{cfg.server.host}:{cfg.server.port}"

        try:
            health = httpx.get(f"{base_url.rstrip('/')}/health", timeout=5.0)
            health.raise_for_status()
        except Exception as e:
            raise SystemExit(
                f"Cannot reach arc-llama server at {base_url}: {e}\nStart one with: arc-llama serve"
            ) from e

        if model is None:
            try:
                resp = httpx.get(f"{base_url.rstrip('/')}/v1/models", timeout=5.0)
                resp.raise_for_status()
                models = resp.json().get("data", [])
                if not models:
                    raise SystemExit("No models available on the server.")
                model = str(models[0].get("id", ""))
            except Exception as e:
                raise SystemExit(f"Could not auto-detect a model: {e}") from e

        root_path = Path(root).expanduser().resolve() if root else Path.cwd()
        state_dir = None
        if cfg.paths.state_dir:
            state_dir = Path(cfg.paths.state_dir).expanduser()

        chat_store = ChatStore(state_dir / "chats" if state_dir else Path(".arc_llama_chats"))
        checkpoint_store = CheckpointStore(
            state_dir / "checkpoints" if state_dir else Path(".arc_llama_checkpoints")
        )

        load_skills(cfg.paths.skills_dir)
        manager = MCPClientManager(cfg.active_mcp_servers(profile))
        await manager.start()
        try:
            app = AgentApp(
                base_url=base_url,
                model=model,
                root=root_path,
                folder=folder,
                chat_store=chat_store,
                checkpoint_store=checkpoint_store,
            )
            await app.run_async()
        finally:
            await manager.stop()

    asyncio.run(_main())
