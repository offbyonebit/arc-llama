"""arc-llama TUI — a Textual app that drives a running `arc-llama serve`.

Run alongside the server (or against a remote one over the network) to monitor
GPUs, models, and load state, and to load/stop models without leaving the
terminal.

    arc-llama tui                          # against http://127.0.0.1:11437
    arc-llama tui --server http://10.0.0.5:11437

Color choices avoid red/green — status is signalled by brightness/dim, not hue.
"""
from __future__ import annotations

from typing import Any

import httpx

try:
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.screen import ModalScreen
    from textual.widgets import Button, DataTable, Footer, Input, Label, Select, Static
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "arc-llama TUI requires textual. Install with: pip install 'arc-llama[tui]'"
    ) from e


KV_OPTIONS = ["f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"]


REFRESH_SECONDS = 2.5


class EditRecipeScreen(ModalScreen[dict | None]):
    """Modal that edits a model's ctx + K/V quant, returns a dict on Save."""

    DEFAULT_CSS = """
    EditRecipeScreen {
        align: center middle;
    }
    EditRecipeScreen > Vertical {
        background: #161b22;
        border: solid #58a6ff;
        padding: 1 2;
        width: 60;
        height: auto;
    }
    EditRecipeScreen Label.title {
        text-style: bold;
        margin-bottom: 1;
    }
    EditRecipeScreen Label.field {
        margin-top: 1;
        color: #8b949e;
    }
    EditRecipeScreen Input { width: 100%; }
    EditRecipeScreen Select { width: 100%; }
    EditRecipeScreen .row { height: auto; margin-top: 1; }
    EditRecipeScreen Horizontal.buttons { margin-top: 2; height: 3; }
    EditRecipeScreen Button { margin-right: 1; }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, model: dict) -> None:
        super().__init__()
        self.model = model

    def compose(self) -> ComposeResult:
        m = self.model
        with Vertical():
            yield Label(f"Edit recipe — {m['name']}", classes="title")
            yield Label("Context length (tokens)", classes="field")
            yield Input(value=str(m.get("ctx") or 8192), id="ctx", type="integer")
            yield Label("KV cache type — keys", classes="field")
            yield Select(
                [(o, o) for o in KV_OPTIONS],
                value=m.get("cache_type_k") or "f16",
                id="kv-k",
                allow_blank=False,
            )
            yield Label("KV cache type — values", classes="field")
            yield Select(
                [(o, o) for o in KV_OPTIONS],
                value=m.get("cache_type_v") or "f16",
                id="kv-v",
                allow_blank=False,
            )
            with Horizontal(classes="buttons"):
                yield Button("Save", id="save", variant="primary")
                yield Button("Cancel", id="cancel")

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#save")
    def _on_save(self) -> None:
        try:
            ctx = int(self.query_one("#ctx", Input).value or "0")
        except ValueError:
            ctx = 0
        if ctx < 256:
            self.query_one("#ctx", Input).focus()
            return
        self.dismiss({
            "ctx": ctx,
            "cache_type_k": self.query_one("#kv-k", Select).value,
            "cache_type_v": self.query_one("#kv-v", Select).value,
        })

    def action_cancel(self) -> None:
        self.dismiss(None)


class StatusBar(Static):
    """Top status line — server URL, policy, last-refreshed."""

    def update_status(self, server: dict | None, last: str) -> None:
        if server is None:
            self.update(f"[b]arc-llama[/b]   [dim]disconnected[/dim]   {last}")
            return
        host = f"{server.get('host')}:{server.get('port')}"
        policy = "single-resident" if server.get("single_resident") else "multi-resident"
        self.update(f"[b]arc-llama[/b]   {host}   [dim]·[/dim]   {policy}   [dim]·[/dim]   {last}")


class ArcLlamaTUI(App):
    """Top-level Textual app."""

    CSS = """
    Screen { background: #0e1116; }
    StatusBar {
        height: 1;
        padding: 0 1;
        background: #161b22;
    }
    .panel-title {
        height: 1;
        padding: 0 1;
        color: #8b949e;
        text-style: bold;
    }
    DataTable { border: solid #30363d; }
    DataTable > .datatable--header {
        background: #161b22;
        color: #8b949e;
        text-style: bold;
    }
    DataTable > .datatable--cursor { background: #30363d; }
    """

    BINDINGS = [
        Binding("r", "refresh", "Refresh", show=True),
        Binding("d", "scan", "Discover", show=True),
        Binding("l", "load_selected", "Load", show=True),
        Binding("s", "stop_selected", "Stop", show=True),
        Binding("S", "stop_all", "Stop all", show=True),
        Binding("e", "edit_selected", "Edit", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(self, server_url: str) -> None:
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._last_status: dict[str, Any] | None = None

    def compose(self) -> ComposeResult:
        yield StatusBar(id="status-bar")
        with Vertical():
            yield Static("GPUs", classes="panel-title")
            yield DataTable(id="gpus", zebra_stripes=False, cursor_type="row")
            yield Static("Models", classes="panel-title")
            yield DataTable(id="models", zebra_stripes=False, cursor_type="row")
        yield Footer()

    async def on_mount(self) -> None:
        self._client = httpx.AsyncClient(base_url=self.server_url, timeout=10.0)
        gpus = self.query_one("#gpus", DataTable)
        gpus.add_columns("PCI", "Arch", "Name", "SYCL", "VRAM", "Enabled")
        models = self.query_one("#models", DataTable)
        models.add_columns("Status", "Name", "GPU", "Port", "ctx", "K/V", "Path")
        await self._refresh()
        self.set_interval(REFRESH_SECONDS, self._refresh)

    async def on_unmount(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    async def _refresh(self) -> None:
        bar = self.query_one("#status-bar", StatusBar)
        if self._client is None:
            bar.update_status(None, "no client")
            return
        try:
            r = await self._client.get("/admin/status")
            r.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            bar.update_status(self._last_status.get("server") if self._last_status else None,
                              f"[#d29922]error: {type(e).__name__}[/]")
            return
        s = r.json()
        self._last_status = s
        bar.update_status(s.get("server"), self._fmt_now())
        self._render(s)

    def _fmt_now(self) -> str:
        from datetime import datetime
        return f"updated {datetime.now().strftime('%H:%M:%S')}"

    def _render(self, s: dict[str, Any]) -> None:
        gpus = self.query_one("#gpus", DataTable)
        gpus.clear()
        for g in s.get("gpus", []):
            vram = "?" if g.get("vram_mb") is None else f"{g['vram_mb']/1024:.1f} GB"
            gpus.add_row(
                g.get("pci_slot", "?"),
                g.get("arch", "?"),
                g.get("name") or "—",
                f"level_zero:{g.get('sycl_index')}",
                vram,
                "yes" if g.get("enabled") else "no",
            )

        models = self.query_one("#models", DataTable)
        # Preserve cursor position if possible.
        prev_cursor = models.cursor_row
        models.clear()
        for m in s.get("models", []):
            loaded = bool(m.get("loaded"))
            status = "[b]LOADED[/]" if loaded else "[dim]idle[/]"
            kv = f"{m.get('cache_type_k') or '?'}/{m.get('cache_type_v') or '?'}"
            path = m.get("path") or "—"
            short = "/".join(p for p in path.split("/") if p)[-50:]
            row_text = (status, m["name"], m.get("gpu_pci_slot", "?"),
                        str(m.get("port") or "?"), str(m.get("ctx") or "?"), kv, short)
            if loaded:
                models.add_row(*row_text)
            else:
                models.add_row(*[f"[dim]{c}[/]" for c in row_text])
        if prev_cursor is not None and models.row_count > 0:
            try:
                models.move_cursor(row=min(prev_cursor, models.row_count - 1))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _selected_model_name(self) -> str | None:
        if self._last_status is None:
            return None
        models = self.query_one("#models", DataTable)
        if models.cursor_row is None or models.cursor_row < 0:
            return None
        try:
            return self._last_status["models"][models.cursor_row]["name"]
        except (IndexError, KeyError):
            return None

    async def action_refresh(self) -> None:
        await self._refresh()

    @work(exclusive=True)
    async def action_scan(self) -> None:
        if self._client is None:
            return
        bar = self.query_one("#status-bar", StatusBar)
        bar.update_status(self._last_status.get("server") if self._last_status else None,
                          "scanning…")
        try:
            r = await self._client.post("/admin/scan", timeout=30.0)
            r.raise_for_status()
            j = r.json()
            added = j.get("added") or []
            msg = (f"scanned {j.get('found', 0)} GGUF(s); "
                   f"registered {len(added)} new"
                   + (f": {', '.join(added)}" if added else ""))
            bar.update_status(self._last_status.get("server") if self._last_status else None, msg)
        except httpx.HTTPError as e:
            bar.update_status(self._last_status.get("server") if self._last_status else None,
                              f"[#d29922]scan failed: {e}[/]")
            return
        await self._refresh()

    @work(exclusive=True)
    async def action_load_selected(self) -> None:
        name = self._selected_model_name()
        if not name or self._client is None:
            return
        bar = self.query_one("#status-bar", StatusBar)
        bar.update_status(self._last_status.get("server") if self._last_status else None,
                          f"loading {name}…")
        try:
            r = await self._client.post(f"/admin/load/{name}", timeout=180.0)
            r.raise_for_status()
        except httpx.HTTPError as e:
            bar.update_status(self._last_status.get("server") if self._last_status else None,
                              f"[#d29922]load {name} failed: {e}[/]")
            return
        await self._refresh()

    @work(exclusive=True)
    async def action_stop_selected(self) -> None:
        name = self._selected_model_name()
        if not name or self._client is None:
            return
        try:
            r = await self._client.post(f"/admin/stop/{name}")
            r.raise_for_status()
        except httpx.HTTPError:
            pass
        await self._refresh()

    @work(exclusive=True)
    async def action_stop_all(self) -> None:
        if self._client is None:
            return
        try:
            r = await self._client.post("/admin/stop-all")
            r.raise_for_status()
        except httpx.HTTPError:
            pass
        await self._refresh()

    def action_edit_selected(self) -> None:
        """Pop the EditRecipeScreen for the currently selected model."""
        if self._last_status is None or self._client is None:
            return
        models = self.query_one("#models", DataTable)
        if models.cursor_row is None or models.cursor_row < 0:
            return
        try:
            model = self._last_status["models"][models.cursor_row]
        except (IndexError, KeyError):
            return

        def _after(result: dict | None) -> None:
            if result is None:
                return
            self._submit_edit(model["name"], result, was_loaded=bool(model.get("loaded")))

        self.push_screen(EditRecipeScreen(model), _after)

    @work(exclusive=True)
    async def _submit_edit(self, name: str, payload: dict, was_loaded: bool) -> None:
        if self._client is None:
            return
        bar = self.query_one("#status-bar", StatusBar)
        srv_info = self._last_status.get("server") if self._last_status else None
        try:
            r = await self._client.post(
                f"/admin/models/{name}/edit", json=payload, timeout=15.0,
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            bar.update_status(srv_info, f"[#d29922]edit {name} failed: {e}[/]")
            return
        msg = f"updated {name}"
        if was_loaded:
            msg += " (was running — stopped)"
        bar.update_status(srv_info, msg)
        await self._refresh()


def run_tui(server_url: str = "http://127.0.0.1:11437") -> None:
    ArcLlamaTUI(server_url).run()
