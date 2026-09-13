"""Fast checks for the packaged, dependency-free browser UI."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).parent.parent / "src" / "arc_llama" / "static"


def test_brand_logo_is_referenced_accessibly_on_both_surfaces() -> None:
    for name in ("index.html", "chat.html"):
        html = (STATIC / name).read_text(encoding="utf-8")
        assert 'class="brand-logo"' in html
        assert '/assets/arc-llama-logo.png?v=' in html
        assert 'alt="Arc Llama logo"' in html

    logo = STATIC / "assets" / "arc-llama-logo.png"
    assert logo.is_file()
    assert logo.stat().st_size > 0
    assert (STATIC / "assets" / "arc-llama-logo-dark.png").is_file()


def test_theme_toggle_and_persistent_theme_are_present_on_both_surfaces() -> None:
    for name, script in (("index.html", "app.js"), ("chat.html", "chat.js")):
        html = (STATIC / name).read_text(encoding="utf-8")
        js = (STATIC / script).read_text(encoding="utf-8")
        assert 'id="theme-toggle"' in html
        assert "arc-llama-theme" in js
        assert 'dataset.theme' in js


def test_assistant_bubbles_do_not_use_the_stray_accent_rule() -> None:
    css = (STATIC / "chat.css").read_text(encoding="utf-8")
    assert "border-left: 3px solid var(--accent)" not in css
    assert ".message.assistant" in css


def test_advanced_details_state_is_preserved_across_model_rerenders() -> None:
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "const openDetailModels = new Set()" in js
    assert "details.open = openDetailModels.has(model.name)" in js
    assert 'details.addEventListener("toggle"' in js
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node.js is unavailable")
@pytest.mark.parametrize("script", sorted(STATIC.glob("*.js")))
def test_javascript_syntax(script: Path):
    completed = subprocess.run(["node", "--check", str(script)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


# ---------------------------------------------------------------------------
# Scan feedback: loading state, progress, and restoration
# ---------------------------------------------------------------------------


def test_scan_status_region_is_present_and_live():
    """The dashboard ships an accessible, polite scan status region."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    assert 'id="scan-status"' in html
    assert 'id="scan-status-text"' in html
    assert 'id="scan-progress"' in html
    assert 'role="status"' in html
    assert 'aria-live="polite"' in html


def test_scan_button_relabels_and_disables_immediately():
    js = (STATIC / "app.js").read_text(encoding="utf-8")

    # The busy and idle labels are named constants so status wording,
    # button label, and tests cannot drift apart.
    assert 'const SCAN_LABEL_IDLE = "Scan for models"' in js
    assert 'const SCAN_LABEL_BUSY = "Scanning…"' in js
    # Loading state is applied before the request is awaited.
    assert "buttonNode.disabled = true" in js
    assert 'buttonNode.setAttribute("aria-busy", "true")' in js
    assert "buttonNode.textContent = SCAN_LABEL_BUSY" in js
    assert js.index("buttonNode.textContent = SCAN_LABEL_BUSY") < js.index('await fetch("/admin/scan"')


def test_scan_shows_honest_indeterminate_progress():
    """The backend exposes no percentage, so the UI shows an indeterminate bar."""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    # Honest wording: no invented percentage in the status copy or markup.
    assert "%" not in html.split('id="scan-status"')[1].split("</div>")[0]
    assert "%" not in js.split("Scanning your folders")[1].split('"')[0]
    # The progress element is an indeterminate bar driven by a CSS animation;
    # no JS ever sets a percentage width on it.
    assert ".scan-progress" in css
    assert "animation" in css
    assert "scan-progress-slide" in css
    assert "scanProgress" not in js.replace("scan-progress", "")


def test_scan_reports_success_and_error_then_restores_button_in_finally():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    scan_body = js[js.index("async function scanModels") : js.index("async function stopAll")]

    # Success path reports what the scan found via the live region.
    assert "Scan finished:" in scan_body
    assert "no new models found" in scan_body
    # Error path explains the failure via the live region.
    assert "Scan failed:" in scan_body
    assert "Check your scan folders and try again" in scan_body
    # Restoration happens in finally, covering success and failure alike.
    finally_body = scan_body[scan_body.index("finally") :]
    assert "scanning = false" in finally_body
    assert "buttonNode.disabled = false" in finally_body
    assert "buttonNode.textContent = SCAN_LABEL_IDLE" in finally_body


@pytest.mark.skipif(NODE is None, reason="Node.js is unavailable")
def test_scan_button_lifecycle_under_stubbed_dom():
    """Drive scanModels through success and failure with a minimal DOM stub and
    prove the button and its aria-busy attribute are restored either way."""
    app = STATIC / "app.js"
    program = f"""
      const fs = require("fs");
      const calls = [];
      let responses = [];
      const elements = new Map();
      function el(id) {{
        if (!elements.has(id)) {{
          elements.set(id, {{
            id,
            disabled: false,
            textContent: "",
            hidden: true,
            innerHTML: "",
            className: "",
            attributes: {{}},
            classList: {{ toggle() {{}}, add() {{}}, remove() {{}} }},
            setAttribute(name, value) {{ this.attributes[name] = value; }},
            removeAttribute(name) {{ delete this.attributes[name]; }},
            querySelector() {{ return el(id + "-child-" + elements.size); }},
            replaceChildren() {{}},
            append() {{}},
            appendChild() {{}},
            addEventListener() {{}},
          }});
        }}
        return elements.get(id);
      }}
      const scanButton = el("scan");
      scanButton.textContent = "Scan for models";
      const statusText = el("scan-status-text");
      globalThis.document = {{
        querySelector: (sel) => el(sel.slice(1)),
        createElement: () => el("created-" + elements.size),
      }};
      globalThis.fetch = async (url, init) => {{
        calls.push({{ url, init, method: init?.method || "GET" }});
        const next = responses.shift();
        if (next instanceof Error) throw next;
        return {{ ok: next.ok, status: next.status, json: async () => next.body }};
      }};
      const source = fs.readFileSync({json.dumps(str(app))}, "utf8");
      // Evaluate definitions only: cut before the DOM event wiring and the
      // top-level init IIFE, which need a fuller browser environment.
      const cut = source.indexOf('$("#connect-frontend").addEventListener');
      (0, eval)(source.slice(0, cut));

      (async () => {{
        const scanRequest = async () => {{
          scanButton.disabled = false;
          scanButton.textContent = "Scan for models";
          delete scanButton.attributes["aria-busy"];
          await scanModels();
          return {{
            disabledAfter: scanButton.disabled,
            labelAfter: scanButton.textContent,
            ariaBusyAfter: scanButton.attributes["aria-busy"] || null,
            statusAfter: statusText.textContent,
          }};
        }};
        // Success: scan finds one model, then the status refresh fails
        // harmlessly (fetchStatus swallows its own errors).
        responses = [
          {{ ok: true, status: 200, body: {{ found: 2, added: ["model-a"] }} }},
          {{ ok: false, status: 503, body: {{}} }},
        ];
        const success = await scanRequest();
        // Error: the scan request itself rejects.
        responses = [new Error("Scan failed (500)"), {{ ok: false, status: 503, body: {{}} }}];
        const failure = await scanRequest();
        process.stdout.write(JSON.stringify({{ success, failure, calls }}));
      }})().catch((err) => {{
        console.error(err);
        process.exit(1);
      }});
    """
    completed = subprocess.run([NODE, "-e", program], check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)
    scan_calls = [c for c in result["calls"] if c["url"] == "/admin/scan"]
    assert len(scan_calls) == 2
    assert all(c["method"] == "POST" for c in scan_calls)
    assert result["success"]["statusAfter"].startswith("Scan finished: found 2, added 1 model.")
    assert result["failure"]["statusAfter"].startswith("Scan failed:")
    for phase in ("success", "failure"):
        assert result[phase]["disabledAfter"] is False
        assert result[phase]["labelAfter"] == "Scan for models"
        assert result[phase]["ariaBusyAfter"] is None


# ---------------------------------------------------------------------------
# Punctuation: no Unicode em dashes in user-visible static UI text
# ---------------------------------------------------------------------------

EM_DASH = "—"


@pytest.mark.parametrize(
    "filename",
    ["index.html", "app.js", "chat.html", "chat.js", "style.css", "chat.css"],
)
def test_no_em_dashes_in_user_visible_ui_text(filename: str):
    content = (STATIC / filename).read_text(encoding="utf-8")
    offending = [i for i, ch in enumerate(content) if ch == EM_DASH]
    assert not offending, (
        f"{filename} contains Unicode em dash characters at offsets {offending[:8]}; "
        "use commas, colons, or a normal hyphen in user-visible text"
    )


@pytest.mark.skipif(NODE is None, reason="Node.js is unavailable")
def test_markdown_url_policy_blocks_script_protocols():
    helper = STATIC / "markdown_safety.js"
    program = f"""
      const safety = require({json.dumps(str(helper))});
      const result = {{
        javascript: safety.safeUrl('javascript:alert(1)'),
        encodedControl: safety.safeUrl('java\\u0000script:alert(1)'),
        data: safety.safeUrl('data:text/html,<script>alert(1)</script>'),
        https: safety.safeUrl('https://example.com/path'),
        relative: safety.safeUrl('/docs'),
        unsafeLink: safety.link('javascript:alert(1)', null, 'click'),
      }};
      process.stdout.write(JSON.stringify(result));
    """
    completed = subprocess.run([NODE, "-e", program], check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)
    assert result["javascript"] is None
    assert result["encodedControl"] is None
    assert result["data"] is None
    assert result["https"] == "https://example.com/path"
    assert result["relative"] == "/docs"
    assert "href" not in result["unsafeLink"]


@pytest.mark.skipif(NODE is None, reason="Node.js is unavailable")
def test_marked_renderer_handles_safe_and_hostile_markdown():
    """Exercise the renderer contract used by the pinned Marked 12 browser script."""
    marked = STATIC / "vendor" / "marked-12.0.2.min.js"
    helper = STATIC / "markdown_safety.js"
    program = f"""
      const {{marked}} = require({json.dumps(str(marked))});
      const safety = require({json.dumps(str(helper))});
      const renderer = new marked.Renderer();
      renderer.html = (text) => safety.escapeHtml(text);
      renderer.link = (href, title, text) => safety.link(href, title, text);
      renderer.image = (href, title, text) => safety.image(href, title, text);
      const inputs = [
        '[bad](javascript:alert(1))',
        '[ok](https://example.com)',
        '<img src=x onerror=alert(1)>',
        '![x](data:text/html,bad)',
      ];
      process.stdout.write(JSON.stringify(inputs.map(x => marked.parse(x, {{renderer}}))));
    """
    completed = subprocess.run([NODE, "-e", program], check=True, capture_output=True, text=True)
    bad_link, good_link, raw_html, bad_image = json.loads(completed.stdout)
    assert "javascript:" not in bad_link
    assert 'href="https://example.com"' in good_link
    assert "onerror" in raw_html and "<img" not in raw_html
    assert "data:" not in bad_image


def test_chat_loads_safety_policy_before_renderer():
    html = (STATIC / "chat.html").read_text(encoding="utf-8")
    assert html.index("markdown_safety.js") < html.index("chat.js")


def test_chat_uses_clear_on_demand_model_status():
    chat_html = (STATIC / "chat.html").read_text(encoding="utf-8")
    chat_js = (STATIC / "chat.js").read_text(encoding="utf-8")

    assert '<span id="status-text">loading models</span>' in chat_html
    assert 'updateStatus("idle")' in chat_js
    assert 'updateStatus("unavailable")' in chat_js


# ---------------------------------------------------------------------------
# Connect a frontend panel
# ---------------------------------------------------------------------------


def test_dashboard_offers_connect_a_frontend_action():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")

    assert 'id="connect-frontend"' in html
    assert "Connect a frontend" in html
    # The action opens the guided dialog, not a new page.
    assert "openFrontendDialog" in js
    assert '$("#frontend-dialog").showModal()' in js


def test_frontend_dialog_lists_three_guided_choices():
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    assert 'id="tab-openwebui"' in html
    assert 'id="tab-ollama"' in html
    assert 'id="tab-generic"' in html
    assert "Open WebUI" in html
    assert "Ollama" in html
    # Panels are tabpanels controlled by the tabs, one visible by default.
    assert 'role="tabpanel"' in html
    assert html.count('role="tabpanel"') == 3


def test_frontend_dialog_uses_no_external_assets():
    """Offline usability: the dialog must rely only on the local page."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    dialog = html[html.index("frontend-dialog") : html.index("</dialog>")]
    assert "http://" not in dialog
    assert "https://" not in dialog
    assert "src=" not in dialog
    assert 'href="' not in dialog


def test_frontend_panel_reads_admin_integration_endpoint():
    js = (STATIC / "app.js").read_text(encoding="utf-8")

    assert 'fetch("/admin/integration"' in js
    assert "authHeaders()" in js


def test_frontend_copies_use_no_credentials():
    """The flow is copy-only guidance; no token fields may appear."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    dialog = html[html.index("frontend-dialog") : html.index("</dialog>")]

    assert "admin_token" not in dialog
    assert "session-token" not in dialog
    assert "Authorization" not in dialog


def test_openwebui_panel_shows_copyable_base_url_and_key_guidance():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")

    assert 'id="openwebui-url"' in html
    assert 'id="copy-openwebui-url"' in html
    # app.js renders the API-key guidance from the backend payload so wording
    # stays consistent with the server, next to a copyable URL.
    assert 'id="openwebui-key-note"' in html
    assert "api_key_guidance" in js


def test_ollama_panel_uses_existing_upstream_flow_copyably():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")

    # The panel shows a copyable upstream add command, sourced from the
    # backend's discovery payload.
    assert 'id="ollama-command"' in html
    assert 'id="copy-ollama-command"' in html
    assert "upstream_add_command" in js
    # The command is copyable, not executed remotely (no POST from the panel).
    assert "upstream/add" not in js


def test_generic_panel_shows_curl_example():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")

    assert 'id="generic-url"' in html
    assert 'id="generic-curl"' in html
    # curl examples are platform-aware: the payload ships both variants and
    # the UI renders the one matching the browser.
    assert "curl_example_windows" in js
    assert "curl_example" in js


def test_frontend_tabs_are_keyboard_accessible():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")

    assert 'role="tablist"' in html
    assert html.count('role="tab"') == 3
    assert 'aria-selected="true"' in html
    assert 'aria-selected="false"' in html
    assert 'tab.setAttribute("aria-selected"' in js


def test_dialog_styles_exist_for_frontend_panel():
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    assert ".frontend-dialog" in css
    assert ".frontend-tab" in css
    assert ".frontend-copy" in css


# ---------------------------------------------------------------------------
# Plugins panel
# ---------------------------------------------------------------------------


def test_plugins_panel_is_present_and_labeled():
    html = (STATIC / "index.html").read_text()

    assert 'id="plugins-title"' in html
    assert "Plugins" in html
    assert 'id="plugin-list"' in html
    assert "arc_llama.plugins" in html


def test_plugins_panel_reads_admin_plugins_endpoint():
    js = (STATIC / "app.js").read_text()

    assert 'fetch("/admin/plugins"' in js
    assert "authHeaders()" in js


def test_plugins_panel_renders_name_status_and_optional_metadata():
    js = (STATIC / "app.js").read_text()

    assert "plugin.name" in js
    assert "plugin.status" in js
    assert "plugin.description" in js
    assert "plugin.version" in js
    assert "plugin.ui" in js
    assert "plugin.api" in js


def test_plugins_panel_empty_state_mentions_entry_point_discovery():
    js = (STATIC / "app.js").read_text()

    assert "No plugins installed" in js
    assert "arc_llama.plugins" in js


def test_plugin_fetch_failure_is_isolated():
    js = (STATIC / "app.js").read_text()
    block = js[js.index("async function fetchPlugins") : js.index("// Connect-a-frontend")]

    # The whole request/render path is wrapped so a failing plugins fetch
    # can never break the model panels it shares the page with.
    assert "try {" in block
    assert "catch (_)" in block


@pytest.mark.skipif(NODE is None, reason="Node.js is unavailable")
def test_plugins_panel_lifecycle_under_stubbed_dom():
    """Exercise renderPlugins/fetchPlugins against a stubbed DOM: metadata
    rendering status pills, and the empty state, with fetch failures
    leaving the page untouched."""
    app = STATIC / "app.js"
    program = f"""
      const fs = require("fs");
      const cards = [];
      const pluginList = {{ replaceChildren() {{ cards.length = 0; }}, appendChild(node) {{ cards.push(node); }} }};
      globalThis.document = {{
        querySelector: (sel) => (sel === "#plugin-list" ? pluginList : null),
        createElement: () => ({{
          className: "", textContent: "", appendChild() {{}}, append() {{}},
          classList: {{ add() {{}} }},
        }}),
      }};
      const source = fs.readFileSync({json.dumps(str(app))}, "utf8");
      const cut = source.indexOf('$("#connect-frontend").addEventListener');
      (0, eval)(source.slice(0, cut));

      (async () => {{
        const out = {{}};
        // Success with metadata: one populated plugin card.
        globalThis.fetch = async () => ({{
          ok: true, status: 200, json: async () => ({{
            plugins: [{{ name: "audio", status: "active", version: "1.0", description: "Audio routes", ui: "Audio panel", api: ["/v1/audio"] }}],
          }}),
        }});
        await fetchPlugins();
        out.loadedCards = cards.length;
        // Failure: keeps whatever was rendered, renders nothing new.
        globalThis.fetch = async () => {{ throw new Error("down"); }};
        await fetchPlugins();
        out.afterFailure = cards.length;
        // Empty catalog: exactly one empty-state note.
        globalThis.fetch = async () => ({{
          ok: true, status: 200, json: async () => ({{ plugins: [] }}),
        }});
        await fetchPlugins();
        out.emptyCards = cards.length;
        process.stdout.write(JSON.stringify(out));
      }})().catch((err) => {{
        console.error(err);
        process.exit(1);
      }});
    """
    completed = subprocess.run([NODE, "-e", program], check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)
    assert result["loadedCards"] == 1
    assert result["afterFailure"] == 1
    assert result["emptyCards"] == 1
