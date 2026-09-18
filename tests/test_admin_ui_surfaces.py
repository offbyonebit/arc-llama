"""Static-UI integration points for the new admin surfaces: the settings
panel's honest memory-fit line, the dashboard Measurements section, and the
chat status polling that feeds them."""

from __future__ import annotations

from pathlib import Path

STATIC = Path(__file__).parent.parent / "src" / "arc_llama" / "static"


def test_settings_panel_renders_vram_fit_line():
    js = (STATIC / "chat.js").read_text()
    assert "function vramFitText" in js
    # Honest wording for every branch: fit with headroom, does-not-fit with
    # an action, unknown capacity, and not-estimated.
    assert "MiB headroom on the assigned GPU" in js
    assert "Reduce context or KV size, or pick a smaller model" in js
    assert "GPU capacity unknown; no fit verdict" in js
    assert "not estimated yet" in js


def test_settings_panel_consumes_admin_status_vram_estimate():
    js = (STATIC / "chat.js").read_text()
    assert "s.vram_estimate" in js
    assert "m.vram_estimate = s.vram_estimate" in js
    assert 'id="s-fit"' in js


def test_apply_settings_refreshes_the_fit_line():
    js = (STATIC / "chat.js").read_text()
    body = js[js.index("async function applySettings") : js.index("function estimateTokens")]
    # After saving recipe edits the status is refetched so the fit line and
    # loaded state re-render instead of showing stale estimates.
    assert "fetchStatus().catch" in body


def test_chat_status_css_exists_for_fit_line():
    css = (STATIC / "chat.css").read_text()
    assert ".s-fit" in css


def test_dashboard_has_measurements_section():
    html = (STATIC / "index.html").read_text()
    assert 'id="measurements"' in html
    assert 'id="measurements-title"' in html
    assert "Measurements" in html


def test_dashboard_renders_only_measured_values():
    js = (STATIC / "app.js").read_text()
    assert "function renderMeasurements" in js
    assert "fetchMeasurements" in js
    # The empty state is explicit that numbers come from real traffic.
    assert "No measurements yet" in js
    # Never invents throughput: generation speed rows come straight from
    # the server's summary, labelled tok/s.
    assert '" tok/s"' in js
    # The section is refreshed on its own slower cadence alongside status.
    assert js.count("setInterval(fetchMeasurements") == 1


def test_measurements_css_exists():
    css = (STATIC / "style.css").read_text()
    assert ".measurements-card" in css
    assert ".measure-rows" in css


def test_dashboard_readiness_shows_memory_fit_from_status():
    js = (STATIC / "app.js").read_text()
    assert "model.vram_estimate" in js
    assert "Memory fit" in js
    assert "Does not fit" in js
    assert "headroom" in js


def test_readiness_tone_classes_exist():
    css = (STATIC / "style.css").read_text()
    assert ".metric strong.tone-ok" in css or "tone-ok" in css
    assert "tone-warn" in css
