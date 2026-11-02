from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from click.testing import CliRunner

from arc_llama.arch import Arch
from arc_llama.config import Config, GPUConfig
from arc_llama.detect import DetectedGPU
from arc_llama.issue_report import (
    ISSUE_TEMPLATE,
    MAX_URL_LENGTH,
    REPO_URL,
    ReportContext,
    build_report_link,
    collect_context,
    redact,
)


def _query(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


def _gpu() -> DetectedGPU:
    return DetectedGPU(
        pci_slot="0000:03:00.0",
        device_id=0xE211,
        arch=Arch.BATTLEMAGE,
        name="Intel Arc Pro B60",
        driver="xe",
        vram_mb=24576,
        drm_card=None,
        drm_render=None,
        sysfs_path="/sys/bus/pci/devices/0000:03:00.0",
    )


def test_redact_removes_tokens_and_home(monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: Path("/users/tester")))
    text = (
        "Authorization: Bearer abc.def-123\n"
        "hf_token = hf_abcdefghijklmnopqrstuvwxyz\n"
        "loading /users/tester/models/x.gguf\n"
        "GET /v1/models?api_key=topsecret"
    )
    out = redact(text)
    assert "abc.def-123" not in out
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in out
    assert "topsecret" not in out
    assert "/users/tester" not in out
    assert "$HOME/models/x.gguf" in out


def test_redact_removes_windows_home_in_every_spelling(monkeypatch):
    from pathlib import PureWindowsPath

    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: PureWindowsPath("C:/Users/Sam")))
    monkeypatch.setattr(sys, "platform", "win32")
    text = (
        "loading c:\\users\\sam\\models\\x.gguf\n"
        "config C:/Users/Sam/AppData/arc.toml\n"
        "No such file: 'C:\\\\Users\\\\Sam\\\\models\\\\y.gguf'"
    )
    out = redact(text)
    assert "sam" not in out.lower()
    assert out.count("$HOME") == 3


def test_issue_link_prefills_form_fields(tmp_path):
    (tmp_path / "qwen.log").write_text("line 1\nerror: device lost\n", encoding="utf-8")
    cfg = Config(gpus=[GPUConfig("0000:03:00.0", 0, "battlemage", backend="vulkan")])
    ctx = collect_context(cfg, detected_gpus=[_gpu()], log_dir=tmp_path)

    link = build_report_link(ctx, title="Model won't load")

    assert link.url.startswith(f"{REPO_URL}/issues/new?")
    q = _query(link.url)
    assert q["template"] == ISSUE_TEMPLATE
    assert q["title"] == "Model won't load"
    assert "Backend: vulkan" in q["environment"]
    assert "Intel Arc Pro B60" in q["gpu"] and "driver xe" in q["gpu"]
    assert "error: device lost" in q["logs"]
    assert link.trimmed is False


def test_discussion_link_uses_qa_category():
    link = build_report_link(ReportContext(), kind="discussion")
    q = _query(link.url)
    assert "/discussions/new?" in link.url
    assert q["category"] == "q-a"
    assert "### Environment" in q["body"]


def test_long_logs_are_trimmed_to_fit():
    ctx = ReportContext(log_name="big.log", log_lines=[f"line {i} " + "x" * 200 for i in range(40)])
    link = build_report_link(ctx)
    assert len(link.url) <= MAX_URL_LENGTH
    assert link.trimmed is True
    assert link.log_lines_included < 40
    assert "line 39" in _query(link.url)["logs"]


def test_log_left_out_entirely_when_nothing_else_fits():
    ctx = ReportContext(log_lines=["x" * 5000] * 10)
    link = build_report_link(ctx, max_length=2000)
    assert link.log_lines_included == 0
    assert "x" * 100 not in link.url
    assert "support-bundle" in _query(link.url)["logs"]


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        build_report_link(ReportContext(), kind="email")


def test_report_command_prints_link_without_browser(tmp_path, monkeypatch):
    import arc_llama.cli as cli_mod

    monkeypatch.setattr(cli_mod, "detect_gpus", lambda enrich=True: [], raising=False)
    result = CliRunner().invoke(
        cli_mod.cli, ["--config", str(tmp_path / "missing.toml"), "report", "--no-open"]
    )
    assert result.exit_code == 0, result.output
    assert f"{REPO_URL}/issues/new?" in result.output


async def test_admin_report_link_endpoint(tmp_path):
    from arc_llama.server import create_app

    (tmp_path / "m.log").write_text("boom\n", encoding="utf-8")
    cfg = Config(gpus=[GPUConfig("0000:03:00.0", 0, "alchemist", name="Intel Arc A770")])
    app = create_app(cfg, plugins=[])
    app.state.cfg = cfg
    app.state.router = SimpleNamespace(log_dir=tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        ok = await client.get("/admin/report-link?kind=issue")
        bad = await client.get("/admin/report-link?kind=email")
    assert ok.status_code == 200
    body = ok.json()
    assert body["url"].startswith(f"{REPO_URL}/issues/new?")
    assert "Intel Arc A770" in body["preview"]["gpu"]
    assert "boom" in body["preview"]["log"]
    assert bad.status_code == 400
