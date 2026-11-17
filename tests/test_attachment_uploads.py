"""Office uploads exercise the existing readers with small, in-test documents."""
from __future__ import annotations

import io
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from arc_llama.api import attachments
from arc_llama.config import Config, ServerConfig
from arc_llama.server import _require_admin

HEADERS = {"Authorization": "Bearer secret"}


@pytest.fixture
def client():
    app = FastAPI()
    app.state.cfg = Config(server=ServerConfig(admin_token="secret"))
    attachments.register_attachment_routes(app, require_admin=_require_admin)
    with TestClient(app) as client:
        yield client


def office(suffix):
    stream = io.BytesIO()
    if suffix == ".docx":
        docx = pytest.importorskip("docx")
        document = docx.Document()
        document.add_heading("Controlled Word title", 1)
        document.add_paragraph("A small Word attachment.")
        document.save(stream)
    elif suffix == ".xlsx":
        openpyxl = pytest.importorskip("openpyxl")
        book = openpyxl.Workbook()
        book.active.title = "Controlled totals"
        book.active.append(["Item", "Cost"])
        book.active.append(["Example", "=2+3"])
        book.save(stream)
        book.close()
    else:
        pptx = pytest.importorskip("pptx")
        presentation = pptx.Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[0])
        slide.shapes.title.text = "Controlled slide title"
        presentation.save(stream)
    return stream.getvalue()


@pytest.mark.parametrize("suffix,expected", [
    (".docx", "A small Word attachment."), (".xlsx", "=2+3"), (".pptx", "Controlled slide title"),
])
def test_office_upload_extracts_existing_reader_text(client, suffix, expected):
    response = client.post("/admin/parse-document", files={"file": ("controlled" + suffix, office(suffix))}, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["filename"] == "controlled" + suffix
    assert expected in response.json()["text"]


def test_office_upload_auth_and_unsupported_or_corrupt(client):
    assert client.post("/admin/parse-document", files={"file": ("x.docx", b"x")}).status_code == 401
    assert client.post("/admin/parse-document", files={"file": ("x.exe", b"x")}, headers=HEADERS).status_code == 400
    assert client.post("/admin/parse-document", files={"file": ("x.docx", b"broken")}, headers=HEADERS).status_code == 422


def test_office_upload_bounds_compressed_and_expanded_size(client, monkeypatch):
    monkeypatch.setattr(attachments, "MAX_UPLOAD_BYTES", 1024)
    response = client.post("/admin/parse-document", files={"file": ("x.docx", b"x" * 1025)}, headers=HEADERS)
    assert response.status_code == 413
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "x" * 4096)
    assert len(stream.getvalue()) < 1024
    response = client.post("/admin/parse-document", files={"file": ("x.docx", stream.getvalue())}, headers=HEADERS)
    assert response.status_code == 413


def test_missing_optional_reader_is_clear(client, monkeypatch):
    from arc_llama.agent import documents
    from arc_llama.agent.tools import ToolError

    content = office(".docx")

    def missing(_module):
        raise ToolError("docx is not installed. " + documents.INSTALL_HINT)

    monkeypatch.setattr(documents, "_library", missing)
    response = client.post("/admin/parse-document", files={"file": ("x.docx", content)}, headers=HEADERS)
    assert response.status_code == 501
    assert "arc-llama[documents]" in response.json()["detail"]
