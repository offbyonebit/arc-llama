"""Document tools for Work's Documents type: read, create and look at Office files and PDFs.

These live in their own registry so coding runs never see them. Writing needs the optional
``documents`` extra (python-docx, openpyxl, python-pptx); reading a page as an image needs
``pdftoppm`` for PDFs, WordCraft for Word files, and LibreOffice for Excel and PowerPoint. Everything stays inside the project folder."""
from __future__ import annotations

import base64
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from arc_llama.agent.tools import (
    TOOLS,
    ToolContext,
    ToolError,
    ToolRegistry,
    ToolResult,
    _ensure_checkpoint,
    _resolve_path,
)

DOCUMENT_TOOLS = ToolRegistry()

# Coding tools that are safe and useful when working with documents. No shell, no patching.
SHARED_TOOLS = frozenset({"read_file", "read_multiple_files", "read_pdf", "list_directory", "search_files"})

MAX_READ_CHARS = 60_000
MAX_SHEET_ROWS = 200
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
OFFICE_SUFFIXES = {".docx", ".xlsx", ".pptx"}
INSTALL_HINT = "Install the document libraries with: pip install 'arc-llama[documents]'"


def documents_registry() -> ToolRegistry:
    """The tool set for a Documents run: shared read tools plus the document tools."""
    registry = TOOLS.select(include=SHARED_TOOLS)
    for name in DOCUMENT_TOOLS.list_tools():
        tool = DOCUMENT_TOOLS.get(name)
        if tool is not None:
            registry.register(tool)
    return registry


def _tool(name: str, description: str, parameters: dict[str, Any], *, confirm: bool = False) -> Any:
    return DOCUMENT_TOOLS.register_function(
        name=name, description=description, parameters=parameters, requires_confirmation=confirm
    )


def _clip(text: str) -> str:
    if len(text) <= MAX_READ_CHARS:
        return text
    return text[:MAX_READ_CHARS] + f"\n... (cut after {MAX_READ_CHARS} characters)"


def _library(module: str) -> Any:
    try:
        return __import__(module)
    except ImportError as exc:
        raise ToolError(f"{module} is not installed. {INSTALL_HINT}") from exc


def _target(path: str, ctx: ToolContext, suffix: str | None = None) -> Path:
    target = _resolve_path(path, ctx.root)
    if suffix and target.suffix.lower() != suffix:
        raise ToolError(f"The file name must end in {suffix}: {path}")
    return target


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _read_docx(target: Path) -> str:
    docx = _library("docx")
    document = docx.Document(str(target))
    lines: list[str] = []
    body = document.element.body
    paragraphs = {p._p: p for p in document.paragraphs}
    tables = {t._tbl: t for t in document.tables}
    for child in body.iterchildren():
        if child in paragraphs:
            para = paragraphs[child]
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name if para.style is not None else "") or ""
            if style.startswith("Heading"):
                level = style.split()[-1]
                depth = int(level) if level.isdigit() else 1
                lines.append("#" * min(depth, 6) + " " + text)
            elif "List" in style:
                lines.append("- " + text)
            else:
                lines.append(text)
        elif child in tables:
            for row in tables[child].rows:
                lines.append("| " + " | ".join(cell.text.strip().replace("\n", " ") for cell in row.cells) + " |")
            lines.append("")
    return "\n".join(lines)


def _read_xlsx(target: Path) -> str:
    openpyxl = _library("openpyxl")
    book = openpyxl.load_workbook(str(target), data_only=False, read_only=True)
    out: list[str] = []
    try:
        for sheet in book.worksheets:
            out.append(f"## Sheet: {sheet.title}")
            for index, row in enumerate(sheet.iter_rows(values_only=True), start=1):
                if index > MAX_SHEET_ROWS:
                    out.append(f"... (more rows after row {MAX_SHEET_ROWS})")
                    break
                if all(cell is None for cell in row):
                    continue
                out.append(f"{index}: " + " | ".join("" if cell is None else str(cell) for cell in row))
            out.append("")
    finally:
        book.close()
    return "\n".join(out)


def _read_pptx(target: Path) -> str:
    pptx = _library("pptx")
    deck = pptx.Presentation(str(target))
    out: list[str] = []
    for number, slide in enumerate(deck.slides, start=1):
        out.append(f"## Slide {number}")
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False) and shape.text_frame.text.strip():
                out.append(shape.text_frame.text.strip())
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            out.append("Notes: " + slide.notes_slide.notes_text_frame.text.strip())
        out.append("")
    return "\n".join(out)


_READERS = {".docx": _read_docx, ".xlsx": _read_xlsx, ".pptx": _read_pptx}


@_tool(
    "read_document",
    "Read the text of a Word (.docx), Excel (.xlsx) or PowerPoint (.pptx) file. Headings, "
    "lists and tables are kept. For PDFs use read_pdf; to see how a page looks use view_page.",
    {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path of the file inside the project folder."}},
        "required": ["path"],
    },
)
def _read_document(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    target = _resolve_path(str(arguments.get("path", "")), ctx.root)
    reader = _READERS.get(target.suffix.lower())
    if reader is None:
        return ToolResult("read_document handles .docx, .xlsx and .pptx files.", error=True)
    if not target.is_file():
        return ToolResult(f"File not found: {arguments.get('path')}", error=True)
    return ToolResult(_clip(reader(target)))


# ---------------------------------------------------------------------------
# Creating
# ---------------------------------------------------------------------------


def _save_checkpointed(ctx: ToolContext, target: Path, save: Any) -> ToolResult:
    checkpoint_id = _ensure_checkpoint(ctx)
    target.parent.mkdir(parents=True, exist_ok=True)
    save(str(target))
    return ToolResult(f"Wrote {target.relative_to(ctx.root.resolve())}", checkpoint_id=checkpoint_id)


@_tool(
    "create_document",
    "Create a Word (.docx) file. 'blocks' is the document in order. Each block is one of: "
    '{"type":"heading","text":"...","level":1}, {"type":"paragraph","text":"..."}, '
    '{"type":"bullets","items":["..."]}, {"type":"numbered","items":["..."]}, '
    '{"type":"table","rows":[["header","header"],["cell","cell"]]}, {"type":"page_break"}.',
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Where to save, ending in .docx."},
            "title": {"type": "string", "description": "Optional title shown at the top."},
            "blocks": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["path", "blocks"],
    },
    confirm=True,
)
def _create_document(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    target = _target(str(arguments.get("path", "")), ctx, ".docx")
    docx = _library("docx")
    blocks = arguments.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        return ToolResult("'blocks' must be a non-empty list.", error=True)
    document = docx.Document()
    if arguments.get("title"):
        document.add_heading(str(arguments["title"]), level=0)
    for block in blocks:
        if not isinstance(block, dict):
            return ToolResult("Every block must be an object.", error=True)
        kind = block.get("type", "paragraph")
        if kind == "heading":
            document.add_heading(str(block.get("text", "")), level=max(1, min(int(block.get("level", 1)), 4)))
        elif kind == "paragraph":
            document.add_paragraph(str(block.get("text", "")))
        elif kind in ("bullets", "numbered"):
            style = "List Bullet" if kind == "bullets" else "List Number"
            for item in block.get("items") or []:
                document.add_paragraph(str(item), style=style)
        elif kind == "table":
            rows = [[str(c) for c in row] for row in block.get("rows") or [] if isinstance(row, list)]
            if not rows:
                return ToolResult("A table needs rows.", error=True)
            width = max(len(r) for r in rows)
            table = document.add_table(rows=len(rows), cols=width)
            table.style = "Table Grid"
            for r, row in enumerate(rows):
                for c, value in enumerate(row):
                    table.cell(r, c).text = value
        elif kind == "page_break":
            document.add_page_break()
        else:
            return ToolResult(f"Unknown block type: {kind}", error=True)
    return _save_checkpointed(ctx, target, document.save)


@_tool(
    "create_spreadsheet",
    "Create an Excel (.xlsx) file. Each sheet has a name and rows; a cell text starting with '=' "
    "is a formula. Numbers should be sent as numbers, not text.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Where to save, ending in .xlsx."},
            "sheets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "rows": {"type": "array", "items": {"type": "array"}},
                    },
                    "required": ["name", "rows"],
                },
            },
        },
        "required": ["path", "sheets"],
    },
    confirm=True,
)
def _create_spreadsheet(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    target = _target(str(arguments.get("path", "")), ctx, ".xlsx")
    openpyxl = _library("openpyxl")
    sheets = arguments.get("sheets")
    if not isinstance(sheets, list) or not sheets:
        return ToolResult("'sheets' must be a non-empty list.", error=True)
    book = openpyxl.Workbook()
    book.remove(book.active)
    for sheet in sheets:
        if not isinstance(sheet, dict) or not isinstance(sheet.get("rows"), list):
            return ToolResult("Every sheet needs a name and a list of rows.", error=True)
        name = str(sheet.get("name") or f"Sheet{len(book.worksheets) + 1}")[:31]
        worksheet = book.create_sheet(title=name)
        for row in sheet["rows"]:
            worksheet.append(list(row) if isinstance(row, list) else [row])
    return _save_checkpointed(ctx, target, book.save)


@_tool(
    "create_presentation",
    "Create a PowerPoint (.pptx) file. Each slide has a title, optional bullets and optional "
    "speaker notes. The first slide is used as the title slide when it has no bullets.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Where to save, ending in .pptx."},
            "slides": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "bullets": {"type": "array", "items": {"type": "string"}},
                        "notes": {"type": "string"},
                    },
                    "required": ["title"],
                },
            },
        },
        "required": ["path", "slides"],
    },
    confirm=True,
)
def _create_presentation(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    target = _target(str(arguments.get("path", "")), ctx, ".pptx")
    pptx = _library("pptx")
    slides = arguments.get("slides")
    if not isinstance(slides, list) or not slides:
        return ToolResult("'slides' must be a non-empty list.", error=True)
    deck = pptx.Presentation()
    for index, slide in enumerate(slides):
        if not isinstance(slide, dict):
            return ToolResult("Every slide must be an object.", error=True)
        bullets = [str(b) for b in slide.get("bullets") or []]
        layout = deck.slide_layouts[0 if index == 0 and not bullets else 1]
        made = deck.slides.add_slide(layout)
        made.shapes.title.text = str(slide.get("title", ""))
        if bullets and len(made.placeholders) > 1:
            frame = made.placeholders[1].text_frame
            frame.text = bullets[0]
            for line in bullets[1:]:
                frame.add_paragraph().text = line
        if slide.get("notes"):
            made.notes_slide.notes_text_frame.text = str(slide["notes"])
    return _save_checkpointed(ctx, target, deck.save)


# ---------------------------------------------------------------------------
# Looking at pages (vision models only)
# ---------------------------------------------------------------------------


WORDCRAFT_SUFFIXES = {".docx", ".doc", ".odt", ".rtf"}


def _wordcraft() -> str | None:
    return shutil.which("wordcraft-cli-local") or shutil.which("wordcraft-cli")


def _wordcraft_page(source: Path, page: int, workdir: Path) -> bytes:
    """Render one page of a Word-type file with WordCraft; no LibreOffice needed."""
    binary = _wordcraft()
    if binary is None:
        raise ToolError("WordCraft is not installed.")
    out = workdir / "page.png"
    cmd = [binary, "render", str(source), str(out), "--page", str(page), "--scale", "0.7"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except (subprocess.SubprocessError, OSError) as exc:
        raise ToolError(f"Could not render page {page} of {source.name}: {exc}") from exc
    if not out.is_file():
        raise ToolError(f"{source.name} has no page {page}.")
    return out.read_bytes()


def _to_pdf(source: Path, workdir: Path) -> Path:
    office = shutil.which("soffice") or shutil.which("libreoffice")
    if office is None:
        raise ToolError("Viewing Excel or PowerPoint pages needs LibreOffice (soffice) on the PATH.")
    # A private profile keeps this from touching or waiting on the user's own LibreOffice.
    cmd = [
        office, f"-env:UserInstallation=file://{workdir}/profile", "--headless",
        "--convert-to", "pdf", "--outdir", str(workdir), str(source),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except (subprocess.SubprocessError, OSError) as exc:
        raise ToolError(f"Could not convert {source.name} to PDF: {exc}") from exc
    pdf = workdir / (source.stem + ".pdf")
    if not pdf.is_file():
        raise ToolError(f"Could not convert {source.name} to PDF.")
    return pdf


def _render_pdf_page(pdf: Path, page: int, workdir: Path) -> bytes:
    if shutil.which("pdftoppm") is None:
        raise ToolError("Viewing pages needs pdftoppm (poppler-utils) on the PATH.")
    prefix = workdir / "page"
    cmd = ["pdftoppm", "-png", "-r", "80", "-f", str(page), "-l", str(page), "-singlefile", str(pdf), str(prefix)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    except (subprocess.SubprocessError, OSError) as exc:
        raise ToolError(f"Could not render page {page}: {exc}") from exc
    out = workdir / "page.png"
    if not out.is_file():
        raise ToolError(f"{pdf.name} has no page {page}.")
    return out.read_bytes()


def _data_url(mime: str, data: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


@_tool(
    "view_page",
    "Look at a page of a PDF, a slide or page of an Office file, or an image, as a picture. "
    "Use it to check layout, charts and anything the text alone does not show.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "A PDF, image, .docx, .xlsx or .pptx file."},
            "page": {"type": "integer", "description": "Page or slide number, starting at 1."},
        },
        "required": ["path"],
    },
)
def _view_page(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    target = _resolve_path(str(arguments.get("path", "")), ctx.root)
    if not target.is_file():
        return ToolResult(f"File not found: {arguments.get('path')}", error=True)
    page = arguments.get("page", 1)
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        return ToolResult("page must be a positive whole number.", error=True)
    suffix = target.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else f"image/{suffix[1:]}"
        return ToolResult(f"Showing {target.name}.", images=(_data_url(mime, target.read_bytes()),))
    if suffix != ".pdf" and suffix not in OFFICE_SUFFIXES:
        return ToolResult("view_page handles PDFs, images, .docx, .xlsx and .pptx files.", error=True)
    with tempfile.TemporaryDirectory(prefix="arc-llama-page-") as tmp:
        workdir = Path(tmp)
        if suffix in WORDCRAFT_SUFFIXES and _wordcraft() is not None:
            png = _wordcraft_page(target, page, workdir)
        else:
            pdf = target if suffix == ".pdf" else _to_pdf(target, workdir)
            png = _render_pdf_page(pdf, page, workdir)
    return ToolResult(f"Showing page {page} of {target.name}.", images=(_data_url("image/png", png),))
