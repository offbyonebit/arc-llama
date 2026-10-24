# Work: Documents

Documents is a second kind of Work next to Code. It is for Word, Excel, PowerPoint and PDF files: read them, summarise them, build new ones, and look at the result.

## What it does

- Reads `.docx`, `.xlsx` and `.pptx` (headings, lists, tables, formulas, slide text and notes), and PDFs through the existing `read_pdf`.
- Creates `.docx`, `.xlsx` and `.pptx` files from structured content. It never edits a file in place; a change is written to a new file, so the original is safe. Every new file asks first and is covered by Undo and checkpoints like any other write.
- Looks at a page, slide or image with `view_page`, so the model can check layout and charts.
- Has no shell, no patching and no web access. Only the project folder.

## Requirements

- **A vision model.** The Documents type is offered only for models with a vision projector (`mmproj`), and the server rejects any other model with a clear message. Pairing a projector is described in the model recipe docs.
- **The built-in agent.** OpenCode and Hermes are not used for Documents.
- `pip install 'arc-llama[documents]'` for python-docx, openpyxl and python-pptx.
- To look at pages: `pdftoppm` (poppler-utils) for PDFs, WordCraft (`wordcraft-cli`) for Word files, and LibreOffice (`soffice`) for Excel and PowerPoint. Without them the model can still read and create files, and `view_page` says what is missing.

## API

`POST /v1/agent` takes `"work_type": "code" | "documents"` (default `code`). `GET /v1/agent/info` lists `work_types` and the `vision_models` Documents can use.

## Not built yet

- Scheduled tasks: saved tasks that run on a timer and wait in an inbox for approval.
- Command isolation with Bubblewrap for the harnesses (the built-in Documents type already has no shell).
- Documents on text-only models, which would need page text extraction instead of page images.
- Task profiles beyond Code and Documents (research, spreadsheets-only, and so on).
