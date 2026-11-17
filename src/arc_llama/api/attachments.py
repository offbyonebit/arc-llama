"""Chat Office text uploads reuse Work's optional readers without adding parsers."""
from __future__ import annotations

import asyncio
import io
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, UploadFile

# Match the existing chat PDF upload limit; also bound the expanded Office archive.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def _extract(content: bytes, suffix: str) -> str:
    from arc_llama.agent.documents import _READERS, _clip

    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Expanded document must be under 50 MB")
    with tempfile.TemporaryDirectory(prefix="arc-chat-document-") as folder:
        path = Path(folder) / ("upload" + suffix)
        path.write_bytes(content)
        return _clip(_READERS[suffix](path))


def register_attachment_routes(app: FastAPI, *, require_admin: Callable[..., Any]) -> None:
    @app.post("/admin/parse-document")
    async def parse_document(file: UploadFile, _auth: None = Depends(require_admin)) -> dict[str, str]:
        from arc_llama.agent.tools import ToolError

        try:
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in {".docx", ".xlsx", ".pptx"}:
                raise HTTPException(status_code=400, detail="Supported documents: .docx, .xlsx and .pptx")
            content = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Document must be under 50 MB")
            text = await asyncio.to_thread(_extract, content, suffix)
            return {"filename": file.filename or "document", "text": text}
        except HTTPException:
            raise
        except ToolError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=422, detail="Could not read this Office document") from exc
        finally:
            await file.close()
