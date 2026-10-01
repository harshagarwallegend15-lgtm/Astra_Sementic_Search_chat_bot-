"""ASTRA INTEL - HTTP API and static frontend host.

The RAG pipeline never imported Streamlit, so it can be served over plain HTTP
instead. This module exposes exactly the operations ``app.py`` wired to widgets -
ask, evidence-only, upload, rebuild, clear, remove, briefings - and serves the
frontend from ``static/``.

Two front ends therefore coexist: the Streamlit app (default, no dependencies)
and this API plus a vanilla HTML/CSS/JS console on port 8502.

Run with::

    python -m uvicorn api:app --port 8502

or ``python api.py`` for the same thing with reload disabled.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402
from src.errors import AstraIntelError  # noqa: E402
from src.logging_utils import setup_logging  # noqa: E402
from src.models import AnswerResult, Citation, RetrievedChunk  # noqa: E402
from src.pipeline import AstraPipeline, IngestResult  # noqa: E402

STATIC_DIR = PROJECT_ROOT / "static"

logger = logging.getLogger("astra.api")


# --------------------------------------------------------------------------
# payloads
# --------------------------------------------------------------------------


class QuestionIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """Reject whitespace-only input here.

        ``min_length`` accepts ``"   "``, and without this the request would
        reach the model with an empty question - a retrieval and billing round
        trip to learn nothing. A blank question is a client error (422), not a
        provider failure.
        """
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be blank")
        return cleaned


def _as_dict(value: Any) -> Any:
    """Serialise the dataclasses in the result tree.

    ``dataclasses.asdict`` is avoided on purpose: ``RetrievedChunk`` holds a
    ``Chunk`` with a ``sections_detected`` style nesting that is cheaper and
    safer to flatten explicitly than to trust a recursive walk over.
    """
    if isinstance(value, RetrievedChunk):
        return {
            "score": round(value.score, 4),
            "dense_score": None if value.dense_score is None else round(value.dense_score, 4),
            "lexical_score": None if value.lexical_score is None else round(value.lexical_score, 4),
            "matched_terms": list(value.matched_terms),
            "chunk": {
                "chunk_id": value.chunk.chunk_id,
                "title": value.chunk.title,
                "filename": value.chunk.filename,
                "page": value.chunk.page,
                "section": value.chunk.section,
                "text": value.chunk.text,
            },
        }
    if isinstance(value, Citation):
        return {
            "marker": value.marker,
            "title": value.title,
            "filename": value.filename,
            "document_id": value.document_id,
            "page": value.page,
            "section": value.section,
            "excerpt": value.excerpt,
            "score": round(value.score, 4),
            "chunk_id": value.chunk_id,
        }
    if isinstance(value, AnswerResult):
        return {
            "question": value.question,
            "answer": value.answer,
            "status": value.status,
            "confidence": value.confidence,
            "notes": list(value.notes),
            "llm_model": value.llm_model,
            "latency_ms": value.latency_ms,
            "citations": [_as_dict(c) for c in value.citations],
            "retrieved": [_as_dict(r) for r in value.retrieved],
        }
    if isinstance(value, IngestResult):
        return {
            "added": [
                {
                    "document_id": m.document_id,
                    "title": m.title,
                    "page_count": m.page_count,
                    "chunk_count": m.chunk_count,
                }
                for m in value.added
            ],
            "duplicates": list(value.duplicates),
            "failures": dict(value.failures),
            "chunk_count": value.chunk_count,
            "elapsed_ms": value.elapsed_ms,
        }
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, dict):
        return {k: _as_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_dict(v) for v in value]
    return value


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------

app = FastAPI(
    title="ASTRA INTEL",
    description="Grounded question answering over indexed PDF documents.",
    version="1.0.0",
)

_pipeline: AstraPipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> AstraPipeline:
    """The one pipeline for the process; the index lives on disk.

    Sync handlers run in Starlette's threadpool, so two concurrent first
    requests would both see ``None``, both construct a pipeline, and both
    load ``data/vectorstore`` - the loser's store would be discarded
    half-read, and each store has its own lock so they do not exclude each
    other. The whole check-then-load-construct sequence is therefore
    serialised.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            settings = get_settings()
            setup_logging(settings.log_level)
            pipeline = AstraPipeline(settings)
            pipeline.load()
            _pipeline = pipeline
    return _pipeline


def reset_pipeline() -> None:
    """Drop the cached pipeline so the next call rebuilds it.

    Used by tests that swap the on-disk index out from under the process, and
    available to callers that mutate ``VECTORSTORE_DIR`` at runtime.
    """
    global _pipeline
    with _pipeline_lock:
        _pipeline = None


def _state() -> dict[str, Any]:
    pipeline = get_pipeline()
    llm = pipeline.llm()
    settings = pipeline.settings
    documents = pipeline.documents()

    # Per-page passage counts, so the console can draw the corpus grid (the
    # waste-dashboard "map" analogue: where in the corpus the material sits).
    # Computed here rather than in the browser because the store is the only
    # thing that knows the real distribution.
    register: list[dict[str, Any]] = []
    for meta in documents:
        histogram: dict[int, int] = {}
        for chunk in pipeline.chunks_for(meta.document_id):
            histogram[chunk.page] = histogram.get(chunk.page, 0) + 1
        entry = _as_dict(meta)
        entry["pages"] = [
            {"page": page, "passages": histogram[page]}
            for page in range(1, (meta.page_count or 0) + 1)
        ]
        entry["indexed_pages"] = len(histogram)
        register.append(entry)

    return {
        "stats": _as_dict(pipeline.stats()),
        "documents": register,
        "settings": settings.public_dict(),
        "llm_ready": llm is not None,
    }


@app.get("/api/state")
def get_state() -> dict[str, Any]:
    return _state()


@app.post("/api/ask")
def ask(payload: QuestionIn) -> dict[str, Any]:
    pipeline = get_pipeline()
    try:
        result = pipeline.answerer().answer(payload.question)
    except AstraIntelError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc
    except Exception as exc:  # noqa: BLE001 - never leak a traceback to the client
        logger.exception("answer failed")
        raise HTTPException(status_code=500, detail=f"Could not answer: {exc}") from exc
    return _as_dict(result)


@app.post("/api/evidence")
def evidence(payload: QuestionIn) -> dict[str, Any]:
    pipeline = get_pipeline()
    try:
        citations = pipeline.answerer(None).evidence(payload.question)
    except AstraIntelError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("evidence failed")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc
    return {"question": payload.question, "citations": _as_dict(citations)}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    payload: list[tuple[str, bytes]] = []
    for item in files:
        payload.append((item.filename or "upload.pdf", await item.read()))
    if not payload:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    pipeline = get_pipeline()
    try:
        result = pipeline.ingest_pdfs(payload)
    except AstraIntelError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("ingest failed")
        raise HTTPException(status_code=500, detail=f"Could not index those files: {exc}") from exc
    return _as_dict(result)


@app.post("/api/rebuild")
def rebuild() -> dict[str, Any]:
    pipeline = get_pipeline()
    try:
        result = pipeline.index_starter_documents(rebuild=True)
    except AstraIntelError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("rebuild failed")
        raise HTTPException(status_code=500, detail=f"Could not rebuild: {exc}") from exc
    return _as_dict(result)


@app.post("/api/clear")
def clear() -> dict[str, Any]:
    get_pipeline().clear_index()
    return {"ok": True}


@app.delete("/api/documents/{document_id}")
def remove(document_id: str) -> dict[str, Any]:
    removed = get_pipeline().remove_document(document_id)
    return {"removed": removed}


@app.post("/api/briefings")
def briefings() -> dict[str, Any]:
    pipeline = get_pipeline()
    if pipeline.llm() is None:
        raise HTTPException(status_code=503, detail="No language model is configured.")
    try:
        return {"summaries": _as_dict(pipeline.summaries())}
    except AstraIntelError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("briefing failed")
        raise HTTPException(status_code=500, detail=f"Could not generate briefings: {exc}") from exc


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@app.get("/api/suggestions")
def suggestions() -> dict[str, Any]:
    """The quick-pick questions, read from ``example-questions.md``.

    Served rather than duplicated into ``static/``, so the console and the
    evaluation script can never disagree about the question list.
    """
    import re

    path = PROJECT_ROOT / "example-questions.md"
    if not path.is_file():
        return {"questions": []}
    found: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\d+[\.\)]\s+(.*\S)\s*$", line.strip())
        if match:
            found.append(match.group(1).strip().strip('"'))
    return {"questions": found}


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8502, log_level="info")