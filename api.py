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
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

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


def _internal_failure(action: str, exc: BaseException) -> HTTPException:
    """Build a 500 that is safe to return to a client.

    ``str(exc)`` is deliberately not forwarded. These handlers catch bare
    ``Exception``, so the stringification can carry provider request metadata,
    filesystem paths, ``KeyError('api_key')`` and similar internals. The real
    cause is logged with a traceback server-side and correlated to the client
    by an id, so a report can still be matched to a log line.
    """
    ref = uuid.uuid4().hex[:12]
    logger.exception("%s failed (ref=%s)", action, ref)
    return HTTPException(
        status_code=500,
        detail=f"{action} failed. Reference: {ref}",
    )


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
    # thing that knows the real distribution, and in one pass because a
    # per-document query would rescan the whole corpus per document.
    histograms = pipeline.page_histograms()
    register: list[dict[str, Any]] = []
    for meta in documents:
        histogram = histograms.get(meta.document_id, {})
        entry = _as_dict(meta)
        entry["pages"] = [
            {"page": page, "passages": histogram.get(page, 0)}
            for page in range(1, (meta.page_count or 0) + 1)
        ]
        entry["indexed_pages"] = len(histogram)
        register.append(entry)

    return {
        "stats": _as_dict(pipeline.stats()),
        "documents": register,
        # `settings` is the human-readable dump (labels like "LLM model") meant
        # for display. `runtime` is the same information under stable
        # snake_case keys for programmatic use, so the console never has to
        # match on a display string. Neither block carries the API key:
        # public_dict reports only whether one is configured.
        "settings": settings.public_dict(),
        "runtime": {
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "llm_base_url": settings.resolved_base_url,
            "embedding_model": settings.embedding_model,
            "llm_ready": llm is not None,
        },
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
        raise _internal_failure("Answering the question", exc) from exc
    return _as_dict(result)


@app.post("/api/evidence")
def evidence(payload: QuestionIn) -> dict[str, Any]:
    pipeline = get_pipeline()
    try:
        citations = pipeline.answerer(None).evidence(payload.question)
    except AstraIntelError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc
    except Exception as exc:  # noqa: BLE001
        raise _internal_failure("Retrieving evidence", exc) from exc
    return {"question": payload.question, "citations": _as_dict(citations)}


# Upload guardrails. Both limits are enforced while the body is still being
# read, not after it has been buffered into memory.
MAX_UPLOAD_FILES = 20
CHUNK_BYTES = 1024 * 1024


async def _close_files(files: list[UploadFile]) -> None:
    """Release every uploaded file's spool handle, ignoring cleanup errors."""
    for item in files:
        try:
            await item.close()
        except Exception:  # noqa: BLE001 - cleanup must not mask the real error
            pass


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Ingest PDFs.

    Declared ``async`` because the files must be streamed. The blocking work
    (PDF extraction, embedding, FAISS build, save) is handed to a thread so it
    cannot stall the event loop and with it every other request; when that
    used to run inline in the coroutine, /api/health stalled for the whole
    duration of an ingest.
    """
    if len(files) > MAX_UPLOAD_FILES:
        # Reject before touching the body, but the files Starlette already
        # spooled still hold open handles, so close them.
        await _close_files(files)
        raise HTTPException(
            status_code=400,
            detail=f"Too many files: {len(files)}. Upload at most {MAX_UPLOAD_FILES} at once.",
        )

    limit = get_settings().max_upload_mb * 1024 * 1024
    payload: list[tuple[str, bytes]] = []
    total = 0
    try:
        for item in files:
            name = item.filename or "upload.pdf"
            buffer = bytearray()
            while True:
                block = await item.read(CHUNK_BYTES)
                if not block:
                    break
                total += len(block)
                if total > limit:
                    # Stop early rather than after the whole body is resident.
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Upload exceeds the {get_settings().max_upload_mb} MB limit. "
                            "Split the batch and retry."
                        ),
                    )
                buffer.extend(block)
            payload.append((name, bytes(buffer)))
    except HTTPException:
        # Every UploadFile spools to a temp file; abandoning them without
        # closing leaks a handle per file and, on Windows, can leave the
        # %TEMP% artefact locked until the garbage collector runs.
        await _close_files(files)
        raise

    if not payload:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    pipeline = get_pipeline()
    try:
        result = await run_in_threadpool(pipeline.ingest_pdfs, payload)
    except AstraIntelError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc
    except Exception as exc:  # noqa: BLE001
        raise _internal_failure("Indexing those files", exc) from exc
    return _as_dict(result)


@app.post("/api/rebuild")
def rebuild(confirm: str = "") -> dict[str, Any]:
    # Destructive: this discards every user-uploaded document along with the
    # starter set. Requiring the confirmation token server-side means the
    # control cannot be triggered by a stray request, a prefetch, or a
    # bookmarked GET.
    if confirm != "rebuild":
        raise HTTPException(
            status_code=400,
            detail="Rebuilding discards every indexed document. Pass confirm=rebuild.",
        )
    pipeline = get_pipeline()
    try:
        result = pipeline.index_starter_documents(rebuild=True)
    except AstraIntelError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc
    except Exception as exc:  # noqa: BLE001
        raise _internal_failure("Rebuilding the index", exc) from exc
    return _as_dict(result)


@app.post("/api/clear")
def clear(confirm: str = "") -> dict[str, Any]:
    if confirm != "clear":
        raise HTTPException(
            status_code=400,
            detail="Clearing destroys the entire index. Pass confirm=clear.",
        )
    try:
        get_pipeline().clear_index()
    except AstraIntelError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc
    except Exception as exc:  # noqa: BLE001
        raise _internal_failure("Clearing the index", exc) from exc
    return {"ok": True}


@app.delete("/api/documents/{document_id}")
def remove(document_id: str) -> dict[str, Any]:
    try:
        removed = get_pipeline().remove_document(document_id)
    except AstraIntelError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc
    except Exception as exc:  # noqa: BLE001
        raise _internal_failure(f"Removing document {document_id}", exc) from exc
    if not removed:
        # Deleting nothing is a client mistake, not a silent success: the
        # console refetches the register either way, and a 200 would suggest
        # the document was removed when it never existed.
        raise HTTPException(
            status_code=404,
            detail=f"No indexed document has the id {document_id!r}.",
        )
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
        raise _internal_failure("Generating briefings", exc) from exc


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Report real readiness, not just that the process is alive.

    The old body was ``{"ok": true}``, which stayed green while the index was
    unreadable and the answer engine dead - the console showed a live system
    that could not answer a single question. Each dependency is probed and the
    HTTP status reflects the worst outcome: ``200`` when the service can serve
    questions, ``503`` when it cannot.
    """
    checks: dict[str, Any] = {}
    degraded: list[str] = []

    try:
        pipeline = get_pipeline()
        stats = pipeline.stats()
        checks["index"] = {
            "ok": bool(stats.get("chunks")),
            "chunks": stats.get("chunks", 0),
            "documents": stats.get("documents", 0),
            "embedding_model": pipeline.settings.embedding_model,
        }
        if not stats.get("chunks"):
            degraded.append("index")
    except Exception as exc:  # noqa: BLE001 - a probe must never raise
        logger.error("Health probe failed on the index: %s", exc)
        checks["index"] = {"ok": False, "error": type(exc).__name__}
        degraded.append("index")

    try:
        pipeline = get_pipeline()
        llm = pipeline.llm()
        # `llm()` builds the client but sends nothing, so this proves
        # configuration and reachability of the SDK, not the provider.
        checks["llm"] = {
            "ok": llm is not None,
            "provider": pipeline.settings.llm_provider,
            "model": pipeline.settings.llm_model,
            "detail": None if llm is not None else "client unavailable",
        }
        if llm is None:
            # Answers still work in evidence-only mode, so this is reported
            # without failing the whole check.
            checks["llm"]["degraded"] = True
    except Exception as exc:  # noqa: BLE001
        logger.error("Health probe failed on the language model: %s", exc)
        checks["llm"] = {"ok": False, "error": type(exc).__name__, "degraded": True}

    body = {
        "status": "degraded" if degraded else "ok",
        "checks": checks,
        "index_ready": checks["index"].get("ok", False),
        "llm_ready": checks["llm"].get("ok", False),
    }
    if degraded:
        # JSONResponse rather than HTTPException so the payload keeps the same
        # shape whether the service is ready or not; a probe should not have to
        # unwrap `detail` to find out what is broken.
        return JSONResponse(status_code=503, content=body)
    return body


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