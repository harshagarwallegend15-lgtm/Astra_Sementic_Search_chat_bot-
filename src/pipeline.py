"""Ingestion pipeline.

Owns the lifecycle of the index: load a persisted FAISS store or build one from
scratch, ingest uploaded PDFs (content-addressed, so re-uploading the same bytes
is a no-op), keep the on-disk store in sync, and expose a single object that the
UI and evaluation scripts both use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Sequence

from .chunker import Chunker
from .config import Settings, get_settings
from .embeddings import get_embedder
from .errors import (
    AstraIntelError,
    DocumentError,
    DuplicateDocumentError,
    NoDocumentsError,
)
from .logging_utils import get_logger
from .models import Chunk, DocumentMeta
from .pdf_processor import PDFProcessor, load_metadata_catalog, metadata_for
from .vector_store import VectorStore

logger = get_logger(__name__)

# How long a failed language-model build is cached before another is attempted.
# Long enough that a hard misconfiguration does not retry on every question,
# short enough that a transient outage heals without restarting the process.
LLM_RETRY_COOLDOWN_S = 60.0


@dataclass
class IngestResult:
    """Outcome of ingesting a batch of PDFs."""

    added: list[DocumentMeta] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    chunk_count: int = 0
    elapsed_ms: int = 0
    reused_cache: bool = False

    @property
    def added_count(self) -> int:
        return len(self.added)


class AstraPipeline:
    """Extracts, chunks, embeds and indexes PDFs; answers grounded questions."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_directories()
        self.processor = PDFProcessor(self.settings)
        self.chunker = Chunker(self.settings)
        self.embedder = get_embedder(self.settings)
        self.store = VectorStore(self.settings, self.embedder)
        self.retriever = None
        self._catalog = load_metadata_catalog(self.settings.sample_metadata_path)
        self._loaded = False
        self._llm = None
        self._llm_failed_until: float | None = None
        self._llm_lock = Lock()

    def llm(self):
        """Return a cached LLM client, building it on first use.

        Built lazily and cached because constructing a client opens a network
        session, and ingestion does not need one. A provider that cannot be
        built is remembered as failed so a misconfigured deployment degrades
        to evidence-only answers instead of retrying and stalling on every
        question.

        The failure record is *not* permanent: it carries a cooldown so a
        transient fault (DNS blip, rate limit, provider restart) heals on its
        own without a restart, while a genuine misconfiguration is not retried
        on every single question. Pass an explicit client to ``answerer()`` to
        override.

        Building is guarded by a lock because the API serves requests on a
        thread pool: without it a burst of concurrent questions each opens its
        own client (and each retries a provider that is already known to be
        down), and a single successful build could be discarded by a slower
        failing sibling.
        """
        client = self._llm
        if client is not None:
            return client
        with self._llm_lock:
            if self._llm is not None:
                return self._llm
            if self._llm_failed_until is not None:
                if time.monotonic() < self._llm_failed_until:
                    return None
                logger.info(
                    "Retrying the language model after its cooldown expired."
                )
                self._llm_failed_until = None
            try:
                from .llm import build_llm_client

                self._llm = build_llm_client(self.settings)
                self._llm_failed_until = None
            except Exception as exc:  # noqa: BLE001 - degrade, never crash the UI
                self._llm_failed_until = time.monotonic() + LLM_RETRY_COOLDOWN_S
                logger.warning(
                    "LLM unavailable (%s); answers will be evidence-only for %.0fs.",
                    exc,
                    LLM_RETRY_COOLDOWN_S,
                )
                self._llm = None
            return self._llm

    # -- lifecycle ----------------------------------------------------------

    def load(self) -> bool:
        """Load the persisted index. Returns True when chunks are available."""
        if self._loaded:
            return not self.store.is_empty
        if self.store.load():
            self._build_retriever()
            self._loaded = True
            logger.info("Loaded persisted index: %d chunks", self.store.chunk_count)
            return not self.store.is_empty
        self._loaded = True
        return False

    def ensure_ready(self) -> None:
        if not self._loaded:
            self.load()
        if self.store.is_empty:
            self._build_retriever()

    def _build_retriever(self) -> None:
        from .retriever import Retriever

        self.retriever = Retriever(self.store, self.settings)

    # -- ingestion ----------------------------------------------------------

    def ingest_pdfs(
        self,
        files: Sequence[tuple[str, bytes]],
        rebuild: bool = False,
    ) -> IngestResult:
        """Ingest ``(filename, bytes)`` pairs into the index.

        ``rebuild`` discards any existing index first, which is what the
        "rebuild index" control uses when the embedding model has changed.
        """
        if not files:
            raise NoDocumentsError("No files were supplied for ingestion.")

        started = time.perf_counter()
        if rebuild:
            self.store.clear()
        self.load()

        result = IngestResult()
        existing = self.store.document_ids()
        new_chunks: list[Chunk] = []
        documents: list[DocumentMeta] = []

        for filename, data in files:
            try:
                extracted = self.processor.process(
                    data, filename, **metadata_for(filename, self._catalog)
                )
                meta = extracted.meta
                if meta.document_id in existing:
                    result.duplicates.append(meta.title)
                    logger.info("Skipping duplicate: %s", meta.title)
                    continue
                chunks = self.chunker.chunk_document(extracted)
                if not chunks:
                    result.failures[filename] = "No usable text was found."
                    continue
                meta.chunk_count = len(chunks)
                documents.append(meta)
                new_chunks.extend(chunks)
                existing.add(meta.document_id)
            except AstraIntelError as exc:
                result.failures[filename] = exc.user_message
                logger.warning("Could not ingest %s: %s", filename, exc)
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
                # Not surfaced verbatim: this text reaches the client through
                # the raised AstraIntelError, and an unexpected exception can
                # carry paths or provider internals.
                result.failures[filename] = (
                    f"{filename} could not be read. Check it is a valid, "
                    "unencrypted PDF."
                )
                logger.exception("Unexpected failure ingesting %s", filename)

        if new_chunks:
            self.store.add(new_chunks, documents=documents)
            self.store.save()
            result.added = documents

        # Any per-file failure is surfaced, even when other files succeeded or
        # were duplicates. Previously a batch mixing one duplicate with one
        # broken file fell through every branch and returned HTTP 200 with the
        # failure buried in the payload, so the console reported a cheerful
        # "Indexed 0 new documents" and the broken PDF vanished silently.
        if result.failures:
            # The values are already the curated `user_message` of each
            # per-file error (never a traceback), so joining them is safe to
            # render. Raising the bare `AstraIntelError` here discarded them
            # in favour of the generic base-class fallback, so a corrupt PDF
            # surfaced as "Something went wrong" instead of naming the file
            # and saying it could not be read.
            detail = "; ".join(
                f"{name}: {reason}" for name, reason in sorted(result.failures.items())
            )
            raise DocumentError(
                f"Some files could not be indexed: {detail}",
                user_message=f"Some files could not be indexed. {detail}",
            )

        self._build_retriever()
        result.chunk_count = self.store.chunk_count
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "Ingestion finished: +%d doc(s), +%d chunk(s), total %d",
            result.added_count,
            len(new_chunks),
            result.chunk_count,
        )
        return result

    def index_starter_documents(self, rebuild: bool = False) -> IngestResult:
        """Ingest the bundled ``sample-documents`` PDFs."""
        directory = Path(self.settings.sample_documents_dir)
        if not directory.is_dir():
            raise NoDocumentsError(
                f"Starter documents not found at {directory}."
            )
        files: list[tuple[str, bytes]] = []
        for path in sorted(directory.glob("*.pdf")):
            files.append((path.name, path.read_bytes()))
        if not files:
            raise NoDocumentsError(f"No PDFs found in {directory}.")
        return self.ingest_pdfs(files, rebuild=rebuild)

    def remove_document(self, document_id: str) -> int:
        removed = self.store.remove_document(document_id)
        self.store.save()
        self._build_retriever()
        return removed

    def clear_index(self) -> None:
        self.store.clear()
        self._build_retriever()

    # -- inspection ---------------------------------------------------------

    def documents(self) -> list[DocumentMeta]:
        return self.store.load_documents()

    def page_histograms(self) -> dict[str, dict[int, int]]:
        """Passage counts per page, keyed by document id.

        One pass over the corpus for all documents. Computing this per
        document instead means each document copies the entire chunk map, so
        the cost was O(documents x corpus) - fine at three documents, not at a
        few hundred.
        """
        histograms: dict[str, dict[int, int]] = {}
        for chunk in self.store.all_chunks():
            pages = histograms.setdefault(chunk.document_id, {})
            pages[chunk.page] = pages.get(chunk.page, 0) + 1
        return histograms

    def chunks_for(self, document_id: str) -> list[Chunk]:
        return [
            chunk
            for chunk in self.store.all_chunks()
            if chunk.document_id == document_id
        ]

    def stats(self) -> dict[str, object]:
        stats = dict(self.store.stats())
        stats["documents"] = len(self.documents())
        stats["llm_provider"] = self.settings.llm_provider
        stats["llm_model"] = self.settings.llm_model
        stats["embedding_model"] = self.settings.embedding_model
        return stats

    # -- answering ----------------------------------------------------------

    def answerer(self, llm=None):
        from .answerer import Answerer

        self.ensure_ready()
        return Answerer(self.retriever, self.llm() if llm is None else llm, self.settings)

    def summaries(self, llm=None, document_id: str | None = None):
        """Summarise every indexed document, or just one.

        ``document_id`` lets the console brief a single document from its row
        instead of regenerating the whole set. Without it an operator wanting
        one summary pays for all of them, and waits for all of them.
        """
        from .summarizer import summarise_documents

        self.ensure_ready()
        payload = [
            (meta.document_id, meta.title, self.chunks_for(meta.document_id))
            for meta in self.documents()
        ]
        if document_id is not None:
            payload = [row for row in payload if row[0] == document_id]
        return summarise_documents(
            self.settings, self.llm() if llm is None else llm, payload
        )
