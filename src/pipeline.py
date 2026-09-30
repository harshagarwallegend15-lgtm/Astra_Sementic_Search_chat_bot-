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
from typing import Sequence

from .chunker import Chunker
from .config import Settings, get_settings
from .embeddings import get_embedder
from .errors import (
    AstraIntelError,
    DuplicateDocumentError,
    NoDocumentsError,
)
from .logging_utils import get_logger
from .models import Chunk, DocumentMeta
from .pdf_processor import PDFProcessor, load_metadata_catalog, metadata_for
from .vector_store import VectorStore

logger = get_logger(__name__)


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
        self._llm_failed = False

    def llm(self):
        """Return a cached LLM client, building it on first use.

        Built lazily and cached because constructing a client opens a network
        session, and ingestion does not need one. A provider that cannot be
        reached is remembered as failed so a misconfigured deployment degrades
        to evidence-only answers instead of retrying and stalling on every
        question. Pass an explicit client to ``answerer()`` to override.
        """
        if self._llm is not None or self._llm_failed:
            return self._llm
        try:
            from .llm import build_llm_client

            self._llm = build_llm_client(self.settings)
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the UI
            logger.warning("LLM unavailable (%s); answers will be evidence-only.", exc)
            self._llm_failed = True
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
                result.failures[filename] = f"Unexpected error: {exc}"
                logger.exception("Unexpected failure ingesting %s", filename)

        if new_chunks:
            self.store.add(new_chunks, documents=documents)
            self.store.save()
            result.added = documents
        elif result.failures and not result.duplicates:
            raise AstraIntelError("; ".join(result.failures.values()) or "Ingestion failed.")

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

    def summaries(self, llm=None):
        from .summarizer import summarise_documents

        self.ensure_ready()
        payload = [
            (meta.document_id, meta.title, self.chunks_for(meta.document_id))
            for meta in self.documents()
        ]
        return summarise_documents(
            self.settings, self.llm() if llm is None else llm, payload
        )
