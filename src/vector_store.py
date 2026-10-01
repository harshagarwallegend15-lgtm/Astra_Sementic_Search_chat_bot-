"""FAISS-backed vector store with full provenance for every chunk.

Built on LangChain's FAISS integration so the retriever is a standard
LangChain component, with two additions this project needs:

* **Scores are returned as cosine similarity.** FAISS reports L2 distance;
  because every vector is L2-normalised on the way in, cosine similarity is
  exactly ``1 - d**2 / 2``. Converting here means the retrieval threshold
  config is expressed in a scale a human can reason about.
* **A readable JSONL manifest** sits next to the FAISS files. The index itself
  is a binary we cannot inspect; the manifest holds one JSON object per chunk
  and is what makes a citation traceable back to a page during debugging.
"""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import Settings, get_settings
from .embeddings import BaseEmbedder, get_embedder
from .errors import VectorStoreError
from .logging_utils import get_logger
from .models import Chunk, DocumentMeta

logger = get_logger(__name__)

INDEX_DIRNAME = "faiss_index"
MANIFEST_FILENAME = "chunks.jsonl"
DOCUMENTS_FILENAME = "documents.json"


def l2_to_cosine(squared_distance: float) -> float:
    """Convert a FAISS squared-L2 distance to cosine similarity.

    FAISS's ``IndexFlatL2.search`` returns **squared** Euclidean distances. For
    L2-normalised vectors ``u`` and ``v``:

        ||u - v||^2 = ||u||^2 + ||v||^2 - 2*u.v = 2 - 2*cos(u, v)

    so ``cos = 1 - d/2`` where ``d`` is the *squared* distance as returned.
    Squaring it again silently compresses every score towards zero, which would
    make the retrieval threshold meaningless.
    """
    return max(-1.0, min(1.0, 1.0 - float(squared_distance) / 2.0))


def chunk_embedding_text(chunk: Chunk) -> str:
    """Text actually handed to the embedding model.

    The document title and section heading are prepended even though they are
    not part of the body text. This is standard practice and it matters here:
    a question about "degree of autonomy" should match the chunk whose section
    is called *Degree of autonomy* even if that phrase is absent from the
    paragraph itself. The stored ``Chunk.text`` is left untouched, so what the
    user is shown as evidence is always verbatim document text.
    """
    header = chunk.title
    if chunk.section:
        header = f"{header} - {chunk.section}"
    return f"{header}\n\n{chunk.text}"


def chunk_to_document(chunk: Chunk) -> Document:
    """Represent a chunk as a LangChain document carrying its provenance.

    ``page_content`` is the *embedding* text (see above). Callers must recover
    the verbatim body via :meth:`VectorStore.search`, which resolves the
    ``chunk_id`` against the in-memory chunk table rather than trusting the
    document store.
    """
    return Document(
        page_content=chunk_embedding_text(chunk),
        metadata={
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "filename": chunk.filename,
            "title": chunk.title,
            "page": int(chunk.page),
            "section": chunk.section or "",
            "source": chunk.source,
            "is_reference": bool(chunk.is_reference),
            "char_start": int(chunk.char_start),
        },
    )


def document_to_chunk(document: Document) -> Chunk:
    """Inverse of :func:`chunk_to_document`."""
    meta = document.metadata
    return Chunk(
        chunk_id=meta["chunk_id"],
        document_id=meta["document_id"],
        filename=meta["filename"],
        title=meta["title"],
        page=int(meta["page"]),
        section=meta.get("section") or None,
        text=document.page_content,
        source=meta.get("source", "User upload"),
        char_start=int(meta.get("char_start", 0)),
        is_reference=bool(meta.get("is_reference", False)),
    )


def _derive_document_meta(chunk: Chunk) -> DocumentMeta:
    """Best-effort metadata when only chunks (not the ingest record) are known."""
    return DocumentMeta(
        document_id=chunk.document_id,
        filename=chunk.filename,
        title=chunk.title,
        page_count=chunk.page,
        char_count=chunk.char_count,
        chunk_count=1,
        source=chunk.source,
    )


class VectorStore:
    """Persistent store of chunk embeddings and their metadata."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: BaseEmbedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedder(self.settings)
        self.directory = Path(self.settings.vectorstore_dir)
        self.index_path = self.directory / INDEX_DIRNAME
        self.manifest_path = self.directory / MANIFEST_FILENAME
        self.documents_path = self.directory / DOCUMENTS_FILENAME
        self._store: FAISS | None = None
        self._chunks: dict[str, Chunk] = {}
        self._documents: dict[str, DocumentMeta] = {}
        self._lock = threading.RLock()

    # -- state -------------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return not self._chunks

    @property
    def chunk_count(self) -> int:
        with self._lock:
            return len(self._chunks)

    @property
    def document_count(self) -> int:
        with self._lock:
            return len(self._documents)

    def all_chunks(self) -> list[Chunk]:
        with self._lock:
            return list(self._chunks.values())

    def document_ids(self) -> set[str]:
        with self._lock:
            if self._documents:
                return set(self._documents)
            return {chunk.document_id for chunk in self._chunks.values()}

    # -- building ----------------------------------------------------------

    def build(
        self,
        chunks: Sequence[Chunk],
        *,
        reset: bool = True,
        documents: Sequence[DocumentMeta] | None = None,
    ) -> None:
        """Create the index from ``chunks``.

        Rebuilding from scratch is the default. Appending to a FAISS index that
        was written by an older embedding model silently produces meaningless
        neighbours, and a silent wrong answer is exactly what this project is
        built to avoid.
        """
        if not chunks:
            raise VectorStoreError("Cannot build an index from zero chunks.")
        with self._lock:
            if reset:
                self._store = None
                self._chunks = {}
                self._documents = {}
                self._remove_disk()
            try:
                store = FAISS.from_documents(
                    [chunk_to_document(chunk) for chunk in chunks],
                    self.embedder,
                )
            except Exception as exc:
                raise VectorStoreError(f"Failed to build the vector index: {exc}") from exc
            self._store = store
            self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
            self._merge_documents(chunks, documents)
            logger.info(
                "Built FAISS index with %d chunks using %s",
                len(chunks),
                self.embedder.name,
            )

    def add(
        self,
        chunks: Sequence[Chunk],
        documents: Sequence[DocumentMeta] | None = None,
    ) -> None:
        """Append ``chunks`` to an existing index."""
        if not chunks:
            return
        with self._lock:
            if self._store is None:
                # `build` resets by default, which is what we want here: there
                # is no live index to append to, so the new chunks become the
                # whole index rather than being merged into a stale manifest.
                self.build(chunks, documents=documents)
                return
            new_chunks = [c for c in chunks if c.chunk_id not in self._chunks]
            if not new_chunks:
                self._merge_documents(chunks, documents)
                return
            try:
                self._store.add_documents(
                    [chunk_to_document(chunk) for chunk in new_chunks]
                )
            except Exception as exc:
                raise VectorStoreError(f"Failed to add chunks to the index: {exc}") from exc
            self._chunks.update({c.chunk_id: c for c in new_chunks})
            self._merge_documents(new_chunks, documents)

    def _merge_documents(
        self,
        chunks: Sequence[Chunk],
        documents: Sequence[DocumentMeta] | None,
    ) -> None:
        """Record real ``DocumentMeta`` where supplied, else derive what we can."""
        for meta in documents or ():
            self._documents[meta.document_id] = meta
        for chunk in chunks:
            if chunk.document_id in self._documents:
                continue
            self._documents[chunk.document_id] = _derive_document_meta(chunk)
        for document_id, meta in self._documents.items():
            count = sum(
                1 for c in self._chunks.values() if c.document_id == document_id
            )
            if count and meta.chunk_count != count:
                meta.chunk_count = count

    def remove_document(self, document_id: str) -> int:
        """Drop every chunk belonging to ``document_id``.

        LangChain's FAISS wrapper has no delete API, so this removes the
        entries from the in-memory docstore and rebuilds the index from the
        surviving chunks. Documents are small, and a rebuild is far more
        reliable than hand-editing FAISS internals.
        """
        with self._lock:
            if not self._chunks:
                return 0
            survivors = [c for c in self._chunks.values() if c.document_id != document_id]
            removed = len(self._chunks) - len(survivors)
            if not removed:
                return 0
            surviving_docs = [
                meta
                for doc_id, meta in self._documents.items()
                if doc_id != document_id
            ]
            if not survivors:
                self.clear()
                return removed
            self.build(survivors, reset=True, documents=surviving_docs)
            return removed

    def clear(self) -> None:
        with self._lock:
            self._store = None
            self._chunks = {}
            self._documents = {}
            self._remove_disk()
            logger.info("Cleared the vector index")

    # -- searching ---------------------------------------------------------

    def search(
        self,
        query: str,
        k: int,
        *,
        document_ids: Iterable[str] | None = None,
        include_references: bool = False,
    ) -> list[tuple[Chunk, float]]:
        """Return ``k`` nearest chunks with cosine similarity scores."""
        with self._lock:
            if self._store is None or not self._chunks:
                raise VectorStoreError(
                    "The vector index is empty. Upload a document first."
                )
            # Over-fetch so that post-filtering still leaves `k` results.
            pool = max(k * 4, 20) if document_ids is not None else max(k, 1)
            try:
                hits = self._store.similarity_search_with_score(query, k=pool)
            except Exception as exc:
                raise VectorStoreError(f"Vector search failed: {exc}") from exc
            # Bind the chunk map that corresponds to *these* hits before
            # releasing the lock. Reading `self._chunks` after the lock is
            # dropped lets a concurrent `add`/`clear`/`build` rebind it, in
            # which case every hit fails to resolve and `search` returns []
            # - which the answerer reports as "the documents do not contain
            # that information". That is a confident, silently wrong refusal,
            # so the snapshot is taken here instead.
            chunk_map = self._chunks

        wanted = set(document_ids) if document_ids is not None else None
        results: list[tuple[Chunk, float]] = []
        for document, distance in hits:
            chunk = chunk_map.get(document.metadata.get("chunk_id", ""))
            if chunk is None:
                continue
            if wanted is not None and chunk.document_id not in wanted:
                continue
            if chunk.is_reference and not include_references:
                continue
            results.append((chunk, l2_to_cosine(distance)))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:k]

    # -- persistence -------------------------------------------------------

    def save(self) -> None:
        """Persist the index, the chunk manifest and document metadata."""
        with self._lock:
            if self._store is None:
                return
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                self._store.save_local(str(self.index_path))
                with self.manifest_path.open("w", encoding="utf-8") as handle:
                    for chunk in self._chunks.values():
                        handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
                self._write_documents()
            except Exception as exc:
                raise VectorStoreError(f"Could not save the vector index: {exc}") from exc
            logger.info("Saved vector index to %s", self.directory)

    def load(self) -> bool:
        """Load a previously saved index. Returns False if there is none."""
        with self._lock:
            if not (self.index_path / "index.faiss").exists() and not self.manifest_path.exists():
                return False
            try:
                if (self.index_path / "index.faiss").exists():
                    # The pickle is one this application wrote into its own
                    # data directory, never a file fetched from the internet.
                    self._store = FAISS.load_local(
                        str(self.index_path),
                        self.embedder,
                        allow_dangerous_deserialization=True,
                    )
                self._chunks = self._read_manifest()
                if not self._documents:
                    self._documents = {
                        item.document_id: item for item in self._read_document_file()
                    }
            except Exception as exc:
                raise VectorStoreError(
                    f"Could not read the vector index at {self.directory}: {exc}",
                    user_message=(
                        "The saved document index could not be read. Use "
                        "'Clear documents' and re-upload, or 'Rebuild index'."
                    ),
                ) from exc
            logger.info(
                "Loaded vector index: %d chunks from %s",
                len(self._chunks),
                self.directory,
            )
            return True

    def _read_manifest(self) -> dict[str, Chunk]:
        chunks: dict[str, Chunk] = {}
        if not self.manifest_path.exists():
            return chunks
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = Chunk.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    logger.warning(
                        "Skipping corrupt manifest line %d: %s", line_no, exc
                    )
                    continue
                chunks[chunk.chunk_id] = chunk
        return chunks

    def _write_documents(self) -> None:
        payload = [meta.to_dict() for meta in self._documents.values()]
        with self.documents_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _read_document_file(self) -> list[DocumentMeta]:
        if not self.documents_path.exists():
            return []
        try:
            payload = json.loads(self.documents_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read %s: %s", self.documents_path, exc)
            return []
        documents: list[DocumentMeta] = []
        for item in payload:
            try:
                documents.append(DocumentMeta.from_dict(item))
            except TypeError as exc:
                logger.warning("Skipping malformed document metadata: %s", exc)
        return documents

    def load_documents(self) -> list[DocumentMeta]:
        """Read the persisted document metadata list."""
        with self._lock:
            if self._documents:
                return list(self._documents.values())
            return self._read_document_file()

    def _remove_disk(self) -> None:
        if self.directory.exists():
            try:
                shutil.rmtree(self.directory)
            except OSError as exc:  # pragma: no cover - windows file locks
                logger.warning("Could not remove %s: %s", self.directory, exc)

    # -- diagnostics -------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "chunks": len(self._chunks),
                "documents": len(self._documents),
                "embedding_model": self.embedder.name,
                "semantic": self.embedder.is_semantic,
                "path": str(self.directory),
            }


__all__ = [
    "VectorStore",
    "chunk_to_document",
    "document_to_chunk",
    "l2_to_cosine",
]
