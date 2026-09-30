"""Typed data structures shared across the pipeline.

These dataclasses are the contract between stages. In particular they encode
the two invariants the whole system depends on:

1. A :class:`Chunk` always carries an exact ``page`` number and a ``section``
   that was *observed* in the PDF (``None`` when no heading could be detected).
2. A :class:`Citation` is always derived from a real :class:`Chunk`, never from
   free text produced by a model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Sequence

# ---------------------------------------------------------------------------
# Ingestion types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextLine:
    """A single visual line of text as laid out on a page.

    ``size`` and ``bold`` are what make heading detection possible: in a
    rendered article the body is a modal font size and headings are strictly
    larger and bold.
    """

    text: str
    size: float
    bold: bool
    x0: float
    y0: float

    @property
    def is_blank(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True)
class SectionHeading:
    """A heading observed in the document."""

    title: str
    level: int
    page: int
    #: 1-based order of the heading within the document.
    ordinal: int


@dataclass
class PageContent:
    """Extracted content of a single page, in reading order."""

    page: int
    lines: list[TextLine] = field(default_factory=list)
    headings: list[SectionHeading] = field(default_factory=list)
    #: Body font size used as the "this is normal text" baseline for the doc.
    body_size: float = 0.0
    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines if not line.is_blank)

    @property
    def char_count(self) -> int:
        return sum(len(line.text) for line in self.lines)


@dataclass
class DocumentMeta:
    """Metadata about one ingested document.

    ``document_id`` is derived from the file *content*, which gives us
    duplicate detection for free: re-uploading the same PDF produces the same
    id no matter what it is renamed to.
    """

    document_id: str
    filename: str
    title: str
    page_count: int
    char_count: int = 0
    chunk_count: int = 0
    source: str = "User upload"
    license: str | None = None
    url: str | None = None
    retrieved: str | None = None
    added_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    #: True when the body font size could be established and headings found.
    sections_detected: bool = False
    headings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "title": self.title,
            "page_count": self.page_count,
            "char_count": self.char_count,
            "chunk_count": self.chunk_count,
            "source": self.source,
            "license": self.license,
            "url": self.url,
            "retrieved": self.retrieved,
            "added_at": self.added_at,
            "sections_detected": self.sections_detected,
            "headings": self.headings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentMeta":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Chunking / retrieval types
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A retrievable unit of document text with full provenance.

    ``page`` is guaranteed to be the PDF page the text was physically printed
    on (1-based). Chunks never span pages, which is what makes that guarantee
    hold.
    """

    chunk_id: str
    document_id: str
    filename: str
    title: str
    page: int
    text: str
    section: str | None = None
    source: str = "User upload"
    char_start: int = 0
    #: Set for chunks that live in a bibliography-style section; excluded from
    #: retrieval by default because they are URLs and citations, not content.
    is_reference: bool = False

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "title": self.title,
            "page": self.page,
            "section": self.section,
            "text": self.text,
            "source": self.source,
            "char_start": self.char_start,
            "is_reference": self.is_reference,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class RetrievedChunk:
    """A chunk plus the scores that got it selected."""

    chunk: Chunk
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    #: Terms from the question that this chunk actually covers.
    matched_terms: tuple[str, ...] = ()

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def label(self) -> str:
        """Human-readable provenance, e.g. ``Electronic warfare - p.3``."""
        return f"{self.chunk.title} - p.{self.chunk.page}"


@dataclass
class Citation:
    """A source the user can go and verify.

    Built exclusively from :class:`Chunk` metadata. Page numbers cannot be
    invented because they are never generated by a model.
    """

    chunk_id: str
    document_id: str
    title: str
    filename: str
    page: int
    section: str | None
    excerpt: str
    score: float = 0.0
    #: The ``[S1]``-style marker the model used, resolved back to this chunk.
    marker: str = ""

    @property
    def label(self) -> str:
        return f"{self.title} - p.{self.page}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "filename": self.filename,
            "page": self.page,
            "section": self.section,
            "excerpt": self.excerpt,
            "score": self.score,
            "marker": self.marker,
        }


# ---------------------------------------------------------------------------
# Answer types
# ---------------------------------------------------------------------------

GroundingStatus = Literal["grounded", "partially_grounded", "ungrounded", "no_context"]


@dataclass
class AnswerResult:
    """The complete, citable result of one question."""

    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    status: GroundingStatus = "ungrounded"
    #: 0..1 confidence derived from retrieval scores, shown in the UI.
    confidence: float = 0.0
    #: Populated when the answer was refused or only partly supported.
    notes: list[str] = field(default_factory=list)
    #: Chunks handed to the model, including any it did not end up citing.
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    llm_model: str | None = None
    latency_ms: int | None = None

    @property
    def is_grounded(self) -> bool:
        return self.status in ("grounded", "partially_grounded")

    @property
    def documents_cited(self) -> list[str]:
        seen: list[str] = []
        for c in self.citations:
            if c.title not in seen:
                seen.append(c.title)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "status": self.status,
            "confidence": round(self.confidence, 3),
            "citations": [c.to_dict() for c in self.citations],
            "notes": self.notes,
            "llm_model": self.ll_model,
            "latency_ms": self.latency_ms,
        }


@dataclass
class DocumentSummary:
    """Grounded summary of a single document."""

    document_id: str
    title: str
    summary: str
    topics: list[str] = field(default_factory=list)
    #: Pages the summary drew from, for transparency.
    pages: list[int] = field(default_factory=list)
    #: False when no LLM was available and a structural fallback was used.
    llm_generated: bool = False
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "summary": self.summary,
            "topics": self.topics,
            "pages": self.pages,
            "llm_generated": self.llm_generated,
            "model": self.model,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_document_id(data: bytes) -> str:
    """Content-addressed document id.

    Using the hash of the bytes (not the filename) means the same PDF uploaded
    twice under two different names is detected as a duplicate.
    """
    return hashlib.sha256(data).hexdigest()[:16]


def make_chunk_id(document_id: str, page: int, index: int) -> str:
    """Human-readable, stable chunk id such as ``a1b2c3d4e5f6g789-p003-02``."""
    return f"{document_id}-p{page:03d}-{index:02d}"


def normalise_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces."""
    return " ".join(text.split())


def truncate(text: str, limit: int, suffix: str = "...") -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))].rstrip() + suffix


def dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


__all__ = [
    "TextLine",
    "SectionHeading",
    "PageContent",
    "DocumentMeta",
    "Chunk",
    "RetrievedChunk",
    "Citation",
    "AnswerResult",
    "DocumentSummary",
    "GroundingStatus",
    "make_document_id",
    "make_chunk_id",
    "normalise_whitespace",
    "truncate",
    "dedupe_preserve_order",
]
