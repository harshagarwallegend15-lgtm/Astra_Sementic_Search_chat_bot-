"""Structural chunking that preserves page numbers and section names.

Two invariants drive the design, and both come straight from the challenge's
citation requirements:

**1. A chunk never spans a page.** Therefore ``Chunk.page`` is always the exact
PDF page the text was printed on, and a citation can never be "roughly right".
The cost is slightly smaller chunks at page boundaries; the benefit is that
"Page 6" in the UI is a fact rather than an estimate.

**2. A chunk belongs to exactly one section.** A heading stack is maintained as
pages are walked in order, so a paragraph that continues across a page break
keeps the section it started in, and a section that starts mid-page is
attributed from that point on.

Within those constraints, chunks are built from whole paragraph blocks wherever
possible. A block that is on its own too large for the target size is split at
sentence boundaries, and consecutive chunks in the same section share a small
overlap so a fact that straddles a split is still retrievable from one chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .config import Settings, get_settings
from .logging_utils import get_logger
from .models import (
    Chunk,
    PageContent,
    SectionHeading,
    make_chunk_id,
    normalise_whitespace,
)
from .pdf_processor import ExtractedDocument, RawBlock, _is_reference_title

logger = get_logger(__name__)

#: Sentences end here. Kept deliberately simple and conservative.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\u201c])")

#: An uppercase acronym followed by a full stop, e.g. "...the U.S. Army".
_ABBREVIATION_TAIL_RE = re.compile(r"\b[A-Z]\.$")

#: A line ending in a hyphen followed by a lowercase letter is almost always a
#: word broken across a line, not a real compound.
_HYPHEN_BREAK_RE = re.compile(r"(\w)-$")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, keeping terminal punctuation.

    A deliberately simple splitter: PDF text does not justify a statistical
    segmenter, and the goal is only to avoid cutting mid-clause.
    """
    if not text.strip():
        return []
    parts: list[str] = []
    for candidate in _SENTENCE_END_RE.split(text):
        candidate = candidate.strip()
        if not candidate:
            continue
        # Re-join fragments that were really just "U.S." or "Dr." splits.
        if parts and _ABBREVIATION_TAIL_RE.search(parts[-1]):
            parts[-1] = f"{parts[-1]} {candidate}"
        else:
            parts.append(candidate)
    return parts or [text.strip()]


def join_lines(lines: Sequence[str]) -> str:
    """Join visual lines into flowing text, undoing line-break hyphenation."""
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
            continue
        match = _HYPHEN_BREAK_RE.search(out)
        if match and line[:1].islower():
            # "electro-" + "magnetic" -> "electromagnetic"
            out = out[:-1] + line
        else:
            out = f"{out} {line}"
    return out.strip()


@dataclass
class _SectionState:
    """The heading stack at a point in the document."""

    #: (level, title) pairs, deepest last.
    titles: list[tuple[int, str]] = field(default_factory=list)
    is_reference: bool = False

    def push(self, heading: SectionHeading) -> None:
        # Pop headings at the same or deeper level: they are siblings, not children.
        while self.titles and self.titles[-1][0] >= heading.level:
            self.titles.pop()
        self.titles.append((heading.level, heading.title))
        if _is_reference_title(heading.title):
            self.is_reference = True

    def pop_to_before(self, level: int) -> None:
        """Reset the reference flag when leaving a reference section."""
        while self.titles and self.titles[-1][0] >= level:
            self.titles.pop()
        if self.titles:
            self.is_reference = _is_reference_title(self.titles[-1][1])

    @property
    def current(self) -> str | None:
        """Deepest heading title, which is the most specific section name."""
        return self.titles[-1][1] if self.titles else None

    @property
    def path(self) -> str | None:
        """Full heading path, e.g. ``Classification types > Degree of autonomy``."""
        if not self.titles:
            return None
        return " > ".join(title for _, title in self.titles)


class Chunker:
    """Converts an :class:`ExtractedDocument` into retrievable chunks."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.chunk_size = self.settings.chunk_size
        self.chunk_overlap = self.settings.chunk_overlap
        self.min_chars = self.settings.min_chunk_chars

    # -- public API --------------------------------------------------------

    def chunk_document(self, document: ExtractedDocument) -> list[Chunk]:
        """Split one document into page-bounded, section-tagged chunks."""
        meta = document.meta
        chunks: list[Chunk] = []
        state = _SectionState()
        per_page_index: dict[int, int] = {}

        for page, blocks in self._blocks_by_page(document):
            for block in blocks:
                if self._is_heading(page, block):
                    heading = self._heading_for(page, block)
                    if heading is not None:
                        state.pop_to_before(heading.level)
                        state.push(heading)
                    continue

                text = join_lines([line.text for line in block.lines])
                if len(text) < self.min_chars:
                    continue

                for piece in self._split_to_size(text):
                    index = per_page_index.get(page.page, 0)
                    per_page_index[page.page] = index + 1
                    chunks.append(
                        Chunk(
                            chunk_id=make_chunk_id(meta.document_id, page.page, index),
                            document_id=meta.document_id,
                            filename=meta.filename,
                            title=meta.title,
                            page=page.page,
                            section=state.current,
                            text=piece,
                            source=meta.source,
                            is_reference=state.is_reference,
                        )
                    )

        meta.chunk_count = len(chunks)
        logger.info(
            "Chunked %s into %d chunks across %d pages",
            meta.filename,
            len(chunks),
            meta.page_count,
        )
        return chunks

    def chunk_documents(self, documents: Iterable[ExtractedDocument]) -> list[Chunk]:
        out: list[Chunk] = []
        for document in documents:
            out.extend(self.chunk_document(document))
        return out

    # -- internals ---------------------------------------------------------

    def _blocks_by_page(
        self, document: ExtractedDocument
    ) -> list[tuple[PageContent, list[RawBlock]]]:
        """Pair each structured page with its raw blocks, in reading order.

        The blocks were already sorted main-column-first during extraction, so
        re-sorting here is only a safeguard for documents that were
        reconstructed from a cache.
        """
        from .pdf_processor import PDFProcessor

        paired: list[tuple[PageContent, list[RawBlock]]] = []
        for index, page in enumerate(document.pages):
            raw_blocks = (
                document.raw_pages[index]
                if index < len(document.raw_pages)
                else []
            )
            width = (
                document.page_widths[index]
                if index < len(document.page_widths)
                else 612.0
            )
            paired.append((page, PDFProcessor._sort_reading_order(raw_blocks, width)))
        return paired

    @staticmethod
    def _is_heading(page: PageContent, block: RawBlock) -> bool:
        text = normalise_whitespace(block.text)
        return any(text == heading.title for heading in page.headings)

    @staticmethod
    def _heading_for(page: PageContent, block: RawBlock) -> SectionHeading | None:
        text = normalise_whitespace(block.text)
        for heading in page.headings:
            if heading.title == text:
                return heading
        return None

    def _split_to_size(self, text: str) -> list[str]:
        """Break a block into pieces no larger than ``chunk_size``."""
        if len(text) <= self.chunk_size:
            return [text]

        pieces: list[str] = []
        current: list[str] = []
        length = 0
        for sentence in split_sentences(text):
            if len(sentence) > self.chunk_size:
                # Pathological single sentence: fall back to a hard slice.
                if current:
                    pieces.append(" ".join(current))
                    current, length = [], 0
                pieces.extend(self._hard_slice(sentence))
                continue
            if length + len(sentence) + 1 > self.chunk_size and current:
                pieces.append(" ".join(current))
                tail = self._overlap_tail(" ".join(current))
                current = [tail] if tail else []
                length = len(current[0]) if current else 0
            current.append(sentence)
            length += len(sentence) + 1
        if current:
            piece = " ".join(current).strip()
            if piece and (not pieces or piece != pieces[-1]):
                pieces.append(piece)
        return [p for p in pieces if p.strip()]

    def _overlap_tail(self, text: str) -> str:
        """Last ``chunk_overlap`` characters of ``text``, cut at a word edge."""
        if self.chunk_overlap <= 0 or len(text) <= self.chunk_overlap:
            return ""
        tail = text[-self.chunk_overlap :]
        space = tail.find(" ")
        return tail[space + 1 :] if 0 <= space < len(tail) - 1 else ""

    def _hard_slice(self, text: str) -> list[str]:
        """Last-resort splitter for a single oversized sentence."""
        out: list[str] = []
        remaining = text
        while len(remaining) > self.chunk_size:
            cut = remaining.rfind(" ", 0, self.chunk_size)
            if cut <= 0:
                cut = self.chunk_size
            out.append(remaining[:cut].strip())
            remaining = remaining[cut:].lstrip()
        if remaining.strip():
            out.append(remaining.strip())
        return out


__all__ = ["Chunker", "split_sentences", "join_lines"]
