"""PDF text extraction with exact page numbers and real section headings.

This module is the foundation of every citation the system produces, so it is
deliberately conservative:

* **Page numbers are never estimated.** A page is the physical PDF page index
  (1-based) of the text that was read from it.
* **Section names are observed, not invented.** A section is recorded only when
  a heading was actually detected in the PDF. When nothing heading-like is
  found, the page's chunks carry ``section=None`` and the UI shows just the
  page number, which is what the challenge asks for.

Heading detection
-----------------
Rendered documents (this project's starter PDFs are Wikipedia print-to-PDF
exports) have a stable typographic hierarchy: body text sits at one modal font
size, and headings sit strictly above it in size, usually bold. We therefore

1. compute the character-weighted modal font size of the whole document - the
   body size, weighting by character count so a 12-word caption cannot
   outvote three paragraphs of body text;
2. treat any line whose size is meaningfully larger than the body size, and
   which is short enough to be a heading, as a heading;
3. rank the distinct heading sizes to assign heading levels 1, 2, 3, ...

This is size-relative, not size-absolute, so it also works on PDFs that have
nothing to do with Wikipedia.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pymupdf

from .config import Settings, get_settings
from .errors import (
    EmptyDocumentError,
    PDFProcessingError,
    UnsupportedFileTypeError,
)
from .logging_utils import get_logger
from .models import (
    DocumentMeta,
    PageContent,
    SectionHeading,
    TextLine,
    make_document_id,
    normalise_whitespace,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

#: A line must be at least this many characters wider than the body font size
#: before we are willing to call it a heading. Guards against superscripts and
#: italic lead-ins that are marginally larger.
HEADING_SIZE_MARGIN: float = 1.06

#: Longer than this and it is a paragraph, not a heading.
MAX_HEADING_CHARS: int = 180

#: A heading should not end in a sentence terminator.
_HEADING_ENDS_SENTENCE = re.compile(r"[.,;:!?)\]]$")

#: x-position (as a fraction of page width) at or beyond which text is treated
#: as belonging to a side column - infoboxes, figure captions, margin notes.
SIDE_COLUMN_RATIO: float = 0.5

#: Headings that introduce a bibliography rather than prose. Their content is
#: kept (it is part of the document) but flagged so retrieval can ignore it.
_REFERENCE_SECTION_TITLES: frozenset[str] = frozenset(
    {
        "references",
        "citations",
        "sources",
        "notes",
        "footnotes",
        "bibliography",
        "works cited",
        "external links",
        "further reading",
        "see also",
        "external references",
        "sources and further reading",
        "notes and references",
    }
)

#: Boilerplate emitted by the PDF generator itself.
_RETRIEVED_FROM_RE = re.compile(r"^\s*retrieved\s+from\b", re.IGNORECASE)
_WIKIPEDIA_URL_RE = re.compile(r"https?://[a-z.]*wikipedia\.org/w/index\.php", re.IGNORECASE)
_LICENSE_TAIL_RE = re.compile(
    r"^\s*(text is available under|this page was last edited|content is available under)",
    re.IGNORECASE,
)
#: A bare number sitting at the foot of a page is a page number, not content.
_PAGE_NUMBER_RE = re.compile(r"^\s*[-–—\[\(]?\s*\d{1,4}\s*[-–—\]\)]?\s*$")

#: Ligatures and invisible characters that break naive string matching.
_CHAR_FIXES = {
    "\u00ad": "",  # soft hyphen
    "\u200b": "",  # zero width space
    "\u200c": "",  # zero width non-joiner
    "\u200d": "",  # zero width joiner
    "\ufeff": "",  # byte order mark
    "\u00a0": " ",  # non-breaking space
    "\u202f": " ",  # narrow non-breaking space
    "\u2044": "/",  # fraction slash
}

#: Title junk appended by the PDF producer.
_TITLE_SUFFIX_RE = re.compile(
    r"\s*[-–—|]\s*(wikipedia|the free encyclopedia|printable version|full screen)\s*$",
    re.IGNORECASE,
)
_TITLE_PREFIX_RE = re.compile(r"^\s*(wikipedia|full screen|printable version)\s*[:\-–—]\s*", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------


def clean_text(raw: str) -> str:
    """Normalise extracted text without destroying meaningful punctuation.

    Em dashes and typographic quotes are preserved because they carry meaning
    in prose; invisible and layout characters are removed.
    """
    if not raw:
        return ""
    text = unicodedata.normalize("NFC", raw)
    for bad, good in _CHAR_FIXES.items():
        text = text.replace(bad, good)
    return re.sub(r"[ \t]+", " ", text).strip()


def _is_reference_title(title: str) -> bool:
    return normalise_whitespace(title).strip().lower().rstrip(":") in _REFERENCE_SECTION_TITLES


# ---------------------------------------------------------------------------
# Extraction result
# ---------------------------------------------------------------------------


@dataclass
class ExtractedDocument:
    """Everything we learned from one PDF.

    ``raw_pages`` keeps the paragraph-level blocks produced during extraction.
    The chunker needs them to cut on paragraph boundaries; the cleaned
    :class:`PageContent` list is what everything downstream reads.
    """

    meta: DocumentMeta
    pages: list[PageContent]
    body_size: float
    raw_pages: list[list["RawBlock"]] = field(default_factory=list)
    page_widths: list[float] = field(default_factory=list)

    @property
    def heading_titles(self) -> list[str]:
        seen: list[str] = []
        for page in self.pages:
            for heading in page.headings:
                if heading.title not in seen:
                    seen.append(heading.title)
        return seen


@dataclass
class RawLine:
    """Intermediate representation of a single visual line."""

    text: str
    size: float
    bold: bool
    x0: float
    y0: float
    page: int
    #: Index of the owning text block, which in a rendered article corresponds
    #: closely to a paragraph. The chunker uses this to avoid cutting a
    #: paragraph in half where possible.
    block_id: int = 0


@dataclass
class RawBlock:
    """A group of lines that belong to one paragraph-level unit."""

    block_id: int
    page: int
    lines: list[RawLine] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def x0(self) -> float:
        return min((line.x0 for line in self.lines), default=0.0)

    @property
    def y0(self) -> float:
        return min((line.y0 for line in self.lines), default=0.0)

    @property
    def size(self) -> float:
        """Font size of the widest line in the block."""
        return max((line.size for line in self.lines), default=0.0)

    @property
    def bold(self) -> bool:
        return bool(self.lines) and all(line.bold for line in self.lines)

    @property
    def char_count(self) -> int:
        return sum(len(line.text) for line in self.lines)


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class PDFProcessor:
    """Turns PDF bytes into pages of clean, positioned, section-tagged text."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # -- public API --------------------------------------------------------

    def process(
        self,
        data: bytes,
        filename: str,
        *,
        title: str | None = None,
        source: str | None = None,
        license_: str | None = None,
        url: str | None = None,
        retrieved: str | None = None,
    ) -> ExtractedDocument:
        """Extract ``data`` as a document named ``filename``.

        Raises :class:`PDFProcessingError` for unreadable files and
        :class:`EmptyDocumentError` when no usable text could be found.
        """
        self._validate_bytes(data, filename)

        document_id = make_document_id(data)
        try:
            with pymupdf.open(stream=data, filetype="pdf") as doc:
                if doc.needs_pass:
                    raise PDFProcessingError(
                        f"{filename} is password protected and cannot be read."
                    )
                page_count = doc.page_count
                raw_pages = [
                    self._read_page(doc, index) for index in range(page_count)
                ]
                page_widths = [float(doc[index].rect.width or 1.0) for index in range(page_count)]
                pdf_title = clean_text(doc.metadata.get("title") or "")
                pdf_author = clean_text(doc.metadata.get("author") or "")
        except PDFProcessingError:
            raise
        except Exception as exc:  # pymupdf raises a wide variety of errors
            raise PDFProcessingError(f"Could not read {filename}: {exc}") from exc

        if page_count == 0:
            raise EmptyDocumentError(f"{filename} contains no pages.")

        raw_blocks = [block for page in raw_pages for block in page]
        self._strip_running_headers(raw_pages)

        if not normalise_whitespace(
            " ".join(line.text for block in raw_blocks for line in block.lines)
        ):
            raise EmptyDocumentError(
                f"{filename} has no extractable text layer on any of its "
                f"{page_count} pages."
            )

        char_total = sum(block.char_count for block in raw_blocks)
        if char_total < self.settings.min_document_chars:
            raise EmptyDocumentError(
                f"{filename} yielded only {char_total} characters of text, "
                f"below the {self.settings.min_document_chars} character minimum. "
                "It is effectively empty."
            )

        body_size = self._body_font_size(raw_blocks)
        pages = self._build_pages(raw_pages, body_size)
        resolved_title = self._resolve_title(title, pdf_title, pages, filename)

        meta = DocumentMeta(
            document_id=document_id,
            filename=filename,
            title=resolved_title,
            page_count=page_count,
            char_count=char_total,
            source=source or pdf_author or "User upload",
            license=license_,
            url=url,
            retrieved=retrieved,
        )
        meta.headings = [h.title for p in pages for h in p.headings]
        meta.sections_detected = bool(meta.headings)
        # De-duplicate while preserving order for the UI listing.
        meta.headings = list(dict.fromkeys(meta.headings))

        logger.info(
            "Extracted %s: %d pages, %d chars, body font %.1fpt, %d headings",
            filename,
            page_count,
            char_total,
            body_size,
            len(meta.headings),
        )
        return ExtractedDocument(
            meta=meta,
            pages=pages,
            body_size=body_size,
            raw_pages=raw_pages,
            page_widths=page_widths,
        )

    # -- validation --------------------------------------------------------

    def _validate_bytes(self, data: bytes, filename: str) -> None:
        if not data:
            raise UnsupportedFileTypeError(f"{filename} is empty (0 bytes).")
        if not data.lstrip()[:5].startswith(b"%PDF"):
            # Some generators emit junk before the header; scan a little way in.
            head = data[:1024]
            if b"%PDF-" not in head:
                raise UnsupportedFileTypeError(
                    f"{filename} does not look like a PDF (missing %PDF header).",
                    user_message=(
                        f"'{filename}' is not a PDF. Please upload a .pdf file."
                    ),
                )
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        if len(data) > max_bytes:
            from .errors import DocumentTooLargeError

            raise DocumentTooLargeError(
                f"{filename} is {len(data) / 1e6:.1f} MB, over the "
                f"{self.settings.max_upload_mb} MB limit."
            )

    # -- page reading ------------------------------------------------------

    def _read_page(self, doc: pymupdf.Document, index: int) -> list[RawBlock]:
        """Read one page into paragraph-level blocks in reading order."""
        page = doc[index]
        try:
            page_dict = page.get_text("dict")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Page %d of %s unreadable: %s", index + 1, doc.name, exc)
            return []

        page_width = page.rect.width or 1.0
        page_height = page.rect.height or 1.0
        blocks: list[RawBlock] = []

        for block_index, block in enumerate(page_dict.get("blocks", [])):
            if block.get("type") != 0:  # 0 == text block
                continue
            raw_block = RawBlock(block_id=block_index, page=index + 1)
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = clean_text("".join(span.get("text", "") for span in spans))
                if not text:
                    continue
                if self._is_page_furniture(text, spans, page_height):
                    continue

                sizes = [
                    round(float(span.get("size", 0.0)), 2)
                    for span in spans
                    if span.get("text", "").strip()
                ]
                # Use the size of the widest span: a heading occasionally
                # contains a trailing superscript reference marker.
                dominant = max(sizes) if sizes else 0.0
                bold = bool(spans) and all(
                    int(span.get("flags", 0)) & (1 << 4) for span in spans
                )
                raw_block.lines.append(
                    RawLine(
                        text=text,
                        size=dominant,
                        bold=bold,
                        x0=float(line["bbox"][0]),
                        y0=float(line["bbox"][1]),
                        page=index + 1,
                        block_id=block_index,
                    )
                )
            if raw_block.lines:
                blocks.append(raw_block)

        return self._sort_reading_order(blocks, page_width)

    @staticmethod
    def _is_page_furniture(text: str, spans: Sequence[dict[str, Any]], page_height: float) -> bool:
        """Reject page numbers, retrieval footers and licence tails."""
        if _RETRIEVED_FROM_RE.match(text) or _LICENSE_TAIL_RE.match(text):
            return True
        if _WIKIPEDIA_URL_RE.search(text) and len(text) < 400:
            return True
        if _PAGE_NUMBER_RE.match(text):
            # Only treat a bare number as a page number when it really sits at
            # the top or bottom of the page; mid-page numbers are content.
            y0 = float(spans[0]["bbox"][1]) if spans and "bbox" in spans[0] else 0.0
            y1 = float(spans[0]["bbox"][3]) if spans and "bbox" in spans[0] else page_height
            if y0 < page_height * 0.08 or y1 > page_height * 0.92:
                return True
        return False

    @staticmethod
    def _sort_reading_order(blocks: list[RawBlock], page_width: float) -> list[RawBlock]:
        """Order blocks the way a human reads the page.

        Rendered articles frequently place an infobox or figure caption in a
        narrow right-hand column. Emitting the main column top-to-bottom first
        and the side column afterwards keeps the narrative contiguous, which
        makes far better chunks than the raw content-stream order.
        """
        if not blocks:
            return []
        split = page_width * SIDE_COLUMN_RATIO
        main = [b for b in blocks if b.x0 < split]
        side = [b for b in blocks if b.x0 >= split]

        # Only treat the page as two-column if the main column is not itself
        # running the full width of the page.
        two_column = bool(side) and bool(main) and max(b.x0 for b in main) < split
        if not two_column:
            return sorted(blocks, key=lambda b: (round(b.y0, 1), b.x0))
        main.sort(key=lambda b: (round(b.y0, 1), b.x0))
        side.sort(key=lambda b: (round(b.y0, 1), b.x0))
        return main + side

    def _strip_running_headers(self, pages: list[list[RawBlock]]) -> None:
        """Drop lines repeated on most pages (running heads and footers).

        Deliberately conservative: a candidate must be reasonably long and
        appear on at least half the pages, so short repeated body text is
        never removed by mistake.
        """
        page_count = len(pages)
        if page_count < 4:
            return

        occurrences: Counter[str] = Counter()
        for page in pages:
            for key in {normalise_whitespace(line.text) for block in page for line in block.lines}:
                occurrences[key] += 1

        threshold = max(3, (page_count * 0.5))
        repeated = {
            key
            for key, count in occurrences.items()
            if count >= threshold and 10 <= len(key) <= 120
        }
        if not repeated:
            return

        removed = 0
        for page in pages:
            kept: list[RawBlock] = []
            for block in page:
                block.lines = [
                    line
                    for line in block.lines
                    if normalise_whitespace(line.text) not in repeated
                ]
                if block.lines:
                    kept.append(block)
                else:
                    removed += 1
            page[:] = kept
        if removed:
            logger.info("Removed %d running header/footer blocks", removed)

    # -- typography --------------------------------------------------------

    @staticmethod
    def _body_font_size(blocks: Iterable[RawBlock]) -> float:
        """Character-weighted modal font size.

        Weighting by character count is what makes this robust: it prevents a
        large figure caption in a 10pt style from being mistaken for body text.
        """
        weights: Counter[float] = Counter()
        for block in blocks:
            for line in block.lines:
                if line.text.strip():
                    weights[round(line.size, 1)] += len(line.text)
        if not weights:
            return 0.0
        return weights.most_common(1)[0][0]

    def _build_pages(
        self, raw_pages: list[list[RawBlock]], body_size: float
    ) -> list[PageContent]:
        """Turn positioned lines into :class:`PageContent` with headings."""
        # Pass 1: decide which sizes are headings, and rank them into levels.
        heading_sizes = sorted(
            {
                round(block.size, 1)
                for page in raw_pages
                for block in page
                if self._is_heading_candidate(block, body_size)
            },
            reverse=True,
        )
        level_of = {size: index + 1 for index, size in enumerate(heading_sizes)}
        logger.debug(
            "Body size %.1fpt; heading sizes %s", body_size, heading_sizes or "none"
        )

        pages: list[PageContent] = []
        ordinal = 0
        for page_index, raw_blocks_page in enumerate(raw_pages, start=1):
            lines: list[TextLine] = []
            headings: list[SectionHeading] = []
            for block in raw_blocks_page:
                if self._is_heading_candidate(block, body_size):
                    size = round(block.size, 1)
                    ordinal += 1
                    headings.append(
                        SectionHeading(
                            title=normalise_whitespace(block.text),
                            level=level_of.get(size, len(heading_sizes) or 1),
                            page=page_index,
                            ordinal=ordinal,
                        )
                    )
                for raw in block.lines:
                    lines.append(
                        TextLine(
                            text=raw.text,
                            size=raw.size,
                            bold=raw.bold,
                            x0=raw.x0,
                            y0=raw.y0,
                        )
                    )
            pages.append(
                PageContent(
                    page=page_index, lines=lines, headings=headings, body_size=body_size
                )
            )
        return pages

    @staticmethod
    def _is_heading_candidate(block: RawBlock, body_size: float) -> bool:
        """A block is a heading if it is one or two short, larger-than-body lines."""
        lines = [line for line in block.lines if line.text.strip()]
        if not lines:
            return False
        if block.size < body_size * HEADING_SIZE_MARGIN:
            return False
        # A heading occupies a line or two; a paragraph of large text does not.
        if len(lines) > 2:
            return False
        text = normalise_whitespace(block.text)
        if len(text) < 2 or len(text) > MAX_HEADING_CHARS:
            return False
        if _HEADING_ENDS_SENTENCE.search(text):
            return False
        if len(text.split()) > 20:
            return False
        return True

    # -- titles ------------------------------------------------------------

    @staticmethod
    def _resolve_title(
        explicit: str | None,
        pdf_title: str,
        pages: list[PageContent],
        filename: str,
    ) -> str:
        """Best available human title, most trustworthy source first."""
        if explicit and explicit.strip():
            return normalise_whitespace(explicit)

        if pdf_title:
            cleaned = _TITLE_SUFFIX_RE.sub("", _TITLE_PREFIX_RE.sub("", pdf_title))
            cleaned = normalise_whitespace(cleaned)
            if len(cleaned) >= 3:
                return cleaned

        # The largest heading on page 1 is the document title in a rendered
        # article.
        if pages:
            first = pages[0]
            if first.headings:
                return normalise_whitespace(min(first.headings, key=lambda h: h.level).title)
            sizes = [line.size for line in first.lines if line.text.strip()]
            if sizes:
                biggest = max(sizes)
                for line in first.lines:
                    if round(line.size, 1) == round(biggest, 1):
                        return normalise_whitespace(line.text)

        stem = Path(filename).stem.replace("-", " ").replace("_", " ")
        return normalise_whitespace(stem).title() or filename


# ---------------------------------------------------------------------------
# Starter-document metadata
# ---------------------------------------------------------------------------


def load_metadata_catalog(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Index ``sample-metadata.json`` by filename.

    Used to attach the real title, licence, source and retrieval date to the
    starter PDFs so citations are properly attributed. A missing or malformed
    file is not fatal - the documents simply fall back to "User upload".
    """
    settings = get_settings()
    target = path or settings.sample_metadata_path
    catalog: dict[str, dict[str, Any]] = {}
    if not target.exists():
        logger.info("No sample metadata at %s; using default document titles", target)
        return catalog
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", target, exc)
        return catalog

    for entry in payload.get("documents", []):
        name = entry.get("file")
        if name:
            catalog[name] = entry
    logger.info("Loaded metadata for %d starter documents", len(catalog))
    return catalog


def metadata_for(filename: str, catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Metadata overrides for ``filename``, if the catalog knows about it."""
    entry = catalog.get(Path(filename).name)
    if not entry:
        return {}
    return {
        "title": entry.get("title"),
        "source": entry.get("source"),
        "license_": entry.get("license"),
        "url": entry.get("url"),
        "retrieved": entry.get("date"),
    }


__all__ = [
    "PDFProcessor",
    "ExtractedDocument",
    "RawLine",
    "clean_text",
    "load_metadata_catalog",
    "metadata_for",
    "HEADING_SIZE_MARGIN",
    "MAX_HEADING_CHARS",
]
