"""Grounded per-document summarisation.

Uses a map-reduce strategy so a 50-page document is never sent to the model in
one prompt: each group of chunks is summarised independently, then the partial
summaries are combined. Falls back to a structural extractive summary when no
LLM is available, so the feature degrades rather than failing.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .citations import build_citations
from .config import Settings
from .grounding import verify
from .llm import LLMClient
from .logging_utils import get_logger
from .models import (
    Chunk,
    DocumentSummary,
    RetrievedChunk,
    dedupe_preserve_order,
    normalise_whitespace,
    truncate,
)
from .prompts import SUMMARY_SYSTEM_PROMPT, build_summary_prompt

logger = get_logger(__name__)


class DocumentSummarizer:
    """Produces a cited summary of a single document."""

    def __init__(self, settings: Settings, llm: LLMClient | None = None) -> None:
        self.settings = settings
        self.llm = llm

    def summarize(
        self,
        title: str,
        document_id: str,
        chunks: Sequence[Chunk],
    ) -> DocumentSummary:
        usable = [chunk for chunk in chunks if not chunk.is_reference]
        if not usable:
            return DocumentSummary(
                document_id=document_id,
                title=title,
                summary="This document has no indexable body text.",
                llm_generated=False,
            )

        if self.llm is None:
            return self._extractive(title, document_id, usable)

        groups = self._group(usable, self.settings.summary_map_chunks)
        partials: list[str] = []
        pages: list[int] = []
        for group in groups:
            pages.extend(chunk.page for chunk in group)
            retrieved = [RetrievedChunk(chunk=chunk, score=1.0) for chunk in group]
            try:
                text = self.llm.complete(
                    SUMMARY_SYSTEM_PROMPT,
                    build_summary_prompt(title, retrieved),
                )
            except Exception as exc:  # noqa: BLE001 - summary must not break the app
                logger.warning("Summary map step failed: %s", exc)
                return self._extractive(title, document_id, usable)
            if text.strip():
                partials.append(text.strip())

        if not partials:
            return self._extractive(title, document_id, usable)

        combined = "\n\n".join(partials)
        if len(partials) > 1:
            reduce_groups = self._split_for_reduce(partials, self.settings.summary_reduce_chunks)
            if len(reduce_groups) > 1:
                merged = self._reduce(title, reduce_groups)
                if merged:
                    combined = merged

        retrieved_all = [RetrievedChunk(chunk=chunk, score=1.0) for chunk in usable]
        report = verify(combined, retrieved_all, require_citations=False)
        if report.unsupported_numbers:
            logger.warning(
                "Summary for %s contained %d unsupported number(s); "
                "using extractive fallback instead.",
                title,
                len(report.unsupported_numbers),
            )
            return self._extractive(title, document_id, usable)

        return DocumentSummary(
            document_id=document_id,
            title=title,
            summary=combined,
            topics=self._topics(usable),
            pages=sorted(set(pages)),
            llm_generated=True,
            model=getattr(self.settings, "llm_model", None),
        )

    def _reduce(self, title: str, partials: Sequence[str]) -> str:
        joined = "\n\n".join(partials)
        if not joined.strip():
            return ""
        try:
            return self.llm.complete(
                SUMMARY_SYSTEM_PROMPT,
                f"Below are partial summaries of the document \"{title}\". "
                "Merge them into one cohesive summary, grouped by section. "
                "Keep every citation marker exactly as it appears, drop "
                "duplicates, and do not introduce any new fact or number.\n\n"
                f"{joined}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Summary reduce step failed: %s", exc)
            return ""

    def _extractive(
        self,
        title: str,
        document_id: str,
        chunks: Sequence[Chunk],
    ) -> DocumentSummary:
        """Structure-only summary used when the LLM is unavailable."""
        by_section: dict[str, list[str]] = defaultdict(list)
        for chunk in chunks:
            if chunk.section:
                by_section[chunk.section].append(chunk.text.strip())

        lines: list[str] = []
        for section, texts in list(by_section.items())[:8]:
            first = truncate(normalise_whitespace(texts[0]), 260)
            lines.append(f"- **{section}** (p. {chunks[0].page}): {first}")

        summary = "\n".join(lines) if lines else truncate(
            normalise_whitespace(chunks[0].text), 400
        )
        return DocumentSummary(
            document_id=document_id,
            title=title,
            summary=summary,
            topics=self._topics(chunks),
            pages=sorted({chunk.page for chunk in chunks}),
            llm_generated=False,
        )

    @staticmethod
    def _group(chunks: Sequence[Chunk], size: int) -> list[list[Chunk]]:
        size = max(1, size)
        return [list(chunks[i : i + size]) for i in range(0, len(chunks), size)]

    @staticmethod
    def _split_for_reduce(partials: Sequence[str], size: int) -> list[list[str]]:
        size = max(1, size)
        return [list(partials[i : i + size]) for i in range(0, len(partials), size)]

    @staticmethod
    def _topics(chunks: Sequence[Chunk]) -> list[str]:
        return dedupe_preserve_order(
            [chunk.section for chunk in chunks if chunk.section]
        )[:12]


def summarise_documents(
    settings: Settings,
    llm: LLMClient | None,
    documents: Sequence[tuple[str, str, Sequence[Chunk]]],
) -> list[DocumentSummary]:
    """Summarise several ``(document_id, title, chunks)`` triples."""
    summarizer = DocumentSummarizer(settings, llm)
    results: list[DocumentSummary] = []
    for document_id, title, chunks in documents:
        try:
            results.append(summarizer.summarize(title, document_id, chunks))
        except Exception as exc:  # noqa: BLE001 - one bad doc must not stop the rest
            logger.warning("Could not summarise %s: %s", title, exc)
    return results


__all__ = ["DocumentSummarizer", "summarise_documents", "build_citations"]
