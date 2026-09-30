"""Question answering orchestration.

Order of operations, which is what makes the honesty guarantees hold:

1. Retrieve with the hybrid retriever.
2. If nothing clears the relevance gate, refuse without calling the LLM.
3. Build a bounded SOURCES prompt; call the LLM.
4. Verify the answer deterministically (markers, unknown markers, numbers).
5. If the answer carries a fabricated number, discard it and replace it with a
   refusal that explains the gap.

Step 5 is the one that answers the "what percentage of UAV missions are fully
autonomous?" question correctly without trusting the model.
"""

from __future__ import annotations

import time
from typing import Iterator, Sequence

from .citations import citations_for_report, evidence_for_chunks
from .config import Settings
from .errors import AnsweringError
from .grounding import (
    GroundingReport,
    find_source_echo,
    normalise_markers,
    strip_trailing_hedge,
    verify,
)
from .llm import LLMClient
from .logging_utils import get_logger
from .models import AnswerResult, RetrievedChunk, truncate
from .prompts import (
    NO_EVIDENCE_TEMPLATE,
    PARTIAL_NOTE_TEMPLATE,
    SYSTEM_PROMPT,
    build_ask_prompt,
    build_refusal_prompt,
)
from .retriever import Retriever

logger = get_logger(__name__)

MAX_CONTEXT_CHARS = 9000


class Answerer:
    """Turns a question plus an index into a citable, verified answer."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient | None,
        settings: Settings,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    # -- public API ---------------------------------------------------------

    def answer(self, question: str) -> AnswerResult:
        started = time.perf_counter()
        question = (question or "").strip()
        if not question:
            raise AnsweringError("Question must not be empty.")

        retrieved = self.retriever.search(question, k=self.settings.top_k)
        retrieved = self._bound_context(retrieved)

        if not retrieved:
            return self._no_evidence_result(question, started)

        if self.llm is None:
            return self._no_llm_result(question, retrieved, started)

        partial = self._is_partial(retrieved)
        prompt = build_ask_prompt(question, retrieved, partial=partial)
        raw = self.llm.complete(SYSTEM_PROMPT, prompt)
        report = verify(raw, retrieved, require_citations=self.settings.strict_citations)

        notes: list[str] = []
        if report.source_echo:
            logger.info("Answer echoed the source block; retrying with a nudge.")
            retried = self._retry_without_echo(question, retrieved, partial)
            if retried:
                raw = retried
                report = verify(
                    raw, retrieved, require_citations=self.settings.strict_citations
                )
            else:
                notes.append(
                    "The model repeated the source text instead of answering. "
                    "The passage is shown below so you can read it directly."
                )

        answer_text: str | None = None
        if report.unsupported_numbers:
            bad_numbers = list(report.unsupported_numbers)
            logger.info(
                "Draft answer contained unsupported number(s) %s; retrying.",
                bad_numbers,
            )
            corrected = self._retry_without_bad_numbers(
                question, retrieved, partial, bad_numbers
            )
            if corrected:
                raw = corrected
                report = verify(
                    raw, retrieved, require_citations=self.settings.strict_citations
                )
                notes.append(
                    "Rewrote the answer to remove "
                    + ", ".join(bad_numbers)
                    + ", which appeared in no retrieved page."
                )
            else:
                logger.info(
                    "Discarding answer with unsupported number(s) %s", bad_numbers
                )
                answer_text, extra = self._fabrication_refusal(
                    question, retrieved, bad_numbers
                )
                notes.extend(extra)
                report = verify(answer_text, retrieved, require_citations=False)

        if answer_text is None:
            answer_text = normalise_markers(raw)
            # Applied last so it also covers the retry paths above. Re-verify
            # afterwards so the reported status describes what is actually shown.
            trimmed = strip_trailing_hedge(answer_text)
            if trimmed != answer_text:
                logger.info("Removed trailing disclaimer from a cited answer.")
                notes.append(
                    "Removed a trailing disclaimer that contradicted the cited "
                    "answer above it."
                )
                answer_text = trimmed
                report = verify(
                    answer_text,
                    retrieved,
                    require_citations=self.settings.strict_citations,
                )
        if partial:
            notes.append(PARTIAL_NOTE_TEMPLATE)
        if report.unknown_markers:
            notes.append(
                "Dropped unknown citation marker(s): "
                + ", ".join(report.unknown_markers)
            )

        status = self._status(report, retrieved)
        citations = citations_for_report(report, retrieved)

        return AnswerResult(
            question=question,
            answer=answer_text.strip(),
            citations=citations,
            status=status,
            confidence=self._confidence(retrieved),
            notes=notes,
            retrieved=retrieved,
            llm_model=self.settings.llm_model,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def stream(self, question: str) -> Iterator[str]:
        """Stream a verified answer.

        Streaming starts only after the grounding gate has passed, so a refusal
        still arrives intact. Note that the deterministic verification runs on
        the *completed* text, so a streamed answer is verified only at the end;
        :meth:`answer` is the authoritative path for programmatic callers.
        """
        question = (question or "").strip()
        if not question:
            raise AnsweringError("Question must not be empty.")

        retrieved = self._bound_context(self.retriever.search(question, k=self.settings.top_k))
        if not retrieved:
            yield self._no_evidence_result(question, time.perf_counter()).answer
            return
        if self.llm is None:
            yield self._no_llm_result(question, retrieved, time.perf_counter()).answer
            return

        prompt = build_ask_prompt(
            question, retrieved, partial=self._is_partial(retrieved)
        )
        buffer: list[str] = []
        for piece in self.llm.stream(SYSTEM_PROMPT, prompt):
            buffer.append(piece)
            yield piece
        final = verify("".join(buffer), retrieved, require_citations=False)
        if final.unsupported_numbers:
            yield (
                "\n\n---\n\n**Grounding check:** this answer contained a value "
                "not present in the retrieved pages and was withheld. "
                + truncate(
                    self._no_evidence_result(question, time.perf_counter()).answer, 200
                )
            )

    def evidence(self, question: str) -> list:
        """The passages a question would use, without calling the LLM."""
        retrieved = self._bound_context(self.retriever.search(question, k=self.settings.top_k))
        return evidence_for_chunks(retrieved)

    # -- internals ----------------------------------------------------------

    def _bound_context(self, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """Keep the highest-ranked chunks that fit the context character budget."""
        bounded: list[RetrievedChunk] = []
        used = 0
        for chunk in chunks:
            length = len(chunk.chunk.text)
            if bounded and used + length > MAX_CONTEXT_CHARS:
                break
            bounded.append(chunk)
            used += length
        return bounded

    def _is_partial(self, chunks: Sequence[RetrievedChunk]) -> bool:
        if not chunks:
            return True
        return chunks[0].score < self.settings.min_retrieval_score + 0.15

    def _status(self, report: GroundingReport, chunks: Sequence[RetrievedChunk]) -> str:
        if report.is_refusal:
            return "partially_grounded" if report.valid_markers else "no_context"
        if not chunks:
            return "no_context"
        if not report.is_grounded or not report.valid_markers:
            return "partially_grounded"
        if self._is_partial(chunks):
            return "partially_grounded"
        return "grounded"

    @staticmethod
    def _confidence(chunks: Sequence[RetrievedChunk]) -> float:
        if not chunks:
            return 0.0
        top = max(chunk.score for chunk in chunks)
        return round(max(0.0, min(1.0, top)), 3)

    def _retry_without_bad_numbers(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        partial: bool,
        bad_numbers: Sequence[str],
    ) -> str:
        """Ask again, naming the numbers that must not be used.

        A single stray digit should not cost the user a whole answer, but it
        must never survive either. So the model gets one chance to restate the
        answer without the offending values; if it still leaks one, the caller
        falls back to a refusal.
        """
        if self.llm is None:
            return ""
        prompt = build_ask_prompt(question, chunks, partial=partial)
        prompt += (
            "\n\nIMPORTANT: your previous draft used the following value(s): "
            + ", ".join(bad_numbers)
            + ". None of them appear in any retrieved page, so they cannot be "
            "reported. Answer the question again in your own words, omitting "
            "those values entirely or restating the point without a "
            "statistic. Do not guess a replacement number. Every sentence must "
            "still end with its [S#] marker."
        )
        try:
            text = normalise_markers(self.llm.complete(SYSTEM_PROMPT, prompt))
        except Exception as exc:  # noqa: BLE001 - retry is best effort
            logger.warning("Number-correction retry failed: %s", exc)
            return ""
        if not text.strip():
            return ""
        # Only accept the rewrite if it is clean and cites something.
        check = verify(text, chunks, require_citations=False)
        if check.unsupported_numbers or check.unknown_markers:
            logger.info(
                "Rewrite still unsupported: %s", check.unsupported_numbers
            )
            return ""
        if find_source_echo(text):
            return ""
        return text

    def _retry_without_echo(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        partial: bool,
    ) -> str:
        """Second attempt with an explicit instruction not to copy the sources."""
        if self.llm is None:
            return ""
        prompt = build_ask_prompt(question, chunks[:3], partial=partial)
        prompt += (
            "\n\nIMPORTANT: your previous reply repeated the SOURCES block "
            "verbatim. This is not acceptable. Write two or three sentences "
            "answering the question directly in your own words, each sentence "
            "ending with its [S#] marker. Do not print the word 'Document', "
            "'Page', 'Section' or 'Text', and do not repeat a passage in full."
        )
        try:
            text = normalise_markers(self.llm.complete(SYSTEM_PROMPT, prompt))
        except Exception as exc:  # noqa: BLE001 - retry is best effort
            logger.warning("Echo retry failed: %s", exc)
            return ""
        return "" if find_source_echo(text) else text

    def _no_evidence_result(self, question: str, started: float) -> AnswerResult:
        count = self.retriever.store.document_count
        plural = "" if count == 1 else "s"
        return AnswerResult(
            question=question,
            answer=NO_EVIDENCE_TEMPLATE.format(
                document_count=count, document_plural=plural
            ),
            citations=[],
            status="no_context",
            confidence=0.0,
            notes=["No retrieved passage cleared the relevance threshold."],
            retrieved=[],
            llm_model=None,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def _no_llm_result(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        started: float,
    ) -> AnswerResult:
        return AnswerResult(
            question=question,
            answer=(
                "No LLM provider is configured, so I cannot generate a written "
                "answer. The retrieved passages are shown below as evidence."
            ),
            citations=evidence_for_chunks(chunks),
            status="ungrounded",
            confidence=self._confidence(chunks),
            notes=["Set LLM_PROVIDER / LLM_MODEL (or LLM_API_KEY) to enable answers."],
            retrieved=chunks,
            llm_model=None,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def _fabrication_refusal(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        values: Sequence[str],
    ) -> tuple[str, list[str]]:
        values_text = ", ".join(values)
        note = (
            f"Withheld: the draft answer contained {values_text}, which does "
            "not appear in any retrieved page."
        )
        fixed = (
            "The supplied documents do not state this. The retrieved pages "
            "do not contain the requested value, so it cannot be reported "
            "from these sources."
        )
        try:
            text = normalise_markers(
                self.llm.complete(SYSTEM_PROMPT, build_refusal_prompt(question, chunks[:3]))
            )
        except Exception as exc:  # noqa: BLE001 - fall back to a fixed refusal
            logger.warning("Refusal phrasing failed: %s", exc)
            return fixed, [note]

        # The refusal path is the last line of defence, so it is verified too.
        # Asked to decline, a small model will sometimes restate the very number
        # it was told to avoid, which would reintroduce the fabrication we just
        # caught. If that happens, use wording that is guaranteed safe.
        if not text.strip():
            return fixed, [note]
        check = verify(text, chunks, require_citations=False)
        if check.unsupported_numbers:
            logger.warning(
                "Refusal draft reintroduced unsupported number(s) %s; "
                "using fixed wording",
                check.unsupported_numbers,
            )
            return fixed, [note]
        return text, [note]
