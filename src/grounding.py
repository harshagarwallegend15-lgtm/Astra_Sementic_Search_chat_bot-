"""Deterministic grounding verification.

Runs after generation and checks, without a second LLM call, that:

* every citation marker resolves to a source that was actually supplied;
* every number, percentage, year and quantity in the answer is present
  verbatim in the retrieved context;
* refusals are recognised so a refusal is never mislabelled as a failure.

The numeric check is what makes the "what percentage of UAV missions are fully
autonomous?" honesty test safe: the documents contain no such statistic, so any
percentage in the answer cannot appear in the context and is rejected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .models import RetrievedChunk, normalise_whitespace, truncate

MARKER_PATTERN = re.compile(r"\[(?:S|s)\s?(\d{1,3})\]")
ANY_BRACKET_CID_PATTERN = re.compile(r"\[(?:S|s)\s?(\d{1,3})\s*,\s*(?:p{1,2}\.?\s*)?(\d{1,4})\]")
# A bare bracketed integer, e.g. "[59]". These are the source document's own
# reference markers (the PDFs are rendered from Wikipedia), not citations into
# this index. Only matched after [S#] markers have already been normalised, so
# a real citation can never be swallowed by this.
BARE_REFERENCE_PATTERN = re.compile(r"\[(?:\d{1,4}(?:\s*[,\-–]\s*\d{1,4})*)\]")
PERCENT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent|per\s?cent)")
NUMBER_PATTERN = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\b")
YEAR_PATTERN = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b")
CURRENCY_PATTERN = re.compile(r"([$€£¥])\s?(\d+(?:\.\d+)?)")

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "million": "1000000", "billion": "1000000000",
}


REFUSAL_PHRASES = (    "do not state", "does not state", "not stated", "do not specify",
    "does not specify", "not specify", "do not contain", "does not contain",
    "do not include", "does not include", "no information", "not available",
    "cannot be answered", "can't be answered", "cannot be determined",
    "no such", "does not appear", "do not appear", "not mention",
    "does not mention", "do not mention", "no mention", "not covered",
    "do not cover", "does not cover", "not reported", "does not report",
    "insufficient information", "not enough information", "cannot answer",
    "i cannot answer", "unable to answer", "not answerable",
    "the documents do not", "this document does not", "no basis",
)

ISSUE_MISSING_MARKER = "answer_has_no_citations"
ISSUE_UNKNOWN_MARKER = "unknown_citation_marker"
ISSUE_UNSUPPORTED_NUMBER = "unsupported_number"
ISSUE_UNCITED_CLAIM = "uncited_claim"
ISSUE_SOURCE_ECHO = "answer_echoes_source_block"

#: Labels the model emitted instead of answering, when it copied the prompt's
#: SOURCES layout. Small models do this often enough to be worth catching.
SOURCE_BLOCK_LABELS = (
    "document:", "page:", "section:", "text:", "begin sources", "end sources",
)


@dataclass(frozen=True)
class GroundingIssue:
    kind: str
    detail: str
    excerpt: str = ""

    def __str__(self) -> str:
        base = f"{self.kind}: {self.detail}"
        return f"{base} ({truncate(self.excerpt, 90)})" if self.excerpt else base


@dataclass
class GroundingReport:
    answer: str
    markers: list[str] = field(default_factory=list)
    valid_markers: list[str] = field(default_factory=list)
    unknown_markers: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    issues: list[GroundingIssue] = field(default_factory=list)
    is_refusal: bool = False
    is_grounded: bool = True
    source_count: int = 0
    source_markers: list[str] = field(default_factory=list)
    #: Set when the answer copied the SOURCES layout instead of answering.
    source_echo: str | None = None

    @property
    def blocking_issues(self) -> list[GroundingIssue]:
        blocking = {ISSUE_UNSUPPORTED_NUMBER, ISSUE_UNKNOWN_MARKER}
        return [issue for issue in self.issues if issue.kind in blocking]

    def summary(self) -> str:
        if self.is_refusal:
            return "Refused: the retrieved passages do not answer the question."
        if not self.issues:
            return f"Grounded in {len(self.valid_markers)} citation(s)."
        parts = [str(issue) for issue in self.issues]
        return f"{len(self.issues)} issue(s): " + "; ".join(parts)


def normalise_markers(text: str) -> str:
    """Collapse compound markers such as ``[S1, p. 2]`` down to ``[S1]``.

    The prompt forbids the model from writing page numbers, but smaller models
    do it anyway. Stripping the page keeps a well-formed citation usable instead
    of discarding the whole answer, and the page is re-derived from the chunk.
    """
    if not text:
        return ""
    text = ANY_BRACKET_CID_PATTERN.sub(lambda m: f"[S{m.group(1)}]", text)
    text = re.sub(r"\[\s*(S\s?\d{1,3})(?:\s*,\s*S\s?\d{1,3})*\]", lambda m: f"[{m.group(1)}]", text)
    # The source PDFs are rendered from Wikipedia, so their text carries
    # reference markers such as "[59][60]" into the chunk, and the model
    # faithfully reproduces them. They point at the document's own bibliography,
    # not at our sources, and leaving them in makes a fabricated-looking
    # citation appear in the answer. Drop them; the real provenance is the
    # [S#] marker.
    text = BARE_REFERENCE_PATTERN.sub("", text)
    return text


def extract_markers(text: str) -> list[str]:
    """Return unique citation markers in order of first appearance."""
    seen: list[str] = []
    for match in MARKER_PATTERN.finditer(text or ""):
        marker = f"S{match.group(1)}"
        if marker not in seen:
            seen.append(marker)
    return seen


def build_context_text(chunks: Sequence[RetrievedChunk]) -> str:
    """Flatten retrieved chunks into one normalised comparison corpus."""
    return normalise_whitespace(" ".join(chunk.chunk.text for chunk in chunks))


def _numeric_claims(text: str) -> list[str]:
    """Extract every numeric claim from the answer, de-duplicated.

    Values are returned in a canonical digit form so that ``75%``, ``75 percent``
    and ``75`` collapse to the same token, and number words such as ``three``
    collapse to ``3``.
    """
    claims: list[str] = []

    def add(value: str) -> None:
        token = value.replace(",", "").rstrip(".").strip()
        if token and token not in claims:
            claims.append(token)

    for match in PERCENT_PATTERN.finditer(text or ""):
        add(match.group(1))
    for match in CURRENCY_PATTERN.finditer(text or ""):
        add(match.group(2))
    for match in YEAR_PATTERN.finditer(text or ""):
        add(match.group(1))
    for match in NUMBER_PATTERN.finditer(text or ""):
        add(match.group(1))

    for word, digits in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text or "", flags=re.IGNORECASE):
            add(digits)

    return claims


def _context_numbers(context: str) -> set[str]:
    """Every numeric value the context supports, in canonical digit form.

    Number words are mapped to digits as well, so an answer saying "three" is
    supported by a context that also spells out "three" rather than writing "3".
    """
    numbers: set[str] = set()
    for match in NUMBER_PATTERN.finditer(context or ""):
        numbers.add(match.group(1).replace(",", "").rstrip("."))
    for match in PERCENT_PATTERN.finditer(context or ""):
        numbers.add(match.group(1).rstrip("."))
    for word, digits in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", context or "", flags=re.IGNORECASE):
            numbers.add(digits)
    return numbers


def find_unsupported_numbers(
    answer: str,
    context: str,
    ignore: Iterable[str] = (),
) -> list[str]:
    """Return numeric claims in ``answer`` that are absent from ``context``.

    ``ignore`` lists values that are structural rather than factual, such as
    citation-marker ids.
    """
    ignored = {str(value).replace(",", "") for value in ignore}
    allowed = _context_numbers(context)
    unsupported: list[str] = []
    for claim in _numeric_claims(answer or ""):
        if claim in ignored or claim in allowed:
            continue
        unsupported.append(claim)
    return unsupported


def detect_refusal(text: str) -> bool:
    """Heuristically detect an explicit 'not in the documents' refusal.

    Phrases such as "does not state" also turn up in a trailing caveat bolted
    onto an otherwise complete answer, and calling that a refusal mislabels
    good output and downgrades its status. The distinguishing signal is
    citations: a refusal asserts that nothing is available, so it has no source
    marker to stand on. A reply that cites first and hedges afterwards has
    answered, whatever its last paragraph says.
    """
    lowered = normalise_whitespace((text or "")).lower()
    if not lowered:
        return False
    positions = [lowered.find(phrase) for phrase in REFUSAL_PHRASES]
    positions = [p for p in positions if p != -1]
    if not positions:
        return False
    first = min(positions)
    return not MARKER_PATTERN.search((text or "")[:first])


# ---------------------------------------------------------------------------
# Trailing disclaimer removal
# ---------------------------------------------------------------------------

#: Discourse markers a disclaimer paragraph often opens with.
_HEDGE_OPENER_RE = re.compile(
    r"^(?:note|however|although|though|while|but|and|also)\b[\s,:;-]*"
    r"|^it(?:'s|\s+is)\s+(?:worth|useful)\s+noting\b[\s,:;-]*",
    re.IGNORECASE,
)

#: The subject of the disclaimer: the documents, not the world.
#: Minimum surviving length after a disclaimer is removed. Deliberately modest:
#: it only exists to stop a strip that would leave nothing useful. The real
#: guard is the citation requirement.
MIN_REMAINDER_CHARS: int = 40

_HEDGE_SUBJECT_RE = re.compile(
    r"\b(?:document|documents|source|sources|passage|passages|text|excerpts?|"
    r"extracts?|material|materials|corpus|context)\b",
    re.IGNORECASE,
)

#: The negated claim itself.
_HEDGE_CLAIM_RE = re.compile(
    # "do not state", but also "do not explicitly state" / "does not appear to
    # provide" - an adverb between the negation and the verb is how models
    # actually hedge, and missing it was a real observed failure.
    r"\b(?:do|does|did)\s+not\s+(?:\w+ly\s+)?(?:explicitly\s+|directly\s+|"
    r"specifically\s+)?(?:state|specify|provide|say|mention|give|contain|"
    r"include|list|offer|address|discuss|compare|cover|report|indicate|"
    r"describe|present)\b"
    r"|\b(?:is|are|was|were)\s+not\s+(?:\w+ly\s+)?(?:stated|specified|provided|"
    r"mentioned|given|listed|included|covered|addressed|described|reported)\b"
    r"|\bno\s+(?:direct\s+|explicit\s+|specific\s+|clear\s+)?"
    r"(?:comparison|statement|statements|data|figure|figures|number|numbers|"
    r"percentage|mention|information|detail|details)\b"
    r"|\bthere\s+(?:is|are)\s+no\b"
    r"|\b(?:but|rather)\s+than\b"
    r"|\bcannot\s+be\s+(?:determined|established|inferred)\b",
    re.IGNORECASE,
)


def _is_hedge_paragraph(paragraph: str) -> bool:
    """Is this an uncited 'the documents do not state...' disclaimer?"""
    text = paragraph.strip()
    if not text or MARKER_PATTERN.search(text):
        # A paragraph that carries its own citation is making a claim about the
        # evidence, not disclaiming it. Never strip it.
        return False
    probe = _HEDGE_OPENER_RE.sub("", text).strip()
    return bool(_HEDGE_SUBJECT_RE.search(probe) and _HEDGE_CLAIM_RE.search(probe))


#: Models very often append the disclaimer *inline* to the end of the last
#: paragraph ("... applications [S7]. However, this source does not provide
#: ..."), so paragraph-level handling alone misses the common case.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?:\]’”\"])\s+(?=[A-Z\"'(\u201c])")

#: A disclaimer may name the documents with a pronoun ("it does not provide",
#: "they do not state") once the subject has already been established. This is
#: only trusted in the final position, after everything else has been cited.
_HEDGE_PRONOUN_RE = re.compile(
    r"^(?:however|but|although|though|while|also|and|yet)?[\s,:;-]*"
    r"(?:it|they|this|these|those|the\s+source|the\s+document|they\s+do)\b",
    re.IGNORECASE,
)


def _is_hedge_sentence(sentence: str) -> bool:
    """Is this a trailing, uncited disclaimer sentence?"""
    text = sentence.strip()
    if len(text) < 15 or MARKER_PATTERN.search(text):
        return False
    if not _HEDGE_CLAIM_RE.search(text):
        return False
    return bool(_HEDGE_SUBJECT_RE.search(text) or _HEDGE_PRONOUN_RE.match(text))


def _strip_trailing_sentences(text: str) -> str:
    """Remove disclaimer sentences bolted onto the end of the last paragraph."""
    # An answer can legitimately be a single paragraph, so fall back to treating
    # the whole answer as the "last paragraph" when there is no blank line.
    head, sep, last = text.rpartition("\n\n")
    if not sep:
        head, last = "", text
    sentences = _SENTENCE_SPLIT_RE.split(last)
    if len(sentences) < 2:
        return text

    index = len(sentences)
    while index > 1 and _is_hedge_sentence(sentences[index - 1]):
        index -= 1
    if index == len(sentences):
        return text

    trimmed_last = " ".join(sentences[:index]).strip()
    remainder = f"{head}\n\n{trimmed_last}".strip() if head else trimmed_last
    # Never leave an answer without a source or without substance.
    if MARKER_PATTERN.search(remainder) and len(remainder) >= MIN_REMAINDER_CHARS:
        return remainder
    return text


def strip_trailing_hedge(text: str) -> str:
    """Drop a trailing disclaimer bolted onto an already-complete answer.

    Prompting the model not to append "the documents do not state anything
    further" after a cited answer is unreliable: even with an explicit rule it
    still happens, and it reads as the system undermining its own answer. The
    same two sentences that grade as ``grounded`` are poor output, so grading
    alone cannot fix it - the sentence has to go.

    Handles both shapes models actually produce: a disclaimer as its own final
    paragraph, and one appended inline to the end of the last paragraph.

    Both passes only ever remove *trailing* text, and only while a cited,
    substantive remainder survives. A genuine refusal ("the documents do not
    state that percentage") has nothing but the disclaimer, so it is left
    intact and keeps its honest status.
    """
    if not text or not text.strip():
        return text

    stripped = text.strip()

    # Pass 1: whole trailing paragraphs.
    for separator in (r"\n\s*\n", r"\n"):
        paragraphs = re.split(separator, stripped)
        if len(paragraphs) < 2:
            continue
        index = len(paragraphs)
        while index > 1 and (
            _is_hedge_paragraph(paragraphs[index - 1])
            # A final paragraph that is nothing but one disclaimer sentence
            # ("However, it does not provide a clear distinction...") names the
            # documents only as a pronoun, so the paragraph check misses it.
            or _is_hedge_sentence(paragraphs[index - 1])
        ):
            index -= 1
        if index == len(paragraphs):
            break
        remainder = "\n\n".join(paragraphs[:index]).strip()
        if MARKER_PATTERN.search(remainder) and len(remainder) >= MIN_REMAINDER_CHARS:
            stripped = remainder
        break

    # Pass 2: disclaimer sentences inside the final paragraph.
    return _strip_trailing_sentences(stripped)


def _strip_marked_spans(text: str) -> str:
    return MARKER_PATTERN.sub(" ", text or "")


def _find_uncited_claims(answer: str, marker_count: int) -> list[str]:
    """Return claim-bearing lines that carry no citation marker."""
    uncited: list[str] = []
    for line in (answer or "").splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) < 15:
            continue
        if stripped.startswith(("#", "|", "```", "---", "**Document", "SOURCES")):
            continue
        if MARKER_PATTERN.search(stripped):
            continue
        letters = sum(character.isalpha() for character in stripped)
        if letters < 8:
            continue
        if stripped.rstrip().endswith(":"):
            continue
        uncited.append(stripped)
        if len(uncited) >= 3:
            break
    return uncited


def find_source_echo(answer: str) -> str | None:
    """Detect an answer that copied the SOURCES block instead of answering.

    Small models sometimes emit ``[S1] Document: X`` / ``Page: 2`` /
    ``Section: Y`` / ``Text: ...``, which is a non-answer: the user gets the
    passage they could not read, not an explanation of it.
    """
    lowered = (answer or "").lower()
    hits = [label for label in SOURCE_BLOCK_LABELS if label in lowered]
    if len(hits) >= 3:
        return ", ".join(hits)
    return None


def verify(
    answer: str,
    chunks: Sequence[RetrievedChunk],
    require_citations: bool = True,
) -> GroundingReport:
    """Verify an answer against the chunks it was allowed to use."""
    answer = normalise_markers(answer or "")
    available = [f"S{index}" for index in range(1, len(chunks) + 1)]
    markers = extract_markers(answer)
    valid = [marker for marker in markers if marker in available]
    unknown = [marker for marker in markers if marker not in available]

    context = build_context_text(chunks)
    is_refusal = detect_refusal(answer)

    ignored = {str(index) for index in range(1, len(chunks) + 1)}
    unsupported = find_unsupported_numbers(answer, context, ignore=ignored)

    issues: list[GroundingIssue] = []
    for marker in unknown:
        issues.append(
            GroundingIssue(
                kind=ISSUE_UNKNOWN_MARKER,
                detail=f"marker [{marker}] was not among the supplied sources",
                excerpt=_marker_context(answer, marker),
            )
        )
    for value in unsupported:
        issues.append(
            GroundingIssue(
                kind=ISSUE_UNSUPPORTED_NUMBER,
                detail=f"value {value!r} does not appear in the retrieved text",
                excerpt=_number_context(answer, value),
            )
        )

    if require_citations and not markers and not is_refusal:
        issues.append(
            GroundingIssue(
                kind=ISSUE_MISSING_MARKER,
                detail="answer contains no citation markers",
                excerpt=truncate(_strip_marked_spans(answer), 90),
            )
        )
    elif require_citations and not is_refusal:
        for line in _find_uncited_claims(answer, len(markers)):
            issues.append(
                GroundingIssue(
                    kind=ISSUE_UNCITED_CLAIM,
                    detail="claim has no citation marker",
                    excerpt=truncate(line, 90),
                )
            )

    blocking = {ISSUE_UNSUPPORTED_NUMBER, ISSUE_UNKNOWN_MARKER}
    is_grounded = not any(issue.kind in blocking for issue in issues)

    echo = find_source_echo(answer)
    if echo:
        issues.insert(
            0,
            GroundingIssue(
                kind=ISSUE_SOURCE_ECHO,
                detail=f"answer copied the source layout ({echo}) instead of answering",
                excerpt=truncate(_strip_marked_spans(answer), 90),
            ),
        )

    if is_refusal and unsupported:
        unsupported_numbers_reported: list[str] = []
    else:
        unsupported_numbers_reported = unsupported

    return GroundingReport(
        answer=answer,
        markers=markers,
        valid_markers=valid,
        unknown_markers=unknown,
        unsupported_numbers=unsupported_numbers_reported,
        issues=issues,
        is_refusal=is_refusal,
        is_grounded=is_grounded,
        source_count=len(chunks),
        source_markers=available,
        source_echo=echo,
    )


def _marker_context(text: str, marker: str) -> str:
    index = text.find(f"[{marker}")
    if index == -1:
        return ""
    return truncate(_strip_marked_spans(text[max(0, index - 90) : index + 90]), 160)


def _number_context(text: str, value: str) -> str:
    pattern = re.compile(
        rf"[^.]{{0,70}}?\b{re.escape(value)}\b[^.]{{0,70}}",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text or "")
    return truncate(_strip_marked_spans(match.group(0)), 160) if match else ""


def strip_all_markers(text: str) -> str:
    """Remove citation markers, for plain-text rendering."""
    return MARKER_PATTERN.sub("", text or "").strip()
