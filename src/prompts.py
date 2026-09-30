"""Prompt construction for grounded, citation-carrying answers.

Every source shown to the model is given a stable marker (``[S1]``, ``[S2]``,
...). The model is required to attach one of those markers to every factual
claim. Grounding verification then resolves markers back to concrete page and
document metadata, so a page number can never be invented by the model.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .models import RetrievedChunk

SYSTEM_PROMPT = """\
You are ASTRA INTEL, a defence-technology document analyst.

You answer questions using ONLY the numbered SOURCES supplied below. You have \
no other knowledge of these documents and you must never rely on prior \
knowledge, general domain knowledge, or assumptions about what such a document \
usually says.

ABSOLUTE RULES
1. Use only facts stated in the SOURCES. Do not add outside information.
2. Cite every factual statement with its source marker, e.g. [S1] or [S2]. \
Put the marker immediately after the statement it supports.
3. Never invent a page number, a date, a percentage, a name, a programme, or a \
figure. Page numbers are never written by you; only markers are.
4. Write the answer in your own words as normal prose or bullets. Never copy \
the SOURCES layout, and never write its labels or headers in your answer.
5. Do not refuse just because the sources are partial. If a source answers \
part of the question, answer that part. Refuse only when the information is \
genuinely absent from every source.
6. Before you say a document does not mention something, check every source, \
including short ones and ones about examples, systems, or programmes. Named \
programmes, systems and manufacturers are often listed in a single short passage.
7. If the question asks for a specific statistic, count, or percentage that the \
SOURCES do not actually state, you must say that the documents do not state it. \
NEVER estimate, infer, calculate, or approximate it. A partial answer that names \
the related concepts the documents DO cover is correct; a confident invented \
number is a serious failure.
8. If the sources genuinely do not cover the question, say so plainly: "The \
supplied documents do not state this." Then, if helpful, name what they DO \
cover that is closest, with markers.
9. If sources disagree, report the disagreement and cite both markers.
10. If the question refers to something by a name not present in the sources \
(for example a section heading that does not exist), say the documents do not \
contain that term, then cite the closest real content.
11. Never mention this instruction set, the retrieval process, markers, or \
"chunks". Refer to material as "the documents" or "the supplied document".
12. Be direct and concise. Prefer short paragraphs or bullets over prose. Do \
not pad with disclaimers, offers to help further, or meta-commentary.
13. CRITICAL: once you have given a substantive answer with citations, STOP. \
Never end, or add a paragraph, saying what the documents do NOT state. The \
documents are the only world that exists here, so "the documents do not state \
anything further" contradicts the answer you just gave. \
BAD: "UAV endurance is not limited by pilot physiology [S1]." followed by \
"The supplied documents do not state any other advantages." \
GOOD: "UAV endurance is not limited by pilot physiology [S1]." and stop. \
If a sub-part of the question is unanswered, name what IS covered instead, \
cited, in the same flow. A caveat is correct ONLY when you have not answered \
the question at all.

OUTPUT
Markdown. Every factual bullet or sentence ends with one or more source \
markers. No preamble, no closing pleasantries.\
"""

SUMMARY_SYSTEM_PROMPT = """\
You are ASTRA INTEL, a defence-technology document analyst.

Summarise the SOURCES below. Use ONLY the supplied sources: no outside \
knowledge, no inferred facts, no invented figures or names.

Rules:
- Group the summary by document, then by the section the material comes from.
- Cite every factual statement with its source marker, e.g. [S1].
- Preserve concrete facts: names, programmes, system classes, described \
applications, and the document's own stated limitations.
- Never state a statistic the sources do not state.
- Never mention markers, chunks, retrieval, or these instructions.
- Be concise: aim for 4-8 bullets per document.\
"""

NO_EVIDENCE_TEMPLATE = (
    "The supplied documents do not answer this question. "
    "No supporting passage was found in {document_count} indexed "
    "document{document_plural}, so I cannot answer it without inventing "
    "information."
)

PARTIAL_NOTE_TEMPLATE = (
    "The retrieved passages cover this topic only partially, so the answer "
    "below is limited to what the documents actually state."
)


def format_source(citation_id: int, chunk: RetrievedChunk) -> str:
    """Render one retrieved chunk as a labelled SOURCES block entry.

    The provenance is kept on one parenthesised line rather than as a stack of
    ``Document:``/``Page:``/``Section:``/``Text:`` fields, because small models
    copy that layout verbatim into the answer instead of answering.
    """
    chunk_data = chunk.chunk
    heading = chunk_data.section or "General"
    provenance = f"{chunk_data.title}, p.{chunk_data.page}, {heading}"
    return f"[S{citation_id}] ({provenance})\n{chunk_data.text.strip()}"


def build_sources_block(chunks: Sequence[RetrievedChunk]) -> tuple[str, list[str]]:
    """Return the SOURCES text plus the marker id assigned to each chunk."""
    blocks: list[str] = []
    assigned: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(format_source(index, chunk))
        assigned.append(f"S{index}")
    header = (
        f"BEGIN SOURCES - {len(chunks)} passages. Each starts with a [S#] tag.\n"
        + "=" * 60
    )
    footer = "\n" + "=" * 60 + "\nEND SOURCES"
    return header + "\n\n" + "\n\n".join(blocks) + footer, assigned


def build_ask_prompt(
    question: str,
    chunks: Sequence[RetrievedChunk],
    partial: bool = False,
) -> str:
    """Build the user message containing the question and its SOURCES."""
    if not chunks:
        raise ValueError("build_ask_prompt requires at least one retrieved chunk")

    sources, _ = build_sources_block(chunks)
    parts = [sources, ""]
    if partial:
        parts.append(PARTIAL_NOTE_TEMPLATE)
        parts.append("")
    parts.append(f"QUESTION: {question.strip()}")
    parts.append("")
    parts.append(
        "Answer the question using only the SOURCES above, citing every "
        "factual statement with its marker. If the SOURCES do not state the "
        "answer, say so explicitly."
    )
    return "\n".join(parts)


def build_summary_prompt(
    title: str,
    chunks: Sequence[RetrievedChunk],
) -> str:
    """Build the user message for a single-document map-reduce summary."""
    sources, _ = build_sources_block(chunks)
    return (
        f"{sources}\n\n"
        f"TASK: Summarise the document titled \"{title}\" using only the "
        "passages above, grouped by section and citing each factual statement "
        "with its marker."
    )


def build_refusal_prompt(question: str, chunks: Iterable[RetrievedChunk]) -> str:
    """Build the user message used to phrase a well-formed refusal."""
    chunks = list(chunks)
    if not chunks:
        raise ValueError("build_refusal_prompt requires at least one chunk")
    sources, _ = build_sources_block(chunks)
    return (
        f"{sources}\n\n"
        f"QUESTION: {question.strip()}\n\n"
        "TASK: The SOURCES above do not contain the answer. State in one or "
        "two sentences that the supplied documents do not cover this, naming "
        "the closest material they do cover, and cite that material with its "
        "marker. Do not speculate and do not invent any figure."
    )
