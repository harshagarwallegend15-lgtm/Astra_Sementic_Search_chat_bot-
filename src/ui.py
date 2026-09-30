"""ASTRA INTEL - Streamlit presentation layer.

Deliberately thin: it renders what the pipeline returns and never does
retrieval, prompting or verification of its own. Every user-triggerable failure
is caught and rendered as a readable message rather than a traceback.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import streamlit as st

from .citations import build_citations
from .errors import AstraIntelError
from .models import AnswerResult, Citation, DocumentMeta, RetrievedChunk

STATUS_COLOURS = {
    "grounded": ("Grounded", "success"),
    "partially_grounded": ("Partially grounded", "warning"),
    "ungrounded": ("Not grounded", "error"),
    "no_context": ("No supporting evidence", "error"),
}


def configure_page() -> None:
    st.set_page_config(
        page_title="ASTRA INTEL",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_header() -> None:
    st.title("ASTRA INTEL")
    st.caption(
        "Defence-technology document analyst. Every answer is grounded in "
        "supplied PDF pages, and every citation points to a real page."
    )


def render_sidebar(
    settings_public: dict[str, Any],
    stats: dict[str, Any],
    documents: Sequence[DocumentMeta],
    llm_ready: bool,
    llm_error: str | None = None,
) -> str:
    """Render the sidebar and return the action chosen by the user."""
    with st.sidebar:
        st.header("Status")
        st.metric("Documents indexed", stats.get("documents", 0))
        st.metric("Passages indexed", stats.get("chunks", 0))

        st.subheader("Model")
        rows = [
            ("LLM", f"{settings_public.get('LLM provider')} / "
                    f"{settings_public.get('LLM model')}"),
            ("Endpoint", settings_public.get("LLM base URL", "")),
            ("Embeddings", settings_public.get("Embedding model", "")),
            ("Retrieval", settings_public.get("retrieval mode", "hybrid")),
        ]
        for label, value in rows:
            st.caption(f"**{label}:** {value}")

        if llm_error:
            st.error(llm_error)
        elif llm_ready:
            st.success("LLM ready")
        else:
            st.info("LLM not configured - retrieval and evidence only.")

        action = "none"
        if documents:
            st.subheader("Indexed documents")
            for meta in documents:
                st.markdown(
                    f"- **{meta.title}**  \n"
                    f"  <small>{meta.page_count} pages · {meta.chunk_count} "
                    f"passages</small>",
                    unsafe_allow_html=True,
                )
        return action


def render_document_manager(
    on_ingest,
    on_rebuild,
    on_clear,
) -> None:
    """Render the document upload / rebuild / clear controls."""
    st.subheader("Documents")
    uploads = st.file_uploader(
        "Add PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Text-based PDFs only. Scanned images have no text layer.",
        key="uploader",
    )

    left, right = st.columns(2)
    add_clicked = left.button(
        "Index uploaded PDFs",
        use_container_width=True,
        disabled=not uploads,
    )
    rebuild_clicked = right.button(
        "Rebuild index",
        use_container_width=True,
        help="Re-extract the bundled sample documents from scratch.",
    )

    if add_clicked and uploads:
        on_ingest([(item.name, item.getvalue()) for item in uploads])
    if rebuild_clicked:
        on_rebuild()

    if st.session_state.get("documents"):
        st.markdown("---")
        if st.button("Clear all documents", use_container_width=True):
            on_clear()


def render_ingest_result(result) -> None:
    if result is None:
        return
    messages: list[str] = []
    if result.added_count:
        titles = ", ".join(meta.title for meta in result.added)
        messages.append(
            f"Indexed {result.added_count} document(s): {titles} "
            f"(+{result.chunk_count} passages in {result.elapsed_ms} ms)"
        )
    if result.duplicates:
        messages.append("Already indexed, skipped: " + ", ".join(result.duplicates))
    for filename, reason in (result.failures or {}).items():
        messages.append(f"**{filename}** - {reason}")
    for message in messages:
        st.success(message)


def render_chat(
    answer_func,
    evidence_func,
    suggested_questions: Sequence[str],
) -> None:
    st.subheader("Ask a question")
    if not st.session_state.get("documents"):
        st.info("Index at least one PDF to start asking questions.")
        return

    with st.expander("Suggested questions", expanded=False):
        for question in suggested_questions:
            if st.button(question, key=f"sugg_{question[:40]}"):
                st.session_state["pending_question"] = question
                st.rerun()

    question = st.text_input(
        "Your question",
        key="question_input",
        placeholder="e.g. What are the three primary mission types of electronic warfare?",
    )
    if st.session_state.pop("pending_question", None):
        question = st.session_state.get("question_input") or question

    columns = st.columns([1, 1, 2])
    ask = columns[0].button("Ask", type="primary", use_container_width=True)
    show_evidence = columns[1].button("Show evidence only", use_container_width=True)
    columns[2].caption(
        "Answers are verified: a value not present in the retrieved pages is "
        "withheld automatically."
    )

    if not ask and not show_evidence:
        return

    question = (question or "").strip()
    if not question:
        st.warning("Enter a question first.")
        return

    if show_evidence:
        _render_evidence_only(question, evidence_func)
        return

    started = time.perf_counter()
    with st.spinner("Searching the documents and composing a grounded answer ..."):
        try:
            result = answer_func(question)
        except AstraIntelError as exc:
            st.error(exc.user_message)
            return
        except Exception as exc:  # noqa: BLE001 - never surface a traceback
            st.error("Something went wrong while answering. See the log for details.")
            st.caption(str(exc))
            return
    _render_result(result, time.perf_counter() - started)


def _render_result(result: AnswerResult, elapsed: float) -> None:
    label, kind = STATUS_COLOURS.get(result.status, ("Unknown", "warning"))
    message = (
        f"{label} · confidence {result.confidence:.2f} · {elapsed:.1f}s"
        + (f" · {result.llm_model}" if result.llm_model else "")
    )
    if kind == "success":
        st.success(message)
    elif kind == "error":
        st.error(message)
    else:
        st.warning(message)

    st.markdown(result.answer)

    if result.notes:
        for note in result.notes:
            st.caption(note)

    if result.citations:
        st.markdown("---")
        st.markdown("**Sources**")
        for citation in result.citations:
            _render_citation(citation)

    with st.expander(f"Retrieved passages ({len(result.retrieved)})", expanded=False):
        for retrieved in result.retrieved:
            chunk = retrieved.chunk
            st.markdown(
                f"- `score={retrieved.score:.3f}` **{chunk.title}** "
                f"p.{chunk.page} · {chunk.section or 'general'}"
            )
            st.text(truncate(chunk.text, 400))


def _render_evidence_only(question: str, evidence_func) -> None:
    with st.spinner("Retrieving passages ..."):
        try:
            citations = evidence_func(question)
        except AstraIntelError as exc:
            st.error(exc.user_message)
            return
    if not citations:
        st.warning("No passage cleared the relevance threshold for that question.")
        return
    st.markdown(f"**{len(citations)} passages above the relevance threshold**")
    for citation in citations:
        _render_citation(citation)


def _render_citation(citation: Citation) -> None:
    marker = citation.marker or "S"
    section = f" · {citation.section}" if citation.section else ""
    st.markdown(
        f"- **[{marker}] {citation.title}** — page {citation.page}{section}  "
        f"<small>(retrieval score {citation.score:.3f})</small>",
        unsafe_allow_html=True,
    )
    with st.expander(f"Verbatim passage — {citation.filename}, p.{citation.page}"):
        st.text(citation.excerpt)
        st.caption(f"chunk id: {citation.chunk_id}")


def render_summary(summaries, on_summarise, llm_ready: bool) -> None:
    st.subheader("Document summaries")
    if st.button("Generate summaries", use_container_width=True, disabled=not llm_ready):
        with st.spinner("Summarising documents (map-reduce). This is slower ..."):
            on_summarise()

    for summary in summaries or ():
        with st.expander(f"{summary.title} ({'LLM' if summary.llm_generated else 'structural'})"):
            st.markdown(summary.summary)
            if summary.topics:
                st.caption("Sections: " + ", ".join(summary.topics[:10]))


def render_config_panel(settings_public: dict[str, Any]) -> None:
    with st.expander("Configuration", expanded=False):
        st.json(settings_public)


def render_chat_history(history: Sequence[dict[str, Any]]) -> None:
    if not history:
        return
    st.subheader("Session history")
    for entry in reversed(history):
        st.markdown(f"**Q:** {entry['question']}")
        st.markdown(f"**A:** {entry['status']} - {entry['summary']}")


def truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


__all__ = [
    "configure_page",
    "render_chat",
    "render_chat_history",
    "render_config_panel",
    "render_document_manager",
    "render_header",
    "render_ingest_result",
    "render_sidebar",
    "render_summary",
    "STATUS_COLOURS",
    "build_citations",
    "RetrievedChunk",
]
