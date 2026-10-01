"""ASTRA INTEL - Streamlit presentation layer.

Deliberately thin: it renders what the pipeline returns and never does retrieval,
prompting or verification of its own. Every user-triggerable failure is caught and
rendered as a readable message rather than a traceback.

Layout is chat-primary. The conversation owns the main pane because it is the only
thing an analyst uses repeatedly; documents, summaries and configuration are
secondary and live in the sidebar, where they stay reachable without pushing the
transcript off screen.

Turns live in ``st.session_state["turns"]`` rather than being re-derived on every
run, so the transcript survives a rerun and each answer keeps its own citations.
"""

from __future__ import annotations

import html
import time
from typing import Any, Sequence

import streamlit as st

from .citations import build_citations
from .errors import AstraIntelError
from .models import AnswerResult, Citation, DocumentMeta, RetrievedChunk

# label, Streamlit alert kind, colour used by the inline status pill
STATUS_COLOURS = {
    "grounded": ("Grounded", "success", "#34d399"),
    "partially_grounded": ("Partially grounded", "warning", "#fbbf24"),
    "ungrounded": ("Not grounded", "error", "#fb7185"),
    "no_context": ("No supporting evidence", "error", "#fb7185"),
}

TURNS_KEY = "turns"

_THEME_CSS = """
<style>
/* ---------- shell ---------- */
.stApp {
  background:
    radial-gradient(1100px 620px at 78% -12%, rgba(56,189,248,.10), transparent 62%),
    radial-gradient(900px 520px at 4% 4%, rgba(99,102,241,.09), transparent 58%),
    #0a0f1c;
}
.block-container { padding-top: 2.1rem; padding-bottom: 5.5rem; max-width: 1180px; }
#MainMenu, footer, header [data-testid="stStatusWidget"] { visibility: hidden; height: 0; }

h1, h2, h3 { letter-spacing: -.015em; color: #f2f7ff; }
.astra-eyebrow {
  font-size: .70rem; font-weight: 700; letter-spacing: .17em; text-transform: uppercase;
  color: #7dd3fc; margin-bottom: .18rem;
}
.astra-sub { color: #93a4c3; font-size: .93rem; margin-bottom: .1rem; }

/* ---------- brand mark ---------- */
.astra-brand { display: flex; align-items: center; gap: .7rem; margin-bottom: .5rem; }
.astra-brand .glyph {
  width: 2.35rem; height: 2.35rem; border-radius: .7rem; flex: 0 0 auto;
  display: grid; place-items: center; font-size: 1.15rem;
  background: linear-gradient(140deg, #0ea5e9, #6366f1);
  box-shadow: 0 6px 20px rgba(14,165,233,.34);
}
.astra-brand .name { font-size: 1.42rem; font-weight: 700; color: #f2f7ff; line-height: 1.1; }
.astra-brand .tag { font-size: .76rem; color: #93a4c3; }

/* ---------- chat ---------- */
[data-testid="stChatMessage"] {
  background: transparent; padding: .1rem 0 .55rem 0; gap: .75rem;
}
[data-testid="stChatMessage"] .stAvatar { display: none; }
[data-testid="stChatMessageContent"] { font-size: .97rem; }
[data-testid="stChatMessageContent"] p { line-height: 1.66; }
[data-testid="stChatInput"] { border-radius: 1rem; }

.astra-turn {
  border-radius: .9rem; padding: .95rem 1.1rem; margin: .1rem 0 .3rem 0;
  border: 1px solid rgba(148,163,184,.16); background: rgba(21,29,49,.72);
}
.astra-turn.user {
  border-color: rgba(56,189,248,.30);
  background: linear-gradient(180deg, rgba(56,189,248,.12), rgba(56,189,248,.05));
}
.astra-q { font-weight: 600; color: #eaf4ff; font-size: .98rem; }

/* ---------- status pill ---------- */
.astra-meta { display: flex; align-items: center; gap: .55rem; flex-wrap: wrap; margin: .1rem 0 .7rem; }
.astra-pill {
  display: inline-block; padding: .13rem .62rem; border-radius: 999px;
  font-size: .715rem; font-weight: 700; letter-spacing: .045em; text-transform: uppercase;
  border: 1px solid currentColor;
}
.astra-pill.Grounded, .astra-pill.grounded { color: #34d399; background: rgba(52,211,153,.11); }
.astra-pill.partially_grounded { color: #fbbf24; background: rgba(251,191,36,.11); }
.astra-pill.ungrounded, .astra-pill.no_context { color: #fb7185; background: rgba(251,113,133,.11); }
.astra-meta .detail {
  font-family: ui-monospace, "Cascadia Mono", Menlo, monospace;
  font-size: .715rem; color: #93a4c3; letter-spacing: .01em;
}

/* ---------- citations ---------- */
.astra-src { margin-top: .5rem; border-top: 1px dashed rgba(148,163,184,.22); padding-top: .55rem; }
.astra-src > .label {
  font-size: .70rem; font-weight: 700; letter-spacing: .15em; text-transform: uppercase;
  color: #7dd3fc; margin-bottom: .3rem;
}
.astra-cite {
  display: block; border-left: 2px solid rgba(56,189,248,.55);
  padding: .34rem .7rem; margin: .3rem 0; border-radius: 0 .5rem .5rem 0;
  background: rgba(56,189,248,.055); font-size: .855rem; color: #dbe7f8;
}
.astra-cite .tag {
  font-family: ui-monospace, Menlo, monospace; color: #7dd3fc; font-weight: 700; margin-right: .35rem;
}
.astra-cite .where { color: #93a4c3; }
.astra-cite .score {
  float: right; font-family: ui-monospace, Menlo, monospace;
  font-size: .72rem; color: #7c8db0;
}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d1425 0%, #0a0f1c 100%);
  border-right: 1px solid rgba(148,163,184,.13);
}
section[data-testid="stSidebar"] .stCaption { color: #8496b5; }
[data-testid="stMetricValue"] { font-size: 1.32rem; color: #f2f7ff; }
[data-testid="stMetricLabel"] { font-size: .74rem; color: #8496b5; }

.astra-kv { font-size: .80rem; line-height: 1.5; }
.astra-kv b { color: #cfe0f5; font-weight: 600; }
.astra-kv span { color: #8496b5; }

.astra-doc {
  font-size: .82rem; border: 1px solid rgba(148,163,184,.14);
  border-radius: .55rem; padding: .42rem .55rem; margin: .28rem 0;
  background: rgba(21,29,49,.55);
}
.astra-doc .t { font-weight: 600; color: #e6eefb; display: block; }
.astra-doc .m { color: #8496b5; font-size: .735rem; }

/* ---------- empty state ---------- */
.astra-hero {
  border: 1px solid rgba(56,189,248,.20); border-radius: 1rem;
  background: linear-gradient(160deg, rgba(56,189,248,.075), rgba(99,102,241,.045));
  padding: 1.35rem 1.5rem; margin: .5rem 0 1.1rem;
}
.astra-hero h3 { margin: 0 0 .3rem 0; font-size: 1.06rem; }
.astra-hero p { margin: 0; color: #a8b8d4; font-size: .90rem; line-height: 1.6; }

/* ---------- widgets ---------- */
.stButton > button {
  border-radius: .55rem; font-weight: 600; font-size: .855rem;
  border: 1px solid rgba(148,163,184,.22); transition: all .14s ease;
}
.stButton > button:hover:not(:disabled) {
  border-color: rgba(56,189,248,.65); color: #bae6fd; transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
  background: linear-gradient(140deg, #0ea5e9, #2563eb); border: none; color: #fff;
  box-shadow: 0 4px 16px rgba(14,165,233,.26);
}
.stButton > button[kind="primary"]:hover { box-shadow: 0 7px 22px rgba(14,165,233,.38); }
[data-testid="stTextInput"] input { border-radius: .55rem; font-size: .93rem; }
[data-testid="stExpander"] {
  border: 1px solid rgba(148,163,184,.15); border-radius: .6rem; background: rgba(21,29,49,.45);
}
.astra-note { color: #8496b5; font-size: .78rem; line-height: 1.55; }
</style>
"""


def apply_theme() -> None:
    """Inject the stylesheet. Cheap enough to re-send on every rerun."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def configure_page() -> None:
    st.set_page_config(
        page_title="ASTRA INTEL",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()


def render_header() -> None:
    st.title("ASTRA INTEL")
    st.markdown(
        '<div class="astra-sub">Grounded defence-technology document analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="astra-note">Every claim is traced to a supplied PDF page. '
        "A number that does not appear in the retrieved text is withheld.</div>",
        unsafe_allow_html=True,
    )
    st.divider()


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------


def render_sidebar(
    settings_public: dict[str, Any],
    stats: dict[str, Any],
    documents: Sequence[DocumentMeta],
    llm_ready: bool,
    llm_error: str | None = None,
) -> None:
    """Status, then the model in use, then what is indexed."""
    with st.sidebar:
        st.markdown(
            '<div class="astra-eyebrow">Workspace</div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns(2)
        left.metric("Documents", stats.get("documents", 0))
        right.metric("Passages", stats.get("chunks", 0))

        rows = (
            ("LLM", f"{settings_public.get('LLM provider')} / {settings_public.get('LLM model')}"),
            ("Endpoint", settings_public.get("LLM base URL", "")),
            ("Embeddings", settings_public.get("Embedding model", "")),
            ("Retrieval", settings_public.get("retrieval mode", "hybrid")),
        )
        st.markdown(
            '<div class="astra-kv">'
            + "".join(f"<div><b>{html.escape(str(label))}:</b> <span>{html.escape(str(value or '-'))}</span></div>"
                      for label, value in rows)
            + "</div>",
            unsafe_allow_html=True,
        )

        if llm_error:
            st.error(llm_error)
        elif llm_ready:
            st.success("LLM ready")
        else:
            st.info("No LLM - retrieval and evidence only.")

        if documents:
            st.markdown(
                '<div class="astra-eyebrow" style="margin-top:.9rem">Indexed</div>',
                unsafe_allow_html=True,
            )
            for meta in documents:
                st.markdown(
                    f'<div class="astra-doc"><span class="t">{html.escape(meta.title)}</span>'
                    f'<span class="m">{meta.page_count} pages · {meta.chunk_count} passages</span></div>',
                    unsafe_allow_html=True,
                )


def render_document_manager(on_ingest, on_rebuild, on_clear) -> None:
    """Upload / rebuild / clear, all inside the sidebar."""
    with st.sidebar:
        st.markdown('<div class="astra-eyebrow">Documents</div>', unsafe_allow_html=True)
        uploads = st.file_uploader(
            "Add PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Text-based PDFs only. Scanned images have no text layer.",
            key="uploader",
        )
        if st.button("Index uploaded PDFs", width="stretch", disabled=not uploads):
            on_ingest([(item.name, item.getvalue()) for item in uploads])
        if st.button(
            "Rebuild sample index",
            width="stretch",
            help="Re-extract the bundled sample documents from scratch.",
        ):
            on_rebuild()

        if st.session_state.get("documents"):
            if st.button("Clear all documents", width="stretch"):
                on_clear()


def render_ingest_result(result) -> None:
    """Ingest feedback, shown as a transient toast next to the composer."""
    if result is None:
        return
    if result.added_count:
        titles = ", ".join(meta.title for meta in result.added)
        st.toast(
            f"Indexed {result.added_count} document(s): {titles} "
            f"(+{result.chunk_count} passages in {result.elapsed_ms} ms)",
            icon="✅",
        )
    if result.duplicates:
        st.toast("Already indexed, skipped: " + ", ".join(result.duplicates), icon="ℹ️")
    for filename, reason in (result.failures or {}).items():
        st.error(f"**{filename}** - {reason}")


def render_summary(summaries, on_summarise, llm_ready: bool) -> None:
    with st.sidebar:
        st.markdown(
            '<div class="astra-eyebrow" style="margin-top:.9rem">Briefings</div>',
            unsafe_allow_html=True,
        )
        if st.button("Generate document briefings", width="stretch", disabled=not llm_ready):
            with st.spinner("Summarising documents (map-reduce). This is slower ..."):
                on_summarise()

        for summary in summaries or ():
            origin = "LLM" if summary.llm_generated else "structural"
            with st.expander(f"{summary.title} ({origin})"):
                st.markdown(summary.summary)
                if summary.topics:
                    st.caption("Sections: " + ", ".join(summary.topics[:10]))


def render_config_panel(settings_public: dict[str, Any]) -> None:
    with st.sidebar:
        with st.expander("Configuration", expanded=False):
            st.json(settings_public)


# --------------------------------------------------------------------------
# conversation
# --------------------------------------------------------------------------


def new_turn(question: str) -> dict[str, Any]:
    return {"question": question, "result": None, "citations": [], "elapsed": 0.0, "error": None}


def push_turn(turn: dict[str, Any]) -> None:
    st.session_state.setdefault(TURNS_KEY, []).append(turn)


def clear_turns() -> None:
    st.session_state[TURNS_KEY] = []


def render_conversation(turns: Sequence[dict[str, Any]], suggested: Sequence[str]) -> None:
    """The transcript. Empty state doubles as the suggestion surface."""
    if not turns:
        st.markdown(
            '<div class="astra-hero"><h3>Ask the corpus</h3>'
            "<p>Retrieval runs across every indexed page, and the answer is verified "
            "against the text that was actually retrieved. If the documents do not "
            "cover something, you get that answer rather than a confident guess.</p></div>",
            unsafe_allow_html=True,
        )
        return

    for turn in turns:
        with st.chat_message("user"):
            st.markdown(f'<div class="astra-turn user"><div class="astra-q">{html.escape(turn["question"])}</div></div>',
                         unsafe_allow_html=True)
        with st.chat_message("assistant"):
            if turn.get("error"):
                st.error(turn["error"])
            elif turn.get("result") is not None:
                _render_result(turn["result"], turn["elapsed"], turn["question"])
            else:
                _render_evidence_turn(turn)


def _render_status_pill(result: AnswerResult, elapsed: float) -> None:
    label, _kind, colour = STATUS_COLOURS.get(result.status, ("Unknown", "warning", "#94a3b8"))
    bits = [f"confidence {result.confidence:.2f}", f"{elapsed:.1f}s"]
    if result.llm_model:
        bits.append(str(result.llm_model))
    st.markdown(
        f'<div class="astra-meta"><span class="astra-pill {html.escape(result.status)}" '
        f'style="color:{colour}">{html.escape(label)}</span>'
        f'<span class="detail">{" · ".join(html.escape(b) for b in bits)}</span></div>',
        unsafe_allow_html=True,
    )


def _render_result(result: AnswerResult, elapsed: float, question: str) -> None:
    _render_status_pill(result, elapsed)
    st.markdown(result.answer)

    for note in result.notes:
        st.caption(note)

    if result.citations:
        _render_sources(result.citations)
        st.download_button(
            "Download answer with sources",
            data=_as_markdown(result, question),
            file_name="astra-intel-answer.md",
            mime="text/markdown",
            key=f"dl_{abs(hash((question, result.answer))) % 10**10}",
        )

    with st.expander(f"Retrieved passages ({len(result.retrieved)})", expanded=False):
        if not result.retrieved:
            st.caption("Nothing cleared the relevance threshold.")
        for retrieved in result.retrieved:
            chunk = retrieved.chunk
            st.markdown(
                f"`score={retrieved.score:.3f}` **{chunk.title}** "
                f"p.{chunk.page} · {chunk.section or 'general'}"
            )
            st.text(truncate(chunk.text, 400))


def _render_sources(citations: Sequence[Citation]) -> None:
    """Citation list. Kept as one block so the marker and its page stay on the
    same line - the smoke test matches ``[S#] ... page N`` exactly."""
    blocks = ['<div class="astra-src"><div class="label">Sources</div>']
    for citation in citations:
        marker = citation.marker or "S"
        section = f" · {citation.section}" if citation.section else ""
        blocks.append(
            f'<span class="cite"><span class="score">{citation.score:.3f}</span>'
            f'<span class="tag">[{html.escape(marker)}]</span>'
            f'<b>{html.escape(citation.title)}</b>'
            f'<span class="where"> — page {citation.page}{html.escape(section)}</span></span>'
        )
    blocks.append("</div>")
    st.markdown("".join(blocks), unsafe_allow_html=True)
    for citation in citations:
        with st.expander(f"Verbatim passage — {citation.filename}, p.{citation.page}"):
            st.text(citation.excerpt)
            st.caption(f"chunk id: {citation.chunk_id}")


def _as_markdown(result: AnswerResult, question: str) -> str:
    lines = [f"# {question}", "", result.answer, "", "## Sources", ""]
    for citation in result.citations:
        section = f" ({citation.section})" if citation.section else ""
        lines.append(f"- **{citation.title}**, page {citation.page}{section}")
        lines.append(f"  > {truncate(citation.excerpt, 400)}")
    return "\n".join(lines)


def _render_evidence_turn(turn: dict[str, Any]) -> None:
    citations = turn.get("citations") or []
    if not citations:
        st.warning("No passage cleared the relevance threshold for that question.")
        return
    st.info(f"{len(citations)} passage(s) above the relevance threshold — no model was called.")
    _render_sources(citations)


def render_chat(answer_func, evidence_func, suggested_questions: Sequence[str]) -> None:
    """Composer plus, when a question was just sent, its answer.

    Both live in one function on purpose: the answer has to be rendered in the
    same script run as the click that produced it, because Streamlit re-executes
    this script from the top every time.
    """
    if not st.session_state.get("documents"):
        st.info("Index at least one PDF in the sidebar to start asking questions.")
        return

    suggested_click: str | None = None
    if suggested_questions:
        with st.expander("Suggested questions", expanded=not st.session_state.get(TURNS_KEY)):
            for question in suggested_questions:
                if st.button(question, key=f"sugg_{abs(hash(question)) % 10**10}", width="stretch"):
                    suggested_click = question

    if suggested_click is not None:
        # Put it in the box as well as answering it, so the question stays visible
        # and editable. Setting the key before the widget exists is the supported
        # way to seed a text_input.
        st.session_state["question_input"] = suggested_click

    question = st.text_input(
        "Your question",
        key="question_input",
        placeholder="e.g. What are the three primary mission types of electronic warfare?",
        label_visibility="collapsed",
    )

    columns = st.columns([1, 1, 3])
    ask = columns[0].button("Ask", type="primary", width="stretch")
    show_evidence = columns[1].button("Show evidence only", width="stretch")
    columns[2].caption(
        "Answers are verified against the retrieved pages."
    )

    should_ask = bool(ask or show_evidence or suggested_click)
    if not should_ask:
        return

    if suggested_click:
        question = suggested_click
    question = (question or "").strip()
    if not question:
        st.warning("Enter a question first.")
        return

    turn = new_turn(question)
    started = time.perf_counter()

    if show_evidence and not ask:
        try:
            turn["citations"] = list(evidence_func(question))
        except AstraIntelError as exc:
            turn["error"] = exc.user_message
        except Exception as exc:  # noqa: BLE001 - never surface a traceback
            turn["error"] = f"Retrieval failed: {exc}"
        turn["elapsed"] = time.perf_counter() - started
        push_turn(turn)
        st.rerun()
        return

    with st.spinner("Searching the documents and composing a grounded answer ..."):
        try:
            turn["result"] = answer_func(question)
        except AstraIntelError as exc:
            turn["error"] = exc.user_message
        except Exception as exc:  # noqa: BLE001 - never surface a traceback
            turn["error"] = f"Could not answer that question: {exc}"
    turn["elapsed"] = time.perf_counter() - started
    push_turn(turn)
    st.rerun()


def render_chat_history(history: Sequence[dict[str, Any]]) -> None:
    """Kept for the evaluation/debug callers that seed a synthetic history."""
    for entry in reversed(list(history or ())):
        with st.chat_message("user"):
            st.markdown(str(entry.get("question", "")))
        with st.chat_message("assistant"):
            st.caption(f"{entry.get('status', '')} - {entry.get('summary', '')}")


def truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


__all__ = [
    "STATUS_COLOURS",
    "TURNS_KEY",
    "apply_theme",
    "clear_turns",
    "configure_page",
    "new_turn",
    "push_turn",
    "render_chat",
    "render_chat_history",
    "render_config_panel",
    "render_conversation",
    "render_document_manager",
    "render_header",
    "render_ingest_result",
    "render_sidebar",
    "render_summary",
    "truncate",
    "build_citations",
    "RetrievedChunk",
]