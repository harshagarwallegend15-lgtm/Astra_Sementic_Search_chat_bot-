"""ASTRA INTEL - application entry point.

Run with::

    streamlit run app.py

Wiring only: session state holds one pipeline and one LLM client, and the
panels in ``src.ui`` render what the pipeline returns.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402
from src.errors import AstraIntelError  # noqa: E402
from src.llm import build_llm_client  # noqa: E402
from src.logging_utils import setup_logging  # noqa: E402
from src.pipeline import AstraPipeline  # noqa: E402
from src.ui import (  # noqa: E402
    TURNS_KEY,
    configure_page,
    render_chat,
    render_config_panel,
    render_conversation,
    render_document_manager,
    render_header,
    render_ingest_result,
    render_sidebar,
    render_summary,
)


@st.cache_resource(show_spinner="Loading embedding model and index ...")
def get_pipeline() -> AstraPipeline:
    """One pipeline per server process; the index is loaded from disk."""
    settings = get_settings()
    setup_logging(settings.log_level)
    pipeline = AstraPipeline(settings)
    pipeline.load()
    return pipeline


@st.cache_resource(show_spinner=False)
def get_llm(settings, provider: str, model: str, base_url: str, key: str):
    """One LLM client per configuration."""
    try:
        return build_llm_client(settings)
    except AstraIntelError as exc:
        st.session_state["llm_error"] = exc.user_message
        return None
    except Exception as exc:  # noqa: BLE001
        st.session_state["llm_error"] = (
            f"Could not connect to the language model: {exc}"
        )
        return None


def suggested_questions() -> list[str]:
    path = PROJECT_ROOT / "example-questions.md"
    if not path.is_file():
        return []
    import re

    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\d+[\.\)]\s+(.*\S)\s*$", line.strip())
        if match:
            questions.append(match.group(1).strip().strip('"'))
    return questions


def _docs_key() -> str:
    return "documents"


def refresh_documents(pipeline: AstraPipeline) -> list:
    st.session_state[_docs_key()] = pipeline.documents()
    return st.session_state[_docs_key()]


def main() -> None:
    configure_page()
    render_header()

    pipeline = get_pipeline()
    settings = pipeline.settings
    llm_error: str | None = st.session_state.get("llm_error")
    llm = get_llm(
        settings,
        settings.llm_provider,
        settings.llm_model,
        settings.resolved_base_url,
        settings.llm_api_key,
    )
    documents = st.session_state.get(_docs_key()) or refresh_documents(pipeline)

    # Sidebar first: status, documents, briefings, configuration. The main pane
    # is left entirely to the conversation.
    render_sidebar(
        settings.public_dict(),
        pipeline.stats(),
        documents,
        llm_ready=llm is not None,
        llm_error=llm_error,
    )
    render_document_manager(
        on_ingest=_make_ingest_handler(pipeline),
        on_rebuild=_make_rebuild_handler(pipeline),
        on_clear=_make_clear_handler(pipeline),
    )
    render_summary(
        st.session_state.get("summaries"),
        _make_summary_func(pipeline, llm),
        llm_ready=llm is not None,
    )
    render_config_panel(settings.public_dict())

    if st.session_state.get("ingest_result") is not None:
        render_ingest_result(st.session_state.pop("ingest_result"))

    questions = suggested_questions()
    render_conversation(st.session_state.get(TURNS_KEY, []), questions)
    render_chat(
        answer_func=_make_answer_func(pipeline, llm),
        evidence_func=_make_evidence_func(pipeline),
        suggested_questions=questions,
    )


def _make_ingest_handler(pipeline: AstraPipeline):
    def handler(files: list[tuple[str, bytes]]) -> None:
        try:
            with st.spinner("Extracting, chunking and embedding ..."):
                result = pipeline.ingest_pdfs(files)
            st.session_state["ingest_result"] = result
        except AstraIntelError as exc:
            st.error(exc.user_message)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not index those files: {exc}")
        finally:
            refresh_documents(pipeline)
            st.rerun()

    return handler


def _make_rebuild_handler(pipeline: AstraPipeline):
    def handler() -> None:
        try:
            with st.spinner("Rebuilding the index from the sample documents ..."):
                result = pipeline.index_starter_documents(rebuild=True)
            st.session_state["ingest_result"] = result
        except AstraIntelError as exc:
            st.error(exc.user_message)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not rebuild the index: {exc}")
        finally:
            refresh_documents(pipeline)
            st.rerun()

    return handler


def _make_clear_handler(pipeline: AstraPipeline):
    def handler() -> None:
        pipeline.clear_index()
        st.session_state["summaries"] = []
        # The transcript cites pages that no longer exist, so it goes too.
        st.session_state[TURNS_KEY] = []
        refresh_documents(pipeline)
        st.rerun()

    return handler


def _make_answer_func(pipeline: AstraPipeline, llm):
    def answer(question: str):
        return pipeline.answerer(llm).answer(question)

    return answer


def _make_evidence_func(pipeline: AstraPipeline):
    def evidence(question: str) -> list:
        return pipeline.answerer(None).evidence(question)

    return evidence


def _make_summary_func(pipeline: AstraPipeline, llm):
    def summarise() -> None:
        try:
            st.session_state["summaries"] = pipeline.summaries(llm)
        except AstraIntelError as exc:
            st.error(exc.user_message)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not generate summaries: {exc}")

    return summarise


if __name__ == "__main__":
    main()
