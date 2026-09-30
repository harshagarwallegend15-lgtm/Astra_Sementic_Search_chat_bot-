"""Tests for the answering orchestration.

These use stub LLMs, so they verify the *logic* of the pipeline - the refusal
path, the fabrication catch, the echo retry - without a running model.
"""

from __future__ import annotations

import pytest

from src.answerer import Answerer
from src.errors import AnsweringError
from src.prompts import SYSTEM_PROMPT
from src.retriever import Retriever
from src.summarizer import DocumentSummarizer
from src.vector_store import VectorStore

from .stubs import EchoingLLM, StubLLM, UnavailableLLM


def build_index(settings, chunks):
    store = VectorStore(settings, None)
    store.build(chunks)
    return store


def make_answerer(settings, chunks, llm):
    store = build_index(settings, chunks)
    retriever = Retriever(store, settings)
    return Answerer(retriever, llm, settings)


def chunks_from(retrieved_list):
    return [r.chunk for r in retrieved_list]


class TestAskPromptWiring:
    def test_system_prompt_is_always_used(self, settings, ew_chunk):
        llm = StubLLM("Three subdivisions [S1].")
        answerer = make_answerer(settings, chunks_from([ew_chunk]), llm)
        answerer.answer("What are the subdivisions?")
        assert llm.calls
        assert llm.calls[0][0] == SYSTEM_PROMPT

    def test_sources_reach_the_model(self, settings, ew_chunk):
        llm = StubLLM("Three subdivisions [S1].")
        make_answerer(settings, chunks_from([ew_chunk]), llm).answer("subdivisions?")
        prompt = llm.user_prompts[0]
        assert "[S1]" in prompt
        assert "electronic attack" in prompt
        assert "subdivisions?" in prompt


class TestGroundedPath:
    def test_clean_answer_is_returned_with_citations(self, settings, ew_chunk):
        llm = StubLLM(
            "Electronic warfare has three subdivisions: attack, protection and "
            "support [S1]."
        )
        result = make_answerer(
            settings, chunks_from([ew_chunk]), llm
        ).answer("What are the subdivisions?")
        assert result.status in ("grounded", "partially_grounded")
        assert result.citations
        assert result.citations[0].page == 2
        assert result.confidence > 0

    def test_unknown_marker_does_not_leak_into_citations(self, settings, ew_chunk):
        llm = StubLLM("Attack and protection [S1] and more [S9].")
        result = make_answerer(
            settings, chunks_from([ew_chunk]), llm
        ).answer("What are the subdivisions?")
        assert [c.marker for c in result.citations] == ["S1"]
        assert any("S9" in note for note in result.notes)


class TestFabricationCatch:
    def test_invented_percentage_is_removed_by_a_retry(self, settings, uav_chunk):
        llm = StubLLM(
            ["80% of UAV missions are fully autonomous [S1].",
             "The supplied documents do not state this. Autonomy varies widely [S1]."]
        )
        result = make_answerer(
            settings, chunks_from([uav_chunk]), llm
        ).answer("What percentage are autonomous?")
        assert "80%" not in result.answer
        assert "do not state" in result.answer.lower()
        assert any("Rewrote the answer to remove 80" in n for n in result.notes)

    def test_retry_still_leaking_falls_back_to_refusal(self, settings, uav_chunk):
        llm = StubLLM(["80% then 90% of missions are autonomous [S1]."])
        result = make_answerer(
            settings, chunks_from([uav_chunk]), llm
        ).answer("What percentage are autonomous?")
        assert "80%" not in result.answer
        assert "90%" not in result.answer
        assert any("Withheld" in note for note in result.notes)

    def test_refusal_fallback_when_second_call_fails(self, settings, uav_chunk):
        class HalfBroken(StubLLM):
            def complete(self, system_prompt, user_prompt, history=None):  # type: ignore[no-untyped-def]
                if self.calls:
                    raise RuntimeError("provider died")
                self.calls.append((system_prompt, user_prompt))
                return "80% of missions are autonomous [S1]."

        result = make_answerer(
            settings, chunks_from([uav_chunk]), HalfBroken("x")
        ).answer("What percentage are autonomous?")
        assert "80%" not in result.answer
        assert "do not state" in result.answer.lower()

    def test_number_correction_prompt_names_the_values(self, settings, uav_chunk):
        llm = StubLLM(
            ["80% of missions are autonomous [S1].",
             "The documents do not state this [S1]."]
        )
        make_answerer(settings, chunks_from([uav_chunk]), llm).answer(
            "What percentage are autonomous?"
        )
        assert len(llm.calls) == 2
        assert "80" in llm.user_prompts[1]
        assert "cannot be" in llm.user_prompts[1]

    def test_supported_numbers_are_kept(self, settings, ew_chunk):
        llm = StubLLM("There are three major subdivisions [S1].")
        result = make_answerer(
            settings, chunks_from([ew_chunk]), llm
        ).answer("How many subdivisions?")
        assert "three" in result.answer
        assert len(llm.calls) == 1
        assert not any("Rewrote" in n for n in result.notes)


class TestEchoRetry:
    def test_echo_triggers_a_nudge_retry(self, settings, ew_chunk):
        llm = EchoingLLM()
        answerer = make_answerer(settings, chunks_from([ew_chunk]), llm)
        answerer.answer("What are the subdivisions?")
        assert len(llm.calls) >= 2
        assert "previous reply repeated" in llm.user_prompts[1]

    def test_persistent_echo_is_flagged_to_the_user(self, settings, ew_chunk):
        result = make_answerer(
            settings, chunks_from([ew_chunk]), EchoingLLM()
        ).answer("What are the subdivisions?")
        assert any("repeated the source text" in note for note in result.notes)
        assert result.citations


class TestNoEvidencePath:
    def test_high_threshold_refuses_without_calling_the_llm(self, settings, ew_chunk):
        settings.min_retrieval_score = 1.01
        llm = StubLLM("should never be used")
        result = make_answerer(
            settings, chunks_from([ew_chunk]), llm
        ).answer("What is the capital of France?")
        assert result.status == "no_context"
        assert llm.calls == []
        assert "cannot answer it" in result.answer

    def test_refusal_names_the_document_count(self, settings, ew_chunk):
        settings.min_retrieval_score = 1.01
        result = make_answerer(
            settings, chunks_from([ew_chunk]), StubLLM()
        ).answer("anything")
        assert "1 indexed document," in result.answer

    def test_empty_question_is_rejected(self, settings, ew_chunk):
        with pytest.raises(AnsweringError):
            make_answerer(settings, chunks_from([ew_chunk]), StubLLM()).answer("   ")


class TestMissingLLM:
    def test_returns_evidence_instead_of_failing(self, settings, ew_chunk):
        result = make_answerer(settings, chunks_from([ew_chunk]), None).answer("subdivisions?")
        assert result.status == "ungrounded"
        assert "No LLM provider is configured" in result.answer
        assert result.citations
        assert any("LLM_PROVIDER" in note for note in result.notes)

    def test_provider_outage_is_reported_cleanly(self, settings, ew_chunk):
        with pytest.raises(Exception):
            make_answerer(
                settings, chunks_from([ew_chunk]), UnavailableLLM()
            ).answer("subdivisions?")


class TestEvidenceOnly:
    def test_evidence_never_calls_the_llm(self, settings, ew_chunk):
        llm = StubLLM("unused")
        answerer = make_answerer(settings, chunks_from([ew_chunk]), llm)
        citations = answerer.evidence("What are the subdivisions?")
        assert citations
        assert llm.calls == []
        assert citations[0].page == 2

    def test_stream_yields_text(self, settings, ew_chunk):
        llm = StubLLM("Three subdivisions [S1].")
        answerer = make_answerer(settings, chunks_from([ew_chunk]), llm)
        text = "".join(answerer.stream("What are the subdivisions?"))
        assert "subdivisions" in text


class TestSummarizer:
    def test_extractive_fallback_without_an_llm(self, settings, ew_chunk):
        summary = DocumentSummarizer(settings, None).summarize(
            "Electronic warfare", ew_chunk.chunk.document_id, [ew_chunk.chunk]
        )
        assert not summary.llm_generated
        assert summary.summary
        assert summary.pages == [2]

    def test_empty_document_is_handled(self, settings):
        summary = DocumentSummarizer(settings, None).summarize("Empty", "d", [])
        assert "no indexable body text" in summary.summary

    def test_llm_summary_is_used_and_checked(self, settings, ew_chunk):
        llm = StubLLM("Electronic warfare covers attacks and protection [S1].")
        summary = DocumentSummarizer(settings, llm).summarize(
            "Electronic warfare", ew_chunk.chunk.document_id, [ew_chunk.chunk]
        )
        assert summary.llm_generated
        assert "attacks" in summary.summary

    def test_llm_summary_with_invented_numbers_falls_back(self, settings, ew_chunk):
        llm = StubLLM("Electronic warfare has 97 subdivisions [S1].")
        summary = DocumentSummarizer(settings, llm).summarize(
            "Electronic warfare", ew_chunk.chunk.document_id, [ew_chunk.chunk]
        )
        assert not summary.llm_generated
        assert "97" not in summary.summary
