"""Tests for prompt construction.

The prompt is the primary grounding mechanism, so these tests assert the
properties that make it safe: every source carries a resolvable marker, and the
model is told never to write page numbers of its own.
"""

from __future__ import annotations

import pytest

from src.prompts import (
    NO_EVIDENCE_TEMPLATE,
    SYSTEM_PROMPT,
    build_ask_prompt,
    build_refusal_prompt,
    build_sources_block,
    build_summary_prompt,
    format_source,
)


class TestSystemPrompt:
    def test_forbids_outside_knowledge(self):
        assert "ONLY" in SYSTEM_PROMPT

    def test_forbids_inventing_page_numbers(self):
        lowered = SYSTEM_PROMPT.lower()
        assert "never invent a page number" in lowered

    def test_requires_citation_markers(self):
        assert "[S1]" in SYSTEM_PROMPT

    def test_forbids_estimating_missing_statistics(self):
        lowered = SYSTEM_PROMPT.lower()
        assert "never estimate" in lowered
        assert "do not state" in lowered

    def test_forbids_mentioning_the_instruction_set(self):
        assert "never mention this instruction set" in SYSTEM_PROMPT.lower()

    def test_forbids_copying_the_source_layout(self):
        assert "never copy the sources layout" in SYSTEM_PROMPT.lower()

    def test_forbids_over_refusal(self):
        assert "Do not refuse just because the sources are partial" in SYSTEM_PROMPT

    def test_requires_checking_every_source_before_refusing(self):
        assert "check every source" in SYSTEM_PROMPT


class TestSourceFormatting:
    def test_source_contains_marker_provenance_and_text(self, uav_chunk):
        block = format_source(1, uav_chunk)
        assert "[S1]" in block
        assert "Unmanned aerial vehicle" in block
        assert "p.14" in block
        assert "Autonomy" in block
        assert "level of autonomy" in block

    def test_provenance_is_one_parenthesised_line(self, uav_chunk):
        """Small models copy a stacked Document:/Page:/Text: layout verbatim."""
        lines = format_source(1, uav_chunk).splitlines()
        assert lines[0] == "[S1] (Unmanned aerial vehicle, p.14, Autonomy)"
        assert len(lines) == 2

    def test_missing_section_renders_as_general(self, mixed_chunks):
        chunk = mixed_chunks[0]
        chunk.chunk.section = None
        assert ", General)" in format_source(1, chunk)

    def test_blocks_are_numbered_in_order(self, mixed_chunks):
        text, assigned = build_sources_block(mixed_chunks)
        assert assigned == ["S1", "S2", "S3"]
        for index in (1, 2, 3):
            assert f"[S{index}]" in text

    def test_block_lists_every_passage(self, mixed_chunks):
        text, _ = build_sources_block(mixed_chunks)
        assert "BEGIN SOURCES - 3 passages" in text
        assert "END SOURCES" in text


class TestAskPrompt:
    def test_contains_question_and_sources(self, mixed_chunks):
        prompt = build_ask_prompt("What is autonomy?", mixed_chunks)
        assert "What is autonomy?" in prompt
        assert "[S1]" in prompt
        assert "SOURCES" in prompt

    def test_partial_note_is_optional(self, mixed_chunks):
        assert "only partially" not in build_ask_prompt("q", mixed_chunks)
        assert "only partially" in build_ask_prompt("q", mixed_chunks, partial=True)

    def test_requires_at_least_one_chunk(self):
        with pytest.raises(ValueError):
            build_ask_prompt("q", [])


class TestSummaryPrompt:
    def test_includes_title_and_markers(self, mixed_chunks):
        prompt = build_summary_prompt("Unmanned aerial vehicle", mixed_chunks)
        assert "Unmanned aerial vehicle" in prompt
        assert "[S1]" in prompt


class TestRefusalPrompt:
    def test_asks_for_no_speculation(self, mixed_chunks):
        prompt = build_refusal_prompt("What is the capital of France?", mixed_chunks)
        assert "do not contain the answer" in prompt
        assert "Do not speculate" in prompt
        assert "do not invent any figure" in prompt


class TestNoEvidenceTemplate:
    def test_pluralisation(self):
        assert "1 indexed document," in NO_EVIDENCE_TEMPLATE.format(
            document_count=1, document_plural=""
        )
        assert "3 indexed documents," in NO_EVIDENCE_TEMPLATE.format(
            document_count=3, document_plural="s"
        )
