"""Tests for citation construction.

The central invariant: a page number in the UI is always derived from a real
chunk, never from model text.
"""

from __future__ import annotations

from src.citations import (
    build_citations,
    citations_for_report,
    evidence_for_chunks,
    format_citation_line,
    format_evidence_block,
    markers_from_answer,
)
from src.grounding import verify


class TestBuildCitations:
    def test_markers_map_to_the_right_pages(self, mixed_chunks):
        citations = build_citations(mixed_chunks)
        assert [c.marker for c in citations] == ["S1", "S2", "S3"]
        assert [c.page for c in citations] == [2, 14, 6]

    def test_subset_of_markers(self, mixed_chunks):
        citations = build_citations(mixed_chunks, ["S3"])
        assert len(citations) == 1
        assert citations[0].title == "Unmanned ground vehicle"
        assert citations[0].page == 6

    def test_unknown_markers_are_skipped(self, mixed_chunks):
        assert build_citations(mixed_chunks, ["S9"]) == []

    def test_duplicate_markers_are_collapsed(self, mixed_chunks):
        assert len(build_citations(mixed_chunks, ["S1", "S1", "S2"])) == 2

    def test_citation_carries_real_provenance(self, ugv_chunk):
        citation = build_citations([ugv_chunk])[0]
        assert citation.chunk_id == ugv_chunk.chunk.chunk_id
        assert citation.filename == ugv_chunk.chunk.filename
        assert citation.section == "Military"
        assert "THeMIS" in citation.excerpt
        assert citation.label == "Unmanned ground vehicle - p.6"

    def test_empty_input(self):
        assert build_citations([]) == []


class TestCitationsForReport:
    def test_uses_only_cited_markers(self, mixed_chunks):
        report = verify("Autonomy varies widely [S2].", mixed_chunks)
        citations = citations_for_report(report, mixed_chunks)
        assert [c.marker for c in citations] == ["S2"]
        assert citations[0].page == 14

    def test_falls_back_to_top_chunk_when_nothing_cited(self, mixed_chunks):
        report = verify("no markers at all", mixed_chunks, require_citations=False)
        citations = citations_for_report(report, mixed_chunks)
        assert len(citations) == 1
        assert citations[0].marker == "S1"

    def test_falls_back_when_marker_is_unknown(self, mixed_chunks):
        report = verify("something [S9]", mixed_chunks)
        assert len(citations_for_report(report, mixed_chunks)) == 1


class TestEvidencePanel:
    def test_evidence_covers_every_passage(self, mixed_chunks):
        citations = evidence_for_chunks(mixed_chunks)
        assert [c.marker for c in citations] == ["S1", "S2", "S3"]

    def test_evidence_block_renders_pages(self, mixed_chunks):
        block = format_evidence_block(evidence_for_chunks(mixed_chunks))
        assert "(p. 2)" in block
        assert "(p. 14)" in block
        assert "THeMIS" in block

    def test_citation_line_format(self, mixed_chunks):
        line = format_citation_line(evidence_for_chunks(mixed_chunks)[0])
        assert line == "[S1] Electronic warfare - p. 2, Subdivisions"


class TestMarkerHelpers:
    def test_markers_from_answer(self):
        assert markers_from_answer("a [S1] b [S2] a [S1]") == ["S1", "S2"]
