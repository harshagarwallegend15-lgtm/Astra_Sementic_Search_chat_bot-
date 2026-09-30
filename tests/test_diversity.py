"""Tests for cross-document context diversity.

Without this, one large document can fill every context slot and a
cross-document question ("across all three documents, what are the risks...")
becomes unanswerable from a single-document context.
"""

from __future__ import annotations

import pytest

from src.config import Settings
from src.retriever import Retriever
from src.vector_store import VectorStore

from .conftest import _chunk


def build(settings, chunks):
    store = VectorStore(settings, None)
    store.build([c.chunk for c in chunks])
    return Retriever(store, settings)


@pytest.fixture
def lopsided(settings):
    """Eight strong UAV passages, one competitive passage from each rival."""
    chunks = [
        _chunk(f"autonomy level {i} describes the risks of autonomous flight", 10 + i,
               "UAV", "Autonomy", doc="uav")
        for i in range(8)
    ] + [
        _chunk("autonomy risks include electronic attack against autonomous systems",
               2, "EW", "Electronic attack", doc="ew"),
        _chunk("autonomy risks include a control system failing to detect obstacles",
               4, "UGV", "Control systems", doc="ugv"),
    ]
    return build(settings, chunks)


class TestDocumentDiversity:
    def test_rival_documents_get_a_slot(self, lopsided):
        docs = {c.chunk.document_id for c in lopsided.search("autonomy risks")}
        assert "uav" in docs
        assert len(docs) >= 2

    def test_all_three_documents_can_reach_the_context(self, lopsided):
        docs = {c.chunk.document_id for c in lopsided.search("autonomy risks")}
        assert docs == {"uav", "ew", "ugv"}

    def test_it_never_exceeds_top_k(self, lopsided):
        assert len(lopsided.search("autonomy risks")) <= lopsided.settings.top_k

    def test_it_adds_evidence_rather_than_replacing_it(self, lopsided):
        """The point is additive: top-ranked passages must survive."""
        results = lopsided.search("autonomy risks")
        texts = {c.chunk.text for c in results}
        assert any("autonomy level 0" in t for t in texts)
        assert any("electronic attack" in t for t in texts)

    def test_results_stay_sorted_by_score(self, lopsided):
        scores = [c.score for c in lopsided.search("autonomy risks")]
        assert scores == sorted(scores, reverse=True)

    def test_disabling_diversity_restores_pure_ranking(self, lopsided):
        lopsided.settings.document_diversity_ratio = 0.0
        docs = [c.chunk.document_id for c in lopsided.search("autonomy risks")]
        assert set(docs) == {"uav"}

    def test_a_high_ratio_requires_closer_competition(self, lopsided):
        lopsided.settings.document_diversity_ratio = 0.99
        docs = {c.chunk.document_id for c in lopsided.search("autonomy risks")}
        assert docs == {"uav"}

    def test_the_ratio_gate_excludes_a_weak_rival(self, lopsided):
        """A rival that is not competitive with the leader is not injected."""
        lopsided.settings.document_diversity_ratio = 0.99
        docs = {c.chunk.document_id for c in lopsided.search("autonomy risks")}
        assert docs == {"uav"}

    def test_the_relevance_floor_excludes_a_weak_rival(self, settings):
        """Independent of diversity: a passage below the floor never reaches
        the context, however many documents it comes from."""
        store = VectorStore(settings, None)
        store.build([
            _chunk(f"quantum entanglement decoherence in photonics lab {i}", 1 + i,
                   "Physics", "Quantum", doc="physics").chunk
            for i in range(6)
        ] + [
            _chunk("radar electronic support measures detect emitters", 2,
                   "EW", "ESM", doc="ew").chunk,
        ])
        settings.min_retrieval_score = 0.5
        retriever = Retriever(store, settings)
        results = retriever.search("quantum entanglement decoherence")
        assert results
        assert {c.chunk.document_id for c in results} == {"physics"}

    def test_an_unrelated_document_is_not_injected_for_a_matching_one(
        self, settings
    ):
        store = VectorStore(settings, None)
        store.build([
            _chunk(f"quantum entanglement decoherence in photonics lab {i}", 1 + i,
                   "Physics", "Quantum", doc="physics").chunk
            for i in range(6)
        ] + [
            _chunk("radar electronic support measures detect emitters", 2,
                   "EW", "ESM", doc="ew").chunk,
        ])
        settings.min_dense_similarity = 0.5
        retriever = Retriever(store, settings)
        docs = {c.chunk.document_id for c in retriever.search("quantum entanglement decoherence")}
        assert docs == {"physics"}

    def test_explicit_document_filter_is_respected(self, lopsided):
        docs = {
            c.chunk.document_id
            for c in lopsided.search("autonomy risks", document_ids={"uav"})
        }
        assert docs == {"uav"}

    def test_single_document_corpus_is_untouched(self, settings):
        store = VectorStore(settings, None)
        store.build([
            _chunk(f"radar jamming disrupts a sensor {i}", 1 + i,
                   "EW", "EA", doc="ew").chunk
            for i in range(5)
        ])
        results = Retriever(store, settings).search("radar jamming disrupts a sensor")
        assert results
        assert {c.chunk.document_id for c in results} == {"ew"}


class TestDiversitySettings:
    def test_default_is_enabled_and_conservative(self):
        s = Settings()
        assert 0 < s.document_diversity_ratio <= 0.6

    def test_it_can_be_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("DOCUMENT_DIVERSITY_RATIO", "0.7")
        assert Settings.from_env().document_diversity_ratio == pytest.approx(0.7)
