"""Tests for the vector store, the hybrid retriever and the answerer.

These use the deterministic hashing embedder so the suite runs offline and in
seconds. Retrieval *quality* against the real PDFs is covered separately by
``scripts/evaluate.py``; what is asserted here is the logic, not the ranking.
"""

from __future__ import annotations

import pytest

from src.chunker import Chunker, join_lines, split_sentences
from src.config import Settings
from src.models import Chunk, RetrievedChunk
from src.vector_store import (
    VectorStore,
    chunk_embedding_text,
    l2_to_cosine,
)
from src.retriever import (
    Retriever,
    asks_for_a_quantity,
    content_terms,
    term_coverage,
)


def make_chunks(count: int = 12, doc: str = "d1", title: str = "Doc") -> list[Chunk]:
    topics = [
        "Electronic warfare divides into electronic attack, protection and support.",
        "Jamming degrades a radar by interfering with its return signal.",
        "Deception uses false targets to mislead an adversary sensor.",
        "An unmanned ground vehicle carries explosive ordnance disposal equipment.",
        "THeMIS is a tracked robot made by Milrem Robotics.",
        "Autonomy levels range from manual piloting to full automation.",
        "A UAV flies under remote control or autonomously along a planned route.",
        "Sense and avoid lets an aircraft detect and avoid other aircraft.",
        "A robot dog is used for inspection and perimeter patrol.",
        "The Ripsaw is a tracked explosive ordnance disposal robot.",
        "Countermeasures reduce the effectiveness of enemy electronic attack.",
        "Battery capacity limits the endurance of small unmanned aircraft.",
    ]
    return [
        Chunk(
            chunk_id=f"{doc}-p{i + 1:03d}-00",
            document_id=doc,
            filename=f"{title}.pdf",
            title=title,
            page=i + 1,
            text=topics[i % len(topics)],
            section=f"Section {i + 1}",
        )
        for i in range(count)
    ]


def _override(store: VectorStore, **kwargs) -> Settings:
    """Settings sharing a store's directories, with per-test overrides."""
    base = {
        "vectorstore_dir": store.directory,
        "uploads_dir": store.directory,
        "embedding_backend": "hashing",
        "min_dense_similarity": -1.0,
        "min_retrieval_score": 0.0,
    }
    base.update(kwargs)
    return Settings(**base)


@pytest.fixture
def store(settings) -> VectorStore:
    store = VectorStore(settings, None)
    store.build(make_chunks())
    return store


class TestL2ToCosine:
    def test_identical_vectors_are_cosine_one(self):
        assert l2_to_cosine(0.0) == pytest.approx(1.0)

    def test_orthogonal_vectors_are_cosine_zero(self):
        assert l2_to_cosine(2.0) == pytest.approx(0.0)

    def test_known_value(self):
        # cos = 0.75  =>  squared L2 = 2 * (1 - 0.75) = 0.5
        assert l2_to_cosine(0.5) == pytest.approx(0.75)

    def test_out_of_range_inputs_are_clamped(self):
        assert l2_to_cosine(10.0) == -1.0
        assert l2_to_cosine(-5.0) == 1.0

    def test_does_not_square_the_distance_again(self):
        """Regression: squaring a squared distance collapses every score."""
        assert l2_to_cosine(1.0) == pytest.approx(0.5)
        assert l2_to_cosine(1.0) != pytest.approx(0.0)


class TestEmbeddingText:
    def test_includes_title_section_and_body(self):
        chunk = make_chunks(1)[0]
        text = chunk_embedding_text(chunk)
        assert "Doc" in text
        assert "Section 1" in text
        assert "Electronic warfare divides" in text


class TestVectorStore:
    def test_build_records_counts(self, store):
        assert store.chunk_count == 12
        assert store.document_count == 1
        assert not store.is_empty

    def test_build_with_no_chunks_raises(self, settings):
        with pytest.raises(Exception):
            VectorStore(settings, None).build([])

    def test_search_returns_known_chunks(self, store):
        hits = store.search("electronic warfare attack protection", 3)
        assert hits
        chunk, score = hits[0]
        assert isinstance(chunk, Chunk)
        assert isinstance(score, float)

    def test_search_scores_are_cosine_in_range(self, store):
        for _chunk, score in store.search("autonomy levels", 5):
            assert -1.0 <= score <= 1.0

    def test_reference_chunks_can_be_excluded(self, settings):
        chunks = make_chunks(3)
        chunks[0].is_reference = True
        store = VectorStore(settings, None)
        store.build(chunks)
        assert all(
            not chunk.is_reference
            for chunk, _ in store.search("electronic", 3)
        )
        assert any(
            chunk.is_reference
            for chunk, _ in store.search("electronic", 3, include_references=True)
        )

    def test_document_filter(self, store):
        other = make_chunks(4, doc="d2", title="Other")
        store.add(other)
        assert store.chunk_count == 16
        hits = store.search("electronic", 10, document_ids={"d2"})
        assert hits
        assert all(chunk.document_id == "d2" for chunk, _ in hits)

    def test_remove_document(self, store):
        store.add(make_chunks(4, doc="d2", title="Other"))
        assert store.remove_document("d2") == 4
        assert store.chunk_count == 12
        assert "d2" not in store.document_ids()

    def test_remove_unknown_document_is_a_noop(self, store):
        assert store.remove_document("nope") == 0

    def test_clear(self, store):
        store.clear()
        assert store.is_empty
        assert store.chunk_count == 0

    def test_save_and_load_round_trip(self, settings):
        first = VectorStore(settings, None)
        first.build(make_chunks(6))
        first.save()
        second = VectorStore(settings, None)
        assert second.load() is True
        assert second.chunk_count == 6
        assert second.document_count == 1

    def test_load_returns_false_without_an_index(self, settings):
        assert VectorStore(settings, None).load() is False

    def test_duplicate_chunk_ids_are_ignored_on_add(self, store):
        before = store.chunk_count
        store.add(make_chunks(3))
        assert store.chunk_count == before


class TestTextHelpers:
    def test_content_terms_drops_stopwords(self):
        terms = content_terms("What are the three main types of drone?")
        assert "the" not in terms
        assert "of" not in terms
        assert "drone" in terms

    def test_term_coverage(self):
        assert term_coverage(["uav", "autonomy"], "UAV autonomy levels") == 1.0
        assert term_coverage(["uav", "jammer"], "UAV autonomy levels") == 0.5
        assert term_coverage(["uav"], "nothing relevant") == 0.0

    def test_split_sentences(self):
        assert len(split_sentences("One. Two! Three?")) == 3

    def test_split_sentences_keeps_abbreviations_together(self):
        assert len(split_sentences("See p. 4 for details. Then stop.")) == 2

    def test_join_lines_joins_words_with_spaces(self):
        assert join_lines(["a word", "b"]) == "a word b"

    def test_join_lines_drops_soft_hyphens(self):
        assert join_lines(["unmanned-", "aerial"]) == "unmannedaerial"

    @pytest.mark.parametrize(
        "question,expected",
        [
            ("What percentage of missions are autonomous?", True),
            ("How many UGV programs are listed?", True),
            ("What is autonomy?", False),
        ],
    )
    def test_asks_for_a_quantity(self, question, expected):
        assert asks_for_a_quantity(question) is expected


class TestRetriever:
    def test_search_returns_ranked_chunks(self, store, settings):
        retriever = Retriever(store, settings)
        hits = retriever.search("electronic attack and protection", k=5)
        assert hits
        assert [hit.score for hit in hits] == sorted(
            (hit.score for hit in hits), reverse=True
        )

    def test_every_hit_has_a_dense_score(self, store, settings):
        retriever = Retriever(store, settings)
        for hit in retriever.search("autonomy", k=8):
            assert hit.dense_score is not None

    def test_dense_floor_drops_weak_matches(self, store):
        settings = _override(store, min_dense_similarity=0.999)
        retriever = Retriever(store, settings)
        assert retriever.search("autonomy levels for unmanned aircraft", k=8) == []

    def test_score_gate_drops_everything(self, store):
        settings = _override(store, min_retrieval_score=1.01)
        assert Retriever(store, settings).search("autonomy", k=5) == []

    def test_document_cap_is_respected(self, store):
        settings = _override(store, max_chunks_per_document=1)
        hits = Retriever(store, settings).search("electronic warfare", k=10)
        assert len(hits) <= 1

    def test_unknown_document_filter_returns_nothing(self, store, settings):
        hits = Retriever(store, settings).search("autonomy", k=5, document_ids={"nope"})
        assert hits == []

    def test_has_evidence_predicate(self, store, settings):
        retriever = Retriever(store, settings)
        assert retriever.has_evidence("electronic warfare attack") in (True, False)

    def test_empty_question_returns_nothing(self, store, settings):
        assert Retriever(store, settings).search("", k=5) == []

    def test_dense_mode_works(self, store):
        settings = _override(store, retrieval_mode="dense")
        assert Retriever(store, settings).search("autonomy", k=5)

    def test_lexical_mode_works(self, store):
        settings = _override(store, retrieval_mode="lexical")
        assert Retriever(store, settings).search("THeMIS", k=5)


class TestChunkerUnits:
    def test_chunk_document_splits_large_text(self, settings):
        from src.models import DocumentMeta, PageContent, SectionHeading, TextLine
        from src.pdf_processor import (
            DocumentMeta as _DM,
            ExtractedDocument,
            RawBlock,
            RawLine,
        )

        long_text = " ".join(
            f"Sentence number {i} about autonomy and control." for i in range(200)
        )
        heading = SectionHeading("Autonomy", 2, 1, 1)
        page = PageContent(
            page=1,
            lines=[TextLine(heading.title, 18.0, True, 50.0, 40.0),
                   TextLine(long_text, 12.0, False, 50.0, 100.0)],
            headings=[heading],
            body_size=12.0,
        )
        raw_page = [
            RawBlock(
                block_id=0,
                page=1,
                lines=[
                    RawLine(
                        heading.title, 18.0, True, 50.0, 40.0, page=1, block_id=0
                    )
                ],
            ),
            RawBlock(
                block_id=1,
                page=1,
                lines=[
                    RawLine(
                        long_text, 12.0, False, 50.0, 100.0, page=1, block_id=1
                    )
                ],
            ),
        ]
        extracted = ExtractedDocument(
            meta=_DM(document_id="d", filename="f.pdf", title="T", page_count=1),
            pages=[page],
            body_size=12.0,
            raw_pages=[raw_page],
        )
        chunks = Chunker(settings).chunk_document(extracted)
        assert len(chunks) > 1
        assert all(chunk.page == 1 for chunk in chunks)
        assert all(chunk.text.strip() for chunk in chunks)
        assert all(chunk.section for chunk in chunks)
