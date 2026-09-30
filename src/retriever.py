"""Hybrid retrieval with an explicit evidence gate.

Three retrievers are combined, because each fails differently:

* **Dense (FAISS + sentence embeddings)** finds passages that mean the same
  thing in different words. Weak on rare proper nouns, which is exactly the
  "name one specific UGV programme" case.
* **BM25** finds rare literal terms - programme names, "sense-and-avoid",
  "ordnance". Strong where dense retrieval drifts.
* **Evidence reranking** then re-orders the fused candidates by how much of the
  question they actually cover, and applies a per-document cap so a synthesis
  question is not monopolised by the largest PDF.

The final score is deliberately built from *absolute* quantities - cosine
similarity and term coverage - rather than from rank positions. A rank-derived
score always makes the top hit look perfect, which would defeat the point of
``MIN_RETRIEVAL_SCORE``. That threshold is the first of three grounding
layers: below it, the answerer refuses without calling the model at all.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from rank_bm25 import BM25Okapi

from .config import Settings, get_settings
from .embeddings import tokenize
from .errors import VectorStoreError
from .logging_utils import get_logger
from .models import Chunk, RetrievedChunk
from .vector_store import VectorStore

logger = get_logger(__name__)

#: Reciprocal-rank-fusion damping constant. The value from the original RRF
#: paper; it keeps a single top-1 hit from dominating a deep list.
RRF_K = 60

#: Weights for the final evidence score. Both terms are absolute.
DENSE_WEIGHT = 0.55
COVERAGE_WEIGHT = 0.45

#: Semantic floor. A cosine similarity below this means the query and the chunk
#: point in meaningfully different directions, and a chunk that is semantically
#: unrelated is not evidence no matter how many rare words it happens to share
#: with the question. Without this floor, "What is the capital of France?"
#: matched a passage that merely happens to contain the word "France".
MIN_DENSE_SIMILARITY = 0.10

#: Cap on how many chunks the dense retriever scores exhaustively. FAISS flat
#: search is O(n) and sub-100 ms at this size, so scoring the whole corpus is
#: cheap and gives every candidate a dense score, which is what makes the
#: semantic floor above applicable to every candidate rather than only the
#: dense top-k.
MAX_EXHAUSTIVE_DENSE = 20_000

#: Question words that carry no retrieval signal.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then than that this these those there here of in on at to
    for from by with without about into over under between during after before while
    is are was were be been being am do does did doing have has had having will would
    shall should can could may might must not no nor so such as it its it's they them
    their he she his her him you your i me my we us our who whom which what when where
    why how all any both each few more most other some only own same too very s t just
    don now also per via within upon across among towards toward using used use
    please tell give show list name explain describe according document documents
    provided pdf page pages section sections mention mentions mentioned state states
    stated say says said
    """.split()
)

#: Tokens that indicate the user is asking for a quantity. Used to keep the
#: honesty path honest: an explicit statistic request should not be satisfied by
#: vaguely related prose.
_QUANTITY_HINT_RE = re.compile(
    r"\b(percent(age)?|%|proportion|share|ratio|rate|how many|how much|number of|"
    r"average|median|statistic|figure)\b",
    re.IGNORECASE,
)


def content_terms(text: str) -> list[str]:
    """Retrieval-relevant tokens: lowercased, de-duplicated, stopword-free."""
    seen: dict[str, None] = {}
    for token in tokenize(text):
        if token in STOPWORDS or len(token) < 2:
            continue
        seen.setdefault(token, None)
    return list(seen)


def term_coverage(terms: Sequence[str], haystack: str) -> float:
    """Fraction of ``terms`` that appear in ``haystack``."""
    if not terms:
        return 0.0
    tokens = set(tokenize(haystack))
    if not tokens:
        return 0.0
    hits = sum(1 for term in terms if term in tokens)
    return hits / len(terms)


def asks_for_a_quantity(question: str) -> bool:
    """True when the question explicitly asks for a figure or statistic."""
    return bool(_QUANTITY_HINT_RE.search(question or ""))


@dataclass
class _Candidate:
    chunk: Chunk
    dense: float | None = None
    lexical: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    coverage: float = 0.0
    rrf: float = 0.0
    score: float = 0.0
    matched: tuple[str, ...] = ()


class BM25Index:
    """Sparse lexical index over the chunk corpus, kept in sync with FAISS."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._fingerprint: tuple[int, str] = (0, "")

    @property
    def size(self) -> int:
        return len(self._chunk_ids)

    def ensure(self, chunks: Sequence[Chunk]) -> None:
        """Rebuild the index if the corpus changed."""
        fingerprint = self._fingerprint_of(chunks)
        if self._bm25 is not None and fingerprint == self._fingerprint:
            return
        corpus = [
            tokenize(f"{chunk.title} {chunk.section or ''} {chunk.text}")
            for chunk in chunks
        ]
        # BM25Okapi fails on an all-empty corpus; guard it.
        if not any(corpus):
            self._bm25 = None
            self._chunk_ids = []
            self._fingerprint = fingerprint
            return
        self._bm25 = BM25Okapi(corpus)
        self._chunk_ids = [chunk.chunk_id for chunk in chunks]
        self._fingerprint = fingerprint
        logger.debug("BM25 index rebuilt over %d chunks", len(self._chunk_ids))

    @staticmethod
    def _fingerprint_of(chunks: Sequence[Chunk]) -> tuple[int, str]:
        if not chunks:
            return (0, "")
        # Cheap but sufficient: chunk count plus the last id and total length.
        total = sum(len(c.text) for c in chunks)
        return (len(chunks), f"{chunks[-1].chunk_id}:{total}")

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            range(len(scores)), key=lambda i: float(scores[i]), reverse=True
        )[:k]
        return [(self._chunk_ids[i], float(scores[i])) for i in ranked]


class Retriever:
    """Finds the chunks that can support an answer."""

    def __init__(
        self,
        vector_store: VectorStore,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = vector_store
        self.bm25 = BM25Index()

    # -- public API --------------------------------------------------------

    def search(
        self,
        question: str,
        *,
        k: int | None = None,
        document_ids: Iterable[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return up to ``k`` chunks ranked by evidence strength.

        Returns an empty list when the index is empty or nothing clears
        ``MIN_RETRIEVAL_SCORE``. Callers treat an empty list as "no evidence".
        """
        k = k or self.settings.top_k
        question = (question or "").strip()
        if not question:
            return []
        if self.store.is_empty:
            raise VectorStoreError("The vector index is empty. Upload a document first.")

        terms = content_terms(question)
        wanted = set(document_ids) if document_ids is not None else None

        candidates = self._collect_candidates(question, k, wanted)
        candidates = self._apply_dense_floor(candidates)
        if not candidates:
            return []

        self._apply_rrf(candidates)
        self._score(candidates, terms)

        ranked = sorted(candidates.values(), key=lambda c: c.score, reverse=True)
        selected = self._apply_document_diversity(ranked, k, wanted)
        selected = self._apply_document_cap(selected, k, wanted)

        results = [
            RetrievedChunk(
                chunk=c.chunk,
                score=round(c.score, 4),
                dense_score=round(c.dense, 4) if c.dense is not None else None,
                lexical_score=round(c.lexical, 4) if c.lexical is not None else None,
                dense_rank=c.dense_rank,
                lexical_rank=c.lexical_rank,
                matched_terms=c.matched,
            )
            for c in selected
            if c.score >= self.settings.min_retrieval_score
        ]
        logger.debug(
            "Retrieved %d/%d chunks for %r (best score %.3f)",
            len(results),
            len(candidates),
            question[:60],
            selected[0].score if selected else 0.0,
        )
        return results

    def has_evidence(self, question: str, **kwargs) -> bool:
        return bool(self.search(question, **kwargs))

    # -- internals ---------------------------------------------------------

    def _collect_candidates(
        self, question: str, k: int, wanted: set[str] | None
    ) -> dict[str, _Candidate]:
        mode = self.settings.retrieval_mode
        pool = max(self.settings.candidate_k, k)
        candidates: dict[str, _Candidate] = {}

        if mode in ("hybrid", "dense"):
            # Score the whole corpus densely so that the semantic floor can be
            # applied uniformly to every candidate, including ones that only
            # the lexical retriever found. The pool is the full chunk count
            # rather than the eligible count, because reference chunks occupy
            # most of the far tail and would otherwise push eligible chunks
            # out of the window.
            dense_k = min(self.store.chunk_count, MAX_EXHAUSTIVE_DENSE)
            dense_hits = self.store.search(
                question,
                dense_k,
                document_ids=wanted,
                include_references=not self.settings.exclude_reference_sections,
            )
            for rank, (chunk, score) in enumerate(dense_hits):
                # Record every chunk's dense score, including weak ones, so the
                # semantic floor can be applied uniformly afterwards rather
                # than only to whichever retriever happened to find it first.
                candidate = candidates.setdefault(chunk.chunk_id, _Candidate(chunk=chunk))
                candidate.dense = score
                candidate.dense_rank = rank

        if mode in ("hybrid", "lexical"):
            eligible = [
                chunk
                for chunk in self.store.all_chunks()
                if (wanted is None or chunk.document_id in wanted)
                and not (
                    chunk.is_reference and self.settings.exclude_reference_sections
                )
            ]
            self.bm25.ensure(eligible)
            by_id = {chunk.chunk_id: chunk for chunk in eligible}
            for rank, (chunk_id, score) in enumerate(
                self.bm25.search(question, pool)
            ):
                chunk = by_id.get(chunk_id)
                if chunk is None:
                    continue
                candidate = candidates.setdefault(chunk_id, _Candidate(chunk=chunk))
                candidate.lexical = score
                candidate.lexical_rank = rank

        return candidates

    def _apply_dense_floor(
        self, candidates: dict[str, _Candidate]
    ) -> dict[str, _Candidate]:
        """Discard chunks that are semantically unrelated to the question.

        Applied after fusion so that it governs every candidate. A chunk found
        only by BM25 still has a dense score, because the dense pass scores the
        whole eligible corpus.
        """
        floor = self.settings.min_dense_similarity
        kept = {
            chunk_id: candidate
            for chunk_id, candidate in candidates.items()
            if candidate.dense is None or candidate.dense >= floor
        }
        if len(kept) != len(candidates):
            logger.debug(
                "Dense floor %.2f removed %d of %d candidates",
                floor,
                len(candidates) - len(kept),
                len(candidates),
            )
        return kept

    @staticmethod
    def _apply_rrf(candidates: dict[str, _Candidate]) -> None:
        """Reciprocal rank fusion of the dense and lexical result lists."""
        for candidate in candidates.values():
            total = 0.0
            if candidate.dense_rank is not None:
                total += 1.0 / (RRF_K + candidate.dense_rank + 1)
            if candidate.lexical_rank is not None:
                total += 1.0 / (RRF_K + candidate.lexical_rank + 1)
            candidate.rrf = total

    def _score(self, candidates: dict[str, _Candidate], terms: Sequence[str]) -> None:
        """Compute the final, absolute evidence score for each candidate."""
        best_rrf = max((c.rrf for c in candidates.values()), default=1.0) or 1.0
        for candidate in candidates.values():
            haystack = " ".join(
                [
                    candidate.chunk.title,
                    candidate.chunk.section or "",
                    candidate.chunk.text,
                ]
            )
            candidate.coverage = term_coverage(terms, haystack)
            tokens = set(tokenize(haystack))
            candidate.matched = tuple(t for t in terms if t in tokens)

            # Dense similarity is absolute. When only the lexical retriever
            # contributed, fall back to the RRF rank share, which is weaker but
            # still monotone.
            dense_component = (
                candidate.dense
                if candidate.dense is not None
                else math.log1p(max(0.0, candidate.lexical or 0.0)) / 6.0
            )
            fused = candidate.rrf / best_rrf
            candidate.score = (
                DENSE_WEIGHT * max(0.0, dense_component)
                + COVERAGE_WEIGHT * candidate.coverage
                + 0.10 * fused
            )

    def _apply_document_diversity(
        self, candidates: list[_Candidate], k: int, wanted: set[str] | None
    ) -> list[_Candidate]:
        """Guarantee competitive documents a slot in the context.

        The opposite failure to the per-document cap: with the cap off, a large
        document can fill every slot. "Across all three documents, what are the
        risks of autonomous systems?" retrieved four passages from the UAV
        guide and none from the other two, so the model could not answer a
        cross-document question from a single-document context.

        This is additive rather than subtractive - it adds one passage from each
        *competitive* rival document and keeps every other passage ranked
        normally, so it cannot evict the evidence the cap used to destroy.

        A document only earns a slot if its best passage is within
        ``document_diversity_ratio`` of the best passage overall, which stops
        an unrelated document from being injected just because it exists.
        """
        ratio = self.settings.document_diversity_ratio
        if ratio <= 0 or wanted is not None or len(candidates) < 2:
            return candidates[:k]

        floor = self.settings.min_retrieval_score
        top_score = candidates[0].score
        if top_score <= 0:
            return candidates[:k]
        threshold = top_score * ratio

        best_per_document: dict[str, _Candidate] = {}
        for candidate in candidates:
            document_id = candidate.chunk.document_id
            if candidate.score < floor:
                break
            if document_id not in best_per_document:
                best_per_document[document_id] = candidate

        if len(best_per_document) < 2:
            return candidates[:k]

        reserved: list[_Candidate] = []
        seen: set[str] = {candidates[0].chunk.document_id}
        # Strongest rivals first, so a truncated reserve favours real evidence.
        rivals = sorted(
            (c for doc, c in best_per_document.items() if doc not in seen),
            key=lambda c: c.score,
            reverse=True,
        )
        for rival in rivals:
            if len(reserved) >= k - 1:
                break
            if rival.score < threshold:
                break
            reserved.append(rival)
            seen.add(rival.chunk.document_id)

        if not reserved:
            return candidates[:k]

        reserved_ids = {id(c) for c in reserved}
        # Fill only the remaining slots. Taking the top k *after* adding the
        # reserved chunks would drop them again when rivals score just below the
        # leader, which is precisely the case this pass exists to fix.
        filler = [c for c in candidates if id(c) not in reserved_ids][
            : k - len(reserved)
        ]
        chosen = filler + reserved
        chosen.sort(key=lambda c: c.score, reverse=True)
        logger.debug(
            "Reserved %d slot(s) for rival document(s): %s",
            len(reserved),
            ", ".join(sorted({c.chunk.document_id for c in reserved})),
        )
        return chosen[:k]

    def _apply_document_cap(
        self, candidates: list[_Candidate], k: int, wanted: set[str] | None
    ) -> list[_Candidate]:
        """Optionally limit how much any single document can contribute.

        Disabled by default (``MAX_CHUNKS_PER_DOCUMENT=0``). Capping produced
        measurably worse answers: for "name one specific UGV program and its
        developer" the cap of 3 was filled by three introductory UGV passages
        that merely repeat the word "UGV", which evicted the passage that
        actually names the programme. A wrong refusal is worse than an uneven
        spread of sources, so the cap is opt-in.
        """
        cap = self.settings.max_chunks_per_document
        if wanted is not None or cap <= 0:
            # When the user has explicitly filtered to specific documents, or
            # capping is switched off, ranking order alone decides.
            return candidates[:k]

        counts: Counter[str] = Counter()
        selected: list[_Candidate] = []
        for candidate in candidates:
            document_id = candidate.chunk.document_id
            if counts[document_id] >= cap:
                continue
            counts[document_id] += 1
            selected.append(candidate)
            if len(selected) >= k:
                break
        return selected


__all__ = [
    "Retriever",
    "BM25Index",
    "content_terms",
    "term_coverage",
    "asks_for_a_quantity",
    "STOPWORDS",
]
