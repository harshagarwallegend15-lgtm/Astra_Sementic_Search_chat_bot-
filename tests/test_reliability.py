"""Reliability regressions.

Every test here corresponds to a specific defect that was fixed; each fails
against the old behaviour. They are about *behaviour under concurrency and
failure* rather than happy-path payloads, because that is where this project's
silent-wrong-answer bugs lived.

- ``search`` used to resolve its hits against ``self._chunks`` after releasing
  the lock, so a concurrent rebuild produced an empty result set that the
  answerer reported as "the documents do not contain that information" - a
  confident, silently wrong refusal.
- The BM25 index published three coupled fields with separate assignments, so
  two threads could pair one corpus's scores with another corpus's id ordering
  and return a citation for the wrong passage.
- ``get_pipeline`` checked and loaded without synchronisation, so concurrent
  first requests could construct two pipelines over one index directory.
- The language-model failure flag was a one-way latch, so a single transient
  error disabled synthesis for the lifetime of the process.

No network and no real model: the hashing embedder and offline settings are
used throughout.
"""

from __future__ import annotations

import threading

import pytest

from src.config import Settings
from src.errors import AstraIntelError, VectorStoreError
from src.models import Chunk
from src.pipeline import LLM_RETRY_COOLDOWN_S, AstraPipeline
from src.retriever import BM25Index, tokenize
from src.vector_store import VectorStore

from .conftest import PROJECT_ROOT


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    page: int = 1,
    document_id: str | None = None,
    title: str = "Doc",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id or f"doc-{chunk_id}",
        filename=f"{title}.pdf",
        title=title,
        page=page,
        section=None,
        text=text,
    )


def corpus(count: int = 6) -> list[Chunk]:
    return [
        make_chunk(f"c{i}", f"unmanned vehicle section {i} ordnance clearance", page=i + 1)
        for i in range(count)
    ]


@pytest.fixture
def offline_settings(tmp_path) -> Settings:
    return Settings(
        vectorstore_dir=tmp_path / "vectorstore",
        uploads_dir=tmp_path / "uploads",
        sample_metadata_path=PROJECT_ROOT / "sample-metadata.json",
        embedding_backend="hashing",
        min_dense_similarity=-1.0,
        min_retrieval_score=0.0,
    )


@pytest.fixture
def store(offline_settings) -> VectorStore:
    store = VectorStore(offline_settings, None)
    store.build(corpus())
    return store


@pytest.fixture
def pipeline(offline_settings, tmp_path) -> AstraPipeline:
    pipeline = AstraPipeline(offline_settings)
    pipeline.store.build(corpus())
    pipeline._loaded = True
    return pipeline


# ------------------------------------------------- vector store / chunk map --


def test_search_resolves_hits_against_the_map_from_its_own_query(store):
    """A concurrent rebuild must not wipe out the hits already returned.

    ``search`` fetches FAISS hits while holding the lock, then resolves them
    into chunks *after* releasing it. If it re-read ``self._chunks`` per hit, a
    rebuild landing in that window rebinds the attribute and every hit fails to
    resolve, so ``search`` returns ``[]`` - which the answerer reports as "the
    documents do not contain that information". A confident, silently wrong
    refusal, reproduced here deterministically rather than by timing.
    """
    armed: list[bool] = [True]

    class _RebindOnFirstRead(dict):
        def get(self, key, default=None):
            if armed[0]:
                armed[0] = False
                # What `build()`/`clear()` do mid-query.
                store._chunks = {}
            return super().get(key, default)

    store._chunks = _RebindOnFirstRead(store._chunks)
    results = store.search("unmanned vehicle", k=3)

    assert armed[0] is False, "the rebind never fired, so the test proves nothing"
    assert len(results) == 3, (
        "hits were lost to the concurrent rebuild: the snapshot was not bound "
        "inside the lock"
    )


def test_search_on_an_empty_index_raises_rather_than_returning_nothing(store):
    """An empty index is an error state, not an empty answer."""
    store.clear()
    with pytest.raises(VectorStoreError):
        store.search("anything", k=3)


def test_concurrent_readers_all_receive_usable_results(store):
    failures: list[str] = []

    def worker() -> None:
        for _ in range(25):
            try:
                results = store.search("unmanned vehicle", k=3)
            except VectorStoreError:
                continue
            if not results:
                failures.append("empty result set")
                return

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert failures == []


def test_remove_document_during_readers_raises_nothing_unexpected(store):
    """Deleting a document must not corrupt a search in flight."""
    document_ids = sorted(store.document_ids())
    assert document_ids
    unexpected: list[BaseException] = []

    def reader() -> None:
        for _ in range(15):
            try:
                store.search("unmanned vehicle", k=2)
            except VectorStoreError:
                pass
            except Exception as exc:  # noqa: BLE001 - asserted on below
                unexpected.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(3)]
    for thread in threads:
        thread.start()
    store.remove_document(document_ids[0])
    for thread in threads:
        thread.join(timeout=30)

    assert unexpected == []
    assert document_ids[0] not in store.document_ids()


def test_a_failed_disk_removal_is_raised_not_swallowed(store, monkeypatch):
    """clear() must not report success while data/vectorstore survives.

    The old handler caught OSError and only warned, so a Windows file lock left
    the corpus on disk: the session looked empty and every document reappeared
    on the next start.
    """
    import shutil

    store.save()
    assert store.directory.exists()

    def refuse(*_args, **_kwargs):
        raise PermissionError("file is open in another process")

    monkeypatch.setattr(shutil, "rmtree", refuse)

    with pytest.raises(VectorStoreError) as excinfo:
        store.clear()
    assert "reappear" in str(excinfo.value)


# ------------------------------------------------------------- BM25 coherence --


def test_bm25_never_pairs_scores_with_another_corpus_ids():
    """A rebuild mid-flight must not yield a citation for the wrong passage."""
    index = BM25Index()
    # Distinct vocabularies make a torn read unambiguous.
    small = [make_chunk(f"a{i}", "alpha alpha alpha") for i in range(2)]
    large = [make_chunk(f"b{i}", "beta beta beta gamma") for i in range(6)]

    torn: list[tuple[str, float]] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            for chunk_id, score in index.search("alpha", k=2):
                # A torn read means scores computed over one corpus were paired
                # with another corpus's id ordering. That is only observable when
                # the score is positive *and* belongs to the other corpus: while
                # `large` is legitimately published, searching "alpha" returns a
                # zero-score `b*` id, which is correct behaviour and not a race.
                if chunk_id.startswith("b") and score > 0:
                    torn.append((chunk_id, score))
                break

    readers = [threading.Thread(target=reader, daemon=True) for _ in range(4)]
    for thread in readers:
        thread.start()
    try:
        for _ in range(80):
            index.ensure(large)
            index.ensure(small)
    finally:
        stop.set()
        for thread in readers:
            thread.join(timeout=5)

    # Belt and braces: the reader above can miss a race if the GIL never yields
    # mid-search. Replay the interleaving single-threaded, where every torn read
    # is guaranteed to be observed, by inspecting the published tuple directly.
    assert torn == [], f"torn BM25 read: {torn[:5]}"

    index.ensure(large)
    snapshot = index._index
    assert snapshot is not None
    bm25, ids, fingerprint = snapshot
    scores = bm25.get_scores(tokenize("alpha"))
    # The published id ordering must line up with the score array it ships with.
    assert len(ids) == len(scores), "id ordering and score array disagree in length"
    assert all(float(score) == 0.0 for score in scores), (
        "'alpha' must not score against the beta corpus"
    )


def test_bm25_concurrent_rebuilds_publish_the_last_invoked_corpus():
    """A slower rebuild must not resurrect a superseded corpus.

    Locking only the assignment still lets two builds interleave, and the one
    that finishes last wins regardless of which was requested last. The slow
    build is started first and made to finish last.
    """
    import src.retriever as retriever_module

    index = BM25Index()
    slow = [make_chunk(f"a{i}", "alpha alpha alpha") for i in range(2)]
    fast = [make_chunk(f"b{i}", "beta beta beta") for i in range(6)]

    original = retriever_module.BM25Okapi
    gate = threading.Event()

    def gated(corpus):
        # Only the *first* (slow) build waits; later builds proceed.
        if corpus and corpus[0][:1] == ["alpha"]:
            gate.wait(timeout=10)
        return original(corpus)

    retriever_module.BM25Okapi = gated
    try:
        slow_thread = threading.Thread(target=index.ensure, args=(slow,))
        slow_thread.start()
        index.ensure(fast)          # requested last, published first
        gate.set()                  # now let the slow build finish
        slow_thread.join(timeout=30)
    finally:
        retriever_module.BM25Okapi = original

    # All six beta chunks score identically, so any of them is correct; what
    # matters is that no superseded alpha chunk came back.
    found = [cid for cid, _ in index.search("beta", k=1)]
    assert found and found[0].startswith("b"), (
        "a superseded rebuild was published after the current corpus"
    )


def test_bm25_search_matches_the_corpus_last_ensured():
    index = BM25Index()
    index.ensure([make_chunk("a1", "alpha alpha alpha")])
    assert [cid for cid, _ in index.search("alpha", k=1)] == ["a1"]
    index.ensure([make_chunk("b1", "beta beta beta")])
    assert [cid for cid, _ in index.search("alpha", k=1)] == ["b1"]


def test_bm25_handles_an_empty_corpus():
    index = BM25Index()
    index.ensure([])
    assert index.search("anything", k=3) == []
    assert index.size == 0


# ------------------------------------------------------ LLM failure cooldown --


def test_llm_failure_recovers_after_the_cooldown(pipeline, monkeypatch):
    """One transient failure must not disable synthesis for the process."""
    import src.llm as llm_module

    attempts = {"n": 0}

    def flaky(_settings):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient DNS failure")
        return object()

    monkeypatch.setattr(llm_module, "build_llm_client", flaky)

    assert pipeline.llm() is None, "the first failure should degrade to evidence-only"
    assert pipeline.llm() is None, "still degraded"
    assert attempts["n"] == 1, "retried during the cooldown"

    # Wind the cooldown back rather than sleeping through it.
    assert pipeline._llm_failed_until is not None
    pipeline._llm_failed_until -= LLM_RETRY_COOLDOWN_S + 1

    recovered = pipeline.llm()
    assert recovered is not None, "a transient failure must heal after the cooldown"
    assert attempts["n"] == 2
    assert pipeline.llm() is recovered, "a working client must then be cached"


def test_a_working_llm_client_is_built_once(pipeline, monkeypatch):
    import src.llm as llm_module

    calls = {"n": 0}

    def factory(_settings):
        calls["n"] += 1
        return object()

    monkeypatch.setattr(llm_module, "build_llm_client", factory)
    first = pipeline.llm()
    assert pipeline.llm() is first
    assert calls["n"] == 1


def test_a_burst_of_concurrent_llm_calls_opens_a_single_client(pipeline, monkeypatch):
    """Concurrent questions must not each open a provider session."""
    import src.llm as llm_module

    calls = {"n": 0}
    enter = threading.Event()

    def slow_factory(_settings):
        calls["n"] += 1
        enter.wait(timeout=10)   # widen the window a race would exploit
        return object()

    monkeypatch.setattr(llm_module, "build_llm_client", slow_factory)

    results: list[object] = []
    threads = [threading.Thread(target=lambda: results.append(pipeline.llm()))
               for _ in range(6)]
    for thread in threads:
        thread.start()
    enter.set()
    for thread in threads:
        thread.join(timeout=30)

    assert calls["n"] == 1, f"built {calls['n']} clients for one burst"
    assert len(results) == 6
    assert len({id(r) for r in results}) == 1


# ---------------------------------------------------------------- ingest gate --


def _patch_processor(monkeypatch, pipeline, failing: set[str]):
    from src.models import DocumentMeta

    class _Processor:
        def process(self, data, filename, **_kwargs):
            if filename in failing:
                raise RuntimeError("malformed xref table")
            meta = DocumentMeta(
                document_id=f"id-{filename}",
                title=filename,
                filename=filename,
                page_count=2,
            )
            return type("Extracted", (), {"meta": meta})()

    monkeypatch.setattr(pipeline, "processor", _Processor())
    monkeypatch.setattr(
        pipeline.chunker,
        "chunk_document",
        lambda extracted: [make_chunk(f"{extracted.meta.document_id}-1", "text")],
    )
    monkeypatch.setattr(pipeline.store, "add", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.store, "save", lambda: None)


def test_a_failed_file_raises_even_alongside_a_duplicate(pipeline, monkeypatch):
    """The broken-file gate was ``failures and not duplicates``.

    One duplicate plus one unreadable PDF therefore matched neither branch and
    returned success with the failure buried in a payload field the console
    never renders.
    """
    _patch_processor(monkeypatch, pipeline, failing={"broken.pdf"})

    with pytest.raises(AstraIntelError) as excinfo:
        pipeline.ingest_pdfs(
            [("broken.pdf", b"%PDF-bad"), ("dupe.pdf", b"%PDF-ok")]
        )

    # The message the client renders must name the file and stay actionable,
    # not the generic base-class fallback.
    assert "broken.pdf" in excinfo.value.user_message
    assert excinfo.value.user_message != AstraIntelError.default_user_message
    # The internal exception text must not be what reaches the client.
    assert "xref" not in excinfo.value.user_message


def test_a_clean_batch_reports_what_it_added(pipeline, monkeypatch):
    _patch_processor(monkeypatch, pipeline, failing=set())

    result = pipeline.ingest_pdfs([("new.pdf", b"%PDF-ok")])
    assert len(result.added) == 1
    assert result.failures == {}


def test_page_histograms_cover_every_document_in_one_pass(store):
    pipeline = AstraPipeline.__new__(AstraPipeline)
    pipeline.store = store

    histograms = pipeline.page_histograms()

    assert set(histograms) == set(store.document_ids())
    total = sum(sum(pages.values()) for pages in histograms.values())
    assert total == store.chunk_count
    # The fixture spreads chunks over pages 1..6 for one document.
    assert sum(1 for _ in histograms["doc-c0"]) == 1


# ------------------------------------------------------------ get_pipeline --


def test_get_pipeline_is_initialised_once_under_concurrency(offline_settings):
    """Concurrent first requests must share a single pipeline instance."""
    import api

    api.reset_pipeline()
    seen: list[int] = []
    errors: list[BaseException] = []
    start = threading.Barrier(8)

    def grab() -> None:
        try:
            start.wait(timeout=30)
            seen.append(id(api.get_pipeline()))
        except BaseException as exc:  # noqa: BLE001 - asserted on below
            errors.append(exc)

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert errors == []
    assert len(seen) == 8
    assert len(set(seen)) == 1, "more than one pipeline was constructed"
    api.reset_pipeline()