"""Shared pytest fixtures.

Tests that would need the embedding model or a live LLM are marked so they can
be deselected with ``-m "not slow"``. Everything marked ``slow`` needs either
a downloaded sentence-transformers model or a running Ollama instance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings  # noqa: E402
from src.models import Chunk, RetrievedChunk  # noqa: E402

EW_SUBVISIONS = (
    "Electronic warfare consists of three major subdivisions: electronic attack "
    "(EA), electronic protection (EP), and electronic warfare support (ES). "
    "Electronic attack is the use of electromagnetic energy to attack enemy "
    "capabilities."
)

UAV_AUTONOMY = (
    "The level of autonomy in UAVs varies widely. UAV manufacturers often build "
    "specific levels of autonomy. Full autonomy is available for specific tasks, "
    "such as airborne refuelling. One approach to quantifying autonomous "
    "capabilities is based on the OODA loop. Operators must approve take-off and "
    "landing manually."
)

UGV_EOD = (
    "Explosive ordnance disposal robots such as THeMIS from Milrem Robotics are "
    "used to inspect and clear suspected explosive devices. The Ripsaw platform "
    "from Howe and Howe Technologies is an example of a tracked EOD robot."
)


def _chunk(text: str, page: int, title: str, section: str, doc: str = "d1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"{doc}-p{page:03d}-00",
            document_id=doc,
            filename=f"{title}.pdf",
            title=title,
            page=page,
            text=text,
            section=section,
        ),
        score=0.6,
        dense_score=0.6,
    )


@pytest.fixture
def ew_chunk() -> RetrievedChunk:
    return _chunk(EW_SUBVISIONS, 2, "Electronic warfare", "Subdivisions", "dew")


@pytest.fixture
def uav_chunk() -> RetrievedChunk:
    return _chunk(UAV_AUTONOMY, 14, "Unmanned aerial vehicle", "Autonomy", "duav")


@pytest.fixture
def ugv_chunk() -> RetrievedChunk:
    return _chunk(UGV_EOD, 6, "Unmanned ground vehicle", "Military", "dugv")


@pytest.fixture
def mixed_chunks(ew_chunk, uav_chunk, ugv_chunk) -> list[RetrievedChunk]:
    return [ew_chunk, uav_chunk, ugv_chunk]


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Offline settings for logic tests.

    The thresholds are relaxed because the hashing fallback embedder produces
    near-zero cosine similarities; the production thresholds are exercised by
    ``scripts/evaluate.py`` and by the explicit threshold tests in
    ``test_retrieval.py``.
    """
    return Settings(
        vectorstore_dir=tmp_path / "vectorstore",
        uploads_dir=tmp_path / "uploads",
        sample_metadata_path=PROJECT_ROOT / "sample-metadata.json",
        embedding_backend="hashing",
        min_dense_similarity=-1.0,
        min_retrieval_score=0.0,
    )


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop a developer's real .env from leaking into the test run."""
    for key in (
        "LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL",
        "EMBEDDING_BACKEND", "EMBEDDING_MODEL", "MIN_DENSE_SIMILARITY",
        "MIN_RETRIEVAL_SCORE", "MAX_CHUNKS_PER_DOCUMENT", "VECTORSTORE_DIR",
        "UPLOADS_DIR", "STRICT_CITATIONS", "LOG_LEVEL", "TOP_K", "CHUNK_SIZE",
        "CHUNK_OVERLAP", "EXCLUDE_REFERENCE_SECTIONS",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(scope="session")
def starter_pdfs() -> list[Path]:
    return sorted((PROJECT_ROOT / "sample-documents").glob("*.pdf"))
