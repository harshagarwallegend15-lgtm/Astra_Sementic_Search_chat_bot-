"""Embedding models, behind a small and stable interface.

The default is a local, free sentence-transformer (``all-MiniLM-L6-v2``,
~90 MB, CPU friendly) which needs no API key and no paid account. If that
model cannot be loaded - no network on first run, no Hugging Face cache, a
locked-down machine - the system degrades to a deterministic hashing
embedder instead of crashing.

Why the fallback exists rather than an error: retrieval, chunking and the
citation machinery are all testable without a neural model, and a developer
should never be blocked from running the test suite by a missing download.
The active model name is always reported so the UI can say which one is in use
and nobody mistakes hashed vectors for semantic ones.

Every embedder subclasses LangChain's ``Embeddings``, so it plugs directly into
the FAISS vector store.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any, Sequence

import numpy as np
from langchain_core.embeddings import Embeddings

from .config import Settings, get_settings
from .errors import EmbeddingError
from .logging_utils import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9]+")

#: Dimensionality of the hashing fallback. Small on purpose: it only has to
#: separate a few hundred chunks, and smaller means faster tests.
_HASH_DIMENSIONS = 512


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, used by both the hashing embedder and BM25."""
    return [match.group(0).lower() for match in _WORD_RE.finditer(text or "")]


class BaseEmbedder(Embeddings):
    """Common interface for the embedding backends."""

    #: Human-readable identifier shown in the UI.
    name: str = "base"
    #: True when the vectors carry real semantic meaning.
    is_semantic: bool = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # noqa: D102
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:  # noqa: D102
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:  # pragma: no cover - abstract
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local sentence-transformer embeddings. No API key, no cost."""

    is_semantic = True

    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.name = model_name
        self._lock = threading.Lock()
        self._model: Any | None = None
        self._dimension: int | None = None

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed.", exc
                ) from exc
            # Keep first-run downloads and encodes from spraying progress bars
            # over the Streamlit console.
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            try:
                logger.info("Loading embedding model %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
                # `get_sentence_embedding_dimension` was renamed upstream; the
                # new name is preferred but the old one is kept as a fallback
                # for older sentence-transformers releases.
                getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
                    self._model, "get_sentence_embedding_dimension"
                )
                self._dimension = int(getter())
                logger.info("Embedding model ready (dim=%d)", self._dimension)
            except Exception as exc:
                raise EmbeddingError(
                    f"Could not load embedding model '{self.model_name}': {exc}",
                    user_message=(
                        f"Could not load the embedding model '{self.model_name}'. "
                        "It may need to be downloaded on first use - check your "
                        "internet connection, or set EMBEDDING_BACKEND=always to "
                        "use the offline fallback."
                    ),
                ) from exc
        return self._model

    def _embed_one(self, text: str) -> list[float]:
        model = self._ensure_loaded()
        # Always pass a list. sentence-transformers 6.x may return a 2-D
        # array for a bare string, which then fails to iterate as scalars.
        vectors = model.encode(
            [text if text and text.strip() else " "],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [float(value) for value in np.asarray(vectors)[0]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batched encode; a single call is far faster than one at a time."""
        if not texts:
            return []
        model = self._ensure_loaded()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]


class HashingEmbedder(BaseEmbedder):
    """Deterministic, dependency-light fallback: hashed word unigrams+bigrams.

    This is a *lexical* embedding, not a semantic one. It exists so the system
    stays usable and testable offline; the UI labels it as such.
    """

    name = "hashing-fallback (lexical, not semantic)"
    is_semantic = False

    def __init__(self, dimensions: int = _HASH_DIMENSIONS) -> None:
        self.dimensions = dimensions
        self._dimension = dimensions

    def _embed_one(self, text: str) -> list[float]:
        tokens = tokenize(text)
        if not tokens:
            return [0.0] * self.dimensions

        features: list[str] = list(tokens)
        features.extend(
            f"{a}_{b}" for a, b in zip(tokens, tokens[1:])
        )

        vector = np.zeros(self.dimensions, dtype=np.float32)
        for feature in features:
            index = _stable_hash(feature) % self.dimensions
            # A sign keeps unrelated collisions from always adding together.
            vector[index] += 1.0 if (_stable_hash(feature) >> 32) & 1 else -1.0

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return [float(value) for value in vector]


def _stable_hash(text: str) -> int:
    """Hash that is identical across processes.

    Python's built-in ``hash`` is salted per process for strings, which would
    make a persisted index unusable across restarts.
    """
    import hashlib

    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big"
    )


_embedder_cache: dict[tuple[str, str], BaseEmbedder] = {}
_cache_lock = threading.Lock()


SEMANTIC_BACKENDS = frozenset({"semantic", "never"})
HASHING_BACKENDS = frozenset({"hashing", "always"})
VALID_BACKENDS = frozenset({"auto"}) | SEMANTIC_BACKENDS | HASHING_BACKENDS


def get_embedder(settings: Settings | None = None) -> BaseEmbedder:
    """Build (and cache) the embedder described by ``settings``.

    ``embedding_backend`` accepts:

    ``hashing`` (alias ``always``)
        Force the deterministic local hashing embedder. No model download, no
        network, much faster. Lower retrieval quality.
    ``semantic`` (alias ``never``)
        Require the sentence-transformers model; fail loudly if it cannot load.
    ``auto`` (default)
        Try the semantic model and fall back to hashing if it cannot load.

    An unrecognised value raises rather than silently meaning "semantic",
    because a typo here would quietly swap the retrieval engine.
    """
    settings = settings or get_settings()
    backend = (settings.embedding_backend or "auto").strip().lower()
    if backend not in VALID_BACKENDS:
        from .errors import ConfigurationError

        raise ConfigurationError(
            f"Unknown EMBEDDING_BACKEND '{settings.embedding_backend}'. "
            "Use 'auto', 'semantic' or 'hashing'."
        )

    key = (settings.embedding_model, backend)

    with _cache_lock:
        cached = _embedder_cache.get(key)
    if cached is not None:
        return cached

    if backend in HASHING_BACKENDS:
        embedder: BaseEmbedder = HashingEmbedder()
    elif backend in SEMANTIC_BACKENDS:
        embedder = SentenceTransformerEmbedder(
            settings.embedding_model, settings.embedding_batch_size
        )
    else:
        semantic = SentenceTransformerEmbedder(
            settings.embedding_model, settings.embedding_batch_size
        )
        try:
            semantic._ensure_loaded()
            embedder = semantic
        except EmbeddingError as exc:
            logger.warning("Falling back to lexical embeddings: %s", exc)
            embedder = HashingEmbedder()

    with _cache_lock:
        _embedder_cache[key] = embedder
    return embedder


def reset_embedder_cache() -> None:
    """Drop cached embedders. Used by tests and by the rebuild-index action."""
    with _cache_lock:
        _embedder_cache.clear()


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity for already-normalised or raw vectors."""
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


__all__ = [
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
    "HashingEmbedder",
    "get_embedder",
    "reset_embedder_cache",
    "cosine_similarity",
    "tokenize",
]
