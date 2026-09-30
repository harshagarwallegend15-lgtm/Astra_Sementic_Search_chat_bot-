"""Tests for the embedding backends and their selection."""

from __future__ import annotations

import pytest

from src.config import Settings
from src.embeddings import (
    HASHING_BACKENDS,
    SEMANTIC_BACKENDS,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
)
from src.errors import ConfigurationError, EmbeddingError


class TestHashingEmbedder:
    def test_dimension_is_stable(self):
        embedder = HashingEmbedder()
        assert len(embedder.embed_query("electronic warfare")) == embedder.dimensions

    def test_output_is_normalised(self):
        vector = HashingEmbedder().embed_query("electronic warfare")
        magnitude = sum(value * value for value in vector) ** 0.5
        assert magnitude == pytest.approx(1.0, abs=1e-6)

    def test_identical_text_gives_identical_vectors(self):
        embedder = HashingEmbedder()
        assert embedder.embed_query("jammer") == embedder.embed_query("jammer")

    def test_different_text_gives_different_vectors(self):
        embedder = HashingEmbedder()
        assert embedder.embed_query("jammer") != embedder.embed_query("lidar")

    def test_batch_matches_single_query(self):
        embedder = HashingEmbedder()
        batch = embedder.embed_documents(["jammer", "lidar"])
        assert batch[0] == embedder.embed_query("jammer")
        assert batch[1] == embedder.embed_query("lidar")

    def test_reports_itself_as_hashing(self):
        embedder = HashingEmbedder()
        assert embedder.name.startswith("hashing")
        assert embedder.is_semantic is False


class TestBackendSelection:
    def test_hashing_backend_returns_hashing(self):
        settings = Settings(embedding_backend="hashing")
        assert isinstance(get_embedder(settings), HashingEmbedder)

    @pytest.mark.parametrize("name", sorted(HASHING_BACKENDS))
    def test_hashing_aliases(self, name):
        assert isinstance(
            get_embedder(Settings(embedding_backend=name)), HashingEmbedder
        )

    def test_backend_is_case_insensitive(self):
        assert isinstance(
            get_embedder(Settings(embedding_backend="HASHING")), HashingEmbedder
        )

    @pytest.mark.parametrize("name", sorted(SEMANTIC_BACKENDS))
    def test_semantic_aliases_reject_the_fallback(self, name):
        settings = Settings(embedding_backend=name, embedding_model="not-a-real-model")
        embedder = get_embedder(settings)
        assert isinstance(embedder, SentenceTransformerEmbedder)

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ConfigurationError, match="EMBEDDING_BACKEND"):
            get_embedder(Settings(embedding_backend="word2vec"))

    def test_a_typo_does_not_silently_mean_semantic(self):
        """The whole point: a bad value must fail loudly, not swap engines."""
        with pytest.raises(ConfigurationError):
            get_embedder(Settings(embedding_backend="hashingg"))

    def test_embedders_are_cached_per_backend(self):
        settings = Settings(embedding_backend="hashing")
        assert get_embedder(settings) is get_embedder(settings)


class TestSemanticEmbedderOffline:
    def test_unloadable_model_raises_embedding_error(self):
        embedder = SentenceTransformerEmbedder("definitely-not-a-real-model-xyz")
        with pytest.raises(EmbeddingError):
            embedder._ensure_loaded()

    def test_semantic_backend_does_not_silently_downgrade(self):
        settings = Settings(
            embedding_backend="semantic", embedding_model="definitely-not-real-xyz"
        )
        embedder = get_embedder(settings)
        with pytest.raises(EmbeddingError):
            embedder.embed_query("anything")

    def test_auto_backend_downgrades_to_hashing(self):
        settings = Settings(
            embedding_backend="auto", embedding_model="definitely-not-real-xyz"
        )
        embedder = get_embedder(settings)
        assert isinstance(embedder, HashingEmbedder)
        assert embedder.name.startswith("hashing")
