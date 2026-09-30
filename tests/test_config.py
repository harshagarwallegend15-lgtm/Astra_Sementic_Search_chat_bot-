"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest

from src.config import PROJECT_ROOT, Settings
from src.errors import ConfigurationError


class TestDefaults:
    def test_ollama_is_the_default_provider(self):
        assert Settings().llm_provider == "ollama"

    def test_chunking_defaults(self):
        settings = Settings()
        assert settings.chunk_size == 900
        assert settings.chunk_overlap == 150
        assert settings.chunk_overlap < settings.chunk_size

    def test_thresholds_are_in_range(self):
        settings = Settings()
        assert 0.0 <= settings.min_retrieval_score <= 1.0
        assert -1.0 <= settings.min_dense_similarity <= 1.0

    def test_dense_floor_is_meaningful_for_minilm(self):
        """all-MiniLM-L6-v2 compresses similarity; the default must reflect that."""
        assert Settings().min_dense_similarity >= 0.20


class TestPaths:
    def test_paths_default_inside_the_project(self):
        settings = Settings()
        assert str(settings.vectorstore_dir).startswith(str(PROJECT_ROOT))
        assert str(settings.uploads_dir).startswith(str(PROJECT_ROOT))

    def test_sample_metadata_path(self):
        assert Settings().sample_metadata_path == PROJECT_ROOT / "sample-metadata.json"

    def test_starter_documents_dir(self):
        assert Settings().sample_documents_dir.name == "sample-documents"

    def test_ensure_directories_creates_them(self, tmp_path):
        settings = Settings(
            vectorstore_dir=tmp_path / "vs",
            uploads_dir=tmp_path / "up",
        )
        settings.ensure_directories()
        assert (tmp_path / "vs").is_dir()
        assert (tmp_path / "up").is_dir()


class TestProviderResolution:
    def test_ollama_default_endpoint(self):
        assert Settings(llm_provider="ollama").resolved_base_url.startswith("http")

    def test_explicit_base_url_wins(self):
        settings = Settings(
            llm_provider="ollama", llm_base_url="http://example.test:9000/"
        )
        assert settings.resolved_base_url == "http://example.test:9000"

    def test_openai_compatible_infers_its_own_base(self):
        settings = Settings(llm_provider="openai", llm_api_key="k")
        assert settings.resolved_base_url.startswith("https://")

    def test_keyless_provider_needs_no_key(self):
        assert not Settings(llm_provider="ollama").provider_needs_key

    def test_hosted_provider_needs_a_key(self):
        assert Settings(llm_provider="openai").provider_needs_key


class TestValidation:
    def test_missing_provider_raises(self):
        with pytest.raises(ConfigurationError):
            Settings(llm_provider="").validate_llm()

    def test_missing_model_raises(self):
        with pytest.raises(ConfigurationError):
            Settings(llm_model="").validate_llm()

    def test_missing_key_for_hosted_provider_raises(self):
        with pytest.raises(ConfigurationError):
            Settings(llm_provider="openai", llm_api_key="").validate_llm()

    def test_ollama_validates_without_a_key(self):
        Settings(llm_provider="ollama", llm_model="llama3.2:3b").validate_llm()

    def test_tiny_chunk_size_is_raised(self):
        settings = Settings(chunk_size=10, chunk_overlap=0)
        settings.validate_chunking()
        assert settings.chunk_size >= 300

    def test_overlap_larger_than_chunk_is_clamped(self):
        settings = Settings(chunk_size=800, chunk_overlap=900)
        settings.validate_chunking()
        assert settings.chunk_overlap < settings.chunk_size


class TestPublicDict:
    def test_never_exposes_the_api_key(self):
        settings = Settings(llm_provider="openai", llm_api_key="sk-secret-value")
        public = settings.public_dict()
        assert "sk-secret-value" not in str(public)
        assert public["API key configured"] == "yes"

    def test_reports_missing_key(self):
        assert Settings(llm_api_key="").public_dict()["API key configured"] == "no"
