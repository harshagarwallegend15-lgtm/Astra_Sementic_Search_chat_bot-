"""Environment-driven configuration.

Everything the system can be tuned with lives here, so there is exactly one
place to look when a retrieval score or a model name needs changing.

Design rules:
  * No secret ever has a default value. ``LLM_API_KEY`` defaults to ``""`` and
    providers that need one refuse to start without it.
  * ``.env`` is loaded if present but real environment variables always win,
    which is what makes ``docker run --env-file`` behave predictably.
  * Invalid values fall back to the default and log a warning rather than
    crashing the UI at import time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Final

from dotenv import load_dotenv

from .errors import ConfigurationError
from .logging_utils import get_logger

logger = get_logger(__name__)

# Project root is the parent of the ``src`` package.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: Providers that do not need an API key.
KEYLESS_PROVIDERS: Final[frozenset[str]] = frozenset({"ollama", "local", "none"})

#: Base URLs for the well-known OpenAI-compatible providers. Users only need to
#: set ``LLM_API_KEY``; the endpoint is inferred.
OPENAI_COMPATIBLE_BASES: Final[dict[str, str]] = {
    "openai": "https://api.openai.com/v1",
    "together": "https://api.together.xyz/v1",
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "lmstudio": "http://localhost:1234/v1",
}

_TRUE: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_FALSE: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None else value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    logger.warning("%s=%r is not a boolean; using %s", name, raw, default)
    return default


@dataclass
class Settings:
    """Runtime configuration for the whole application."""

    # -- LLM ---------------------------------------------------------------
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2:3b"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_temperature: float = 0.0
    llm_max_tokens: int = 900
    llm_timeout: int = 180
    #: Retries for transient provider faults (429, 502/503/504, socket
    #: resets). Without this a single rate-limit response aborts the question
    #: and starts the failure cooldown, so one blip costs the user an answer.
    llm_max_retries: int = 2
    llm_num_ctx: int = 8192

    # -- Embeddings --------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    #: ``auto`` tries MiniLM and degrades to a local hashing embedder if the
    #: model cannot be loaded (no network, no cache). ``semantic`` requires
    #: MiniLM and fails loudly instead. ``hashing`` forces the deterministic
    #: offline fallback, which is what the test suite uses so that ``pytest``
    #: never needs a model download.
    embedding_backend: str = "auto"
    embedding_batch_size: int = 64

    # -- Chunking ----------------------------------------------------------
    chunk_size: int = 900
    chunk_overlap: int = 150
    #: Chunks below this many characters are dropped as noise.
    min_chunk_chars: int = 60

    # -- Retrieval ---------------------------------------------------------
    top_k: int = 8
    #: How many candidates each retriever contributes before fusion.
    candidate_k: int = 24
    retrieval_mode: str = "hybrid"
    #: Fused evidence score below this is treated as "no evidence".
    #: Calibrated against the default embedding model; see README.
    min_retrieval_score: float = 0.30
    #: Semantic floor. A chunk whose cosine similarity to the question is below
    #: this is not evidence, however many rare words it happens to share.
    #: Note: all-MiniLM-L6-v2 compresses its similarity range (unrelated pairs
    #: sit around 0.1-0.3, strong matches 0.6-0.8), so this is a coarse
    #: relevance filter and deliberately NOT a truth oracle. The authoritative
    #: grounding layers are the system prompt and the post-hoc verifier.
    min_dense_similarity: float = 0.25
    #: Cap on chunks taken from any single document. 0 disables the cap, which
    #: is the default: capping was measured to cause false refusals on
    #: "name one specific programme" style questions, because the cap filled
    #: with introductory passages that merely repeat the question's noun and
    #: evicted the passage that actually carried the answer.
    max_chunks_per_document: int = 0
    #: Reserve a context slot for each competitive rival document. A rival earns
    #: a slot when its best passage scores at least this fraction of the overall
    #: best passage. This is additive: it adds cross-document evidence without
    #: evicting the top-ranked passages. Raise it to require broader coverage;
    #: set 0 to disable and rank purely by score.
    document_diversity_ratio: float = 0.45
    exclude_reference_sections: bool = True

    # -- Upload validation -------------------------------------------------
    max_upload_mb: int = 50
    #: A page with fewer characters than this is treated as effectively blank.
    min_chars_per_page: int = 40
    #: A document with fewer characters than this is rejected as empty.
    min_document_chars: int = 200

    # -- Summarisation -----------------------------------------------------
    summary_map_chunks: int = 10
    summary_reduce_chunks: int = 8

    # -- Behaviour ---------------------------------------------------------
    #: Refuse to show an answer that carries no valid citation.
    strict_citations: bool = False
    log_level: str = "INFO"

    # -- Paths -------------------------------------------------------------
    vectorstore_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "vectorstore")
    uploads_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "uploads")
    sample_metadata_path: Path = field(default_factory=lambda: PROJECT_ROOT / "sample-metadata.json")

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from the process environment (and ``.env``)."""
        load_dotenv(PROJECT_ROOT / ".env", override=False)

        def _path(name: str, default: Path) -> Path:
            raw = _env(name)
            if not raw:
                return default
            path = Path(raw)
            return path if path.is_absolute() else PROJECT_ROOT / path

        return cls(
            llm_provider=_env("LLM_PROVIDER", "ollama").lower(),
            llm_model=_env("LLM_MODEL", "llama3.2:3b"),
            llm_api_key=_env("LLM_API_KEY"),
            llm_base_url=_env("LLM_BASE_URL"),
            llm_temperature=_env_float("LLM_TEMPERATURE", 0.0),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", 900),
            llm_timeout=_env_int("LLM_TIMEOUT", 180),
llm_max_retries=_env_int("LLM_MAX_RETRIES", 2),
            llm_num_ctx=_env_int("LLM_NUM_CTX", 8192),
            embedding_model=_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            embedding_backend=_env("EMBEDDING_BACKEND", "auto").lower(),
            embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 64),
            chunk_size=_env_int("CHUNK_SIZE", 900),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 150),
            min_chunk_chars=_env_int("MIN_CHUNK_CHARS", 60),
            top_k=_env_int("TOP_K", 8),
            candidate_k=_env_int("CANDIDATE_K", 24),
            retrieval_mode=_env("RETRIEVAL_MODE", "hybrid").lower(),
            min_retrieval_score=_env_float("MIN_RETRIEVAL_SCORE", 0.30),
            min_dense_similarity=_env_float("MIN_DENSE_SIMILARITY", 0.25),
            max_chunks_per_document=_env_int("MAX_CHUNKS_PER_DOCUMENT", 0),
            document_diversity_ratio=_env_float("DOCUMENT_DIVERSITY_RATIO", 0.45),
            exclude_reference_sections=_env_bool("EXCLUDE_REFERENCE_SECTIONS", True),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", 50),
            min_chars_per_page=_env_int("MIN_CHARS_PER_PAGE", 40),
            min_document_chars=_env_int("MIN_DOCUMENT_CHARS", 200),
            summary_map_chunks=_env_int("SUMMARY_MAP_CHUNKS", 10),
            summary_reduce_chunks=_env_int("SUMMARY_REDUCE_CHUNKS", 8),
            strict_citations=_env_bool("STRICT_CITATIONS", False),
            log_level=_env("LOG_LEVEL", "INFO").upper(),
            vectorstore_dir=_path("VECTORSTORE_DIR", PROJECT_ROOT / "data" / "vectorstore"),
            uploads_dir=_path("UPLOADS_DIR", PROJECT_ROOT / "data" / "uploads"),
        )

    # ------------------------------------------------------------------
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def sample_documents_dir(self) -> Path:
        return PROJECT_ROOT / "sample-documents"

    @property
    def resolved_base_url(self) -> str:
        """Base URL for the chat endpoint.

        Explicit ``LLM_BASE_URL`` always wins. Otherwise it is inferred from
        the provider name, which keeps ``.env`` to a minimum for the common
        case.
        """
        if self.llm_base_url:
            return self.llm_base_url.rstrip("/")
        if self.llm_provider in OPENAI_COMPATIBLE_BASES:
            return OPENAI_COMPATIBLE_BASES[self.llm_provider]
        if self.llm_provider == "ollama":
            return _env("OLLAMA_HOST", "http://localhost:11434")
        return ""

    @property
    def provider_needs_key(self) -> bool:
        return self.llm_provider not in KEYLESS_PROVIDERS

    def validate_llm(self) -> None:
        """Raise :class:`ConfigurationError` if the LLM cannot possibly work."""
        if not self.llm_provider:
            raise ConfigurationError("LLM_PROVIDER is not set")
        if not self.llm_model:
            raise ConfigurationError("LLM_MODEL is not set")
        if self.provider_needs_key and not self.llm_api_key:
            raise ConfigurationError(
                f"Provider '{self.llm_provider}' requires LLM_API_KEY, but none is set.",
                user_message=(
                    f"The '{self.llm_provider}' provider needs an API key. Add "
                    "LLM_API_KEY to your .env file and restart."
                ),
            )

    def validate_chunking(self) -> None:
        if self.chunk_size < 120:
            logger.warning("CHUNK_SIZE=%d is very small; raising to 300", self.chunk_size)
            self.chunk_size = 300
        if self.chunk_overlap >= self.chunk_size:
            logger.warning(
                "CHUNK_OVERLAP=%d must be smaller than CHUNK_SIZE=%d; clamping",
                self.chunk_overlap,
                self.chunk_size,
            )
            self.chunk_overlap = max(0, self.chunk_size // 4)

    def ensure_directories(self) -> None:
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> dict[str, Any]:
        """Config for display in the UI. Never includes the API key."""
        return {
            "LLM provider": self.llm_provider,
            "LLM model": self.llm_model,
            "LLM base URL": self.resolved_base_url or "(provider default)",
            "API key configured": "yes" if self.llm_api_key else "no",
            "Embedding model": self.embedding_model,
            "Embedding backend": self.embedding_backend,
            "TOP_K": self.top_k,
            "MIN_RETRIEVAL_SCORE": self.min_retrieval_score,
            "CHUNK_SIZE": self.chunk_size,
            "CHUNK_OVERLAP": self.chunk_overlap,
            "Retrieval mode": self.retrieval_mode,
        }


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None or refresh:
        _settings = Settings.from_env()
        _settings.validate_chunking()
    return _settings


__all__ = [
    "Settings",
    "get_settings",
    "PROJECT_ROOT",
    "KEYLESS_PROVIDERS",
    "OPENAI_COMPATIBLE_BASES",
]
