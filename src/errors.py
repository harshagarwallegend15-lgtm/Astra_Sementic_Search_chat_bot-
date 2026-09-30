"""Typed exceptions for ASTRA INTEL.

The UI must never show a raw traceback, but it must be able to show an
actionable message. Every failure the user can trigger therefore raises one of
these, carrying a ``user_message`` that is safe to render directly.

The full traceback is still logged (see ``logging_utils.setup_logging``) so
debugging information is never lost.
"""

from __future__ import annotations


class AstraIntelError(Exception):
    """Base class for all errors raised by this application."""

    #: Fallback shown when a subclass does not provide a specific message.
    default_user_message = "Something went wrong while processing your request."

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.user_message = user_message or self.default_user_message

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
class ConfigurationError(AstraIntelError):
    default_user_message = (
        "The application is not configured correctly. Check your .env file."
    )


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------
class DocumentError(AstraIntelError):
    default_user_message = "The document could not be processed."


class UnsupportedFileTypeError(DocumentError):
    default_user_message = "Only PDF files are supported."


class DocumentTooLargeError(DocumentError):
    default_user_message = "That file is too large to process."


class DuplicateDocumentError(DocumentError):
    default_user_message = "This document has already been uploaded."


class PDFProcessingError(DocumentError):
    default_user_message = (
        "The PDF could not be read. It may be corrupt or password protected."
    )


class EmptyDocumentError(DocumentError):
    """Raised when a PDF opens cleanly but carries no usable text.

    This is deliberately distinct from :class:`PDFProcessingError`: a scanned
    image-only PDF is a *valid* PDF that we simply cannot read, and the user
    deserves to be told which case they hit.
    """

    default_user_message = (
        "No text could be extracted from this PDF. It is most likely a scanned "
        "image with no text layer."
    )


class NoDocumentsError(AstraIntelError):
    default_user_message = "Please upload at least one PDF document first."


# --------------------------------------------------------------------------
# Retrieval / indexing
# --------------------------------------------------------------------------
class EmbeddingError(AstraIntelError):
    default_user_message = (
        "The embedding model could not be loaded. Check EMBEDDING_MODEL and "
        "your internet connection."
    )


class VectorStoreError(AstraIntelError):
    default_user_message = (
        "The document index could not be read or written. Try rebuilding the index."
    )


class NoRelevantContextError(AstraIntelError):
    """Retrieval found nothing confident enough to answer from."""

    default_user_message = (
        "I couldn't find enough relevant information in the provided documents."
    )


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
class LLMConfigurationError(AstraIntelError):
    default_user_message = (
        "No language model is configured. Set LLM_PROVIDER, LLM_MODEL and any "
        "required API key in your .env file."
    )


class LLMRequestError(AstraIntelError):
    default_user_message = (
        "The language model could not be reached. It may be offline or "
        "rate-limiting requests."
    )


class LLMResponseError(AstraIntelError):
    default_user_message = "The language model returned an unreadable response."


class LLMError(AstraIntelError):
    """Generic LLM failure used as the catch-all at provider boundaries."""

    default_user_message = "The language model could not complete this request."


class AnsweringError(AstraIntelError):
    default_user_message = "The question could not be answered."


__all__ = [
    "AstraIntelError",
    "ConfigurationError",
    "DocumentError",
    "UnsupportedFileTypeError",
    "DocumentTooLargeError",
    "DuplicateDocumentError",
    "PDFProcessingError",
    "EmptyDocumentError",
    "NoDocumentsError",
    "EmbeddingError",
    "VectorStoreError",
    "NoRelevantContextError",
    "LLMConfigurationError",
    "LLMRequestError",
    "LLMResponseError",
    "LLMError",
    "AnsweringError",
]
