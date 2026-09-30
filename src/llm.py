"""LLM provider abstraction.

Exposes a single ``complete``/``stream`` surface over LangChain chat models so
the rest of the application is provider-agnostic. Ollama is the default local
provider; any OpenAI-compatible endpoint works via ``LLM_BASE_URL``.
"""

from __future__ import annotations

from typing import Iterator, Protocol, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama

from .config import Settings
from .errors import LLMError
from .logging_utils import get_logger

logger = get_logger(__name__)

SYSTEM_ROLE = "system"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"


class LLMClient(Protocol):
    """Minimal chat interface used by the answering pipeline."""

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Sequence[BaseMessage] | None = None,
    ) -> str:
        ...

    def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Sequence[BaseMessage] | None = None,
    ) -> Iterator[str]:
        ...


class BaseLLMClient:
    """Shared message assembly, history handling and error wrapping."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build_model(self) -> BaseChatModel:
        raise NotImplementedError

    def _messages(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Sequence[BaseMessage] | None,
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        if history:
            messages.extend(history)
        messages.append(HumanMessage(content=user_prompt))
        return messages

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Sequence[BaseMessage] | None = None,
    ) -> str:
        messages = self._messages(system_prompt, user_prompt, history)
        try:
            response = self.build_model().invoke(messages)
        except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
            raise LLMError(f"LLM request failed: {exc}") from exc
        return _message_text(response)

    def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Sequence[BaseMessage] | None = None,
    ) -> Iterator[str]:
        messages = self._messages(system_prompt, user_prompt, history)
        try:
            for chunk in self.build_model().stream(messages):
                text = _message_text(chunk)
                if text:
                    yield text
        except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
            raise LLMError(f"LLM streaming failed: {exc}") from exc


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(parts)
    return str(content)


class OllamaLLMClient(BaseLLMClient):
    """Local Ollama client (default provider)."""

    def build_model(self) -> BaseChatModel:
        kwargs: dict[str, object] = {
            "model": self.settings.llm_model,
            "temperature": self.settings.llm_temperature,
            "num_ctx": self.settings.llm_num_ctx,
            "num_predict": self.settings.llm_max_tokens,
            "timeout": self.settings.llm_timeout,
            "repeat_penalty": 1.1,
        }
        base_url = self.settings.resolved_base_url
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOllama(**kwargs)


class OpenAICompatibleLLMClient(BaseLLMClient):
    """Client for any OpenAI-compatible chat-completions endpoint.

    Works with hosted providers (OpenAI, Together, Groq, LM Studio, vLLM) and
    with self-hosted servers that expose ``/v1/chat/completions``.
    """

    def build_model(self) -> BaseChatModel:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMError(
                "langchain-openai is required for the OpenAI-compatible "
                "provider. Install it with: pip install langchain-openai"
            ) from exc

        kwargs: dict[str, object] = {
            "model": self.settings.llm_model,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "timeout": self.settings.llm_timeout,
            "api_key": self.settings.llm_api_key or "not-needed",
        }
        base_url = self.settings.resolved_base_url
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)


def build_llm_client(settings: Settings) -> BaseLLMClient:
    """Instantiate the client for the configured provider."""
    settings.validate_llm()
    provider = settings.llm_provider.lower()
    if provider == "ollama":
        client: BaseLLMClient = OllamaLLMClient(settings)
    elif provider in {"openai", "openai-compatible", "openai_compatible"}:
        client = OpenAICompatibleLLMClient(settings)
    else:
        raise LLMError(
            f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. "
            "Use 'ollama' or 'openai'."
        )
    logger.info(
        "LLM client ready: provider=%s model=%s endpoint=%s",
        provider,
        settings.llm_model,
        settings.resolved_base_url or "provider default",
    )
    return client


def to_history(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    return list(messages)


def user_message(text: str) -> HumanMessage:
    return HumanMessage(content=text)


def assistant_message(text: str) -> AIMessage:
    return AIMessage(content=text)


__all__ = [
    "AIMessage",
    "ASSISTANT_ROLE",
    "BaseLLMClient",
    "BaseMessage",
    "HumanMessage",
    "LLMClient",
    "LLMError",
    "OllamaLLMClient",
    "OpenAICompatibleLLMClient",
    "SYSTEM_ROLE",
    "SystemMessage",
    "USER_ROLE",
    "assistant_message",
    "build_llm_client",
    "to_history",
    "user_message",
]
