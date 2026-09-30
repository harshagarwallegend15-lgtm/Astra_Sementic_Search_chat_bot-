"""Test fixtures stubbed LLM clients.

Used to test the answering orchestration without a running Ollama instance.
"""

from __future__ import annotations

from typing import Iterator, Sequence

from src.config import Settings
from src.models import RetrievedChunk


class StubLLM:
    """Records the prompts it receives and returns canned responses."""

    def __init__(self, responses: Sequence[str] | str = "Stub answer [S1].") -> None:
        self.responses = (
            [responses] if isinstance(responses, str) else list(responses)
        )
        self.calls: list[tuple[str, str]] = []
        self._index = 0

    def _next(self) -> str:
        if self._index < len(self.responses):
            value = self.responses[self._index]
        else:
            value = self.responses[-1] if self.responses else ""
        self._index += 1
        return value

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Sequence[object] | None = None,
    ) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self._next()

    def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Sequence[object] | None = None,
    ) -> Iterator[str]:
        self.calls.append((system_prompt, user_prompt))
        for piece in self._next().split(" "):
            yield piece + " "

    @property
    def user_prompts(self) -> list[str]:
        return [prompt for _, prompt in self.calls]


class EchoingLLM(StubLLM):
    """Always copies the SOURCES block, to exercise the echo retry."""

    def complete(self, system_prompt, user_prompt, history=None):  # type: ignore[no-untyped-def]
        self.calls.append((system_prompt, user_prompt))
        return (
            "[S1] Document: Unmanned aerial vehicle\n"
            "Page: 14\n"
            "Section: Autonomy\n"
            "Text: The level of autonomy in UAVs varies widely."
        )


class UnavailableLLM:
    """An LLM client whose every call fails."""

    def complete(self, system_prompt, user_prompt, history=None):  # type: ignore[no-untyped-def]
        from src.errors import LLMError

        raise LLMError("simulated provider outage")

    def stream(self, system_prompt, user_prompt, history=None):  # type: ignore[no-untyped-def]
        from src.errors import LLMError

        raise LLMError("simulated provider outage")
        yield ""  # pragma: no cover


__all__ = [
    "EchoingLLM",
    "StubLLM",
    "UnavailableLLM",
    "RetrievedChunk",
    "Settings",
]
