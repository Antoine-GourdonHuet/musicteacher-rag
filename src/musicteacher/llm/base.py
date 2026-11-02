"""Provider-agnostic LLM interface.

Every experiment talks to an `LLMClient`, so the generator can be swapped (local Ollama model,
a cached wrapper, a fake in tests) without touching the rest of the code.
"""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Message:
    role: str  # "user" or "assistant"
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False
    raw: Any = None


class LLMClient(Protocol):
    model: str

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Return the model's reply to `messages`."""
        ...
