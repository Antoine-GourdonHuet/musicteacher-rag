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


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolTurn:
    """One reply of a tool-using model: either text, or one or more tool calls."""

    content: str
    tool_calls: list[ToolCall]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False


class ToolLLMClient(Protocol):
    model: str

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ToolTurn:
        """One model turn; `messages` use the chat format (user / assistant / tool roles)."""
        ...
