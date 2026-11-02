"""Deterministic `LLMClient` for tests: no network, scripted replies, records every call."""

from typing import Any

from musicteacher.llm.base import LLMResponse, Message, ToolTurn


class FakeLLMClient:
    def __init__(
        self,
        replies: list[str] | None = None,
        model: str = "fake-model",
        turns: list[ToolTurn] | None = None,
    ) -> None:
        self.model = model
        self._replies = list(replies or [])
        self._turns = list(turns or [])  # scripted replies for chat_with_tools
        self.calls: list[dict] = []

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "system": system, "temperature": temperature})
        text = self._replies.pop(0) if self._replies else "fake answer"
        return LLMResponse(
            text=text,
            input_tokens=sum(len(m.content.split()) for m in messages),
            output_tokens=len(text.split()),
        )

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ToolTurn:
        self.calls.append(
            {"messages": [dict(m) for m in messages], "tools": tools, "system": system}
        )
        return self._turns.pop(0) if self._turns else ToolTurn(content="fake answer", tool_calls=[])
