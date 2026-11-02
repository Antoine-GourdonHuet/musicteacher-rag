"""Deterministic `LLMClient` for tests: no network, scripted replies, records every call."""

from musicteacher.llm.base import LLMResponse, Message


class FakeLLMClient:
    def __init__(self, replies: list[str] | None = None, model: str = "fake-model") -> None:
        self.model = model
        self._replies = list(replies or [])
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
