"""Disk cache around any `LLMClient`.

Re-running an evaluation must return exactly the same outputs without calling the model again.
The key is a hash of everything that can change the answer: model, system prompt, messages,
temperature and max_tokens.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import diskcache

from musicteacher.llm.base import LLMClient, LLMResponse, Message, ToolCall, ToolTurn
from musicteacher.llm.usage import UsageTracker


def cache_key(
    model: str,
    system: str | None,
    messages: list[Message],
    temperature: float,
    max_tokens: int,
) -> str:
    blob = json.dumps(
        {
            "model": model,
            "system": system,
            "messages": [[m.role, m.content] for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CachedLLMClient:
    def __init__(
        self, inner: LLMClient, cache_dir: Path, usage: UsageTracker | None = None
    ) -> None:
        self.inner = inner
        self.model = inner.model
        self.usage = usage or UsageTracker()
        self._cache = diskcache.Cache(str(cache_dir))

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        key = cache_key(self.model, system, messages, temperature, max_tokens)
        hit = self._cache.get(key)
        if hit is not None:
            response = LLMResponse(**hit, cached=True)
        else:
            response = self.inner.complete(
                messages, system=system, temperature=temperature, max_tokens=max_tokens
            )
            # `raw` (the full provider payload) is not stored: only what the experiments need.
            self._cache.set(
                key,
                {
                    "text": response.text,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "latency_s": response.latency_s,
                },
            )
        self.usage.record(response)
        return response

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ToolTurn:
        """Same caching for tool-using turns; the key also covers the conversation and the tools."""
        blob = json.dumps(
            {
                "kind": "chat_with_tools",
                "model": self.model,
                "system": system,
                "messages": messages,
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        key = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        hit = self._cache.get(key)
        if hit is not None:
            turn = ToolTurn(
                content=hit["content"],
                tool_calls=[ToolCall(c["name"], c["arguments"]) for c in hit["tool_calls"]],
                input_tokens=hit["input_tokens"],
                output_tokens=hit["output_tokens"],
                latency_s=hit["latency_s"],
                cached=True,
            )
        else:
            turn = self.inner.chat_with_tools(  # type: ignore[attr-defined]
                messages, tools, system=system, temperature=temperature, max_tokens=max_tokens
            )
            self._cache.set(
                key,
                {
                    "content": turn.content,
                    "tool_calls": [
                        {"name": c.name, "arguments": c.arguments} for c in turn.tool_calls
                    ],
                    "input_tokens": turn.input_tokens,
                    "output_tokens": turn.output_tokens,
                    "latency_s": turn.latency_s,
                },
            )
        self.usage.record(turn)
        return turn

    def close(self) -> None:
        self._cache.close()
