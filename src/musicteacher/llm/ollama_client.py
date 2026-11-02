"""`LLMClient` backed by a local Ollama server (https://docs.ollama.com/api).

Uses the non-streaming `POST /api/chat` endpoint directly through `httpx`: no SDK, so every field we
send and read is visible here.
"""

import json
import time
from typing import Any

import httpx

from musicteacher.llm.base import LLMResponse, Message, ToolCall, ToolTurn


class OllamaClient:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        *,
        seed: int = 42,
        think: bool | None = False,
        timeout_s: float = 300.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.seed = seed
        # Thinking models (e.g. Qwen 3.5) reason before answering; we disable it for shorter, more
        # reproducible answers. Set to None for models that do not support the `think` field.
        self.think = think
        self._http = http_client or httpx.Client(base_url=base_url, timeout=timeout_s)

    def build_payload(
        self,
        messages: list[Message],
        *,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        chat = [{"role": "system", "content": system}] if system else []
        chat += [{"role": m.role, "content": m.content} for m in messages]
        payload: dict = {
            "model": self.model,
            "messages": chat,
            "stream": False,
            "options": {"temperature": temperature, "seed": self.seed, "num_predict": max_tokens},
        }
        if self.think is not None:
            payload["think"] = self.think
        return payload

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        payload = self.build_payload(
            messages, system=system, temperature=temperature, max_tokens=max_tokens
        )
        start = time.perf_counter()
        response = self._http.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return LLMResponse(
            text=data["message"]["content"],
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_s=time.perf_counter() - start,
            raw=data,
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
        """Native Ollama tool calling: the model may reply with `tool_calls` instead of text."""
        chat = ([{"role": "system", "content": system}] if system else []) + messages
        payload: dict = {
            "model": self.model,
            "messages": chat,
            "tools": tools,
            "stream": False,
            "options": {"temperature": temperature, "seed": self.seed, "num_predict": max_tokens},
        }
        if self.think is not None:
            payload["think"] = self.think
        start = time.perf_counter()
        response = self._http.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        message = data["message"]
        calls = [
            ToolCall(c["function"]["name"], parse_arguments(c["function"].get("arguments")))
            for c in message.get("tool_calls") or []
        ]
        return ToolTurn(
            content=message.get("content", ""),
            tool_calls=calls,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_s=time.perf_counter() - start,
        )


def parse_arguments(raw: object) -> dict[str, Any]:
    """Tool arguments arrive as a dict, or sometimes as a JSON string; anything else -> {}."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}
