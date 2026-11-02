"""Running counter of LLM usage: calls, cache hits, tokens and generation time.

The generator runs locally, so the cost to track is time rather than money.
"""

from dataclasses import dataclass

from musicteacher.llm.base import LLMResponse, ToolTurn


@dataclass
class UsageTracker:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    def record(self, response: LLMResponse | ToolTurn) -> None:
        self.calls += 1
        if response.cached:
            self.cache_hits += 1
            return
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.latency_s += response.latency_s

    def summary(self) -> str:
        return (
            f"LLM calls: {self.calls} ({self.cache_hits} cached) | "
            f"tokens in/out: {self.input_tokens}/{self.output_tokens} | "
            f"generation time: {self.latency_s:.1f}s"
        )
