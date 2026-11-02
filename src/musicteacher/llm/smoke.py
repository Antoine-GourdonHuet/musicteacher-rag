"""Smoke test against the real local model: one call, then the same call served from the cache.

Usage: uv run python -m musicteacher.llm.smoke
"""

import time

from musicteacher.config import get_settings
from musicteacher.llm.base import Message
from musicteacher.llm.cache import CachedLLMClient
from musicteacher.llm.ollama_client import OllamaClient


def main() -> None:
    settings = get_settings()
    inner = OllamaClient(
        settings.generator_model, settings.ollama_base_url, seed=settings.seed
    )
    client = CachedLLMClient(inner, settings.cache_dir / "llm")
    question = [Message("user", "Name the three notes of a C major triad. Answer in one line.")]

    for attempt in (1, 2):
        start = time.perf_counter()
        reply = client.complete(question, temperature=settings.temperature, max_tokens=100)
        wall_s = time.perf_counter() - start
        print(
            f"[call {attempt}] cached={reply.cached} wall time={wall_s:.3f}s "
            f"tokens in/out={reply.input_tokens}/{reply.output_tokens}"
        )
        print(f"  {reply.text.strip()}")
    print(client.usage.summary())
    client.close()


if __name__ == "__main__":
    main()
