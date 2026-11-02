"""Answer generation: one prompt for every retrieval condition (C1-C4), a closed-book one for C0.

The only difference between C1-C4 is which passages are retrieved; the prompt, the model, the
temperature, `top_k` and `max_tokens` stay the same. C0 has no passages, so it cannot use the
"answer only from the context" instruction: it gets the same rules minus the context ones.
"""

from musicteacher.llm.base import LLMClient, LLMResponse, Message
from musicteacher.retrieval.search import Hit

RAG_SYSTEM = """You are a music theory and guitar teaching assistant.
Answer the question using ONLY the numbered context passages.
- Cite the passages you use with their number in brackets, for example [2].
- If the passages do not contain the answer, say that you do not know. Never use outside knowledge.
- Answer in the same language as the question, in 1 to 4 sentences."""

CLOSED_BOOK_SYSTEM = """You are a music theory and guitar teaching assistant.
Answer the question from your own knowledge.
- Answer in the same language as the question, in 1 to 4 sentences."""

MAX_ANSWER_TOKENS = 300


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] {hit.title} — {hit.section}\n{hit.text}" for i, hit in enumerate(hits, start=1)
    )


def generate_answer(
    client: LLMClient, question: str, hits: list[Hit] | None, temperature: float = 0.0
) -> LLMResponse:
    """`hits=None` means closed book (C0); otherwise the top-k passages are given as context."""
    if hits is None:
        system, content = CLOSED_BOOK_SYSTEM, f"Question: {question}"
    else:
        system = RAG_SYSTEM
        content = f"Context passages:\n\n{format_context(hits)}\n\nQuestion: {question}"
    return client.complete(
        [Message("user", content)],
        system=system,
        temperature=temperature,
        max_tokens=MAX_ANSWER_TOKENS,
    )
