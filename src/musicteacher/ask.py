"""Ask the assistant a question from the command line.

Usage:
    uv run python -m musicteacher.ask "Qu'est-ce que le mode dorien ?"
    uv run python -m musicteacher.ask "What is a capo?" --mode dense
    uv run python -m musicteacher.ask "Transpose C - Am - F - G up a minor third." --agent

Default: RAG with hybrid search + reranker (condition C4). `--mode` picks another retrieval mode
(dense, bm25, hybrid, hybrid_rerank). `--agent` uses the agent with tools, required to call a tool
(condition C5b). Needs Ollama running with the generator model, and the indexes built.
"""

import argparse
import sys

import httpx

from musicteacher.agent.loop import AgentResult, run_agent
from musicteacher.agent.tools import ToolRegistry
from musicteacher.generation.rag import generate_answer
from musicteacher.llm.base import LLMClient, ToolLLMClient
from musicteacher.retrieval.search import MODES, Hit, Retriever


def format_sources(hits: list[Hit]) -> str:
    return "\n".join(f"  [{h.rank}] {h.title} — {h.section} ({h.language})  {h.url}" for h in hits)


def rag_answer(
    question: str, retriever: Retriever, llm: LLMClient, mode: str, k: int
) -> tuple[str, list[Hit]]:
    hits = retriever.search(question, mode, k=k)
    return generate_answer(llm, question, hits).text.strip(), hits


def agent_answer(
    question: str, llm: ToolLLMClient, registry: ToolRegistry
) -> tuple[str, AgentResult, list[Hit]]:
    result = run_agent(llm, registry, question, require_tool=True)
    hits = [
        h for step in result.steps if step.tool == "search_corpus" and step.ok for h in step.data
    ]
    return result.answer, result, hits


def main() -> None:
    from musicteacher.agent.tools import build_registry
    from musicteacher.config import get_settings
    from musicteacher.llm.cache import CachedLLMClient
    from musicteacher.llm.ollama_client import OllamaClient
    from musicteacher.retrieval.backends import load_retriever

    parser = argparse.ArgumentParser(description="Ask the music theory assistant a question.")
    parser.add_argument("question")
    parser.add_argument("--mode", default="hybrid_rerank", choices=MODES)
    parser.add_argument("--agent", action="store_true", help="agent with tools (C5b)")
    args = parser.parse_args()

    settings = get_settings()
    llm = CachedLLMClient(
        OllamaClient(settings.generator_model, settings.ollama_base_url, seed=settings.seed),
        settings.cache_dir / "llm",
    )
    try:
        if args.agent:
            # Retrieval on the CPU leaves the GPU to the LLM, which runs in the same loop.
            retriever = load_retriever(settings, with_reranker=True, device="cpu")
            registry = build_registry(retriever, k=settings.top_k)
            answer, result, hits = agent_answer(args.question, llm, registry)
            print(f"\n{answer}\n")
            for step in result.steps:
                status = "ok" if step.ok else "error"
                print(f"  tool: {step.tool}({step.arguments}) -> {status}")
        else:
            retriever = load_retriever(settings, with_reranker=args.mode == "hybrid_rerank")
            answer, hits = rag_answer(args.question, retriever, llm, args.mode, settings.top_k)
            print(f"\n{answer}\n")
    except httpx.ConnectError:
        sys.exit(f"Cannot reach Ollama at {settings.ollama_base_url}: start the Ollama app first.")
    finally:
        llm.close()
    if hits:
        print("\nSources:\n" + format_sources(hits))


if __name__ == "__main__":
    main()
