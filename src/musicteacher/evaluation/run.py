"""Run one experimental condition on the evaluation set and write a result JSON.

Conditions (only the retrieval step changes):
    C0 no retrieval (closed book) | C1 dense | C2 BM25 | C3 hybrid (RRF) | C4 hybrid + reranker

Three phases, so the GPU only holds one family of models at a time (8 GB):
1. retrieval for every question (embedding + reranker models), then free the GPU;
2. answer generation with the generator LLM (Ollama, disk cache);
3. judging with the judge LLM (another model family, disk cache).

Usage:
    uv run python -m musicteacher.evaluation.run --condition C4 [--split dev] [--limit 5]
    (--split: dev | heldout | all)
"""

import argparse
import gc
import hashlib
import json
import random
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from musicteacher.config import Settings, get_settings
from musicteacher.evaluation.dataset import EvalQuestion, load_questions
from musicteacher.evaluation.judge import CORRECTNESS_SYSTEM, FAITHFULNESS_SYSTEM, judge_answer
from musicteacher.evaluation.metrics import recall_at_k, reciprocal_rank, summarise
from musicteacher.generation.rag import (
    CLOSED_BOOK_SYSTEM,
    MAX_ANSWER_TOKENS,
    RAG_SYSTEM,
    format_context,
    generate_answer,
)
from musicteacher.retrieval.search import Hit

CONDITIONS = {"C0": None, "C1": "dense", "C2": "bm25", "C3": "hybrid", "C4": "hybrid_rerank"}
RETRIEVE_DEPTH = 10  # retrieved for MRR@10; only the top_k first are given to the generator
METRICS = [
    "recall_at_5",
    "article_recall_at_5",
    "mrr_at_10",
    "correctness",
    "faithfulness",
    "relevancy",
]


def git_info(root: Path) -> dict:
    git = shutil.which("git") or r"C:\Program Files\Git\cmd\git.exe"
    try:
        commit = subprocess.run(
            [git, "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            [git, "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


def prompt_hash(*prompts: str) -> str:
    return hashlib.sha256("\n---\n".join(prompts).encode("utf-8")).hexdigest()[:12]


def retrieval_scores(q: EvalQuestion, hits: list[Hit], top_k: int) -> dict:
    """Chunk-level recall@5 and MRR@10 against the source chunks; article-level recall@5 too."""
    if not q.source_chunk_ids:  # agentic questions have no source passage
        return {"recall_at_5": None, "article_recall_at_5": None, "mrr_at_10": None}
    retrieved = [h.chunk_id for h in hits]
    source_articles = sorted({c.split("#")[0] for c in q.source_chunk_ids})
    return {
        "recall_at_5": recall_at_k(retrieved, q.source_chunk_ids, top_k),
        "article_recall_at_5": recall_at_k(
            [c.split("#")[0] for c in retrieved], source_articles, top_k
        ),
        "mrr_at_10": reciprocal_rank(retrieved, q.source_chunk_ids),
    }


def retrieve_all(questions: list[EvalQuestion], mode: str, settings: Settings) -> list[list[Hit]]:
    import torch

    from musicteacher.retrieval.backends import load_retriever

    retriever = load_retriever(settings, with_reranker=mode == "hybrid_rerank")
    results = []
    for i, q in enumerate(questions, start=1):
        results.append(retriever.search(q.question, mode, k=RETRIEVE_DEPTH))
        print(f"  retrieval {i}/{len(questions)}", end="\r")
    print()
    del retriever
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # leave the GPU to the Ollama models
    return results


def run(condition: str, split: str, limit: int | None, settings: Settings) -> Path:
    from musicteacher.llm.cache import CachedLLMClient
    from musicteacher.llm.ollama_client import OllamaClient
    from musicteacher.llm.usage import UsageTracker

    mode = CONDITIONS[condition]
    questions = load_questions(settings.evals_dir / "questions.jsonl")
    if split != "all":
        questions = [q for q in questions if q.split == split]
    if limit:  # a reproducible random sample, so a smoke run mixes question types
        questions = sorted(
            random.Random(settings.seed).sample(questions, limit), key=lambda q: q.id
        )
    print(f"{condition} ({mode or 'closed book'}) on {len(questions)} questions [{split}]")
    started = time.perf_counter()

    # Phase 1: retrieval
    all_hits = retrieve_all(questions, mode, settings) if mode else [None] * len(questions)

    # Phase 2: generation
    gen_usage = UsageTracker()
    generator = CachedLLMClient(
        OllamaClient(settings.generator_model, settings.ollama_base_url, seed=settings.seed),
        settings.cache_dir / "llm",
        usage=gen_usage,
    )
    answers = []
    for i, (q, hits) in enumerate(zip(questions, all_hits, strict=True), start=1):
        context_hits = hits[: settings.top_k] if hits is not None else None
        answers.append(generate_answer(generator, q.question, context_hits, settings.temperature))
        print(f"  generation {i}/{len(questions)} | {gen_usage.summary()}", end="\r")
    print()
    generator.close()

    # Phase 3: judging
    judge_usage = UsageTracker()
    judge = CachedLLMClient(
        OllamaClient(
            settings.judge_model, settings.ollama_base_url, seed=settings.seed, think=None
        ),
        settings.cache_dir / "llm",
        usage=judge_usage,
    )
    rows = []
    for i, (q, hits, answer) in enumerate(zip(questions, all_hits, answers, strict=True), start=1):
        context_hits = hits[: settings.top_k] if hits is not None else None
        # Faithfulness only makes sense for QA questions answered from passages, not for the
        # computations of agentic questions: the judge sees the context only for QA questions.
        context = (
            format_context(context_hits)
            if context_hits is not None and q.qtype != "agentic"
            else None
        )
        verdict = judge_answer(judge, q.question, q.reference_answer, answer.text, context)
        retrieval = (
            retrieval_scores(q, hits, settings.top_k)
            if hits is not None
            else dict.fromkeys(["recall_at_5", "article_recall_at_5", "mrr_at_10"])
        )
        rows.append(
            {
                "id": q.id,
                "lang": q.lang,
                "qtype": q.qtype,
                "cross_lingual": q.cross_lingual,
                "split": q.split,
                "question": q.question,
                "reference_answer": q.reference_answer,
                "answer": answer.text,
                "retrieved": [h.chunk_id for h in hits] if hits is not None else [],
                "source_chunk_ids": q.source_chunk_ids,
                **retrieval,
                **verdict,
                "gen_tokens_in": answer.input_tokens,
                "gen_tokens_out": answer.output_tokens,
                "gen_latency_s": round(answer.latency_s, 2),
            }
        )
        print(f"  judging {i}/{len(questions)} | {judge_usage.summary()}", end="\r")
    print()
    judge.close()

    result = {
        "condition": condition,
        "retrieval_mode": mode,
        "split": split,
        "limit": limit,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_info(settings.data_dir.parent),
        "config": {
            "generator_model": settings.generator_model,
            "judge_model": settings.judge_model,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "top_k": settings.top_k,
            "retrieve_depth": RETRIEVE_DEPTH,
            "fusion_depth": settings.fusion_depth,
            "rrf_k": settings.rrf_k,
            "rerank_candidates": settings.rerank_candidates,
            "temperature": settings.temperature,
            "seed": settings.seed,
            "max_answer_tokens": MAX_ANSWER_TOKENS,
            "prompt_hash": prompt_hash(
                RAG_SYSTEM, CLOSED_BOOK_SYSTEM, CORRECTNESS_SYSTEM, FAITHFULNESS_SYSTEM
            ),
        },
        "n_questions": len(rows),
        "judge_invalid": sum(not r["judge_valid"] for r in rows),
        "usage": {"generator": gen_usage.summary(), "judge": judge_usage.summary()},
        "wall_time_s": round(time.perf_counter() - started, 1),
        "metrics": summarise(rows, METRICS, seed=settings.seed),
        "questions": rows,
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    suffix = f"_n{limit}" if limit else ""
    out = settings.evals_dir / "results" / f"{condition}_{split}{suffix}_{stamp}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one condition of the experiment.")
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--split", default="dev", choices=["dev", "heldout", "all"])
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    args = parser.parse_args()

    out = run(args.condition, args.split, args.limit, get_settings())
    result = json.loads(out.read_text(encoding="utf-8"))
    print(f"\nSaved {out}")
    for name, value in result["metrics"]["overall"].items():
        if value["mean"] is not None:
            print(
                f"  {name:20} {value['mean']:.3f}  "
                f"[{value['ci_low']:.3f}, {value['ci_high']:.3f}]  n={value['n']}"
            )
    print(f"  judge invalid: {result['judge_invalid']} | wall time {result['wall_time_s']}s")


if __name__ == "__main__":
    main()
