"""Sanity check of the LLM judge with deliberately broken answers ("evaluate the evaluator").

Takes answers that the judge scored as correct and faithful in a real run, then:
1. injects a fabricated claim into each answer -> faithfulness should drop below 1;
2. swaps answers between questions -> correctness should drop to 0.
The detection rates say how far the judge's faithfulness and correctness scores can be trusted.

Usage: uv run python -m musicteacher.evaluation.judge_check [--condition C1] [--n 20]
"""

import argparse
import json
import random

from musicteacher.config import get_settings
from musicteacher.evaluation.judge import judge_answer
from musicteacher.evaluation.report import latest_full_runs
from musicteacher.generation.rag import format_context
from musicteacher.llm.base import LLMClient
from musicteacher.retrieval.search import Hit

FABRICATED = {
    "en": "This was invented by Johann Sebastian Bach in 1950 to tune electric pianos [1].",
    "fr": "Cela a été inventé par Jean-Sébastien Bach en 1950 pour accorder les pianos [1].",
}


def check(client: LLMClient, rows: list[dict], contexts: dict[str, str]) -> dict:
    """Detection rates of injected fabrications (faithfulness) and swapped answers (correctness)."""
    fabrication_caught, swap_caught = [], []
    shuffled = rows[1:] + rows[:1]  # each question gets the answer of another question
    for row, other in zip(rows, shuffled, strict=True):
        broken = f"{row['answer'].strip()} {FABRICATED[row['lang']]}"
        verdict = judge_answer(
            client, row["question"], row["reference_answer"], broken, contexts[row["id"]]
        )
        if verdict["faithfulness"] is not None:
            fabrication_caught.append(verdict["faithfulness"] < 1)
        verdict = judge_answer(
            client, row["question"], row["reference_answer"], other["answer"], contexts[row["id"]]
        )
        if verdict["correctness"] is not None:
            swap_caught.append(verdict["correctness"] < 1)

    def rate(values: list[bool]) -> dict:
        return {
            "caught": sum(values),
            "n": len(values),
            "rate": round(sum(values) / len(values), 3),
        }

    return {"fabrication": rate(fabrication_caught), "swapped_answer": rate(swap_caught)}


def main() -> None:
    from musicteacher.llm.cache import CachedLLMClient
    from musicteacher.llm.ollama_client import OllamaClient
    from musicteacher.retrieval.backends import load_records

    parser = argparse.ArgumentParser(description="Sanity check of the LLM judge.")
    parser.add_argument("--condition", default="C1")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    settings = get_settings()

    run = latest_full_runs(settings.evals_dir / "results")[args.condition]
    candidates = [
        q
        for q in run["questions"]
        if q["qtype"] != "agentic" and q["correctness"] == 1 and q["faithfulness"] == 1
    ]
    rows = sorted(
        random.Random(settings.seed).sample(candidates, min(args.n, len(candidates))),
        key=lambda q: q["id"],
    )
    records = {
        r["chunk_id"]: r for r in load_records(settings.data_dir / "processed" / "chunks.jsonl")
    }
    contexts = {}
    for q in rows:
        hits = [
            Hit(i, 0.0, c, records[c]["title"], records[c]["section"], "", "", records[c]["text"])
            for i, c in enumerate(q["retrieved"][: settings.top_k], start=1)
        ]
        contexts[q["id"]] = format_context(hits)

    judge = CachedLLMClient(
        OllamaClient(
            settings.judge_model, settings.ollama_base_url, seed=settings.seed, think=None
        ),
        settings.cache_dir / "llm",
    )
    result = {
        "condition": args.condition,
        "source_file": run["_file"],
        "judge_model": settings.judge_model,
        **check(judge, rows, contexts),
    }
    judge.close()
    out = settings.evals_dir / "results" / "judge_check.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
