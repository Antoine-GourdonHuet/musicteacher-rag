"""Sanity check of the LLM judge with answers whose verdict is known ("evaluate the evaluator").

Takes QA answers that the judge scored as correct in a real run, then measures:
1. fabrication caught: a fabricated claim is appended -> faithfulness must be 0 (sensitivity);
2. verbatim accepted: the answer is replaced by two sentences copied from passage [1] of the
   context -> faithfulness must be 1 (no false alarm on text that is literally supported);
3. swapped answer caught: the answer of another question is given -> correctness must drop.
Acceptance threshold fixed before looking at the results: >= 90% on each test.

Usage: uv run python -m musicteacher.evaluation.judge_check [--condition C1] [--n 20]
"""

import argparse
import json
import random
import re

from musicteacher.config import get_settings
from musicteacher.evaluation.judge import judge_correctness, judge_faithfulness
from musicteacher.evaluation.report import latest_full_runs
from musicteacher.generation.rag import format_context
from musicteacher.llm.base import LLMClient
from musicteacher.retrieval.search import Hit

FABRICATED = {
    "en": "This was invented by Johann Sebastian Bach in 1950 to tune electric pianos [1].",
    "fr": "Cela a été inventé par Jean-Sébastien Bach en 1950 pour accorder les pianos [1].",
}
THRESHOLD = 0.9


def verbatim_answer(passage: str) -> str:
    """The first two sentences of a passage, cited: an answer that is supported by construction."""
    sentences = re.split(r"(?<=[.!?])\s+", passage.replace("\n", " ").strip())
    return " ".join(sentences[:2]) + " [1]"


def rate(values: list[bool]) -> dict:
    n = len(values)
    return {"ok": sum(values), "n": n, "rate": round(sum(values) / n, 3) if n else None}


def check(
    client: LLMClient, rows: list[dict], contexts: dict[str, str], first_passages: dict[str, str]
) -> dict:
    fabrication, verbatim, swapped = [], [], []
    others = rows[1:] + rows[:1]  # each question gets the answer of another question
    for row, other in zip(rows, others, strict=True):
        context = contexts[row["id"]]
        broken = f"{row['answer'].strip()} {FABRICATED[row['lang']]}"
        verdict = judge_faithfulness(client, broken, context)
        if verdict["faithfulness"] is not None:
            fabrication.append(verdict["faithfulness"] == 0)
        verdict = judge_faithfulness(client, verbatim_answer(first_passages[row["id"]]), context)
        if verdict["faithfulness"] is not None:
            verbatim.append(verdict["faithfulness"] == 1)
        verdict = judge_correctness(
            client, row["question"], row["reference_answer"], other["answer"]
        )
        if verdict["correctness"] is not None:
            swapped.append(verdict["correctness"] < 1)
    result = {
        "fabrication_caught": rate(fabrication),
        "verbatim_accepted": rate(verbatim),
        "swapped_answer_caught": rate(swapped),
    }
    result["passes_threshold"] = all(
        r["rate"] is not None and r["rate"] >= THRESHOLD for r in result.values()
    )
    return result


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
        if q["qtype"] != "agentic" and q["correctness"] == 1 and q["retrieved"]
    ]
    rows = sorted(
        random.Random(settings.seed).sample(candidates, min(args.n, len(candidates))),
        key=lambda q: q["id"],
    )
    records = {
        r["chunk_id"]: r for r in load_records(settings.data_dir / "processed" / "chunks.jsonl")
    }
    contexts, first_passages = {}, {}
    for q in rows:
        hits = [
            Hit(i, 0.0, c, records[c]["title"], records[c]["section"], "", "", records[c]["text"])
            for i, c in enumerate(q["retrieved"][: settings.top_k], start=1)
        ]
        contexts[q["id"]] = format_context(hits)
        first_passages[q["id"]] = hits[0].text

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
        **check(judge, rows, contexts, first_passages),
    }
    judge.close()
    out = settings.evals_dir / "results" / "judge_check.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
