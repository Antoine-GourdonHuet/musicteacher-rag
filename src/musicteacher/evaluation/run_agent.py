"""Condition C5: the agent with tools, on the same questions and with the same judge as C0-C4.

Extra metrics:
- tool_selection: the first tool the agent calls is the expected one (`expected_tool` for agentic
  questions, `search_corpus` for QA questions);
- args_correct (agentic only): the arguments of the expected tool call match `expected_tool_args`
  (semitones compared modulo 12, notes and keys compared by pitch);
- retrieval metrics come from the first `search_corpus` call (top 5).
Faithfulness (QA only) is judged against every passage the agent received; a QA answer given
without any search is judged against an empty context, so unsupported claims count as unfaithful.

Usage: uv run python -m musicteacher.evaluation.run_agent [--split all] [--limit 5]
       [--require-tool]   (condition C5b: ask once for a tool call if the first reply has none)
"""

import argparse
import json
import random
import time
from datetime import UTC, datetime

from musicteacher.agent.loop import AGENT_SYSTEM, AgentResult, run_agent
from musicteacher.agent.music import parse_key, pitch_class
from musicteacher.agent.tools import build_registry
from musicteacher.config import Settings, get_settings
from musicteacher.evaluation.dataset import EvalQuestion, load_questions
from musicteacher.evaluation.judge import CORRECTNESS_SYSTEM, FAITHFULNESS_SYSTEM, judge_answer
from musicteacher.evaluation.metrics import summarise
from musicteacher.evaluation.run import METRICS, git_info, prompt_hash, retrieval_scores

AGENT_METRICS = METRICS + ["tool_selection", "args_correct"]
MAX_STEPS = 4
NO_CONTEXT = "(no passages were retrieved)"


def args_match(tool: str, expected: dict, actual: dict) -> bool:
    try:
        if tool == "transpose":
            same_chords = [str(c).strip() for c in actual["chords"]] == expected["chords"]
            return same_chords and (int(actual["semitones"]) - expected["semitones"]) % 12 == 0
        if tool == "interval":
            return all(
                pitch_class(str(actual[k])) == pitch_class(expected[k]) for k in ("note1", "note2")
            )
        if tool == "key_info":
            (tonic_a, mode_a), (tonic_e, mode_e) = (
                parse_key(actual["key"]),
                parse_key(expected["key"]),
            )
            return mode_a == mode_e and pitch_class(tonic_a) == pitch_class(tonic_e)
    except (KeyError, TypeError, ValueError):
        return False
    return False


def score_question(judge_llm, q: EvalQuestion, result: AgentResult, top_k: int) -> dict:
    """Tool, retrieval and judge metrics for one agent run."""
    tools_called = [s.tool for s in result.steps]
    expected_tool = q.expected_tool or "search_corpus"
    searches = [s for s in result.steps if s.tool == "search_corpus" and s.ok]

    retrieval = dict.fromkeys(["recall_at_5", "article_recall_at_5", "mrr_at_10"])
    if q.qtype != "agentic" and searches:
        retrieval = retrieval_scores(q, searches[0].data, top_k)
    elif q.qtype != "agentic":
        retrieval = {"recall_at_5": 0.0, "article_recall_at_5": 0.0, "mrr_at_10": 0.0}

    args_correct = None
    if q.qtype == "agentic":
        expected_calls = [s for s in result.steps if s.tool == q.expected_tool]
        args_correct = float(
            bool(expected_calls)
            and args_match(q.expected_tool, q.expected_tool_args or {}, expected_calls[0].arguments)
        )

    context = None
    if q.qtype != "agentic":
        context = "\n\n".join(s.result for s in searches) or NO_CONTEXT
    verdict = judge_answer(judge_llm, q.question, q.reference_answer, result.answer, context)

    return {
        "id": q.id,
        "lang": q.lang,
        "qtype": q.qtype,
        "cross_lingual": q.cross_lingual,
        "split": q.split,
        "question": q.question,
        "reference_answer": q.reference_answer,
        "answer": result.answer,
        "retrieved": [h.chunk_id for h in searches[0].data] if searches else [],
        "source_chunk_ids": q.source_chunk_ids,
        **retrieval,
        **verdict,
        "tool_selection": float(bool(tools_called) and tools_called[0] == expected_tool),
        "args_correct": args_correct,
        "expected_tool": expected_tool,
        "tools_called": tools_called,
        "agent_turns": result.turns,
        "agent_stopped": result.stopped,
        "agent_nudged": result.nudged,
        "trace": [
            {"tool": s.tool, "arguments": s.arguments, "ok": s.ok, "result": s.result[:500]}
            for s in result.steps
        ],
        "gen_tokens_in": result.input_tokens,
        "gen_tokens_out": result.output_tokens,
        "gen_latency_s": round(result.latency_s, 2),
    }


def run(split: str, limit: int | None, settings: Settings, require_tool: bool = False):
    from musicteacher.llm.cache import CachedLLMClient
    from musicteacher.llm.ollama_client import OllamaClient
    from musicteacher.llm.usage import UsageTracker
    from musicteacher.retrieval.backends import load_retriever

    questions: list[EvalQuestion] = load_questions(settings.evals_dir / "questions.jsonl")
    if split != "all":
        questions = [q for q in questions if q.split == split]
    if limit:
        questions = sorted(
            random.Random(settings.seed).sample(questions, limit), key=lambda q: q.id
        )
    condition = "C5b" if require_tool else "C5"
    print(f"{condition} (agent with tools) on {len(questions)} questions [{split}]")
    started = time.perf_counter()

    # Retrieval on the CPU: the GPU is left to the LLM, which runs in the same loop.
    retriever = load_retriever(settings, with_reranker=True, device="cpu")
    registry = build_registry(retriever, mode="hybrid_rerank", k=settings.top_k)
    agent_usage, judge_usage = UsageTracker(), UsageTracker()
    agent_llm = CachedLLMClient(
        OllamaClient(settings.generator_model, settings.ollama_base_url, seed=settings.seed),
        settings.cache_dir / "llm",
        usage=agent_usage,
    )
    judge_llm = CachedLLMClient(
        OllamaClient(
            settings.judge_model, settings.ollama_base_url, seed=settings.seed, think=None
        ),
        settings.cache_dir / "llm",
        usage=judge_usage,
    )
    # Two phases, like C0-C4: the agent and the judge models do not fit together in 8 GB of GPU
    # memory, and alternating them per question makes Ollama reload a model every time.
    agent_runs = []
    for i, q in enumerate(questions, start=1):
        agent_runs.append(
            run_agent(
                agent_llm, registry, q.question, max_steps=MAX_STEPS, require_tool=require_tool
            )
        )
        tools = [s.tool for s in agent_runs[-1].steps]
        print(f"  agent {i}/{len(questions)} {q.id}: tools={tools}", flush=True)
    agent_llm.close()
    rows = []
    for i, (q, agent_run) in enumerate(zip(questions, agent_runs, strict=True), start=1):
        rows.append(score_question(judge_llm, q, agent_run, settings.top_k))
        print(f"  judging {i}/{len(questions)}", end="\r", flush=True)
    print()
    judge_llm.close()

    result = {
        "condition": condition,
        "retrieval_mode": "agent",
        "split": split,
        "limit": limit,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_info(settings.data_dir.parent),
        "config": {
            "generator_model": settings.generator_model,
            "judge_model": settings.judge_model,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "search_mode": "hybrid_rerank",
            "top_k": settings.top_k,
            "max_steps": MAX_STEPS,
            "require_tool": require_tool,
            "temperature": settings.temperature,
            "seed": settings.seed,
            "tools": list(registry.tools),
            "prompt_hash": prompt_hash(AGENT_SYSTEM, CORRECTNESS_SYSTEM, FAITHFULNESS_SYSTEM),
        },
        "n_questions": len(rows),
        "judge_invalid": sum(not r["judge_valid"] for r in rows),
        "usage": {"agent": agent_usage.summary(), "judge": judge_usage.summary()},
        "wall_time_s": round(time.perf_counter() - started, 1),
        "metrics": summarise(rows, AGENT_METRICS, seed=settings.seed),
        "questions": rows,
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    suffix = f"_n{limit}" if limit else ""
    out = settings.evals_dir / "results" / f"{condition}_{split}{suffix}_{stamp}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run condition C5 (agent with tools).")
    parser.add_argument("--split", default="all", choices=["dev", "heldout", "all"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--require-tool",
        action="store_true",
        help="C5b: if the first reply uses no tool, ask the model once to call a tool",
    )
    args = parser.parse_args()
    out = run(args.split, args.limit, get_settings(), require_tool=args.require_tool)
    result = json.loads(out.read_text(encoding="utf-8"))
    print(f"\nSaved {out}")
    for name, value in result["metrics"]["overall"].items():
        if value["mean"] is not None:
            print(
                f"  {name:20} {value['mean']:.3f}  "
                f"[{value['ci_low']:.3f}, {value['ci_high']:.3f}]  n={value['n']}"
            )
    agentic = result["metrics"]["by_qtype"].get("agentic", {})
    for name in ("tool_selection", "args_correct", "correctness"):
        if agentic.get(name, {}).get("mean") is not None:
            print(f"  agentic {name:14} {agentic[name]['mean']:.3f}  n={agentic[name]['n']}")
    print(f"  judge invalid: {result['judge_invalid']} | wall time {result['wall_time_s']}s")


if __name__ == "__main__":
    main()
