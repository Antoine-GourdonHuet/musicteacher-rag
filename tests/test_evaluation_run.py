import json

import pytest

from musicteacher.evaluation.dataset import EvalQuestion
from musicteacher.evaluation.judge import judge_answer
from musicteacher.evaluation.metrics import bootstrap_ci, recall_at_k, reciprocal_rank, summarise
from musicteacher.evaluation.run import retrieval_scores
from musicteacher.generation.rag import CLOSED_BOOK_SYSTEM, RAG_SYSTEM, generate_answer
from musicteacher.llm.fake import FakeLLMClient
from musicteacher.retrieval.search import Hit


def hit(chunk_id: str) -> Hit:
    return Hit(1, 1.0, chunk_id, "Capo", "Introduction", "en", "u", f"text of {chunk_id}")


def test_recall_and_reciprocal_rank():
    retrieved = ["a#0", "b#0", "c#0"]
    assert recall_at_k(retrieved, ["b#0"], k=2) == 1.0
    assert recall_at_k(retrieved, ["c#0"], k=2) == 0.0
    assert recall_at_k(retrieved, ["a#0", "z#0"], k=3) == 0.5  # multihop: half found
    assert reciprocal_rank(retrieved, ["c#0"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(retrieved, ["z#0"]) == 0.0


def test_bootstrap_ci_brackets_the_mean_and_shrinks_with_more_data():
    small = bootstrap_ci([0, 1] * 5)
    large = bootstrap_ci([0, 1] * 500)
    assert small["mean"] == large["mean"] == 0.5
    assert small["ci_low"] <= 0.5 <= small["ci_high"]
    assert (large["ci_high"] - large["ci_low"]) < (small["ci_high"] - small["ci_low"])
    assert bootstrap_ci([])["n"] == 0


def test_summarise_breaks_down_by_group_and_skips_missing_values():
    rows = [
        {"lang": "en", "qtype": "definitional", "cross_lingual": False, "m": 1.0},
        {"lang": "fr", "qtype": "definitional", "cross_lingual": True, "m": 0.0},
        {"lang": "fr", "qtype": "agentic", "cross_lingual": False, "m": None},
    ]
    summary = summarise(rows, ["m"])
    assert summary["overall"]["m"]["n"] == 2
    assert summary["by_lang"]["en"]["m"]["mean"] == 1.0
    assert summary["by_qtype"]["agentic"]["m"]["n"] == 0
    assert summary["cross_lingual"]["m"]["mean"] == 0.0


def test_retrieval_scores_chunk_and_article_level():
    q = EvalQuestion(
        id="en-defi-001",
        lang="en",
        qtype="definitional",
        question="q",
        reference_answer="a",
        source_chunk_ids=["en-capo#002"],
    )
    hits = [hit("en-capo#000"), hit("en-guitar#010"), hit("en-capo#002")]
    scores = retrieval_scores(q, hits, top_k=5)
    assert scores == {
        "recall_at_5": 1.0,
        "article_recall_at_5": 1.0,
        "mrr_at_10": pytest.approx(1 / 3),
    }
    agentic = q.model_copy(update={"qtype": "agentic", "source_chunk_ids": []})
    assert retrieval_scores(agentic, hits, top_k=5)["recall_at_5"] is None


def test_generation_uses_context_prompt_or_closed_book():
    client = FakeLLMClient(["answer [1]", "answer"])
    generate_answer(client, "What is a capo?", [hit("en-capo#000")])
    generate_answer(client, "What is a capo?", None)
    rag_call, closed_call = client.calls
    assert rag_call["system"] == RAG_SYSTEM
    assert "[1] Capo — Introduction" in rag_call["messages"][0].content
    assert closed_call["system"] == CLOSED_BOOK_SYSTEM
    assert "Context" not in closed_call["messages"][0].content


def test_judge_parses_scores_and_flags_invalid_replies():
    good = json.dumps({"correctness": 1, "faithfulness": 0.5, "relevancy": 1, "reason": "ok"})
    verdict = judge_answer(FakeLLMClient([good]), "q", "ref", "ans", context="[1] passage")
    assert (verdict["correctness"], verdict["faithfulness"], verdict["relevancy"]) == (1, 0.5, 1)
    assert verdict["judge_valid"] is True

    closed = judge_answer(FakeLLMClient([good]), "q", "ref", "ans", context=None)
    assert closed["faithfulness"] is None  # no context: faithfulness is not defined
    assert closed["judge_valid"] is True

    odd = json.dumps({"correctness": 0.7, "faithfulness": 1, "relevancy": 1})
    assert judge_answer(FakeLLMClient([odd]), "q", "r", "a", "[1] p")["judge_valid"] is False
    assert judge_answer(FakeLLMClient(["no json"]), "q", "r", "a", "[1] p")["judge_valid"] is False
