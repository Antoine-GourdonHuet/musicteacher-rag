import json

from musicteacher.evaluation.metrics import summarise
from musicteacher.evaluation.report import build_report, latest_full_runs, paired_difference
from musicteacher.evaluation.run import METRICS


def make_run(condition: str, mode, correctness: list[float], limit=None) -> dict:
    questions = [
        {
            "id": f"q{i}",
            "lang": "en" if i % 2 else "fr",
            "qtype": "definitional",
            "cross_lingual": i == 0,
            "split": "heldout" if i == 0 else "dev",
            "recall_at_5": 1.0,
            "article_recall_at_5": 1.0,
            "mrr_at_10": 1.0,
            "correctness": c,
            "faithfulness": 1.0,
            "relevancy": 1.0,
            "gen_latency_s": 2.0,
        }
        for i, c in enumerate(correctness)
    ]
    return {
        "condition": condition,
        "retrieval_mode": mode,
        "split": "all",
        "limit": limit,
        "n_questions": len(questions),
        "judge_invalid": 0,
        "config": {"generator_model": "g", "judge_model": "j", "top_k": 5, "temperature": 0.0},
        "metrics": summarise(questions, METRICS),
        "questions": questions,
    }


def test_paired_difference_is_per_question():
    a = make_run("C3", "hybrid", [1, 1, 1, 0])
    b = make_run("C1", "dense", [0, 1, 1, 0])
    diff = paired_difference(a, b, "correctness")
    assert diff["mean"] == 0.25
    assert diff["n"] == 4


def test_latest_full_runs_ignores_smoke_runs(tmp_path):
    for name, run in [
        ("C1_all_20260101T000000.json", make_run("C1", "dense", [0, 1])),
        ("C1_all_20260102T000000.json", make_run("C1", "dense", [1, 1])),
        ("C1_dev_n5_20260103T000000.json", make_run("C1", "dense", [0, 0], limit=5)),
    ]:
        (tmp_path / name).write_text(json.dumps(run), encoding="utf-8")
    runs = latest_full_runs(tmp_path)
    assert runs["C1"]["_file"] == "C1_all_20260102T000000.json"


def test_report_contains_tables_and_comparisons():
    runs = {
        "C0": make_run("C0", None, [0, 0, 1, 0]),
        "C1": make_run("C1", "dense", [0, 1, 1, 0]),
        "C3": make_run("C3", "hybrid", [1, 1, 1, 0]),
    }
    for r in runs.values():
        r["_file"] = f"{r['condition']}.json"
    report = build_report(runs)
    assert "## Main table" in report
    assert "C3 − C1 (H1: hybrid vs dense)" in report
    assert "C1 (dense)" in report and "C0 (no retrieval)" in report
    assert "### heldout" in report


def test_report_has_an_agent_section_with_paired_agentic_comparison():
    c4 = make_run("C4", "hybrid_rerank", [0, 0, 1, 1])
    c5 = make_run("C5", "agent", [1, 1, 1, 1])
    for run in (c4, c5):
        for q in run["questions"]:
            q["qtype"] = "agentic"
            q.update({"tool_selection": 1.0, "args_correct": 1.0, "agent_stopped": "answer"})
        run["_file"] = f"{run['condition']}.json"
    report = build_report({"C4": c4, "C5": c5})
    assert "## Agent with tools (C5)" in report
    assert "| C5 − C4 (agentic only) | +0.50" in report
    assert "Agent stop reasons: answer 4" in report


def test_readme_block_is_written_between_markers_only(tmp_path):
    from musicteacher.evaluation.report import readme_block, update_readme

    runs = {"C0": make_run("C0", None, [0, 1]), "C1": make_run("C1", "dense", [1, 1])}
    corpus = {
        "articles": 3,
        "chunks": 9,
        "chunks_en": 5,
        "chunks_fr": 4,
        "chunk_tokens_total": 1200,
    }
    block = readme_block(runs, corpus, None, seed=42)
    assert "Corpus: 3 Wikipedia articles, 9 chunks (5 EN / 4 FR), 1,200 tokens." in block
    assert "| C1 − C0: retrieval vs no retrieval |" in block

    readme = tmp_path / "README.md"
    readme.write_text(
        "intro\n<!-- results:start -->\nold\n<!-- results:end -->\noutro\n", encoding="utf-8"
    )
    update_readme(readme, block)
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("intro\n<!-- results:start -->\n<!-- Generated")
    assert "old" not in text and text.endswith("<!-- results:end -->\noutro\n")
