import json
from collections import Counter

import pytest

from musicteacher.evaluation import review
from musicteacher.evaluation.dataset import EvalQuestion, load_questions


def make_draft(i: int, lang: str = "en", qtype: str = "definitional", origin: str = "llm_draft"):
    return {
        "id": f"{lang}-{qtype[:4]}-{i:03d}",
        "lang": lang,
        "qtype": qtype,
        "question": f"Question {i}?",
        "reference_answer": f"Answer {i}.",
        "source_passage": "Some passage.",
        "origin": origin,
        "review": {"copy_run_words": 0, "copy_flag": False},
    }


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_review_records_decisions_and_resumes(tmp_path, monkeypatch, capsys):
    drafts, state = tmp_path / "drafts.jsonl", tmp_path / "state.jsonl"
    write_jsonl(drafts, [make_draft(1), make_draft(2), make_draft(3)])

    answers = iter(["a", "e", "Better question?", "", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    review.review(drafts, state)

    decisions = review.latest_decisions(review.read_jsonl(state))
    assert decisions["en-defi-001"]["decision"] == "accept"
    assert decisions["en-defi-002"]["edited"] is True
    assert decisions["en-defi-002"]["question"]["question"] == "Better question?"
    assert decisions["en-defi-002"]["question"]["reference_answer"] == "Answer 2."
    assert "en-defi-003" not in decisions  # quit before deciding

    answers = iter(["r"])
    review.review(drafts, state)  # resumes on draft 3 only
    assert review.latest_decisions(review.read_jsonl(state))["en-defi-003"]["decision"] == "reject"


def test_splits_are_stratified_and_deterministic():
    questions = [
        EvalQuestion(id=f"{lang}-{t}-{i}", lang=lang, qtype=t, question="q", reference_answer="a")
        for lang in ("en", "fr")
        for t in ("definitional", "conceptual")
        for i in range(10)
    ]
    first = review.assign_splits(questions, seed=42)
    assert first == review.assign_splits(questions, seed=42)
    heldout = Counter((q.lang, q.qtype) for q in first if q.split == "heldout")
    assert set(heldout.values()) == {2}  # 20% of each group of 10


def test_freeze_writes_reviewed_questions_and_changelog(tmp_path):
    state, out, changelog = (
        tmp_path / "state.jsonl",
        tmp_path / "q.jsonl",
        tmp_path / "CHANGELOG.md",
    )
    changelog.write_text("# Changelog\n", encoding="utf-8")
    decisions = [
        {"id": "en-defi-001", "decision": "accept", "edited": False, "question": make_draft(1)},
        {"id": "en-defi-002", "decision": "reject", "edited": False, "question": make_draft(2)},
        {
            "id": "ag-001",
            "decision": "accept",
            "edited": False,
            "question": {
                **make_draft(1, qtype="agentic", origin="template"),
                "id": "ag-001",
                "expected_tool": "transpose",
                "expected_tool_args": {"chords": ["C"], "semitones": 2},
            },
        },
    ]
    write_jsonl(state, decisions)

    review.freeze(state, out, changelog, seed=42)

    questions = load_questions(out)
    assert [q.id for q in questions] == ["ag-001", "en-defi-001"]
    assert {q.origin for q in questions} == {"llm_draft_human_reviewed", "template_human_reviewed"}
    assert "evaluation set frozen" in changelog.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        review.freeze(state, out, changelog, seed=42)  # never overwrite a frozen set
