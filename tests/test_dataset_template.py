from musicteacher.config import ROOT
from musicteacher.evaluation.dataset import load_questions


def test_template_is_valid_and_ids_are_unique():
    questions = load_questions(ROOT / "evals" / "questions_template.jsonl")
    assert len(questions) >= 3
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids))


def test_agentic_questions_declare_a_tool():
    questions = load_questions(ROOT / "evals" / "questions_template.jsonl")
    for q in questions:
        if q.qtype == "agentic":
            assert q.expected_tool is not None
