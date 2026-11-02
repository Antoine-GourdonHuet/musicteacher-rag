import json

from musicteacher.evaluation.review import accepted_questions
from musicteacher.evaluation.verify import TOOLS_REVIEWER, verify_draft
from musicteacher.llm.fake import FakeLLMClient

QA = {
    "id": "en-defi-001",
    "lang": "en",
    "qtype": "definitional",
    "question": "What is a capo used for?",
    "reference_answer": "It raises the pitch of all strings.",
    "source_passage": "A capo clamps the strings to raise their pitch.",
    "origin": "llm_draft",
    "review": {"copy_run_words": 2, "copy_flag": False},
    "auto_filter": {"kept": True, "reasons": []},
}
AGENTIC = {
    "id": "ag-001",
    "lang": "en",
    "qtype": "agentic",
    "question": "Transpose C up two semitones.",
    "reference_answer": "D",
    "origin": "template",
    "expected_tool": "transpose",
    "expected_tool_args": {"chords": ["C"], "semitones": 2},
}
OK = json.dumps(
    {
        "answerable": True,
        "supported": True,
        "trivia": False,
        "reason": "fine",
        "evidence": "A capo clamps the strings to raise their pitch.",
    }
)
BAD = json.dumps({"answerable": True, "supported": False, "trivia": False, "reason": "wrong"})


def test_verifier_accepts_and_rejects_qa_with_its_own_origin():
    accepted = verify_draft(FakeLLMClient([OK]), "ministral-3:8b", QA)
    rejected = verify_draft(FakeLLMClient([BAD]), "ministral-3:8b", QA)
    assert accepted["decision"] == "accept"
    assert accepted["reviewer"] == "ministral-3:8b"
    assert accepted["question"]["origin"] == "llm_draft_llm_verified"
    assert rejected["decision"] == "reject"
    assert "reference answer not supported" in rejected["reasons"]


def test_agentic_questions_are_accepted_without_llm_call():
    client = FakeLLMClient()
    decision = verify_draft(client, "ministral-3:8b", AGENTIC)
    assert decision["decision"] == "accept"
    assert decision["reviewer"] == TOOLS_REVIEWER
    assert client.calls == []


def test_frozen_origins_are_not_relabelled_as_human_reviewed():
    decisions = {
        "en-defi-001": verify_draft(FakeLLMClient([OK]), "ministral-3:8b", QA),
        "ag-001": verify_draft(FakeLLMClient(), "ministral-3:8b", AGENTIC),
    }
    origins = {q.id: q.origin for q in accepted_questions(decisions)}
    assert origins == {"en-defi-001": "llm_draft_llm_verified", "ag-001": "template"}
