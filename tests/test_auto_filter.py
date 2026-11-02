import json

from musicteacher.evaluation.auto_filter import filter_draft, rule_reasons
from musicteacher.llm.fake import FakeLLMClient

DRAFT = {
    "id": "en-defi-001",
    "qtype": "definitional",
    "question": "What is a capo used for?",
    "reference_answer": "It raises the pitch of all strings.",
    "source_passage": "A capo clamps the strings to raise their pitch.",
    "review": {"copy_run_words": 2, "copy_flag": False},
}


def verdict(**values) -> str:
    base = {"answerable": True, "supported": True, "trivia": False, "reason": "ok"}
    return json.dumps({**base, **values})


def test_rules_catch_meta_references_and_copies():
    assert rule_reasons({**DRAFT, "question": "According to the passage, what is a capo?"}) == [
        "mentions the passage/text"
    ]
    assert rule_reasons({**DRAFT, "question": "Que dit le texte sur le capo ?"}) == [
        "mentions the passage/text"
    ]
    assert rule_reasons({**DRAFT, "review": {"copy_flag": True}}) == [
        "copies 6+ words from the passage"
    ]
    assert rule_reasons(DRAFT) == []


def test_good_draft_is_kept():
    assert filter_draft(FakeLLMClient([verdict()]), DRAFT) == {"kept": True, "reasons": []}


def test_checker_failures_drop_the_draft():
    client = FakeLLMClient([verdict(supported=False, reason="answer adds facts")])
    result = filter_draft(client, DRAFT)
    assert result["kept"] is False
    assert "reference answer not supported" in result["reasons"]
    assert "checker: answer adds facts" in result["reasons"]


def test_invalid_checker_reply_drops_the_draft():
    result = filter_draft(FakeLLMClient(["no idea"]), DRAFT)
    assert result == {"kept": False, "reasons": ["checker gave no valid JSON"]}


def test_rule_failure_skips_the_llm_and_agentic_is_always_kept():
    client = FakeLLMClient()
    filter_draft(client, {**DRAFT, "question": "In this text, what is a capo?"})
    filter_draft(client, {**DRAFT, "qtype": "agentic"})
    assert client.calls == []
