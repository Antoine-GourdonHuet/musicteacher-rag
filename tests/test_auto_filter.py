import json

from musicteacher.evaluation.auto_filter import evidence_found, filter_draft, rule_reasons
from musicteacher.llm.fake import FakeLLMClient

DRAFT = {
    "id": "en-defi-001",
    "qtype": "definitional",
    "question": "What is a capo used for?",
    "reference_answer": "It raises the pitch of all strings.",
    "source_passage": "A capo clamps the strings to raise their pitch.",
    "review": {"copy_run_words": 2, "copy_flag": False},
}


EVIDENCE = "A capo clamps the strings to raise their pitch."


def verdict(**values) -> str:
    base = {
        "answerable": True,
        "supported": True,
        "trivia": False,
        "reason": "ok",
        "evidence": EVIDENCE,
    }
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


def test_answer_mentioning_the_text_is_caught():
    draft = {**DRAFT, "reference_answer": "The text describes it as a clamp."}
    assert rule_reasons(draft) == ["mentions the passage/text"]


def test_evidence_must_be_a_verbatim_quote_of_at_least_five_words():
    passage = "Le « palm mute » consiste à poser la main près du chevalet."
    assert evidence_found("le palm mute consiste à poser la main", passage)
    assert not evidence_found("poser la main sur la touche", passage)  # invented
    assert not evidence_found("palm mute", passage)  # too short to prove anything
    long_passage = "If two distinctly different themes are alternated indefinitely, then this "
    long_passage += "gives rise to a simple binary form."
    slip = "If two different themes are alternated indefinitely, then this gives rise to a form."
    assert evidence_found(slip, long_passage)  # small copying slip, 8+ words still match
    translated = "Si deux thèmes différents alternent indéfiniment, cela donne une forme binaire."
    assert not evidence_found(translated, long_passage)  # a translation is not evidence


def test_missing_or_invented_evidence_drops_the_draft():
    result = filter_draft(
        FakeLLMClient([verdict(evidence="A capo makes the guitar louder.")]), DRAFT
    )
    assert result["kept"] is False
    assert "no verbatim evidence quoted from the passage" in result["reasons"]
