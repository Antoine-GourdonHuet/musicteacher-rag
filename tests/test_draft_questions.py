import json
from collections import Counter

from musicteacher.evaluation.agentic_questions import agentic_questions
from musicteacher.evaluation.draft_questions import (
    Slot,
    draft,
    longest_shared_run,
    parse_json_reply,
    plan_slots,
    sample_passages,
)
from musicteacher.llm.fake import FakeLLMClient


def test_plan_has_80_balanced_slots():
    slots = plan_slots()
    assert len(slots) == 80
    pairs = Counter((s.question_lang, s.passage_lang) for s in slots)
    assert pairs == {("fr", "fr"): 34, ("en", "en"): 34, ("fr", "en"): 6, ("en", "fr"): 6}
    assert Counter(s.qtype for s in slots)["multihop"] == 10


def fake_records():
    records = []
    for lang in ("fr", "en"):
        for cat in ("chords", "guitar"):
            for article in range(6):
                for section in ("Intro", "History"):
                    records.append(
                        {
                            "chunk_id": f"{lang}-{cat}{article}#{section}",
                            "article_id": f"{lang}-{cat}{article}",
                            "language": lang,
                            "category": cat,
                            "section": section,
                            "title": f"{cat} {article}",
                            "text": "some text",
                            "n_tokens": 150,
                        }
                    )
    return records


def test_sampling_is_deterministic_uses_each_article_once_and_alternates_categories():
    slots = [Slot("fr", "fr", "definitional")] * 4 + [Slot("en", "en", "multihop")] * 2
    picks = sample_passages(slots, fake_records(), seed=42)
    assert picks == sample_passages(slots, fake_records(), seed=42)
    articles = [p[0]["article_id"] for p in picks]
    assert len(articles) == len(set(articles))
    assert [p[0]["category"] for p in picks[:4]] == ["chords", "guitar", "chords", "guitar"]
    for multihop in picks[4:]:
        assert len(multihop) == 2
        assert multihop[0]["section"] != multihop[1]["section"]


def test_sampling_skips_short_chunks():
    records = fake_records()
    for r in records:
        if r["category"] == "chords":
            r["n_tokens"] = 50
    picks = sample_passages([Slot("en", "en", "conceptual")] * 3, records, seed=1)
    assert {p[0]["category"] for p in picks} == {"guitar"}


def test_parse_json_reply_tolerates_extra_text():
    reply = 'Sure!\n{"question": "Q?", "reference_answer": "A."}\nHope it helps.'
    assert parse_json_reply(reply) == {"question": "Q?", "reference_answer": "A."}
    assert parse_json_reply("no json here") is None
    assert parse_json_reply('{"question": "", "reference_answer": "A."}') is None


def test_longest_shared_run_counts_consecutive_words():
    passage = "The Dorian mode has a raised sixth compared with the natural minor scale."
    assert longest_shared_run("What does the Dorian mode have?", passage) == 3
    assert longest_shared_run("Quelle est la sixte du mode dorien ?", passage) == 1  # "mode"


def test_draft_builds_a_reviewable_question():
    passage = {
        "chunk_id": "en-dorian_mode#004",
        "article_id": "en-dorian_mode",
        "category": "scales_modes",
        "title": "Dorian mode",
        "section": "Modern Dorian mode",
        "text": "The Dorian mode has a raised sixth compared with the natural minor scale.",
    }
    reply = json.dumps({"question": "En quoi le dorien diffère-t-il ?", "reference_answer": "R."})
    client = FakeLLMClient(replies=[reply])

    result = draft(client, Slot("fr", "en", "conceptual"), [passage], 7)

    assert result["id"] == "fr-conc-007"
    assert result["cross_lingual"] is True
    assert result["source_chunk_ids"] == ["en-dorian_mode#004"]
    assert result["origin"] == "llm_draft"
    assert result["review"] == {"copy_run_words": 0, "copy_flag": False}
    assert "French" in client.calls[0]["messages"][0].content


def test_agentic_questions_have_tools_and_computed_answers():
    questions = agentic_questions()
    assert len(questions) == 20
    assert Counter(q.expected_tool for q in questions) == {
        "transpose": 8,
        "interval": 6,
        "key_info": 6,
    }
    assert questions[0].reference_answer == "Eb - Cm - Ab - Bb"
    assert len({q.id for q in questions}) == 20
    assert Counter(q.lang for q in questions) == {"en": 10, "fr": 10}
