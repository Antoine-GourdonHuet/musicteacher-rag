import json

from musicteacher.evaluation.judge_check import FABRICATED, check, verbatim_answer
from musicteacher.llm.fake import FakeLLMClient


def correctness(score):
    return json.dumps({"correctness": score, "relevancy": 1, "reason": "r"})


def claims(*items):
    return json.dumps({"unsupported_claims": list(items), "faithful": not items})


ROWS = [
    {"id": "a", "lang": "en", "question": "Qa?", "reference_answer": "Ra", "answer": "Aa"},
    {"id": "b", "lang": "fr", "question": "Qb?", "reference_answer": "Rb", "answer": "Ab"},
]
CONTEXTS = {"a": "[1] ctx a", "b": "[1] ctx b"}
FIRST = {"a": "One. Two. Three.", "b": "Un. Deux."}


def test_verbatim_answer_takes_two_sentences_and_cites_them():
    assert verbatim_answer("A capo clamps.\nIt raises the pitch. Third one.") == (
        "A capo clamps. It raises the pitch. [1]"
    )


def test_check_measures_the_three_error_types():
    # per row: fabricated (faithfulness), verbatim (faithfulness), swapped (correctness)
    client = FakeLLMClient(
        [
            claims("Bach 1950"),  # a: fabrication caught
            claims(),  # a: verbatim accepted
            correctness(0),  # a: swap caught
            claims(),  # b: fabrication missed
            claims("too strict"),  # b: verbatim rejected (false alarm)
            correctness(0),  # b: swap caught
        ]
    )
    result = check(client, ROWS, CONTEXTS, FIRST)
    assert result["fabrication_caught"] == {"ok": 1, "n": 2, "rate": 0.5}
    assert result["verbatim_accepted"] == {"ok": 1, "n": 2, "rate": 0.5}
    assert result["swapped_answer_caught"] == {"ok": 2, "n": 2, "rate": 1.0}
    assert result["passes_threshold"] is False

    prompts = [c["messages"][0].content for c in client.calls]
    assert FABRICATED["en"] in prompts[0]
    assert "Answer: One. Two. [1]" in prompts[1]
    assert "Answer: Ab" in prompts[2]  # question a judged with the answer of question b
