import json

from musicteacher.evaluation.judge_check import FABRICATED, check
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


def test_check_counts_detections_and_sends_broken_answers():
    # per row: fabricated answer (correctness + faithfulness calls), then swapped answer (same)
    client = FakeLLMClient(
        [
            correctness(1),
            claims("Bach 1950"),  # row a: fabrication caught
            correctness(0),
            claims(),  # row a: swap caught
            correctness(1),
            claims(),  # row b: fabrication missed
            correctness(1),
            claims(),  # row b: swap missed
        ]
    )
    result = check(client, ROWS, CONTEXTS)
    assert result["fabrication"] == {"caught": 1, "n": 2, "rate": 0.5}
    assert result["swapped_answer"] == {"caught": 1, "n": 2, "rate": 0.5}

    prompts = [c["messages"][0].content for c in client.calls]
    assert FABRICATED["en"] in prompts[0]
    assert "Answer: Ab" in prompts[2]  # question a judged with the answer of question b
    assert FABRICATED["fr"] in prompts[4]
