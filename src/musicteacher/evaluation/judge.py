"""LLM-as-a-judge for the generated answers (documented replacement for RAGAS).

Why not RAGAS: with a local model it needs LangChain wrappers, which this project avoids on purpose,
and its multi-step prompts are fragile with 8B models. This judge uses one explicit rubric, one
call per answer, temperature 0, JSON output, and a judge model from another family than the
generator (Mistral judges Qwen) to limit self-preference.

Scores are on a 3-level scale: 1 (yes), 0.5 (partly), 0 (no).
- correctness:  same key facts as the reference answer (works for every condition, C0 included)
- faithfulness: every claim of the answer is supported by the context passages (None for C0)
- relevancy:    the answer addresses the question that was asked (a refusal scores 0, as in
                RAGAS answer relevancy, which penalises noncommittal answers)
"""

from musicteacher.evaluation.draft_questions import parse_json_object
from musicteacher.llm.base import LLMClient, Message

LEVELS = (0.0, 0.5, 1.0)

JUDGE_SYSTEM = """You grade answers given by a music theory and guitar teaching assistant.
Score each criterion with 1 (yes), 0.5 (partly) or 0 (no):
- "correctness": the answer gives the same key facts as the reference answer. Extra correct
  details are fine. A refusal ("I don't know") or a contradiction scores 0.
- "faithfulness" (only if context passages are given): every factual claim of the answer is
  supported by the context passages. A refusal scores 1 (it claims nothing unsupported).
  Write null if no context is given.
- "relevancy": the answer addresses the question that was asked (not another question).
  A refusal or a noncommittal answer ("the passages do not say") scores 0.
The question and answer may be in French or English; judge the meaning, not the language.
Reply with JSON only, reason in at most 20 words:
{"correctness": 0/0.5/1, "faithfulness": 0/0.5/1/null, "relevancy": 0/0.5/1, "reason": "..."}"""


def _level(value: object) -> float | None:
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return score if score in LEVELS else None


def judge_answer(
    client: LLMClient, question: str, reference: str, answer: str, context: str | None
) -> dict:
    parts = []
    if context is not None:
        parts.append(f"Context passages:\n{context}")
    parts += [f"Question: {question}", f"Reference answer: {reference}", f"Answer: {answer}"]
    reply = client.complete(
        [Message("user", "\n\n".join(parts))],
        system=JUDGE_SYSTEM,
        temperature=0.0,
        max_tokens=300,
    )
    verdict = parse_json_object(reply.text) or {}
    scores = {
        "correctness": _level(verdict.get("correctness")),
        "faithfulness": _level(verdict.get("faithfulness")) if context is not None else None,
        "relevancy": _level(verdict.get("relevancy")),
    }
    required = ["correctness", "relevancy"] + (["faithfulness"] if context is not None else [])
    return {
        **scores,
        "judge_valid": all(scores[k] is not None for k in required),
        "judge_reason": str(verdict.get("reason", ""))[:300],
    }
