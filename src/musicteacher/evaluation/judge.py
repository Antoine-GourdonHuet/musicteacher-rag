"""LLM-as-a-judge for the generated answers (documented replacement for RAGAS).

Why not RAGAS: with a local model it needs LangChain wrappers, which this project avoids on purpose,
and its multi-step prompts are fragile with 8B models. This judge uses explicit rubrics,
temperature 0, JSON output, and a judge model from another family than the generator (Mistral
judges Qwen) to limit self-preference.

Two separate calls per answer:
1. correctness and relevancy (0 / 0.5 / 1), against the reference answer — every condition;
2. faithfulness, only when context passages were given: the judge must LIST the claims of the
   answer that the passages do not support; faithful (1) only if that list is empty, else 0.
   A first version asked for all three scores in one call: on answers with a planted false claim,
   the judge often returned `null` for faithfulness instead of 0, so unfaithful answers silently
   dropped out of the average. Forcing a list of unsupported claims removed that escape hatch.
"""

from musicteacher.evaluation.draft_questions import parse_json_object
from musicteacher.llm.base import LLMClient, Message

LEVELS = (0.0, 0.5, 1.0)

CORRECTNESS_SYSTEM = """You grade answers given by a music theory and guitar teaching assistant.
Score each criterion with 1 (yes), 0.5 (partly) or 0 (no):
- "correctness": the answer gives the same key facts as the reference answer. Extra correct
  details are fine. A refusal ("I don't know") or a contradiction scores 0.
- "relevancy": the answer addresses the question that was asked (not another question).
  A refusal or a noncommittal answer ("the passages do not say") scores 0.
The question and answer may be in French or English; judge the meaning, not the language.
Reply with JSON only, reason in at most 20 words:
{"correctness": 0/0.5/1, "relevancy": 0/0.5/1, "reason": "..."}"""

FAITHFULNESS_SYSTEM = """You check whether an answer is faithful to the context passages.
List every factual claim of the answer that is NOT supported by the context passages.
Citation markers like [1] are not claims. A refusal ("the passages do not say") makes no claim.
The passages and the answer may be in different languages; compare the meaning.
Reply with JSON only:
{"unsupported_claims": ["claim 1", "..."], "faithful": true/false}
"faithful" is true only if "unsupported_claims" is empty."""


def _level(value: object) -> float | None:
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return score if score in LEVELS else None


def judge_correctness(client: LLMClient, question: str, reference: str, answer: str) -> dict:
    content = f"Question: {question}\n\nReference answer: {reference}\n\nAnswer: {answer}"
    reply = client.complete(
        [Message("user", content)], system=CORRECTNESS_SYSTEM, temperature=0.0, max_tokens=300
    )
    verdict = parse_json_object(reply.text) or {}
    return {
        "correctness": _level(verdict.get("correctness")),
        "relevancy": _level(verdict.get("relevancy")),
        "judge_reason": str(verdict.get("reason", ""))[:300],
    }


def judge_faithfulness(client: LLMClient, answer: str, context: str) -> dict:
    content = f"Context passages:\n{context}\n\nAnswer: {answer}"
    reply = client.complete(
        [Message("user", content)], system=FAITHFULNESS_SYSTEM, temperature=0.0, max_tokens=400
    )
    verdict = parse_json_object(reply.text)
    claims = verdict.get("unsupported_claims") if verdict else None
    if not isinstance(claims, list):
        return {"faithfulness": None, "unsupported_claims": []}
    claims = [str(c) for c in claims if str(c).strip()]
    # The list decides: a non-empty list means unfaithful, whatever the "faithful" flag says.
    return {"faithfulness": 0.0 if claims else 1.0, "unsupported_claims": claims}


def judge_answer(
    client: LLMClient, question: str, reference: str, answer: str, context: str | None
) -> dict:
    result = judge_correctness(client, question, reference, answer)
    if context is not None:
        result.update(judge_faithfulness(client, answer, context))
    else:
        result.update({"faithfulness": None, "unsupported_claims": []})
    required = ["correctness", "relevancy"] + (["faithfulness"] if context is not None else [])
    result["judge_valid"] = all(result[k] is not None for k in required)
    return result
