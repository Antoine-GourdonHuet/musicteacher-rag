"""Automatic pre-filter of the drafted QA questions, before the human review.

A draft is dropped if any check fails:
1. its question or answer refers to "the passage" / "the text" (a student would never say that);
2. it copies 6+ consecutive words from the passage (flagged at drafting time);
3. an LLM pass judges that the question cannot be answered from the passage alone, that the
   reference answer is not supported, or that the question is trivia about the text;
4. the LLM must quote the sentence that supports the answer, and the quote must really be in the
   passage (checked in code): the model has to show its evidence, not just say "yes".

The verdict is stored in each draft as `auto_filter` ({"kept": bool, "reasons": [...]}) and
`evaluation.review` only shows the kept drafts. Agentic questions are always kept (their answers are
computed). Limitation: the checking model is the same one that drafted the questions.

Usage: uv run python -m musicteacher.evaluation.auto_filter
"""

import json
import re

from musicteacher.config import get_settings
from musicteacher.evaluation.draft_questions import longest_shared_run, parse_json_object, words
from musicteacher.llm.base import LLMClient, Message

META_REFERENCE = re.compile(
    r"\b(passages?|the text|this text|le texte|ce texte|l'extrait|cet extrait|the excerpt)\b",
    re.IGNORECASE,
)

CHECK_SYSTEM = """You check evaluation questions for a music theory teaching assistant.
Given a passage, a question and a reference answer, decide:
- "answerable": true if the question can be fully answered using ONLY the passage.
- "supported": true if the reference answer is correct according to the passage.
- "evidence": copy, word for word, the ONE sentence of the passage that best supports the
  reference answer. The passage may be in another language than the question: copy the sentence
  in the passage's ORIGINAL language, never translate it. Empty string if there is none.
- "trivia": true if the question is about a specific person, performer, date, song, anecdote or
  quotation mentioned in the text, rather than about a music concept or technique.
  Trivia examples: "What did singer X alternate with bird calls?", "Which band used this technique
  in 1971?", "Who composed the operetta mentioned?".
  NOT trivia: definitions, rules, techniques, how or why something works, even when the passage
  answers them directly (that is expected).
Reply with JSON only, reason in at most 15 words:
{"answerable": true/false, "supported": true/false, "evidence": "...", "trivia": true/false,
"reason": "..."}"""

MIN_EVIDENCE_WORDS = 5
EVIDENCE_RUN_WORDS = 8


def evidence_found(evidence: str, passage: str) -> bool:
    """True if the quote really comes from the passage: at least 8 consecutive words of it (or
    the whole quote, if shorter, with a minimum of 5) appear in the passage. Tolerates small
    copying slips (punctuation, a changed word) but not paraphrases or translations."""
    n_words = len(words(evidence))
    if n_words < MIN_EVIDENCE_WORDS:
        return False
    return longest_shared_run(evidence, passage) >= min(EVIDENCE_RUN_WORDS, n_words)


def rule_reasons(draft: dict) -> list[str]:
    reasons = []
    if META_REFERENCE.search(draft["question"]) or META_REFERENCE.search(draft["reference_answer"]):
        reasons.append("mentions the passage/text")
    if (draft.get("review") or {}).get("copy_flag"):
        reasons.append("copies 6+ words from the passage")
    return reasons


def llm_reasons(client: LLMClient, draft: dict) -> list[str]:
    prompt = (
        f"Passage:\n{draft['source_passage']}\n\n"
        f"Question: {draft['question']}\n"
        f"Reference answer: {draft['reference_answer']}"
    )
    reply = client.complete(
        [Message("user", prompt)], system=CHECK_SYSTEM, temperature=0.0, max_tokens=400
    )
    verdict = parse_json_object(reply.text)
    if verdict is None:
        return ["checker gave no valid JSON"]
    reasons = []
    if verdict.get("answerable") is not True:
        reasons.append("not answerable from the passage")
    if verdict.get("supported") is not True:
        reasons.append("reference answer not supported")
    if verdict.get("trivia") is not False:
        reasons.append("trivia about the text, not a music concept")
    if not evidence_found(str(verdict.get("evidence", "")), draft["source_passage"]):
        reasons.append("no verbatim evidence quoted from the passage")
    if reasons and verdict.get("reason"):
        reasons.append(f"checker: {verdict['reason']}")
    return reasons


def filter_draft(client: LLMClient, draft: dict) -> dict:
    if draft["qtype"] == "agentic":
        return {"kept": True, "reasons": []}
    reasons = rule_reasons(draft)
    if not reasons:  # no need to spend an LLM call on a draft already rejected by the rules
        reasons = llm_reasons(client, draft)
    return {"kept": not reasons, "reasons": reasons}


def main() -> None:
    from musicteacher.llm.cache import CachedLLMClient
    from musicteacher.llm.ollama_client import OllamaClient

    settings = get_settings()
    path = settings.evals_dir / "drafts.jsonl"
    drafts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    inner = OllamaClient(settings.generator_model, settings.ollama_base_url, seed=settings.seed)
    client = CachedLLMClient(inner, settings.cache_dir / "llm")

    for i, draft in enumerate(drafts, start=1):
        draft["auto_filter"] = filter_draft(client, draft)
        verdict = "kept" if draft["auto_filter"]["kept"] else "DROPPED"
        print(f"[{i}/{len(drafts)}] {draft['id']}: {verdict} {draft['auto_filter']['reasons']}")
    client.close()

    path.write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in drafts), encoding="utf-8"
    )
    qa = [d for d in drafts if d["qtype"] != "agentic"]
    kept = [d for d in qa if d["auto_filter"]["kept"]]
    print(f"\nQA kept: {len(kept)}/{len(qa)} | agentic kept: {len(drafts) - len(qa)}")
    print(
        f"kept FR/EN: {sum(d['lang'] == 'fr' for d in kept)}/{sum(d['lang'] == 'en' for d in kept)}"
    )
    print(f"kept cross-lingual: {sum(d['cross_lingual'] for d in kept)}")
    print(client.usage.summary())


if __name__ == "__main__":
    main()
