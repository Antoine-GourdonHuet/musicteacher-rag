"""Draft evaluation questions with the local LLM, to be reviewed one by one by a human.

1. Plan 80 slots: 34 FR + 34 EN same-language, 12 cross-lingual (6 FR questions on EN passages,
   6 EN questions on FR passages). Each slot has a target question type, so types stay balanced;
   ~10 slots are multihop and use two sections of the same article.
2. Sample passages (chunks >= 120 tokens) round-robin over topic categories, seed fixed, at most one
   question per article.
3. Ask the LLM for {"question", "reference_answer"} in JSON, paraphrasing the passage.
4. Measure copying: the longest run of words shared by the question and the passage. Runs of 6+
   words are flagged for the reviewer.
5. Append the 20 template-based agentic questions and write `evals/drafts.jsonl`.

Usage: uv run python -m musicteacher.evaluation.draft_questions
"""

import json
import random
import re
from dataclasses import dataclass

from musicteacher.config import get_settings
from musicteacher.evaluation.agentic_questions import agentic_questions
from musicteacher.evaluation.dataset import EvalQuestion
from musicteacher.llm.base import LLMClient, Message

MIN_CHUNK_TOKENS = 120
COPY_FLAG_WORDS = 6
SINGLE_TYPES = ["definitional", "conceptual", "terminology"]
LANGUAGE_NAMES = {"fr": "French", "en": "English"}

QTYPE_INSTRUCTIONS = {
    "definitional": "asks what a concept, technique or term is (a definition).",
    "conceptual": "asks why or how something works (an explanation, not a definition).",
    "terminology": (
        "asks for the name of a concept from its description, or the difference between two "
        "closely related terms."
    ),
    "multihop": (
        "can only be answered by combining information from BOTH passages "
        "(passage 1 and passage 2)."
    ),
}

SYSTEM_PROMPT = """You write evaluation questions for a music theory and guitar teaching assistant.
Rules:
- The question must be answerable using ONLY the passage(s) given. No outside knowledge needed.
- Paraphrase: do NOT copy phrases from the passage. Use your own words and a natural student tone.
- One precise idea per question. No yes/no questions. No question with several valid answers.
- Do not mention "the passage" or "the text" in the question.
- The reference answer is 1 to 3 sentences, faithful to the passage.
- Reply with JSON only: {"question": "...", "reference_answer": "..."}"""


@dataclass(frozen=True)
class Slot:
    question_lang: str
    passage_lang: str
    qtype: str


def plan_slots(same_language: int = 34, cross_each_way: int = 6, multihop_each: int = 5):
    """Slots in a fixed order: per language, `multihop_each` multihop + rotating single types."""
    slots = []
    for lang in ("fr", "en"):
        for i in range(same_language - multihop_each):
            slots.append(Slot(lang, lang, SINGLE_TYPES[i % 3]))
        slots += [Slot(lang, lang, "multihop")] * multihop_each
    for question_lang, passage_lang in (("fr", "en"), ("en", "fr")):
        for i in range(cross_each_way):
            slots.append(Slot(question_lang, passage_lang, SINGLE_TYPES[i % 3]))
    return slots


def sample_passages(slots: list[Slot], records: list[dict], seed: int) -> list[list[dict]]:
    """For each slot, 1 chunk (2 for multihop) in the passage language; categories round-robin."""
    rng = random.Random(seed)
    eligible = [r for r in records if r["n_tokens"] >= MIN_CHUNK_TOKENS]
    by_lang_cat: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for r in eligible:
        articles = by_lang_cat.setdefault((r["language"], r["category"]), {})
        articles.setdefault(r["article_id"], []).append(r)
    categories = sorted({r["category"] for r in eligible})
    used_articles: set[str] = set()
    next_category = {"fr": 0, "en": 0}
    picks = []
    for slot in slots:
        need_two = slot.qtype == "multihop"
        for _ in range(len(categories)):
            cat = categories[next_category[slot.passage_lang] % len(categories)]
            next_category[slot.passage_lang] += 1
            candidates = [
                chunks
                for article_id, chunks in sorted(
                    by_lang_cat.get((slot.passage_lang, cat), {}).items()
                )
                if article_id not in used_articles
                and (not need_two or len({c["section"] for c in chunks}) >= 2)
            ]
            if candidates:
                break
        else:
            raise ValueError(f"no passage left for {slot}")
        chunks = rng.choice(candidates)
        if need_two:
            first = rng.choice(chunks)
            second = rng.choice([c for c in chunks if c["section"] != first["section"]])
            chosen = sorted([first, second], key=lambda c: c["chunk_id"])
        else:
            chosen = [rng.choice(chunks)]
        used_articles.add(chosen[0]["article_id"])
        picks.append(chosen)
    return picks


def build_prompt(slot: Slot, passages: list[dict]) -> str:
    blocks = [
        f"Passage {i + 1} (from '{p['title']}', section '{p['section']}'):\n{p['text']}"
        for i, p in enumerate(passages)
    ]
    language = LANGUAGE_NAMES[slot.question_lang]
    return (
        "\n\n".join(blocks)
        + f"\n\nWrite ONE question that {QTYPE_INSTRUCTIONS[slot.qtype]}"
        + f"\nWrite the question AND the reference answer in {language}."
    )


def parse_json_object(text: str) -> dict | None:
    """The {...} object in an LLM reply, tolerating text around it."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_json_reply(text: str) -> dict | None:
    """Question and answer from a drafting reply, if both are non-empty."""
    data = parse_json_object(text)
    if data is None:
        return None
    question = str(data.get("question", "")).strip()
    answer = str(data.get("reference_answer", "")).strip()
    return {"question": question, "reference_answer": answer} if question and answer else None


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def longest_shared_run(a: str, b: str) -> int:
    """Length (in words) of the longest sequence of consecutive words found in both texts."""
    wa, wb = words(a), words(b)
    best = 0
    previous = [0] * (len(wb) + 1)
    for i in range(1, len(wa) + 1):
        current = [0] * (len(wb) + 1)
        for j in range(1, len(wb) + 1):
            if wa[i - 1] == wb[j - 1]:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def draft(client: LLMClient, slot: Slot, passages: list[dict], number: int) -> dict | None:
    reply = client.complete(
        [Message("user", build_prompt(slot, passages))],
        system=SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=400,
    )
    parsed = parse_json_reply(reply.text)
    if parsed is None:
        return None
    passage_text = "\n\n".join(p["text"] for p in passages)
    question = EvalQuestion(
        id=f"{slot.question_lang}-{slot.qtype[:4]}-{number:03d}",
        lang=slot.question_lang,
        qtype=slot.qtype,
        question=parsed["question"],
        reference_answer=parsed["reference_answer"],
        source_doc=passages[0]["article_id"],
        source_passage=passage_text,
        source_chunk_ids=[p["chunk_id"] for p in passages],
        category=passages[0]["category"],
        cross_lingual=slot.question_lang != slot.passage_lang,
        origin="llm_draft",
    )
    overlap = longest_shared_run(parsed["question"], passage_text)
    return {
        **question.model_dump(),
        "review": {"copy_run_words": overlap, "copy_flag": overlap >= COPY_FLAG_WORDS},
    }


def main() -> None:
    from musicteacher.llm.cache import CachedLLMClient
    from musicteacher.llm.ollama_client import OllamaClient
    from musicteacher.retrieval.backends import load_records

    settings = get_settings()
    records = load_records(settings.data_dir / "processed" / "chunks.jsonl")
    slots = plan_slots()
    picks = sample_passages(slots, records, seed=settings.seed)
    inner = OllamaClient(settings.generator_model, settings.ollama_base_url, seed=settings.seed)
    client = CachedLLMClient(inner, settings.cache_dir / "llm")

    drafts, failures = [], []
    for number, (slot, passages) in enumerate(zip(slots, picks, strict=True), start=1):
        result = draft(client, slot, passages, number)
        status = "ok" if result else "FAILED (no valid JSON)"
        langs = f"{slot.question_lang}<-{slot.passage_lang}"
        print(f"[{number}/{len(slots)}] {langs} {slot.qtype}: {status}")
        if result:
            drafts.append(result)
        else:
            failures.append(number)
    client.close()

    drafts += [{**q.model_dump(), "review": {}} for q in agentic_questions()]
    out = settings.evals_dir / "drafts.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for d in drafts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    qa = [d for d in drafts if d["qtype"] != "agentic"]
    print(f"\nWrote {len(drafts)} drafts to {out}")
    print(f"QA: {len(qa)} (failed: {failures}) | agentic: {len(drafts) - len(qa)}")
    print(f"cross-lingual: {sum(d['cross_lingual'] for d in qa)}")
    flagged = sum(d["review"]["copy_flag"] for d in qa)
    print(f"flagged for copying (>= {COPY_FLAG_WORDS} words): {flagged}")
    print(client.usage.summary())


if __name__ == "__main__":
    main()
