"""Independent LLM verification of the drafted questions (replaces the human review).

The drafts were written by the generator model (Qwen). A second model from another family
(Mistral, `judge_model`) checks each draft kept by `auto_filter`: answerable from the passage,
reference answer supported by the passage, not trivia. Only drafts accepted by both the automatic
filter and this verifier are kept. Agentic questions are accepted as is: their answers are computed
by the unit-tested music tools.

Decisions are written to `evals/review_state.jsonl` (same format as the human review tool), then
`python -m musicteacher.evaluation.review --freeze` builds the final set.

Usage: uv run python -m musicteacher.evaluation.verify
"""

from datetime import UTC, datetime

from musicteacher.config import get_settings
from musicteacher.evaluation.auto_filter import llm_reasons
from musicteacher.evaluation.review import read_jsonl
from musicteacher.llm.base import LLMClient

TOOLS_REVIEWER = "unit-tested music tools"


def verify_draft(client: LLMClient, verifier: str, draft: dict) -> dict:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    if draft["qtype"] == "agentic":
        return {
            "id": draft["id"],
            "decision": "accept",
            "edited": False,
            "reviewer": TOOLS_REVIEWER,
            "reasons": [],
            "question": draft,
            "decided_at": now,
        }
    reasons = llm_reasons(client, draft)
    return {
        "id": draft["id"],
        "decision": "reject" if reasons else "accept",
        "edited": False,
        "reviewer": verifier,
        "reasons": reasons,
        "question": {**draft, "origin": "llm_draft_llm_verified"},
        "decided_at": now,
    }


def main() -> None:
    import json

    from musicteacher.llm.cache import CachedLLMClient
    from musicteacher.llm.ollama_client import OllamaClient

    settings = get_settings()
    drafts = read_jsonl(settings.evals_dir / "drafts.jsonl")
    kept = [d for d in drafts if (d.get("auto_filter") or {}).get("kept", True)]
    # Mistral models do not support the `think` field: do not send it.
    inner = OllamaClient(
        settings.judge_model, settings.ollama_base_url, seed=settings.seed, think=None
    )
    client = CachedLLMClient(inner, settings.cache_dir / "llm")

    decisions = []
    for i, draft in enumerate(kept, start=1):
        decision = verify_draft(client, settings.judge_model, draft)
        decisions.append(decision)
        print(f"[{i}/{len(kept)}] {draft['id']}: {decision['decision']} {decision['reasons']}")
    client.close()

    state = settings.evals_dir / "review_state.jsonl"
    state.write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in decisions), encoding="utf-8"
    )
    accepted = [d for d in decisions if d["decision"] == "accept"]
    qa = [d for d in accepted if d["question"]["qtype"] != "agentic"]
    print(f"\nKept by auto_filter: {len(kept)}/{len(drafts)}")
    print(f"Accepted: {len(qa)} QA + {len(accepted) - len(qa)} agentic")
    print(
        f"QA FR/EN: {sum(d['question']['lang'] == 'fr' for d in qa)}/"
        f"{sum(d['question']['lang'] == 'en' for d in qa)}, "
        f"cross-lingual: {sum(d['question']['cross_lingual'] for d in qa)}"
    )
    print(client.usage.summary())


if __name__ == "__main__":
    main()
