"""Terminal tool to review the drafted questions one by one, then freeze the evaluation set.

Review (resumable: decisions are saved after every answer in `evals/review_state.jsonl`):
    uv run python -m musicteacher.evaluation.review
Freeze (accepted questions -> `evals/questions.jsonl` with a stratified 20% held-out split):
    uv run python -m musicteacher.evaluation.review --freeze
"""

import argparse
import json
import random
import textwrap
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from musicteacher.config import get_settings
from musicteacher.evaluation.dataset import EvalQuestion

TARGET_QA = 50
TARGET_AGENTIC = 12
HELDOUT_FRACTION = 0.2
REVIEWED_ORIGIN = {"llm_draft": "llm_draft_human_reviewed", "template": "template_human_reviewed"}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def latest_decisions(state: list[dict]) -> dict[str, dict]:
    """Last decision per draft id (a later decision overrides an earlier one)."""
    return {d["id"]: d for d in state}


def counts(decisions: dict[str, dict]) -> tuple[int, int]:
    accepted = [d for d in decisions.values() if d["decision"] == "accept"]
    agentic = sum(d["question"]["qtype"] == "agentic" for d in accepted)
    return len(accepted) - agentic, agentic


def show(draft: dict, position: str, accepted_qa: int, accepted_agentic: int) -> None:
    wrap = textwrap.TextWrapper(width=100, replace_whitespace=False)
    print("\n" + "=" * 100)
    print(
        f"{position}  |  accepted QA {accepted_qa}/{TARGET_QA}  "
        f"agentic {accepted_agentic}/{TARGET_AGENTIC}"
    )
    flags = [draft["lang"].upper(), draft["qtype"]]
    if draft.get("cross_lingual"):
        flags.append("CROSS-LINGUAL")
    if draft.get("category"):
        flags.append(draft["category"])
    print(f"{draft['id']}  [{' | '.join(flags)}]")
    review = draft.get("review") or {}
    if review.get("copy_flag"):
        print(
            f"!! COPY WARNING: {review['copy_run_words']} consecutive words taken from the passage"
        )
    if draft.get("source_passage"):
        print("-" * 100 + "\nPASSAGE:")
        for paragraph in draft["source_passage"].split("\n"):
            print(wrap.fill(paragraph))
    if draft.get("expected_tool"):
        print("-" * 100)
        print(f"TOOL: {draft['expected_tool']}  ARGS: {json.dumps(draft['expected_tool_args'])}")
    print("-" * 100)
    print(f"QUESTION: {draft['question']}")
    print(f"ANSWER:   {draft['reference_answer']}")


def ask_edit(draft: dict) -> dict:
    print("Press Enter to keep the current text.")
    question = input("New question: ").strip() or draft["question"]
    answer = input("New answer:   ").strip() or draft["reference_answer"]
    return {**draft, "question": question, "reference_answer": answer}


def review(drafts_path: Path, state_path: Path) -> None:
    all_drafts = read_jsonl(drafts_path)
    drafts = [d for d in all_drafts if (d.get("auto_filter") or {}).get("kept", True)]
    decisions = latest_decisions(read_jsonl(state_path))
    pending = [d for d in drafts if d["id"] not in decisions]
    print(
        f"{len(all_drafts)} drafts, {len(all_drafts) - len(drafts)} dropped by the automatic "
        f"filter, {len(decisions)} already decided, {len(pending)} to review."
    )
    print("Commands: [a]ccept  [e]dit then accept  [r]eject  [s]kip  [q]uit (progress is saved)")

    for i, draft in enumerate(pending, start=1):
        qa, agentic = counts(decisions)
        show(draft, f"{i}/{len(pending)}", qa, agentic)
        while True:
            choice = input("> ").strip().lower()
            if choice in {"a", "e", "r", "s", "q"}:
                break
            print("Type a, e, r, s or q.")
        if choice == "q":
            break
        if choice == "s":
            continue
        question = ask_edit(draft) if choice == "e" else draft
        decision = {
            "id": draft["id"],
            "decision": "reject" if choice == "r" else "accept",
            "edited": choice == "e",
            "question": question,
            "decided_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        append_jsonl(state_path, decision)
        decisions[draft["id"]] = decision

    qa, agentic = counts(decisions)
    print(f"\nAccepted so far: {qa} QA, {agentic} agentic.")


def assign_splits(questions: list[EvalQuestion], seed: int) -> list[EvalQuestion]:
    """20% held out within each (language, question type) group, so both splits stay balanced."""
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[EvalQuestion]] = {}
    for q in questions:
        groups.setdefault((q.lang, q.qtype), []).append(q)
    heldout: set[str] = set()
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda q: q.id)
        n_heldout = round(len(members) * HELDOUT_FRACTION)
        heldout |= {q.id for q in rng.sample(members, n_heldout)}
    return [
        q.model_copy(update={"split": "heldout" if q.id in heldout else "dev"}) for q in questions
    ]


def accepted_questions(decisions: dict[str, dict]) -> list[EvalQuestion]:
    questions = []
    for d in decisions.values():
        if d["decision"] != "accept":
            continue
        data = {k: v for k, v in d["question"].items() if k != "review"}
        data["origin"] = REVIEWED_ORIGIN.get(data["origin"], data["origin"])
        questions.append(EvalQuestion.model_validate(data))
    return sorted(questions, key=lambda q: q.id)


def freeze(
    state_path: Path,
    out_path: Path,
    changelog: Path,
    seed: int,
    drafts_path: Path | None = None,
) -> list[EvalQuestion]:
    if out_path.exists():
        raise SystemExit(
            f"{out_path} already exists: the set is frozen. Log changes in CHANGELOG.md."
        )
    decisions = latest_decisions(read_jsonl(state_path))
    questions = assign_splits(accepted_questions(decisions), seed)
    with out_path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(q.model_dump_json() + "\n")

    split_counts = Counter(q.split for q in questions)
    by_group = Counter((q.lang, q.qtype) for q in questions)
    rejected = sum(d["decision"] == "reject" for d in decisions.values())
    drafts = read_jsonl(drafts_path) if drafts_path else []
    auto_dropped = sum(not (d.get("auto_filter") or {}).get("kept", True) for d in drafts)
    edited = sum(d["edited"] for d in decisions.values() if d["decision"] == "accept")
    groups_text = ", ".join(f"{lang}/{qtype} {n}" for (lang, qtype), n in sorted(by_group.items()))
    date = datetime.now(UTC).date().isoformat()
    entry = (
        f"\n## {date}: evaluation set frozen\n\n"
        f"- {len(drafts) or 'n/a'} drafts; {auto_dropped} dropped by the automatic filter "
        "(evaluation/auto_filter.py); the rest reviewed one by one by a human.\n"
        f"- {len(questions)} questions accepted ({edited} edited), "
        f"{rejected} rejected by the reviewer.\n"
        f"- QA: {sum(q.qtype != 'agentic' for q in questions)}, "
        f"agentic: {sum(q.qtype == 'agentic' for q in questions)}, "
        f"cross-lingual: {sum(q.cross_lingual for q in questions)}.\n"
        f"- Split (stratified by language x type, seed {seed}): "
        f"dev {split_counts['dev']}, heldout {split_counts['heldout']}.\n"
        f"- Per group: {groups_text}.\n"
        "- The held-out split is never used for tuning.\n"
    )
    with changelog.open("a", encoding="utf-8") as f:
        f.write(entry)
    print(entry)
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--freeze", action="store_true", help="write evals/questions.jsonl")
    args = parser.parse_args()
    settings = get_settings()
    evals = settings.evals_dir
    if args.freeze:
        freeze(
            evals / "review_state.jsonl",
            evals / "questions.jsonl",
            evals / "CHANGELOG.md",
            settings.seed,
            drafts_path=evals / "drafts.jsonl",
        )
    else:
        review(evals / "drafts.jsonl", evals / "review_state.jsonl")


if __name__ == "__main__":
    main()
