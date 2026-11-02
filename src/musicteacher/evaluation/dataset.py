"""Evaluation dataset schema and loader."""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel


class EvalQuestion(BaseModel):
    id: str
    lang: Literal["fr", "en"]
    qtype: Literal["definitional", "conceptual", "terminology", "multihop", "agentic"]
    question: str
    reference_answer: str
    source_doc: str | None = None  # id from data/sources.csv
    source_passage: str | None = None  # supporting text copied from the corpus
    source_chunk_ids: list[str] = []  # chunks of data/processed/chunks.jsonl (2 for multihop)
    category: str | None = None  # topic category from data/topics.csv
    cross_lingual: bool = False  # True if the supporting passage is in the other language
    split: Literal["dev", "heldout"] = "dev"
    origin: Literal[
        "manual", "llm_draft", "llm_draft_human_reviewed", "template", "template_human_reviewed"
    ] = "manual"
    expected_tool: str | None = None  # agentic questions only
    expected_tool_args: dict[str, Any] | None = None


def load_questions(path: Path) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(EvalQuestion.model_validate(json.loads(line)))
    return questions
