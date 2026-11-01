# MusicTeacher (RAG)

Bilingual (FR/EN) agentic RAG assistant for music theory education, built as a controlled experiment:
**does hybrid retrieval + reranking improve answer faithfulness?**

> Status: work in progress.

## Quick start

```powershell
uv sync --group dev
copy .env.example .env      # then fill MUSICTEACHER_ANTHROPIC_API_KEY
uv run pytest -q
```

## Results

_Generated from `evals/results/` once the experiments are run. No number is reported before it is measured._

## Architecture

_Diagram to come._

## Limitations

_To be completed (LLM-drafted questions with human review, small sample, Wikipedia-only corpus,
judge bias, single annotator)._
