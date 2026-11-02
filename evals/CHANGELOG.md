# Evaluation set changelog

Log every change to the question set made after the first system run (what, why, when).

## 2026-11-02: evaluation set frozen

- 100 drafts; 24 dropped by the automatic filter (evaluation/auto_filter.py).
- Remaining drafts decided by: ministral-3:8b: 56, unit-tested music tools: 20.
- 73 questions accepted (0 edited), 3 rejected at review.
- QA: 53, agentic: 20, cross-lingual: 8.
- Split (stratified by language x type, seed 42): dev 59, heldout 14.
- Per group: en/agentic 10, en/conceptual 10, en/definitional 8, en/multihop 4, en/terminology 7, fr/agentic 10, fr/conceptual 9, fr/definitional 6, fr/multihop 2, fr/terminology 7.
- The held-out split is never used for tuning.
