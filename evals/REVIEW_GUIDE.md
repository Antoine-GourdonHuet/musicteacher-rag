# Reviewing the LLM-drafted questions

You are the quality filter. Be strict: 50 good questions beat 80 mediocre ones.

For each draft, check:

1. **Answerable from the passage shown.** If you must guess, reject.
2. **Correct.** Does the reference answer match the passage AND what you know about music theory? If not, edit or reject.
3. **One idea per question** (multihop questions need two passages).
4. **Not a copy of the passage's wording.** If the question reuses long phrases, reword it.
5. **Not trivial or ambiguous.** Reject yes/no questions and questions with several valid answers.
6. **Language and cross-lingual flag are right.**
7. **Agentic questions:** the expected tool and arguments are right. Convention: use flats (Eb, not D#).

Targets after review: ~50 QA questions (about 25 FR / 25 EN, ~10 cross-lingual) and ~12 agentic.
Never change a question after seeing system outputs without logging it in `evals/CHANGELOG.md`.
