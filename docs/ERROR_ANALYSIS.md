# Error analysis — best condition (C4: hybrid retrieval + reranker)

Source: `evals/results/C4_all_20261102T150914.json` (73 questions). A *failure* is any answer with
correctness < 1 or faithfulness = 0: **22 failures** (13 QA, 9 agentic). Fifteen are analysed below:
all 13 QA failures and 2 representative agentic ones (the 7 other agentic failures are identical refusals).

Each case was read against its question, reference answer, retrieved passages and judge verdict.
Categories: retrieval miss · wrong/incomplete use of good context · hallucination despite good context ·
language issue · bad question · judge error · computation without a tool.
No domain-expert annotation was available: categories reflect a careful reading, not a music-theory review.

## The 15 cases

| # | Question | Recall@5 | Correct. | Faith. | Category | What happened |
|---|---|---|---|---|---|---|
| 1 | en-conc-076 (cross-lingual) | 0 | 0 | 0 | **Retrieval miss (cross-lingual)** | Source is French; only English chunks retrieved. The model correctly refuses. Faithfulness 0 is a judge false alarm (a refusal makes no claim). |
| 2 | fr-conc-070 (cross-lingual) | 0 | 0.5 | 1 | **Retrieval miss (cross-lingual)** | Source is English; French chunks retrieved instead. Plausible partial answer from them. |
| 3 | fr-conc-073 (cross-lingual) | 0 | 0 | 0 | **Retrieval miss (cross-lingual)** | Source is English; French chord-notation chunks retrieved. Correct refusal; faithfulness 0 is a false alarm. |
| 4 | en-mult-066 (multihop) | 0.5 | 1 | 0 | **Retrieval miss (partial)** | 1 of the 2 source chunks retrieved; the model answers the first half and says the passages lack the second. Faithfulness 0 is a false alarm (the flagged claim is in passage [1]). |
| 5 | en-term-077 (cross-lingual) | 0 | 0.5 | 1 | **Language issue** | French source missed, but an English passage gives the right term ("arpeggiator"). The reference uses the French word; the judge scores 0.5. |
| 6 | en-defi-053 | 1 | 1 | 0 | **Hallucination despite good context** | Correct core answer, then a garbled extra sentence mixing bass and guitar scale lengths. Justified flag. |
| 7 | en-defi-059 | 1 | 1 | 0 | **Hallucination despite good context** | Adds "this facilitates substitutions and modulations in jazz", an extrapolation beyond the passages. Justified flag. |
| 8 | en-defi-050 | 1 | 1 | 0 | **Hallucination (minor)** | Repeats the question's own wording ("tight vocal arrangements"), which the passage does not state. Borderline. |
| 9 | en-conc-036 | 1 | 0 | 1 | **Incomplete use of good context** | The right chunk is ranked 1st; the answer covers the 7th but omits the 9th/11th/13th extensions. The judge also wrongly calls "major-minor seventh" a mislabel (it is a valid synonym). |
| 10 | en-mult-064 (multihop) | 1 | 0 | 0 | **Bad question** | Both source chunks retrieved, but the question ("relationship between tempo and duration" for metrical vs free rhythm) is convoluted; the model refuses. Faithfulness 0 is a false alarm. |
| 11 | en-term-046 | 1 | 1 | 0 | **Judge error** | Correct ("plagal cadence"), stated verbatim in passage [1]; the judge flags it as unsupported. |
| 12 | fr-defi-007 | 1 | 1 | 0 | **Judge error** | Correct definition of *degré*; the flagged detail (numbering from 1 at the tonic) is in the passage. |
| 13 | fr-defi-010 | 1 | 0.5 | 1 | **Judge error** | Correct answer plus a correct detail from passage [3] (open vs filled note heads); the judge calls it wrong. |
| 14 | ag-002 (agentic) | – | 0 | – | **Computation without a tool** | Transposes G-Em-C-D down a tone but writes Em → Cm instead of Dm. |
| 15 | ag-011 (agentic) | – | 0 | – | **Computation without a tool** | "The passages do not contain the interval between A and Eb": refuses instead of computing. 7 other agentic failures are the same refusal. |

## Summary

| Category | Cases (of 15) |
|---|---|
| Retrieval miss | 4 (3 cross-lingual, 1 partial multihop) |
| Language issue | 1 |
| Hallucination despite good context | 3 (1 minor) |
| Incomplete use of good context | 1 |
| Bad question | 1 |
| Judge error | 3 |
| Computation without a tool | 2 (representing 9 agentic failures) |

**Findings**

1. **Cross-lingual retrieval is the main retrieval weakness.** All 4 cross-lingual QA failures come from a
   miss: when the source passage is in the other language, the retriever tends to return passages in the
   question's language. This matches the low cross-lingual recall@5 of C4 (0.50 on 8 questions vs 0.89 overall).
2. **Hallucinations are rare and small.** When the right passage is retrieved, the generator stays close to it;
   the 3 unfaithful answers add one extra sentence beyond the passages, never a wrong core answer.
3. **The generator prefers refusing to guessing.** 4 QA failures and 8 agentic failures are refusals. This keeps
   faithfulness high but costs correctness, and it is exactly why computation questions need tools (C5).
4. **The judge is the largest single source of error in this sample.** Of the 9 answers judged unfaithful in C4,
   3 flags are justified and 6 are false alarms (3 refusals, 3 supported claims). The reported faithfulness of C4
   (0.83) is therefore likely an underestimate; correctness has at least 2 harsh verdicts (cases 5 and 13).
   The judge passed controlled tests (20/20 planted fabrications caught, 20/20 verbatim answers accepted,
   20/20 swapped answers caught: `evals/results/judge_check.json`), but real answers are subtler than those tests.

**Judge agreement.** Agreement between the judge and a human annotator was not measured (no domain-expert
reviewer). On the 13 QA failures above (26 scores: correctness and faithfulness), the judge's verdict
disagrees with our reading in 8 (6 faithfulness false alarms, 2 harsh correctness scores). This sample only covers failures, so it says
nothing about false negatives (bad answers the judge accepted).
