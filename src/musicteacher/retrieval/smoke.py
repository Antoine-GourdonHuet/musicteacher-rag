"""Smoke test of the four retrieval modes on 10 hand-picked queries (5 FR, 5 EN, 2 cross-lingual).

Prints the top 3 of each mode per query, marks hits from the expected article with "*", and writes
the same report as Markdown to `data/processed/retrieval_smoke.md`. This is a sanity check, not an
evaluation: the measured comparison comes later on the reviewed question set.

Usage: uv run python -m musicteacher.retrieval.smoke
"""

import time

from musicteacher.config import get_settings
from musicteacher.retrieval.backends import load_retriever
from musicteacher.retrieval.search import MODES, Hit

# (query, expected article ids, note).
# Cross-lingual = French query whose topic only exists in English.
QUERIES = [
    ("Quelle note caractérise le mode dorien ?", {"fr-mode_dorien", "en-dorian_mode"}, "FR"),
    ("À quoi sert un capodastre sur une guitare ?", {"fr-capo", "en-capo"}, "FR"),
    ("Qu'est-ce que la voix de poitrine ?", {"fr-voix_de_poitrine", "en-chest_voice"}, "FR"),
    ("Qu'est-ce qu'un accord de neuvième ?", {"en-ninth_chord"}, "FR -> EN only"),
    (
        "Comment fonctionne la progression ii-V-I en jazz ?",
        {"en-ii_v_i_progression"},
        "FR -> EN only",
    ),
    (
        "What is a tritone substitution?",
        {"en-tritone_substitution", "fr-substitution_tritonique"},
        "EN",
    ),
    (
        "How does the circle of fifths relate keys?",
        {"en-circle_of_fifths", "fr-cycle_des_quintes"},
        "EN",
    ),
    ("Difference between a hammer-on and a pull-off", {"en-hammer_on", "en-pull_off"}, "EN"),
    ("What is a hemiola?", {"en-hemiola", "fr-hémiole"}, "EN"),
    ("What is the passaggio in singing?", {"en-passaggio", "fr-passaggio"}, "EN"),
]


def label(hit: Hit, expected: set[str]) -> str:
    article_id = hit.chunk_id.split("#")[0]
    mark = "*" if article_id in expected else " "
    section = hit.section.split(" > ")[-1]
    return f"{mark}{hit.title} › {section} ({hit.language})"[:60]


def main() -> None:
    retriever = load_retriever(get_settings())
    lines = ["# Retrieval smoke test (top 3 per mode, * = expected article)", ""]
    found = dict.fromkeys(MODES, 0)
    latency = dict.fromkeys(MODES, 0.0)

    for query, expected, note in QUERIES:
        results = {}
        for mode in MODES:
            start = time.perf_counter()
            results[mode] = retriever.search(query, mode, k=3)
            latency[mode] += time.perf_counter() - start
            found[mode] += any(h.chunk_id.split("#")[0] in expected for h in results[mode])
        lines += [f"## {query}  [{note}]", "", "| # | " + " | ".join(MODES) + " |"]
        lines.append("|---" * (len(MODES) + 1) + "|")
        for i in range(3):
            cells = [label(results[m][i], expected) if i < len(results[m]) else "" for m in MODES]
            lines.append(f"| {i + 1} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| mode | queries with expected article in top 3 | mean latency |")
    lines.append("|---|---|---|")
    for mode in MODES:
        mean_ms = 1000 * latency[mode] / len(QUERIES)
        lines.append(f"| {mode} | {found[mode]}/{len(QUERIES)} | {mean_ms:.0f} ms |")

    report = "\n".join(lines)
    print(report)
    out = get_settings().data_dir / "processed" / "retrieval_smoke.md"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
