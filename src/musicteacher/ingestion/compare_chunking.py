"""Compare the two chunking strategies on the real corpus.

Prints, for each strategy: number of chunks, token-size distribution, share of chunks that start or
end mid-sentence, and the same example passage cut both ways. Writes the numbers to
`data/processed/chunking_comparison.json`.

Usage: uv run python -m musicteacher.ingestion.compare_chunking
"""

import json
import statistics

from musicteacher.config import get_settings
from musicteacher.ingestion.chunking import (
    Chunk,
    HFTokenizer,
    chunk_fixed,
    chunk_sections,
    load_raw_articles,
)
from musicteacher.ingestion.clean import split_sections

SENTENCE_ENDINGS = (".", "!", "?", "…", ":", ")", '"', "»")
EXAMPLE_ARTICLE = "en-dorian_mode"


def describe(chunks: list[Chunk]) -> dict:
    sizes = sorted(c.n_tokens for c in chunks)
    deciles = statistics.quantiles(sizes, n=10)
    texts = [c.text.strip() for c in chunks]
    return {
        "chunks": len(chunks),
        "chunks_en": sum(c.language == "en" for c in chunks),
        "chunks_fr": sum(c.language == "fr" for c in chunks),
        "tokens_total": sum(sizes),
        "tokens_min": sizes[0],
        "tokens_p10": round(deciles[0]),
        "tokens_median": round(statistics.median(sizes)),
        "tokens_p90": round(deciles[-1]),
        "tokens_max": sizes[-1],
        "under_64_tokens": sum(s < 64 for s in sizes),
        "starts_mid_sentence_pct": round(
            100 * sum(not t[:1].isupper() for t in texts) / len(texts), 1
        ),
        "ends_mid_sentence_pct": round(
            100 * sum(not t.endswith(SENTENCE_ENDINGS) for t in texts) / len(texts), 1
        ),
    }


def main() -> None:
    settings = get_settings()
    tok = HFTokenizer(settings.embedding_model)
    articles = load_raw_articles(settings.data_dir / "raw")

    results: dict[str, list[Chunk]] = {"fixed_300_overlap_50": [], "sections_max_300": []}
    for meta, text in articles:
        sections = split_sections(text, meta["language"])
        results["fixed_300_overlap_50"] += chunk_fixed(meta, sections, tok, size=300, overlap=50)
        results["sections_max_300"] += chunk_sections(meta, sections, tok, max_tokens=300)

    summary = {name: describe(chunks) for name, chunks in results.items()}
    print(f"Articles: {len(articles)}\n")
    keys = list(next(iter(summary.values())))
    print(f"{'':28}" + "".join(f"{name:>24}" for name in summary))
    for key in keys:
        print(f"{key:28}" + "".join(f"{summary[name][key]:>24}" for name in summary))

    for name, chunks in results.items():
        example = [c for c in chunks if c.article_id == EXAMPLE_ARTICLE][1]
        print(f"\n--- {name}: {example.chunk_id} [{example.section}] ({example.n_tokens} tokens)")
        print(example.text[:600] + (" [...]" if len(example.text) > 600 else ""))

    out = settings.data_dir / "processed" / "chunking_comparison.json"
    out.write_text(json.dumps({"articles": len(articles), **summary}, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
