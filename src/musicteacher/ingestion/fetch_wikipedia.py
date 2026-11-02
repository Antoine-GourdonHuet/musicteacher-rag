"""Fetch the corpus from Wikipedia (EN + FR) through the official MediaWiki API.

Reads the frozen topic list `data/topics.csv`, downloads each article as plain text (TextExtracts,
section headings kept as `== Heading ==`), saves it to `data/raw/{lang}/{id}.txt` with a metadata
JSON, then rebuilds `data/sources.csv` and prints real corpus statistics.

API etiquette: descriptive User-Agent, one request at a time, a pause between requests, retries with
backoff on rate limiting, and already-downloaded articles are skipped.

Usage: uv run python -m musicteacher.ingestion.fetch_wikipedia
"""

import csv
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from musicteacher.config import get_settings

LICENSE = "CC BY-SA 4.0"
SOURCES_FIELDS = [
    "id",
    "title",
    "url",
    "license",
    "language",
    "retrieved_at",
    "approx_tokens",
    "notes",
]


@dataclass(frozen=True)
class Article:
    lang: str
    title: str
    category: str

    @property
    def id(self) -> str:
        slug = re.sub(r"[^\w]+", "_", self.title.lower()).strip("_")
        return f"{self.lang}-{slug}"


def load_topics(path: Path) -> list[Article]:
    """One row per topic; the French column is empty when there is no French equivalent."""
    articles = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            articles.append(Article("en", row["title_en"], row["category"]))
            if row["title_fr"]:
                articles.append(Article("fr", row["title_fr"], row["category"]))
    ids = [a.id for a in articles]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate article ids in {path}: {sorted(duplicates)}")
    return articles


def estimate_tokens(text: str) -> int:
    """Rough estimate (~4 characters per token). Real token counts are measured at chunking time."""
    return len(text) // 4


def fetch_article(
    http: httpx.Client, article: Article, *, max_retries: int = 4, backoff_s: float = 5.0
) -> dict | None:
    """Return title, url, revision id and plain text, or None if the page does not exist."""
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "redirects": 1,
        "prop": "extracts|info",
        "explaintext": 1,
        "exsectionformat": "wiki",
        "inprop": "url",
        "titles": article.title,
    }
    url = f"https://{article.lang}.wikipedia.org/w/api.php"
    for attempt in range(max_retries + 1):
        try:
            response = http.get(url, params=params)
        except httpx.TransportError as exc:  # network drop, timeout, DNS failure...
            if attempt == max_retries:
                raise
            wait_s = backoff_s * 2**attempt
            print(f"  network error on {article.title!r} ({exc}), retrying in {wait_s:.0f}s")
            time.sleep(wait_s)
            continue
        if response.status_code in (429, 503) and attempt < max_retries:
            # Honour the server's Retry-After when given, otherwise back off exponentially.
            retry_after = response.headers.get("Retry-After", "")
            wait_s = float(retry_after) if retry_after.isdigit() else backoff_s * 2**attempt
            print(f"  HTTP {response.status_code} on {article.title!r}, retrying in {wait_s:.0f}s")
            time.sleep(wait_s)
            continue
        response.raise_for_status()
        break
    page = response.json()["query"]["pages"][0]
    if page.get("missing") or not page.get("extract"):
        return None
    return {
        "title": page["title"],
        "url": page["fullurl"],
        "revid": page.get("lastrevid"),
        "text": page["extract"],
    }


def save_article(raw_dir: Path, article: Article, fetched: dict, retrieved_at: str) -> None:
    lang_dir = raw_dir / article.lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    (lang_dir / f"{article.id}.txt").write_text(fetched["text"], encoding="utf-8")
    meta = {
        "id": article.id,
        "title": fetched["title"],
        "requested_title": article.title,
        "url": fetched["url"],
        "language": article.lang,
        "category": article.category,
        "revid": fetched["revid"],
        "retrieved_at": retrieved_at,
        "characters": len(fetched["text"]),
        "approx_tokens": estimate_tokens(fetched["text"]),
    }
    (lang_dir / f"{article.id}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_metadata(raw_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(raw_dir.glob("*/*.json"))]


def write_sources(path: Path, metadata: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SOURCES_FIELDS)
        writer.writeheader()
        for m in metadata:
            writer.writerow(
                {
                    "id": m["id"],
                    "title": m["title"],
                    "url": m["url"],
                    "license": LICENSE,
                    "language": m["language"],
                    "retrieved_at": m["retrieved_at"],
                    "approx_tokens": m["approx_tokens"],
                    "notes": f"revid={m['revid']}; category={m['category']}",
                }
            )


def corpus_stats(metadata: list[dict]) -> dict:
    stats: dict = {}
    for lang in sorted({m["language"] for m in metadata}):
        subset = [m for m in metadata if m["language"] == lang]
        stats[lang] = {
            "articles": len(subset),
            "characters": sum(m["characters"] for m in subset),
            "approx_tokens": sum(m["approx_tokens"] for m in subset),
        }
    stats["total"] = {
        key: sum(v[key] for v in stats.values())
        for key in ("articles", "characters", "approx_tokens")
    }
    return stats


def main() -> None:
    settings = get_settings()
    data_dir = settings.data_dir
    raw_dir = data_dir / "raw"
    articles = load_topics(data_dir / "topics.csv")

    missing = []
    headers = {"User-Agent": settings.wikipedia_user_agent}
    # Force IPv4: on some networks the IPv6 route to Wikimedia hangs until timeout.
    transport = httpx.HTTPTransport(local_address="0.0.0.0")
    with httpx.Client(headers=headers, timeout=60.0, transport=transport) as http:
        for i, article in enumerate(articles, start=1):
            if (raw_dir / article.lang / f"{article.id}.txt").exists():
                continue
            start = time.perf_counter()
            fetched = fetch_article(http, article)
            elapsed = time.perf_counter() - start
            if fetched is None:
                missing.append(f"{article.lang}:{article.title}")
                print(f"[{i}/{len(articles)}] MISSING {article.lang}:{article.title}")
            else:
                retrieved_at = datetime.now(UTC).isoformat(timespec="seconds")
                save_article(raw_dir, article, fetched, retrieved_at)
                print(f"[{i}/{len(articles)}] {article.lang}:{fetched['title']} ({elapsed:.1f}s)")
            time.sleep(settings.wikipedia_delay_s)

    metadata = load_metadata(raw_dir)
    write_sources(data_dir / "sources.csv", metadata)
    stats = corpus_stats(metadata)
    print(f"\nTopics requested: {len(articles)} | missing: {len(missing)} {missing}")
    for lang, s in stats.items():
        print(
            f"{lang:>5}: {s['articles']} articles, {s['characters']:,} characters, "
            f"~{s['approx_tokens']:,} tokens (estimated)"
        )


if __name__ == "__main__":
    main()
