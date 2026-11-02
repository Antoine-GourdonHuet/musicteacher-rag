import csv

import httpx
import pytest

from musicteacher.ingestion.fetch_wikipedia import (
    Article,
    corpus_stats,
    fetch_article,
    load_metadata,
    load_topics,
    save_article,
    write_sources,
)


def write_topics(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "title_en", "title_fr"])
        writer.writeheader()
        writer.writerows(rows)


def test_load_topics_adds_french_only_when_present(tmp_path):
    path = tmp_path / "topics.csv"
    write_topics(
        path,
        [
            {"category": "scales_modes", "title_en": "Dorian mode", "title_fr": "Mode dorien"},
            {"category": "chords", "title_en": "Slash chord", "title_fr": ""},
        ],
    )
    articles = load_topics(path)
    assert [(a.lang, a.title) for a in articles] == [
        ("en", "Dorian mode"),
        ("fr", "Mode dorien"),
        ("en", "Slash chord"),
    ]


def test_load_topics_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "topics.csv"
    row = {"category": "chords", "title_en": "Power chord", "title_fr": ""}
    write_topics(path, [row, row])
    with pytest.raises(ValueError, match="duplicate"):
        load_topics(path)


def test_article_id_is_a_safe_filename():
    assert (
        Article("fr", "Accord de septième d'espèce", "chords").id
        == "fr-accord_de_septième_d_espèce"
    )
    assert Article("en", "Ii–V–I progression", "harmony").id == "en-ii_v_i_progression"


def fake_wikipedia(pages: dict[str, dict], calls: list):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        title = request.url.params["titles"]
        page = pages.get(title, {"title": title, "missing": True})
        return httpx.Response(200, json={"query": {"pages": [page]}})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_article_uses_plain_text_extracts():
    calls: list = []
    page = {
        "title": "Mode dorien",
        "fullurl": "https://fr.wikipedia.org/wiki/Mode_dorien",
        "lastrevid": 123,
        "extract": "Le mode de ré...\n\n== Histoire ==\n...",
    }
    http = fake_wikipedia({"Mode dorien": page}, calls)

    fetched = fetch_article(http, Article("fr", "Mode dorien", "scales_modes"))

    request = calls[0]
    assert request.url.host == "fr.wikipedia.org"
    assert request.url.params["explaintext"] == "1"
    assert request.url.params["prop"] == "extracts|info"
    assert fetched == {
        "title": "Mode dorien",
        "url": "https://fr.wikipedia.org/wiki/Mode_dorien",
        "revid": 123,
        "text": "Le mode de ré...\n\n== Histoire ==\n...",
    }


def test_fetch_article_returns_none_for_missing_page():
    http = fake_wikipedia({}, [])
    assert fetch_article(http, Article("en", "No such page", "x")) is None


def test_fetch_article_retries_when_rate_limited():
    responses = [
        httpx.Response(429),
        httpx.Response(
            200,
            json={
                "query": {
                    "pages": [{"title": "Capo", "fullurl": "u", "lastrevid": 1, "extract": "text"}]
                }
            },
        ),
    ]
    http = httpx.Client(transport=httpx.MockTransport(lambda request: responses.pop(0)))
    fetched = fetch_article(http, Article("en", "Capo", "guitar"), backoff_s=0)
    assert fetched["text"] == "text"


def test_fetch_article_retries_after_network_error():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            raise httpx.ConnectError("network unreachable")
        page = {"title": "Capo", "fullurl": "u", "lastrevid": 1, "extract": "text"}
        return httpx.Response(200, json={"query": {"pages": [page]}})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    fetched = fetch_article(http, Article("en", "Capo", "guitar"), backoff_s=0)
    assert fetched["text"] == "text"
    assert len(attempts) == 2


def test_save_then_sources_and_stats(tmp_path):
    raw = tmp_path / "raw"
    fetched = {
        "title": "Capo",
        "url": "https://en.wikipedia.org/wiki/Capo",
        "revid": 7,
        "text": "x" * 400,
    }
    save_article(raw, Article("en", "Capo", "guitar"), fetched, "2026-11-02T10:00:00+00:00")
    assert (raw / "en" / "en-capo.txt").read_text(encoding="utf-8") == "x" * 400

    metadata = load_metadata(raw)
    write_sources(tmp_path / "sources.csv", metadata)
    with (tmp_path / "sources.csv").open(encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["license"] == "CC BY-SA 4.0"
    assert row["language"] == "en"
    assert row["approx_tokens"] == "100"
    assert "revid=7" in row["notes"]

    stats = corpus_stats(metadata)
    assert stats["en"] == {"articles": 1, "characters": 400, "approx_tokens": 100}
    assert stats["total"]["articles"] == 1
