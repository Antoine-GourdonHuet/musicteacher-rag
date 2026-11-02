import numpy as np
from qdrant_client import QdrantClient

from musicteacher.ingestion.build_index import (
    COLLECTION,
    build_bm25,
    build_chunks,
    build_qdrant,
    chunk_records,
    index_text,
)
from musicteacher.ingestion.chunking import Chunk, WhitespaceTokenizer
from musicteacher.retrieval.bm25_tokens import bm25_terms, bm25_tokenize


def test_bm25_keeps_note_names_and_drops_stopwords():
    assert bm25_terms("The C major chord is C, E and G") == ["c", "major", "chord", "c", "e", "g"]
    assert bm25_terms("Le mode de ré commence sur D") == ["mode", "ré", "commence", "d"]


def test_index_text_prefixes_title_and_section():
    chunk = Chunk("en-x#000", "en-x", "en", "Dorian mode", "Modern Dorian mode", "Body.", 1)
    assert index_text(chunk) == "Dorian mode — Modern Dorian mode\nBody."


ARTICLES = [
    (
        {"id": "en-dorian", "language": "en", "title": "Dorian mode", "url": "u1", "category": "m"},
        "The Dorian mode has a minor third and a major sixth.",
    ),
    (
        {"id": "en-capo", "language": "en", "title": "Capo", "url": "u2", "category": "g"},
        "A capo clamps the strings of a guitar to raise the pitch.",
    ),
]


def test_chunk_records_carry_article_metadata():
    records = chunk_records(build_chunks(ARTICLES, WhitespaceTokenizer()), ARTICLES)
    assert [(r["chunk_id"], r["url"], r["category"]) for r in records] == [
        ("en-dorian#000", "u1", "m"),
        ("en-capo#000", "u2", "g"),
    ]
    assert records[1]["index_text"].startswith("Capo — Introduction\n")


def test_bm25_index_finds_the_matching_chunk():
    records = chunk_records(build_chunks(ARTICLES, WhitespaceTokenizer()), ARTICLES)
    retriever = build_bm25([r["index_text"] for r in records])
    ids, _scores = retriever.retrieve(bm25_tokenize(["guitar capo"]), k=1, show_progress=False)
    assert ids[0][0] == 1


def test_qdrant_collection_stores_vectors_and_payload():
    records = chunk_records(build_chunks(ARTICLES, WhitespaceTokenizer()), ARTICLES)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    client = QdrantClient(":memory:")

    build_qdrant(client, records, vectors)
    build_qdrant(client, records, vectors)  # rebuilding replaces, never duplicates

    assert client.count(COLLECTION).count == 2
    hit = client.query_points(COLLECTION, query=[0.1, 0.9], limit=1).points[0]
    assert hit.id == 1
    assert hit.payload["title"] == "Capo"
