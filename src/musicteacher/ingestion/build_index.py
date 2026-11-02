"""Build the retrieval indexes from the raw corpus.

1. Clean every article and chunk it by sections (max 300 bge-m3 tokens).
2. Write all chunks to `data/processed/chunks.jsonl`: the single list both indexes share.
3. Dense index: embed `index_text` with BAAI/bge-m3 and store it in a local Qdrant collection,
   with metadata (article, language, section, url, category, text) as payload.
4. Sparse index: BM25 (`bm25s`) over the same `index_text`, saved to `data/index/bm25`.

The indexed text is "Title — Section" + chunk text, so a chunk keeps the context it came from.

Usage: uv run python -m musicteacher.ingestion.build_index
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

import bm25s
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from musicteacher.config import get_settings
from musicteacher.ingestion.chunking import (
    Chunk,
    HFTokenizer,
    Tokenizer,
    chunk_sections,
    load_raw_articles,
)
from musicteacher.ingestion.clean import split_sections
from musicteacher.retrieval.bm25_tokens import bm25_tokenize

COLLECTION = "chunks"
MAX_TOKENS = 300


def index_text(chunk: Chunk) -> str:
    return f"{chunk.title} — {chunk.section}\n{chunk.text}"


def build_chunks(articles: list[tuple[dict, str]], tok: Tokenizer) -> list[Chunk]:
    chunks = []
    for meta, text in articles:
        sections = split_sections(text, meta["language"])
        chunks += chunk_sections(meta, sections, tok, max_tokens=MAX_TOKENS)
    return chunks


def chunk_records(chunks: list[Chunk], articles: list[tuple[dict, str]]) -> list[dict]:
    """Chunk fields + article url/category + the exact text that is indexed."""
    meta_by_id = {meta["id"]: meta for meta, _ in articles}
    return [
        {
            **asdict(c),
            "url": meta_by_id[c.article_id]["url"],
            "category": meta_by_id[c.article_id]["category"],
            "index_text": index_text(c),
        }
        for c in chunks
    ]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_qdrant(client: QdrantClient, records: list[dict], vectors: np.ndarray) -> None:
    """(Re)create the collection; point id = position in chunks.jsonl."""
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION, vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE)
    )
    batch = 256
    for start in range(0, len(records), batch):
        client.upsert(
            COLLECTION,
            points=[
                PointStruct(id=start + i, vector=vectors[start + i].tolist(), payload=record)
                for i, record in enumerate(records[start : start + batch])
            ],
        )


def build_bm25(texts: list[str], save_dir: Path | None = None) -> bm25s.BM25:
    """BM25 document i = chunk i of chunks.jsonl (same order as the Qdrant point ids)."""
    retriever = bm25s.BM25()
    retriever.index(bm25_tokenize(texts), show_progress=False)
    if save_dir is not None:
        retriever.save(str(save_dir))
    return retriever


def main() -> None:
    from musicteacher.retrieval.embedder import embed, load_embedder

    settings = get_settings()
    index_dir = settings.data_dir / "index"
    timings = {}

    start = time.perf_counter()
    articles = load_raw_articles(settings.data_dir / "raw")
    chunks = build_chunks(articles, HFTokenizer(settings.embedding_model))
    records = chunk_records(chunks, articles)
    write_jsonl(settings.data_dir / "processed" / "chunks.jsonl", records)
    timings["chunking_s"] = time.perf_counter() - start
    texts = [r["index_text"] for r in records]
    print(f"{len(articles)} articles -> {len(records)} chunks")

    start = time.perf_counter()
    model = load_embedder(settings.embedding_model)
    vectors = embed(model, texts)
    timings["embedding_s"] = time.perf_counter() - start
    print(f"Embedded {vectors.shape[0]} chunks, dim {vectors.shape[1]}, device {model.device}")

    start = time.perf_counter()
    client = QdrantClient(path=str(index_dir / "qdrant"))
    build_qdrant(client, records, vectors)
    count = client.count(COLLECTION).count
    client.close()
    timings["qdrant_s"] = time.perf_counter() - start

    start = time.perf_counter()
    build_bm25(texts, index_dir / "bm25")
    timings["bm25_s"] = time.perf_counter() - start

    stats = {
        "articles": len(articles),
        "chunks": len(records),
        "chunks_en": sum(r["language"] == "en" for r in records),
        "chunks_fr": sum(r["language"] == "fr" for r in records),
        "chunk_tokens_total": sum(r["n_tokens"] for r in records),
        "qdrant_points": count,
        "embedding_model": settings.embedding_model,
        "embedding_dim": int(vectors.shape[1]),
        "timings_s": {k: round(v, 1) for k, v in timings.items()},
    }
    (index_dir / "index_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
