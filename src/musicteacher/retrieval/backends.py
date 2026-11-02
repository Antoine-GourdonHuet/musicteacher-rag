"""Real backends for `Retriever`: Qdrant dense search, BM25 search and the cross-encoder reranker.

Kept apart from `search.py` so the search logic can be unit-tested without loading any model.
"""

import json
from functools import cache
from pathlib import Path

import bm25s
import torch
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from musicteacher.config import Settings, get_settings
from musicteacher.ingestion.build_index import COLLECTION
from musicteacher.retrieval.bm25_tokens import bm25_tokenize
from musicteacher.retrieval.embedder import MAX_SEQ_LENGTH, embed, load_embedder
from musicteacher.retrieval.search import Backend, Hit, Reranker, Retriever


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dense_backend(settings: Settings, client: QdrantClient) -> Backend:
    model = load_embedder(settings.embedding_model)

    def search(query: str, n: int) -> list[tuple[int, float]]:
        vector = embed(model, [query])[0].tolist()
        points = client.query_points(COLLECTION, query=vector, limit=n).points
        return [(int(p.id), float(p.score)) for p in points]

    return search


def bm25_backend(index_dir: Path) -> Backend:
    retriever = bm25s.BM25.load(str(index_dir))

    def search(query: str, n: int) -> list[tuple[int, float]]:
        ids, scores = retriever.retrieve(bm25_tokenize([query]), k=n, show_progress=False)
        # bm25s pads with zero-score documents when few match; drop them.
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if s > 0]

    return search


def cross_encoder_reranker(model_name: str) -> Reranker:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(model_name, device=device, max_length=MAX_SEQ_LENGTH)
    if device == "cuda":
        model.half()

    def rerank(query: str, texts: list[str]) -> list[float]:
        return model.predict([(query, t) for t in texts], batch_size=16).tolist()

    return rerank


def load_retriever(settings: Settings | None = None, *, with_reranker: bool = True) -> Retriever:
    settings = settings or get_settings()
    index_dir = settings.data_dir / "index"
    records = load_records(settings.data_dir / "processed" / "chunks.jsonl")
    client = QdrantClient(path=str(index_dir / "qdrant"))
    return Retriever(
        records,
        dense=dense_backend(settings, client),
        bm25=bm25_backend(index_dir / "bm25"),
        reranker=cross_encoder_reranker(settings.reranker_model) if with_reranker else None,
        fusion_depth=settings.fusion_depth,
        rrf_k=settings.rrf_k,
        rerank_candidates=settings.rerank_candidates,
    )


@cache
def default_retriever() -> Retriever:
    """Loaded once per process: models and indexes take a few seconds to load."""
    return load_retriever()


def search(query: str, mode: str = "hybrid_rerank", k: int = 5) -> list[Hit]:
    """Search the corpus. `mode` is one of dense, bm25, hybrid, hybrid_rerank."""
    return default_retriever().search(query, mode, k)
