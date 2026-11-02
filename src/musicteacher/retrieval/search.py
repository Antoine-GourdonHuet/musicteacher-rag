"""One entry point for the four retrieval modes compared in the experiment.

- dense:          bge-m3 vectors, cosine similarity in Qdrant
- bm25:           exact-term scoring with bm25s
- hybrid:         dense top-N + BM25 top-N fused with Reciprocal Rank Fusion (RRF)
- hybrid_rerank:  hybrid top-20 re-scored by a cross-encoder (bge-reranker-v2-m3)

Every backend returns positions in `chunks.jsonl` (= Qdrant point ids = BM25 document ids), so
results from different modes refer to the same chunks.
"""

from collections.abc import Callable
from dataclasses import dataclass

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")

# A backend takes (query, n) and returns [(chunk position, score)] best first.
Backend = Callable[[str, int], list[tuple[int, float]]]
# A reranker takes (query, candidate texts) and returns one relevance score per text.
Reranker = Callable[[str, list[str]], list[float]]


@dataclass(frozen=True)
class Hit:
    rank: int
    score: float
    chunk_id: str
    title: str
    section: str
    language: str
    url: str
    text: str


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank of d), rank from 1.

    Only ranks are used, never raw scores, so a cosine similarity (0-1) and a BM25 score (0-30)
    can be combined without any normalisation.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class Retriever:
    def __init__(
        self,
        records: list[dict],
        dense: Backend,
        bm25: Backend,
        reranker: Reranker | None = None,
        *,
        fusion_depth: int = 50,
        rrf_k: int = 60,
        rerank_candidates: int = 20,
    ) -> None:
        self.records = records
        self._dense = dense
        self._bm25 = bm25
        self._reranker = reranker
        self.fusion_depth = fusion_depth
        self.rrf_k = rrf_k
        self.rerank_candidates = rerank_candidates

    def search(self, query: str, mode: str = "hybrid_rerank", k: int = 5) -> list[Hit]:
        if mode == "dense":
            ranked = self._dense(query, k)
        elif mode == "bm25":
            ranked = self._bm25(query, k)
        elif mode == "hybrid":
            ranked = self._hybrid(query)[:k]
        elif mode == "hybrid_rerank":
            ranked = self._rerank(query, self._hybrid(query)[: self.rerank_candidates])[:k]
        else:
            raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
        return [self._hit(rank, pos, score) for rank, (pos, score) in enumerate(ranked, start=1)]

    def _hybrid(self, query: str) -> list[tuple[int, float]]:
        dense_ids = [pos for pos, _ in self._dense(query, self.fusion_depth)]
        bm25_ids = [pos for pos, _ in self._bm25(query, self.fusion_depth)]
        return rrf_fuse([dense_ids, bm25_ids], k=self.rrf_k)

    def _rerank(self, query: str, candidates: list[tuple[int, float]]) -> list[tuple[int, float]]:
        if self._reranker is None:
            raise ValueError("hybrid_rerank needs a reranker")
        texts = [self.records[pos]["index_text"] for pos, _ in candidates]
        scores = self._reranker(query, texts)
        rescored = [(pos, float(s)) for (pos, _), s in zip(candidates, scores, strict=True)]
        return sorted(rescored, key=lambda item: item[1], reverse=True)

    def _hit(self, rank: int, pos: int, score: float) -> Hit:
        r = self.records[pos]
        return Hit(
            rank=rank,
            score=score,
            chunk_id=r["chunk_id"],
            title=r["title"],
            section=r["section"],
            language=r["language"],
            url=r["url"],
            text=r["text"],
        )
