import pytest

from musicteacher.retrieval.search import Retriever, rrf_fuse

RECORDS = [
    {
        "chunk_id": f"en-doc{i}#000",
        "title": f"Doc {i}",
        "section": "Introduction",
        "language": "en",
        "url": f"u{i}",
        "text": f"text {i}",
        "index_text": f"Doc {i} — Introduction\ntext {i}",
    }
    for i in range(6)
]


def fixed_backend(ranking: list[int]):
    """Fake backend: always returns `ranking`, with decreasing fake scores."""

    def search(query: str, n: int) -> list[tuple[int, float]]:
        return [(pos, 1.0 / (r + 1)) for r, pos in enumerate(ranking[:n])]

    return search


def test_rrf_rewards_documents_ranked_well_by_both():
    # doc 1 is 2nd for dense and 1st for BM25; doc 0 is 1st for dense only.
    fused = rrf_fuse([[0, 1, 2], [1, 3]], k=60)
    assert fused[0][0] == 1
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1] == (0, pytest.approx(1 / 61))


def test_rrf_ignores_raw_score_scales():
    # Only positions matter: the same rankings always give the same fusion.
    assert rrf_fuse([[2, 0], [0, 2]]) == rrf_fuse([[2, 0], [0, 2]])
    assert {doc for doc, _ in rrf_fuse([[2, 0], [0, 2]])} == {0, 2}


def make_retriever(reranker=None, **kwargs) -> Retriever:
    return Retriever(
        RECORDS,
        dense=fixed_backend([0, 1, 2, 3]),
        bm25=fixed_backend([3, 1, 4]),
        reranker=reranker,
        **kwargs,
    )


def test_dense_and_bm25_modes_return_their_own_ranking():
    retriever = make_retriever()
    assert [h.chunk_id for h in retriever.search("q", "dense", k=2)] == [
        "en-doc0#000",
        "en-doc1#000",
    ]
    assert [h.chunk_id for h in retriever.search("q", "bm25", k=2)] == [
        "en-doc3#000",
        "en-doc1#000",
    ]


def test_hybrid_mode_fuses_both_rankings():
    hits = make_retriever().search("q", "hybrid", k=3)
    # doc 1 (2nd + 2nd) and doc 3 (4th + 1st) beat doc 0 (1st dense only).
    assert [h.chunk_id for h in hits] == ["en-doc1#000", "en-doc3#000", "en-doc0#000"]
    assert [h.rank for h in hits] == [1, 2, 3]


def test_hybrid_rerank_reorders_the_top_candidates_by_reranker_score():
    def reranker(query: str, texts: list[str]) -> list[float]:
        return [10.0 if "text 4" in t else 0.0 for t in texts]  # loves doc 4

    hits = make_retriever(reranker=reranker, rerank_candidates=5).search("q", "hybrid_rerank", k=2)
    assert hits[0].chunk_id == "en-doc4#000"
    assert hits[0].score == 10.0


def test_reranker_only_sees_the_top_candidates():
    seen = []

    def reranker(query: str, texts: list[str]) -> list[float]:
        seen.extend(texts)
        return [0.0] * len(texts)

    make_retriever(reranker=reranker, rerank_candidates=2).search("q", "hybrid_rerank", k=2)
    assert len(seen) == 2


def test_unknown_mode_and_missing_reranker_are_errors():
    retriever = make_retriever()
    with pytest.raises(ValueError, match="unknown mode"):
        retriever.search("q", "magic")
    with pytest.raises(ValueError, match="reranker"):
        retriever.search("q", "hybrid_rerank")
