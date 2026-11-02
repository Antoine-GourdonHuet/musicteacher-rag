"""Retrieval metrics, bootstrap confidence intervals and per-group summaries."""

import numpy as np


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Share of the relevant items found in the top k (1.0 or 0.0 with one relevant chunk)."""
    if not relevant:
        raise ValueError("recall needs at least one relevant item")
    top = set(retrieved[:k])
    return sum(item in top for item in relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """1 / rank of the first relevant item (0 if none was retrieved)."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def bootstrap_ci(
    values: list[float], n_resamples: int = 1000, seed: int = 42, confidence: float = 0.95
) -> dict:
    """Mean with a percentile bootstrap confidence interval.

    Resample the questions with replacement many times, recompute the mean each time; the middle
    95% of those means is the interval. Wide interval = the mean is not precise (few questions).
    """
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(seed)
    means = rng.choice(data, size=(n_resamples, data.size), replace=True).mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return {
        "mean": round(float(data.mean()), 4),
        "ci_low": round(float(low), 4),
        "ci_high": round(float(high), 4),
        "n": int(data.size),
    }


def summarise(rows: list[dict], metrics: list[str], seed: int = 42) -> dict:
    """Mean + 95% CI of each metric: overall, per language, per question type, cross-lingual.

    A row without a value for a metric (None, e.g. no retrieval metric for agentic questions) is
    left out of that metric only.
    """

    def block(subset: list[dict]) -> dict:
        return {
            m: bootstrap_ci([r[m] for r in subset if r.get(m) is not None], seed=seed)
            for m in metrics
        }

    def groups(key: str) -> dict:
        values = sorted({r[key] for r in rows})
        return {v: block([r for r in rows if r[key] == v]) for v in values}

    return {
        "overall": block(rows),
        "by_lang": groups("lang"),
        "by_qtype": groups("qtype"),
        "cross_lingual": block([r for r in rows if r["cross_lingual"]]),
    }
