"""Top-k ranking metrics with binary relevance.

Every function takes a ranked list of item ids (best first) and the set of relevant ids.
Duplicates in the ranked list are ignored after their first occurrence, so a model cannot
score twice with the same item.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _dedupe(ranked: Sequence[int]) -> list[int]:
    return list(dict.fromkeys(int(x) for x in ranked))


def _hits(ranked: Sequence[int], relevant: Iterable[int], k: int) -> np.ndarray:
    """Binary hit vector of length <= k."""
    rel = {int(x) for x in relevant}
    return np.array([1.0 if x in rel else 0.0 for x in _dedupe(ranked)[:k]])


def dcg_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float:
    hits = _hits(ranked, relevant, k)
    if hits.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, hits.size + 2))
    return float((hits * discounts).sum())


def ndcg_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """DCG normalized by the ideal DCG (all relevant items ranked first)."""
    relevant = {int(x) for x in relevant}
    if not relevant:
        return 0.0
    n_ideal = min(len(relevant), k)
    idcg = float((1.0 / np.log2(np.arange(2, n_ideal + 2))).sum())
    return dcg_at_k(ranked, relevant, k) / idcg


def recall_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Share of relevant items found in the top k."""
    relevant = {int(x) for x in relevant}
    if not relevant:
        return 0.0
    return float(_hits(ranked, relevant, k).sum()) / len(relevant)


def hit_rate_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """1 if at least one relevant item is in the top k, else 0."""
    return float(_hits(ranked, relevant, k).sum() > 0)


def reciprocal_rank(ranked: Sequence[int], relevant: Iterable[int]) -> float:
    """1 / rank of the first relevant item (0 if none)."""
    rel = {int(x) for x in relevant}
    for rank, x in enumerate(_dedupe(ranked), 1):
        if x in rel:
            return 1.0 / rank
    return 0.0
