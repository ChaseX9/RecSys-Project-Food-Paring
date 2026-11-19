"""Reference predictors. All statistics come from the recipes allowed by
``EvalSetup.train_recipe_mask``, never from the evaluated recipes.

Note: ``n_recipes`` in ingredients_vocab.csv counts ALL recipes (test included), so it must
not be used as a popularity score.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp


def load_train_recipe_ingredient(processed_dir: str | Path, train_recipe_mask: np.ndarray) -> sp.csr_matrix:
    """Binary recipes x ingredients matrix restricted to the allowed recipes."""
    m = sp.load_npz(Path(processed_dir) / "recipe_ingredient.npz").tocsr()
    if m.shape[0] != len(train_recipe_mask):
        raise ValueError("recipe_ingredient.npz and recipes.parquet are not aligned")
    return m[np.asarray(train_recipe_mask, dtype=bool)]


def _top_k(scores: np.ndarray, exclude, k: int) -> list[int]:
    scores = scores.astype(np.float64, copy=True)
    scores[list(exclude)] = -np.inf
    k = min(k, int(np.isfinite(scores).sum()))
    if k <= 0:
        return []
    top = np.argpartition(-scores, k - 1)[:k]
    # tie-break by ingredient id for determinism
    return [int(i) for i in sorted(top, key=lambda i: (-scores[i], i))]


class RandomBaseline:
    def __init__(self, n_items: int, seed: int = 42):
        self.n_items = n_items
        self.rng = np.random.default_rng(seed)

    def __call__(self, query_ids, user_id, k):
        return _top_k(self.rng.random(self.n_items), query_ids, k)


class PopularityBaseline:
    """Most frequent ingredients in train recipes, same list for everyone."""

    def __init__(self, train_ri: sp.csr_matrix):
        self.pop = np.asarray(train_ri.sum(axis=0)).ravel()

    def __call__(self, query_ids, user_id, k):
        return _top_k(self.pop, query_ids, k)


class CooccurrenceBaseline:
    """Sum of train co-occurrence counts with the query ingredients (strong, non-learned)."""

    def __init__(self, train_ri: sp.csr_matrix):
        co = (train_ri.T @ train_ri).tocsr()
        co.setdiag(0)
        co.eliminate_zeros()
        self.co = co

    def __call__(self, query_ids, user_id, k):
        scores = np.asarray(self.co[list(query_ids)].sum(axis=0)).ravel()
        return _top_k(scores, query_ids, k)


def make_baselines(names, train_ri: sp.csr_matrix, seed: int = 42) -> dict:
    factories = {
        "random": lambda: RandomBaseline(train_ri.shape[1], seed),
        "popularity": lambda: PopularityBaseline(train_ri),
        "cooccurrence": lambda: CooccurrenceBaseline(train_ri),
    }
    unknown = set(names) - set(factories)
    if unknown:
        raise ValueError(f"unknown baselines: {sorted(unknown)}")
    return {n: factories[n]() for n in names}
