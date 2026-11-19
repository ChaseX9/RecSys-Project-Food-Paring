"""Beyond-accuracy metrics, computed on the same prediction lists as NDCG/Recall.

They answer "is the model just recommending popular ingredients?" and "can each
recommendation be explained by real evidence?". All statistics come from train recipes.

- coverage@k   share of the catalog that appears in at least one top-k list
- longtail@k   share of recommended items outside the most popular head
- novelty@k    mean self-information -log2(p_i) of recommended items (higher = less obvious)
- grounded@k   share of recommended items backed by an explanation signal: strong NPMI with a
               query ingredient or chemical similarity above the threshold used by
               src/explain/reasons.py (explanation fidelity: the reason shown is actually true)
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp


def item_popularity(train_ri: sp.csr_matrix) -> np.ndarray:
    """Number of train recipes containing each ingredient."""
    return np.asarray(train_ri.sum(axis=0)).ravel().astype(np.float64)


def longtail_mask(pop: np.ndarray, head_frac: float = 0.2) -> np.ndarray:
    """True for long-tail items, i.e. outside the ``head_frac`` most popular ones."""
    n_head = int(round(head_frac * len(pop)))
    order = np.argsort(-pop, kind="stable")
    tail = np.ones(len(pop), dtype=bool)
    tail[order[:n_head]] = False
    return tail


def beyond_accuracy(predictions: Mapping[int, Sequence[int]], cases: pd.DataFrame,
                    pop: np.ndarray, n_recipes: int, k: int = 10, head_frac: float = 0.2,
                    npmi: sp.csr_matrix | None = None, chemical: sp.csr_matrix | None = None,
                    npmi_thr: float = 0.15, chem_thr: float = 0.05) -> dict:
    n_items = len(pop)
    tail = longtail_mask(pop, head_frac)
    self_info = -np.log2((pop + 1.0) / (n_recipes + 1.0))
    seen: set[int] = set()
    n_rec = n_tail = n_grounded = 0
    novelty = 0.0
    for case_id, query in zip(cases["case_id"], cases["query_ids"]):
        q = [int(x) for x in query]
        qs = set(q)
        top = [int(x) for x in predictions.get(int(case_id), []) if int(x) not in qs][:k]
        if not top:
            continue
        seen.update(top)
        n_rec += len(top)
        n_tail += int(tail[top].sum())
        novelty += float(self_info[top].sum())
        if npmi is not None or chemical is not None:
            ok = np.zeros(len(top), dtype=bool)
            if npmi is not None:
                ok |= np.asarray(npmi[q][:, top].max(axis=0).todense()).ravel() > npmi_thr
            if chemical is not None:
                chem = np.asarray(chemical[q][:, top].mean(axis=0)).ravel()
                ok |= chem > chem_thr
            n_grounded += int(ok.sum())
    out = {
        f"coverage@{k}": len(seen) / n_items,
        f"longtail@{k}": n_tail / n_rec if n_rec else 0.0,
        f"novelty@{k}": novelty / n_rec if n_rec else 0.0,
    }
    if npmi is not None or chemical is not None:
        out[f"grounded@{k}"] = n_grounded / n_rec if n_rec else 0.0
    return out
