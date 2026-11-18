"""Evaluation harness: predictions in, metrics out.

Two entry points:
- ``evaluate_predictions``: already computed predictions {case_id: ranked ingredient ids},
  e.g. mock predictions or a model's dumped output.
- ``evaluate``: any predictor ``predict_fn(query_ids, user_id, k) -> ranked ingredient ids``.

Query ingredients are removed from every prediction before scoring (they are known to the
user and can never be a hidden target), so all predictors are scored the same way.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import hit_rate_at_k, ndcg_at_k, recall_at_k, reciprocal_rank

PredictFn = Callable[[list[int], object, int], Sequence[int]]


def evaluate_predictions(predictions: Mapping[int, Sequence[int]], cases: pd.DataFrame,
                         ks: Sequence[int] = (10, 20), return_per_case: bool = False):
    """Average NDCG@k, Recall@k, HitRate@k and MRR over all cases.

    A case with no prediction counts as an empty list (score 0) and is reported in
    ``n_missing``, so a predictor cannot look better by skipping hard cases.
    """
    ks = sorted(set(int(k) for k in ks))
    per_case = []
    n_missing = 0
    for case_id, query, target in zip(cases["case_id"], cases["query_ids"], cases["target_ids"]):
        ranked = predictions.get(int(case_id))
        if ranked is None:
            n_missing += 1
            ranked = []
        q = {int(x) for x in query}
        ranked = [int(x) for x in ranked if int(x) not in q]
        row = {"case_id": int(case_id), "mrr": reciprocal_rank(ranked, target)}
        for k in ks:
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, target, k)
            row[f"recall@{k}"] = recall_at_k(ranked, target, k)
            row[f"hit@{k}"] = hit_rate_at_k(ranked, target, k)
        per_case.append(row)

    df = pd.DataFrame(per_case)
    metrics_cols = [c for c in df.columns if c != "case_id"] if len(df) else []
    summary = {c: float(df[c].mean()) for c in metrics_cols}
    summary["n_cases"] = int(len(cases))
    summary["n_missing"] = int(n_missing)
    return (summary, df) if return_per_case else summary


def predict_all(predict_fn: PredictFn, cases: pd.DataFrame, k: int) -> dict[int, list[int]]:
    preds = {}
    for case_id, user_id, query in zip(cases["case_id"], cases["user_id"], cases["query_ids"]):
        user_id = None if user_id is None or (isinstance(user_id, float) and np.isnan(user_id)) else user_id
        preds[int(case_id)] = [int(x) for x in predict_fn(list(query), user_id, k)]
    return preds


def evaluate(predict_fn: PredictFn, cases: pd.DataFrame, ks: Sequence[int] = (10, 20),
             return_per_case: bool = False):
    """Run ``predict_fn`` on every case, then score with ``evaluate_predictions``."""
    preds = predict_all(predict_fn, cases, max(ks))
    return evaluate_predictions(preds, cases, ks, return_per_case)


def from_hybrid_ranker(ranker) -> PredictFn:
    """Adapter for HybridRanker.recommend (branch model-train), which takes names.

    Exact vocabulary names resolve exactly in HybridRanker._resolve, so no fuzzy matching
    happens here.
    """
    def predict(query_ids, user_id, k):
        names = [ranker.id_to_name[int(i)] for i in query_ids]
        uid = int(user_id) if user_id is not None else None
        return [r["ingredient_id"] for r in ranker.recommend(names, user_id=uid, top_k=k)]
    return predict
