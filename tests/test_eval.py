import math

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.eval.baselines import (CooccurrenceBaseline, PopularityBaseline, RandomBaseline,
                                load_train_recipe_ingredient)
from src.eval.harness import evaluate, evaluate_predictions
from src.eval.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from src.eval.protocol import build_cases, holdout_recipe_ids, mask_ingredients


# ---------------------------------------------------------------- metrics
def test_ndcg_perfect_and_zero():
    assert ndcg_at_k([3, 1, 2], {3, 1}, 10) == pytest.approx(1.0)
    assert ndcg_at_k([4, 5, 6], {1, 2}, 10) == 0.0
    assert ndcg_at_k([1, 2], set(), 10) == 0.0


def test_ndcg_hand_computed():
    # hits at ranks 2 and 3: (1/log2(3) + 1/log2(4)) / (1 + 1/log2(3))
    expected = (1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3))
    assert ndcg_at_k([5, 1, 7], {1, 7}, 3) == pytest.approx(expected)  # ~0.6934


def test_ndcg_cut_at_k():
    assert ndcg_at_k([9, 8, 1], {1}, 2) == 0.0
    assert ndcg_at_k([9, 8, 1], {1}, 3) == pytest.approx(0.5)


def test_recall_hit_mrr():
    assert recall_at_k([1, 2, 7], {1, 7, 9}, 2) == pytest.approx(1 / 3)
    assert recall_at_k([1, 2, 7], {1, 7, 9}, 3) == pytest.approx(2 / 3)
    assert hit_rate_at_k([4, 7], {7}, 1) == 0.0 and hit_rate_at_k([4, 7], {7}, 2) == 1.0
    assert reciprocal_rank([4, 5, 7], {7}) == pytest.approx(1 / 3)


def test_duplicates_not_counted_twice():
    assert recall_at_k([1, 1, 1], {1, 2}, 3) == pytest.approx(0.5)
    assert ndcg_at_k([1, 1], {1}, 2) == pytest.approx(1.0)


# ---------------------------------------------------------------- harness on mock predictions
def _cases():
    return pd.DataFrame({
        "case_id": [0, 1, 2],
        "user_id": [10, 11, 12],
        "recipe_id": [100, 101, 102],
        "query_ids": [[0, 1], [2, 3], [4, 5]],
        "target_ids": [[6], [7, 8], [9]],
    })


def test_mock_predictions_return_metrics():
    cases = _cases()
    perfect = {0: [6, 20, 21], 1: [7, 8, 22], 2: [9]}
    r = evaluate_predictions(perfect, cases, ks=(10, 20))
    assert r["ndcg@10"] == pytest.approx(1.0) and r["recall@20"] == pytest.approx(1.0)
    assert r["n_cases"] == 3 and r["n_missing"] == 0

    wrong = {0: [30], 1: [31], 2: [32]}
    r = evaluate_predictions(wrong, cases)
    assert r["ndcg@10"] == 0.0 and r["recall@20"] == 0.0


def test_missing_predictions_score_zero():
    r = evaluate_predictions({0: [6]}, _cases(), ks=(10,))
    assert r["n_missing"] == 2
    assert r["ndcg@10"] == pytest.approx(1 / 3)


def test_query_ingredients_are_removed_before_scoring():
    # predicting the query ingredient 0 first must not push the hit down to rank 2
    r = evaluate_predictions({0: [0, 6], 1: [], 2: []}, _cases(), ks=(1,))
    assert r["hit@1"] == pytest.approx(1 / 3)


def test_evaluate_with_predict_fn():
    cases = _cases()
    oracle = dict(zip(cases.case_id, cases.target_ids))
    lookup = {tuple(q): t for q, t in zip(cases.query_ids, cases.target_ids)}
    r = evaluate(lambda q, u, k: lookup[tuple(q)], cases, ks=(10,))
    assert r["ndcg@10"] == pytest.approx(1.0)
    assert evaluate_predictions(oracle, cases, ks=(10,))["ndcg@10"] == pytest.approx(1.0)


# ---------------------------------------------------------------- protocol
def test_mask_ingredients_partition():
    rng = np.random.default_rng(0)
    q, t = mask_ingredients(list(range(10)), rng, hidden_ratio=0.2, min_query=2)
    assert len(t) == 2 and len(q) == 8
    assert set(q).isdisjoint(t) and set(q) | set(t) == set(range(10))
    assert mask_ingredients([1, 2], rng, hidden_ratio=0.2, min_query=2) is None


def _write_processed(d):
    # recipes 0-2 have train interactions, recipes 3-4 only appear in val/test
    recipes = pd.DataFrame({
        "recipe_id": [0, 1, 2, 3, 4],
        "name": list("abcde"),
        "ingredient_ids": [np.array(x, dtype=np.int32) for x in
                           ([0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 4, 5, 1], [2, 4, 5])],
        "is_train_recipe": [True, True, True, False, False],
    })
    recipes.to_parquet(d / "recipes.parquet", index=False)
    pd.DataFrame({"user_id": [1, 2, 3], "recipe_id": [3, 0, 4]}).to_parquet(d / "interactions_test.parquet")
    pd.DataFrame({"user_id": [1, 2], "recipe_id": [1, 2]}).to_parquet(d / "interactions_val.parquet")
    indptr = np.cumsum([0] + [len(x) for x in recipes.ingredient_ids])
    ri = sp.csr_matrix((np.ones(indptr[-1], dtype=np.float32),
                        np.concatenate(recipes.ingredient_ids.tolist()), indptr), shape=(5, 6))
    sp.save_npz(d / "recipe_ingredient.npz", ri)


def test_unseen_recipes_protocol_has_no_train_recipe(tmp_path):
    _write_processed(tmp_path)
    s = build_cases(tmp_path, split="test", protocol="unseen_recipes", min_query=2)
    assert set(s.cases.recipe_id) == {3, 4}      # recipe 0 has train interactions -> excluded
    assert list(s.train_recipe_mask) == [True, True, True, False, False]
    assert s.cases.user_id.tolist() == [1, 3]
    # deterministic
    s2 = build_cases(tmp_path, split="test", protocol="unseen_recipes", min_query=2)
    assert s.cases.equals(s2.cases)


def test_baselines_never_see_eval_recipes(tmp_path):
    # ingredients 4 and 5 only appear in recipes 3 and 4 (val/test only): zero train signal
    _write_processed(tmp_path)
    s = build_cases(tmp_path, split="test", protocol="unseen_recipes", min_query=2)
    ri = load_train_recipe_ingredient(tmp_path, s.train_recipe_mask)
    assert ri.shape[0] == 3
    assert PopularityBaseline(ri).pop[[4, 5]].sum() == 0
    assert CooccurrenceBaseline(ri).co[:, [4, 5]].nnz == 0


def test_recipe_holdout_protocol(tmp_path):
    _write_processed(tmp_path)
    recipes = pd.read_parquet(tmp_path / "recipes.parquet")
    held = set(holdout_recipe_ids(recipes, frac=0.4, seed=0))
    assert len(held) == 2
    val = build_cases(tmp_path, split="val", protocol="recipe_holdout", holdout_frac=0.4, seed=0, min_query=2)
    test = build_cases(tmp_path, split="test", protocol="recipe_holdout", holdout_frac=0.4, seed=0, min_query=2)
    assert set(val.cases.recipe_id).isdisjoint(test.cases.recipe_id)
    assert set(val.cases.recipe_id) | set(test.cases.recipe_id) <= held
    allowed = set(recipes.recipe_id[test.train_recipe_mask])
    assert allowed.isdisjoint(held)


def test_baselines_exclude_query_and_rank():
    ri = sp.csr_matrix(np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 1, 0]], dtype=np.float32))
    assert PopularityBaseline(ri)([0], None, 2) == [1, 2]
    assert CooccurrenceBaseline(ri)([1], None, 3) == [0, 2, 3]   # 3 never co-occurs -> last
    r = RandomBaseline(4, seed=0)([0], None, 10)
    assert sorted(r) == [1, 2, 3]
