"""Validation signals for training (Week 3, Q1: do the curves converge?).

Two signals, both computed on the *val* split only (val recipes are disjoint from test
recipes, see src/eval/protocol.py), never on test:

- val BPR loss: BPR on ingredient pairs that co-occur in val recipes. These recipes have no
  train interaction, so their pairs are not used to build the graph: the loss measures
  generalization, not memorization. The same loss on a fixed sample of train edges gives a
  comparable train curve.
- val NDCG@10: the evaluation harness on val cases with the hybrid ranker, i.e. exactly the
  metric reported at the end. Used to keep the best epoch (early stopping).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.eval.harness import evaluate, from_hybrid_ranker
from src.eval.protocol import build_cases
from src.ranking.hybrid import HybridRanker, RankingConfig


@dataclass
class BPRTriples:
    src: np.ndarray
    pos: np.ndarray
    neg: np.ndarray


def _fixed_negatives(n: int, n_nodes: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n_nodes, size=n, dtype=np.int64)


def val_pair_triples(cases: pd.DataFrame, recipes: pd.DataFrame, n_nodes: int,
                     seed: int = 42, max_pairs: int = 200_000) -> BPRTriples:
    """All ingredient pairs (i < j) of the val recipes, with fixed random negatives."""
    ings = dict(zip(recipes["recipe_id"], recipes["ingredient_ids"]))
    pairs = set()
    for rid in pd.unique(cases["recipe_id"]):
        ids = np.unique(np.asarray(ings[rid], dtype=np.int64))
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                pairs.add((int(ids[a]), int(ids[b])))
    arr = np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    if len(arr) > max_pairs:
        arr = arr[rng.choice(len(arr), size=max_pairs, replace=False)]
    return BPRTriples(arr[:, 0], arr[:, 1], _fixed_negatives(len(arr), n_nodes, rng))


def train_edge_triples(adj: sp.csr_matrix, n: int, seed: int = 42) -> BPRTriples:
    """Fixed sample of train graph edges (i < j) with fixed negatives, same size as val."""
    coo = sp.triu(adj, k=1).tocoo()
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(coo.nnz, size=min(n, coo.nnz), replace=False)
    return BPRTriples(coo.row[idx].astype(np.int64), coo.col[idx].astype(np.int64),
                      _fixed_negatives(len(idx), adj.shape[0], rng))


def bpr_numpy(emb: np.ndarray, t: BPRTriples) -> float:
    """BPR loss of fixed triples on given embeddings (no gradient, deterministic)."""
    s, p, n = emb[t.src], emb[t.pos], emb[t.neg]
    diff = (s * p).sum(1) - (s * n).sum(1)
    return float(np.mean(np.logaddexp(0.0, -diff)))   # -log(sigmoid(diff)), stable


def build_validation(processed_dir: str | Path, adj: sp.csr_matrix, ranking: dict,
                     eval_cfg: dict, max_cases: int | None = None) -> dict:
    """Everything train_nlgcl needs to validate each epoch."""
    processed_dir = Path(processed_dir)
    setup = build_cases(processed_dir, split="val", protocol="unseen_recipes",
                        hidden_ratio=eval_cfg["hidden_ratio"], min_query=eval_cfg["min_query"],
                        seed=eval_cfg["seed"], max_cases=max_cases)
    recipes = pd.read_parquet(processed_dir / "recipes.parquet", columns=["recipe_id", "ingredient_ids"])
    n_nodes = adj.shape[0]
    val_t = val_pair_triples(setup.cases, recipes, n_nodes, seed=eval_cfg["seed"])
    train_t = train_edge_triples(adj, len(val_t.src), seed=eval_cfg["seed"])

    vocab = pd.read_csv(processed_dir / "ingredients_vocab.csv")
    chem_path = processed_dir / "chemical_similarity.npz"
    chem = sp.load_npz(chem_path).tocsr() if chem_path.exists() else None
    profiles = sp.load_npz(processed_dir / "user_profiles.npz").tocsr()
    uids = np.load(processed_dir / "user_profile_ids.npy")
    rcfg = RankingConfig(**{k: v for k, v in ranking.items() if k in RankingConfig.__dataclass_fields__})

    def ndcg_fn(emb: np.ndarray) -> dict:
        ranker = HybridRanker(emb, vocab, chem, None, profiles, uids, rcfg)
        r = evaluate(from_hybrid_ranker(ranker), setup.cases, ks=(10, 20))
        return {"val_ndcg@10": r["ndcg@10"], "val_recall@20": r["recall@20"]}

    def validate(emb: np.ndarray) -> dict:
        out = {"val_bpr": bpr_numpy(emb, val_t), "train_bpr_fixed": bpr_numpy(emb, train_t)}
        out.update(ndcg_fn(emb))
        return out

    print(f"Validation: {len(setup.cases):,} val cases, {len(val_t.src):,} val pairs "
          f"(val recipes are disjoint from train and test)")
    return {"validate": validate}


ValidateFn = Callable[[np.ndarray], dict]
