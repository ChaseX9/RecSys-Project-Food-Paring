"""Rating-aware user profiles (ratings 1-5, train interactions only).

Level 1 (counting, explicit):
  weight(u, r) = rating - b_u, where b_u is the user's mean rating shrunk toward the global
  mean (``shrink_k`` pseudo-ratings), so a user who gives 5 to everything still gets a
  (small) positive weight instead of an empty profile. Negative weights only count when the
  rating is <= ``dislike_max_rating`` (a 4/5 from a generous user is not a dislike).
  like[u, i]    = sum of positive weights of the recipes containing i, times idf(i)
  dislike[u, i] = same with negative weights
  idf(i) = log(n_recipes / df(i)): garlic or onion, present everywhere, no longer dominate.
  confidence(u) = n_u / (n_u + conf_k): users with few ratings are trusted less.

Level 2 (embedding space, generalizes):
  each recipe = mean of its ingredients; user signal S[u] = sum_r weight(u, r) * recipe_r,
  user vector = S[u] @ E (ingredient embeddings). A user who likes ginger moves toward
  lemongrass too, even without having rated it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class RatingProfiles:
    user_ids: np.ndarray        # sorted train user ids, row order of every matrix below
    like: sp.csr_matrix         # users x ingredients, >= 0, L1-normalized rows
    dislike: sp.csr_matrix      # users x ingredients, >= 0, L1-normalized rows
    signal: sp.csr_matrix       # users x ingredients, signed, for the embedding user vector
    confidence: np.ndarray      # (n_users,) in [0, 1)
    n_ratings: np.ndarray       # (n_users,)

    def user_vectors(self, embeddings: np.ndarray) -> np.ndarray:
        """Level 2: users x d, L2-normalized (zero rows stay zero)."""
        u = np.asarray(self.signal @ embeddings, dtype=np.float32)
        n = np.linalg.norm(u, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return u / n


def _l1_rows(m: sp.csr_matrix) -> sp.csr_matrix:
    s = np.asarray(abs(m).sum(axis=1)).ravel()
    s[s == 0] = 1.0
    return (sp.diags(1.0 / s) @ m).tocsr()


def user_recipe_weights(train: pd.DataFrame, recipe_index: dict, n_recipes: int,
                        shrink_k: float = 5.0, dislike_max_rating: float = 3.0):
    """Signed users x recipes matrix of centered, shrunk ratings."""
    g = float(train["rating"].mean())
    stats = train.groupby("user_id")["rating"].agg(["sum", "count"])
    base = (stats["sum"] + shrink_k * g) / (stats["count"] + shrink_k)
    user_ids = np.sort(train["user_id"].unique())
    u_idx = pd.Series(np.arange(len(user_ids)), index=user_ids)

    w = train["rating"].to_numpy(float) - train["user_id"].map(base).to_numpy(float)
    w[(w < 0) & (train["rating"].to_numpy() > dislike_max_rating)] = 0.0
    rows = train["user_id"].map(u_idx).to_numpy()
    cols = train["recipe_id"].map(recipe_index).to_numpy()
    keep = ~pd.isna(cols) & (w != 0)
    W = sp.csr_matrix((w[keep], (rows[keep], cols[keep].astype(np.int64))),
                      shape=(len(user_ids), n_recipes))
    W.sum_duplicates()
    n_ratings = stats.loc[user_ids, "count"].to_numpy()
    return user_ids, W, n_ratings


def build_rating_profiles(processed_dir: str | Path, shrink_k: float = 5.0,
                          dislike_max_rating: float = 3.0, conf_k: float = 10.0,
                          use_idf: bool = True) -> RatingProfiles:
    processed_dir = Path(processed_dir)
    recipes = pd.read_parquet(processed_dir / "recipes.parquet", columns=["recipe_id", "is_train_recipe"])
    recipe_index = dict(zip(recipes["recipe_id"], range(len(recipes))))
    ri = sp.load_npz(processed_dir / "recipe_ingredient.npz").tocsr()
    train = pd.read_parquet(processed_dir / "interactions_train.parquet",
                            columns=["user_id", "recipe_id", "rating"])

    user_ids, W, n_ratings = user_recipe_weights(train, recipe_index, ri.shape[0],
                                                 shrink_k, dislike_max_rating)

    # idf over train recipes only
    train_ri = ri[recipes["is_train_recipe"].to_numpy(dtype=bool)]
    df = np.asarray((train_ri > 0).sum(axis=0)).ravel().astype(np.float64)
    idf = np.log((train_ri.shape[0] + 1.0) / (df + 1.0)) if use_idf else np.ones_like(df)

    Wp, Wn = W.maximum(0).tocsr(), (-W).maximum(0).tocsr()
    like = _l1_rows((Wp @ ri) @ sp.diags(idf))
    dislike = _l1_rows((Wn @ ri) @ sp.diags(idf))
    signal = (W @ _l1_rows(ri)).tocsr()          # recipe = mean of its ingredients
    confidence = n_ratings / (n_ratings + conf_k) if conf_k > 0 else np.ones(len(n_ratings))
    return RatingProfiles(user_ids, like.tocsr(), dislike.tocsr(), signal, confidence, n_ratings)
