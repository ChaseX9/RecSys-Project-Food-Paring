"""Rating-aware user profiles (ratings 1-5, train interactions only).

Recipe weights:
  weight(u, r) = rating - b_u, where b_u is the user's mean rating shrunk toward the global
  mean (``shrink_k`` pseudo-ratings), so a user who gives 5 to everything still gets a
  positive weight instead of an empty profile. A negative weight only counts when the
  rating is strictly below ``dislike_below_rating`` (default 3: only 1/5 and 2/5 are
  dislikes; a 3/5 or 4/5 from a generous user is not). Weights are then divided by the
  user's mean |weight|, so one typical rating weighs about 1 for every user.

Ingredient preference (level 1), from recipe-level ratings:
  P[u, i] = weight of the liked recipes containing i, N[u, i] = same for disliked recipes
  net[u, i] = (P - N) / (P + N + alpha)                 in (-1, 1)
  A rating is about a whole recipe, not about each ingredient. ``net`` measures how
  *consistently* an ingredient goes with liked recipes: flour in 25 liked and 2 disliked
  cakes stays clearly liked (+0.8), while an ingredient only seen in disliked recipes
  becomes negative. ``alpha`` (smoothing) keeps one single recipe from being decisive.
  (The former ``like`` / ``dislike`` profiles, normalized separately, let 2 disliked cakes
  weigh as much as 25 liked ones: they are kept for display only.)

Embedding user vector (level 2): S[u] = sum_r weight(u, r) * mean of the recipe's
ingredients, user vector = S[u] @ E (ingredient embeddings).

Users of the app (no Food.com history) rate recipes too: ``RatingData.profile_for`` builds the
same profile for one user at request time, from their app ratings (plus the train ratings of a
linked Food.com user), with the train global mean. No retraining needed.
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
    like: sp.csr_matrix         # users x ingredients, >= 0, L1-normalized rows (display)
    dislike: sp.csr_matrix      # users x ingredients, >= 0, L1-normalized rows (display)
    signal: sp.csr_matrix       # users x ingredients, signed, for the embedding user vector
    confidence: np.ndarray      # (n_users,) in [0, 1]
    n_ratings: np.ndarray       # (n_users,)
    net: sp.csr_matrix | None = None   # users x ingredients, consistency score in (-1, 1)

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
                        shrink_k: float = 5.0, dislike_below_rating: float = 3.0,
                        global_mean: float | None = None):
    """Signed users x recipes matrix of centered, shrunk ratings (scaled per user).

    ``global_mean``: the mean the user means are shrunk toward (default: mean of ``train``;
    pass the train mean when scoring a single new user)."""
    g = float(train["rating"].mean()) if global_mean is None else float(global_mean)
    stats = train.groupby("user_id")["rating"].agg(["sum", "count"])
    base = (stats["sum"] + shrink_k * g) / (stats["count"] + shrink_k)
    user_ids = np.sort(train["user_id"].unique())
    u_idx = pd.Series(np.arange(len(user_ids)), index=user_ids)

    w = train["rating"].to_numpy(float) - train["user_id"].map(base).to_numpy(float)
    w[(w < 0) & (train["rating"].to_numpy() >= dislike_below_rating)] = 0.0
    rows = train["user_id"].map(u_idx).to_numpy()
    cols = train["recipe_id"].map(recipe_index).to_numpy()
    keep = ~pd.isna(cols) & (w != 0)
    W = sp.csr_matrix((w[keep], (rows[keep], cols[keep].astype(np.int64))),
                      shape=(len(user_ids), n_recipes))
    W.sum_duplicates()
    # one typical rating ~ 1 for every user (generous raters have tiny raw weights)
    scale = np.asarray(abs(W).sum(axis=1)).ravel() / np.maximum(np.diff(W.indptr), 1)
    scale[scale == 0] = 1.0
    W = (sp.diags(1.0 / scale) @ W).tocsr()
    n_ratings = stats.loc[user_ids, "count"].to_numpy()
    return user_ids, W, n_ratings


def consistency_profile(P: sp.csr_matrix, N: sp.csr_matrix, alpha: float = 2.0) -> sp.csr_matrix:
    """(P - N) / (P + N + alpha), element-wise on the non-zeros of P + N."""
    S = (P + N).tocsr()
    S.sum_duplicates()
    inv = S.copy()
    inv.data = 1.0 / (inv.data + alpha)
    net = (P - N).tocsr().multiply(inv).tocsr()
    net.eliminate_zeros()
    return net


def build_rating_profiles(processed_dir: str | Path, shrink_k: float = 5.0,
                          dislike_below_rating: float = 3.0, conf_k: float = 10.0,
                          use_idf: bool = True, alpha: float = 2.0) -> RatingProfiles:
    processed_dir = Path(processed_dir)
    recipes = pd.read_parquet(processed_dir / "recipes.parquet", columns=["recipe_id", "is_train_recipe"])
    recipe_index = dict(zip(recipes["recipe_id"], range(len(recipes))))
    ri = sp.load_npz(processed_dir / "recipe_ingredient.npz").tocsr()
    train = pd.read_parquet(processed_dir / "interactions_train.parquet",
                            columns=["user_id", "recipe_id", "rating"])

    user_ids, W, n_ratings = user_recipe_weights(train, recipe_index, ri.shape[0],
                                                 shrink_k, dislike_below_rating)

    # idf over train recipes only
    train_ri = ri[recipes["is_train_recipe"].to_numpy(dtype=bool)]
    df = np.asarray((train_ri > 0).sum(axis=0)).ravel().astype(np.float64)
    idf = np.log((train_ri.shape[0] + 1.0) / (df + 1.0)) if use_idf else np.ones_like(df)

    return profiles_from_weights(user_ids, W, n_ratings, ri, idf, conf_k, alpha)


def profiles_from_weights(user_ids, W: sp.csr_matrix, n_ratings: np.ndarray, ri: sp.csr_matrix,
                          idf: np.ndarray, conf_k: float = 10.0, alpha: float = 2.0) -> RatingProfiles:
    """Ingredient profiles from the signed users x recipes weights (see the module docstring)."""
    Wp, Wn = W.maximum(0).tocsr(), (-W).maximum(0).tocsr()
    P, N = (Wp @ ri).tocsr(), (Wn @ ri).tocsr()
    like = _l1_rows(P @ sp.diags(idf))
    dislike = _l1_rows(N @ sp.diags(idf))
    net = consistency_profile(P, N, alpha)
    signal = (W @ _l1_rows(ri)).tocsr()          # recipe = mean of its ingredients
    n_ratings = np.asarray(n_ratings)
    confidence = n_ratings / (n_ratings + conf_k) if conf_k > 0 else np.ones(len(n_ratings))
    return RatingProfiles(np.asarray(user_ids), like.tocsr(), dislike.tocsr(), signal, confidence,
                          n_ratings, net)


@dataclass
class RatingData:
    """What it takes to build a rating profile for one user at request time."""
    ri: sp.csr_matrix               # recipes x ingredients (rows = recipes.parquet order)
    recipe_index: dict              # recipe_id -> row of ri
    idf: np.ndarray                 # (n_ingredients,)
    global_mean: float              # mean train rating
    train_ratings: sp.csr_matrix    # Food.com users x recipes, train ratings (0 = not rated)
    train_user_row: dict            # Food.com user_id -> row of train_ratings
    shrink_k: float = 5.0
    dislike_below_rating: float = 3.0
    conf_k: float = 10.0
    alpha: float = 2.0

    def ratings_of(self, user_id) -> list[tuple[int, float]]:
        """Train ratings of a Food.com user ([] if unknown)."""
        try:
            row = self.train_user_row.get(int(user_id))
        except (TypeError, ValueError):
            return []
        if row is None:
            return []
        r = self.train_ratings.getrow(row)
        recipe_ids = self._row_to_recipe[r.indices]
        return list(zip(recipe_ids.tolist(), r.data.tolist()))

    @property
    def _row_to_recipe(self) -> np.ndarray:
        if not hasattr(self, "_r2r"):
            arr = np.empty(len(self.recipe_index), dtype=np.int64)
            for rid, row in self.recipe_index.items():
                arr[row] = rid
            self._r2r = arr
        return self._r2r

    def profile_for(self, ratings) -> RatingProfiles | None:
        """One-row RatingProfiles from ``ratings`` [(recipe_id, rating 1-5)], None if empty.
        Same formula as build_rating_profiles, shrunk toward the train global mean; a later
        rating of the same recipe replaces the earlier one."""
        latest = {}
        for rid, rating in ratings or []:
            if rid in self.recipe_index:
                latest[int(rid)] = float(rating)
        if not latest:
            return None
        df = pd.DataFrame({"user_id": 0, "recipe_id": list(latest), "rating": list(latest.values())})
        user_ids, W, n_ratings = user_recipe_weights(df, self.recipe_index, self.ri.shape[0],
                                                     self.shrink_k, self.dislike_below_rating,
                                                     global_mean=self.global_mean)
        return profiles_from_weights(user_ids, W, n_ratings, self.ri, self.idf, self.conf_k,
                                     self.alpha)


def load_rating_data(processed_dir: str | Path, shrink_k: float = 5.0,
                     dislike_below_rating: float = 3.0, conf_k: float = 10.0,
                     use_idf: bool = True, alpha: float = 2.0) -> RatingData:
    processed_dir = Path(processed_dir)
    recipes = pd.read_parquet(processed_dir / "recipes.parquet", columns=["recipe_id", "is_train_recipe"])
    recipe_index = dict(zip(recipes["recipe_id"], range(len(recipes))))
    ri = sp.load_npz(processed_dir / "recipe_ingredient.npz").tocsr()
    train = pd.read_parquet(processed_dir / "interactions_train.parquet",
                            columns=["user_id", "recipe_id", "rating"])
    train_ri = ri[recipes["is_train_recipe"].to_numpy(dtype=bool)]
    df = np.asarray((train_ri > 0).sum(axis=0)).ravel().astype(np.float64)
    idf = np.log((train_ri.shape[0] + 1.0) / (df + 1.0)) if use_idf else np.ones_like(df)
    users = np.sort(train["user_id"].unique())
    user_row = {int(u): i for i, u in enumerate(users)}
    cols = train["recipe_id"].map(recipe_index)
    keep = cols.notna().to_numpy()
    rows = train["user_id"].map(user_row).to_numpy()[keep]
    R = sp.csr_matrix((train["rating"].to_numpy(float)[keep], (rows, cols[keep].astype(np.int64))),
                      shape=(len(users), ri.shape[0]))
    R.sum_duplicates()
    return RatingData(ri, recipe_index, idf, float(train["rating"].mean()), R, user_row,
                      shrink_k, dislike_below_rating, conf_k, alpha)
