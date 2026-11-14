from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def build_recipe_ingredient_matrix(recipes: pd.DataFrame, n_ingredients: int) -> sp.csr_matrix:
    """Binary recipes x ingredients matrix (row i = recipes.iloc[i])."""
    indptr = np.cumsum([0] + [len(x) for x in recipes["ingredient_ids"]])
    indices = np.concatenate([np.asarray(x, dtype=np.int64) for x in recipes["ingredient_ids"]])
    data = np.ones(len(indices), dtype=np.float32)
    return sp.csr_matrix((data, indices, indptr), shape=(len(recipes), n_ingredients))


def build_user_profiles(train: pd.DataFrame, recipe_index: dict[int, int],
                        recipe_ing: sp.csr_matrix, positive_rating_gt: float):
    """Return (user_ids, L1-normalized users x ingredients profiles).

    Only train interactions are used, so val/test never leak into the profiles.
    """
    pos = train[train["rating"] > positive_rating_gt]
    user_ids = np.sort(train["user_id"].unique())  # every train user, even without positives
    u_idx = {u: i for i, u in enumerate(user_ids)}
    rows = pos["user_id"].map(u_idx).to_numpy()
    cols = pos["recipe_id"].map(recipe_index).to_numpy()
    ur = sp.csr_matrix((np.ones(len(pos), dtype=np.float32), (rows, cols)),
                       shape=(len(user_ids), recipe_ing.shape[0]))
    prof = (ur @ recipe_ing).tocsr()
    norm = np.asarray(prof.sum(axis=1)).ravel()
    norm[norm == 0] = 1.0  # empty profiles stay at zero
    prof = sp.diags(1.0 / norm) @ prof
    return user_ids, prof.tocsr()
