"""Evaluation protocol: masked-ingredient completion.

Our model recommends ingredients from ingredients, while the held-out data is
user -> recipe interactions. We bridge the two by hiding part of a held-out recipe's
ingredients: the remaining ingredients are the query, the hidden ones are the ground
truth, and the user who interacted with the recipe is passed for personalization.

Two protocols, both leakage-free:

- ``unseen_recipes`` (option a, default): cases come from val/test interactions whose
  recipe has NO train interaction (``is_train_recipe == False``). The graph, NPMI,
  popularity and user profiles are all built on train recipes only, so the model has
  never seen these recipes. Small but clean (4,426 such recipes, shared by val and test).

- ``recipe_holdout`` (option b): a random fraction of ALL recipes is held out at recipe
  level. Cases are one per held-out recipe, without a user (pure pairing quality). The
  returned ``train_recipe_mask`` excludes the held-out recipes, and the graph/model must be
  rebuilt with that same mask (see ``holdout_recipe_ids``) for the comparison to be fair.
  User profiles are not rebuilt, which is why this protocol does not pass a user_id.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROTOCOLS = ("unseen_recipes", "recipe_holdout")


@dataclass
class EvalSetup:
    cases: pd.DataFrame            # case_id, user_id, recipe_id, query_ids, target_ids
    train_recipe_mask: np.ndarray  # bool, aligned with recipes.parquet rows: what models may see
    protocol: str
    split: str


def mask_ingredients(ingredient_ids: Sequence[int], rng: np.random.Generator,
                     hidden_ratio: float = 0.2, min_query: int = 2):
    """Split one recipe into (query, target). Returns None if the recipe is too short.

    At least one ingredient is hidden and at least ``min_query`` stay in the query.
    """
    ids = np.asarray(ingredient_ids, dtype=np.int64)
    n = len(ids)
    n_hidden = max(1, int(round(hidden_ratio * n)))
    if n - n_hidden < min_query:
        return None
    perm = rng.permutation(n)
    target = sorted(int(x) for x in ids[perm[:n_hidden]])
    query = sorted(int(x) for x in ids[perm[n_hidden:]])
    return query, target


def holdout_recipe_ids(recipes: pd.DataFrame, frac: float, seed: int) -> np.ndarray:
    """Deterministic set of held-out recipe ids for protocol b.

    The graph builder must exclude these recipes with the same (frac, seed).
    """
    rng = np.random.default_rng(seed)
    ids = np.sort(recipes["recipe_id"].to_numpy())
    n = int(round(frac * len(ids)))
    return np.sort(rng.choice(ids, size=n, replace=False))


def _make_cases(rows: pd.DataFrame, recipe_ings: dict, hidden_ratio: float,
                min_query: int, seed: int, max_cases: int | None) -> pd.DataFrame:
    rows = rows.sort_values(["recipe_id", "user_id"], na_position="first").reset_index(drop=True)
    if max_cases is not None and len(rows) > max_cases:
        rows = rows.sample(n=max_cases, random_state=seed).sort_index()
    rng = np.random.default_rng(seed)
    out = []
    for user_id, recipe_id in zip(rows["user_id"], rows["recipe_id"]):
        split = mask_ingredients(recipe_ings[recipe_id], rng, hidden_ratio, min_query)
        if split is None:
            continue
        out.append({"user_id": user_id, "recipe_id": int(recipe_id),
                    "query_ids": split[0], "target_ids": split[1]})
    cases = pd.DataFrame(out, columns=["user_id", "recipe_id", "query_ids", "target_ids"])
    cases.insert(0, "case_id", np.arange(len(cases)))
    return cases


def build_cases(processed_dir: str | Path, split: str = "test", protocol: str = "unseen_recipes",
                hidden_ratio: float = 0.2, min_query: int = 2, seed: int = 42,
                max_cases: int | None = None, holdout_frac: float = 0.05) -> EvalSetup:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}, expected one of {PROTOCOLS}")
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    processed_dir = Path(processed_dir)
    recipes = pd.read_parquet(processed_dir / "recipes.parquet")
    if "is_train_recipe" not in recipes.columns:
        raise ValueError("recipes.parquet has no is_train_recipe column: rerun scripts/build_data.py")
    recipe_ings = dict(zip(recipes["recipe_id"], recipes["ingredient_ids"]))
    is_train = recipes["is_train_recipe"].to_numpy(dtype=bool)

    if protocol == "unseen_recipes":
        inter = pd.read_parquet(processed_dir / f"interactions_{split}.parquet",
                                columns=["user_id", "recipe_id"])
        unseen = set(recipes.loc[~is_train, "recipe_id"])
        rows = inter[inter["recipe_id"].isin(unseen)]
        mask = is_train
    else:
        held = holdout_recipe_ids(recipes, holdout_frac, seed)
        # val and test get disjoint halves of the held-out recipes
        half = len(held) // 2
        chosen = held[:half] if split == "val" else held[half:]
        rows = pd.DataFrame({"user_id": [None] * len(chosen), "recipe_id": chosen})
        mask = is_train & ~recipes["recipe_id"].isin(set(held)).to_numpy()

    cases = _make_cases(rows, recipe_ings, hidden_ratio, min_query, seed, max_cases)
    return EvalSetup(cases=cases, train_recipe_mask=mask, protocol=protocol, split=split)
