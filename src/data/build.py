from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from .flavor import build_flavor_features, load_flavorgraph
from .ingredients import load_aliases, normalize_recipe
from .profiles import build_recipe_ingredient_matrix, build_user_profiles


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# recipes
def build_recipes_and_vocab(cfg: dict, root: Path, stats: dict):
    raw = root / cfg["paths"]["raw_dir"] / cfg["paths"]["recipes_file"]
    df = pd.read_csv(raw, usecols=["RecipeId", "Name", "RecipeIngredientParts"])
    stats["recipes_raw"] = len(df)
    aliases = load_aliases(root / cfg["paths"]["aliases_file"])

    df["ingredients"] = [normalize_recipe(v, aliases) for v in df["RecipeIngredientParts"]]
    stop = set(cfg["ingredients"]["stop_ingredients"])
    df["ingredients"] = [[i for i in ing if i not in stop] for ing in df["ingredients"]]

    # Vocabulary by recipe frequency (rule-based cleaning, no train/test information)
    counts = Counter(i for ing in df["ingredients"] for i in ing)
    stats["ingredients_distinct_before_filter"] = len(counts)
    kept = sorted((i for i, c in counts.items() if c >= cfg["ingredients"]["min_recipes"]),
                  key=lambda i: (-counts[i], i))
    vocab = {name: idx for idx, name in enumerate(kept)}
    stats["ingredients_vocab_size"] = len(vocab)

    df["ingredient_ids"] = [[vocab[i] for i in ing if i in vocab] for ing in df["ingredients"]]
    n = df["ingredient_ids"].map(len)
    mask = (n >= cfg["recipes"]["min_ingredients"]) & (n <= cfg["recipes"]["max_ingredients"])
    df = df[mask].reset_index(drop=True)
    stats["recipes_kept"] = len(df)

    recipes = pd.DataFrame({
        "recipe_id": df["RecipeId"].astype("int64"),
        "name": df["Name"],
        "ingredient_ids": [np.asarray(x, dtype=np.int32) for x in df["ingredient_ids"]],
    })
    vocab_df = pd.DataFrame({
        "ingredient_id": list(vocab.values()), "name": list(vocab.keys()),
        "n_recipes": [counts[k] for k in vocab],
    })
    return recipes, vocab_df


# interactions
def build_interactions(cfg: dict, root: Path, recipe_ids: set[int], stats: dict) -> pd.DataFrame:
    raw = root / cfg["paths"]["raw_dir"] / cfg["paths"]["reviews_file"]
    df = pd.read_csv(raw, usecols=["ReviewId", "RecipeId", "AuthorId", "Rating", "DateSubmitted"])
    df.columns = ["review_id", "recipe_id", "user_id", "rating", "timestamp"]
    stats["interactions_raw"] = len(df)
    stats["rating_distribution_raw"] = {str(k): int(v) for k, v in
                                        df["rating"].value_counts().sort_index().items()}

    if cfg["interactions"]["drop_rating_zero"]:
        df = df[df["rating"] > 0]
    stats["interactions_after_drop_zero"] = len(df)
    df = df[df["recipe_id"].isin(recipe_ids)]
    stats["interactions_on_kept_recipes"] = len(df)
    df = df.drop_duplicates(["user_id", "recipe_id"], keep="last")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Iterative k-core (users, then recipes) until stable
    u_min = cfg["interactions"]["min_user_interactions"]
    r_min = cfg["interactions"]["min_recipe_interactions"]
    while True:
        before = len(df)
        df = df[df.groupby("user_id")["recipe_id"].transform("size") >= u_min]
        df = df[df.groupby("recipe_id")["user_id"].transform("size") >= r_min]
        if len(df) == before:
            break
    stats["interactions_after_kcore"] = len(df)
    return df.reset_index(drop=True)


def leave_last_out_split(df: pd.DataFrame) -> pd.DataFrame:
    """Per user: last interaction -> test, second to last -> val, the rest -> train.

    Ties on the timestamp are broken by review_id.
    """
    df = df.sort_values(["user_id", "timestamp", "review_id"]).reset_index(drop=True)
    rank_from_end = df.groupby("user_id").cumcount(ascending=False)
    df["split"] = np.where(rank_from_end == 0, "test", np.where(rank_from_end == 1, "val", "train"))
    return df


def mark_train_recipes(recipes: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """Add is_train_recipe: True if the recipe has at least one train interaction.

    Graph statistics (e.g. NPMI) must only use these recipes, otherwise recipes that appear
    only in val/test leak into the graph.
    """
    train_ids = set(interactions.loc[interactions["split"] == "train", "recipe_id"])
    out = recipes.copy()
    out["is_train_recipe"] = out["recipe_id"].isin(train_ids)
    return out


# main
def run(config_path: str | Path = "configs/data.yaml", root: str | Path = ".") -> dict:
    root = Path(root).resolve()
    cfg = load_config(root / config_path)
    out = root / cfg["paths"]["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    stats: dict = {}

    print("[1/6] Recipes and ingredients...")
    recipes, vocab_df = build_recipes_and_vocab(cfg, root, stats)
    print(f"      {stats['recipes_kept']:,} recipes, {stats['ingredients_vocab_size']:,} ingredients")

    print("[2/6] Interactions...")
    inter = build_interactions(cfg, root, set(recipes["recipe_id"]), stats)

    # Keep only recipes that appear in the final interactions
    recipes = recipes[recipes["recipe_id"].isin(set(inter["recipe_id"]))].reset_index(drop=True)
    stats["recipes_with_interactions"] = len(recipes)

    print("[3/6] Leave-last-out split...")
    inter = leave_last_out_split(inter)
    stats["split_sizes"] = {k: int(v) for k, v in inter["split"].value_counts().items()}
    stats["n_users"] = int(inter["user_id"].nunique())
    stats["n_recipes"] = int(inter["recipe_id"].nunique())
    stats["sparsity"] = float(1 - len(inter) / (stats["n_users"] * stats["n_recipes"]))
    stats["positive_share_train"] = float(
        (inter.loc[inter["split"] == "train", "rating"] > cfg["profiles"]["positive_rating_gt"]).mean())

    recipes = mark_train_recipes(recipes, inter)
    stats["train_recipes"] = int(recipes["is_train_recipe"].sum())
    stats["recipes_only_in_val_test"] = int((~recipes["is_train_recipe"]).sum())

    print("[4/6] User profiles (train only)...")
    n_ing = len(vocab_df)
    recipe_ing = build_recipe_ingredient_matrix(recipes, n_ing)
    recipe_index = {rid: i for i, rid in enumerate(recipes["recipe_id"])}
    train = inter[inter["split"] == "train"]
    user_ids, profiles = build_user_profiles(train, recipe_index, recipe_ing,
                                             cfg["profiles"]["positive_rating_gt"])
    stats["users_with_empty_profile"] = int((np.asarray(profiles.sum(axis=1)).ravel() == 0).sum())

    print("[5/6] FlavorGraph features...")
    fg_cfg = cfg.get("flavorgraph", {})
    fg_dir = root / fg_cfg.get("dir", "data/raw/flavorgraph")
    fg_ready = fg_cfg.get("enabled", False) and (fg_dir / fg_cfg["nodes_file"]).exists()

    print("[6/6] Writing to", out)
    recipes.to_parquet(out / "recipes.parquet", index=False)
    vocab_df.to_csv(out / "ingredients_vocab.csv", index=False)
    for name in ("train", "val", "test"):
        inter[inter["split"] == name].drop(columns="split").to_parquet(out / f"interactions_{name}.parquet", index=False)
    sp.save_npz(out / "recipe_ingredient.npz", recipe_ing)       # rows follow the order of recipes.parquet
    sp.save_npz(out / "user_profiles.npz", profiles)             # rows follow the order of user_profile_ids.npy
    np.save(out / "user_profile_ids.npy", user_ids)
    if fg_ready:
        nodes, edges = load_flavorgraph(fg_dir, fg_cfg["nodes_file"], fg_cfg["edges_file"])
        stats["flavorgraph"] = build_flavor_features(
            vocab_df, nodes, edges, load_aliases(root / cfg["paths"]["aliases_file"]), out)
    else:
        print("      FlavorGraph files not found, skipping (plan B: co-occurrence only)")
        stats["flavorgraph"] = None
    stats["top_ingredients"] = vocab_df.head(30)[["name", "n_recipes"]].to_dict("records")
    stats["rarest_kept_ingredients"] = vocab_df.tail(15)[["name", "n_recipes"]].to_dict("records")
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return stats
