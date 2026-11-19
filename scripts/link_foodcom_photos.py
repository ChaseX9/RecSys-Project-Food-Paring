"""Link our recipes to their own photo on Food.com (from the Kaggle "Food.com - Recipes and
Reviews" dataset by irkaal, whose RecipeId are the same Food.com ids as ours).

Download the dataset from https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews
(only recipes.parquet is needed), put it under data/raw/, then from the repo root:
    python scripts/link_foodcom_photos.py                       # finds data/raw/**/recipes.parquet
    python scripts/link_foodcom_photos.py --source path/to/recipes.parquet

Output: app/recipe_photos.parquet  (recipe_id, path), first photo of each recipe that has one.
``path`` is the part of the Food.com CDN URL after ``/v1/img/recipes/``; the app rebuilds the
URL with a small size (``PHOTO_URL`` in app/common.py).

The photos are NOT downloaded: they belong to the Food.com users who posted them, so the app
loads them from Food.com's CDN when a page shows them (needs internet; without it the recipe
falls back to the TheMealDB photo or the ingredient collage).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OURS = ROOT / "data" / "processed" / "recipes.parquet"
OUT = ROOT / "app" / "recipe_photos.parquet"
MARKER = "/v1/img/recipes/"


def first_photo_path(images) -> str | None:
    """CDN path of the first photo (``images`` is a list / array of URLs, or empty)."""
    try:
        urls = list(images)
    except TypeError:
        return None
    for url in urls:
        if isinstance(url, str) and MARKER in url:
            return url.split(MARKER, 1)[1]
    return None


def link(kaggle: pd.DataFrame, ours: pd.DataFrame) -> pd.DataFrame:
    """recipe_id -> photo path, for our recipes that have a photo on Food.com."""
    k = kaggle.dropna(subset=["RecipeId"]).assign(recipe_id=lambda d: d["RecipeId"].astype("int64"))
    k["path"] = k["Images"].map(first_photo_path)
    k = k.dropna(subset=["path"]).drop_duplicates("recipe_id")
    out = ours[["recipe_id"]].merge(k[["recipe_id", "path"]], on="recipe_id", how="inner")
    return out.sort_values("recipe_id").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", type=Path, help="Kaggle recipes.parquet (default: search data/raw)")
    args = p.parse_args()
    source = args.source or next(iter(sorted((ROOT / "data" / "raw").rglob("recipes.parquet"))), None)
    if source is None or not source.exists():
        sys.exit("Kaggle recipes.parquet not found: download it into data/raw/ (see the docstring).")

    kaggle = pd.read_parquet(source, columns=["RecipeId", "Images"])
    ours = pd.read_parquet(OURS, columns=["recipe_id"])
    out = link(kaggle, ours)
    out.to_parquet(OUT, index=False)
    print(f"{len(out):,} / {len(ours):,} recipes have a Food.com photo "
          f"({len(out) / len(ours):.0%}) -> {OUT.relative_to(ROOT)} "
          f"({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
