"""Match Food.com recipes to TheMealDB dish photos and download small thumbnails.

Run once (needs internet), from the repo root:
    python scripts/fetch_recipe_images.py            # match + download
    python scripts/fetch_recipe_images.py --no-download

Outputs:
    app/recipe_images.csv        recipe_id, image, meal (TheMealDB name), match ("exact" | "dish")
    app/images/recipes/*.webp    160 px thumbnails (one per TheMealDB dish, shared by recipes)

Matching is deliberately strict (a wrong photo is worse than none): after normalization
(lowercase, accents, plurals, filler words like "easy" / "best" / "crock pot" removed), a
recipe matches a dish if
  - exact: same words ("Beef Stroganov (Stroganoff)" -> "Beef stroganoff"), or
  - dish: every word of the dish is in the recipe title, the dish has 2+ words, and the dish
    contains the recipe's head noun (its last word outside parentheses), so "Pumpkin Pie Dip"
    or "Apple Pie Muffins" do not get an apple pie photo.
The most specific dish (most words) wins. Recipes without a photo get a collage of their
ingredient photos in the app (app/common.py: recipe_img).

Images: TheMealDB (https://www.themealdb.com), free API for educational / development use.
"""
from __future__ import annotations

import argparse
import html
import io
import re
import string
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import httpx
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "data" / "processed" / "recipes.parquet"
OUT_CSV = ROOT / "app" / "recipe_images.csv"
OUT_DIR = ROOT / "app" / "images" / "recipes"
SEARCH_URL = "https://www.themealdb.com/api/json/v1/1/search.php?f={}"
SIZE = 160

FILLER = set("""
recipe recipes easy best the and with my mom moms grandma grandmas homemade quick simple
delicious ever world famous style old fashioned perfect super ultimate yummy great good really
awesome classic traditional authentic healthy low fat lowfat skinny light our your from for
made scratch bake baked crock pot crockpot slow cooker oven copycat version kittencal kittencals
amazing fabulous wonderful excellent incredible fantastic favorite favourite special
""".split())
SYNONYMS = {"lasagne": "lasagna", "yoghurt": "yogurt", "aubergine": "eggplant",
            "courgette": "zucchini", "prawn": "shrimp", "chilli": "chili", "doughnut": "donut",
            "stroganov": "stroganoff", "tso": "tsos"}


def words(name: str) -> list[str]:
    """Normalized words of a title, in order (filler words and words < 3 letters dropped)."""
    s = unicodedata.normalize("NFKD", html.unescape(name)).encode("ascii", "ignore").decode().lower()
    out = []
    for w in re.findall(r"[a-z]+", s):
        w = SYNONYMS.get(w, w)
        if w in FILLER or len(w) < 3:
            continue
        # one stem for singular and plural: cookie / cookies -> cooky, berry / berries -> berry
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("ie") and len(w) > 3:
            w = w[:-2] + "y"
        elif w.endswith("es") and w[:-2].endswith(("ch", "sh", "ss", "x", "o")):
            w = w[:-2]
        elif w.endswith("s") and not w.endswith(("ss", "us")):
            w = w[:-1]
        out.append(SYNONYMS.get(w, w))
    return out


def head_noun(name: str) -> str | None:
    """Last normalized word of the title outside parentheses (the dish itself)."""
    ws = words(re.sub(r"\([^)]*\)", " ", html.unescape(name)))
    return ws[-1] if ws else None


def fetch_meals(client: httpx.Client) -> list[dict]:
    meals = {}
    for ch in string.ascii_lowercase + string.digits:
        data = client.get(SEARCH_URL.format(ch)).json()
        for m in data.get("meals") or []:
            meals[m["idMeal"]] = {"id": m["idMeal"], "name": m["strMeal"].strip(),
                                  "thumb": m["strMealThumb"]}
        time.sleep(0.1)
    return list(meals.values())


def match_recipes(recipes: pd.DataFrame, meals: list[dict]) -> pd.DataFrame:
    dishes = [(frozenset(words(m["name"])), m) for m in meals]
    dishes = [(t, m) for t, m in dishes if t]
    by_word = defaultdict(list)
    for t, m in dishes:
        for w in t:
            by_word[w].append((t, m))
    rows = []
    for rid, name in zip(recipes["recipe_id"], recipes["name"]):
        r = frozenset(words(name))
        head = head_noun(name)
        if not r or head is None:
            continue
        best, kind = None, None
        for t, m in by_word.get(head, ()):
            if t == r:
                cand = "exact"
            elif len(t) >= 2 and t <= r:
                cand = "dish"
            else:
                continue
            if best is None or len(t) > len(best[0]) or (cand == "exact" and kind != "exact"):
                best, kind = (t, m), cand
        if best:
            rows.append({"recipe_id": int(rid), "meal_id": best[1]["id"], "meal": best[1]["name"],
                         "thumb": best[1]["thumb"], "match": kind})
    return pd.DataFrame(rows)


def download(client: httpx.Client, url: str, dest: Path) -> bool:
    for suffix in ("/preview", ""):  # TheMealDB serves a 250 px preview of every photo
        try:
            r = client.get(url + suffix)
            if r.status_code != 200:
                continue
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img.thumbnail((SIZE, SIZE))
            img.save(dest, "WEBP", quality=80)
            return True
        except (httpx.HTTPError, OSError):
            continue
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--no-download", action="store_true", help="only write the mapping")
    args = p.parse_args()

    recipes = pd.read_parquet(RECIPES, columns=["recipe_id", "name"])
    with httpx.Client(timeout=30) as client:
        meals = fetch_meals(client)
        print(f"TheMealDB: {len(meals)} dishes")
        matches = match_recipes(recipes, meals)
        print(f"Matched {len(matches):,} / {len(recipes):,} recipes "
              f"({matches['match'].value_counts().to_dict()}) to "
              f"{matches['meal_id'].nunique()} dishes")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        ok = set()
        if not args.no_download:
            for k, (mid, thumb) in enumerate(matches.drop_duplicates("meal_id")[["meal_id", "thumb"]]
                                             .itertuples(index=False), 1):
                dest = OUT_DIR / f"{mid}.webp"
                if dest.exists() or download(client, thumb, dest):
                    ok.add(mid)
                if k % 50 == 0:
                    print(f"  {k} photos")
                time.sleep(0.05)
        else:
            ok = {p.stem for p in OUT_DIR.glob("*.webp")}
    matches = matches[matches["meal_id"].isin(ok)]
    matches["image"] = "images/recipes/" + matches["meal_id"] + ".webp"
    matches[["recipe_id", "image", "meal", "match"]].sort_values("recipe_id").to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV.relative_to(ROOT)} ({len(matches):,} recipes with a photo)")


if __name__ == "__main__":
    main()
