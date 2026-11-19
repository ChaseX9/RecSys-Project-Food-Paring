"""Match our ingredient vocab to TheMealDB's ingredient photos and download small thumbnails.

Run once (needs internet), from the repo root:
    python scripts/fetch_ingredient_images.py            # match + download
    python scripts/fetch_ingredient_images.py --no-download

Outputs:
    app/ingredient_images.csv      name, image, match ("exact" | "simplified" | "head"), source
    app/images/ingredients/*.webp  96 px thumbnails (shared by every vocab name mapped to them)

Images: TheMealDB (https://www.themealdb.com), free API for educational / development use.
Check the "head" matches in the CSV: they are guesses (e.g. "smoked trout fillet" -> trout).
"""
from __future__ import annotations

import argparse
import io
import re
import time
from pathlib import Path

import httpx
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VOCAB = ROOT / "data" / "processed" / "ingredients_vocab.csv"
OUT_CSV = ROOT / "app" / "ingredient_images.csv"
OUT_DIR = ROOT / "app" / "images" / "ingredients"
LIST_URL = "https://www.themealdb.com/api/json/v1/1/list.php?i=list"
IMG_URL = "https://www.themealdb.com/images/ingredients/{}-small.png"
SIZE = 96

# Words that describe a state or a cut, not the ingredient itself.
QUALIFIERS = set("""
fresh freshly dried dry ground chopped minced diced sliced grated shredded crushed whole large
small medium extra virgin light low fat low-fat reduced-fat fat-free nonfat skim lean boneless
skinless raw cooked frozen canned plain unsalted salted sweetened unsweetened pure granulated
packed fine coarse finely hot cold warm soft softened melted toasted roasted baby mini organic
instant quick prepared refrigerated seasoned regular natural sodium reduced-sodium fillets fillet
""".split())

# US names in Food.com vs UK names in TheMealDB, plus a few common synonyms.
ALIASES = {
    "eggplant": "aubergine", "scallion": "spring onions", "green onion": "spring onions",
    "powdered sugar": "icing sugar", "confectioners sugar": "icing sugar",
    "all-purpose flour": "plain flour", "flour": "plain flour", "cornstarch": "cornflour",
    "baking soda": "bicarbonate of soda", "heavy cream": "double cream",
    "whipping cream": "double cream", "ground beef": "minced beef", "shrimp": "prawns",
    "garbanzo bean": "chickpeas", "bell pepper": "red pepper", "green bell pepper": "green pepper",
    "red bell pepper": "red pepper", "yellow bell pepper": "yellow pepper",
    "chile": "chilli", "chili powder": "chilli powder", "red chili pepper": "red chilli",
    "jalapeno": "jalapeno", "arugula": "rocket", "zucchini": "courgettes",
    "beet": "beetroot", "molasses": "black treacle", "ketchup": "tomato ketchup",
    "chicken broth": "chicken stock", "beef broth": "beef stock",
    "vegetable broth": "vegetable stock", "fish broth": "fish stock",
    "tomato paste": "tomato puree", "corn": "sweetcorn", "green chily": "green chilli",
    "red chily": "red chilli", "chily": "chilli", "half-and-half": "single cream",
    "chicken breast half": "chicken breast", "pecan": "pecan nuts", "pineapple": "pineapple chunks",
    "celery rib": "celery", "basil leaf": "basil leaves", "sour cream": "sour cream",
    "margarine": "butter", "vegetable shortening": "butter", "shortening": "butter",
    "cayenne": "cayenne pepper", "barbecue sauce": "barbeque sauce", "pasta": "penne pasta",
    "gingerroot": "ginger", "broccoli floret": "broccoli", "dill weed": "dill",
    "parsley flake": "parsley", "corn kernel": "sweetcorn", "kernel corn": "sweetcorn",
    "lemon rind": "lemon zest", "lemon peel": "lemon zest", "orange rind": "orange zest",
    "orange peel": "orange zest", "lime rind": "lime zest", "kalamata olive": "black olives",
    "soymilk": "soya milk", "corn syrup": "golden syrup", "light corn syrup": "golden syrup",
    "applesauce": "apple sauce", "bittersweet chocolate": "dark chocolate",
    "semi-sweet chocolate chip": "chocolate chips", "sherry wine": "sherry",
    "tortilla chip": "tortilla chips", "chili sauce": "sweet chilli sauce",
    "date": "medjool dates", "cool whip": "whipped cream", "splenda sugar substitute": "sugar",
    "artichoke heart": "artichoke", "cauliflower": "cauliflower florets",
}

# Trailing words we can drop to reach a more specific name ("feta cheese" -> "feta",
# better than the head-noun guess "cheese").
GENERIC_TAILS = {"cheese", "leaf", "rib", "stalk", "sprig", "half", "piece", "nut"}


def norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 -]", " ", s.lower().replace("&", " and "))
    return re.sub(r"\s+", " ", s).strip()


def singular(word: str) -> str:
    for suffix, repl in (("ies", "y"), ("oes", "o"), ("ches", "ch"), ("shes", "sh"), ("s", "")):
        if word.endswith(suffix) and len(word) > len(suffix) + 2 and not word.endswith("ss"):
            return word[: -len(suffix)] + repl
    return word


def key(s: str) -> str:
    """Comparison key: normalised, every word singular."""
    return " ".join(singular(w) for w in norm(s).split())


def build_index(meal_names: list[str]) -> dict[str, str]:
    index: dict[str, str] = {}
    for name in meal_names:
        index.setdefault(key(name), name)
    return index


def match(name: str, index: dict[str, str]) -> tuple[str, str] | None:
    """(TheMealDB name, how) for one vocab name, or None."""
    aliases = {key(a): key(b) for a, b in ALIASES.items()}

    def lookup(k: str) -> str | None:
        return index.get(k) or index.get(aliases.get(k, ""))

    # "tuna in water", "chicken with skin": the ingredient is what comes before
    k = re.split(r" (?:in|with|packed in) ", " " + key(name))[0].strip() or key(name)
    if hit := lookup(k):
        return hit, "exact"
    words = [w for w in k.split() if w not in QUALIFIERS]
    simple = " ".join(words)
    if simple and simple != k and (hit := lookup(simple)):
        return hit, "simplified"
    if len(words) > 1 and words[-1] in GENERIC_TAILS and (hit := lookup(" ".join(words[:-1]))):
        return hit, "simplified"
    for start in range(1, len(words)):  # longest ending first: "smoked trout" -> "trout"
        if hit := lookup(" ".join(words[start:])):
            return hit, "head"
    return None


def slug(meal_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", meal_name.lower()).strip("_")


def download(meal_names: set[str]) -> set[str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok: set[str] = set()
    with httpx.Client(timeout=20, follow_redirects=True) as client:
        for i, name in enumerate(sorted(meal_names), 1):
            out = OUT_DIR / f"{slug(name)}.webp"
            if out.exists():
                ok.add(name)
                continue
            try:
                r = client.get(IMG_URL.format(name.replace(" ", "%20")))
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert("RGBA")
                img.thumbnail((SIZE, SIZE))
                img.save(out, "WEBP", quality=80)
                ok.add(name)
            except Exception as e:  # missing image on their side: fall back to an emoji in the app
                print(f"  ! {name}: {e}")
            if i % 50 == 0:
                print(f"  {i}/{len(meal_names)} images")
            time.sleep(0.05)  # be gentle with a free service
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    meal_names = [m["strIngredient"] for m in httpx.get(LIST_URL, timeout=20).json()["meals"]]
    vocab = pd.read_csv(VOCAB)
    counts = dict(zip(vocab["name"], vocab.get("n_recipes", pd.Series(1, index=vocab.index))))

    # Some TheMealDB ingredients have no photo (404): drop them and match again, so that
    # e.g. "ground black pepper" falls back to another name instead of having no image.
    no_photo: set[str] = set()
    while True:
        index = build_index([n for n in meal_names if n not in no_photo])
        rows = []
        for name in vocab["name"].astype(str):
            if m := match(name, index):
                rows.append({"name": name, "meal": m[0], "match": m[1]})
        found = {r["meal"] for r in rows}
        available = found if args.no_download else download(found)
        if available == found:
            break
        no_photo |= found - available

    rows = [{"name": r["name"], "image": f"images/ingredients/{slug(r['meal'])}.webp",
             "match": r["match"], "source": r["meal"]} for r in rows]
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    total = sum(counts.values())
    covered = sum(counts[r["name"]] for r in rows)
    by = pd.Series([r["match"] for r in rows]).value_counts().to_dict()
    print(f"\n{len(rows)}/{len(vocab)} ingredients have a photo ({by}), "
          f"{len(available)} distinct images")
    print(f"weighted by recipe count: {covered / total:.1%} of ingredient uses are covered")
    print(f"-> {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
