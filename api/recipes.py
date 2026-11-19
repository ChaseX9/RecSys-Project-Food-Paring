"""Recipe ratings given in the app, and the recipe catalog used to ask for them.

Why: the personalization (src/data/rating_profiles.py) learns a user's taste from recipes rated
1-5. Food.com users have that history; app users get it by rating recipes here. Their ratings
are stored in ``recipe_ratings`` and folded into the profile on every /recommend (no retraining).

Catalog (Food.com recipes, data/processed/recipes.parquet):
- search: every word of the query in the recipe name, most rated first;
- onboarding: popular recipes that are as different as possible from each other
  (greedy on ingredient Jaccard), so that 10 ratings already cover many tastes;
- next: one recipe to rate, chosen among popular ones for the ingredients it would teach
  about (not yet seen in the user's rated recipes), randomized so each visit differs.
Recipes containing an ingredient banned by the user's diet / allergies are never proposed. The
recipe name is checked too: the cleaned ingredient lists sometimes lose the main ingredient
("Pulled Pork (Crock Pot)" = onion, ginger ale, barbecue sauce).
"""
from __future__ import annotations

import html
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .db import connect

SCHEMA = """
CREATE TABLE IF NOT EXISTS recipe_ratings (
    user_id    TEXT    NOT NULL,
    recipe_id  INTEGER NOT NULL,
    rating     INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    created_at TEXT    NOT NULL,   -- ISO 8601, UTC (last time the rating was set)
    PRIMARY KEY (user_id, recipe_id)
);
"""
ONBOARDING_SIZE = 10
POOL_ONBOARDING = 400     # most rated recipes considered for the onboarding set
POOL_NEXT = 3000          # most rated recipes considered for "next recipe to rate"
MIN_INGREDIENTS = 3


# ------------------------------------------------------------------ storage
def init_db(db_path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def rate(db_path, user_id: str, recipe_id: int, rating: int) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO recipe_ratings (user_id, recipe_id, rating, created_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (user_id, recipe_id) DO UPDATE SET rating = excluded.rating, "
            "created_at = excluded.created_at",
            (user_id, int(recipe_id), int(rating), datetime.now(timezone.utc).isoformat()))


def unrate(db_path, user_id: str, recipe_id: int) -> bool:
    with connect(db_path) as conn:
        cur = conn.execute("DELETE FROM recipe_ratings WHERE user_id = ? AND recipe_id = ?",
                           (user_id, int(recipe_id)))
        return cur.rowcount > 0


def ratings_of(db_path, user_id: str) -> list[dict]:
    """The user's ratings, most recent first."""
    with connect(db_path) as conn:
        rows = conn.execute("SELECT recipe_id, rating, created_at FROM recipe_ratings WHERE user_id = ? "
                            "ORDER BY created_at DESC, recipe_id", (user_id,)).fetchall()
    return [dict(r) for r in rows]


def rating_pairs(db_path, user_id: str) -> list[tuple[int, int]]:
    """[(recipe_id, rating)] for the ranker."""
    return [(r["recipe_id"], r["rating"]) for r in ratings_of(db_path, user_id)]


def delete_user_ratings(conn, user_id: str) -> None:
    conn.execute("DELETE FROM recipe_ratings WHERE user_id = ?", (user_id,))


# ------------------------------------------------------------------ catalog
# Recipe titles name dishes, not ingredients: the ranker's whole-word ingredient keywords miss
# "Porketta", "BBQ Ribs" or "Buffalo Wings". These title rules are applied on top of them.
_MEAT_TITLE = re.compile(
    r"\b(?:pork|chicken|beef|lamb|veal|turkey|bacon|sausage|steak|meat|venison|duck|goose|quail"
    r"|brisket|jerky|chorizo|pepperoni|salami|prosciutto|pancetta|carnitas|gyro)"
    r"|\b(?:ham|hams|ribs?|ribeye|riblets|wings?|burgers?|hamburgers?|cheeseburgers?"
    r"|sloppy joes?|pot roast|chili con carne)\b")
_FISH_TITLE = re.compile(
    r"\b(?:fish|salmon|tuna|shrimp|prawn|crab|lobster|tilapia|scallop|clam|mussel|oyster|anchov"
    r"|sardine|trout|halibut|catfish|crawfish|calamari|squid|octopus|seafood|sushi)"
    r"|\bcod\b")
_TITLE_WORDS_IGNORED = {"with", "recipe", "easy", "best", "oven", "crock", "style", "homemade",
                        "quick", "simple", "delicious", "ever", "world", "famous", "baked", "the",
                        "and", "your", "great", "good", "yummy", "super", "really", "awesome"}


def title_ban(diet: str | None) -> re.Pattern | None:
    """Extra rule on recipe titles for a diet (meat for every diet, fish except pescatarian)."""
    d = (diet or "").lower()
    if d in ("vegetarian", "vegan"):
        return re.compile(_MEAT_TITLE.pattern + "|" + _FISH_TITLE.pattern)
    if d == "pescatarian":
        return _MEAT_TITLE
    return None


@dataclass
class Ban:
    """What a user must never be offered: ingredient ids, a regex for the recipe name (same
    keywords as the ranker) and extra title rules for the diet (``title_ban``)."""
    ingredients: set[int] = field(default_factory=set)
    name: re.Pattern | None = None
    title: re.Pattern | None = None


NO_BAN = Ban()


def _title_words(lower_name: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", lower_name) if w not in _TITLE_WORDS_IGNORED}


class RecipeCatalog:
    """Recipes with their ingredients and Food.com popularity (train ratings)."""

    def __init__(self, recipes: pd.DataFrame, vocab: list[str], counts: pd.Series,
                 means: pd.Series):
        """``recipes``: recipe_id, name, ingredient_ids (vocab ids); ``counts`` / ``means``:
        number and mean of train ratings per recipe_id."""
        self.vocab = list(vocab)
        self.ids = recipes["recipe_id"].to_numpy(np.int64)
        self.names = [html.unescape(str(n)).strip() for n in recipes["name"]]
        self.lower = [n.lower() for n in self.names]
        self.ingredients = [np.asarray(x, dtype=np.int64) for x in recipes["ingredient_ids"]]
        self.row = {int(r): i for i, r in enumerate(self.ids)}
        self.n_ratings = counts.reindex(self.ids).fillna(0).to_numpy(np.int64)
        self.mean_rating = means.reindex(self.ids).to_numpy(float)
        ok = np.array([len(x) >= MIN_INGREDIENTS for x in self.ingredients])
        order = np.argsort(-self.n_ratings, kind="stable")
        self.popular = [int(i) for i in order if ok[i] and self.n_ratings[i] > 0]

    @classmethod
    def load(cls, processed_dir: str | Path) -> "RecipeCatalog | None":
        processed_dir = Path(processed_dir)
        paths = [processed_dir / f for f in ("recipes.parquet", "ingredients_vocab.csv",
                                            "interactions_train.parquet")]
        if not all(p.exists() for p in paths):
            return None
        recipes = pd.read_parquet(paths[0], columns=["recipe_id", "name", "ingredient_ids"])
        vocab = pd.read_csv(paths[1]).sort_values("ingredient_id")["name"].tolist()
        train = pd.read_parquet(paths[2], columns=["recipe_id", "rating"])
        g = train.groupby("recipe_id")["rating"]
        return cls(recipes, vocab, g.count(), g.mean())

    # ---------------------------------------------------------------- helpers
    def __contains__(self, recipe_id) -> bool:
        return int(recipe_id) in self.row

    def card(self, i: int, user_rating: int | None = None) -> dict:
        mean = self.mean_rating[i]
        return {"recipe_id": int(self.ids[i]), "name": self.names[i],
                "ingredients": [self.vocab[j] for j in self.ingredients[i]],
                "n_ratings": int(self.n_ratings[i]),
                "mean_rating": None if np.isnan(mean) else round(float(mean), 2),
                "user_rating": user_rating}

    def _allowed(self, i: int, ban: Ban) -> bool:
        for pattern in (ban.name, ban.title):
            if pattern is not None and pattern.search(self.lower[i]):
                return False
        return not ban.ingredients or not any(int(j) in ban.ingredients for j in self.ingredients[i])

    def _rows(self, recipe_ids) -> set[int]:
        return {self.row[int(r)] for r in recipe_ids if int(r) in self.row}

    # ---------------------------------------------------------------- queries
    def search(self, query: str, limit: int = 20, ban: Ban = NO_BAN,
               user_ratings: dict | None = None) -> list[dict]:
        words = [w for w in query.lower().split() if w]
        if not words:
            return []
        hits = [i for i, n in enumerate(self.lower) if all(w in n for w in words)
                and self._allowed(i, ban)]
        hits.sort(key=lambda i: (-self.n_ratings[i], self.names[i]))
        ur = user_ratings or {}
        return [self.card(i, ur.get(int(self.ids[i]))) for i in hits[:limit]]

    def onboarding(self, ban: Ban = NO_BAN, rated=(), n: int = ONBOARDING_SIZE) -> list[dict]:
        """Popular and varied: greedy, each pick far from the ones already picked (Jaccard on
        ingredients and on title words: the title often names the main ingredient)."""
        skip = self._rows(rated)
        pool = [i for i in self.popular[:POOL_ONBOARDING * 3]
                if i not in skip and self._allowed(i, ban)][:POOL_ONBOARDING]
        if not pool:
            return []
        sets = {i: set(self.ingredients[i].tolist()) for i in pool}
        words = {i: _title_words(self.lower[i]) for i in pool}

        def jaccard(a: set, b: set) -> float:
            return len(a & b) / len(a | b) if a and b else 0.0

        pop = np.log1p(self.n_ratings[pool].astype(float))
        pop = dict(zip(pool, (pop - pop.min()) / (pop.max() - pop.min() or 1.0)))
        picked = [pool[0]]
        while len(picked) < min(n, len(pool)):
            def score(i):
                closest = max(max(jaccard(sets[i], sets[p]), jaccard(words[i], words[p]))
                              for p in picked)
                return 0.7 * (1.0 - closest) + 0.3 * pop[i]
            picked.append(max((i for i in pool if i not in picked), key=score))
        return [self.card(i) for i in picked]

    def next_to_rate(self, ban: Ban = NO_BAN, rated=(), skip=(),
                     seed: int | None = None, top: int = 30) -> dict | None:
        """One recipe to rate: popular, made of ingredients the user's ratings don't cover yet."""
        rated_rows = self._rows(rated)
        skip_rows = self._rows(skip) | rated_rows
        seen = set()
        for i in rated_rows:
            seen.update(self.ingredients[i].tolist())
        pool = [i for i in self.popular[:POOL_NEXT] if i not in skip_rows
                and self._allowed(i, ban)]
        if not pool:
            return None
        pop = np.log1p(self.n_ratings[pool].astype(float))
        pop = (pop - pop.min()) / (pop.max() - pop.min() or 1.0)
        novelty = np.array([np.mean([j not in seen for j in self.ingredients[i].tolist()])
                            for i in pool])
        score = 0.6 * novelty + 0.4 * pop
        best = [pool[k] for k in np.argsort(-score, kind="stable")[:top]]
        i = random.Random(seed).choice(best)
        card = self.card(i)
        card["new_ingredients"] = [self.vocab[j] for j in self.ingredients[i].tolist()
                                   if j not in seen][:4]
        return card
