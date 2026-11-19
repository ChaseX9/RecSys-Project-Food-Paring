"""Mock recommender used by POST /recommend until the real HybridRanker is wired in (week 3).

TODO(week 3): replace `mock_recommend` with HybridRanker.recommend() (src/ranking/hybrid.py)
followed by attach_explanations() (src/explain/reasons.py), both on branch model-train,
which provides the `explanation` field.

Only `mock_recommend` needs to be replaced: it returns items in the same shape the API
contract expects (rank, name, score, emb_score, chem_score, pref_score, explanation).
"""
from __future__ import annotations

import csv
import hashlib
import random
from functools import lru_cache
from pathlib import Path

VOCAB_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "ingredients_vocab.csv"

FALLBACK_VOCAB = (
    "garlic", "onion", "lemon", "olive oil", "basil", "thyme", "rosemary", "parsley",
    "tomato", "ginger", "soy sauce", "honey", "black pepper", "butter", "cumin",
    "paprika", "cilantro", "lime", "mushroom", "parmesan", "chili", "dill", "mint",
    "cinnamon", "vanilla", "apple", "carrot", "shallot", "white wine", "yogurt",
)

# Same staples HybridRanker bans: rarely meaningful as "pairings".
STAPLES = {"water", "salt", "ice"}

# Fixed candidate pool size (= max top_k): the pool never depends on top_k, so any
# top_k returns a prefix of the same ranked list.
POOL_SIZE = 50


@lru_cache(maxsize=4)
def load_vocab(path: str | Path = VOCAB_PATH) -> tuple[str, ...]:
    """Ingredient names from the processed vocab CSV, or a small fallback list if absent."""
    path = Path(path)
    if not path.exists():
        return FALLBACK_VOCAB
    with path.open(newline="", encoding="utf-8") as f:
        names = tuple(row["name"].strip().lower() for row in csv.DictReader(f) if row.get("name"))
    return names or FALLBACK_VOCAB


def _seed(ingredients: list[str], allergies: list[str] | None, diet: str | None) -> int:
    # top_k is deliberately not part of the seed (see POOL_SIZE).
    key = "|".join([
        ",".join(sorted({i.strip().lower() for i in ingredients})),
        ",".join(sorted({a.strip().lower() for a in allergies or []})),
        (diet or "").strip().lower(),
    ])
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def mock_recommend(
    ingredients: list[str],
    top_k: int = 10,
    user_id: str | None = None,
    allergies: list[str] | None = None,
    diet: str | None = None,
    vocab_path: str | Path = VOCAB_PATH,
    disliked: list[str] | None = None,
) -> list[dict]:
    """Deterministic fake recommendations: same query -> same output.

    A fixed pool of POOL_SIZE candidates is drawn and ranked, then cut to top_k, so
    top_k=5 is exactly the first 5 of top_k=10. `diet` only feeds the seed; allergies are
    a simple substring filter. The mock knows no user profiles, so pref_score is always 0.0
    (what HybridRanker returns for an unknown user_id).
    """
    query = {i.strip().lower() for i in ingredients}
    banned = {d.strip().lower() for d in disliked or []}  # feedback loop: never show a disliked item
    allergy_keys = [a.strip().lower() for a in allergies or [] if a.strip()]

    candidates = [
        n for n in dict.fromkeys(load_vocab(vocab_path))
        if n not in query and n not in STAPLES and n not in banned
        and not any(k in n for k in allergy_keys)
    ]
    rng = random.Random(_seed(ingredients, allergies, diet))
    pool = rng.sample(candidates, min(POOL_SIZE, len(candidates)))

    pref = 0.0  # no user profiles in the mock -> unknown user
    items = []
    for name in pool:
        emb = rng.uniform(0.3, 0.95)
        chem = rng.uniform(0.0, 0.8)
        items.append({"name": name, "emb_score": round(emb, 4), "chem_score": round(chem, 4),
                      "pref_score": pref,
                      "score": round(0.6 * emb + 0.25 * chem + 0.15 * pref, 4)})
    # Stable sort with name as tie-breaker so the ranking is fully deterministic.
    items.sort(key=lambda d: (-d["score"], d["name"]))

    query_str = ", ".join(sorted(query))
    return [
        {"rank": rank, "name": it["name"], "score": it["score"], "emb_score": it["emb_score"],
         "chem_score": it["chem_score"], "pref_score": it["pref_score"],
         "explanation": f"[mock] {it['name']} is often paired with {query_str}.", "feedback": None}
        for rank, it in enumerate(items[:top_k], 1)
    ]


# --------------------------------------------------------------------------- real model
ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = ROOT / "artifacts" / "ingredient_embeddings.npy"
PROCESSED_DIR = ROOT / "data" / "processed"


class ModelRecommender:
    """Trained NLGCL embeddings + HybridRanker + explanations, loaded once at startup."""

    def __init__(self, processed_dir: str | Path = PROCESSED_DIR,
                 embeddings_path: str | Path = EMBEDDINGS_PATH, ranker=None):
        import pandas as pd
        import scipy.sparse as sp

        from src.ranking.hybrid import load_ranker

        processed_dir = Path(processed_dir)
        self.ranker = ranker or load_ranker(processed_dir, embeddings_path)
        self.npmi = self.ranker.npmi
        self.chemical = self.ranker.chemical
        self.compounds = self.ing_comp = None
        comp, ic = processed_dir / "compounds.csv", processed_dir / "ingredient_compounds.npz"
        if comp.exists() and ic.exists():
            self.compounds = pd.read_csv(comp)
            self.ing_comp = sp.load_npz(ic).tocsr()

    def recommend(self, ingredients, top_k=10, user_id=None, allergies=None, diet=None,
                  liked=None, disliked=None) -> list[dict]:
        from src.explain.reasons import attach_explanations

        recs = self.ranker.recommend(ingredients, user_id=user_id, allergies=allergies, diet=diet,
                                     top_k=top_k, liked=liked, disliked=disliked)
        recs = attach_explanations(recs, ingredients, self.npmi, self.chemical,
                                   self.compounds, self.ing_comp)
        out = []
        for r in recs:
            expl = r["explanation"]
            if r.get("feedback"):
                expl = f"{expl} + {r['feedback']}"
            out.append({k: r[k] for k in ("rank", "name", "score", "emb_score", "chem_score", "pref_score")}
                       | {"explanation": expl, "feedback": r.get("feedback")})
        return out

    def profile(self, user_id, liked=None, disliked=None, n: int = 10) -> dict:
        """The user's preference profile (rating history updated with feedback)."""
        return self.ranker.profile_summary(user_id, liked, disliked, n)


def model_available(embeddings_path: str | Path = EMBEDDINGS_PATH) -> bool:
    return Path(embeddings_path).exists()
