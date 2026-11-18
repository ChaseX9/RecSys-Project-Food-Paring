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
) -> list[dict]:
    """Deterministic fake recommendations: same query -> same output.

    A fixed pool of POOL_SIZE candidates is drawn and ranked, then cut to top_k, so
    top_k=5 is exactly the first 5 of top_k=10. `diet` only feeds the seed; allergies are
    a simple substring filter. The mock knows no user profiles, so pref_score is always 0.0
    (what HybridRanker returns for an unknown user_id).
    """
    query = {i.strip().lower() for i in ingredients}
    allergy_keys = [a.strip().lower() for a in allergies or [] if a.strip()]

    candidates = [
        n for n in dict.fromkeys(load_vocab(vocab_path))
        if n not in query and n not in STAPLES and not any(k in n for k in allergy_keys)
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
         "explanation": f"[mock] {it['name']} is often paired with {query_str}."}
        for rank, it in enumerate(items[:top_k], 1)
    ]
