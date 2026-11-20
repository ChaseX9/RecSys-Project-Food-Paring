"""Hybrid ranking for food pairing.

Score = α · embedding + β · chemical + γ · preference + δ · co-occurrence
Hard filters: allergies / diet / exclude query ingredients.

With ``normalize`` each component is min-max scaled to [0, 1] per query, so the weights
mean what they say (raw cosine, Jaccard and L1 profiles live on very different scales).
``emb_similarity: dot`` keeps the embedding norm, which LightGCN-style models use to encode
how often an ingredient is used; cosine throws it away and over-recommends rare ingredients.
The co-occurrence term counts pairs in *train* recipes only (no leakage).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class RankingConfig:
    alpha_emb: float = 0.6
    beta_chem: float = 0.25
    gamma_pref: float = 0.15
    delta_cooc: float = 0.0
    emb_similarity: str = "cosine"   # "cosine" | "dot"
    normalize: bool = False          # per-query min-max of each component
    # personalization (see src/data/rating_profiles.py)
    pref_source: str = "frequency"   # frequency (5/5 only) | rating | embedding | hybrid (rating + embedding)
    pref_mode: str = "additive"      # additive: + gamma*pref | multiplicative: pairing * (1 + gamma*pref)
    neg_weight: float = 0.5          # weight of the "dislike" profile (rating source)
    profile_shrink_k: float = 5.0    # pseudo-ratings pulling a user's mean toward the global mean
    dislike_below_rating: float = 3.0  # only ratings strictly below this count as dislikes (1/5, 2/5)
    profile_combine: str = "net"     # net: consistency (P - N) / (P + N + alpha) | separate: like - neg_weight * dislike
    profile_alpha: float = 2.0       # smoothing of the consistency score (in typical ratings)
    profile_conf_k: float = 10.0     # confidence = n_ratings / (n_ratings + k); 0 = no confidence scaling
    profile_idf: bool = True         # down-weight ubiquitous ingredients in rating profiles
    # feedback loop (explicit like / dislike on ingredients, see api/db.py)
    # The feedback updates the user's preference vector (see preference()):
    #   like / cooked -> preference of the ingredient = feedback_like_value, and its close
    #                    neighbours get feedback_like_value * closeness
    #   dislike       -> preference -1 and the ingredient is banned; neighbours untouched
    feedback_ban_disliked: bool = True   # a disliked ingredient is never recommended again
    feedback_sim_threshold: float = 0.65 # cosine (NLGCL space) above which two ingredients are "close"
    feedback_like_value: float = 1.5     # preference given to a liked ingredient (profile values are in [-1, 1])
    top_k: int = 20
    # diet / allergy keyword filters, matched as whole words (optional plural s/es).
    # Compound words ("crabmeat", "soybean", ...) must be listed explicitly since
    # whole-word matching no longer catches them as substrings.
    allergy_keywords: dict[str, list[str]] = field(default_factory=lambda: {
        "gluten": ["wheat", "flour", "bread", "pasta", "barley", "rye", "couscous", "semolina",
                   "breadcrumb", "cornbread", "buckwheat noodle", "soba"],
        "dairy": ["milk", "butter", "cheese", "cream", "yogurt", "parmesan", "mozzarella", "cheddar",
                  "buttermilk", "butterscotch", "cheesecake", "creamed", "bechamel"],
        "nut": ["almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut", "peanut", "nut",
                "chestnut"],
        "shellfish": ["shrimp", "crab", "lobster", "shellfish", "clam", "mussel", "oyster",
                      "crabmeat", "crawfish", "crayfish"],
        "egg": ["egg"],
        "soy": ["soy", "tofu", "edamame", "soymilk", "soya", "soybean", "tofutti"],
    })
    # keyword -> preceding words that make it a different product
    # (e.g. "peanut butter" is not dairy, "water chestnut" is not a nut). Allergen
    # safety is preserved by the other lists: "peanut butter" is still banned for
    # "nut" via "peanut".
    keyword_exceptions: dict[str, list[str]] = field(default_factory=lambda: {
        "butter": ["peanut", "almond", "cashew", "coconut", "cocoa", "apple"],
        "chestnut": ["water"],
    })


PREF_SOURCES = ("frequency", "rating", "embedding", "hybrid")

MEAT_KEYWORDS = ("beef", "pork", "chicken", "lamb", "bacon", "ham", "sausage", "turkey",
                 "veal", "duck", "meat", "steak", "prosciutto", "salami",
                 "hamburger", "duckling",
                 "pancetta", "kielbasa", "lardon", "chorizo", "pepperoni", "liver",
                 "goose", "quail", "pheasant", "rabbit", "venison")
FISH_KEYWORDS = ("fish", "salmon", "tuna", "cod", "anchovy", "sardine", "trout", "shrimp",
                 "crab", "lobster", "shellfish",
                 "crabmeat", "crawfish", "crayfish", "catfish", "swordfish",
                 "clam", "mussel", "scallop", "squid", "octopus", "oyster", "caviar",
                 "halibut", "haddock", "sole", "sea bass", "snapper", "mahi mahi", "grouper",
                 "flounder", "herring", "mackerel", "tilapia",
                 "oyster sauce", "fish sauce")
# animal-derived products that are neither meat nor fish but excluded for vegetarians
VEGETARIAN_EXTRA_KEYWORDS = ("gelatin", "worcestershire sauce")


@lru_cache(maxsize=None)
def _keyword_regex(
    keywords: tuple[str, ...],
    exceptions: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> re.Pattern:
    """Compile one whole-word regex for ``keywords`` (plural s/es optional).

    ``exceptions`` maps a keyword to preceding qualifiers that cancel the match
    ("peanut" before "butter"). Cached: each keyword set is compiled once.
    """
    exc = dict(exceptions)
    alts = []
    for k in sorted(set(keywords), key=len, reverse=True):
        guard = "".join(f"(?<!{re.escape(q)} )" for q in exc.get(k, ()))
        alts.append(guard + re.escape(k))
    return re.compile(r"\b(?:" + "|".join(alts) + r")(?:e?s)?\b")


class HybridRanker:
    def __init__(
        self,
        embeddings: np.ndarray,
        vocab: pd.DataFrame,
        chemical: sp.csr_matrix | None = None,
        npmi: sp.csr_matrix | None = None,
        user_profiles: sp.csr_matrix | None = None,
        user_ids: np.ndarray | None = None,
        config: RankingConfig | None = None,
        cooccurrence: sp.csr_matrix | None = None,
        rating_profiles=None,
        rating_data=None,
    ):
        """``rating_data`` (src.data.rating_profiles.RatingData): lets the ranker build a
        profile at request time from recipe ratings given in the app (``recipe_ratings``)."""
        self.cfg = config or RankingConfig()
        if self.cfg.emb_similarity not in ("cosine", "dot"):
            raise ValueError(f"emb_similarity must be cosine or dot, got {self.cfg.emb_similarity!r}")
        if self.cfg.pref_source not in PREF_SOURCES:
            raise ValueError(f"pref_source must be one of {PREF_SOURCES}, got {self.cfg.pref_source!r}")
        if self.cfg.pref_mode not in ("additive", "multiplicative"):
            raise ValueError(f"pref_mode must be additive or multiplicative, got {self.cfg.pref_mode!r}")
        if self.cfg.pref_source != "frequency" and rating_profiles is None:
            raise ValueError(f"pref_source={self.cfg.pref_source!r} needs rating_profiles")
        raw = embeddings.astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.emb_unit = raw / norms                  # cosine space, used for user vectors
        self.emb = raw
        if self.cfg.emb_similarity == "cosine":
            self.emb = self.emb_unit
        self.cooc = None
        if cooccurrence is not None:
            self.cooc = cooccurrence.tocsr().astype(np.float32)
            self.cooc.data = np.log1p(self.cooc.data)

        self.vocab = vocab.reset_index(drop=True)
        self.name_to_id = {n: i for i, n in enumerate(self.vocab["name"])}
        self.id_to_name = dict(enumerate(self.vocab["name"]))
        self.chemical = chemical
        self.npmi = npmi
        self.user_profiles = user_profiles
        self.user_id_to_row = (
            {int(u): i for i, u in enumerate(user_ids)} if user_ids is not None else {}
        )
        self.rating_profiles = rating_profiles
        self.rating_data = rating_data
        self.raw_emb = raw
        self.rating_row = {}
        self.user_vecs = None
        if rating_profiles is not None:
            self.rating_row = {int(u): i for i, u in enumerate(rating_profiles.user_ids)}
            if self.cfg.pref_source in ("embedding", "hybrid"):
                self.user_vecs = rating_profiles.user_vectors(raw)

    def preference(self, user_id, liked: Sequence[str] | None = None,
                   disliked: Sequence[str] | None = None, recipe_ratings=None) -> np.ndarray:
        """Preference of ``user_id`` for every ingredient, updated with explicit feedback.

        Starts from the profile learned from the user's ratings (zeros for a user without
        history, e.g. a new app user), then applies the feedback: a liked ingredient and its
        close neighbours are raised, a disliked ingredient is set to -1 (its neighbours are
        left untouched). A new user is therefore personalized from their first like.
        """
        pref = self._history_preference(user_id, recipe_ratings).copy()
        liked_ids = self._resolve_exact(liked)
        if liked_ids:
            close, _ = self._closeness(liked_ids)
            pref = np.maximum(pref, self.cfg.feedback_like_value * close)
        disliked_ids = self._resolve_exact(disliked)
        if disliked_ids:
            pref[disliked_ids] = -1.0
        return pref.astype(np.float32)

    def profile_summary(self, user_id, liked=None, disliked=None, n: int = 10,
                        recipe_ratings=None) -> dict:
        """Top liked / disliked ingredients of the (feedback-updated) profile, for display."""
        pref = self.preference(user_id, liked, disliked, recipe_ratings)
        pos = [int(i) for i in np.argsort(-pref)[:n] if pref[i] > 0]
        neg = [int(i) for i in np.argsort(pref)[:n] if pref[i] < 0]
        return {"likes": [{"name": self.id_to_name[i], "value": round(float(pref[i]), 3)} for i in pos],
                "dislikes": [{"name": self.id_to_name[i], "value": round(float(pref[i]), 3)} for i in neg],
                "has_rating_history": bool(self._history_preference(user_id, recipe_ratings).any())}

    def _history_preference(self, user_id, recipe_ratings=None) -> np.ndarray:
        """Preference learned from the user's past ratings (zeros for an unknown user).

        frequency: share of the ingredient in the user's 5/5 recipes (min-max scaled).
        rating:    like - neg_weight * dislike, both min-max scaled (in [-neg_weight, 1]).
        embedding: min-max scaled cosine between the user vector and each ingredient.
        hybrid:    mean of rating and embedding.
        rating / embedding / hybrid are multiplied by the user's confidence.

        ``recipe_ratings`` [(recipe_id, rating)]: ratings given in the app. With rating_data, the
        profile is rebuilt with the same formula from these ratings plus the train ratings of
        ``user_id`` if it is a Food.com user (rating sources only; frequency ignores them).
        """
        n = len(self.vocab)
        zeros = np.zeros(n, dtype=np.float32)
        if recipe_ratings and self.rating_data is not None and self.cfg.pref_source != "frequency":
            return self._app_rating_preference(user_id, tuple((int(r), float(v))
                                                              for r, v in recipe_ratings))
        if user_id is None:
            return zeros
        try:
            uid = int(user_id)
        except (TypeError, ValueError):
            return zeros
        src = self.cfg.pref_source
        if src == "frequency":
            row = self.user_id_to_row.get(uid)
            if row is None or self.user_profiles is None:
                return zeros
            return _minmax(np.asarray(self.user_profiles.getrow(row).todense()).ravel().astype(np.float32))
        row = self.rating_row.get(uid)
        if row is None:
            return zeros
        vec = self.user_vecs[row] if self.user_vecs is not None else None
        return self._pref_from_profile(self.rating_profiles, row, vec)

    def _app_rating_preference(self, user_id, ratings: tuple) -> np.ndarray:
        """Profile rebuilt from app ratings (+ Food.com train ratings of user_id), cached:
        a heavy Food.com user takes ~0.25 s and the fridge page asks once per ingredient."""
        cache = self.__dict__.setdefault("_app_pref_cache", {})
        key = (str(user_id), ratings)
        if key not in cache:
            rp = self.rating_data.profile_for(self.rating_data.ratings_of(user_id) + list(ratings))
            if rp is None:
                pref = np.zeros(len(self.vocab), dtype=np.float32)
            else:
                vec = rp.user_vectors(self.raw_emb)[0] \
                    if self.cfg.pref_source in ("embedding", "hybrid") else None
                pref = self._pref_from_profile(rp, 0, vec)
            if len(cache) >= 256:
                cache.pop(next(iter(cache)))
            cache[key] = pref
        return cache[key]

    def _pref_from_profile(self, rp, row: int, user_vec) -> np.ndarray:
        """Preference vector of row ``row`` of RatingProfiles ``rp`` (rating sources)."""
        src = self.cfg.pref_source
        zeros = np.zeros(len(self.vocab), dtype=np.float32)
        parts = []
        if src in ("rating", "hybrid"):
            if self.cfg.profile_combine == "net" and rp.net is not None:
                parts.append(np.asarray(rp.net.getrow(row).todense()).ravel().astype(np.float32))
            else:
                like = _minmax(np.asarray(rp.like.getrow(row).todense()).ravel().astype(np.float32))
                dis = _minmax(np.asarray(rp.dislike.getrow(row).todense()).ravel().astype(np.float32))
                parts.append(like - self.cfg.neg_weight * dis)
        if src in ("embedding", "hybrid"):
            u = user_vec
            parts.append(_minmax(self.emb_unit @ u) if u is not None and np.any(u) else zeros)
        return (np.mean(parts, axis=0) * float(rp.confidence[row])).astype(np.float32)

    # ------------------------------------------------------------------ public
    def recommend(
        self,
        query: Sequence[str],
        user_id: int | None = None,
        allergies: Sequence[str] | None = None,
        diet: str | None = None,
        top_k: int | None = None,
        exclude: Sequence[str] | None = None,
        liked: Sequence[str] | None = None,
        disliked: Sequence[str] | None = None,
        candidates: Sequence[str] | None = None,
        recipe_ratings=None,
    ) -> list[dict]:
        """Return ranked pairing suggestions with scores and explanation hooks.

        ``recipe_ratings`` [(recipe_id, rating 1-5)]: recipes rated in the app, folded into the
        user's rating profile (see _history_preference).

        ``liked`` / ``disliked``: the user's explicit feedback on ingredients (feedback loop).
        It updates the preference vector (see preference()); disliked ingredients are also
        banned. Only likes spread to close neighbours.

        ``candidates``: if given, only these ingredients (exact names) can be returned,
        e.g. what the user has in the fridge. An empty list returns nothing.
        """
        top_k = top_k or self.cfg.top_k
        q_ids = self._resolve(query)
        if not q_ids:
            return []

        # embedding score = mean similarity (cosine or dot) to query ingredients
        q_emb = self.emb[q_ids].mean(axis=0)
        emb_score = self.emb @ q_emb  # (N,)

        # chemical score
        chem_score = np.zeros(len(self.vocab), dtype=np.float32)
        if self.chemical is not None:
            for i in q_ids:
                row = self.chemical.getrow(i)
                chem_score[row.indices] += row.data
            if len(q_ids) > 1:
                chem_score /= len(q_ids)

        # preference score: rating history updated with explicit feedback (see preference())
        liked_ids = [i for i in self._resolve_exact(liked) if i not in q_ids]
        disliked_ids = self._resolve_exact(disliked)
        pref_score = self.preference(user_id, [self.id_to_name[i] for i in liked_ids], disliked,
                                     recipe_ratings)
        fb_pos, fb_pos_src = self._closeness(liked_ids)

        # co-occurrence score = sum of log(1 + count) with the query (train recipes)
        cooc_score = np.zeros(len(self.vocab), dtype=np.float32)
        if self.cooc is not None and self.cfg.delta_cooc:
            cooc_score = np.asarray(self.cooc[q_ids].sum(axis=0)).ravel().astype(np.float32)

        if self.cfg.normalize:
            emb_score, chem_score, cooc_score = (
                _minmax(x) for x in (emb_score, chem_score, cooc_score))

        pairing = (
            self.cfg.alpha_emb * emb_score
            + self.cfg.beta_chem * chem_score
            + self.cfg.delta_cooc * cooc_score
        )
        if self.cfg.pref_mode == "multiplicative":
            # preference only re-orders plausible pairings: a poor pairing stays poor
            score = pairing * (1.0 + self.cfg.gamma_pref * pref_score)
        else:
            score = pairing + self.cfg.gamma_pref * pref_score


        # hard filters
        ban = set(q_ids)
        if exclude:
            ban |= set(self._resolve(exclude))
        ban |= self._allergy_ban(allergies)
        ban |= self._diet_ban(diet)
        if self.cfg.feedback_ban_disliked:
            ban |= set(disliked_ids)
        if candidates is not None:
            allowed = set(self._resolve_exact(candidates))
            ban |= set(range(len(self.vocab))) - allowed

        score[list(ban)] = -np.inf
        # also ban ultra-generic staples that are rarely "pairings"
        for staple in ("water", "salt", "ice"):
            if staple in self.name_to_id:
                score[self.name_to_id[staple]] = -np.inf

        top = np.argpartition(-score, min(top_k, len(score) - 1))[:top_k]
        top = top[np.argsort(-score[top])]

        results = []
        for rank, idx in enumerate(top, 1):
            if not np.isfinite(score[idx]):
                break
            results.append({
                "rank": rank,
                "ingredient_id": int(idx),
                "name": self.id_to_name[idx],
                "score": float(score[idx]),
                "emb_score": float(emb_score[idx]),
                "chem_score": float(chem_score[idx]),
                "pref_score": float(pref_score[idx]),
                "cooc_score": float(cooc_score[idx]),
                "feedback": self._feedback_reason(idx, fb_pos, fb_pos_src),
                "query_ids": q_ids,
            })
        return results

    # ---------------------------------------------------------------- helpers
    def _resolve_exact(self, names: Sequence[str] | None) -> list[int]:
        """Exact vocabulary names only: feedback must never hit a fuzzy look-alike."""
        out = []
        for n in names or []:
            i = self.name_to_id.get(str(n).strip().lower())
            if i is not None:
                out.append(i)
        return list(dict.fromkeys(out))

    def _closeness(self, ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """For every ingredient: how close it is to the closest of ``ids`` (0..1), and which one."""
        n = len(self.vocab)
        if not ids:
            return np.zeros(n, dtype=np.float32), np.full(n, -1)
        sims = self.emb_unit @ self.emb_unit[ids].T          # (N, len(ids))
        best = sims.argmax(axis=1)
        thr = self.cfg.feedback_sim_threshold
        close = np.clip((sims.max(axis=1) - thr) / (1.0 - thr), 0.0, 1.0).astype(np.float32)
        close[ids] = 1.0
        src = np.asarray(ids)[best]
        src[ids] = ids
        return close, src

    def _feedback_reason(self, idx, pos, pos_src, min_closeness: float = 0.25) -> str | None:
        if pos[idx] < min_closeness:      # tiny effects are not worth an explanation
            return None
        other = self.id_to_name[int(pos_src[idx])]
        return "you liked it" if int(pos_src[idx]) == int(idx) else f"close to {other}, which you liked"

    def _resolve(self, names: Sequence[str]) -> list[int]:
        ids = []
        for n in names:
            key = n.strip().lower()
            if key in self.name_to_id:
                ids.append(self.name_to_id[key])
            else:
                # fuzzy: substring match on vocab
                for vocab_name, vid in self.name_to_id.items():
                    if key in vocab_name or vocab_name in key:
                        ids.append(vid)
                        break
        return list(dict.fromkeys(ids))  # unique, order-preserving

    def _match_keywords(self, keywords: Sequence[str]) -> set[int]:
        """Ids of vocab names containing any keyword as a whole word."""
        if not keywords:
            return set()
        exceptions = tuple(sorted(
            (k, tuple(qs)) for k, qs in self.cfg.keyword_exceptions.items() if k in keywords
        ))
        pattern = _keyword_regex(tuple(keywords), exceptions)
        return {idx for name, idx in self.name_to_id.items() if pattern.search(name)}

    def _allergy_ban(self, allergies: Sequence[str] | None) -> set[int]:
        if not allergies:
            return set()
        ban = set()
        for a in allergies:
            keys = self.cfg.allergy_keywords.get(a.lower(), [a.lower()])
            ban |= self._match_keywords(keys)
        return ban

    def ban_pattern(self, diet: str | None, allergies: Sequence[str] | None) -> re.Pattern | None:
        """Whole-word regex of every keyword banned by ``diet`` / ``allergies`` (same lists as
        _diet_ban / _allergy_ban), to filter free text such as recipe names. None if nothing."""
        keys = []
        for a in allergies or []:
            keys += self.cfg.allergy_keywords.get(a.lower(), [a.lower()])
        d = (diet or "").lower()
        dairy = self.cfg.allergy_keywords.get("dairy", [])
        egg = self.cfg.allergy_keywords.get("egg", [])
        if d == "vegan":
            keys += list(MEAT_KEYWORDS + FISH_KEYWORDS + VEGETARIAN_EXTRA_KEYWORDS) + dairy + egg + ["honey"]
        elif d == "vegetarian":
            keys += list(MEAT_KEYWORDS + FISH_KEYWORDS + VEGETARIAN_EXTRA_KEYWORDS)
        elif d == "pescatarian":
            keys += list(MEAT_KEYWORDS)
        if not keys:
            return None
        exceptions = tuple(sorted(
            (k, tuple(qs)) for k, qs in self.cfg.keyword_exceptions.items() if k in keys))
        return _keyword_regex(tuple(dict.fromkeys(keys)), exceptions)

    def _diet_ban(self, diet: str | None) -> set[int]:
        if not diet:
            return set()
        diet = diet.lower()
        meat = list(MEAT_KEYWORDS)
        fish = list(FISH_KEYWORDS)
        veg_extra = list(VEGETARIAN_EXTRA_KEYWORDS)
        dairy = self.cfg.allergy_keywords.get("dairy", [])
        egg = self.cfg.allergy_keywords.get("egg", [])
        if diet in ("vegan",):
            return self._match_keywords(meat + fish + veg_extra + dairy + egg + ["honey"])
        if diet in ("vegetarian",):
            return self._match_keywords(meat + fish + veg_extra)
        if diet in ("pescatarian",):
            return self._match_keywords(meat)
        return set()


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def ranking_config_from_yaml(path: str | Path = "configs/model.yaml") -> RankingConfig:
    """RankingConfig from the ``ranking`` block of configs/model.yaml (unknown keys ignored)."""
    import yaml

    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    with open(path, encoding="utf-8") as f:
        block = (yaml.safe_load(f) or {}).get("ranking", {}) or {}
    return RankingConfig(**{k: v for k, v in block.items() if k in RankingConfig.__dataclass_fields__})


def train_cooccurrence(processed_dir: str | Path) -> sp.csr_matrix:
    """Ingredient co-occurrence counts over train recipes only (is_train_recipe)."""
    processed_dir = Path(processed_dir)
    ri = sp.load_npz(processed_dir / "recipe_ingredient.npz").tocsr()
    mask = pd.read_parquet(processed_dir / "recipes.parquet",
                           columns=["is_train_recipe"])["is_train_recipe"].to_numpy(dtype=bool)
    ri = ri[mask]
    co = (ri.T @ ri).tocsr()
    co.setdiag(0)
    co.eliminate_zeros()
    return co


def load_ranker(
    processed_dir: str | Path,
    embedding_path: str | Path,
    config: RankingConfig | None = None,
) -> HybridRanker:
    """Load everything the ranker needs. ``config=None`` reads configs/model.yaml."""
    config = config or ranking_config_from_yaml()
    processed_dir = Path(processed_dir)
    emb = np.load(embedding_path)
    vocab = pd.read_csv(processed_dir / "ingredients_vocab.csv")
    if emb.shape[0] != len(vocab):
        # Happens after a data rebuild changed the vocabulary: the model must be retrained.
        raise ValueError(
            f"{embedding_path} has {emb.shape[0]} ingredients but "
            f"{processed_dir / 'ingredients_vocab.csv'} has {len(vocab)}: the model is out of date. "
            "Retrain it with `python scripts/train_nlgcl.py` (then commit artifacts/ingredient_embeddings.npy).")
    chem = None
    chem_path = processed_dir / "chemical_similarity.npz"
    if chem_path.exists():
        chem = sp.load_npz(chem_path).tocsr()
    npmi = None
    npmi_path = processed_dir / "ingredient_npmi.npz"
    if npmi_path.exists():
        npmi = sp.load_npz(npmi_path).tocsr()
    profiles, uids = None, None
    if (processed_dir / "user_profiles.npz").exists():
        profiles = sp.load_npz(processed_dir / "user_profiles.npz").tocsr()
        uids = np.load(processed_dir / "user_profile_ids.npy")
    cooc = train_cooccurrence(processed_dir) if config.delta_cooc else None
    rp = rating_profiles_for(processed_dir, config)
    rd = rating_data_for(processed_dir, config)
    return HybridRanker(emb, vocab, chem, npmi, profiles, uids, config, cooccurrence=cooc,
                        rating_profiles=rp, rating_data=rd)


def rating_data_for(processed_dir: str | Path, config: RankingConfig):
    """Data to build app users' profiles from their recipe ratings (None if not available)."""
    if config.pref_source == "frequency" or not (Path(processed_dir) / "interactions_train.parquet").exists():
        return None
    from src.data.rating_profiles import load_rating_data
    return load_rating_data(processed_dir, config.profile_shrink_k, config.dislike_below_rating,
                            config.profile_conf_k, config.profile_idf, config.profile_alpha)


def rating_profiles_for(processed_dir: str | Path, config: RankingConfig):
    """Rating-aware profiles when the config needs them (cached per settings), else None."""
    if config.pref_source == "frequency":
        return None
    return _rating_profiles_cached(str(Path(processed_dir).resolve()), config.profile_shrink_k,
                                   config.dislike_below_rating, config.profile_conf_k,
                                   config.profile_idf, config.profile_alpha)


@lru_cache(maxsize=8)
def _rating_profiles_cached(processed_dir: str, shrink_k: float, dislike_below: float, conf_k: float,
                            use_idf: bool, alpha: float):
    from src.data.rating_profiles import build_rating_profiles
    return build_rating_profiles(processed_dir, shrink_k, dislike_below, conf_k, use_idf, alpha)
