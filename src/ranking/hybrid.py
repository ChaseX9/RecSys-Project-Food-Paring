"""Hybrid ranking for food pairing.

Score = α · embedding_sim + β · chemical_sim + γ · preference_score
Hard filters: allergies / diet / exclude query ingredients.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    top_k: int = 20
    # simple diet / allergy keyword filters (lowercase substrings)
    allergy_keywords: dict[str, list[str]] = field(default_factory=lambda: {
        "gluten": ["wheat", "flour", "bread", "pasta", "barley", "rye", "couscous", "semolina"],
        "dairy": ["milk", "butter", "cheese", "cream", "yogurt", "parmesan", "mozzarella", "cheddar"],
        "nut": ["almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut", "peanut", "nut"],
        "shellfish": ["shrimp", "crab", "lobster", "shellfish", "clam", "mussel", "oyster"],
        "egg": ["egg"],
        "soy": ["soy", "tofu", "edamame", "soymilk"],
    })


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
    ):
        self.emb = embeddings.astype(np.float32)
        # L2-normalize for cosine
        norms = np.linalg.norm(self.emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.emb = self.emb / norms

        self.vocab = vocab.reset_index(drop=True)
        self.name_to_id = {n: i for i, n in enumerate(self.vocab["name"])}
        self.id_to_name = dict(enumerate(self.vocab["name"]))
        self.chemical = chemical
        self.npmi = npmi
        self.user_profiles = user_profiles
        self.user_id_to_row = (
            {int(u): i for i, u in enumerate(user_ids)} if user_ids is not None else {}
        )
        self.cfg = config or RankingConfig()

    # ------------------------------------------------------------------ public
    def recommend(
        self,
        query: Sequence[str],
        user_id: int | None = None,
        allergies: Sequence[str] | None = None,
        diet: str | None = None,
        top_k: int | None = None,
        exclude: Sequence[str] | None = None,
    ) -> list[dict]:
        """Return ranked pairing suggestions with scores and explanation hooks."""
        top_k = top_k or self.cfg.top_k
        q_ids = self._resolve(query)
        if not q_ids:
            return []

        # embedding score = mean cosine to query ingredients
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

        # preference score
        pref_score = np.zeros(len(self.vocab), dtype=np.float32)
        if user_id is not None and self.user_profiles is not None:
            row = self.user_id_to_row.get(int(user_id))
            if row is not None:
                pref_score = np.asarray(self.user_profiles.getrow(row).todense()).ravel()

        score = (
            self.cfg.alpha_emb * emb_score
            + self.cfg.beta_chem * chem_score
            + self.cfg.gamma_pref * pref_score
        )

        # hard filters
        ban = set(q_ids)
        if exclude:
            ban |= set(self._resolve(exclude))
        ban |= self._allergy_ban(allergies)
        ban |= self._diet_ban(diet)

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
                "query_ids": q_ids,
            })
        return results

    # ---------------------------------------------------------------- helpers
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

    def _allergy_ban(self, allergies: Sequence[str] | None) -> set[int]:
        if not allergies:
            return set()
        ban = set()
        for a in allergies:
            keys = self.cfg.allergy_keywords.get(a.lower(), [a.lower()])
            for name, idx in self.name_to_id.items():
                if any(k in name for k in keys):
                    ban.add(idx)
        return ban

    def _diet_ban(self, diet: str | None) -> set[int]:
        if not diet:
            return set()
        diet = diet.lower()
        ban = set()
        meat = ["beef", "pork", "chicken", "lamb", "bacon", "ham", "sausage", "turkey",
                "veal", "duck", "meat", "steak", "prosciutto", "salami"]
        fish = ["fish", "salmon", "tuna", "cod", "anchovy", "sardine", "trout", "shrimp",
                "crab", "lobster", "shellfish"]
        dairy = self.cfg.allergy_keywords.get("dairy", [])
        egg = self.cfg.allergy_keywords.get("egg", [])
        if diet in ("vegan",):
            for name, idx in self.name_to_id.items():
                if any(k in name for k in meat + fish + dairy + egg + ["honey", "gelatin"]):
                    ban.add(idx)
        elif diet in ("vegetarian",):
            for name, idx in self.name_to_id.items():
                if any(k in name for k in meat + fish):
                    ban.add(idx)
        elif diet in ("pescatarian",):
            for name, idx in self.name_to_id.items():
                if any(k in name for k in meat):
                    ban.add(idx)
        return ban


def load_ranker(
    processed_dir: str | Path,
    embedding_path: str | Path,
    config: RankingConfig | None = None,
) -> HybridRanker:
    processed_dir = Path(processed_dir)
    emb = np.load(embedding_path)
    vocab = pd.read_csv(processed_dir / "ingredients_vocab.csv")
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
    return HybridRanker(emb, vocab, chem, npmi, profiles, uids, config)
