"""Lightweight natural-language explanations for pairings."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp


def explain_pair(
    query_names: Sequence[str],
    candidate_name: str,
    candidate_id: int,
    query_ids: Sequence[int],
    npmi: sp.csr_matrix | None = None,
    chemical: sp.csr_matrix | None = None,
    compounds_df: pd.DataFrame | None = None,
    ingredient_compounds: sp.csr_matrix | None = None,
    emb_score: float = 0.0,
    chem_score: float = 0.0,
) -> str:
    """Return a short human-readable reason string."""
    parts: list[str] = []

    # co-occurrence / NPMI
    if npmi is not None and query_ids:
        best_npmi = 0.0
        best_q = None
        for q in query_ids:
            val = npmi[q, candidate_id]
            if hasattr(val, "toarray"):
                val = float(val.toarray().ravel()[0]) if val.nnz else 0.0
            else:
                val = float(val)
            if val > best_npmi:
                best_npmi = val
                best_q = q
        if best_npmi > 0.15:
            qname = query_names[0] if len(query_names) == 1 else "your ingredients"
            if best_q is not None and len(query_names) == len(query_ids):
                # try to recover name
                pass
            parts.append(f"often paired together in recipes (NPMI={best_npmi:.2f})")

    # shared compounds
    if (
        chemical is not None
        and ingredient_compounds is not None
        and compounds_df is not None
        and query_ids
        and chem_score > 0.05
    ):
        shared = _shared_compounds(query_ids, candidate_id, ingredient_compounds, compounds_df)
        if shared:
            names = ", ".join(shared[:3])
            more = f" (+{len(shared)-3} more)" if len(shared) > 3 else ""
            parts.append(f"share aromatic compounds ({names}{more})")
        elif chem_score > 0.1:
            parts.append("chemically similar flavor profile")

    if emb_score > 0.5 and not parts:
        parts.append("close in the learned pairing embedding space")

    if not parts:
        parts.append("suggested by the pairing model")

    return " + ".join(parts)


def _shared_compounds(
    query_ids: Sequence[int],
    cand_id: int,
    ing_comp: sp.csr_matrix,
    compounds_df: pd.DataFrame,
    max_show: int = 5,
) -> list[str]:
    cand_row = ing_comp.getrow(cand_id)
    cand_set = set(cand_row.indices.tolist())
    if not cand_set:
        return []
    shared_idx: set[int] = set()
    for q in query_ids:
        q_set = set(ing_comp.getrow(q).indices.tolist())
        shared_idx |= (cand_set & q_set)
    if not shared_idx:
        return []
    # map compound_idx → name
    if "compound_idx" in compounds_df.columns:
        name_map = dict(zip(compounds_df["compound_idx"], compounds_df["name"]))
    else:
        name_map = dict(enumerate(compounds_df["name"]))
    names = [str(name_map[i]).replace("_", " ") for i in sorted(shared_idx) if i in name_map]
    return names[:max_show]


def attach_explanations(
    results: list[dict],
    query_names: Sequence[str],
    npmi: sp.csr_matrix | None = None,
    chemical: sp.csr_matrix | None = None,
    compounds_df: pd.DataFrame | None = None,
    ingredient_compounds: sp.csr_matrix | None = None,
) -> list[dict]:
    for r in results:
        r["explanation"] = explain_pair(
            query_names=query_names,
            candidate_name=r["name"],
            candidate_id=r["ingredient_id"],
            query_ids=r.get("query_ids", []),
            npmi=npmi,
            chemical=chemical,
            compounds_df=compounds_df,
            ingredient_compounds=ingredient_compounds,
            emb_score=r.get("emb_score", 0.0),
            chem_score=r.get("chem_score", 0.0),
        )
    return results
