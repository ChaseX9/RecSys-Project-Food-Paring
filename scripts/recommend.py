#!/usr/bin/env python3
"""CLI demo: recommend pairings for given ingredients."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import scipy.sparse as sp

from src.explain.reasons import attach_explanations
from src.ranking.hybrid import RankingConfig, load_ranker


def main():
    p = argparse.ArgumentParser(description="Food pairing recommender")
    p.add_argument("ingredients", nargs="+", help="Query ingredients (e.g. salmon lemon)")
    p.add_argument("--user-id", type=int, default=None)
    p.add_argument("--allergies", nargs="*", default=None, help="e.g. gluten dairy nut")
    p.add_argument("--diet", default=None, choices=["vegan", "vegetarian", "pescatarian"])
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--embeddings", default="artifacts/ingredient_embeddings.npy")
    args = p.parse_args()

    processed = ROOT / args.processed_dir
    emb_path = ROOT / args.embeddings
    if not emb_path.exists():
        sys.exit(f"Embeddings not found at {emb_path}. Run scripts/train_nlgcl.py first.")

    ranker = load_ranker(processed, emb_path)
    results = ranker.recommend(
        query=args.ingredients,
        user_id=args.user_id,
        allergies=args.allergies,
        diet=args.diet,
        top_k=args.top_k,
    )

    npmi = None
    chem = None
    compounds = None
    ing_comp = None
    npmi_p = processed / "ingredient_npmi.npz"
    if npmi_p.exists():
        npmi = sp.load_npz(npmi_p).tocsr()
    chem_p = processed / "chemical_similarity.npz"
    if chem_p.exists():
        chem = sp.load_npz(chem_p).tocsr()
    comp_p = processed / "compounds.csv"
    ic_p = processed / "ingredient_compounds.npz"
    if comp_p.exists() and ic_p.exists():
        compounds = pd.read_csv(comp_p)
        ing_comp = sp.load_npz(ic_p).tocsr()

    results = attach_explanations(
        results, args.ingredients, npmi, chem, compounds, ing_comp
    )

    print(f"\nPairings for: {', '.join(args.ingredients)}")
    if args.allergies:
        print(f"  allergies filtered: {args.allergies}")
    if args.diet:
        print(f"  diet: {args.diet}")
    print("-" * 60)
    for r in results:
        print(
            f"  {r['rank']:2d}. {r['name']:<30s}  "
            f"score={r['score']:.3f}  (emb={r['emb_score']:.3f} chem={r['chem_score']:.3f})"
        )
        print(f"      → {r['explanation']}")
    print()


if __name__ == "__main__":
    main()
