#!/usr/bin/env python3
"""Evaluate baselines (and the trained model when available) on held-out recipes.

    python scripts/evaluate.py                    # test split, protocol from configs/eval.yaml
    python scripts/evaluate.py --split val
    python scripts/evaluate.py --embeddings artifacts/ingredient_embeddings.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.eval.baselines import load_train_recipe_ingredient, make_baselines
from src.eval.harness import evaluate, from_hybrid_ranker
from src.eval.protocol import build_cases


def main():
    p = argparse.ArgumentParser(description="Food pairing evaluation harness")
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--split", choices=["val", "test"], default=None)
    p.add_argument("--protocol", choices=["unseen_recipes", "recipe_holdout"], default=None)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--embeddings", default=None, help="Evaluate HybridRanker with these embeddings")
    args = p.parse_args()

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    split = args.split or cfg["split"]
    protocol = args.protocol or cfg["protocol"]
    max_cases = args.max_cases or cfg.get("max_cases")
    processed = ROOT / cfg["processed_dir"]
    ks = cfg["ks"]

    setup = build_cases(processed, split=split, protocol=protocol,
                        hidden_ratio=cfg["hidden_ratio"], min_query=cfg["min_query"],
                        seed=cfg["seed"], max_cases=max_cases, holdout_frac=cfg["holdout_frac"])
    cases = setup.cases
    print(f"Protocol={protocol}  split={split}  cases={len(cases):,}  "
          f"avg query={cases.query_ids.map(len).mean():.1f}  avg hidden={cases.target_ids.map(len).mean():.1f}")

    train_ri = load_train_recipe_ingredient(processed, setup.train_recipe_mask)
    predictors = make_baselines(cfg["baselines"], train_ri, seed=cfg["seed"])

    if args.embeddings:
        if protocol == "recipe_holdout":
            print("WARNING: the model must be trained on a graph that excludes the held-out recipes.")
        from src.ranking.hybrid import load_ranker  # branch model-train
        predictors["hybrid_ranker"] = from_hybrid_ranker(load_ranker(processed, ROOT / args.embeddings))

    results = {}
    for name, fn in predictors.items():
        results[name] = evaluate(fn, cases, ks)

    cols = [f"ndcg@{k}" for k in ks] + [f"recall@{k}" for k in ks] + [f"hit@{k}" for k in ks] + ["mrr"]
    print(f"\n{'model':<16s}" + "".join(f"{c:>11s}" for c in cols))
    for name, r in results.items():
        print(f"{name:<16s}" + "".join(f"{r[c]:>11.4f}" for c in cols))

    out_dir = ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{protocol}_{split}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"protocol": protocol, "split": split, "config": cfg, "results": results}, f, indent=2)
    print(f"\nSaved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
