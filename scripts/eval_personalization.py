#!/usr/bin/env python3
"""Does personalization help, and is it really about *this* user?

    python scripts/eval_personalization.py               # variants of configs/personalization.yaml
    python scripts/eval_personalization.py --splits val

For every variant (same cases as the other evaluations):
- ndcg@10 / recall@20 with the true user of each case,
- ndcg@10 with a shuffled user (same cases, user ids permuted). If the profile carries real
  taste information, the true user must score higher: lift = true - shuffled > 0.
  This separates "the user's tastes help" from "any profile adds generic popularity".
- changed@10: average number of top-10 items that differ from the non-personalized list.

Writes artifacts/eval/personalization_<protocol>_<split>.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from src.eval.harness import evaluate_predictions, from_hybrid_ranker, predict_all
from src.eval.protocol import build_cases
from src.ranking.hybrid import load_ranker, ranking_config_from_yaml


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    p = argparse.ArgumentParser(description="Personalization study")
    p.add_argument("--config", default="configs/personalization.yaml")
    p.add_argument("--splits", nargs="+", choices=["val", "test"], default=None)
    p.add_argument("--max-cases", type=int, default=None)
    args = p.parse_args()

    pcfg = load_yaml(ROOT / args.config)
    ecfg = load_yaml(ROOT / pcfg["eval_config"])
    processed = ROOT / ecfg["processed_dir"]
    base = ranking_config_from_yaml(pcfg["model_config"])
    emb = ROOT / pcfg["embeddings"]
    if not emb.exists():
        sys.exit(f"{emb} not found: run scripts/train_nlgcl.py first")
    rankers = {name: load_ranker(processed, emb, replace(base, **(ov or {})))
               for name, ov in pcfg["variants"].items()}
    out_dir = ROOT / ecfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits or pcfg.get("splits", ["val", "test"]):
        cases = build_cases(processed, split=split, protocol=ecfg["protocol"],
                            hidden_ratio=ecfg["hidden_ratio"], min_query=ecfg["min_query"],
                            seed=ecfg["seed"], max_cases=args.max_cases or ecfg.get("max_cases")).cases
        shuffled = cases.copy()
        shuffled["user_id"] = np.random.default_rng(pcfg.get("shuffle_seed", 0)).permutation(
            cases["user_id"].to_numpy())
        print(f"\n=== split={split}  cases={len(cases):,}")

        ref = None
        results = {}
        for name, ranker in rankers.items():
            fn = from_hybrid_ranker(ranker)
            preds = predict_all(fn, cases, 20)
            r = evaluate_predictions(preds, cases, (10, 20))
            rs = evaluate_predictions(predict_all(fn, shuffled, 20), shuffled, (10, 20))
            if ref is None:
                ref = preds
            changed = np.mean([len(set(preds[c][:10]) - set(ref[c][:10])) for c in preds])
            cov = len({i for v in preds.values() for i in v[:10]}) / len(ranker.vocab)
            results[name] = {"ndcg@10": r["ndcg@10"], "recall@20": r["recall@20"], "mrr": r["mrr"],
                             "ndcg@10_shuffled_user": rs["ndcg@10"],
                             "user_lift": r["ndcg@10"] - rs["ndcg@10"],
                             "changed@10": float(changed), "coverage@10": cov}
            x = results[name]
            print(f"{name:<22s} ndcg@10={x['ndcg@10']:.4f}  recall@20={x['recall@20']:.4f}  "
                  f"shuffled={x['ndcg@10_shuffled_user']:.4f}  lift={x['user_lift']:+.4f}  "
                  f"changed={x['changed@10']:.2f}  cov={x['coverage@10']:.3f}")

        stem = f"personalization_{ecfg['protocol']}_{split}"
        (out_dir / f"{stem}.json").write_text(json.dumps(
            {"split": split, "n_cases": len(cases), "config": pcfg, "results": results}, indent=2),
            encoding="utf-8")
        cols = ["ndcg@10", "recall@20", "ndcg@10_shuffled_user", "user_lift", "changed@10", "coverage@10"]
        md = [f"# Personalization: split={split}, {len(cases):,} cases", "",
              "| variant | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
        md += [f"| {n} | " + " | ".join(f"{r[c]:.4f}" for c in cols) + " |" for n, r in results.items()]
        (out_dir / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"Saved -> {(out_dir / stem).relative_to(ROOT)}.json / .md")


if __name__ == "__main__":
    main()
