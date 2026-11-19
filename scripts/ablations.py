#!/usr/bin/env python3
"""Ablation study: same harness, same cases, innovation switched on/off.

    python scripts/ablations.py                  # all variants of configs/ablations.yaml, val + test
    python scripts/ablations.py --splits val     # tuning only
    python scripts/ablations.py --retrain        # ignore cached embeddings in artifacts/ablations/
    python scripts/ablations.py --max-cases 300  # quick smoke run

Train-time variants (NLGCL vs LightGCN, graph edges) are trained once and cached in
artifacts/ablations/<variant>/. Each one is then ranked with every rank-time variant
(hybrid score components), next to the random / popularity / co-occurrence baselines.

Writes artifacts/eval/ablations_<protocol>_<split>.json and .md (table for the report).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.sparse as sp
import yaml

from src.eval.baselines import load_train_recipe_ingredient, make_baselines
from src.eval.beyond import beyond_accuracy, item_popularity
from src.eval.harness import evaluate_predictions, from_hybrid_ranker, predict_all
from src.eval.protocol import build_cases
from src.graph.build_graph import build_ingredient_graph
from src.ranking.hybrid import RankingConfig, load_ranker
from src.training.train import save_artifacts, train_nlgcl

TRAIN_KEYS = {"lambda_cl", "n_groups", "temperature", "graph_alpha", "epochs"}


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def train_variant(name: str, overrides: dict, mcfg: dict, processed: Path, out_dir: Path,
                  retrain: bool, device: str) -> Path:
    unknown = set(overrides) - TRAIN_KEYS
    if unknown:
        raise ValueError(f"train variant {name!r}: unknown keys {sorted(unknown)}")
    emb_path = out_dir / name / "ingredient_embeddings.npy"
    if emb_path.exists() and not retrain:
        print(f"[{name}] cached embeddings: {emb_path.relative_to(ROOT)}")
        return emb_path

    g, m, t = mcfg.get("graph", {}), mcfg.get("model", {}), mcfg.get("train", {})
    alpha = overrides.get("graph_alpha", g.get("alpha", 0.7))
    if alpha == g.get("alpha", 0.7):
        graph = {"norm_adj": sp.load_npz(processed / "ingredient_norm_adj.npz").tocsr(),
                 "adj": sp.load_npz(processed / "ingredient_adj.npz").tocsr()}
    else:  # rebuilt in memory, data/processed is left untouched
        graph = build_ingredient_graph(processed, alpha=alpha, top_k=g.get("top_k", 50),
                                       min_cooccur=g.get("min_cooccur", 2))
    lambda_cl = overrides.get("lambda_cl", t.get("lambda_cl", 1e-5))
    print(f"[{name}] training  alpha={alpha}  lambda_cl={lambda_cl}  "
          f"edges={graph['adj'].nnz // 2:,}")
    model, emb, meta = train_nlgcl(
        norm_adj=graph["norm_adj"], adj=graph["adj"],
        emb_dim=m.get("emb_dim", 64), n_layers=m.get("n_layers", 3),
        n_groups=overrides.get("n_groups", m.get("n_groups", 2)),
        temperature=overrides.get("temperature", m.get("temperature", 0.2)),
        lambda_cl=lambda_cl, lambda_reg=t.get("lambda_reg", 1e-4), lr=t.get("lr", 1e-3),
        epochs=overrides.get("epochs", t.get("epochs", 40)),
        batch_size=t.get("batch_size", 4096), device=device, seed=t.get("seed", 42),
        log_every=t.get("log_every", 5),
    )
    meta["variant"] = {"name": name, "overrides": overrides, "graph_alpha": alpha,
                       "lambda_cl": lambda_cl}
    save_artifacts(emb, meta, model, out_dir / name)
    return emb_path


def to_markdown(results: dict, cols: list[str], title: str) -> str:
    lines = [f"### {title}", "", "| model | " + " | ".join(cols) + " |",
             "|---|" + "---:|" * len(cols)]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r.get(c, float('nan')):.4f}" for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="Ablation study for the NLGCL food pairing model")
    p.add_argument("--config", default="configs/ablations.yaml")
    p.add_argument("--splits", nargs="+", choices=["val", "test"], default=None)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    acfg = load_yaml(ROOT / args.config)
    mcfg = load_yaml(ROOT / acfg["model_config"])
    ecfg = load_yaml(ROOT / acfg["eval_config"])
    processed = ROOT / ecfg["processed_dir"]
    protocol = ecfg["protocol"]
    if protocol != "unseen_recipes":
        raise SystemExit("ablations need protocol unseen_recipes: the trained graph only "
                         "excludes non-train recipes, not a recipe_holdout set")
    ks = ecfg["ks"]
    k_beyond = acfg.get("beyond_k", 10)
    splits = args.splits or acfg.get("splits", ["val", "test"])
    max_cases = args.max_cases or ecfg.get("max_cases")

    # 1. train-time variants (cached)
    emb_paths = {name: train_variant(name, ov or {}, mcfg, processed, ROOT / acfg["out_dir"],
                                     args.retrain, args.device)
                 for name, ov in acfg["train_variants"].items()}

    base_rank = RankingConfig(**{k: v for k, v in mcfg.get("ranking", {}).items()
                                 if k in RankingConfig.__dataclass_fields__})
    npmi = sp.load_npz(processed / "ingredient_npmi.npz").tocsr()
    chem_path = processed / "chemical_similarity.npz"
    chemical = sp.load_npz(chem_path).tocsr() if chem_path.exists() else None

    acc_cols = [f"ndcg@{k}" for k in ks] + [f"recall@{k}" for k in ks] + ["mrr"]
    beyond_cols = [f"coverage@{k_beyond}", f"longtail@{k_beyond}", f"novelty@{k_beyond}",
                   f"grounded@{k_beyond}"]
    out_dir = ROOT / ecfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        setup = build_cases(processed, split=split, protocol=protocol,
                            hidden_ratio=ecfg["hidden_ratio"], min_query=ecfg["min_query"],
                            seed=ecfg["seed"], max_cases=max_cases,
                            holdout_frac=ecfg["holdout_frac"])
        cases = setup.cases
        train_ri = load_train_recipe_ingredient(processed, setup.train_recipe_mask)
        pop = item_popularity(train_ri)
        print(f"\n=== split={split}  protocol={protocol}  cases={len(cases):,}")

        predictors = dict(make_baselines(ecfg["baselines"], train_ri, seed=ecfg["seed"]))
        for tname, path in emb_paths.items():
            for rname, rov in acfg["rank_variants"].items():
                ranker = load_ranker(processed, path, replace(base_rank, **(rov or {})))
                predictors[f"{tname}/{rname}"] = from_hybrid_ranker(ranker)

        results = {}
        for name, fn in predictors.items():
            t0 = time.time()
            preds = predict_all(fn, cases, max(ks + [k_beyond]))
            r = evaluate_predictions(preds, cases, ks)
            r.update(beyond_accuracy(preds, cases, pop, train_ri.shape[0], k=k_beyond,
                                     head_frac=acfg.get("longtail_head_frac", 0.2),
                                     npmi=npmi, chemical=chemical,
                                     npmi_thr=acfg.get("grounded_npmi", 0.15),
                                     chem_thr=acfg.get("grounded_chem", 0.05)))
            r["latency_ms"] = 1000 * (time.time() - t0) / max(len(cases), 1)
            results[name] = r
            print(f"{name:<28s} ndcg@10={r.get('ndcg@10', 0):.4f}  recall@20={r.get('recall@20', 0):.4f}"
                  f"  cov@{k_beyond}={r[f'coverage@{k_beyond}']:.3f}"
                  f"  tail@{k_beyond}={r[f'longtail@{k_beyond}']:.3f}  ({r['latency_ms']:.1f} ms/query)")

        stem = f"ablations_{protocol}_{split}"
        with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump({"protocol": protocol, "split": split, "n_cases": len(cases),
                       "ablation_config": acfg, "model_config": mcfg, "eval_config": ecfg,
                       "results": results}, f, indent=2)
        md = (f"# Ablations: protocol={protocol}, split={split}, {len(cases):,} cases\n\n"
              + to_markdown(results, acc_cols, "Accuracy") + "\n"
              + to_markdown(results, beyond_cols, f"Beyond accuracy (top-{k_beyond})"))
        (out_dir / f"{stem}.md").write_text(md, encoding="utf-8")
        print(f"Saved -> {(out_dir / stem).relative_to(ROOT)}.json / .md")


if __name__ == "__main__":
    main()
