#!/usr/bin/env python3
"""Grid search of the hybrid ranking weights on the val split (fast, vectorized).

    python scripts/tune_ranking.py                      # NLGCL vs LightGCN, val then test
    python scripts/tune_ranking.py --max-cases 500      # quick run

Score of the served ranker (src/ranking/hybrid.py, normalize: true, pref multiplicative):

    (alpha * emb + beta * chem + delta * cooc) * (1 + gamma * pref)

each component min-max scaled per query. The components are computed once per case, then
every weight combination is scored with numpy (a few ms each) instead of calling
HybridRanker.recommend thousands of times. The scores are the same as the harness (checked
against src.eval.harness for the served config at startup).

Only ratios between alpha / beta / delta matter (the score is scale-free), so delta is fixed
on a grid and alpha / beta vary. Two families are reported:
- best overall (any weights),
- "model-led": the embedding weighs at least as much as the co-occurrence (alpha >= delta),
  so the ranking is driven by the GNN and an encoder gain (NLGCL vs LightGCN) reaches it.

Weights are chosen on val only; test is printed for the chosen settings, never searched.
Writes artifacts/eval/tune_ranking.json and .md.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from src.eval.harness import evaluate, from_hybrid_ranker
from src.eval.protocol import build_cases
from src.ranking.hybrid import _minmax, load_ranker, ranking_config_from_yaml

STAPLES = ("water", "salt", "ice")


def components(ranker, cases, sim: str) -> dict[str, np.ndarray]:
    """Per-case min-max scaled emb / chem / cooc and raw pref, shape (n_cases, n_items)."""
    n_items = len(ranker.vocab)
    emb_m = ranker.emb_unit if sim == "cosine" else ranker.raw_emb
    out = {k: np.zeros((len(cases), n_items), dtype=np.float32) for k in ("emb", "chem", "cooc", "pref")}
    ban = np.zeros((len(cases), n_items), dtype=bool)
    staples = [ranker.name_to_id[s] for s in STAPLES if s in ranker.name_to_id]
    for c, (q, uid) in enumerate(zip(cases["query_ids"], cases["user_id"])):
        q = [int(x) for x in q]
        out["emb"][c] = _minmax(emb_m @ emb_m[q].mean(axis=0))
        chem = np.zeros(n_items, dtype=np.float32)
        if ranker.chemical is not None:
            chem = np.asarray(ranker.chemical[q].sum(axis=0)).ravel().astype(np.float32) / len(q)
        out["chem"][c] = _minmax(chem)
        out["cooc"][c] = _minmax(np.asarray(ranker.cooc[q].sum(axis=0)).ravel().astype(np.float32))
        uid = None if uid is None or (isinstance(uid, float) and np.isnan(uid)) else int(uid)
        out["pref"][c] = ranker.preference(uid)
        ban[c, q] = True
        ban[c, staples] = True
    out["ban"] = ban
    return out


def target_matrix(cases, n_items: int) -> np.ndarray:
    t = np.zeros((len(cases), n_items), dtype=bool)
    for c, tg in enumerate(cases["target_ids"]):
        t[c, [int(x) for x in tg]] = True
    return t


def score_grid_point(comp, targets, a, b, d, g, k_max=20) -> dict[str, float]:
    s = a * comp["emb"] + b * comp["chem"] + d * comp["cooc"]
    if g:
        s = s * (1.0 + g * comp["pref"])
    s[comp["ban"]] = -np.inf
    top = np.argpartition(-s, k_max, axis=1)[:, :k_max]
    order = np.argsort(-np.take_along_axis(s, top, axis=1), axis=1)
    top = np.take_along_axis(top, order, axis=1)
    hits = np.take_along_axis(targets, top, axis=1).astype(np.float32)
    n_rel = targets.sum(axis=1).astype(np.float32)
    disc = 1.0 / np.log2(np.arange(2, k_max + 2))
    res = {}
    for k in (10, 20):
        dcg = hits[:, :k] @ disc[:k]
        idcg = np.array([disc[:min(int(n), k)].sum() for n in n_rel])
        res[f"ndcg@{k}"] = float(np.mean(np.where(idcg > 0, dcg / np.maximum(idcg, 1e-12), 0.0)))
        res[f"recall@{k}"] = float(np.mean(hits[:, :k].sum(axis=1) / np.maximum(n_rel, 1)))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--eval-config", default="configs/eval.yaml")
    p.add_argument("--models", nargs="+", default=["nlgcl=artifacts/ingredient_embeddings.npy",
                                                    "lightgcn=artifacts/ablations/lightgcn/ingredient_embeddings.npy"])
    p.add_argument("--max-cases", type=int, default=None)
    args = p.parse_args()

    with open(ROOT / args.eval_config, encoding="utf-8") as f:
        ecfg = yaml.safe_load(f)
    processed = ROOT / ecfg["processed_dir"]
    base = ranking_config_from_yaml()
    models = dict(m.split("=", 1) for m in args.models)

    alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]
    betas = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
    deltas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    gammas = [base.gamma_pref]
    sims = ["cosine", "dot"]

    setups = {split: build_cases(processed, split=split, protocol=ecfg["protocol"],
                                 hidden_ratio=ecfg["hidden_ratio"], min_query=ecfg["min_query"],
                                 seed=ecfg["seed"], max_cases=args.max_cases or ecfg.get("max_cases"),
                                 holdout_frac=ecfg["holdout_frac"]).cases
              for split in ("val", "test")}

    report = {"served": {"alpha_emb": base.alpha_emb, "beta_chem": base.beta_chem,
                         "delta_cooc": base.delta_cooc, "gamma_pref": base.gamma_pref,
                         "emb_similarity": base.emb_similarity}, "models": {}}
    for mname, emb_path in models.items():
        ranker = load_ranker(processed, ROOT / emb_path, replace(base, delta_cooc=base.delta_cooc or 0.7))
        n_items = len(ranker.vocab)
        comps = {}
        for split, cases in setups.items():
            t0 = time.time()
            comps[split] = {sim: components(ranker, cases, sim) for sim in sims}
            comps[split]["targets"] = target_matrix(cases, n_items)
            print(f"[{mname}] {split}: {len(cases):,} cases, components in {time.time() - t0:.0f}s")

        # sanity check: the vectorized scorer must match the real harness on the served config
        if args.max_cases is None or args.max_cases <= 500:
            sub = setups["val"].head(300)
            ref = evaluate(from_hybrid_ranker(load_ranker(processed, ROOT / emb_path, base)), sub, (10, 20))
            c = {k: v[:300] for k, v in comps["val"][base.emb_similarity].items()}
            mine = score_grid_point(c, comps["val"]["targets"][:300], base.alpha_emb, base.beta_chem,
                                    base.delta_cooc, base.gamma_pref)
            assert abs(ref["ndcg@10"] - mine["ndcg@10"]) < 2e-3, (ref["ndcg@10"], mine["ndcg@10"])
            print(f"[{mname}] check vs harness: ndcg@10 {ref['ndcg@10']:.4f} / {mine['ndcg@10']:.4f}")

        rows = []
        t0 = time.time()
        for sim, a, b, d, g in itertools.product(sims, alphas, betas, deltas, gammas):
            if a == b == d == 0:
                continue
            r = score_grid_point(comps["val"][sim], comps["val"]["targets"], a, b, d, g)
            rows.append({"emb_similarity": sim, "alpha_emb": a, "beta_chem": b, "delta_cooc": d,
                         "gamma_pref": g, **{f"val_{k}": v for k, v in r.items()}})
        print(f"[{mname}] {len(rows)} grid points in {time.time() - t0:.0f}s")

        def on_test(row):
            r = score_grid_point(comps["test"][row["emb_similarity"]], comps["test"]["targets"],
                                 row["alpha_emb"], row["beta_chem"], row["delta_cooc"], row["gamma_pref"])
            return {**row, **{f"test_{k}": v for k, v in r.items()}}

        def best(pred):
            cand = [r for r in rows if pred(r)]
            return on_test(max(cand, key=lambda r: r["val_ndcg@10"])) if cand else None

        served = on_test(next(r for r in rows if r["emb_similarity"] == base.emb_similarity
                              and r["alpha_emb"] == base.alpha_emb and r["beta_chem"] == base.beta_chem
                              and r["delta_cooc"] == base.delta_cooc))
        picks = {
            "served": served,
            "best": best(lambda r: True),
            "best_chem>0": best(lambda r: r["beta_chem"] > 0),
            # chemistry retuned only, everything else as served
            "served_best_beta": best(lambda r: r["emb_similarity"] == base.emb_similarity
                                     and r["alpha_emb"] == base.alpha_emb
                                     and r["delta_cooc"] == base.delta_cooc),
            "model_led": best(lambda r: r["alpha_emb"] >= r["delta_cooc"] and r["delta_cooc"] > 0),
            "model_led_chem>0": best(lambda r: r["alpha_emb"] >= r["delta_cooc"] and r["delta_cooc"] > 0
                                     and r["beta_chem"] > 0),
            "emb_only": best(lambda r: r["beta_chem"] == 0 and r["delta_cooc"] == 0),
        }
        # beta sweep at the best overall alpha / delta / similarity (how much chemistry costs)
        bo = picks["best"]
        sweep = [on_test(r) for r in rows if r["emb_similarity"] == bo["emb_similarity"]
                 and r["alpha_emb"] == bo["alpha_emb"] and r["delta_cooc"] == bo["delta_cooc"]]
        report["models"][mname] = {"picks": picks, "beta_sweep": sweep, "grid": rows}
        for name, r in picks.items():
            if r:
                print(f"[{mname}] {name:<18s} {r['emb_similarity']:<6s} a={r['alpha_emb']:<4} "
                      f"b={r['beta_chem']:<5} d={r['delta_cooc']:<4} "
                      f"val ndcg@10={r['val_ndcg@10']:.4f} test ndcg@10={r['test_ndcg@10']:.4f} "
                      f"test recall@20={r['test_recall@20']:.4f}")

    out = ROOT / ecfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tune_ranking.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    lines = ["# Ranking weights: grid search on val, test reported", "",
             "| model | pick | sim | alpha | beta | delta | val ndcg@10 | test ndcg@10 | test recall@20 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for mname, m in report["models"].items():
        for name, r in m["picks"].items():
            if r:
                lines.append(f"| {mname} | {name} | {r['emb_similarity']} | {r['alpha_emb']} | {r['beta_chem']} | "
                             f"{r['delta_cooc']} | {r['val_ndcg@10']:.4f} | {r['test_ndcg@10']:.4f} | "
                             f"{r['test_recall@20']:.4f} |")
    (out / "tune_ranking.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved -> {(out / 'tune_ranking').relative_to(ROOT)}.json / .md")


if __name__ == "__main__":
    main()
