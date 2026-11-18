#!/usr/bin/env python3
"""Plot training / validation curves from a train_meta.json (Week 3 report, Q1).

    python scripts/plot_training.py                                  # artifacts/train_meta.json
    python scripts/plot_training.py --meta artifacts/runs/x/train_meta.json --out curves.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description="Plot NLGCL training curves")
    p.add_argument("--meta", default="artifacts/train_meta.json")
    p.add_argument("--out", default=None, help="default: training_curves.png next to the meta file")
    args = p.parse_args()

    meta_path = ROOT / args.meta
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    h = meta["history"]
    ep = [r["epoch"] for r in h]
    has_val = any("val_bpr" in r for r in h)
    vh = [r for r in h if "val_bpr" in r]
    vep = [r["epoch"] for r in vh]
    best = meta.get("best_epoch")

    n = 3 if has_val else 1
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4), squeeze=False)
    axes = axes[0]

    ax = axes[0]
    ax.plot(ep, [r["loss"] for r in h], label="total loss (BPR + lambda_cl * CL)")
    ax.plot(ep, [r["bpr"] for r in h], label="BPR (mini-batches)")
    ax.set_title("Training loss")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=8)

    if has_val:
        ax = axes[1]
        ax.plot(vep, [r["train_bpr_fixed"] for r in vh], label="train BPR (fixed train edges)")
        ax.plot(vep, [r["val_bpr"] for r in vh], label="val BPR (pairs of unseen val recipes)")
        ax.set_title("BPR loss: train vs validation")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)

        ax = axes[2]
        ax.plot(vep, [r["val_ndcg@10"] for r in vh], label="val NDCG@10")
        ax.plot(vep, [r["val_recall@20"] for r in vh], label="val Recall@20")
        ax.set_title("Validation ranking metrics (hybrid ranker)")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)

    if best:
        for ax in axes:
            ax.axvline(best, color="grey", ls="--", lw=1)
        axes[-1].annotate(f"best epoch {best}", (best, axes[-1].get_ylim()[0]),
                          textcoords="offset points", xytext=(4, 6), fontsize=8, color="grey")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"NLGCL  lambda_cl={meta.get('lambda_cl')}  epochs run={meta.get('epochs_run', len(h))}",
                 fontsize=10)
    fig.tight_layout()
    out = ROOT / args.out if args.out else meta_path.parent / "training_curves.png"
    fig.savefig(out, dpi=150)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
