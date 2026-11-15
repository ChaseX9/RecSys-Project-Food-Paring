#!/usr/bin/env python3
"""Train NLGCL on the ingredient graph and write embeddings to artifacts/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scipy.sparse as sp
import yaml

from src.graph.build_graph import build_ingredient_graph, save_graph
from src.training.train import save_artifacts, train_nlgcl


def main():
    p = argparse.ArgumentParser(description="Train NLGCL ingredient embeddings")
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--out-dir", default="artifacts")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--rebuild-graph", action="store_true")
    args = p.parse_args()

    cfg_path = ROOT / args.config
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}

    processed = ROOT / args.processed_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_path = processed / "ingredient_norm_adj.npz"
    if args.rebuild_graph or not graph_path.exists():
        print("Building ingredient graph …")
        gcfg = cfg.get("graph", {})
        graph = build_ingredient_graph(
            processed,
            alpha=gcfg.get("alpha", 0.7),
            top_k=gcfg.get("top_k", 50),
            min_cooccur=gcfg.get("min_cooccur", 2),
        )
        save_graph(graph, processed)
        print("Graph meta:", graph["meta"])
    else:
        print(f"Loading existing graph from {processed}")
        graph = {
            "norm_adj": sp.load_npz(processed / "ingredient_norm_adj.npz").tocsr(),
            "adj": sp.load_npz(processed / "ingredient_adj.npz").tocsr(),
        }

    mcfg = cfg.get("model", {})
    tcfg = cfg.get("train", {})
    epochs = args.epochs or tcfg.get("epochs", 40)

    print(
        f"Training NLGCL  emb={mcfg.get('emb_dim', 64)}  L={mcfg.get('n_layers', 3)}  "
        f"G={mcfg.get('n_groups', 2)}  epochs={epochs}  device={args.device}"
    )
    model, embeddings, meta = train_nlgcl(
        norm_adj=graph["norm_adj"],
        adj=graph["adj"],
        emb_dim=mcfg.get("emb_dim", 64),
        n_layers=mcfg.get("n_layers", 3),
        n_groups=mcfg.get("n_groups", 2),
        temperature=mcfg.get("temperature", 0.2),
        lambda_cl=tcfg.get("lambda_cl", 1e-5),
        lambda_reg=tcfg.get("lambda_reg", 1e-4),
        lr=tcfg.get("lr", 1e-3),
        epochs=epochs,
        batch_size=tcfg.get("batch_size", 4096),
        device=args.device,
        seed=tcfg.get("seed", 42),
        log_every=tcfg.get("log_every", 5),
    )
    save_artifacts(embeddings, meta, model, out_dir)
    print("Training finished.")


if __name__ == "__main__":
    main()
