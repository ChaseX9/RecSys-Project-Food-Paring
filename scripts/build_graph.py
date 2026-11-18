#!/usr/bin/env python3
"""Build the ingredient–ingredient graph (NPMI + chemical) and save to data/processed."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph.build_graph import build_ingredient_graph, save_graph


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--alpha", type=float, default=0.7, help="NPMI weight (1-alpha = chemical)")
    p.add_argument("--top-k", type=int, default=50, help="Max neighbors per ingredient")
    p.add_argument("--min-cooccur", type=int, default=2)
    args = p.parse_args()

    processed = ROOT / args.processed_dir
    print(f"Building graph from {processed} ...")
    graph = build_ingredient_graph(
        processed,
        alpha=args.alpha,
        top_k=args.top_k,
        min_cooccur=args.min_cooccur,
    )
    print("Meta:", graph["meta"])
    save_graph(graph, processed)
    print("Done.")


if __name__ == "__main__":
    main()
