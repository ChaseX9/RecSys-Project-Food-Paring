import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.build import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build data/processed from data/raw")
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    stats = run(args.config, ROOT)
    print("\n=== Summary ===")
    for k, v in stats.items():
        if not isinstance(v, (list, dict)):
            print(f"{k}: {v}")
    print("split_sizes:", stats["split_sizes"])
