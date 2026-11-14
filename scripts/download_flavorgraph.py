import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/lamypark/FlavorGraph/master/input"
FILES = ["nodes_191120.csv", "edges_191120.csv"]
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "flavorgraph"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        print("Downloading", name)
        urllib.request.urlretrieve(f"{BASE}/{name}", OUT / name)
