"""The trained model is versioned so a fresh clone serves real recommendations without training.
These tests catch a model that no longer matches the data (e.g. vocabulary rebuilt, model not
retrained), which otherwise only fails at the first /recommend."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ranking.hybrid import load_ranker

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "artifacts" / "ingredient_embeddings.npy"
META = ROOT / "artifacts" / "train_meta.json"
VOCAB = ROOT / "data" / "processed" / "ingredients_vocab.csv"


def test_versioned_model_matches_versioned_vocabulary():
    assert EMB.exists(), "artifacts/ingredient_embeddings.npy must be committed (run scripts/train_nlgcl.py)"
    emb = np.load(EMB, mmap_mode="r")
    n_vocab = len(pd.read_csv(VOCAB))
    assert emb.shape[0] == n_vocab, (
        f"model has {emb.shape[0]} ingredients, vocabulary has {n_vocab}: "
        "retrain with scripts/train_nlgcl.py and commit artifacts/ingredient_embeddings.npy")
    assert np.isfinite(emb).all()
    meta = json.loads(META.read_text(encoding="utf-8"))
    assert meta["n_nodes"] == n_vocab and meta["emb_dim"] == emb.shape[1]


def test_load_ranker_rejects_out_of_date_model(tmp_path):
    pd.DataFrame({"ingredient_id": range(3), "name": ["a", "b", "c"]}).to_csv(
        tmp_path / "ingredients_vocab.csv", index=False)
    np.save(tmp_path / "emb.npy", np.zeros((4, 8)))
    with pytest.raises(ValueError, match="out of date"):
        load_ranker(tmp_path, tmp_path / "emb.npy")
