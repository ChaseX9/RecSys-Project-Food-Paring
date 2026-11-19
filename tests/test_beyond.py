import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.eval.beyond import beyond_accuracy, longtail_mask


def _cases():
    return pd.DataFrame({"case_id": [0, 1], "query_ids": [[0], [1]], "target_ids": [[2], [3]]})


def test_longtail_mask_excludes_head():
    pop = np.array([100, 50, 10, 5, 1], dtype=float)
    assert longtail_mask(pop, 0.4).tolist() == [False, False, True, True, True]


def test_coverage_and_longtail_hand_computed():
    pop = np.array([100, 50, 10, 5, 1], dtype=float)   # head (40%) = items 0 and 1
    preds = {0: [0, 1, 2], 1: [1, 4]}                  # query items are dropped first
    r = beyond_accuracy(preds, _cases(), pop, n_recipes=100, k=10, head_frac=0.4)
    # recommended after removing queries: case 0 -> [1, 2], case 1 -> [4]
    assert r["coverage@10"] == pytest.approx(3 / 5)
    assert r["longtail@10"] == pytest.approx(2 / 3)
    assert "grounded@10" not in r


def test_grounded_uses_npmi_or_chemical():
    pop = np.ones(5)
    npmi = sp.csr_matrix(([0.5], ([0], [2])), shape=(5, 5))   # 0-2 strongly paired
    chem = sp.csr_matrix(([0.2], ([1], [4])), shape=(5, 5))   # 1-4 share compounds
    preds = {0: [2, 3], 1: [4, 3]}
    r = beyond_accuracy(preds, _cases(), pop, n_recipes=10, k=10, npmi=npmi, chemical=chem)
    assert r["grounded@10"] == pytest.approx(2 / 4)
