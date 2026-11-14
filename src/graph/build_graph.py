"""Ingredient–ingredient graph construction for NLGCL-style training.

Edges combine:
  - NPMI from co-occurrence in *train* recipes only (no leakage)
  - chemical (Jaccard) similarity from FlavorGraph compounds

The resulting sparse adjacency is symmetrically normalized for LightGCN.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def npmi_from_cooccurrence(
    recipe_ing: sp.csr_matrix,
    train_mask: np.ndarray | None = None,
    min_count: int = 2,
    eps: float = 1e-12,
) -> sp.csr_matrix:
    """Pointwise Mutual Information (normalized) between ingredients.

    Parameters
    ----------
    recipe_ing : (n_recipes, n_ingredients) binary CSR
    train_mask : bool array of length n_recipes; if given, only those rows are used
    min_count  : drop pairs with fewer than this many co-occurrences
    """
    if train_mask is not None:
        recipe_ing = recipe_ing[train_mask]

    n_recipes = recipe_ing.shape[0]
    # ingredient frequency (number of train recipes containing i)
    freq = np.asarray(recipe_ing.sum(axis=0)).ravel().astype(np.float64)
    # co-occurrence matrix
    co = (recipe_ing.T @ recipe_ing).tocoo()
    # remove diagonal and rare pairs
    keep = (co.row != co.col) & (co.data >= min_count)
    i, j, c = co.row[keep], co.col[keep], co.data[keep].astype(np.float64)

    p_i = freq[i] / n_recipes
    p_j = freq[j] / n_recipes
    p_ij = c / n_recipes
    pmi = np.log((p_ij + eps) / (p_i * p_j + eps))
    # NPMI in [-1, 1]
    npmi = pmi / (-np.log(p_ij + eps))
    # keep only positive associations
    pos = npmi > 0
    i, j, v = i[pos], j[pos], npmi[pos].astype(np.float32)
    n = recipe_ing.shape[1]
    return sp.csr_matrix((v, (i, j)), shape=(n, n))


def fuse_graphs(
    npmi: sp.csr_matrix,
    chemical: sp.csr_matrix | None,
    alpha: float = 0.7,
    top_k: int | None = 50,
) -> sp.csr_matrix:
    """Weighted sum of NPMI and chemical similarity, optional top-k sparsification.

    alpha : weight on NPMI (1-alpha on chemical). If chemical is None, uses pure NPMI.
    top_k : keep at most top_k neighbors per node (after fusion); None = keep all.
    """
    if chemical is None or chemical.nnz == 0:
        fused = npmi.copy()
    else:
        # align shapes
        n = npmi.shape[0]
        chem = chemical[:n, :n].tocsr()
        # scale chemical to [0,1] if needed (Jaccard already is)
        fused = (alpha * npmi + (1.0 - alpha) * chem).tocsr()
        fused.eliminate_zeros()

    if top_k is not None and top_k > 0:
        fused = _topk_per_row(fused, top_k)

    # symmetrize (average)
    fused = ((fused + fused.T) * 0.5).tocsr()
    fused.eliminate_zeros()
    return fused


def _topk_per_row(mat: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Keep the k largest entries per row."""
    mat = mat.tocsr()
    rows, cols, data = [], [], []
    for i in range(mat.shape[0]):
        start, end = mat.indptr[i], mat.indptr[i + 1]
        if start == end:
            continue
        idx = mat.indices[start:end]
        val = mat.data[start:end]
        if len(val) > k:
            top = np.argpartition(val, -k)[-k:]
            idx, val = idx[top], val[top]
        rows.append(np.full(len(idx), i, dtype=np.int32))
        cols.append(idx.astype(np.int32))
        data.append(val.astype(np.float32))
    if not rows:
        return sp.csr_matrix(mat.shape, dtype=np.float32)
    return sp.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=mat.shape,
    )


def normalize_adjacency(adj: sp.csr_matrix) -> sp.csr_matrix:
    """Symmetric normalization D^{-1/2} A D^{-1/2} (zero-degree nodes stay zero)."""
    adj = adj.tocsr().astype(np.float32)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg_inv_sqrt = np.zeros_like(deg)
    mask = deg > 0
    deg_inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    D = sp.diags(deg_inv_sqrt)
    return (D @ adj @ D).tocsr()


def build_ingredient_graph(
    processed_dir: str | Path,
    alpha: float = 0.7,
    top_k: int = 50,
    min_cooccur: int = 2,
) -> dict:
    """End-to-end graph builder. Returns dict with adj, norm_adj, npmi, meta."""
    processed_dir = Path(processed_dir)
    recipe_ing = sp.load_npz(processed_dir / "recipe_ingredient.npz").tocsr()
    recipes = pd.read_parquet(processed_dir / "recipes.parquet")
    train_mask = recipes["is_train_recipe"].to_numpy() if "is_train_recipe" in recipes.columns else None

    npmi = npmi_from_cooccurrence(recipe_ing, train_mask, min_count=min_cooccur)

    chem_path = processed_dir / "chemical_similarity.npz"
    chemical = sp.load_npz(chem_path).tocsr() if chem_path.exists() else None

    adj = fuse_graphs(npmi, chemical, alpha=alpha, top_k=top_k)
    norm_adj = normalize_adjacency(adj)

    meta = {
        "n_ingredients": adj.shape[0],
        "n_edges": int(adj.nnz // 2),
        "npmi_nnz": int(npmi.nnz),
        "chemical_nnz": int(chemical.nnz) if chemical is not None else 0,
        "alpha": alpha,
        "top_k": top_k,
    }
    return {
        "adj": adj,
        "norm_adj": norm_adj,
        "npmi": npmi,
        "chemical": chemical,
        "meta": meta,
    }


def save_graph(graph: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_dir / "ingredient_adj.npz", graph["adj"])
    sp.save_npz(out_dir / "ingredient_norm_adj.npz", graph["norm_adj"])
    sp.save_npz(out_dir / "ingredient_npmi.npz", graph["npmi"])
    if graph.get("chemical") is not None:
        sp.save_npz(out_dir / "ingredient_chemical.npz", graph["chemical"])
    import json
    with open(out_dir / "graph_meta.json", "w") as f:
        json.dump(graph["meta"], f, indent=2)


def load_norm_adj(processed_dir: str | Path) -> sp.csr_matrix:
    path = Path(processed_dir) / "ingredient_norm_adj.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/build_graph.py first."
        )
    return sp.load_npz(path).tocsr()
