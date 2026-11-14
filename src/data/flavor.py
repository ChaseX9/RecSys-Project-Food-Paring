from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .ingredients import normalize_ingredient

COMPOUND_EDGES = ("ingr-fcomp", "ingr-dcomp")


def load_flavorgraph(directory: Path, nodes_file: str, edges_file: str):
    # keep_default_na=False: some ingredient names ("nan", "NA") must stay strings
    nodes = pd.read_csv(directory / nodes_file, keep_default_na=False)
    edges = pd.read_csv(directory / edges_file)
    return nodes, edges


def map_vocab_to_flavorgraph(vocab: pd.DataFrame, nodes: pd.DataFrame,
                             aliases: dict[str, str] | None = None) -> pd.DataFrame:
    """One row per (vocab ingredient, FlavorGraph node) pair matched on the normalized name."""
    ing = nodes[nodes["node_type"] == "ingredient"].copy()
    ing["norm"] = [normalize_ingredient(n.replace("_", " "), aliases) for n in ing["name"]]
    merged = vocab[["ingredient_id", "name"]].merge(
        ing[["node_id", "norm"]], left_on="name", right_on="norm", how="inner")
    return merged[["ingredient_id", "name", "node_id"]]


def build_compound_matrix(mapping: pd.DataFrame, nodes: pd.DataFrame, edges: pd.DataFrame,
                          n_ingredients: int):
    """Binary ingredients x compounds matrix; FlavorGraph nodes merged to the same ingredient are unioned."""
    node_type = nodes.set_index("node_id")["node_type"]
    e = edges[edges["edge_type"].isin(COMPOUND_EDGES)]
    # edges are stored as (ingredient, compound), but guard against the reverse order
    swap = node_type.reindex(e["id_1"]).to_numpy() == "compound"
    ing_node = np.where(swap, e["id_2"], e["id_1"])
    comp_node = np.where(swap, e["id_1"], e["id_2"])
    pairs = pd.DataFrame({"node_id": ing_node, "compound_node": comp_node}).merge(
        mapping[["ingredient_id", "node_id"]], on="node_id")
    pairs = pairs[["ingredient_id", "compound_node"]].drop_duplicates()

    comp_nodes = np.sort(pairs["compound_node"].unique())
    col = {c: i for i, c in enumerate(comp_nodes)}
    rows = pairs["ingredient_id"].to_numpy()
    cols = pairs["compound_node"].map(col).to_numpy()
    mat = sp.csr_matrix((np.ones(len(pairs), dtype=np.float32), (rows, cols)),
                        shape=(n_ingredients, len(comp_nodes)))
    compounds = nodes.set_index("node_id").loc[comp_nodes, "name"].reset_index()
    compounds.insert(0, "compound_idx", np.arange(len(comp_nodes)))
    return mat, compounds


def jaccard_similarity(mat: sp.csr_matrix) -> sp.csr_matrix:
    """Jaccard similarity of compound sets between ingredients (zero diagonal, sparse)."""
    inter = (mat @ mat.T).tocoo()
    size = np.asarray(mat.sum(axis=1)).ravel()
    keep = inter.row != inter.col
    r, c, v = inter.row[keep], inter.col[keep], inter.data[keep]
    sim = v / (size[r] + size[c] - v)
    return sp.csr_matrix((sim.astype(np.float32), (r, c)), shape=(mat.shape[0], mat.shape[0]))


def build_flavorgraph_cooccurrence(mapping: pd.DataFrame, edges: pd.DataFrame, n_ingredients: int):
    """Ingredient-ingredient scores from FlavorGraph ('ingr-ingr' edges), max over merged nodes."""
    e = edges[edges["edge_type"] == "ingr-ingr"][["id_1", "id_2", "score"]]
    m = mapping[["ingredient_id", "node_id"]]
    e = e.merge(m.rename(columns={"ingredient_id": "a", "node_id": "id_1"}), on="id_1")
    e = e.merge(m.rename(columns={"ingredient_id": "b", "node_id": "id_2"}), on="id_2")
    e = e[e["a"] != e["b"]]
    e = pd.concat([e[["a", "b", "score"]], e.rename(columns={"a": "b", "b": "a"})[["a", "b", "score"]]])
    e = e.groupby(["a", "b"], as_index=False)["score"].max()
    return sp.csr_matrix((e["score"].to_numpy(dtype=np.float32), (e["a"], e["b"])),
                         shape=(n_ingredients, n_ingredients))


def build_flavor_features(vocab: pd.DataFrame, nodes: pd.DataFrame, edges: pd.DataFrame,
                          aliases: dict[str, str] | None, out: Path) -> dict:
    n = len(vocab)
    mapping = map_vocab_to_flavorgraph(vocab, nodes, aliases)
    comp_mat, compounds = build_compound_matrix(mapping, nodes, edges, n)
    chem = jaccard_similarity(comp_mat)
    cooc = build_flavorgraph_cooccurrence(mapping, edges, n)

    mapped_ids = set(mapping["ingredient_id"])
    with_comp = np.asarray(comp_mat.sum(axis=1)).ravel() > 0
    weight = vocab["n_recipes"].to_numpy(dtype=float)
    top100 = slice(0, 100)

    status = pd.DataFrame({"ingredient_id": vocab["ingredient_id"], "name": vocab["name"],
                           "in_flavorgraph": vocab["ingredient_id"].isin(mapped_ids),
                           "has_compounds": with_comp})
    status.to_csv(out / "flavor_mapping.csv", index=False)
    compounds.to_csv(out / "compounds.csv", index=False)
    sp.save_npz(out / "ingredient_compounds.npz", comp_mat)
    sp.save_npz(out / "chemical_similarity.npz", chem)
    sp.save_npz(out / "flavorgraph_cooccurrence.npz", cooc)

    in_fg = status["in_flavorgraph"].to_numpy()
    return {
        "flavorgraph_matched_ingredients": int(in_fg.sum()),
        "flavorgraph_matched_share_recipe_weighted": float(weight[in_fg].sum() / weight.sum()),
        "ingredients_with_compounds": int(with_comp.sum()),
        "ingredients_with_compounds_share_recipe_weighted": float(weight[with_comp].sum() / weight.sum()),
        "top100_ingredients_with_compounds": int(with_comp[top100].sum()),
        "n_compounds": int(comp_mat.shape[1]),
        "chemical_similarity_nonzero_pairs": int(chem.nnz),
    }
