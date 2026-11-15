"""Training loop for NLGCL on the ingredient graph."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.nlgcl import NLGCL, sparse_to_torch


class EdgeDataset(Dataset):
    """Observed positive edges (i, j) for BPR sampling."""

    def __init__(self, adj: sp.csr_matrix):
        coo = adj.tocoo()
        # keep upper triangle to avoid duplicates
        mask = coo.row < coo.col
        self.src = coo.row[mask].astype(np.int64)
        self.dst = coo.col[mask].astype(np.int64)
        self.n_nodes = adj.shape[0]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return int(self.src[idx]), int(self.dst[idx])


def collate_bpr(batch, n_nodes: int, n_neg: int = 1):
    src = torch.tensor([b[0] for b in batch], dtype=torch.long)
    pos = torch.tensor([b[1] for b in batch], dtype=torch.long)
    neg = torch.randint(0, n_nodes, (len(batch) * n_neg,), dtype=torch.long)
    if n_neg > 1:
        src = src.repeat_interleave(n_neg)
        pos = pos.repeat_interleave(n_neg)
    return src, pos, neg


def train_nlgcl(
    norm_adj: sp.csr_matrix,
    adj: sp.csr_matrix,
    emb_dim: int = 64,
    n_layers: int = 3,
    n_groups: int = 2,
    temperature: float = 0.2,
    lambda_cl: float = 1e-5,
    lambda_reg: float = 1e-4,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 4096,
    device: str = "cpu",
    seed: int = 42,
    log_every: int = 5,
) -> tuple[NLGCL, np.ndarray, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device)

    n_nodes = norm_adj.shape[0]
    model = NLGCL(
        n_nodes=n_nodes,
        emb_dim=emb_dim,
        n_layers=n_layers,
        n_groups=n_groups,
        temperature=temperature,
        scope="heterogeneous",
    ).to(device)
    model.set_neighbor_lists(adj)

    norm_adj_t = sparse_to_torch(norm_adj, device)
    adj_t = sparse_to_torch(adj, device)

    ds = EdgeDataset(adj)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_bpr(b, n_nodes),
        num_workers=0,
        drop_last=False,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "bpr": 0.0, "cl": 0.0}
        n_batches = 0
        for src, pos, neg in loader:
            src, pos, neg = src.to(device), pos.to(device), neg.to(device)
            # contrastive nodes = unique nodes in the batch
            cl_nodes = torch.unique(torch.cat([src, pos]))
            out = model.forward_loss(
                norm_adj_t, adj_t, src, pos, neg,
                cl_nodes=cl_nodes,
                lambda_cl=lambda_cl,
                lambda_reg=lambda_reg,
            )
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            for k in totals:
                v = out[k]
                totals[k] += float(v.detach() if hasattr(v, 'detach') else v)
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
        avg["epoch"] = epoch
        history.append(avg)
        if epoch % log_every == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"epoch {epoch:03d}  loss={avg['loss']:.4f}  "
                f"bpr={avg['bpr']:.4f}  cl={avg['cl']:.4f}  ({elapsed:.1f}s)"
            )

    model.eval()
    with torch.no_grad():
        final, _ = model.encode(norm_adj_t)
        embeddings = final.cpu().numpy()

    meta = {
        "n_nodes": n_nodes,
        "emb_dim": emb_dim,
        "n_layers": n_layers,
        "n_groups": n_groups,
        "epochs": epochs,
        "history": history,
        "time_sec": time.time() - t0,
    }
    return model, embeddings, meta


def save_artifacts(
    embeddings: np.ndarray,
    meta: dict,
    model: NLGCL | None,
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "ingredient_embeddings.npy", embeddings)
    with open(out_dir / "train_meta.json", "w") as f:
        # history may contain non-serializable floats already ok
        json.dump(meta, f, indent=2)
    if model is not None:
        torch.save(model.state_dict(), out_dir / "nlgcl_state.pt")
    print(f"Saved embeddings {embeddings.shape} → {out_dir}")
