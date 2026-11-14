"""NLGCL-inspired LightGCN for ingredient–ingredient graphs.

Key ideas from the paper (RecSys '25):
  - No graph augmentations.
  - Contrastive views = consecutive GNN layers.
  - Heterogeneous-scope positives: node at layer g ↔ its graph neighbors at layer g+1.
  - Multi-task: BPR on observed pairs + neighbor-layer InfoNCE + L2.

Adapted to a homogeneous ingredient graph (pairing instead of user–item).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_to_torch(mat: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    """CSR → torch sparse_coo_tensor on device."""
    mat = mat.tocoo()
    indices = torch.from_numpy(
        np.vstack([mat.row, mat.col]).astype(np.int64)
    ).to(device)
    values = torch.from_numpy(mat.data.astype(np.float32)).to(device)
    return torch.sparse_coo_tensor(indices, values, mat.shape).coalesce()


class LightGCNEncoder(nn.Module):
    """Plain LightGCN (no non-linearities, no weight matrices)."""

    def __init__(self, n_nodes: int, emb_dim: int, n_layers: int, dropout: float = 0.0):
        super().__init__()
        self.n_nodes = n_nodes
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(n_nodes, emb_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, norm_adj: torch.Tensor) -> list[torch.Tensor]:
        """Return list of layer embeddings [E^(0), E^(1), ..., E^(L)]."""
        ego = self.embedding.weight
        if self.dropout > 0 and self.training:
            # edge dropout on the sparse adj is expensive; simple feature dropout instead
            ego = F.dropout(ego, p=self.dropout, training=True)
        layers = [ego]
        x = ego
        for _ in range(self.n_layers):
            x = torch.sparse.mm(norm_adj, x)
            layers.append(x)
        return layers

    def final_embedding(self, layers: list[torch.Tensor]) -> torch.Tensor:
        """Mean over layers (LightGCN readout)."""
        return torch.stack(layers, dim=0).mean(dim=0)


class NLGCL(nn.Module):
    """Full model: encoder + BPR + neighbor-layer contrastive loss."""

    def __init__(
        self,
        n_nodes: int,
        emb_dim: int = 64,
        n_layers: int = 3,
        n_groups: int = 2,
        temperature: float = 0.2,
        dropout: float = 0.0,
        scope: str = "heterogeneous",  # "heterogeneous" | "entire"
    ):
        super().__init__()
        assert scope in ("heterogeneous", "entire")
        self.encoder = LightGCNEncoder(n_nodes, emb_dim, n_layers, dropout)
        self.n_layers = n_layers
        self.n_groups = min(n_groups, n_layers)  # G ≤ L (Theorem 1: prefer lower layers)
        self.temperature = temperature
        self.scope = scope
        self._neighbor_lists: list[np.ndarray] | None = None

    def set_neighbor_lists(self, adj: sp.csr_matrix) -> None:
        """Pre-compute neighbor index arrays for fast positive sampling."""
        adj = adj.tocsr()
        self._neighbor_lists = [
            adj.indices[adj.indptr[i] : adj.indptr[i + 1]].copy()
            for i in range(adj.shape[0])
        ]

    def encode(self, norm_adj: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        layers = self.encoder(norm_adj)
        final = self.encoder.final_embedding(layers)
        return final, layers

    # ------------------------------------------------------------------ BPR
    def bpr_loss(
        self,
        emb: torch.Tensor,
        src: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
    ) -> torch.Tensor:
        """Classic BPR on (src, pos, neg) triples."""
        s = emb[src]
        p = emb[pos]
        n = emb[neg]
        pos_score = (s * p).sum(dim=-1)
        neg_score = (s * n).sum(dim=-1)
        return -F.logsigmoid(pos_score - neg_score).mean()

    # --------------------------------------------------------- NLGCL contrastive
    def neighbor_layer_loss(
        self,
        layers: list[torch.Tensor],
        adj: torch.Tensor,
        node_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Simplified but faithful heterogeneous-scope NLGCL loss.

        For each group g ∈ {0 … G-1}:
          positive: node u at layer g  ↔  neighbors of u at layer g+1
          negatives: other nodes (in-batch or full, depending on memory)

        We use a practical in-batch approximation: for a mini-batch of nodes,
        positives are pulled via the sparse adjacency mask, negatives are the
        rest of the batch.
        """
        if node_ids is None:
            # full-graph (small graphs only)
            node_ids = torch.arange(layers[0].size(0), device=layers[0].device)

        # Build dense neighbor mask for the batch (B x N) is too big;
        # instead we sample a fixed number of positives per node from adj.
        loss = 0.0
        n_valid = 0
        tau = self.temperature

        for g in range(self.n_groups):
            e_g = layers[g][node_ids]           # (B, d)
            e_next = layers[g + 1]              # (N, d)

            # Sample positive neighbors for each batch node from the (normalized) adj
            # We use the original (non-normalized) adj for neighbor identity.
            # For efficiency: one random positive neighbor per node when available.
            pos_idx = self._sample_neighbors(adj, node_ids)  # (B,) long, -1 if isolated
            valid = pos_idx >= 0
            if valid.sum() == 0:
                continue

            e_g_v = e_g[valid]
            pos_v = e_next[pos_idx[valid]]
            # InfoNCE with in-batch negatives (all other valid nodes' next-layer emb)
            # positives: (B_v, d), keys: (B_v, d) for pos + we use all e_next[node_ids] as candidates
            candidates = e_next[node_ids]  # (B, d)
            logits = (e_g_v @ candidates.T) / tau  # (B_v, B)
            # positive column index inside the batch
            # map pos_idx → position in node_ids
            # For simplicity treat the sampled neighbor embedding as the sole positive
            # and use other batch members as negatives (standard SimCLR-style).
            pos_sim = (e_g_v * pos_v).sum(-1, keepdim=True) / tau  # (B_v, 1)
            # logits against batch
            neg_logits = logits
            # concatenate positive as first column
            logits_full = torch.cat([pos_sim, neg_logits], dim=1)
            labels = torch.zeros(logits_full.size(0), dtype=torch.long, device=logits_full.device)
            loss = loss + F.cross_entropy(logits_full, labels)
            n_valid += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=layers[0].device)
        return loss / n_valid

    def _sample_neighbors(self, adj: torch.Tensor, node_ids: torch.Tensor) -> torch.Tensor:
        """One random neighbor per node; -1 if degree 0. Uses precomputed lists when available."""
        out = torch.full((node_ids.size(0),), -1, dtype=torch.long, device=node_ids.device)
        if self._neighbor_lists is not None:
            ids = node_ids.tolist()
            for bi, n in enumerate(ids):
                neigh = self._neighbor_lists[n]
                if len(neigh) > 0:
                    out[bi] = int(neigh[np.random.randint(len(neigh))])
            return out
        # fallback: sparse adj scan (slow)
        adj = adj.coalesce()
        indices = adj.indices()
        row, col = indices[0].tolist(), indices[1].tolist()
        from collections import defaultdict
        neigh = defaultdict(list)
        node_set = {int(n): i for i, n in enumerate(node_ids.tolist())}
        for r, c in zip(row, col):
            if r in node_set:
                neigh[r].append(c)
        for n, bi in node_set.items():
            if neigh[n]:
                out[bi] = neigh[n][np.random.randint(len(neigh[n]))]
        return out

    def forward_loss(
        self,
        norm_adj: torch.Tensor,
        adj: torch.Tensor,
        src: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
        cl_nodes: Optional[torch.Tensor] = None,
        lambda_cl: float = 1e-5,
        lambda_reg: float = 1e-4,
    ) -> dict[str, torch.Tensor]:
        final, layers = self.encode(norm_adj)
        bpr = self.bpr_loss(final, src, pos, neg)
        cl = self.neighbor_layer_loss(layers, adj, cl_nodes)
        reg = self.encoder.embedding.weight.pow(2).mean()
        total = bpr + lambda_cl * cl + lambda_reg * reg
        return {"loss": total, "bpr": bpr.detach(), "cl": cl.detach(), "reg": reg.detach()}
