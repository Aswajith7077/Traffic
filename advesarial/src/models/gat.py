import torch
import torch.nn as nn


class GATLayer(nn.Module):
    """Graph Attention Concat (GAC), Section 4.2.2.

    For each node, concatenates its own embedding with up to 4 neighbors'
    attention-weighted embeddings (self ‖ neighbor_1 ‖ ... ‖ neighbor_4),
    zero-padding when fewer than 4 neighbors exist. This is concatenation,
    not the summation aggregation of a standard GAT — output dim is 5F.
    """

    def __init__(self, feature_dim, max_neighbors=4):
        super().__init__()
        self.F = feature_dim
        self.max_neighbors = max_neighbors
        self.attn = nn.Parameter(torch.randn(2 * feature_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, H, adj_list):
        N, feature_dim = H.shape
        Z = []

        for i in range(N):
            hi = H[i]
            neighbors = list(adj_list[i])[: self.max_neighbors]

            weighted = []
            if neighbors:
                scores = torch.stack(
                    [self.leaky_relu(torch.dot(self.attn, torch.cat([hi, H[j]]))) for j in neighbors]
                )
                alpha = torch.softmax(scores, dim=0)
                weighted = [alpha[k] * H[j] for k, j in enumerate(neighbors)]

            pad = self.max_neighbors - len(weighted)
            if pad > 0:
                weighted = weighted + [torch.zeros(feature_dim, device=H.device, dtype=H.dtype) for _ in range(pad)]

            zi = torch.cat([hi, *weighted], dim=-1)  # (5F,)
            Z.append(zi)

        return torch.stack(Z)  # (N, 5F)
