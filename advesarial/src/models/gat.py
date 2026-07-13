import torch
import torch.nn as nn


class GATLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attn = nn.Parameter(torch.randn(2 * feature_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, H, adj_list):
        N, feature_dim = H.shape
        Z = []

        for i in range(N):
            hi = H[i]
            neighbors = adj_list[i]

            if len(neighbors) == 0:
                Z.append(hi)
                continue

            scores = []
            h_neighbors = []

            for j in neighbors:
                hj = H[j]
                e_ij = self.leaky_relu(torch.dot(self.attn, torch.cat([hi, hj])))
                scores.append(e_ij)
                h_neighbors.append(hj)

            scores = torch.stack(scores)
            alpha = torch.softmax(scores, dim=0)

            agg = sum(alpha[k] * h_neighbors[k] for k in range(len(neighbors)))

            zi = hi + agg  # variable neighbors supported
            Z.append(zi)

        return torch.stack(Z)  # (N, feature_dim)
