import torch
import torch.nn as nn

from .gat import GATLayer


class SubPolicy(nn.Module):
    def __init__(self, k, F):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(k, F), nn.ReLU(), nn.Linear(F, F))
        self.gat = GATLayer(F)

    def fuse_global(self, z, global_feat):
        # z: (N, 5F)
        # global_feat: F_g, the Meta-Policy's current-timestep global
        # Transformer embedding — shape (d_reg,) or (N, d_reg).

        if global_feat.dim() == 1:
            global_feat = global_feat.unsqueeze(0).repeat(z.size(0), 1)

        return torch.cat([z, global_feat], dim=1)

    def forward(self, o, adj_list, global_feat):
        # Step 1: encode
        h = self.encoder(o)  # (N, F)

        # Step 2: graph attention concat (GAC)
        z = self.gat(h, adj_list)  # (N, 5F)

        # Step 3: fuse with Meta-Policy's global feature F_g
        final = self.fuse_global(z, global_feat)  # (N, 5F + d_reg)

        return final
