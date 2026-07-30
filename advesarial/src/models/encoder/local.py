import torch.nn as nn


class LocalEncoder(nn.Module):
    def __init__(self, in_dim=10, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, obs):
        return self.mlp(obs)
