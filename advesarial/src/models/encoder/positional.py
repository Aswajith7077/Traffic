import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    """
    1. max_regions:    Maximum number of regions
    2. d_model:        Dimension of the model
    3. scaling_factor: Scaling factor for the positional encoding
    """

    def __init__(self, max_regions: int, d_model: int, scaling_factor: float = 10000.0):

        super().__init__()

        self.max_regions = max_regions
        self.d_model = d_model
        self.scaling_factor = scaling_factor

        positional_encoding = torch.zeros(max_regions, d_model)
        position = torch.arange(0, max_regions).unsqueeze(1).float()
        div_term = scaling_factor ** (2 * torch.arange(0, d_model, 2) / d_model)

        positional_encoding[:, 0::2] = torch.sin(position / div_term)
        positional_encoding[:, 1::2] = torch.cos(position / div_term)

        self.positional_encoding = positional_encoding

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.positional_encoding[:seq_len].unsqueeze(0)
