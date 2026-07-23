import torch.nn as nn
from schema import TransformerEncoderConfig

from .positional import PositionalEncoding


class TransformerEncoding(nn.Module):
    def __init__(self, config: TransformerEncoderConfig):
        super().__init__()

        self.positional_encoder = PositionalEncoding(config.d_model, config.max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=config.num_layers
        )

    def forward(self, x):
        """
        x: (batch, M+1, d_model)
        """
        x = self.positional_encoder(x)
        x = self.encoder(x)
        return x
