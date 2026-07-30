import torch
from schema import TransformerEncoderConfig
from torch import nn

from .positional import PositionalEncoder


class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerEncoderConfig):
        super().__init__()

        self.input_proj = nn.Linear(10, config.d_model)

        self.pos_encoder = PositionalEncoder(
            d_model=config.d_model,
            max_regions=config.max_regions,
            scaling_factor=config.scaling_factor,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            norm_first=True,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        self.global_token = nn.Parameter(torch.randn(1, 1, config.d_model))

    def forward(self, regional_states):
        """
        regional_states shape:
            (T, M, feature_dim)

        Example:
            (100, 33, 10)
        """

        regional_states = self.input_proj(regional_states)
        # (T, M, 128)

        batch_size = regional_states.size(0)

        global_token = self.global_token.expand(batch_size, -1, -1)
        # (T, 1, 128)

        x = torch.cat([global_token, regional_states], dim=1)
        # (T, M+1, 128)

        x = self.pos_encoder(x)

        encoded = self.transformer_encoder(x)
        # (T, M+1, 128)

        global_embedding = encoded[:, 0, :]
        subregion_embeddings = encoded[:, 1:, :]

        return global_embedding, subregion_embeddings
