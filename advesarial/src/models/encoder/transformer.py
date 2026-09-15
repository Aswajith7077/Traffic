import torch
from schema import TransformerEncoderConfig
from torch import nn

from .positional import PositionalEncoder


class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerEncoderConfig):
        super().__init__()

        self.input_proj = nn.Linear(config.d_reg, config.d_model)

        self.pos_encoder = PositionalEncoder(
            d_model=config.d_model,
            max_regions=config.max_regions,
            scaling_factor=config.scaling_factor,
        )

        # Table 6 specifies "no layer norm"; nn.TransformerEncoderLayer always
        # keeps its two internal per-layer norms (no vanilla-PyTorch way to
        # drop those without a custom block), but we at least skip the extra
        # final norm nn.TransformerEncoder would otherwise apply, and use
        # norm_first=False so the layer's own norms don't dominate the tiny
        # d_model=4 residual stream.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            norm_first=False,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        self.global_token = nn.Parameter(torch.randn(1, 1, config.d_model))

    def forward(self, regional_states):
        """
        regional_states: (T, M, d_reg) — T past regional-state snapshots
        (Meta-Policy's temporal window) across M subregions.

        Returns:
            global_embedding: (T, d_model) — E_G^t per timestep; the current
                global feature F_g fed to the Sub-Policy is global_embedding[-1].
            subregion_embeddings: (T, M, d_model) — E_z^t per timestep, fed to
                the LSTM sub-goal generator.
        """

        regional_states = self.input_proj(regional_states)
        # (T, M, d_model)

        batch_size = regional_states.size(0)

        global_token = self.global_token.expand(batch_size, -1, -1)
        # (T, 1, d_model)

        x = torch.cat([global_token, regional_states], dim=1)
        # (T, M+1, d_model)

        x = self.pos_encoder(x)

        encoded = self.transformer_encoder(x)
        # (T, M+1, d_model)

        global_embedding = encoded[:, 0, :]
        subregion_embeddings = encoded[:, 1:, :]

        return global_embedding, subregion_embeddings
