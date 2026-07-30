import torch
import torch.nn as nn


class SubGoalGenerator(nn.Module):
    def __init__(self, d_reg, d_hidden, M, d_g, num_layers=1):
        super().__init__()

        self.M = M
        self.d_hidden = d_hidden

        # LSTM processes each subregion sequence
        self.lstm = nn.LSTM(
            input_size=d_reg,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
        )

        self.ffn = nn.Sequential(nn.Linear(M * d_hidden + d_reg, d_g), nn.ReLU(), nn.Linear(d_g, d_g))

    def forward(self, locals, global_encoding):
        # locals: (batch_size, M, d_reg)
        # global_encoding: (batch_size, d_reg)

        out, _ = self.lstm(locals)  # (batch_size, M, d_hidden)

        batch_size = locals.size(0)
        out_flat = out.reshape(batch_size, -1)  # (batch_size, M * d_hidden)

        combined = torch.cat([out_flat, global_encoding], dim=-1)  # (batch_size, M * d_hidden + d_reg)

        subgoal_vector = self.ffn(combined)  # (batch_size, d_g)
        return subgoal_vector
