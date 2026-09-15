import torch.nn as nn


class SubGoalGenerator(nn.Module):
    """LSTM-based sub-goal generation, Section 4.1.2.

    Runs an LSTM over the temporal axis (T) independently per subregion,
    takes each subregion's final hidden state h_z^T, concatenates across
    subregions M, and projects to a d_g-dim sub-goal G via a feedforward phi.
    A small linear head then splits G into the two scalars (G_w, G_q) that
    the adversarial/goal-reward losses compare against (W_global, Q_global).
    """

    def __init__(self, d_model, d_hidden, M, d_g, num_layers=4):
        super().__init__()

        self.M = M
        self.d_hidden = d_hidden

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
        )

        self.phi = nn.Sequential(nn.Linear(M * d_hidden, d_g), nn.ReLU(), nn.Linear(d_g, d_g))
        self.goal_head = nn.Linear(d_g, 2)

    def forward(self, subregion_embeddings):
        """
        subregion_embeddings: (T, M, d_model) — per-timestep subregion
        embeddings from the Transformer encoder, across the T-step window.

        Returns:
            G: (1, d_g) sub-goal vector
            goal: (1, 2) -> (G_w, G_q)
        """

        T, M, d_model = subregion_embeddings.shape

        # LSTM sees each subregion as one batch element, T as its sequence axis.
        x = subregion_embeddings.permute(1, 0, 2)  # (M, T, d_model)
        _, (h_n, _) = self.lstm(x)
        h_T = h_n[-1]  # (M, d_hidden): final layer's hidden state per subregion

        concat = h_T.reshape(1, self.M * self.d_hidden)
        G = self.phi(concat)
        goal = self.goal_head(G)

        return G, goal
