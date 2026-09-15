import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """Shared trunk + actor + dual-branch critic, per Table 6.

    Table 6 uses a fused observation dim of 334 (5*66 GAC output + 4-dim
    global feature) with a 334->256->128->114 shared trunk, a 114->57->8
    actor, and a dual-branch critic (branch 1: 57 -> concat(57, first 57 of
    the shared feature) -> 1 value; branch 2: 57 -> 56 latent plan). This
    repo's per-intersection observation is 10-dim (not the paper's 66), so
    the fused dim is 5*10+4=54 here; trunk/branch widths are scaled down
    proportionally rather than reusing the paper's absolute sizes verbatim.
    The latent-plan branch is implemented for architectural fidelity but is
    not used by any of the paper's described loss functions.
    """

    def __init__(self, state_dimension, action_dimension, shared_hidden=(128, 64, 32), branch_hidden=16, dropout=0.1):
        super().__init__()

        h1, h2, shared_out = shared_hidden
        self.shared_out = shared_out
        self.partial_dim = min(branch_hidden, shared_out)

        self.shared = nn.Sequential(
            nn.Linear(state_dimension, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, shared_out),
        )

        self.actor = nn.Sequential(
            nn.Linear(shared_out, branch_hidden),
            nn.ReLU(),
            nn.Linear(branch_hidden, action_dimension),
        )

        self.critic_branch1 = nn.Linear(shared_out, branch_hidden)
        self.critic_value = nn.Linear(branch_hidden + self.partial_dim, 1)

        self.critic_branch2 = nn.Sequential(
            nn.Linear(shared_out, branch_hidden),
            nn.ReLU(),
            nn.Linear(branch_hidden, branch_hidden),
        )

    def forward(self, state):
        shared_feature = self.shared(state)  # no activation after the last shared layer (Table 6)

        action_prob = F.softmax(self.actor(shared_feature), dim=-1)

        branch1 = F.relu(self.critic_branch1(shared_feature))
        partial = shared_feature[..., : self.partial_dim]
        state_value = self.critic_value(torch.cat([branch1, partial], dim=-1))

        latent_plan = self.critic_branch2(shared_feature)  # unused by current losses; kept for Table 6 fidelity

        return action_prob, state_value, latent_plan
