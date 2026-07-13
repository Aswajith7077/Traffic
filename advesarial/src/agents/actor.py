import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, state_dimension, action_dimension):
        super().__init__()
        self.affine = nn.Linear(state_dimension, 128)
        self.actor = nn.Linear(128, action_dimension)
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.affine(state))
        action_prob = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_prob, state_value
