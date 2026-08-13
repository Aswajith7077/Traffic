import math

import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, state_dimension, action_dimension):
        super().__init__()
        self.affine = nn.Linear(state_dimension, 128)
        self.actor = nn.Linear(128, action_dimension)
        self.critic_affine = nn.Linear(state_dimension, 128)
        self.critic = nn.Linear(128, 1)
        self._init_weights()

    def _init_weights(self):
        for layer in (self.affine, self.critic_affine):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, state):
        x = F.relu(self.affine(state))
        action_prob = F.softmax(self.actor(x), dim=-1)
        v = F.relu(self.critic_affine(state))
        state_value = self.critic(v)
        return action_prob, state_value
