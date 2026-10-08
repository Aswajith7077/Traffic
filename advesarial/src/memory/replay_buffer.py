import random
from collections import deque

import torch


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Sample WITHOUT replacement: entries hold live autograd graphs (states
        # are no longer detached in sample.py), and each graph can only be
        # backwarded through once. This makes the update on-policy REINFORCE.
        items = list(self.buffer)
        batch = random.sample(items, batch_size)
        batch_ids = {id(item) for item in batch}
        self.buffer = deque(
            (item for item in items if id(item) not in batch_ids),
            maxlen=self.buffer.maxlen,
        )

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(states),
            torch.stack(actions),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)
