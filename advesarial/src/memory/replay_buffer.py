import random
from collections import deque

import torch


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(states),
            torch.stack(actions),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
        )

    def sample_recent(self, batch_size):
        batch = list(self.buffer)[-batch_size:]

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
