import torch


def compute_gae(rewards, values, dones, bootstrap_value, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation, used by the Sub-Policy's PPO update.

    rewards, values, dones: (T, N) tensors — T rollout steps, N agents.
    bootstrap_value: (N,) value estimate for the state after the last step.

    Returns:
        advantages: (T, N)
        returns: (T, N) — advantages + values, i.e. the PPO value targets.
    """

    T, N = rewards.shape
    advantages = torch.zeros(T, N, dtype=torch.float32)
    gae = torch.zeros(N, dtype=torch.float32)

    next_value = bootstrap_value
    for t in reversed(range(T)):
        mask = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values
    return advantages, returns
