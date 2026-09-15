import torch


class RolloutBuffer:
    """Fixed-length on-policy rollout storage for the Sub-Policy's PPO update.

    Stores raw per-step inputs (obs, F_g) rather than the already-fused GAC
    output, so the PPO update can recompute local_encoder -> GAC -> fuse ->
    actor_critic fresh each epoch — otherwise gradients could never reach
    local_encoder/GAT (storing a detached, already-fused state would sever
    the graph and leave those submodules untrained).

    Collects `rollout_length` steps across all N agents, then is consumed by
    one PPO update and cleared (on-policy — no cross-rollout reuse).
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.f_g = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        # Per-step Meta-Policy targets, needed for the Sub-Policy's goal-alignment term.
        self.goals = []
        self.w_globals = []
        self.q_globals = []

    def add(self, obs, f_g, action, log_prob, value, reward, done, goal, w_global, q_global):
        self.obs.append(obs)
        self.f_g.append(f_g)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.goals.append(goal)
        self.w_globals.append(w_global)
        self.q_globals.append(q_global)

    def __len__(self):
        return len(self.obs)

    def get(self):
        """Returns stacked (T, ...) tensors for every stored field."""

        return {
            "obs": torch.stack(self.obs),  # (T, N, obs_dim)
            "f_g": torch.stack(self.f_g),  # (T, d_reg)
            "actions": torch.stack(self.actions),  # (T, N)
            "log_probs": torch.stack(self.log_probs),  # (T, N)
            "values": torch.stack(self.values),  # (T, N)
            "rewards": torch.stack(self.rewards),  # (T, N)
            "dones": torch.stack(self.dones),  # (T, N)
            "goals": torch.stack(self.goals),  # (T, 2)
            "w_globals": torch.stack(self.w_globals),  # (T,)
            "q_globals": torch.stack(self.q_globals),  # (T,)
        }
