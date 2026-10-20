import torch

"""

action: Set of { ah, al1, al2, ... , aln }
n -> number of intersections

There are dynamic phases

"""


class Environment:
    def __init__(self, traci_service, max_t=3600):
        self.traci_service = traci_service
        self.t = 0
        self.max_t = max_t
        self.intersections = self.traci_service.get_all_intersections()

        # Weights for the global goal-reward, Section 4.3.1 (beta_q, beta_w).
        self.beta_q = 0.5
        self.beta_w = 0.5

        # Fairness penalty weights (Section 4): P_fair = lam_var*Var(w/W_ref)
        # + lam_starve*max(0, (W - W_max)/W_max) + lam_jain*(1 - J). Initial, untuned values.
        self.lam_var = 1.0
        self.lam_starve = 10.0
        self.lam_jain = 10.0

        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_alpha = 0.01

        self.adjacency_list = self.traci_service.get_adjacency_list()

    def _apply_action(self, action):

        for i, intersection in enumerate(self.intersections):
            act = action[i].item()
            self.traci_service.set_phase(intersection, act)

    def step(self, action, goal=None):
        """
        goal: optional (G_w, G_q) sub-goal from the Meta-Policy, used to
        compute the global goal-reward r_g (Section 4.3.1). When omitted,
        r_g is 0 (e.g. evaluation runs that don't need the training signal).
        """

        self._apply_action(action)

        self.traci_service.step()
        self.traci_service.update_wait_timers()

        next_state = self.traci_service.get_observations()
        reward = self._compute_reward(goal)
        done = self.t >= self.max_t

        self.t += 1

        return next_state, reward, done

    def compute_goal_reward(self, goal):
        """r_g^t = -[beta_q*(W_global - G_q) + beta_w*(Q_global - G_w)], Section 4.3.1."""

        if goal is None:
            return 0.0

        G_w, G_q = goal
        W_global, Q_global = self.traci_service.compute_global_state_now()
        W_global = W_global.sum()
        Q_global = Q_global.sum()

        return -(self.beta_q * (W_global - G_q) + self.beta_w * (Q_global - G_w))

    def _compute_fairness_reward(self):
        """Per-intersection R_fair_j = -P_fair_j (dimensionless penalties, Section 4)."""
        w_max = self.traci_service.max_wait
        rewards = []
        for i in self.intersections:
            jain, var, max_wait = self.traci_service.get_intersection_fairness(i)
            starve = max(max_wait - w_max, 0.0) / w_max
            rewards.append(-(self.lam_var * var + self.lam_starve * starve + self.lam_jain * (1.0 - jain)))
        return torch.tensor(rewards, dtype=torch.float32)

    def _compute_local_reward(self):
        """Per-intersection local reward r_i^t = -(ql+wt+dt+ps-ss), Section 4.3.1,
        plus the per-intersection fairness reward R_fair_i (Section 4).

        Returns a (N,) tensor — one reward per agent, using that agent's own
        intersection metrics (not a team-average).
        """

        efficiency = torch.tensor(
            [self.traci_service.get_intersection_reward(i) for i in self.intersections],
            dtype=torch.float32,
        )
        return efficiency + self._compute_fairness_reward()

    def _compute_reward(self, goal=None):
        r_i = self._compute_local_reward()  # (N,)
        r_g = self.compute_goal_reward(goal)  # scalar, shared across agents

        total_reward = r_i + r_g  # (N,) trajectory reward r^t = r_i^t + r_g^t
        normalized_reward = self._normalize_reward(total_reward)

        return normalized_reward

    def _normalize_reward(self, reward):
        # Running mean/var normalization (scalar EMA over the batch of N
        # per-agent rewards) for training stability. Not part of the paper's
        # formula, but a standard, low-risk addition for PPO/GAE.
        batch_mean = reward.mean().item()
        self.reward_mean = (1 - self.reward_alpha) * self.reward_mean + self.reward_alpha * batch_mean
        self.reward_var = (1 - self.reward_alpha) * self.reward_var + self.reward_alpha * (
            batch_mean - self.reward_mean
        ) ** 2
        normalized_reward = (reward - self.reward_mean) / (self.reward_var**0.5 + 1e-8)

        return normalized_reward

    def reset(self):
        self.t = 0
        self.reward_mean = 0.0
        self.reward_var = 1.0
        return self.traci_service.get_observations()
