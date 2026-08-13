import torch

"""

action: Set of { ah, al1, al2, ... , aln }
n -> number of intersections

There are dynamic phases

"""


class Environment:
    def __init__(self, traci_service, max_t=3600, control_interval=10):
        self.traci_service = traci_service
        self.t = 0
        self.max_t = max_t
        self.control_interval = control_interval
        self.intersections = self.traci_service.get_all_intersections()

        self.running_mean = torch.zeros(10)
        self.running_var = torch.ones(10)
        self.count = 1e-4
        self.alpha = 0.01

        self.beta1 = 0.1
        self.beta2 = 0.1

        self.prev_queue = 0
        self.prev_wait = 0

        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 1e-4
        self.reward_alpha = 0.01
        self.normalize_rewards = True

        self.adjacency_list = self.traci_service.get_adjacency_list()

    def _apply_action(self, action):

        for i, intersection in enumerate(self.intersections):
            act = action[i].item()
            self.traci_service.set_phase(intersection, act)

    def step(self, action, needs_obs=True):

        # get observation
        # get reward
        # get done
        # get info

        self._apply_action(action)

        self.traci_service.step(self.control_interval)

        next_state = self.traci_service.get_observations() if needs_obs else None
        reward = self._compute_reward()
        done = self.t >= self.max_t

        self.t += self.control_interval

        return next_state, reward, done

    def restart(self):
        """Restart the SUMO simulation for a new training episode."""
        self.t = 0
        self.prev_queue = 0
        self.prev_wait = 0
        self.traci_service.reset_simulation()
        self.intersections = self.traci_service.get_all_intersections()
        self.adjacency_list = self.traci_service.get_adjacency_list()

    def compute_global_reward(self):
        total_queue_length = 0
        total_waiting_time = 0
        for intersection in self.intersections:
            total_queue_length += self.traci_service.total_queue_length(intersection)
            total_waiting_time += self.traci_service.total_waiting_time(intersection)

        # Use raw totals directly for reward calculation (negative for congestion)
        reward = -(self.beta1 * total_waiting_time + self.beta2 * total_queue_length)

        return reward

    def _compute_reward(self):

        local_reward = 0
        total_queue_length = 0
        total_waiting_time = 0
        emergency_delay = 0

        for intersection in self.intersections:
            local_reward += self.traci_service.get_intersection_reward(intersection)
            total_queue_length += self.traci_service.total_queue_length(intersection)
            total_waiting_time += self.traci_service.total_waiting_time(intersection)
            emergency_delay += self.traci_service.get_emergency_waiting_time(intersection)

        num = len(self.intersections) + 1e-6
        local_reward = local_reward / num

        delta_queue = self.prev_queue - total_queue_length
        delta_wait = self.prev_wait - total_waiting_time

        self.prev_queue = total_queue_length
        self.prev_wait = total_waiting_time

        R_eff = -(self.beta1 * delta_queue + self.beta2 * delta_wait) + local_reward
        R_emergency = -emergency_delay

        total_reward = 0.7 * R_eff + 0.3 * R_emergency

        if self.normalize_rewards:
            total_reward = self._normalize_reward(total_reward)

        return float(total_reward)

    def _normalize_reward(self,reward):
        self.reward_mean = (1 - self.reward_alpha) * self.reward_mean + self.reward_alpha * reward
        self.reward_var = (1 - self.reward_alpha) * self.reward_var + self.reward_alpha * (reward - self.reward_mean)**2
        normalized_reward = (reward - self.reward_mean) / (torch.sqrt(torch.tensor(self.reward_var)) + 1e-8)

        return normalized_reward

    def reset(self):
        self.t = 0
        return self.get_observations()
