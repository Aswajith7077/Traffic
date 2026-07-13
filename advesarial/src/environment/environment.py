import torch
import numpy as np

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

        self.adjacency_list = self.traci_service.get_adjacency_list()

    def _apply_action(self, action):

        for i, intersection in enumerate(self.intersections):
            act = action[i].item()
            self.traci_service.set_phase(intersection, act)

    def step(self, action):

        # get observation
        # get reward
        # get done
        # get info

        self._apply_action(action)

        self.traci_service.step()

        next_state = self.traci_service.get_observations()
        reward = self._compute_reward()
        done = self.t >= self.max_t

        self.t += 1

        return next_state, reward, done

    def compute_global_reward(self):
        total_queue_length = 0
        total_waiting_time = 0
        for intersection in self.intersections:
            total_queue_length += self.traci_service.total_queue_length(intersection)
            total_waiting_time += self.traci_service.total_waiting_time(intersection)

        # Use raw totals directly for reward calculation (negative for congestion)
        reward = -(self.beta1 * total_waiting_time + self.beta2 * total_queue_length)

        return reward

    def compute_jain_index(self,wait_times):
        n = len(wait_times)

        if n == 0:
            return 0.0

        numerator = sum(wait_times) ** 2
        denominator = n * sum(w ** 2 for w in wait_times)

        if denominator == 0:
            return 0.0  # or 1.0 depending on your interpretation

        return numerator / denominator

    def _compute_reward(self):

        def norm(x):
            return x / (abs(x) + 1e-6)

        queues = []
        non_emv_wait_times = []
        ped_wait_times = []
        local_utilities = []

        local_reward = 0
        total_queue_length = 0
        total_waiting_time = 0
        emission_penalty = 0
        emergency_delay = 0
        my_utility = 0.0
        ped_conflicts = 0


        for intersection in self.intersections:
            local_reward += self.traci_service.get_intersection_reward(intersection)
            total_queue_length += self.traci_service.total_queue_length(intersection)
            total_waiting_time += self.traci_service.total_waiting_time(intersection)
            emission_penalty += self.traci_service.get_stop_count(intersection)
            emergency_delay += self.traci_service.get_emergency_waiting_time(intersection)
            q = self.traci_service.total_queue_length(intersection)

            util_i = - (self.traci_service.total_waiting_time(intersection) + 
                       0.5 * self.traci_service.total_queue_length(intersection))
            local_utilities.append(util_i)

            # if intersection == self.current_intersection:   # or average over batch
            my_utility = util_i

            non_emv_waits = self.traci_service.get_non_emv_waiting_times(intersection)   # list of floats
            non_emv_wait_times.extend(non_emv_waits)

            ped_waits = self.traci_service.get_pedestrian_waiting_times(intersection)     # list of floats
            ped_wait_times.extend(ped_waits)
            ped_conflicts += self.traci_service.get_pedestrian_conflict_count(intersection)

            queues.append(q)

        num = len(self.intersections) + 1e-6
        local_reward = local_reward / num


        delta_queue = self.prev_queue - total_queue_length
        delta_wait = self.prev_wait - total_waiting_time

        self.prev_queue = total_queue_length
        self.prev_wait = total_waiting_time

        # Fairness Computation

        var_wait = np.var(non_emv_wait_times) if non_emv_wait_times else 0.0
        max_wait_penalty = max(max(non_emv_wait_times) - 90.0, 0.0) if non_emv_wait_times else 0.0

        jain = self.compute_jain_index(non_emv_wait_times)

        max_other_utility = max(local_utilities) if local_utilities else 0.0
        envy = max(max_other_utility - my_utility, 0.0)   # classic envy measure

        max_ped_wait = max(ped_wait_times) if ped_wait_times else 0.0
        


        R_eff = -(self.beta1 * delta_queue + self.beta2 * delta_wait) + local_reward
        R_fair = (
            -0.3 * var_wait
            -0.25 * max_wait_penalty
            -0.15 * (1.0 - jain)
            -0.5 * envy                     # ← Envy-free penalty (forces interaction)
            -0.5 * max_ped_wait
            -1.2 * ped_conflicts
        )
        R_emergency = -emergency_delay
        R_emission = -emission_penalty

        total_reward = (
            0.4 * R_eff +
            0.2 * R_fair +
            0.3 * R_emergency +
            0.1 * R_emission
        )

        normalized_reward = self._normalize_reward(total_reward)

        return float(normalized_reward)

    def _normalize_reward(self,reward):
        self.reward_mean = (1 - self.reward_alpha) * self.reward_mean + self.reward_alpha * reward
        self.reward_var = (1 - self.reward_alpha) * self.reward_var + self.reward_alpha * (reward - self.reward_mean)**2
        normalized_reward = (reward - self.reward_mean) / (torch.sqrt(torch.tensor(self.reward_var)) + 1e-8)

        return normalized_reward

    def reset(self):
        self.t = 0
        return self.get_observations()
