from collections import deque


class PhaseTracker:
    def __init__(self, window_size=20):
        self.phase_history = deque(maxlen=window_size)

    def update(self, current_phase):
        self.phase_history.append(current_phase)

    def get_entropy(self):
        return compute_phase_entropy(list(self.phase_history))
