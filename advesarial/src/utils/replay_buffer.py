from collections import deque

import torch


class RegionalStateBuffer:
    """Sliding window of the last T regional-state snapshots for the Meta-Policy's
    Transformer+LSTM (Section 4.1). Zero-padded at the start of an episode.
    """

    def __init__(self, window_size, num_regions, d_reg):
        self.window_size = window_size
        self.num_regions = num_regions
        self.d_reg = d_reg
        self._buffer = deque(maxlen=window_size)
        self.reset()

    def reset(self):
        self._buffer.clear()
        zeros = torch.zeros(self.num_regions, self.d_reg)
        for _ in range(self.window_size):
            self._buffer.append(zeros)

    def push(self, regional_state: torch.Tensor):
        """regional_state: (num_regions, d_reg) snapshot for the current timestep."""
        self._buffer.append(regional_state)

    def get_window(self) -> torch.Tensor:
        """Returns (T, num_regions, d_reg)."""
        return torch.stack(list(self._buffer), dim=0)
