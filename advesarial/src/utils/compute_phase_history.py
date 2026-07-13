import math


def compute_phase_entropy(phase_history):
    """
    phase_history: list of phase ids
    Example: [0,0,1,2,0,3,1]
    """

    total = len(phase_history)

    if total == 0:
        return 0.0

    freq = {}

    for phase in phase_history:
        freq[phase] = freq.get(phase, 0) + 1

    entropy = 0.0

    for count in freq.values():
        p = count / total
        entropy -= p * math.log(p + 1e-9)

    return entropy
