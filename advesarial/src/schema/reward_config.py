import os

from pydantic import BaseModel, Field, model_validator


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


class RewardConfig(BaseModel):
    """Reward weights, Eq 23-24 (Section 4) and the hierarchical/fairness weights
    of Sections 3.8/4.1 — a single user-configurable surface for every coefficient
    that used to be a hardcoded `Environment`/`TraciService` attribute.

    Defaults match the paper's stated rationale ("R_eff is assigned the largest
    weight... followed by fairness then emergency responsiveness") while staying
    overridable per-run via env vars, since these are tunable hyperparameters,
    not fixed constants.
    """

    w1: float = Field(default_factory=lambda: _env_float("REWARD_W1", 0.6))  # R_eff weight
    w2: float = Field(default_factory=lambda: _env_float("REWARD_W2", 0.3))  # R_fair weight
    w3: float = Field(default_factory=lambda: _env_float("REWARD_W3", 0.1))  # R_emergency weight
    beta_q: float = Field(default_factory=lambda: _env_float("BETA_Q", 0.5))
    beta_w: float = Field(default_factory=lambda: _env_float("BETA_W", 0.5))
    lam_var: float = Field(default_factory=lambda: _env_float("LAM_VAR", 1.0))
    lam_starve: float = Field(default_factory=lambda: _env_float("LAM_STARVE", 10.0))
    lam_jain: float = Field(default_factory=lambda: _env_float("LAM_JAIN", 10.0))

    @model_validator(mode="after")
    def _check_eq24(self):
        if not (self.w1 >= self.w2 >= self.w3 >= 0):
            raise ValueError(f"Eq 24 requires w1 >= w2 >= w3 >= 0, got w1={self.w1}, w2={self.w2}, w3={self.w3}")
        total = self.w1 + self.w2 + self.w3
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Eq 24 requires w1+w2+w3=1, got {total}")
        return self
