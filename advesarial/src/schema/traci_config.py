import os

from pydantic import BaseModel, Field


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


class TraciConfig(BaseModel):
    config_path: str
    use_gui: bool = False
    step_length: float = 1.0
    delay: float = 0.0
    green_duration: float = 5.0
    yellow_duration: float = 3.0
    min_green_steps: int = 5
    max_wait: float = 90.0  # W_max: per-vehicle waiting threshold (s) that triggers the starvation override
    wait_ref: float = 30.0  # W_ref: reference time (s) used to normalize waiting times in the fairness terms

    # City-wide multi-EMV routing extension (Section 3.2), default off. When off,
    # EMVs still get local preemption at the intersection they're currently
    # approaching (Section 5.2's baseline ablation) but no anticipatory,
    # route-wide bias is computed or fed into observations.
    use_emv_routing: bool = Field(default_factory=lambda: _env_bool("USE_EMV_ROUTING"))
    # Accident/incident injection (Section 3.3), default off.
    use_accidents: bool = Field(default_factory=lambda: _env_bool("USE_ACCIDENTS"))
