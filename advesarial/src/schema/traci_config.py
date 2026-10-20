from pydantic import BaseModel


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
