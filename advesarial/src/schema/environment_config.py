from pydantic import BaseModel


class EnvironmentConfig(BaseModel):
    use_gui: bool = False
    max_steps: int = 4000
    history_len: int = 10
    step_length: float = 1.0
    delay: float = 1.0
    min_green_steps: int = 5
    config_path: str = "sumo/manhattan.sumocfg"
