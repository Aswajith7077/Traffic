from pydantic import BaseModel


class TraciConfig(BaseModel):
    config_path: str = "sumo/simple.sumocfg"
    use_gui: bool = False
    step_length: float = 1.0
    delay: float = 0.0
