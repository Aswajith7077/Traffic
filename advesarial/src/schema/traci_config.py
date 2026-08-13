from pydantic import BaseModel


class TraciConfig(BaseModel):
    config_path: str
    use_gui: bool = False
    step_length: float = 1.0
    delay: float = 0.0
    tripinfo_output: str | None = None
    min_green_time: int = 1
    yellow_seconds: float = 5.0
