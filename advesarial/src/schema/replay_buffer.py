from pydantic import BaseModel


class ReplayBufferItem(BaseModel):
    state: dict
    action: dict
    reward: float
    next_state: dict
    done: bool
