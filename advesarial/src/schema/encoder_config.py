from pydantic import BaseModel


class TransformerEncoderConfig(BaseModel):
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_regions: int = 200
    scaling_factor: float = 10000.0
