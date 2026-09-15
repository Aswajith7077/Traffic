from pydantic import BaseModel


class TransformerEncoderConfig(BaseModel):
    # Paper defaults (Table 6): 3 layers, 2 heads, hidden dim (d_model) 4,
    # feedforward 165, regional state dim (d_reg, Appendix A) 4.
    d_reg: int = 4
    d_model: int = 4
    nhead: int = 2
    num_layers: int = 3
    dim_feedforward: int = 165
    dropout: float = 0.1
    max_regions: int = 200
    scaling_factor: float = 10000.0
