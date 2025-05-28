from ....cstp_v3 import modelling
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field

@dataclass
class Protein_Arguments:
    protein_projection_input_dim: Optional[int] = field(default=1280)   # default to the last layer
    protein_projection_output_dim: Optional[int] = field(default=5120)
    text_projection_input_dim: Optional[int] = field(default=5120)
    text_projection_output_dim: Optional[int] = field(default=5120)
    n_heads: Optional[int] = field(default=8)
    n_layers: Optional[int] = field(default=1)
    alpha: Optional[float] = field(default=0.5)

def build_protein_projector(cstp_chackpoint_path):
    cstp_model = modelling.CSTPLightning.load_from_checkpoint(
        checkpoint_path = cstp_chackpoint_path,
        protein_projection_input_dim  = Protein_Arguments.protein_projection_input_dim,
        protein_projection_output_dim = Protein_Arguments.protein_projection_output_dim,
        text_projection_input_dim = Protein_Arguments.text_projection_input_dim,
        text_projection_output_dim = Protein_Arguments.text_projection_output_dim,
        n_heads = Protein_Arguments.n_heads,
        n_layers = Protein_Arguments.n_layers,
        alpha = Protein_Arguments.alpha,
    )
    for param in cstp_model.parameters():
        param.requires_grad = False
    cstp_model.eval()
    return cstp_model