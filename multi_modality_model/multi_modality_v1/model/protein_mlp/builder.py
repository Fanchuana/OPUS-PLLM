import torch
import torch.nn as nn
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
@dataclass
class Switch_Arguments:
    hidden_size: Optional[int] = field(default=768)
    mm_hidden_size: Optional[int] = field(default=1280)

def build_switch_projector(model_args, n_tokens = 8):
    projector_type = getattr(model_args, 'switch_projector_type', 'mlp2x_gelu')
    Switch_Arguments.hidden_size = model_args.hidden_size * n_tokens
    Switch_Arguments.mm_hidden_size = 5120 if model_args.pretrain_protein_projector_ckpt!=None else 1280
    if projector_type == 'linear':
        print('Linear Projector!')
        return nn.Linear(Switch_Arguments.mm_hidden_size, Switch_Arguments.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        print('Multi-Layer Projector!')
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(Switch_Arguments.mm_hidden_size, Switch_Arguments.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(Switch_Arguments.hidden_size, Switch_Arguments.hidden_size))
        return nn.Sequential(*modules)

