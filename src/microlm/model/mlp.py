"""Feed-forward blocks: GELU, ReLU², and the gated SwiGLU / GEGLU.

All variants are parameter-matched through ModelConfig.ffn_hidden: a gated FFN has three
d x h matrices instead of two, so its hidden size is 2/3 of the non-gated one. Without
that correction a "SwiGLU vs GELU" comparison is really a comparison of 1.5x the FFN
parameters.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        h = cfg.ffn_hidden
        self.kind = cfg.ffn_type
        self.gated = cfg.gated
        self.up = nn.Linear(cfg.d_model, 2 * h if self.gated else h, bias=cfg.bias)
        if self.gated:
            self.up.weight.muon_split = (h, h)  # gate and value are separate matrices
        self.down = nn.Linear(h, cfg.d_model, bias=cfg.bias)
        self.down.weight.is_residual_out = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.up(x)
        if self.kind == "gelu":
            a = F.gelu(u)
        elif self.kind == "relu2":
            a = F.relu(u).square()
        else:
            gate, value = u.chunk(2, dim=-1)
            a = (F.silu(gate) if self.kind == "swiglu" else F.gelu(gate)) * value
        return self.down(a)
