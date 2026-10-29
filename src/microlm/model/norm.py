"""LayerNorm and RMSNorm. Placement (pre/post) is decided in transformer.Block."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


class LayerNorm(nn.Module):
    def __init__(self, dim: int, bias: bool, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):
    """x / rms(x) * weight, with the statistic computed in fp32."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def make_norm(cfg: ModelConfig, dim: int | None = None) -> nn.Module:
    dim = dim or cfg.d_model
    if cfg.norm_type == "layernorm":
        return LayerNorm(dim, cfg.bias, cfg.norm_eps)
    return RMSNorm(dim, cfg.norm_eps)
