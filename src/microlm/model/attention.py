"""Causal self-attention: MHA / GQA / MQA, optional QK-norm, RoPE or ALiBi.

Two code paths compute the same function:
  fast   F.scaled_dot_product_attention (Flash / memory-efficient kernels)
  naive  explicit softmax(QK^T / sqrt(d) + bias) V, which also exposes the attention
         probabilities for diagnostics (entropy, induction scores)
`tests/test_model.py` checks they agree for every positional variant.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .norm import RMSNorm
from .pos import apply_rope


@dataclass
class AttnContext:
    """Per-forward tables shared by every layer."""

    rope: tuple[torch.Tensor, torch.Tensor] | None = None  # cos, sin: (T, d_head)
    alibi: torch.Tensor | None = None  # (1, H, T, T) additive bias incl. causal -inf
    naive: bool = False
    probs_hook: Callable[[torch.Tensor], None] | None = None  # called with each layer's (B, H, T, T) probs


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head, self.n_kv, self.d_head = cfg.n_head, cfg.kv_heads, cfg.d_head
        q_dim, kv_dim = self.n_head * self.d_head, self.n_kv * self.d_head
        self.qkv = nn.Linear(cfg.d_model, q_dim + 2 * kv_dim, bias=cfg.bias)
        self.qkv.weight.muon_split = (q_dim, kv_dim, kv_dim)  # orthogonalize q, k, v separately
        self.proj = nn.Linear(q_dim, cfg.d_model, bias=cfg.bias)
        self.proj.weight.is_residual_out = True
        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.d_head, cfg.norm_eps)
            self.k_norm = RMSNorm(self.d_head, cfg.norm_eps)

    def forward(self, x: torch.Tensor, ctx: AttnContext) -> torch.Tensor:
        B, T, _ = x.shape
        H, Hkv, Dh = self.n_head, self.n_kv, self.d_head
        q, k, v = self.qkv(x).split([H * Dh, Hkv * Dh, Hkv * Dh], dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, Hkv, Dh).transpose(1, 2)
        v = v.view(B, T, Hkv, Dh).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if ctx.rope is not None:
            cos, sin = ctx.rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if Hkv != H:
            k = k.repeat_interleave(H // Hkv, dim=1)
            v = v.repeat_interleave(H // Hkv, dim=1)

        if ctx.naive or ctx.probs_hook is not None:
            y = self._naive(q, k, v, ctx)
        elif ctx.alibi is not None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx.alibi.to(q.dtype))
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, H * Dh))

    def _naive(self, q, k, v, ctx: AttnContext) -> torch.Tensor:
        T = q.shape[-2]
        scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if ctx.alibi is not None:
            scores = scores + ctx.alibi.float()
        else:
            causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
        probs = scores.softmax(dim=-1)
        if ctx.probs_hook is not None:
            ctx.probs_hook(probs.detach())
        return (probs @ v.float()).to(q.dtype)
