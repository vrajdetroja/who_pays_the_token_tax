"""Positional information: learned absolute, sinusoidal, RoPE, ALiBi, or none (NoPE).

learned and sinusoidal are added to the input embeddings; RoPE rotates queries and keys;
ALiBi adds a per-head linear distance penalty to the attention logits. NoPE relies on the
causal mask alone. Everything except `learned` is defined at any length, which is what the
length-extrapolation experiment needs.
"""

from __future__ import annotations

import math

import torch


def sinusoidal_table(T: int, d: int, device=None) -> torch.Tensor:
    pos = torch.arange(T, dtype=torch.float32, device=device)[:, None]
    i = torch.arange(0, d, 2, dtype=torch.float32, device=device)
    div = torch.exp(-math.log(10000.0) * i / d)
    table = torch.zeros(T, d, device=device)
    table[:, 0::2] = torch.sin(pos * div)
    table[:, 1::2] = torch.cos(pos * div[: d // 2])
    return table


def rope_cos_sin(T: int, d_head: int, base: float, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (T, d_head) for the rotate-half RoPE formulation, in fp32."""
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float32, device=device) / d_head))
    t = torch.arange(T, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D). Rotation is done in fp32 and cast back to x's dtype."""
    xf = x.float()
    x1, x2 = xf.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return (xf * cos + rotated * sin).to(x.dtype)


def alibi_slopes(n_head: int) -> torch.Tensor:
    """Head slopes from Press et al. (2021), including their recipe for non-powers of two."""

    def pow2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]

    if math.log2(n_head).is_integer():
        s = pow2(n_head)
    else:
        closest = 2 ** math.floor(math.log2(n_head))
        s = pow2(closest) + pow2(2 * closest)[0::2][: n_head - closest]
    return torch.tensor(s, dtype=torch.float32)


def alibi_bias(n_head: int, T: int, device=None) -> torch.Tensor:
    """Additive attention bias (1, H, T, T): -slope * (i - j) for j <= i, -inf for j > i."""
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    dist = (i - j).float()
    bias = -alibi_slopes(n_head).to(device)[:, None, None] * dist
    return bias.masked_fill(j > i, float("-inf"))[None]

