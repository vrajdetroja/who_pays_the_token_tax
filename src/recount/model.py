"""The study's model: microlm's Transformer, with unreachable vocabulary rows masked.

A dose level's tokenizer can never produce ids 256 + k .. 16,382, yet they stay in the
embedding table so every level has identical parameters. Masking sets their logits to -inf
in training and evaluation (PREREGISTRATION.md, decision 6): the softmax runs over the rows
the level can produce, so no bits are spent on impossible tokens. That loss would grow with
r and shrink with training length, so it would inflate the short as-deployed branches most
and bias S toward "accounting". Masked rows receive no gradient.

With nothing to mask (r = 1) the call goes straight to the base model, so level-1.0 runs
compute exactly what microlm computes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from microlm.model import Transformer


class DoseLM(nn.Module):
    def __init__(self, base: Transformer, unreachable: list[int] | None = None):
        super().__init__()
        self.base = base
        self.cfg = base.cfg
        mask = torch.zeros(base.cfg.vocab_size, dtype=torch.bool)
        if unreachable:
            mask[torch.as_tensor(unreachable, dtype=torch.long)] = True
        self.register_buffer("unreachable", mask, persistent=False)
        self.masked = bool(mask.any())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, *, reduction: str = "mean"):
        if not self.masked:
            return self.base(idx, targets, reduction=reduction)
        logits, _ = self.base(idx)
        logits = logits.masked_fill(self.unreachable, float("-inf"))
        if targets is None:
            return logits, None
        B, T = idx.shape
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), reduction=reduction)
        return logits, loss.view(B, T) if reduction == "none" else loss


def build(cfg, unreachable: list[int] | None, mask: bool) -> DoseLM:
    """Model for a run: masked if the study says so and the level has unreachable rows."""
    return DoseLM(Transformer(cfg), unreachable if mask else None)
