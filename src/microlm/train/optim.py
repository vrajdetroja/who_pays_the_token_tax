"""Optimizers: AdamW (torch), Lion, and Muon, behind one constructor.

Parameter roles:
  matrices    2-D weights inside the blocks (attention, FFN)
  embeddings  token / position embeddings and the untied head
  vectors     norm gains and biases: never weight-decayed

AdamW and Lion treat matrices and embeddings alike (both decayed). Muon orthogonalizes
updates for `matrices` only and hands embeddings and vectors to AdamW, as in Jordan et al.
(2024). Every param group carries `base_lr`; the trainer sets lr = base_lr * multiplier.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import TrainConfig


def param_roles(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    roles: dict[str, list[nn.Parameter]] = {"matrices": [], "embeddings": [], "vectors": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2:
            roles["vectors"].append(p)
        elif name.startswith(("tok_emb", "pos_emb", "head")):
            roles["embeddings"].append(p)
        else:
            roles["matrices"].append(p)
    return roles


# ---------------------------------------------------------------------------------------


class Lion(torch.optim.Optimizer):
    """Chen et al. (2023): sign of an interpolated momentum, decoupled weight decay."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                update = m.mul(b1).add_(p.grad, alpha=1 - b1).sign_()
                p.mul_(1 - lr * wd).add_(update, alpha=-lr)
                m.mul_(b2).add_(p.grad, alpha=1 - b2)


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximately replace G by U V^T from its SVD (Jordan et al., 2024).

    A quintic iteration whose coefficients are tuned to push singular values toward ~1
    quickly rather than exactly; they end up in roughly [0.7, 1.2], which is enough.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.is_cuda else G.float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Nesterov momentum, then orthogonalize each logical matrix, scaled by sqrt(max(1, rows/cols)).

    Fused weights (QKV, gate+value) carry a `muon_split` attribute and are orthogonalized
    block by block, so fusing a matmul for speed does not change the optimizer.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1 - mom)
                g = p.grad.lerp(buf, mom)  # Nesterov
                sizes = getattr(p, "muon_split", None) or (p.size(0),)
                parts = []
                for block in g.split(list(sizes), dim=0):
                    scale = max(1.0, block.size(0) / block.size(1)) ** 0.5
                    parts.append(newton_schulz(block, group["ns_steps"]) * scale)
                p.mul_(1 - lr * wd).add_(torch.cat(parts, dim=0), alpha=-lr)


class CombinedOptimizer:
    """Steps several optimizers as one; exposes their param groups as a single list."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    def step(self):
        for o in self.optimizers:
            o.step()

    def zero_grad(self, set_to_none: bool = True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return [o.state_dict() for o in self.optimizers]

    def load_state_dict(self, states):
        for o, s in zip(self.optimizers, states, strict=True):
            o.load_state_dict(s)


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> CombinedOptimizer:
    roles = param_roles(model)
    fused = {"fused": True} if next(model.parameters()).is_cuda else {}
    betas = (cfg.beta1, cfg.beta2)

    def adamw(groups):
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=betas, eps=cfg.eps, **fused)

    def g(params, lr, wd):
        return {"params": params, "lr": lr, "base_lr": lr, "weight_decay": wd}

    if cfg.optimizer == "adamw":
        opts = [adamw([
            g(roles["matrices"] + roles["embeddings"], cfg.lr, cfg.weight_decay),
            g(roles["vectors"], cfg.lr, 0.0),
        ])]
    elif cfg.optimizer == "lion":
        opts = [Lion([
            g(roles["matrices"] + roles["embeddings"], cfg.lr, cfg.weight_decay),
            g(roles["vectors"], cfg.lr, 0.0),
        ], betas=betas)]
    elif cfg.optimizer == "muon":
        muon = Muon([g(roles["matrices"], cfg.muon_lr, cfg.weight_decay)],
                    momentum=cfg.muon_momentum, ns_steps=cfg.muon_ns_steps)
        opts = [muon, adamw([
            g(roles["embeddings"], cfg.lr, cfg.weight_decay),
            g(roles["vectors"], cfg.lr, 0.0),
        ])]
    else:
        raise ValueError(cfg.optimizer)
    n_assigned = sum(len(gr["params"]) for o in opts for gr in o.param_groups)
    assert n_assigned == sum(1 for p in model.parameters() if p.requires_grad), "param left without an optimizer"
    return CombinedOptimizer(opts)
