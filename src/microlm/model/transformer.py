"""The decoder-only transformer, assembled from a ModelConfig."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .attention import Attention, AttnContext
from .config import ModelConfig, param_counts
from .mlp import MLP
from .norm import LayerNorm, RMSNorm, make_norm
from .pos import alibi_bias, rope_cos_sin, sinusoidal_table


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.post = cfg.norm_position == "post"
        self.norm1 = make_norm(cfg)
        self.attn = Attention(cfg)
        self.norm2 = make_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, ctx: AttnContext) -> torch.Tensor:
        if self.post:  # Vaswani et al.: normalize after the residual add
            x = self.norm1(x + self.attn(x, ctx))
            return self.norm2(x + self.mlp(x))
        x = x + self.attn(self.norm1(x), ctx)
        return x + self.mlp(self.norm2(x))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.padded_vocab, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model) if cfg.pos_type == "learned" else None
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.final_norm = make_norm(cfg) if cfg.final_norm else nn.Identity()
        self.head = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.padded_vocab, bias=False)
        self.reset_parameters()

    # -- initialization ------------------------------------------------------------------

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """N(0, init_std) for every matrix and embedding; residual output projections get
        init_std / sqrt(2 * n_layer) so the residual stream's variance does not grow with
        depth (GPT-2). Biases start at zero, norm gains at one. Padding rows are zeroed."""
        cfg = self.cfg
        if cfg.init_scheme == "torch":
            self._torch_init()
            return
        resid_std = cfg.init_std / math.sqrt(2 * cfg.n_layer) if cfg.scale_residual_init else cfg.init_std
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                std = resid_std if getattr(module.weight, "is_residual_out", False) else cfg.init_std
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (LayerNorm, RMSNorm)):
                nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
        self.tok_emb.weight[cfg.vocab_size :].zero_()
        if self.head is not None:
            self.head.weight[cfg.vocab_size :].zero_()

    @torch.no_grad()
    def _torch_init(self) -> None:
        """nn.TransformerEncoderLayer's initialization: Xavier-uniform on the fused QKV matrix
        (as nn.MultiheadAttention's in_proj), PyTorch's default Linear init (Kaiming-uniform,
        a=sqrt(5)) elsewhere, zero biases. Embeddings keep N(0, init_std), matching the
        reference run that showed post-LN training under this scheme."""
        cfg = self.cfg
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=cfg.init_std)
            elif isinstance(module, (LayerNorm, RMSNorm)):
                nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.qkv.weight)
        self.tok_emb.weight[cfg.vocab_size :].zero_()
        if self.head is not None:
            self.head.weight[cfg.vocab_size :].zero_()

    # -- bookkeeping --------------------------------------------------------------------

    def param_counts(self) -> dict[str, int]:
        """Counts from the live module (padding rows excluded), in param_counts' conventions."""
        cfg = self.cfg
        pad = (cfg.padded_vocab - cfg.vocab_size) * cfg.d_model
        all_params = sum(p.numel() for p in self.parameters())
        emb = self.tok_emb.weight.numel() - pad + (self.pos_emb.weight.numel() if self.pos_emb is not None else 0)
        head = self.head.weight.numel() - pad if self.head is not None else 0
        total = all_params - pad * (1 if self.head is None else 2)
        return {"total": total, "non_embedding": total - emb - head, "embedding": emb, "head": head}

    def check_param_counts(self) -> dict[str, int]:
        live, analytic = self.param_counts(), param_counts(self.cfg)
        if live != analytic:
            raise AssertionError(f"param count mismatch: module {live} vs analytic {analytic}")
        return live

    # -- forward ------------------------------------------------------------------------

    def _context(self, T: int, device, naive: bool, probs_hook) -> AttnContext:
        cfg = self.cfg
        ctx = AttnContext(naive=naive, probs_hook=probs_hook)
        if cfg.pos_type == "rope":
            ctx.rope = rope_cos_sin(T, cfg.d_head, cfg.rope_base, device)
        elif cfg.pos_type == "alibi":
            ctx.alibi = alibi_bias(cfg.n_head, T, device)
        return ctx

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        reduction: str = "mean",
        naive_attn: bool = False,
        probs_hook: Callable[[torch.Tensor], None] | None = None,
        resid_hook: Callable[[torch.Tensor], None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (logits, loss). logits are over the true vocab (padding dropped).

        reduction="none" returns per-token losses of shape (B, T), used by the evaluators.
        probs_hook / resid_hook are called once per layer, in order, with that layer's attention
        probabilities (B, H, T, T) and output residual stream; for diagnostics. probs_hook forces
        the naive attention path. Pass `some_list.append` to collect them.
        """
        cfg = self.cfg
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            if T > cfg.max_seq_len:
                raise ValueError(
                    f"learned positions are defined only up to {cfg.max_seq_len}; got length {T}"
                )
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        elif cfg.pos_type == "sinusoidal":
            table = sinusoidal_table(T, cfg.d_model, idx.device)
            if cfg.sinusoidal_rms is not None:  # the raw table's RMS is exactly 1/sqrt(2)
                table = table * (cfg.sinusoidal_rms * math.sqrt(2.0))
            x = x + table.to(x.dtype)

        ctx = self._context(T, idx.device, naive_attn, probs_hook)
        for block in self.blocks:
            x = block(x, ctx)
            if resid_hook is not None:
                resid_hook(x.detach())
        x = self.final_norm(x)

        w = self.tok_emb.weight if self.head is None else self.head.weight
        logits = F.linear(x, w[: cfg.vocab_size])
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.float().view(-1, cfg.vocab_size), targets.reshape(-1), reduction=reduction
        )
        if reduction == "none":
            loss = loss.view(B, T)
        return logits, loss
