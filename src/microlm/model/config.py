"""ModelConfig: every architectural ablation is a change to this dataclass, never to code."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

POS_TYPES = ("learned", "sinusoidal", "rope", "alibi", "nope")
FFN_TYPES = ("gelu", "swiglu", "geglu", "relu2")
GATED_FFNS = ("swiglu", "geglu")
NORM_TYPES = ("layernorm", "rmsnorm")
INIT_SCHEMES = ("gpt2", "torch")

# Fields added after runs already existed. At their default value they are left out of the
# run hash, so adding a field never changes the ID of an earlier run (which would make the
# protocol scripts re-run finished work). Asserted against real run IDs in tests/test_train.py.
HASH_OPTIONAL = {"init_scheme": "gpt2", "sinusoidal_rms": None}


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    n_layer: int = 6
    d_model: int = 384
    n_head: int = 6
    n_kv_head: int | None = None  # None = n_head (MHA); 1 = MQA; otherwise GQA
    max_seq_len: int = 512  # training context; sizes the learned position table

    pos_type: str = "learned"
    rope_base: float = 10000.0
    # None: the raw sin/cos table (RMS 1/sqrt(2) ~ 0.71, far above token embeddings at
    # init_std 0.02). A number rescales the table to that RMS, e.g. init_std, so a fixed table
    # starts at the same magnitude as a learned one.
    sinusoidal_rms: float | None = None

    ffn_type: str = "gelu"
    # FFN parameter count as a multiple of d_model, expressed for a *non-gated* FFN: the
    # FFN gets 2 * ffn_mult * d_model^2 weights whatever its type. A gated FFN has three
    # matrices, so its hidden size is (2/3) * ffn_mult * d_model. See `ffn_hidden`.
    ffn_mult: float = 4.0
    ffn_hidden_override: int | None = None
    ffn_round_to: int = 8

    norm_type: str = "layernorm"
    norm_position: str = "pre"  # pre | post
    final_norm: bool = True
    norm_eps: float = 1e-5
    qk_norm: bool = False

    bias: bool = False
    tie_embeddings: bool = True
    # gpt2: N(0, init_std) for every matrix, residual projections scaled by 1/sqrt(2 * n_layer).
    # torch: PyTorch's nn.TransformerEncoderLayer defaults (Xavier-uniform QKV, PyTorch's
    # default Linear init elsewhere); embeddings stay N(0, init_std). Post-LN collapses to
    # the unigram solution under gpt2 init and trains under torch init.
    init_scheme: str = "gpt2"
    init_std: float = 0.02
    scale_residual_init: bool = True  # gpt2 scheme only: residual output projections get std / sqrt(2 * n_layer)
    pad_vocab_to: int = 64  # embedding rows are padded for kernel efficiency; padded logits are dropped

    def __post_init__(self):
        from ..train.config import check_types

        check_types(self, "model")
        if self.pos_type not in POS_TYPES:
            raise ValueError(f"pos_type must be one of {POS_TYPES}")
        if self.ffn_type not in FFN_TYPES:
            raise ValueError(f"ffn_type must be one of {FFN_TYPES}")
        if self.norm_type not in NORM_TYPES:
            raise ValueError(f"norm_type must be one of {NORM_TYPES}")
        if self.init_scheme not in INIT_SCHEMES:
            raise ValueError(f"init_scheme must be one of {INIT_SCHEMES}")
        if self.norm_position not in ("pre", "post"):
            raise ValueError("norm_position must be 'pre' or 'post'")
        if self.d_model % self.n_head:
            raise ValueError("d_model must be divisible by n_head")
        if self.n_head % self.kv_heads:
            raise ValueError("n_head must be divisible by n_kv_head")
        if self.pos_type == "rope" and self.d_head % 2:
            raise ValueError("RoPE needs an even head dimension")

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head

    @property
    def kv_heads(self) -> int:
        return self.n_kv_head or self.n_head

    @property
    def gated(self) -> bool:
        return self.ffn_type in GATED_FFNS

    @property
    def ffn_hidden(self) -> int:
        if self.ffn_hidden_override is not None:
            return self.ffn_hidden_override
        h = self.ffn_mult * self.d_model * (2 / 3 if self.gated else 1)
        r = self.ffn_round_to
        return max(r, int(round(h / r)) * r)

    @property
    def padded_vocab(self) -> int:
        p = self.pad_vocab_to
        return -(-self.vocab_size // p) * p

    def to_dict(self) -> dict:
        return asdict(self)

    def hash_dict(self) -> dict:
        """to_dict() without late-added fields that sit at their defaults (see HASH_OPTIONAL)."""
        d = self.to_dict()
        for k, default in HASH_OPTIONAL.items():
            if d.get(k) == default:
                d.pop(k)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown ModelConfig fields: {sorted(unknown)}")
        return cls(**d)


def param_counts(cfg: ModelConfig) -> dict[str, int]:
    """Analytic parameter counts, excluding vocab padding rows.

    Conventions (stated once, used in every table):
      embedding      token embedding + learned position table (if any)
      head           the unembedding matrix if untied, else 0 (it *is* the embedding)
      non_embedding  everything else: attention, FFN, norms (Kaplan et al.'s N)
      total          embedding + head + non_embedding
    `tests/test_model.py` asserts these match the instantiated module exactly.
    """
    d, L, V = cfg.d_model, cfg.n_layer, cfg.vocab_size
    Dh, H, Hkv = cfg.d_head, cfg.n_head, cfg.kv_heads
    b = 1 if cfg.bias else 0
    norm = d * (2 if (cfg.norm_type == "layernorm" and cfg.bias) else 1)

    attn = d * (H + 2 * Hkv) * Dh + b * (H + 2 * Hkv) * Dh  # qkv
    attn += H * Dh * d + b * d  # output projection
    if cfg.qk_norm:
        attn += 2 * Dh
    h = cfg.ffn_hidden
    up = 2 * h if cfg.gated else h
    ffn = d * up + b * up + h * d + b * d
    block = attn + ffn + 2 * norm
    non_emb = L * block + (norm if cfg.final_norm else 0)

    emb = V * d + (cfg.max_seq_len * d if cfg.pos_type == "learned" else 0)
    head = 0 if cfg.tie_embeddings else V * d
    return {
        "total": emb + head + non_emb,
        "non_embedding": non_emb,
        "embedding": emb,
        "head": head,
    }


def flops_per_token(cfg: ModelConfig, seq_len: int) -> dict[str, float]:
    """Training FLOPs per token (forward + backward).

    `6N` is the textbook estimate with N = non-embedding params. It omits two terms that are
    not small at this scale: the unembedding matmul (2*d*V per token forward, paid whether or
    not the weights are tied) and attention's QK^T and AV products (4*T*d per layer forward).
    `total` adds both, following the PaLM appendix-B convention (no discount for the causal
    mask). MFU in this repo is computed from `total`.
    """
    pc = param_counts(cfg)
    d, L, V = cfg.d_model, cfg.n_layer, cfg.vocab_size
    six_n = 6.0 * pc["non_embedding"]
    head = 6.0 * d * V
    attn = 12.0 * L * seq_len * d
    return {"6N": six_n, "head": head, "attention": attn, "total": six_n + head + attn}
