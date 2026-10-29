import itertools

import pytest
import torch

from microlm.model import ModelConfig, Transformer, flops_per_token, param_counts

TINY = dict(vocab_size=300, n_layer=2, d_model=64, n_head=4, max_seq_len=32)


def tiny(**kw) -> ModelConfig:
    return ModelConfig(**{**TINY, **kw})


VARIANTS = [
    dict(pos_type=p) for p in ("learned", "sinusoidal", "rope", "alibi", "nope")
] + [
    dict(ffn_type=f) for f in ("swiglu", "geglu", "relu2")
] + [
    dict(norm_type="rmsnorm"),
    dict(norm_position="post"),
    dict(final_norm=False),
    dict(qk_norm=True),
    dict(bias=True),
    dict(bias=True, norm_type="layernorm"),
    dict(tie_embeddings=False),
    dict(n_kv_head=2),
    dict(n_kv_head=1, pos_type="rope"),
    dict(n_head=8, n_kv_head=2, pos_type="alibi", qk_norm=True),
]


@pytest.mark.parametrize("kw", VARIANTS, ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items()))
def test_analytic_param_count_matches_module(kw):
    Transformer(tiny(**kw)).check_param_counts()


@pytest.mark.parametrize("ffn", ["swiglu", "geglu"])
@pytest.mark.parametrize("d", [256, 384, 512, 768])
def test_gated_ffn_is_parameter_matched(ffn, d):
    base = param_counts(ModelConfig(d_model=d, n_head=d // 64, ffn_type="gelu"))["non_embedding"]
    gated = param_counts(ModelConfig(d_model=d, n_head=d // 64, ffn_type=ffn))["non_embedding"]
    assert abs(gated - base) / base < 0.005, (gated, base)


def test_unmatched_swiglu_would_fail():
    # The trap the assertion above guards against: 4d hidden size with three matrices.
    cfg = ModelConfig(ffn_type="swiglu", ffn_hidden_override=4 * 512)
    base = param_counts(ModelConfig())["non_embedding"]
    assert param_counts(cfg)["non_embedding"] / base > 1.3


def test_baseline_size():
    pc = param_counts(ModelConfig())
    assert 10.6e6 < pc["non_embedding"] < 10.65e6  # 12 * L * d^2 = 10.62M plus norms


@pytest.mark.parametrize("L,d", [(4, 768), (8, 512), (16, 384), (32, 256)])
def test_depth_width_table(L, d):
    n = param_counts(ModelConfig(n_layer=L, d_model=d, n_head=d // 64))["non_embedding"]
    assert abs(n - 12 * L * d * d) / n < 0.002


def test_flops_terms():
    f = flops_per_token(ModelConfig(), 1024)
    assert f["total"] == f["6N"] + f["head"] + f["attention"]
    assert f["head"] == 6 * 384 * 8192


@pytest.mark.parametrize("kw", [v for v in VARIANTS if "ffn_type" not in v and "bias" not in v],
                         ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items()))
def test_fast_attention_matches_naive(kw):
    torch.manual_seed(0)
    model = Transformer(tiny(**kw)).eval()
    idx = torch.randint(0, 300, (3, 32))
    with torch.no_grad():
        fast, _ = model(idx)
        naive, _ = model(idx, naive_attn=True)
    torch.testing.assert_close(fast, naive, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("pos", ["learned", "sinusoidal", "rope", "alibi", "nope"])
def test_causality(pos):
    torch.manual_seed(0)
    model = Transformer(tiny(pos_type=pos)).eval()
    idx = torch.randint(0, 300, (1, 32))
    idx2 = idx.clone()
    idx2[0, 20:] = (idx2[0, 20:] + 1) % 300
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(idx2)
    torch.testing.assert_close(a[:, :20], b[:, :20])
    assert not torch.allclose(a[:, 20:], b[:, 20:])


def test_extrapolation_support():
    for pos in ("sinusoidal", "rope", "alibi", "nope"):
        Transformer(tiny(pos_type=pos))(torch.randint(0, 300, (1, 100)))
    with pytest.raises(ValueError, match="learned positions"):
        Transformer(tiny(pos_type="learned"))(torch.randint(0, 300, (1, 100)))


def test_loss_at_init_is_log_vocab():
    torch.manual_seed(0)
    model = Transformer(tiny())
    idx = torch.randint(0, 300, (8, 32))
    targets = torch.randint(0, 300, (8, 32))  # independent of the input: tied weights would leak it
    _, loss = model(idx, targets)
    assert abs(loss.item() - torch.log(torch.tensor(300.0)).item()) < 0.1


def test_padding_is_invisible():
    model = Transformer(tiny())
    assert model.tok_emb.weight.shape[0] == 320
    logits, _ = model(torch.randint(0, 300, (1, 8)))
    assert logits.shape[-1] == 300


def test_per_token_loss_and_diagnostics():
    model = Transformer(tiny())
    idx = torch.randint(0, 300, (2, 16))
    probs, resid = [], []
    _, loss = model(idx, idx, reduction="none", probs_hook=probs.append, resid_hook=resid.append)
    assert loss.shape == (2, 16)
    assert len(probs) == 2 and probs[0].shape == (2, 4, 16, 16)
    torch.testing.assert_close(probs[0].sum(-1), torch.ones(2, 4, 16))
    assert len(resid) == 2


def test_residual_init_scaling():
    torch.manual_seed(0)
    cfg = ModelConfig(n_layer=8, d_model=256, n_head=4, vocab_size=300)
    m = Transformer(cfg)
    std_out = m.blocks[0].attn.proj.weight.std().item()
    std_in = m.blocks[0].attn.qkv.weight.std().item()
    assert abs(std_in - 0.02) < 0.001
    assert abs(std_out - 0.02 / 4) < 0.0005


def test_param_tags_survive_device_move():
    m = Transformer(tiny(ffn_type="swiglu")).to(torch.float64)
    assert m.blocks[0].attn.qkv.weight.muon_split == (64, 64, 64)
    assert m.blocks[0].mlp.up.weight.muon_split == (m.cfg.ffn_hidden,) * 2


def test_torch_init_scheme():
    torch.manual_seed(0)
    m = Transformer(tiny(init_scheme="torch", norm_position="post"))
    d = 64
    qkv = m.blocks[0].attn.qkv.weight  # (3d, d): Xavier-uniform bound sqrt(6 / (d + 3d))
    assert abs(qkv.abs().max().item() - (6 / (4 * d)) ** 0.5) < 0.02
    up = m.blocks[0].mlp.up.weight  # PyTorch default Linear: U(+-1/sqrt(fan_in))
    assert up.abs().max().item() <= 1 / d**0.5 + 1e-6 and up.std().item() > 0.05
    assert abs(m.tok_emb.weight[:300].std().item() - 0.02) < 0.002
    m.check_param_counts()


def test_sinusoidal_rms():
    from microlm.model.pos import sinusoidal_table

    assert abs(sinusoidal_table(100, 64).pow(2).mean().sqrt().item() - 2 ** -0.5) < 1e-6
    torch.manual_seed(0)
    a = Transformer(tiny(pos_type="sinusoidal", sinusoidal_rms=0.02))
    idx = torch.randint(0, 300, (1, 16))
    with torch.no_grad():
        x_tok = a.tok_emb(idx)
        out = []
        a.blocks[0].register_forward_pre_hook(lambda mod, args: out.append(args[0]))
        a(idx)
    pe = out[0] - x_tok
    assert abs(pe.pow(2).mean().sqrt().item() - 0.02) < 1e-4
