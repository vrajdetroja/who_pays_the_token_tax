import json
from pathlib import Path
import math
import shutil

import numpy as np
import pytest
import torch

from microlm.data.loader import TokenData, WindowStream
from microlm.model import ModelConfig, Transformer
from microlm.train.config import RunConfig, TrainConfig, diff, load_run_config
from microlm.train.optim import build_optimizer, newton_schulz, param_roles
from microlm.train.schedule import lr_multiplier
from microlm.train.trainer import train


def tiny_run(data_dir, out_dir, name="tiny", model_kw=None, **train_kw) -> RunConfig:
    model = ModelConfig(**{"n_layer": 2, "d_model": 64, "n_head": 4, "max_seq_len": 64, **(model_kw or {})})
    kw = dict(data_dir=str(data_dir), out_dir=str(out_dir), seq_len=64, tokens_per_step=64 * 8,
              total_tokens=64 * 8 * 12, micro_batch_size=4, warmup_steps=3, eval_tokens=64 * 8,
              eval_interval=4, diag_interval=4, ckpt_interval=4, log_interval=100, dtype="fp32",
              compile=False, device="cpu", lr=3e-3)
    kw.update(train_kw)
    return RunConfig(name=name, model=model, train=TrainConfig(**kw))


def losses(run_dir):
    return [json.loads(l)["train_loss"] for l in (run_dir / "metrics.jsonl").read_text().splitlines()]


# -- resumption -----------------------------------------------------------------------


@pytest.mark.parametrize("optimizer", ["adamw", "muon", "lion"])
def test_resume_is_bitwise_identical(synthetic_data, tmp_path, optimizer):
    full = train(tiny_run(synthetic_data, tmp_path / "a", optimizer=optimizer))
    assert train(tiny_run(synthetic_data, tmp_path / "b", optimizer=optimizer, stop_at_step=6)) is None
    resumed = train(tiny_run(synthetic_data, tmp_path / "b", optimizer=optimizer))
    a, b = tmp_path / "a" / full["run_id"], tmp_path / "b" / resumed["run_id"]
    assert full["run_id"] == resumed["run_id"]
    assert len(losses(a)) == len(losses(b)) == 12
    assert losses(a) == losses(b)  # exact, not approximate
    assert full["val_iid_loss"] == resumed["val_iid_loss"]


def test_seed_controls_init_only(synthetic_data, tmp_path):
    s0 = train(tiny_run(synthetic_data, tmp_path / "x", total_tokens=64 * 8 * 2))
    s1 = train(tiny_run(synthetic_data, tmp_path / "y", total_tokens=64 * 8 * 2, seed=1))
    assert s0["run_id"] != s1["run_id"]
    assert s0["val_iid_loss"] != s1["val_iid_loss"]


def test_finished_run_is_skipped(synthetic_data, tmp_path):
    r = tiny_run(synthetic_data, tmp_path, total_tokens=64 * 8 * 2)
    first = train(r)
    again = train(tiny_run(synthetic_data, tmp_path, total_tokens=64 * 8 * 2))
    assert first == again


def test_training_reduces_loss(synthetic_data, tmp_path):
    s = train(tiny_run(synthetic_data, tmp_path, total_tokens=64 * 8 * 40, lr=1e-2))
    assert s["val_iid_loss"] < math.log(300) - 1.5


# -- run identity ---------------------------------------------------------------------


def test_run_id_ignores_operational_fields():
    a = load_run_config("configs/baseline.yaml")
    b = load_run_config("configs/baseline.yaml", ["train.micro_batch_size=8", "train.eval_interval=7",
                                                   "train.device=cpu", "train.compile=false"])
    assert a.run_id() == b.run_id()
    c = load_run_config("configs/baseline.yaml", ["train.lr=6e-3"])
    assert a.run_id() != c.run_id()


def test_every_ablation_changes_what_it_says(tmp_path):
    from pathlib import Path

    base = load_run_config("configs/baseline.yaml")
    for path in _generated("ablations", "ladder", "reserve"):
        d = diff(base, load_run_config(path))
        assert d, f"{path.name} is identical to the baseline"
        assert all(k.startswith(("model.", "train.")) for k in d)


def test_unknown_fields_rejected():
    with pytest.raises(ValueError, match="unknown"):
        load_run_config("configs/baseline.yaml", ["model.pos_typo=rope"])
    with pytest.raises(ValueError, match="unknown"):
        load_run_config("configs/baseline.yaml", ["train.lr_typo=1"])


def test_baseline_step_arithmetic():
    t = load_run_config("configs/baseline.yaml").train
    assert t.windows_per_step == 256 and t.total_steps == 1620 and t.grad_accum == 4


def _generated(*dirs):
    from pathlib import Path

    return [p for d in dirs for p in sorted(Path("configs", d).glob("*.yaml"))]


def test_vocab_variants_are_matched():
    from microlm.model import flops_per_token, param_counts

    base = load_run_config("configs/baseline.yaml")
    base_params = param_counts(base.model)["total"]
    base_flops = flops_per_token(base.model, 512)["total"] * base.train.total_tokens
    for path in _generated("ablations"):
        if not path.stem.startswith("vocab_"):
            continue
        r = load_run_config(path)
        r.model.vocab_size = int(path.stem.split("_")[1])
        assert abs(param_counts(r.model)["total"] / base_params - 1) < 0.05, path.stem
        flops = flops_per_token(r.model, 512)["total"] * r.train.total_steps * r.train.tokens_per_step
        assert abs(flops / base_flops - 1) < 0.01, path.stem


def test_ladder_shares_the_schedule():
    for path in _generated("ladder"):
        t = load_run_config(path).train
        assert t.total_steps == 1620 and t.warmup_steps == 200, path.stem


# -- schedules ------------------------------------------------------------------------


def _cfg(**kw):
    # 1000 steps at the default tokens_per_step
    return TrainConfig(**{"total_tokens": TrainConfig().tokens_per_step * 1000, **kw})


def test_warmup_is_linear():
    c = _cfg(warmup_steps=100)
    assert lr_multiplier(0, c) == pytest.approx(0.01)
    assert lr_multiplier(49, c) == pytest.approx(0.5)
    assert lr_multiplier(99, c) == pytest.approx(1.0)


@pytest.mark.parametrize("sched", ["cosine", "linear", "wsd"])
@pytest.mark.parametrize("lo", [0.0, 0.1])
def test_schedules_end_at_min(sched, lo):
    c = _cfg(schedule=sched, min_lr_ratio=lo, warmup_steps=50)
    assert lr_multiplier(50, c) == pytest.approx(1.0)
    assert lr_multiplier(999, c) == pytest.approx(lo)
    vals = [lr_multiplier(s, c) for s in range(50, 1000)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))  # monotone after warmup


def test_wsd_is_flat_until_decay():
    c = _cfg(schedule="wsd", decay_frac=0.2, warmup_steps=10)
    assert all(lr_multiplier(s, c) == 1.0 for s in range(10, 800))
    assert lr_multiplier(801, c) < 1.0


def test_wsd_branch_shares_prefix():
    # A shorter WSD run is a valid branch of a longer one until its own decay starts.
    long, short = _cfg(schedule="wsd"), _cfg(schedule="wsd", total_tokens=TrainConfig().tokens_per_step * 500)
    assert all(lr_multiplier(s, long) == lr_multiplier(s, short) for s in range(400))


# -- optimizers -----------------------------------------------------------------------


def test_newton_schulz_orthogonalizes():
    # The quintic is tuned to land singular values in roughly [0.7, 1.2], not exactly at 1.
    torch.manual_seed(0)
    for shape in [(192, 64), (64, 256), (1536, 512)]:
        s = torch.linalg.svdvals(newton_schulz(torch.randn(shape)))
        assert s.min() > 0.6 and s.max() < 1.2, (shape, s.min(), s.max())
    # Square Gaussian matrices have near-zero singular values; 5 steps lift almost all of them.
    s = torch.linalg.svdvals(newton_schulz(torch.randn(512, 512)))
    assert ((s > 0.6) & (s < 1.25)).float().mean() > 0.95
    s = torch.linalg.svdvals(newton_schulz(torch.randn(512, 512), steps=10))
    assert s.min() > 0.6


@pytest.mark.parametrize("opt", ["adamw", "lion", "muon"])
def test_optimizer_covers_every_param_and_never_decays_vectors(opt):
    m = Transformer(ModelConfig(n_layer=2, d_model=64, n_head=4, vocab_size=300, bias=True, tie_embeddings=False))
    o = build_optimizer(m, TrainConfig(optimizer=opt))
    for g in o.param_groups:
        if any(p.dim() < 2 for p in g["params"]):
            assert all(p.dim() < 2 for p in g["params"]) and g["weight_decay"] == 0.0


def test_muon_only_touches_hidden_matrices():
    m = Transformer(ModelConfig(n_layer=2, d_model=64, n_head=4, vocab_size=300, tie_embeddings=False))
    roles = param_roles(m)
    expected = {id(m.tok_emb.weight), id(m.pos_emb.weight), id(m.head.weight)}
    assert expected == {id(p) for p in roles["embeddings"]}
    assert all(p.dim() == 2 for p in roles["matrices"])


# -- data -----------------------------------------------------------------------------


def test_window_stream_is_deterministic_and_bounded(synthetic_data):
    data = TokenData(synthetic_data)
    s1 = WindowStream(data.split("train"), 64, "shuffled", data_seed=0)
    s2 = WindowStream(data.split("train"), 64, "shuffled", data_seed=0)
    np.testing.assert_array_equal(s1.windows(10, 5), s2.windows(10, 5))
    x, y = s1.batch(0, 3, "cpu")
    torch.testing.assert_close(x[:, 1:], y[:, :-1])
    with pytest.raises(IndexError, match="repeat data"):
        s1.windows(s1.n_windows - 1, 2)
    seq = WindowStream(data.split("train"), 64, "sequential")
    np.testing.assert_array_equal(seq.windows(0, 1)[0], data.split("train")[:65])


def test_manifest_detects_corruption(synthetic_data, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(synthetic_data, bad)
    arr = np.memmap(bad / "train.bin", dtype=np.uint16, mode="r+")
    arr[100] ^= 1
    arr.flush()
    del arr
    with pytest.raises(ValueError, match="sha256"):
        TokenData(bad)


# -- evaluation -----------------------------------------------------------------------


def test_extrapolation_eval_on_trained_run(synthetic_data, tmp_path):
    from microlm.data.loader import TokenData
    from microlm.eval.losses import evaluate_tokens, extrapolation_eval
    from microlm.eval.run_eval import load_run_model

    s = train(tiny_run(synthetic_data, tmp_path, total_tokens=64 * 8 * 4))
    s_rope = train(tiny_run(synthetic_data, tmp_path, "rope", {"pos_type": "rope"}, total_tokens=64 * 8 * 4))
    data = TokenData(synthetic_data)
    for summary, can_extrapolate in [(s, False), (s_rope, True)]:
        model, cfg, step = load_run_model(tmp_path / summary["run_id"], "cpu")
        assert step == 4
        r = extrapolation_eval(model, data.split("val_long"), data.bos_id, data.token_bytes, [64, 128, 256], "cpu", "fp32")
        assert r["n_docs"] == 4
        assert ("loss" in r["lengths"][256]) == can_extrapolate
        assert "loss" in r["lengths"][64]
        assert len(r["lengths"][64]["bucket_loss"]) == 64
        # At the training length, the reloaded model reproduces the trainer's own eval.
        ev = evaluate_tokens(model, data.split("val_iid"), data.token_bytes, 64, None, 4, "cpu", "fp32")
        assert abs(ev["loss"] - summary["val_iid_loss"]) < 1e-9


def test_bpb_matches_definition(synthetic_data, tmp_path):
    from microlm.data.loader import TokenData
    from microlm.eval.losses import evaluate_tokens

    data = TokenData(synthetic_data)
    model = Transformer(ModelConfig(n_layer=1, d_model=32, n_head=2, max_seq_len=64, vocab_size=data.vocab_size))
    r = evaluate_tokens(model, data.split("val_iid"), data.token_bytes, 64, None, 4, "cpu", "fp32")
    toks = data.split("val_iid")[1 : r["n_tokens"] + 1]
    assert r["n_bytes"] == len(data.tokenizer.decode_bytes(int(t) for t in toks))
    assert r["bpb"] == pytest.approx(r["loss"] * r["n_tokens"] / r["n_bytes"] / math.log(2))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a non-CPU device")
def test_resume_on_accelerator(synthetic_data, tmp_path):
    # Regression: loading the checkpoint onto the device moved the RNG states there, and
    # torch.set_rng_state rejects non-CPU tensors, so every GPU resume crashed.
    kw = dict(device="mps", dtype="bf16")
    assert train(tiny_run(synthetic_data, tmp_path, stop_at_step=6, **kw)) is None
    resumed = train(tiny_run(synthetic_data, tmp_path, **kw))
    assert resumed["steps"] == 12
    assert len(losses(tmp_path / resumed["run_id"])) == 12


def test_exponent_floats_parse_as_numbers(tmp_path):
    # Regression: PyYAML read "1e-3" as a string, and every such LR crashed inside AdamW.
    r = load_run_config("configs/baseline.yaml", ["train.lr=1e-3", "train.muon_lr=2E-2", "train.eps=1e-08"])
    assert r.train.lr == 0.001 and r.train.muon_lr == 0.02 and r.train.eps == 1e-8
    assert isinstance(r.train.total_tokens, int) and isinstance(r.train.weight_decay, float)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("inherit: " + str(Path("configs/baseline.yaml").resolve()) + "\nname: x\ntrain:\n  lr: 3e-4\n")
    assert load_run_config(cfg).train.lr == 0.0003
    assert load_run_config("configs/baseline.yaml", ["train.lr=1"]).train.lr == 1.0  # int widened


def test_wrong_types_fail_with_the_field_name():
    with pytest.raises(ValueError, match="train.lr must be float"):
        load_run_config("configs/baseline.yaml", ["train.lr=fast"])
    with pytest.raises(ValueError, match="model.n_layer must be int"):
        load_run_config("configs/baseline.yaml", ["model.n_layer=6.5"])


@pytest.mark.parametrize("config,overrides,expected", [
    ("configs/baseline.yaml", ["train.lr=0.004", "train.seed=3"], "baseline__31a6c6843b"),
    ("configs/baseline.yaml", ["train.lr=0.002", "train.seed=0"], "baseline__3b89ac5f65"),
    ("configs/ablations/pos_rope.yaml", ["train.lr=0.004", "train.seed=0"], "pos_rope__023e84e51d"),
    ("configs/ablations/ffn_geglu.yaml", ["train.lr=0.004", "train.seed=0"], "ffn_geglu__ca93f8ac3f"),
])
def test_finished_runs_keep_their_ids(config, overrides, expected):
    # Real IDs of finished runs. If a code change alters them, the protocol scripts would
    # re-run finished work; late-added fields must stay out of the hash at their defaults.
    r = load_run_config(config, overrides)
    r.model.vocab_size = 8192
    assert r.run_id("7cb8d6a46f11f26e:a101f5ae9b99271e") == expected
