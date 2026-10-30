import dataclasses
import json
import random

import numpy as np
import pytest
import torch

from microlm.model import Transformer
from microlm.model.config import ModelConfig
from microlm.train.checkpoint import load_checkpoint
from recount.eval.protocols import evaluate_stream, windows
from recount.smoke import smoke_study, write_text
from recount.tok.build import build_shards, build_tokenizers
from recount.tok.shard import ShardData
from recount.train import plan
from recount.train.stream import DocStream
from recount.train.trainer import find_run, train_job


@pytest.mark.parametrize("n,w,s", [(10, 4, 2), (11, 4, 4), (3, 8, 4), (1000, 64, 32), (999, 48, 12), (65, 64, 1)])
def test_every_target_is_scored_exactly_once(n, w, s):
    seen = np.zeros(n, dtype=int)
    for b, e, first in windows(n, w, s):
        assert e - b <= w and b < first <= e + 1
        seen[first : e + 1] += 1
        # each scored target sees at least w - s tokens of context, or everything before it
        assert all(t - b >= min(w - s, t) for t in range(first, e + 1)) or b == 0
    assert seen[0] == 0 and (seen[1:] == 1).all()


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return Transformer(ModelConfig(vocab_size=300, n_layer=2, d_model=32, n_head=2, max_seq_len=64, pos_type="rope"))


def test_bits_per_byte_does_not_depend_on_batching(tiny_model):
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 299, size=700).astype(np.uint16)
    tb = torch.tensor([1 + (i % 3) for i in range(300)])
    a = evaluate_stream(tiny_model, tokens, tb, 64, 32, "cpu", batch_size=1, dtype="fp32")
    b = evaluate_stream(tiny_model, tokens, tb, 64, 32, "cpu", batch_size=7, dtype="fp32")
    assert a.n_bytes == b.n_bytes == int(tb[torch.from_numpy(tokens[1:].astype(np.int64))].sum())
    assert a.bpb == pytest.approx(b.bpb, abs=1e-6)


def test_context_shuffle_changes_only_unscored_context(tiny_model):
    tokens = np.arange(2000, dtype=np.uint16) % 299
    tb = torch.ones(300, dtype=torch.long)
    plain = evaluate_stream(tiny_model, tokens, tb, 64, 64, "cpu", dtype="fp32")
    shuf = evaluate_stream(tiny_model, tokens, tb, 64, 64, "cpu", dtype="fp32", shuffle_seed=0)
    assert plain.bpb == pytest.approx(shuf.bpb)  # stride = window: no unscored context to replace
    half = evaluate_stream(tiny_model, tokens, tb, 64, 32, "cpu", dtype="fp32", shuffle_seed=0)
    assert half.n_bytes == plain.n_bytes


@pytest.fixture(scope="module")
def smoke_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("smoke")
    study = smoke_study()
    write_text(root, "smk_Cyrl", "Cyrl", seed=1)
    tok_dir = root / "data" / "tokenizers" / "smk_Cyrl"
    build_tokenizers("smk_Cyrl", root / "data" / "text" / "smk_Cyrl", tok_dir, study, 1024 - 257)
    build_shards("smk_Cyrl", root / "data" / "text" / "smk_Cyrl", tok_dir, root / "data" / "shards", study)
    return root, study


def test_doc_stream_reads_the_same_text_at_every_level(smoke_root):
    root, study = smoke_root
    texts = []
    for lv in study.levels:
        d = ShardData(plan.shard_dir(root, "smk_Cyrl", lv, "main"))
        s = DocStream(d.split("train"), d.bos_id, study.seq_len(lv), seed=3)
        texts.append(d.tokenizer.decode_bytes(s.stream[: 5000].tolist())[:2000])
    assert texts[0] == texts[1] == texts[2]


def test_resumed_run_is_bitwise_identical(smoke_root):
    root, study = smoke_root
    job = [j for j in plan.unit_jobs(study, root, "smk_Cyrl", "M", 0, [1.5], deployed=False) if j.parent is None][0]
    op = {"compile": False, "device": "cpu", "log_interval": 10_000, "micro_batch_size": 8, "ckpt_interval": 10_000}
    full = train_job(job, study, root, op)
    w_full = load_checkpoint(find_run(root, job.name) / "model_final.pt")["model"]
    for p in (root / "runs").glob(f"{job.name}__*"):
        import shutil
        shutil.rmtree(p)
    assert train_job(job, study, root, op | {"stop_at_step": job.steps // 3}) == {}
    resumed = train_job(job, study, root, op)
    w_res = load_checkpoint(find_run(root, job.name) / "model_final.pt")["model"]
    assert all(torch.equal(w_full[k], w_res[k]) for k in w_full)
    assert full["steps"] == resumed["steps"] == job.steps


def test_branch_starts_from_its_parent_and_keeps_its_schedule(smoke_root):
    root, study = smoke_root
    jobs = plan.unit_jobs(study, root, "smk_Cyrl", "S", 1, [1.0, 2.25])
    op = {"compile": False, "device": "cpu", "log_interval": 10_000, "micro_batch_size": 8}
    for j in jobs:
        train_job(j, study, root, op)
    dep = next(j for j in jobs if j.endpoint == "deployed")
    summ = json.loads((find_run(root, dep.name) / "summary.json").read_text())
    parent = next(j for j in jobs if j.name == dep.parent)
    assert dep.branch_step in parent.keep_ckpts
    assert summ["trained_tokens"] == (dep.steps - dep.branch_step) * study.training["windows_per_step"] * dep.seq_len
    # the as-deployed branch reads (to within a window) the reference's token budget
    ref = next(j for j in jobs if j.level == 1.0 and j.parent is None)
    assert abs(dep.train_tokens - ref.train_tokens) <= study.training["windows_per_step"] * dep.seq_len


def _tiny_base():
    torch.manual_seed(0)
    return Transformer(ModelConfig(vocab_size=300, n_layer=1, d_model=32, n_head=2, max_seq_len=64, pos_type="rope"))


def test_masked_model_is_the_softmax_over_reachable_rows():
    import torch.nn.functional as F
    from recount.model import DoseLM

    base = _tiny_base()
    unreachable = list(range(200, 299))  # 299 plays the separator: reachable
    m = DoseLM(base, unreachable)
    x, y = torch.randint(0, 200, (2, 16)), torch.randint(0, 200, (2, 16))
    logits, loss = m(x, y, reduction="none")
    assert torch.isinf(logits[..., 200:299]).all() and torch.isfinite(logits[..., :200]).all()
    keep = torch.tensor([i for i in range(300) if i not in unreachable])
    raw, _ = base(x)
    ref = F.cross_entropy(raw[..., keep].float().reshape(-1, len(keep)), torch.searchsorted(keep, y).reshape(-1),
                          reduction="none").view(2, 16)
    torch.testing.assert_close(loss, ref)
    loss.mean().backward()
    assert base.tok_emb.weight.grad[200:299].abs().max() == 0  # masked rows learn nothing


def test_unmasked_model_is_exactly_microlm_and_checkpoints_round_trip():
    from recount.model import DoseLM

    base = _tiny_base()
    x, y = torch.randint(0, 300, (2, 16)), torch.randint(0, 300, (2, 16))
    assert torch.equal(DoseLM(base, None)(x, y)[1], base(x, y)[1])
    a = DoseLM(base, list(range(250, 299)))
    b = DoseLM(_tiny_base(), list(range(250, 299)))
    b.load_state_dict(a.state_dict())
    assert "unreachable" not in a.state_dict()  # the mask comes from the tokenizer, not the checkpoint


def test_masked_evaluation_wastes_no_probability(tiny_model):
    from recount.model import DoseLM

    m = DoseLM(tiny_model, list(range(150, 299)))
    tokens = (np.arange(500) % 150).astype(np.uint16)
    res = evaluate_stream(m, tokens, torch.ones(300, dtype=torch.long), 64, 32, "cpu", dtype="fp32",
                          unreachable=torch.arange(150, 299))
    assert res.wasted_mass == 0.0


def test_identity_branches_read_what_the_deployed_branches_read(smoke_root):
    root, study = smoke_root
    study.endpoints = study.endpoints | {"identity_branches": True}
    jobs = plan.unit_jobs(study, root, "smk_Cyrl", "M", 0, study.levels)
    ref = next(j for j in jobs if j.level == 1.0 and j.parent is None)
    for r in (1.5, 2.25):
        ident = next(j for j in jobs if j.endpoint == f"identity_{r:g}")
        dep = next(j for j in jobs if j.endpoint == "deployed" and j.level == r)
        assert ident.parent == ref.name and ident.branch_step in ref.keep_ckpts
        assert abs(ident.content_fraction - dep.content_fraction) < 1e-12
        assert abs(ident.steps - dep.content_fraction * ref.steps) <= 1


def test_a_model_can_take_its_own_seeds(smoke_root):
    root, study = smoke_root
    study = dataclasses.replace(study, models=study.models | {"M": study.models["M"] | {"seeds": [0, 1, 2]}})
    jobs = plan.grid(study, root, "main", langs=["smk_Cyrl"])
    seeds = {m: sorted({j.seed for j in jobs if j.model == m}) for m in ("M", "S")}
    assert seeds == {"M": [0, 1, 2], "S": list(study.seeds)}
