import math

import pandas as pd
import pytest
import torch

from microlm.model import Transformer
from microlm.model.config import ModelConfig
from recount.analysis import appendix as AP
from recount.eval import canonical as C
from recount.model import DoseLM
from recount.tok.dose import DoseTokenizer


def tiny(active: int):
    """'ab', 'abc' and 'bc' as merges; a random 1-layer model over their 260-entry id space."""
    tok = DoseTokenizer([(97, 98), (256, 99), (98, 99)], active=active)
    cfg = ModelConfig(vocab_size=tok.vocab_size, n_layer=1, d_model=16, n_head=2, max_seq_len=32, pos_type="rope")
    torch.manual_seed(0)
    base = Transformer(cfg)
    with torch.no_grad():  # sharpen the random logits so tokenizations differ in probability
        base.tok_emb.weight.mul_(40)
    return tok, DoseLM(base, tok.unreachable_ids()).eval()


def exact_nats(lm, tok, text: bytes) -> tuple[float, int]:
    """-log of the summed probability of every reachable token sequence spelling `text`."""
    ids = {tok.vocab_bytes[i]: i for i in range(256 + tok.active)}

    def spell(pos, seq):
        if pos == len(text):
            yield list(seq)
        for end in range(pos + 1, len(text) + 1):
            if text[pos:end] in ids:
                yield from spell(end, seq + [ids[text[pos:end]]])

    probs = [math.exp(-C.sequence_nats(lm, tok.bos_id, t, "cpu")) for t in spell(0, [])]
    return -math.log(sum(probs)), len(probs)


def test_allowed_tables_list_exactly_the_reachable_tokens_at_each_offset():
    tok, _ = tiny(active=3)
    ids, lens, valid = C.allowed_tables(C.reachable_trie(tok), b"abc", "cpu")
    at = lambda j: sorted((int(i), int(n)) for i, n, v in zip(ids[j], lens[j], valid[j]) if v)  # noqa: E731
    assert at(0) == [(97, 1), (256, 2), (257, 3)] and at(1) == [(98, 1), (258, 2)] and at(2) == [(99, 1)]
    tok1, _ = tiny(active=1)  # only 'ab' is reachable
    ids, lens, valid = C.allowed_tables(C.reachable_trie(tok1), b"abc", "cpu")
    assert sorted(int(i) for i, v in zip(ids[0], valid[0]) if v) == [97, 256]


@pytest.mark.parametrize("active", [3, 1])
def test_importance_sampling_recovers_the_exact_marginal(active):
    tok, lm = tiny(active)
    text = b"abcab"
    exact, n_seqs = exact_nats(lm, tok, text)
    assert n_seqs > 1
    canonical = tok.encode_fast(text.decode())
    c = C.sequence_nats(lm, tok.bos_id, canonical, "cpu")
    gen = torch.Generator().manual_seed(0)
    logw, is_canon = C.sample_weights(lm, tok.bos_id, [text, b"cab"], C.reachable_trie(tok),
                                      [canonical, tok.encode_fast("cab")], 8192, gen, "cpu", chunk=3000)
    est = C.marginal_nats(c, logw[0], is_canon[0])
    assert exact <= c and est <= c  # the marginal never scores worse than the canonical sequence
    assert est == pytest.approx(exact, abs=0.03)


def test_last_logits_match_the_full_forward():
    tok, lm = tiny(active=1)
    idx = torch.tensor([[tok.bos_id, 97, 98, 256]])
    full = lm(idx)[0][:, -1].float()
    assert torch.allclose(C.last_logits(lm, idx), full, atol=1e-5)
    assert torch.isinf(C.last_logits(lm, idx)[0, 257])  # 'abc' is unreachable at this level


def test_the_appendix_gap_table_subtracts_each_units_reference(tmp_path):
    rows = []
    for level, endpoint, gap_nats in [(1.0, "final", 0.2), (2.25, "final", 0.5), (2.25, "deployed", 0.9)]:
        for s in range(3):
            rows.append({"lang": "eus_Latn", "model": "M", "seed": 0, "level": level, "endpoint": endpoint, "draws": 8,
                         "untrained": False, "sentence": s, "n_bytes": 100, "canonical_tokens": 20,
                         "canonical_nats": 150.0, "marginal_nats": 150.0 - gap_nats,
                         "marginal_nats_half": 150.0 - gap_nats / 2, "draws_canonical": 0.5})
    path = tmp_path / "gap.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    t = AP.canonical_gap(path).set_index(["level", "endpoint"])
    assert t.loc[(2.25, "deployed"), "gap_bpb"] == pytest.approx(0.9 / 100 / math.log(2))
    assert t.loc[(2.25, "final"), "gap_minus_reference"] == pytest.approx(0.3 / 100 / math.log(2))
    assert "reaches D" in AP.canonical_gap_md(t.reset_index())
