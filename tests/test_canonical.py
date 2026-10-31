import math

import pandas as pd
import pytest
import torch

from microlm.model import Transformer
from microlm.model.config import ModelConfig
from recount.analysis import appendix as AP
from recount.eval import canonical as C
from recount.eval import canonical_beam as CB
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


# The beam (amendment 4)


def test_top_per_key_keeps_the_best_of_each_key_and_every_forced_row():
    keys, score = torch.tensor([0, 0, 0, 1, 1]), torch.tensor([3.0, 1.0, 2.0, 5.0, 4.0])
    always = torch.tensor([False, True, False, False, False])
    assert CB.top_per_key(keys, score, always, 1).tolist() == [True, True, False, True, False]
    assert CB.top_per_key(keys, score, always, 2).all()


@pytest.mark.parametrize("active", [3, 1])
def test_a_beam_that_keeps_every_spelling_is_exact_enumeration(active):
    tok, lm = tiny(active)
    texts = [b"abcab", b"cab", b"abcabcab", b"b"]
    canons = [tok.encode_fast(t.decode()) for t in texts]
    # 5 partial spellings per forward pass: several passes at most steps
    others, canon_lp, count = CB.beam_mass(lm, tok.bos_id, texts, C.reachable_trie(tok), canons, 10_000, "cpu", chunk=5)
    for t, c, o, b, k in zip(texts, canons, others, canon_lp, count):
        exact, n_seqs = exact_nats(lm, tok, t)
        cn = C.sequence_nats(lm, tok.bos_id, c, "cpu")
        assert b == pytest.approx(-cn, abs=1e-4)  # the beam's own copy of the canonical log-probability
        assert k == n_seqs - 1  # every other spelling completed
        assert CB.marginal_nats(cn, o - b) == pytest.approx(exact, abs=1e-4)


def test_a_narrow_beam_is_a_lower_bound():
    tok, lm = tiny(3)
    text = b"abcabcab"
    c = tok.encode_fast(text.decode())
    cn, (exact, _) = C.sequence_nats(lm, tok.bos_id, c, "cpu"), exact_nats(lm, tok, text)
    for beam in [1, 2, 4]:
        others, canon_lp, _ = CB.beam_mass(lm, tok.bos_id, [text], C.reachable_trie(tok), [c], beam, "cpu")
        assert exact - 1e-6 <= CB.marginal_nats(cn, others[0] - canon_lp[0]) <= cn


def test_exact_scoring_matches_one_sequence_at_a_time():
    tok, lm = tiny(3)
    rows = C.allowed_rows(C.reachable_trie(tok), b"abcab")
    seqs = CB.spellings(rows)
    assert len(seqs) == CB.count_spellings(rows) == 8
    got = CB.spelling_nats(lm, tok.bos_id, seqs, "cpu", batch=3)
    assert got == pytest.approx([C.sequence_nats(lm, tok.bos_id, s, "cpu") for s in seqs], abs=1e-4)


def test_openings_end_where_every_level_keeps_the_sentences_own_tokens():
    toks = {1.0: tiny(3)[0], 2.25: tiny(1)[0]}
    # 'abcab' is [abc][ab] at the widest level, so the openings 'ab' and 'abca' would end mid-token
    assert CB.openings([b"abcab"], toks, cap=4, min_bytes=1) == [b"abc"]
    assert CB.openings([b"abcab"], toks, cap=100, min_bytes=1) == [b"abcab"]
    assert CB.openings([b"abcab"], toks, cap=4, min_bytes=4) == [None]


def test_the_beam_table_reports_what_reaches_d_convergence_and_the_openings(tmp_path):
    unit = {"lang": "eus_Latn", "model": "M", "seed": 0, "untrained": False}
    beams, opens = [], []
    for level, endpoint, gap16, gap64 in [(1.0, "final", 0.1, 0.1), (2.25, "final", 0.4, 0.6), (2.25, "deployed", 0.5, 0.5)]:
        at = unit | {"level": level, "endpoint": endpoint}
        for beam, gap, n in [(16, gap16, 4), (64, gap64, 2)]:
            beams += [at | {"beam": beam, "sentence": s, "n_bytes": 100, "canonical_tokens": 20, "canonical_nats": 150.0,
                            "marginal_nats": 150.0 - gap, "completed": 3, "canonical_check": 0.0} for s in range(n)]
        opens.append(at | {"beam": 16, "sentence": 0, "n_bytes": 20, "spellings": 50, "canonical_tokens": 4,
                           "canonical_nats": 30.0, "exact_nats": 30.0 - math.log(1.5), "beam_nats": 30.0 - math.log(1.25)})
        opens.append(at | {"beam": 16, "sentence": 1, "n_bytes": 0, "spellings": 0, "canonical_tokens": 0,
                           "canonical_nats": math.nan, "exact_nats": math.nan, "beam_nats": math.nan})  # skipped
    pd.DataFrame(beams).to_csv(tmp_path / "beam.csv", index=False)
    pd.DataFrame(opens).to_csv(tmp_path / "openings.csv", index=False)
    t = AP.canonical_beam(tmp_path / "beam.csv", tmp_path / "openings.csv").set_index(["level", "endpoint"])
    assert t.loc[(2.25, "final"), "gap_bpb"] == pytest.approx(0.4 / 100 / math.log(2))
    assert t.loc[(2.25, "final"), "reaches_d"] == pytest.approx(0.3 / 100 / math.log(2))
    assert not t.loc[(2.25, "final"), "converged"] and t.loc[(2.25, "deployed"), "converged"]  # B = 64 rose by half
    assert t.loc[(1.0, "final"), "openings"] == 1  # the skipped sentence is not an opening
    assert t.loc[(1.0, "final"), "recovered_mean"] == pytest.approx(0.5)  # 0.25 of the exact 0.5
    md = AP.canonical_beam_md(t.reset_index())
    assert "reaches D" in md and "recovered, lowest" in md
