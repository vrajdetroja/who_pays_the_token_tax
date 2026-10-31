import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from recount.analysis import xglm as AX
from recount.eval import xglm as X


def test_the_language_table_is_the_model_cards():
    assert len(X.LANGS) == 30 and len({code for code, _, _ in X.LANGS.values()}) == 30
    assert sum(share for _, share, _ in X.LANGS.values()) == pytest.approx(1.0, abs=2e-4)  # the card's rounding


def test_hc3_is_the_sum_of_squared_leave_one_out_shifts():
    rng = np.random.default_rng(0)
    n = 30
    Xm = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    y = Xm @ [1.0, -2.0, 0.5] + rng.normal(size=n) * (1 + np.abs(Xm[:, 1]))  # heteroskedastic
    fit = AX.ols_hc3(y, Xm, ["const", "a", "b"])
    beta = np.linalg.lstsq(Xm, y, rcond=None)[0]
    shifts = [beta - np.linalg.lstsq(np.delete(Xm, i, 0), np.delete(y, i), rcond=None)[0] for i in range(n)]
    cov = sum(np.outer(d, d) for d in shifts)
    q = stats.t.ppf(0.975, n - 3)
    for j, name in enumerate(["const", "a", "b"]):
        c = fit["coef"][name]
        assert c["estimate"] == pytest.approx(beta[j])
        assert c["se"] == pytest.approx(math.sqrt(cov[j, j]))
        assert c["lo"] == pytest.approx(beta[j] - q * c["se"])
    assert fit["df"] == 27


def test_regressors_follow_the_amendment():
    rows = [{"model": "564M", "lang": lang, "sentence": s, "n_bytes": nb, "n_tokens": nt, "n_unk": unk, "nats": nats}
            for lang, nb, nt, unk, nats in [("en", 100, 20, 0, 50.0), ("ta", 300, 40, 2, 80.0)] for s in range(1012)]
    t = AX.languages(pd.DataFrame(rows), tool_premiums={"en": 1.0, "ta": 2.5}).set_index("lang")
    assert t.loc["ta", "bits_per_sentence"] == pytest.approx(80 / math.log(2))
    assert t.loc["ta", "b"] == 7.5 and t.loc["ta", "token_premium"] == 2.0 and t.loc["ta", "p_flores"] == 3.0
    assert t.loc["ta", "unk_share"] == pytest.approx(2 / 40)
    assert t.loc["ta", "x_data"] == pytest.approx(math.log(500e9 * 0.0060 * 7.5 / 2.5))
    assert t.loc["ta", "x_data_flores"] == pytest.approx(math.log(500e9 * 0.0060 * 7.5 / 3.0))
    assert t.loc["ta", "x_fert"] == pytest.approx(math.log(2)) and t.loc["en", "x_fert"] == 0
    with pytest.raises(ValueError, match="incomplete"):
        AX.languages(pd.DataFrame(rows[:-1]), tool_premiums={"en": 1.0, "ta": 2.5})


def test_the_sign_is_called_only_when_the_positive_control_holds():
    fit = lambda data, fert: {"coef": {"x_data": dict(zip(["lo", "hi"], data)), "x_fert": dict(zip(["lo", "hi"], fert))}}  # noqa: E731
    assert AX.verdict(fit((-2, -1), (1, 2))) == "agrees"
    assert AX.verdict(fit((-2, -1), (-2, -1))) == "disagrees"
    assert AX.verdict(fit((-2, -1), (-1, 1))) == "inconclusive"
    assert AX.verdict(fit((-2, 0.5), (1, 2))).startswith("inconclusive: the positive control")


def test_batched_scoring_matches_the_models_own_loss():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.XGLMConfig(vocab_size=64, d_model=16, num_layers=2, attention_heads=2, ffn_dim=32,
                                  max_position_embeddings=64)
    torch.manual_seed(0)
    model = transformers.XGLMForCausalLM(cfg).eval()
    gen = torch.Generator().manual_seed(1)
    seqs = [[2] + torch.randint(4, 64, (n,), generator=gen).tolist() for n in [1, 5, 9, 17, 30]]
    # 40 positions per batch and 7 rows per head chunk: several padded batches, several chunks each
    nats = X.score(model, seqs, "cpu", pad_id=cfg.pad_token_id, max_tokens=40, head_rows=7)
    for s, got in zip(seqs, nats):
        ids = torch.tensor([s])
        with torch.no_grad():
            loss = model(input_ids=ids, labels=ids).loss.item()  # mean over the len(s) - 1 predicted tokens
        assert got == pytest.approx(loss * (len(s) - 1), rel=1e-5)
