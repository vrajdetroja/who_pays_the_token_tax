import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sst

from recount import runs_table
from recount.analysis import analyze as A
from recount.analysis import decompose as D
from recount.analysis.simulate import Truth, simulate
from recount.analysis.stats import fit_log_slope, hksj, invert_ratio_test, pool, sesoi, t_interval, text_lost
from recount.data import flores
from recount.study import load_study


@pytest.fixture(scope="module")
def sim_runs(tmp_path_factory):
    path = tmp_path_factory.mktemp("sim") / "runs.csv"
    runs_table.append_rows(path, simulate(Truth(), seed=0))
    return runs_table.load(path)


def test_shapley_decomposition_is_exact_for_arbitrary_cells():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0.05, 0.03, size=(500, 4)), columns=["D_bb", "D_bt", "D_tb", "D_tt"])
    out = D.add_shapley(df)
    np.testing.assert_allclose(out["residual"] + out["phi_budget"] + out["phi_context"], out["deployed"], atol=1e-12)
    # Each channel is the average of its two marginal effects.
    np.testing.assert_allclose(out["phi_budget"], 0.5 * ((df.D_tb - df.D_bb) + (df.D_tt - df.D_bt)))


def test_shapley_puts_an_interaction_half_in_each_channel():
    df = pd.DataFrame({"D_bb": [0.0], "D_bt": [0.0], "D_tb": [0.0], "D_tt": [0.02]})  # pure interaction
    out = D.add_shapley(df)
    assert out["phi_budget"][0] == pytest.approx(0.01) and out["phi_context"][0] == pytest.approx(0.01)


def test_hksj_with_equal_variances_is_the_t_interval_of_the_means():
    y = np.array([0.010, 0.022, 0.015, 0.031, 0.008])
    p = hksj(y, np.full(5, 1e-6))
    m, lo, hi = t_interval(y)
    assert p.estimate == pytest.approx(m) and p.lo == pytest.approx(lo) and p.hi == pytest.approx(hi)
    assert p.df == 4


def test_hksj_matches_hand_computation_with_unequal_variances():
    y, v = np.array([0.1, 0.3, 0.2]), np.array([0.01, 0.02, 0.04])
    w = 1 / v
    yf = (w * y).sum() / w.sum()
    Q = (w * (y - yf) ** 2).sum()
    tau2 = max(0, (Q - 2) / (w.sum() - (w**2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    mu = (ws * y).sum() / ws.sum()
    se = np.sqrt((ws * (y - mu) ** 2).sum() / 2 / ws.sum())
    p = hksj(y, v)
    assert p.estimate == pytest.approx(mu) and p.se == pytest.approx(se)
    assert p.hi - p.estimate == pytest.approx(sst.t.ppf(0.975, 2) * se)
    assert hksj(y, v, modified=True).se >= p.se


def test_pooled_within_variance_mode_ignores_lucky_seed_trios():
    rows = [{"lang": l, "seed": s, "x": base + e} for l, base, es in
            [("a", 0.01, [0.0, 0.0001, -0.0001]), ("b", 0.02, [0.004, -0.004, 0.0]), ("c", 0.015, [0.003, -0.002, -0.001])]
            for s, e in enumerate(es)]
    df = pd.DataFrame(rows)
    # Per-language weights would hand language "a" (tiny spread) almost all the weight.
    assert abs(pool(df, "x", within="per_language").estimate - 0.01) < abs(pool(df, "x").estimate - 0.01)


def test_ratio_interval_by_test_inversion_agrees_with_the_bootstrap(sim_runs):
    c = D.cells(sim_runs)
    c = c[(c.model == "M") & (c.level == 2.25)]
    inv = invert_ratio_test(c, "D_bb", "D_tt")
    s_hat = 1 - c["D_bb"].mean() / c["D_tt"].mean()
    assert inv["bounded"] and inv["contiguous"] and inv["lo"] < s_hat < inv["hi"]
    boot = A.bootstrap_ratio(c, "D_bb", "D_tt", draws=4000)
    assert abs(boot["lo"] - inv["lo"]) < 0.03 and abs(boot["hi"] - inv["hi"]) < 0.03


def test_simulated_truth_is_recovered(sim_runs):
    t = Truth()
    res = A.run(sim_runs, load_study())
    lv = res["models"]["M"]["levels"]["2.25"]
    for key, truth in [("D_bb", t.rho), ("phi_context", t.gamma)]:
        p = lv["pooled"][key]
        assert p["lo"] < truth < p["hi"], key
    assert 0.8 < lv["identity_ratio"] < 1.2  # kappa = 1 in the simulation
    assert lv["H1_contrast"]["supported"] and not lv["H1_contrast"]["falsified"]
    assert res["controls"]["nc2_stride"]["pass"]
    assert res["models"]["M"]["data_curve"]["all_monotone"]


def test_h1_is_falsified_when_the_residual_dominates(tmp_path):
    path = tmp_path / "runs.csv"
    runs_table.append_rows(path, simulate(Truth(rho=0.12, kappa=0.2, gamma=0.0), seed=1))
    lv = A.run(runs_table.load(path), load_study())["models"]["M"]["levels"]["2.25"]
    assert lv["H1_contrast"]["falsified"] and lv["share"]["estimate"] < 0.5


def test_sealed_output_hides_every_contrast_mean(sim_runs, tmp_path):
    public, sealed = A.split_sealed(A.run(sim_runs, load_study()))

    def keys(o):
        return set(o) | set().union(*(keys(v) for v in o.values())) if isinstance(o, dict) else set()

    assert not keys(public) & {"pooled", "share", "H1_contrast", "residual_tost", "per_language", "H5_dose_step"}
    assert "noise" in json.dumps(public) and "share" in json.dumps(sealed)
    (tmp_path / "results").mkdir()
    digest = A.seal(sealed, "t", tmp_path)
    assert (tmp_path / "results" / "seals" / "t.sha256").read_text().startswith(digest)


def test_data_curve_slope_and_exchange_rate():
    f = np.array([0.125, 0.25, 0.5, 1.0])
    fit = fit_log_slope(f, 1.0 + 0.07 * np.log(1 / f))
    assert fit["b"] == pytest.approx(0.07) and fit["r2"] == pytest.approx(1.0)
    assert sesoi(0.07) == pytest.approx(0.07 * np.log(1 / 0.9))
    assert text_lost(sesoi(0.07), 0.07) == pytest.approx(0.10)


def test_runs_table_keeps_the_latest_evaluation(tmp_path):
    p = tmp_path / "r.csv"
    base = {"lang": "x", "model": "M", "seed": 0, "level": 1.0, "tokenizer": "main", "endpoint": "final",
            "ctx": "tok", "eval_set": "heldout", "control": ""}
    runs_table.append_rows(p, [base | {"bpb": 1.0, "timestamp": "2026-01-01T00:00:00Z"},
                               base | {"bpb": 0.9, "timestamp": "2026-01-02T00:00:00Z"}])
    df = runs_table.load(p)
    assert len(df) == 1 and df["bpb"].iloc[0] == 0.9
    with pytest.raises(ValueError, match="unknown"):
        runs_table.append_rows(p, [base | {"bogus": 1}])


@pytest.mark.skipif(not flores.ROOT.exists(), reason="needs the FLORES-200 download")
def test_decontamination_flags_a_planted_flores_sentence():
    from recount.data import decontam

    ref = flores.load("eus_Latn", "devtest")[:50]
    clean = "Hau testu garbi bat da, ez du FLORESekin ezer partekatzen. " * 5
    planted = clean + " " + ref[7].upper().replace(",", " ,")  # normalization must still match
    zh = flores.load("zho_Hans", "devtest")[:50]
    flags = decontam.contaminated([clean, planted], ref, by_char=False, workers=2)
    assert flags == [False, True]
    assert decontam.contaminated(["无关的文本。" * 10, "前言" + zh[3]], zh, by_char=True, workers=2) == [False, True]


@pytest.mark.parametrize("kappa", [1.0, 0.5])
def test_matched_identity_recovers_the_budget_channel_ratio(tmp_path, kappa):
    path = tmp_path / "runs.csv"
    runs_table.append_rows(path, simulate(Truth(kappa=kappa), seed=2))
    lv = A.run(runs_table.load(path), load_study())["models"]["M"]["levels"]["2.25"]
    assert lv["identity_matched"]["ratio"] == pytest.approx(kappa, abs=0.12)
