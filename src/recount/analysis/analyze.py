"""The preregistered analysis, end to end (PREREGISTRATION.md §7).

    python -m recount.analysis.analyze --runs results/all_runs.csv --out results/analysis.json
    python -m recount.analysis.analyze --runs results/scout_runs.csv --seal scout

Primary test (H1): model M, r = 2.25, held-out FineWeb-2, pooled over languages. With
L = D(tok, tok) / 2 - D(byte, byte) per (language, seed), "content accounting removes at
least half the penalty" (S >= 0.5) is L >= 0, and H1 is falsified iff the pooled 95% HKSJ
interval for L lies entirely below 0. The interval for S itself is the set of s whose
contrast (1 - s) D(tok, tok) - D(byte, byte) is not rejected (Fieller's construction); a
parametric bootstrap is reported beside it. Everything else is secondary.

With --seal, every number built from a level contrast's *mean* goes to sealed/<name>.json
(git-ignored), only its SHA-256 is written to results/seals/, and the printout shows noise,
controls, and within-level quantities only. Open the sealed file after the main analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import runs_table
from ..study import Study, load_study
from . import decompose as D
from .stats import (bootstrap_ratio, hksj, invert_ratio_test, language_summaries, pool, sesoi, t_interval,
                    text_lost, tost, within_variances)


def pool_difference(a: pd.DataFrame, b: pd.DataFrame, col: str, within: str = "pooled", level: float = 0.95):
    """Pooled per-language difference mean_a - mean_b for unpaired groups (e.g. two model
    sizes), each language's variance the sum of the two means' variances."""
    sa, sb = language_summaries(a, col), language_summaries(b, col)
    common = sa.index.intersection(sb.index)
    sa, sb = sa.loc[common], sb.loc[common]
    v = within_variances(sa, within) + within_variances(sb, within)
    return hksj((sa["mean"] - sb["mean"]).to_numpy(), v, level)


def noise(c: pd.DataFrame, col: str) -> dict:
    """Seed noise of a paired level contrast (sigma_delta) and its between-language spread."""
    summ = language_summaries(c, col)
    dof = summ["n"] - 1
    sigma = float(np.sqrt((dof * summ["var"]).sum() / dof.sum()))
    tau = math.sqrt(hksj(summ["mean"].to_numpy(), within_variances(summ)).tau2) if len(summ) > 1 else float("nan")
    return {"sigma_delta": sigma, "tau": tau, "languages": len(summ), "seeds_per_language": summ["n"].tolist()}


def per_language(c: pd.DataFrame, cols: list[str]) -> dict:
    out = {}
    for lang, g in c.groupby("lang"):
        out[lang] = {col: dict(zip(("mean", "lo", "hi"), t_interval(g[col]))) for col in cols}
    return out


def analyze_model(runs: pd.DataFrame, study: Study, model: str, eval_set: str, within: str) -> dict:
    a = study.analysis
    curve = D.data_curve(runs, eval_set)
    curve = curve[curve["model"] == model]
    b_pool = pool(curve, "b", within=within)
    b = b_pool.estimate
    bound = sesoi(b, a["sesoi_text_fraction"])
    c_all = D.cells(runs, eval_set)
    c_all = c_all[c_all["model"] == model]
    out: dict = {"data_curve": {"b": b_pool.to_dict(), "sesoi": bound,
                                "all_monotone": bool(curve["monotone"].all()),
                                "min_r2": float(curve["r2"].min()),
                                "mean_curvature": float(curve["curvature"].mean())},
                 "levels": {}}
    cols = ["D_tt", "D_tb", "D_bt", "D_bb", "phi_budget", "phi_context", "share"]
    matched = D.identity_matched(runs, eval_set)
    matched = matched[matched["model"] == model]
    for level, c in c_all.groupby("level"):
        c = c.copy()
        lr = float(np.log(c["r_measured"]).mean())  # realized dose, not the nominal level
        c["H1_L"] = (1 - a["falsification_share"]) * c["D_tt"] - c["D_bb"]  # >= 0 iff S >= threshold
        curve_b = curve.set_index(["lang", "seed"])["b"]
        b_unit = c.set_index(["lang", "seed"]).index.map(curve_b).to_numpy()
        c["identity_gap"] = c["phi_budget"] - b_unit * np.log(c["r_measured"].to_numpy())
        pooled = {col: pool(c, col, within=within).to_dict() for col in cols[:-1] + ["identity_gap"]}
        L = pool(c, "H1_L", within=within)
        mu_bb, mu_tt = pooled["D_bb"]["estimate"], pooled["D_tt"]["estimate"]
        share = 1 - mu_bb / mu_tt
        res = {
            "n_units": len(c),
            "pooled": pooled,
            "share": {"estimate": share,
                      "ci_fieller_hksj": invert_ratio_test(c, "D_bb", "D_tt", within=within),
                      "ci_bootstrap": bootstrap_ratio(c, "D_bb", "D_tt", draws=a["bootstrap_draws"],
                                                      seed=a["bootstrap_seed"], within=within)},
            "H1_contrast": L.to_dict() | {"falsified": bool(L.hi < 0), "supported": bool(L.lo > 0)},
            "residual_tost": tost(c, "D_bb", bound, within=within),
            "identity_ratio": pooled["phi_budget"]["estimate"] / (b * lr),
            "text_lost": {k: float(text_lost(pooled[k]["estimate"], b)) for k in ("D_tt", "D_bb")},
            "noise": {"D_bb": noise(c, "D_bb"), "D_tt": noise(c, "D_tt")},
            "per_language": per_language(c, cols),
        }
        m = c.merge(matched, on=["lang", "model", "seed", "level"], how="inner")
        if len(m) and m["lang"].nunique() > 1:  # H3, matched form (identity branches)
            m["identity_gap_matched"] = m["phi_budget"] - m["matched_prediction"]
            pm = pool(m, "matched_prediction", within=within)
            res["identity_matched"] = {
                "prediction": pm.to_dict(),
                "gap": pool(m, "identity_gap_matched", within=within).to_dict(),
                "ratio": pool(m, "phi_budget", within=within).estimate / pm.estimate,
            }
        out["levels"][f"{level:g}"] = res
    if {1.5, 2.25} <= set(c_all["level"]):
        w = c_all.pivot_table(index=["lang", "seed"], columns="level", values="D_tt").dropna().reset_index()
        w["step"] = w[2.25] - w[1.5]
        out["H5_dose_step"] = pool(w, "step", within=within).to_dict()
    return out


def controls(runs: pd.DataFrame, study: Study, eval_set: str, bound_by_model: dict[str, float]) -> dict:
    out: dict = {}
    sw = D.swap_control(runs, study.tokenizer["swap_level"], eval_set)
    for model, g in sw.groupby("model"):
        m, lo, hi = t_interval(g["diff"])
        _, lo90, hi90 = t_interval(g["diff"], 0.90)
        bound = bound_by_model.get(model, float("nan"))
        out[f"nc1_swap_{model}"] = {"mean": m, "lo": lo, "hi": hi, "n": len(g),
                                    "inside_seed_noise": bool(lo <= 0 <= hi),
                                    "equivalent_within_sesoi": bool(-bound < lo90 and hi90 < bound)}
    st = D.control_effect(runs, "stride", eval_set)
    if len(st):
        out["nc2_stride"] = {"max_abs_diff": float(st["diff"].abs().max()), "n": len(st),
                             "pass": bool(st["diff"].abs().max() < 0.001)}
    sh = D.control_effect(runs, "shuffle", eval_set)
    for model, g in sh.groupby("model"):
        p = pool(g.groupby(["lang", "seed"], as_index=False)["diff"].mean(), "diff")
        out[f"pc2_shuffle_{model}"] = p.to_dict() | {
            "informative": bool(p.lo > bound_by_model.get(model, 0.0))}
    return out


def check_predictions(result: dict, predictions: dict) -> list[dict]:
    """Compare filled-in numeric predictions (study.yaml) with the pooled estimates."""
    rows = []
    lv = result["models"].get("M", {}).get("levels", {}).get("2.25")
    if not lv:
        return rows
    ratio = lv["identity_matched"]["ratio"] if "identity_matched" in lv else lv["identity_ratio"]
    observed = {"share_2.25": lv["share"]["estimate"], "D_tt_2.25": lv["pooled"]["D_tt"]["estimate"],
                "D_bb_2.25": lv["pooled"]["D_bb"]["estimate"], "identity_ratio_2.25": ratio}
    for key, rng in (predictions or {}).items():
        if rng is None or key not in observed:
            continue
        lo, hi = rng
        rows.append({"prediction": key, "range": [lo, hi], "observed": observed[key],
                     "held": bool(lo <= observed[key] <= hi)})
    return rows


def run(runs: pd.DataFrame, study: Study, eval_set: str = "heldout", within: str = "pooled") -> dict:
    result: dict = {"eval_set": eval_set, "within": within, "models": {}}
    bounds = {}
    for model in sorted(runs["model"].unique()):
        try:
            result["models"][model] = analyze_model(runs, study, model, eval_set, within)
            bounds[model] = result["models"][model]["data_curve"]["sesoi"]
        except ValueError as e:  # e.g. the scout has no deployed endpoints
            result["models"][model] = {"error": str(e)}
    if {"M", "S"} <= set(result["models"]) and all("levels" in result["models"][m] for m in ("M", "S")):
        c = D.cells(runs, eval_set)
        cm, cs = c[(c.model == "M") & (c.level == 2.25)], c[(c.model == "S") & (c.level == 2.25)]
        if len(cm) and len(cs):
            result["H4_size_M_minus_S"] = pool_difference(cm, cs, "D_bb", within).to_dict()
    result["controls"] = controls(runs, study, eval_set, bounds)
    result["predictions"] = check_predictions(result, study.predictions)
    return result


# ---------------------------------------------------------------------------------------
# Sealing (the scout)

SEALED_KEYS = {"pooled", "share", "H1_contrast", "residual_tost", "identity_ratio", "identity_matched", "text_lost",
               "per_language", "H5_dose_step", "H4_size_M_minus_S", "predictions"}


def split_sealed(obj):
    """(public, sealed): drop every key whose value depends on the mean of a level contrast."""
    if isinstance(obj, dict):
        pub, sea = {}, {}
        for k, v in obj.items():
            if k in SEALED_KEYS:
                sea[k] = v
            else:
                p, s = split_sealed(v)
                pub[k] = p
                if s not in ({}, None):
                    sea[k] = s
        return pub, sea
    return obj, None


def seal(sealed: dict, name: str, root: Path) -> str:
    text = json.dumps(sealed, indent=2, sort_keys=True, default=float)
    (root / "sealed").mkdir(exist_ok=True)
    path = root / "sealed" / f"{name}.json"
    path.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    (root / "results" / "seals").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (root / "results" / "seals" / f"{name}.sha256").write_text(f"{digest}  sealed/{name}.json  {stamp}\n")
    return digest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/all_runs.csv")
    ap.add_argument("--study", default="configs/study.yaml")
    ap.add_argument("--eval-set", default="heldout", choices=["heldout", "flores"])
    ap.add_argument("--within", default="pooled", choices=["pooled", "per_language"])
    ap.add_argument("--metric", default="bpb")
    ap.add_argument("--out", default=None, help="write the full result as JSON (not allowed with --seal)")
    ap.add_argument("--seal", default=None, metavar="NAME", help="scout mode: seal level contrasts under NAME")
    args = ap.parse_args()

    runs = runs_table.load(args.runs)
    if args.eval_set == "flores":
        runs = runs.assign(bpb=runs["bits_per_sentence"])  # within a language, a fixed rescaling
    result = run(runs, load_study(args.study), args.eval_set, args.within)
    if args.seal:
        if args.out:
            raise SystemExit("--out would write sealed numbers; use one or the other")
        public, sealed = split_sealed(result)
        digest = seal(sealed, args.seal, Path.cwd())
        print(json.dumps(public, indent=2, default=float))
        print(f"\nsealed {len(json.dumps(sealed))} bytes of level contrasts: sha256 {digest}")
        return
    text = json.dumps(result, indent=2, default=float)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
