"""The four accounting cells at each dose level and their Shapley decomposition.

For one (language, model, seed) and a level r > 1, D(budget, context) is the penalty in
bits per byte against the same unit's level-1.0 reference:

    D(byte, byte)  full content, byte-matched context    the content-matched residual
    D(byte, tok)   full content, 512-token context
    D(tok,  byte)  reference token budget, byte context
    D(tok,  tok)   reference token budget, 512 tokens    the as-deployed tax

Averaging over the two orders of switching {budget, context} from byte to tok (Shapley)
splits the difference exactly:

    D(tok, tok) = D(byte, byte) + phi_budget + phi_context

and the headline share is S = (phi_budget + phi_context) / D(tok, tok) = 1 - D(byte, byte) / D(tok, tok).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import apply, fit_log_slope

UNIT = ["lang", "model", "seed"]


def _main(runs: pd.DataFrame, eval_set: str, tokenizer: str = "main") -> pd.DataFrame:
    return runs[(runs["eval_set"] == eval_set) & (runs["control"] == "") & (runs["tokenizer"] == tokenizer)]


def reference(runs: pd.DataFrame, eval_set: str = "heldout", metric: str = "bpb") -> pd.Series:
    """Level-1.0 full-content BPB per unit. Its two context protocols are the same window."""
    ref = _main(runs, eval_set)
    ref = ref[(ref["level"] == 1.0) & (ref["endpoint"] == "final")]
    tok = ref[ref["ctx"] == "tok"].set_index(UNIT)[metric]
    byte = ref[ref["ctx"] == "byte"].set_index(UNIT)[metric]
    if len(byte) and not np.allclose(tok.reindex(byte.index), byte, rtol=0, atol=1e-12):
        raise ValueError("level-1.0 reference differs between its two (identical) context protocols")
    if tok.index.duplicated().any():
        raise ValueError("duplicate reference rows")
    return tok.rename("ref")


CELLS = {("final", "byte"): "bb", ("final", "tok"): "bt", ("deployed", "byte"): "tb", ("deployed", "tok"): "tt"}


def cells(runs: pd.DataFrame, eval_set: str = "heldout", metric: str = "bpb",
          tokenizer: str = "main", require_all: bool = True) -> pd.DataFrame:
    """One row per (lang, model, seed, level > 1): the cells D_bb, D_bt, D_tb, D_tt and, when
    all four exist, the Shapley channels and the share S. With require_all, units missing a
    cell are dropped with a printed note; without it (the scout), whatever exists is kept."""
    ref = reference(runs, eval_set, metric)
    df = _main(runs, eval_set, tokenizer)
    df = df[(df["level"] > 1.0) & df["endpoint"].isin(["final", "deployed"])]
    wide = df.pivot_table(index=UNIT + ["level"], columns=["endpoint", "ctx"], values=metric, aggfunc="first")
    present = [c for c in CELLS if c in wide.columns]
    if require_all and len(present) < len(CELLS):
        raise ValueError(f"no rows at all for cells {[c for c in CELLS if c not in present]}")
    wide = wide[present]
    incomplete = wide.isna().any(axis=1)
    if require_all and incomplete.any():
        print(f"note: dropping {int(incomplete.sum())} incomplete units: {list(wide.index[incomplete])}")
        wide = wide[~incomplete]
    out = wide.copy()
    out.columns = [f"bpb_{CELLS[c]}" for c in present]
    out = out.reset_index()
    r = np.asarray(out.set_index(UNIT).index.map(ref), dtype=float)
    if np.isnan(r).any():
        if require_all:
            raise ValueError("a unit has level > 1 endpoints but no level-1.0 reference")
        out, r = out[~np.isnan(r)].copy(), r[~np.isnan(r)]  # e.g. the scout's swap-control units
    out["ref"] = r
    for c in present:
        out[f"D_{CELLS[c]}"] = out[f"bpb_{CELLS[c]}"] - out["ref"]
    # The realized dose (held-out r of this level's tokenizer), for predictions in ln r.
    rm = df.groupby(UNIT + ["level"])["r_measured"].first() if "r_measured" in df else None
    out["r_measured"] = out.set_index(UNIT + ["level"]).index.map(rm) if rm is not None else out["level"]
    out["r_measured"] = out["r_measured"].astype(float).fillna(out["level"])
    return add_shapley(out) if len(present) == len(CELLS) else out


def add_shapley(df: pd.DataFrame) -> pd.DataFrame:
    """phi_budget, phi_context, residual, deployed, share, and the exactness check."""
    bb, bt, tb, tt = df["D_bb"], df["D_bt"], df["D_tb"], df["D_tt"]
    df = df.copy()
    df["phi_budget"] = 0.5 * ((tb - bb) + (tt - bt))
    df["phi_context"] = 0.5 * ((bt - bb) + (tt - tb))
    df["residual"] = bb
    df["deployed"] = tt
    df["share"] = 1.0 - bb / tt
    gap = (df["residual"] + df["phi_budget"] + df["phi_context"] - df["deployed"]).abs().max()
    if gap > 1e-9:
        raise AssertionError(f"Shapley decomposition does not sum to D(tok, tok): max gap {gap:.3g}")
    return df


def data_curve(runs: pd.DataFrame, eval_set: str = "heldout", metric: str = "bpb") -> pd.DataFrame:
    """Per unit: BPB of the level-1.0 data-curve branches plus the full run, the slope b
    (BPB per e-fold of content), and whether BPB falls monotonically as content grows."""
    df = _main(runs, eval_set)
    df = df[(df["level"] == 1.0) & (df["ctx"] == "tok")]
    df = df[(df["endpoint"] == "final") | df["endpoint"].str.startswith("curve_")]
    df = df.assign(fraction=df["endpoint"].map(lambda e: 1.0 if e == "final" else float(e.removeprefix("curve_"))))

    def one(g: pd.DataFrame) -> dict:
        g = g.sort_values("fraction")
        fit = fit_log_slope(g["fraction"], g[metric])
        return fit | {"n_points": len(g), "monotone": bool(np.all(np.diff(g[metric].to_numpy()) < 0)),
                      "fractions": tuple(g["fraction"])}

    return apply(df, one, UNIT)


def identity_matched(runs: pd.DataFrame, eval_set: str = "heldout", metric: str = "bpb") -> pd.DataFrame:
    """Per (unit, level r): the BPB cost, at r = 1, of reading only the bytes the level-r
    as-deployed branch reads (bpb of the reference's identity_<r> branch minus the
    reference). If finer segmentation neither helped nor hurt per byte read, the budget
    channel would equal it: the matched form of the accounting identity (H3)."""
    ref = reference(runs, eval_set, metric)
    df = _main(runs, eval_set)
    df = df[(df["level"] == 1.0) & (df["ctx"] == "tok") & df["endpoint"].str.startswith("identity_")]
    if df.empty:
        return pd.DataFrame(columns=UNIT + ["level", "matched_prediction"])
    out = df.assign(level=df["endpoint"].str.removeprefix("identity_").astype(float))
    r = np.asarray(out.set_index(UNIT).index.map(ref), dtype=float)
    return out.assign(matched_prediction=out[metric].to_numpy() - r)[UNIT + ["level", "matched_prediction"]]


def swap_control(runs: pd.DataFrame, level: float, eval_set: str = "heldout", metric: str = "bpb") -> pd.DataFrame:
    """Negative control 1: per (lang, model, seed), BPB of the swapped tokenizer minus the
    main one at the same level, full content, byte context."""
    df = runs[(runs["eval_set"] == eval_set) & (runs["control"] == "") & (runs["level"] == level)
              & (runs["endpoint"] == "final") & (runs["ctx"] == "byte")]
    wide = df.pivot_table(index=UNIT, columns="tokenizer", values=metric, aggfunc="first").dropna()
    if "swap" not in wide:
        return pd.DataFrame(columns=UNIT + ["diff"])
    return wide.assign(diff=wide["swap"] - wide["main"]).reset_index()


def control_effect(runs: pd.DataFrame, control: str, eval_set: str = "heldout", metric: str = "bpb") -> pd.DataFrame:
    """Positive control 2 / negative control 2: BPB under `control` minus the matching main
    evaluation of the same endpoint and context protocol."""
    keys = UNIT + ["level", "tokenizer", "endpoint", "ctx"]
    base = runs[(runs["eval_set"] == eval_set) & (runs["control"] == "")].set_index(keys)[metric]
    ctl = runs[(runs["eval_set"] == eval_set) & (runs["control"] == control)].set_index(keys)[metric]
    both = pd.concat({"main": base, control: ctl}, axis=1, join="inner")
    return both.assign(diff=both[control] - both["main"]).reset_index()
