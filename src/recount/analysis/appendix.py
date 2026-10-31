"""The appendix tables, regenerated from a runs table and the analysis JSON.

    python -m recount.analysis.appendix --runs results/all_runs.csv --out results/appendix

Every table here is descriptive; the confirmatory numbers are the pooled ones in
results/analysis.json. Each table is written as Markdown (for the paper) and CSV.

    conditions     every endpoint's training: tokens, bytes read, steps, window and training
                   FLOPs, against the unit's r = 1 reference. The residual is content-matched,
                   not compute-matched: this table shows by how much.
    per_language   per (model, level, language): the four cells, both channels and the share,
                   mean over seeds with a t-interval (df = seeds - 1)
    controls       per (model, language): the data curve, the context shuffle, the stride
                   check, and the swap tokenizer (Russian only)
    seeds_needed   per (model, language): the seed SD of the r = 2.25 residual and the seeds a
                   per-language claim would need at this scale
    canonical_gap  per scored endpoint: how much canonical scoring overstates BPB, and how much
                   of that reaches the level contrast (recount.eval.canonical; once it has run)
    canonical_beam the same by beam search (amendment 4; recount.eval.canonical_beam), with the
                   wider beam on a subset and exact enumeration on sentence openings beside it
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .. import runs_table
from ..study import load_study
from . import decompose as D
from .stats import apply, t_interval

LANG_NAMES = {"eus_Latn": "Basque", "rus_Cyrl": "Russian", "arb_Arab": "Arabic", "cmn_Hani": "Chinese",
              "tam_Taml": "Tamil"}
TRAIN = ["train_tokens", "train_bytes", "steps", "train_flops"]


def _endpoints(runs: pd.DataFrame, eval_set: str = "heldout") -> pd.DataFrame:
    """One row per trained endpoint. Its byte-context row carries the training window,
    round(512 r), which every condition at level r trains with."""
    df = runs[(runs["eval_set"] == eval_set) & (runs["control"] == "") & (runs["ctx"] == "byte")]
    return df.drop_duplicates(D.UNIT + ["level", "tokenizer", "endpoint"])


def condition_name(level: float, tokenizer: str, endpoint: str) -> str:
    if endpoint == "final":
        if level == 1.0:
            return "reference: full tokenizer, one pass over the content"
        return "content-matched: one pass over the content" + (" (swap tokenizer)" if tokenizer == "swap" else "")
    if endpoint == "deployed":
        return "as deployed: the reference's token budget"
    if endpoint.startswith("identity_"):
        return f"identity branch: the bytes the r = {endpoint.removeprefix('identity_')} as-deployed branch reads"
    if endpoint.startswith("curve_"):
        return f"data curve: {float(endpoint.removeprefix('curve_')):.1%} of the content"
    return endpoint


def conditions(runs: pd.DataFrame, eval_set: str = "heldout") -> pd.DataFrame:
    """Training facts per condition, each unit against its own r = 1 reference."""
    u = _endpoints(runs, eval_set)
    ref = u[(u["level"] == 1.0) & (u["endpoint"] == "final") & (u["tokenizer"] == "main")].set_index(D.UNIT)[TRAIN]
    u = u.join(ref.add_suffix("_ref"), on=D.UNIT)
    u = u.assign(tokens_ratio=u["train_tokens"] / u["train_tokens_ref"],
                 bytes_fraction=u["train_bytes"] / u["train_bytes_ref"],
                 flops_ratio=u["train_flops"] / u["train_flops_ref"],
                 flops_per_byte_ratio=(u["train_flops"] / u["train_bytes"]) / (u["train_flops_ref"] / u["train_bytes_ref"]))
    g = u.groupby(["model", "level", "tokenizer", "endpoint"], sort=False)
    out = g.agg(units=("seed", "size"), window=("window", "first"), steps=("steps", "mean"),
                tokens_ratio=("tokens_ratio", "mean"), bytes_fraction=("bytes_fraction", "mean"),
                flops_ratio=("flops_ratio", "mean"), flops_ratio_min=("flops_ratio", "min"),
                flops_ratio_max=("flops_ratio", "max"), flops_per_byte_ratio=("flops_per_byte_ratio", "mean"),
                train_flops_mean=("train_flops", "mean")).reset_index()
    out["condition"] = [condition_name(lv, t, e) for lv, t, e in zip(out["level"], out["tokenizer"], out["endpoint"])]
    order = {"final": 0, "deployed": 1}
    out["_k"] = [(0 if lv == 1.0 and e == "final" else 1 if lv == 1.0 else 2, order.get(e, 3), e)
                 for lv, e in zip(out["level"], out["endpoint"])]
    return out.sort_values(["model", "_k", "level", "tokenizer"]).drop(columns="_k").reset_index(drop=True)


def per_language(runs: pd.DataFrame, eval_set: str = "heldout") -> pd.DataFrame:
    """Per (model, level, language): mean over seeds and t-interval of every cell and channel;
    the share is 1 - mean D(byte, byte) / mean D(tok, tok), the pooled S's definition."""
    c = D.cells(runs, eval_set)
    rows = []
    for (model, level, lang), g in c.groupby(["model", "level", "lang"]):
        row = {"model": model, "level": level, "lang": lang, "seeds": len(g)}
        for col in ("D_tt", "D_tb", "D_bt", "D_bb", "phi_budget", "phi_context"):
            m, lo, hi = t_interval(g[col])
            row |= {col: m, f"{col}_lo": lo, f"{col}_hi": hi}
        row["share"] = 1.0 - g["D_bb"].mean() / g["D_tt"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def controls(runs: pd.DataFrame, swap_level: float, eval_set: str = "heldout") -> pd.DataFrame:
    """Per (model, language): positive 1 (data curve), positive 2 (shuffle), negative 2
    (stride) and, where it exists, negative 1 (swap)."""
    curve = D.data_curve(runs, eval_set)
    sh = D.control_effect(runs, "shuffle", eval_set)
    sh = sh.groupby(D.UNIT, as_index=False)["diff"].mean()  # one rise per (language, seed), as pooled
    st = D.control_effect(runs, "stride", eval_set)
    sw = D.swap_control(runs, swap_level, eval_set)
    rows = []
    for (model, lang), g in curve.groupby(["model", "lang"]):
        row = {"model": model, "lang": lang, "seeds": len(g)}
        row |= dict(zip(("b", "b_lo", "b_hi"), t_interval(g["b"])))
        row |= {"curve_monotone": int(g["monotone"].sum()), "curve_min_r2": float(g["r2"].min())}
        s = sh[(sh["model"] == model) & (sh["lang"] == lang)]["diff"]
        row |= dict(zip(("shuffle_rise", "shuffle_lo", "shuffle_hi"), t_interval(s))) if len(s) > 1 else {}
        t = st[(st["model"] == model) & (st["lang"] == lang)]["diff"].abs()
        row |= {"stride_max_abs": float(t.max()) if len(t) else float("nan"), "stride_over_limit": int((t > 0.001).sum()),
                "stride_evaluations": len(t)}
        w = sw[(sw["model"] == model) & (sw["lang"] == lang)]["diff"] if len(sw) else pd.Series(dtype=float)
        if len(w) > 1:
            row |= dict(zip(("swap_diff", "swap_lo", "swap_hi"), t_interval(w)))
        rows.append(row)
    return pd.DataFrame(rows)


def seeds_for_detection(sd: float, delta: float, alpha: float = 0.05, power: float = 0.8, n_max: int = 1000) -> int:
    """Fewest seeds for a two-sided one-sample t-test to detect a true effect of `delta`
    with the given power (noncentral t)."""
    for n in range(2, n_max + 1):
        df, nc = n - 1, delta * math.sqrt(n) / sd
        crit = stats.t.ppf(1 - alpha / 2, df)
        upper, lower = stats.nct.sf(crit, df, nc), stats.nct.cdf(-crit, df, nc)  # NaN once a tail underflows
        if (float(nc > crit) if math.isnan(upper) else upper) + (0.0 if math.isnan(lower) else lower) >= power:
            return n
    return n_max


def seeds_for_equivalence(sd: float, delta: float, alpha: float = 0.05, power: float = 0.8, n_max: int = 1000) -> int:
    """Fewest seeds for two one-sided tests against +-delta to pass with the given power when
    the true effect is 0 (normal approximation for the sampling of the mean)."""
    z = stats.norm.ppf(1 - (1 - power) / 2)
    for n in range(2, n_max + 1):
        if delta * math.sqrt(n) / sd >= stats.t.ppf(1 - alpha, n - 1) + z:
            return n
    return n_max


def seeds_needed(runs: pd.DataFrame, sesoi_by_model: dict[str, float], level: float = 2.25,
                 col: str = "D_bb", eval_set: str = "heldout") -> pd.DataFrame:
    """Per (model, language) at `level`: the seed SD of `col` and the seeds a per-language
    claim about it would need, with the model's SESOI as the effect of interest."""
    c = D.cells(runs, eval_set)
    c = c[c["level"] == level]
    rows = []
    for (model, lang), g in c.groupby(["model", "lang"]):
        sd, delta = float(g[col].std(ddof=1)), sesoi_by_model[model]
        rows.append({"model": model, "lang": lang, "seeds_run": len(g), "sd": sd, "sesoi": delta,
                     "seeds_to_detect_sesoi": seeds_for_detection(sd, delta),
                     "seeds_to_show_negligible": seeds_for_equivalence(sd, delta)})
    return pd.DataFrame(rows)


def canonical_gap(path: Path) -> pd.DataFrame:
    """Per scored endpoint: the canonical-tokenization gap in bits per byte (canonical minus
    marginal nats, summed over sentences, over their bytes), the same from half the draws, the
    share of draws that were the canonical sequence, and the gap minus the same unit's r = 1
    reference gap: the part that reaches the level contrast D."""
    g = pd.read_csv(path)
    g = g[~g["untrained"].astype(str).str.lower().eq("true")]  # dry runs score fresh weights
    if g.empty:
        raise ValueError(f"{path}: no rows from trained models")
    ln2 = math.log(2)

    def one(d: pd.DataFrame) -> dict:
        b = d["n_bytes"].sum()
        return {"sentences": len(d), "canonical_bpb": d["canonical_nats"].sum() / b / ln2,
                "gap_bpb": (d["canonical_nats"] - d["marginal_nats"]).sum() / b / ln2,
                "gap_bpb_half_draws": (d["canonical_nats"] - d["marginal_nats_half"]).sum() / b / ln2,
                "draws_canonical": d["draws_canonical"].mean()}

    t = apply(g, one, ["model", "seed", "lang", "level", "endpoint"])
    unit = ["model", "seed", "lang"]
    ref = t[(t["level"] == 1.0) & (t["endpoint"] == "final")].set_index(unit)["gap_bpb"]
    return t.assign(gap_minus_reference=t["gap_bpb"] - np.asarray(t.set_index(unit).index.map(ref), dtype=float))


UNCONVERGED = (0.25, 0.00001)  # amendment 4: B = 64 raising the subset's gap by more than both


def canonical_beam(beam_path: Path, openings_path: Path | None = None, beam: int = 16, check: int = 64) -> pd.DataFrame:
    """Per scored endpoint, amendment 4's beam: the gap at `beam` over all its sentences and the
    part that reaches D; on the check subset, the gap at `beam` and at `check` and whether
    `beam` had converged there; and on the openings, the exact gap, the beam's gap, and the
    share of the exact non-canonical probability the beam recovered (mean and lowest)."""
    g = pd.read_csv(beam_path)
    g = g[~g["untrained"].astype(str).str.lower().eq("true")]  # dry runs score fresh weights
    if g.empty:
        raise ValueError(f"{beam_path}: no rows from trained models")
    ln2, key, unit = math.log(2), ["model", "seed", "lang", "level", "endpoint"], ["model", "seed", "lang"]

    def gap(d: pd.DataFrame, col: str = "marginal_nats") -> float:
        return (d["canonical_nats"] - d[col]).sum() / d["n_bytes"].sum() / ln2

    t = apply(g[g["beam"] == beam], lambda d: {"sentences": len(d), "gap_bpb": gap(d)}, key)
    ref = t[(t["level"] == 1.0) & (t["endpoint"] == "final")].set_index(unit)["gap_bpb"]
    t["reaches_d"] = t["gap_bpb"] - np.asarray(t.set_index(unit).index.map(ref), dtype=float)
    wide, narrow = g[g["beam"] == check], g[g["beam"] == beam]
    if not wide.empty:
        both = narrow.merge(wide[key + ["sentence"]], on=key + ["sentence"])
        sub = apply(both, lambda d: {"check_sentences": len(d), "gap_check_narrow": gap(d)}, key).merge(
            apply(wide, lambda d: {"gap_check_wide": gap(d)}, key), on=key)
        rise = sub["gap_check_wide"] - sub["gap_check_narrow"]
        sub["converged"] = ~((rise > UNCONVERGED[0] * sub["gap_check_narrow"]) & (rise > UNCONVERGED[1]))
        t = t.merge(sub, on=key, how="left")
    if openings_path is not None and Path(openings_path).exists():
        o = pd.read_csv(openings_path)
        o = o[~o["untrained"].astype(str).str.lower().eq("true") & (o["n_bytes"] > 0)]

        def opened(d: pd.DataFrame) -> dict:
            exact, got = np.expm1(d["canonical_nats"] - d["exact_nats"]), np.expm1(d["canonical_nats"] - d["beam_nats"])
            share = (got / exact)[exact > 0]  # the non-canonical probability, relative to the canonical's
            return {"openings": len(d), "opening_bytes": d["n_bytes"].mean(), "opening_gap_exact": gap(d, "exact_nats"),
                    "opening_gap_beam": gap(d, "beam_nats"), "recovered_mean": share.mean(), "recovered_min": share.min()}

        t = t.merge(apply(o, opened, key), on=key, how="left")
    return t


# ---------------------------------------------------------------------------------------
# Markdown


def _ci(m: float, lo: float, hi: float, nd: int = 3) -> str:
    if any(map(math.isnan, (m, lo, hi))):
        return f"{m:+.{nd}f}" if not math.isnan(m) else ""
    return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def conditions_md(t: pd.DataFrame) -> str:
    parts = ["# Training conditions\n",
             "Means over the units (language × seed) of each condition. Ratios are against each unit's own "
             "r = 1 reference, which reads the whole content once. Training FLOPs per token follow microlm: "
             "6 × non-embedding parameters, plus the unembedding (6 d V) and attention (12 L T d, growing with "
             "the window T), times the tokens trained. Every condition at level r trains with windows of "
             "round(512 r) tokens, so each step reads the same bytes.\n"]
    for model, g in t.groupby("model", sort=False):
        parts.append(f"\n## Model {model}\n")
        rows = [[f"{lv:g}", c, str(int(n)), str(int(w)), f"{s:,.0f}", f"{tr:.2f}", f"{bf:.0%}",
                 f"{fr:.2f} ({lo:.2f}–{hi:.2f})", f"{fb:.2f}"]
                for lv, c, n, w, s, tr, bf, fr, lo, hi, fb in zip(
                    g["level"], g["condition"], g["units"], g["window"], g["steps"], g["tokens_ratio"],
                    g["bytes_fraction"], g["flops_ratio"], g["flops_ratio_min"], g["flops_ratio_max"],
                    g["flops_per_byte_ratio"])]
        parts.append(_table(["r", "condition", "units", "window", "steps", "tokens ×ref", "bytes read",
                             "FLOPs ×ref (range)", "FLOPs per byte ×ref"], rows))
    return "\n".join(parts) + "\n"


def per_language_md(t: pd.DataFrame) -> str:
    parts = ["# Per-language results\n",
             "Held-out FineWeb-2, bits per byte against each language's r = 1 reference. Mean over seeds with a "
             "95% t-interval (df = seeds − 1). Descriptive only: the confirmatory results are pooled over "
             "languages. The share is 1 − mean D(byte, byte) / mean D(tok, tok).\n"]
    for (model, level), g in t.groupby(["model", "level"]):
        parts.append(f"\n## Model {model}, r = {level:g} ({int(g['seeds'].iloc[0])} seeds)\n")
        rows = [[LANG_NAMES.get(r.lang, r.lang)] + [_ci(getattr(r, c), getattr(r, c + "_lo"), getattr(r, c + "_hi"))
                                                    for c in ("D_tt", "D_tb", "D_bt", "D_bb", "phi_budget", "phi_context")]
                + [f"{r.share:.2f}"] for r in g.itertuples()]
        parts.append(_table(["language", "D(tok, tok)", "D(tok, byte)", "D(byte, tok)", "D(byte, byte)",
                             "budget channel", "context channel", "share"], rows))
    return "\n".join(parts) + "\n"


def controls_md(t: pd.DataFrame, sesoi_by_model: dict[str, float]) -> str:
    parts = ["# Controls per language\n",
             "Held-out FineWeb-2, bits per byte. Positive 1: the data-curve slope b (BPB per e-fold of content; "
             "12.5/25/50/100%), mean over seeds with a t-interval, how many seeds' curves fall monotonically, and "
             "the lowest R². Positive 2: the rise in BPB when each window's unscored left context is replaced by "
             "distant text (mean over a seed's evaluations, then over seeds). Negative 2: stride ¼ window against "
             "the main ½, the largest |difference| and how many evaluations exceed the registered 0.001. "
             "Negative 1: the Russian swap tokenizer at r = 1.5 minus the main one.\n"]
    for model, g in t.groupby("model"):
        parts.append(f"\n## Model {model} (SESOI {sesoi_by_model[model]:.4f})\n")
        rows = []
        for r in g.itertuples():
            swap = _ci(r.swap_diff, r.swap_lo, r.swap_hi, 4) if "swap_diff" in g and not math.isnan(r.swap_diff) else "—"
            rows.append([LANG_NAMES.get(r.lang, r.lang), _ci(r.b, r.b_lo, r.b_hi), f"{r.curve_monotone}/{r.seeds}",
                         f"{r.curve_min_r2:.3f}", _ci(r.shuffle_rise, r.shuffle_lo, r.shuffle_hi, 4),
                         f"{r.stride_max_abs:.4f} ({r.stride_over_limit}/{r.stride_evaluations})", swap])
        parts.append(_table(["language", "data-curve b", "monotone", "min R²", "shuffle rise",
                             "stride max |Δ| (over 0.001)", "swap − main"], rows))
    return "\n".join(parts) + "\n"


def seeds_needed_md(t: pd.DataFrame, level: float) -> str:
    rows = [[r.model, LANG_NAMES.get(r.lang, r.lang), str(r.seeds_run), f"{r.sd:.4f}", f"{r.sesoi:.4f}",
             str(r.seeds_to_detect_sesoi), str(r.seeds_to_show_negligible)] for r in t.itertuples()]
    return "\n".join([
        "# Seeds needed per language\n",
        f"For the content-matched residual D(byte, byte) at r = {level:g}: its seed SD within each language "
        "(from the seeds run, so itself rough: 4 degrees of freedom for M, 2 for S), and the seeds a "
        "per-language claim would need at that SD, with the model's SESOI (the cost of losing 10% of the "
        "training text) as the effect of interest. *Detect*: a two-sided t-test at α = 0.05 finds a true "
        "residual of one SESOI with power 0.8. *Show negligible*: two one-sided tests against ±SESOI pass "
        "with power 0.8 when the true residual is 0. The pooled analysis needs neither: it pools the five "
        "languages, which is why per-language results are descriptive.\n",
        _table(["model", "language", "seeds run", "seed SD", "SESOI", "seeds to detect", "seeds to show negligible"], rows),
    ]) + "\n"


def canonical_gap_md(t: pd.DataFrame) -> str:
    parts = ["# The canonical-tokenization gap\n",
             "FLORES-200 devtest sentences, each scored alone after BOS. *Gap*: bits per byte by which scoring "
             "only the canonical token sequence overstates the text's BPB, estimated by importance sampling over "
             "the token sequences that spell it (Chirkova et al., 2023; recount.eval.canonical); half-draw beside "
             "it as a convergence check. With 32 draws under the trained models this is a lower bound far below "
             "the gap: a draw almost never follows the canonical sequence, and the draws that leave it come back "
             "with negligible weight (PREREGISTRATION.md, amendment 4). `canonical_beam.md` has amendment 4's beam. "
             "*Reaches D*: the gap minus the same unit's r = 1 reference gap, the part that would inflate the level "
             "contrast. Mean row: over languages.\n"]
    for (model, seed), g in t.groupby(["model", "seed"]):
        parts.append(f"\n## Model {model}, seed {seed} ({int(g['sentences'].iloc[0])} sentences per endpoint)\n")
        g = g.sort_values(["level", "endpoint", "lang"])
        rows = []
        for (level, endpoint), h in g.groupby(["level", "endpoint"], sort=False):
            name = "reference" if level == 1.0 else ("content-matched" if endpoint == "final" else "as deployed")
            for r in h.itertuples():
                rows.append([f"{level:g}", name, LANG_NAMES.get(r.lang, r.lang), f"{r.canonical_bpb:.4f}",
                             f"{r.gap_bpb:.5f}", f"{r.gap_bpb_half_draws:.5f}", f"{r.draws_canonical:.0%}",
                             "" if level == 1.0 else f"{r.gap_minus_reference:+.5f}"])
            rows.append([f"{level:g}", name, "**mean**", f"{h['canonical_bpb'].mean():.4f}", f"{h['gap_bpb'].mean():.5f}",
                         f"{h['gap_bpb_half_draws'].mean():.5f}", f"{h['draws_canonical'].mean():.0%}",
                         "" if level == 1.0 else f"{h['gap_minus_reference'].mean():+.5f}"])
        parts.append(_table(["r", "endpoint", "language", "canonical BPB", "gap", "gap, half the draws",
                             "draws canonical", "reaches D"], rows))
    return "\n".join(parts) + "\n"


def canonical_beam_md(t: pd.DataFrame, beam: int = 16, check: int = 64) -> str:
    parts = ["# The canonical-tokenization gap, by beam search (amendment 4)\n",
             "The sentences of canonical_gap.md, each scored alone after BOS. *Gap*: bits per byte by which scoring "
             "only the canonical token sequence overstates the text's BPB, from a beam over the token sequences that "
             "spell it (recount.eval.canonical_beam): a lower bound, exact once the beam keeps every spelling. "
             "*Reaches D*: the gap minus the same unit's r = 1 reference gap. *Subset*: the first sentences at both "
             f"widths; B = {beam} is called unconverged where B = {check} raises the subset's gap by more than "
             f"{UNCONVERGED[0]:.0%} and by more than {UNCONVERGED[1]:g} bits per byte. *Openings*: each subset "
             "sentence's longest opening whose canonical tokens at every level are the sentence's own and that has at "
             "most 5,000 spellings at r = 1, every spelling scored; *recovered*: the share of the exact non-canonical "
             "probability the beam found. Mean row: over languages.\n"]
    def name(level: float, endpoint: str) -> str:
        return "reference" if level == 1.0 else ("content-matched" if endpoint == "final" else "as deployed")

    has_check, has_open = "gap_check_wide" in t, "openings" in t
    for (model, seed), g in t.groupby(["model", "seed"]):
        g = g.sort_values(["level", "endpoint", "lang"])
        parts.append(f"\n## Model {model}, seed {seed} ({int(g['sentences'].iloc[0])} sentences per endpoint)\n")
        rows, opened = [], []
        for (level, endpoint), h in g.groupby(["level", "endpoint"], sort=False):
            mean = h.drop(columns="converged", errors="ignore").mean(numeric_only=True).to_dict()
            labelled = [(LANG_NAMES.get(x["lang"], x["lang"]), x) for x in h.to_dict("records")]
            for label, r in labelled + [("**mean**", mean)]:
                row = [f"{level:g}", name(level, endpoint), label, f"{r['gap_bpb']:.6f}",
                       "" if level == 1.0 else f"{r['reaches_d']:+.6f}"]
                if has_check:
                    conv = "" if pd.isna(r.get("converged")) else ("yes" if r["converged"] else "no")
                    row += [f"{r['gap_check_narrow']:.6f}", f"{r['gap_check_wide']:.6f}", conv]
                rows.append(row)
                if has_open:
                    opened.append([f"{level:g}", name(level, endpoint), label, f"{r['openings']:.0f}",
                                   f"{r['opening_bytes']:.1f}", f"{r['opening_gap_exact']:.6f}",
                                   f"{r['opening_gap_beam']:.6f}", f"{r['recovered_mean']:.1%}",
                                   f"{r['recovered_min']:.1%}"])
        header = ["r", "endpoint", "language", f"gap, B = {beam}", "reaches D"]
        header += [f"subset, B = {beam}", f"subset, B = {check}", "converged"] if has_check else []
        parts.append(_table(header, rows))
        if has_open:
            parts.append(f"\n### Openings: exact against the beam (B = {beam})\n")
            parts.append(_table(["r", "endpoint", "language", "openings", "bytes", "exact gap", "beam gap",
                                 "recovered, mean", "recovered, lowest"], opened))
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/all_runs.csv")
    ap.add_argument("--analysis", default="results/analysis.json", help="for each model's SESOI")
    ap.add_argument("--study", default="configs/study.yaml")
    ap.add_argument("--canonical", default="results/canonical_gap.csv", help="used if it exists")
    ap.add_argument("--canonical-beam", default="results/canonical_beam.csv", help="used if it exists")
    ap.add_argument("--canonical-openings", default="results/canonical_openings.csv")
    ap.add_argument("--out", default="results/appendix")
    args = ap.parse_args()
    runs = runs_table.load(args.runs)
    study = load_study(args.study)
    analysis = json.loads(Path(args.analysis).read_text())
    sesoi = {m: v["data_curve"]["sesoi"] for m, v in analysis["models"].items() if "data_curve" in v}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    level = max(study.levels)
    tables = {
        "conditions": (conditions(runs), conditions_md),
        "per_language": (per_language(runs), per_language_md),
        "controls": (controls(runs, study.tokenizer["swap_level"]), lambda t: controls_md(t, sesoi)),
        "seeds_needed": (seeds_needed(runs, sesoi, level), lambda t: seeds_needed_md(t, level)),
    }
    if Path(args.canonical).exists():
        tables["canonical_gap"] = (canonical_gap(Path(args.canonical)), canonical_gap_md)
    if Path(args.canonical_beam).exists():
        tables["canonical_beam"] = (canonical_beam(Path(args.canonical_beam), Path(args.canonical_openings)),
                                    canonical_beam_md)
    for name, (table, md) in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
        (out / f"{name}.md").write_text(md(table))
        print(f"wrote {out / name}.md/.csv ({len(table)} rows)")


if __name__ == "__main__":
    main()
