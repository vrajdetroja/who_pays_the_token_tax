"""The scout report: everything the pilot decision needs, nothing sealed.

    python -m recount.analysis.scout --runs results/scout_runs.csv
    python -m recount.analysis.scout --runs results/scout2_runs.csv --name scout2   # amendment 1

The first scout's untagged runs used the learning rate before amendment 1: reproduce its
report with `--default-lr 2e-3`.

Printed: the learning-rate check (each LR's BPB
*relative to the best LR at the same level*, never across levels), seed noise of the paired
level contrast (its SD, never its mean) and the between-language spread, the four controls,
and dose accuracy. Every level contrast's mean goes to sealed/scout.json via `analyze --seal`
logic; only its SHA-256 is written to results/seals/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import runs_table
from ..study import load_study
from . import analyze as A
from . import decompose as D


def lr_check(runs: pd.DataFrame, default_lr: float) -> pd.DataFrame:
    """BPB minus the best LR's BPB, within each (language, level). The default-LR point is
    the seed-0 variance run, whose tokenizer column carries no '+lr=' tag."""
    df = runs[(runs["eval_set"] == "heldout") & (runs["control"] == "") & (runs["endpoint"] == "final")
              & (runs["ctx"] == "byte") & (runs["seed"] == 0) & runs["tokenizer"].str.startswith("main")]
    lr = df["tokenizer"].str.extract(r"\+lr=([0-9.e-]+)")[0].astype(float)
    df = df.assign(lr=lr)
    tagged_langs = set(df.loc[df["lr"].notna(), "lang"])
    df = df[df["lang"].isin(tagged_langs)].copy()
    rows = []
    for (lang, level), g in df.groupby(["lang", "level"]):
        best = g["bpb"].min()
        for _, r in g.iterrows():
            rows.append({"lang": lang, "level": level, "lr": r["lr"] if pd.notna(r["lr"]) else default_lr,
                         "bpb_minus_best": r["bpb"] - best})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/scout_runs.csv")
    ap.add_argument("--study", default="configs/study.yaml")
    ap.add_argument("--root", default=".")
    ap.add_argument("--name", default="scout", help="seal name: sealed/<name>.json, results/seals/<name>.sha256")
    ap.add_argument("--default-lr", type=float, default=None,
                    help="learning rate of the runs without an '+lr=' tag (default: the study's)")
    args = ap.parse_args()
    study = load_study(args.study)
    root = Path(args.root)
    runs = runs_table.load(args.runs)
    print("\n== learning-rate check (BPB minus the best LR at the same level; across-level numbers withheld)")
    lrc = lr_check(runs, args.default_lr or study.training["lr"])
    print(lrc.to_string(index=False, float_format=lambda x: f"{x:.4f}") if len(lrc) else "  no LR-check runs")

    print("\n== seed noise and controls (level-contrast means are sealed)")
    design = runs[~runs["tokenizer"].str.contains(r"\+")]
    cells = D.cells(design, require_all=False)
    curve = D.data_curve(design)
    sealed: dict = {"contrast_means": {}}
    bounds = {}
    for model, cm in cells.groupby("model"):
        cv = curve[curve["model"] == model]
        b = A.pool(cv, "b") if cv["lang"].nunique() > 1 else None
        b_est = b.estimate if b is not None else float(cv["b"].mean())
        bounds[model] = A.sesoi(b_est, study.analysis["sesoi_text_fraction"])
        print(f"  model {model}: data-curve slope b = {b_est:.4f} BPB per e-fold"
              + (f" [{b.lo:.4f}, {b.hi:.4f}]" if b is not None else "")
              + f"; monotone in every unit: {bool(cv['monotone'].all())}; SESOI = {bounds[model]:.4f}")
        for level, cl in cm.groupby("level"):
            if (cl.groupby("lang")["seed"].nunique() < 2).any():
                continue  # e.g. r = 1.5, which the scout trains only for the swap control
            for col in [x for x in ("D_bb", "D_bt") if x in cl]:
                n = A.noise(cl, col)
                print(f"    r = {level:g} {col}: sigma_delta = {n['sigma_delta']:.4f} BPB, tau = {n['tau']:.4f} "
                      f"({n['languages']} languages, seeds {n['seeds_per_language']})")
                sealed["contrast_means"][f"{model}/{level:g}/{col}"] = cl.groupby("lang")[col].mean().to_dict()
    for k, v in A.controls(design, study, "heldout", bounds).items():
        if "estimate" in v:  # pooled shuffle effect: within an endpoint, safe to show
            v = {kk: v[kk] for kk in ("estimate", "lo", "hi", "informative")}
        print(f"  {k}: " + json.dumps({a: (round(b, 5) if isinstance(b, float) else b) for a, b in v.items()}))
    digest = A.seal(sealed, args.name, root)
    print(f"  sealed level contrasts: sha256 {digest} (results/seals/{args.name}.sha256)")

    print("\n== dose accuracy (held-out r within ±3% of target)")
    for rep in sorted((root / "data" / "tokenizers").glob("*/dose_report.json")):
        d = json.loads(rep.read_text())
        cells = [f"r={k}: {v['r_verify']:.3f} (k={v['active']})" for k, v in d["levels"].items()]
        if "swap" in d:
            cells.append(f"swap: {d['swap']['r_verify']:.3f} (shares {d['swap']['shared_fraction']:.0%} of merges)")
        print(f"  {d['lang']}: " + ", ".join(cells))
    print("\n== decide: instrument works -> run the design; sigma_delta > 0.005 -> "
          "add seeds for M; NC1 fails -> redesign the dose; shuffle uninformative -> drop the "
          "context channel from the claims; divergence at 2.25 -> lower its LR. Amend PREREGISTRATION.md before the "
          "main grid, saying what you had seen.")
    np.seterr(all="ignore")


if __name__ == "__main__":
    main()
