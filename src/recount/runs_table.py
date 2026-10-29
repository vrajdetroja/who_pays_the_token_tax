"""results/all_runs.csv: one row per (trained endpoint, evaluation). The analysis reads
nothing else, and every figure is regenerated from it.

An endpoint is one trained model: the end of a run, or a cooldown branch of one.

    endpoint     what was trained                                   budget
    final        one pass over the language's full content          byte (r > 1), both (r = 1)
    deployed     cooldown branch at the reference token budget T    tok
    curve_<f>    level-1.0 cooldown branch at fraction f of content (data curve)

Each endpoint is evaluated under both context protocols (ctx = tok: a 512-token window;
ctx = byte: a 512*r-token window, the same bytes as the reference's). At r = 1 the two
protocols coincide and both rows carry the same number, so every level has all four cells.
`control` marks control evaluations ("" for the main ones): "shuffle" (positive control 2)
and "stride" (negative control 2). `tokenizer` is "main", or "swap" for negative control 1.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pandas as pd

COLUMNS = [
    # identity
    "lang", "model", "seed", "level", "tokenizer", "endpoint", "budget", "ctx", "eval_set", "control",
    # measurement
    "bpb", "nats", "n_bytes", "n_tokens", "n_sentences", "bits_per_sentence", "wasted_mass",
    "window", "stride",
    # what was trained
    "r_measured", "content_fraction", "train_tokens", "train_bytes", "steps", "train_flops",
    # provenance
    "run_id", "endpoint_id", "git_sha", "timestamp",
]
KEY = ["lang", "model", "seed", "level", "tokenizer", "endpoint", "ctx", "eval_set", "control"]
ENDPOINTS = ("final", "deployed")


def curve_endpoint(fraction: float) -> str:
    return f"curve_{fraction:g}"


def budget_of(endpoint: str, level: float) -> str:
    if endpoint == "final":
        return "both" if level == 1.0 else "byte"
    if endpoint == "deployed":
        return "tok"
    return "curve"


def append_rows(path: str | os.PathLike, rows: list[dict]) -> None:
    """Append rows, writing the header if the file is new. Unknown keys are an error."""
    path = Path(path)
    for r in rows:
        extra = set(r) - set(COLUMNS)
        if extra:
            raise ValueError(f"unknown all_runs columns: {sorted(extra)}")
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})


def load(path: str | os.PathLike) -> pd.DataFrame:
    """Read the table; a re-evaluated endpoint keeps only its latest row per KEY."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""],
                     dtype={"control": str, "tokenizer": str, "endpoint": str, "ctx": str})
    df["control"] = df["control"].fillna("")
    df["tokenizer"] = df["tokenizer"].fillna("main")
    if "timestamp" in df and df["timestamp"].notna().any():
        df = df.sort_values("timestamp", kind="stable")
    return df.drop_duplicates(KEY, keep="last").reset_index(drop=True)
