"""Byte premiums (Arnett, Chang & Bergen, 2024): bytes needed to encode the same content in a
language, relative to English. Content budgets are scaled by them so every language's
model reads the same amount of information.

Looked up exactly as the authors' tool does (`get_pairwise_premium`, use case 1): the
precomputed `byte_coef` column of all_merged_20240223.tsv, whose source is NLLB, FLORES, or
the Bible in that order of preference. One override: FineWeb-2's Mandarin (cmn_Hani) maps
to the tool's zho_hans row (NLLB) rather than cmn_hans (Bible only), following the tool's
own source preference.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TSV = Path("data/raw/byte_premium/all_merged_20240223.tsv")
ALIASES = {"cmn_hani": "zho_hans", "cmn_hans": "zho_hans"}


def table(path: Path = TSV) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return df[df["byte_coef"].notna()].set_index("lang")


def premium(lang: str, path: Path = TSV, reference: str = "eng_latn") -> dict:
    df = table(path)
    key = ALIASES.get(lang.lower(), lang.lower())
    if key not in df.index:
        raise KeyError(f"{lang}: not in the byte premium table")
    row, ref = df.loc[key], df.loc[reference]
    return {"lang": lang, "row": key, "premium": float(row["byte_coef"] / ref["byte_coef"]), "source": row["source"]}
