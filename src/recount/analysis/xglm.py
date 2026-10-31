"""The XGLM check's regression (PREREGISTRATION.md, amendment 3; exploratory, reported apart from
the confirmatory analysis).

Per language, from `recount.eval.xglm`'s sentence scores: bits per FLORES-200 devtest sentence;
x_data = ln(T · b / p), the English-equivalent text seen in training (T = 500B × the card's
upsampled share, b = bytes per XGLM token on devtest, p = the tool's byte premium); and
x_fert = ln(the token premium against English). OLS of bits per sentence on [1, x_data, x_fert],
one fit per model, with HC3 errors (the sum of the squared leave-one-out shifts of the fit) and
t intervals on n − k degrees of freedom. β_data < 0 is the positive control: unless its interval
lies below 0, β_fert's sign is called inconclusive.

    python -m recount.analysis.xglm --sentences results/xglm/sentences.csv --out results/xglm
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..data import premiums
from ..eval.xglm import LANGS, REVISIONS, TRAIN_TOKENS

UNK_LIMIT = 0.01  # above this share of <unk> tokens a language leaves the sensitivity fit
PREDICTED = 10.0  # bits per sentence per e-fold of premium at model M's scale (amendment 3)


def languages(sentences: pd.DataFrame, tool_premiums: dict[str, float] | None = None) -> pd.DataFrame:
    g = (sentences.groupby(["model", "lang"])
         .agg(n=("sentence", "size"), bytes=("n_bytes", "sum"), tokens=("n_tokens", "sum"), unk=("n_unk", "sum"),
              nats=("nats", "sum"))
         .reset_index())
    if (g["n"] != 1012).any():
        raise ValueError(f"incomplete languages: {g.loc[g['n'] != 1012, ['model', 'lang', 'n']].to_dict('records')}")
    en = g[g["lang"] == "en"].set_index("model")
    if tool_premiums is None:
        tool_premiums = {iso: premiums.premium(LANGS[iso][0])["premium"] for iso in g["lang"].unique()}
    g["name"] = g["lang"].map(lambda l: LANGS[l][2])
    g["flores"] = g["lang"].map(lambda l: LANGS[l][0])
    g["share"] = g["lang"].map(lambda l: LANGS[l][1])
    g["bits_per_sentence"] = g["nats"] / g["n"] / math.log(2)
    g["b"] = g["bytes"] / g["tokens"]
    g["token_premium"] = g["tokens"] / g["model"].map(en["tokens"])
    g["unk_share"] = g["unk"] / g["tokens"]
    g["p_tool"] = g["lang"].map(tool_premiums)
    g["p_flores"] = g["bytes"] / g["model"].map(en["bytes"])
    g["x_data"] = np.log(TRAIN_TOKENS * g["share"] * g["b"] / g["p_tool"])
    g["x_data_flores"] = np.log(TRAIN_TOKENS * g["share"] * g["b"] / g["p_flores"])
    g["x_fert"] = np.log(g["token_premium"])
    return g


def ols_hc3(y, X, names: list[str], level: float = 0.95) -> dict:
    y, X = np.asarray(y, float), np.asarray(X, float)
    n, k = X.shape
    inv = np.linalg.inv(X.T @ X)
    beta = inv @ X.T @ y
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, inv, X)
    u = X * (e / (1 - h))[:, None]
    se = np.sqrt(np.diag(inv @ u.T @ u @ inv))
    q = stats.t.ppf(0.5 + level / 2, n - k)
    r2 = 1 - e @ e / np.sum((y - y.mean()) ** 2)
    return {"n": n, "df": n - k, "r2": float(r2), "coef": {
        name: {"estimate": float(b), "se": float(s), "lo": float(b - q * s), "hi": float(b + q * s)}
        for name, b, s in zip(names, beta, se)}}


def verdict(fit: dict) -> str:
    data, fert = fit["coef"]["x_data"], fit["coef"]["x_fert"]
    if not data["hi"] < 0:
        return "inconclusive: the positive control failed"
    return "agrees" if fert["lo"] > 0 else "disagrees" if fert["hi"] < 0 else "inconclusive"


def fits(table: pd.DataFrame) -> dict:
    out = {}
    for model in [m for m in REVISIONS if m in set(table["model"])]:  # smallest first
        g = table[table["model"] == model]
        two = lambda d, x: ols_hc3(d["bits_per_sentence"], np.column_stack([np.ones(len(d)), d[x], d["x_fert"]]),  # noqa: E731
                                   ["const", "x_data", "x_fert"])
        primary, flores_p = two(g, "x_data"), two(g, "x_data_flores")
        heavy = g.loc[g["unk_share"] > UNK_LIMIT, "lang"].tolist()
        out[model] = {
            "primary": primary | {"verdict": verdict(primary)},
            "fert_only": ols_hc3(g["bits_per_sentence"], np.column_stack([np.ones(len(g)), g["x_fert"]]),
                                 ["const", "x_fert"]),
            "flores_premiums": flores_p | {"verdict": verdict(flores_p)},
            "without_unk_heavy": None if not heavy else two(g[~g["lang"].isin(heavy)], "x_data") | {"dropped": heavy},
            "correlation_x_data_x_fert": float(np.corrcoef(g["x_data"], g["x_fert"])[0, 1]),
        }
        if heavy:
            out[model]["without_unk_heavy"]["verdict"] = verdict(out[model]["without_unk_heavy"])
    return out


def ci(c: dict, digits: int = 1) -> str:
    return f"{c['estimate']:+.{digits}f} [{c['lo']:+.{digits}f}, {c['hi']:+.{digits}f}]"


def to_markdown(table: pd.DataFrame, result: dict) -> str:
    models = list(result)
    wide = table.pivot(index="lang", columns="model", values="bits_per_sentence")
    one = table.drop_duplicates("lang").set_index("lang")  # the regressors are the same for both models
    one = one.loc[[l for l in LANGS if l in one.index]]
    lines = ["# The XGLM check (amendment 3; exploratory)", "",
             "Bits per FLORES-200 devtest sentence (1,012 sentences, each scored alone after `</s>`), the "
             "regressors, and OLS fits with HC3 errors and t intervals. The prediction: β_fert > 0, about "
             f"{PREDICTED:.0f} bits per sentence per e-fold of premium at model M's scale and less at XGLM's.", "",
             "## Fits", "",
             "| model | fit | n | β_fert (bits/sentence per e-fold) | β_data | R² | sign |", "|---|---|---|---|---|---|---|"]
    labels = {"primary": "primary", "flores_premiums": "FLORES-direct premiums",
              "without_unk_heavy": f"without `<unk>` > {UNK_LIMIT:.0%}", "fert_only": "x_fert alone"}
    for m in models:
        for key, label in labels.items():
            f = result[m][key]
            if f is None:
                lines.append(f"| {m} | {label} | – | no language above the limit | | | |")
                continue
            data = ci(f["coef"]["x_data"]) if "x_data" in f["coef"] else "–"
            lines.append(f"| {m} | {label} | {f['n']} | {ci(f['coef']['x_fert'])} | {data} | {f['r2']:.2f} | "
                         f"{f.get('verdict', '–')} |")
    lines += ["", "Correlation of x_data and x_fert across the languages: "
              f"{result[models[0]]['correlation_x_data_x_fert']:+.2f}.", "", "## Languages", "",
              "| language | FLORES | upsampled share | bytes/token | byte premium (tool, FLORES) | token premium | "
              "`<unk>` | x_data | x_fert | " + " | ".join(f"bits/sentence {m}" for m in models) + " |",
              "|---|---|---|---|---|---|---|---|---|" + "---|" * len(models)]
    for iso, r in one.iterrows():
        lines.append(f"| {r['name']} | {r['flores']} | {r['share']:.4f} | {r['b']:.2f} | {r['p_tool']:.2f}, "
                     f"{r['p_flores']:.2f} | {r['token_premium']:.2f} | {r['unk_share']:.2%} | {r['x_data']:.2f} | "
                     f"{r['x_fert']:+.2f} | " + " | ".join(f"{wide.loc[iso, m]:.1f}" for m in models) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sentences", type=Path, default=Path("results/xglm/sentences.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/xglm"))
    args = ap.parse_args()
    table = languages(pd.read_csv(args.sentences))
    result = fits(table)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "languages.csv", index=False)
    (args.out / "fit.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "xglm.md").write_text(to_markdown(table, result))
    for m, r in result.items():
        f = r["primary"]["coef"]
        print(f"{m}: β_fert {ci(f['x_fert'])}, β_data {ci(f['x_data'])}: {r['primary']['verdict']}")


if __name__ == "__main__":
    main()
