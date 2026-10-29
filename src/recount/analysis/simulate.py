"""Simulated all_runs.csv with known ground truth, in the exact schema the real pipeline
writes. The analysis is written and tested against this before any model is trained.

    python -m recount.analysis.simulate --out results/simulated_runs.csv --seed 0

Generative model, per language i, at dose level r (all effects in bits per byte):

    reference          beta_i (+ size_gap for model S)
    data curve         bpb(f) = beta_i + b_i * ln(1/f)
    residual           rho * ln r / ln 2.25        + N(0, tau)   D(byte, byte)
    budget channel     kappa * b_i * ln r          + N(0, tau)   kappa = 1: granularity changes nothing
    context channel    gamma * ln r / ln 2.25      + N(0, tau)
    interaction        iota * ln r / ln 2.25                     split equally by Shapley

and every trained endpoint gets independent seed noise N(0, sigma_delta / sqrt 2), so a
paired level contrast has SD sigma_delta. Its two context evaluations share that noise
(same weights) plus a little protocol noise.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from ..runs_table import append_rows, budget_of, curve_endpoint

LANGS = ["eus_Latn", "rus_Cyrl", "arb_Arab", "cmn_Hani", "tam_Taml"]
BASE_BPB = {"eus_Latn": 1.10, "rus_Cyrl": 0.92, "arb_Arab": 1.02, "cmn_Hani": 1.25, "tam_Taml": 0.78}
BYTES_PER_SENTENCE = {"eus_Latn": 150.0, "rus_Cyrl": 260.0, "arb_Arab": 210.0, "cmn_Hani": 140.0, "tam_Taml": 400.0}


@dataclass
class Truth:
    rho: float = 0.015  # content-matched residual at r = 2.25
    kappa: float = 1.0  # budget channel / (b ln r)
    gamma: float = 0.006  # context channel at r = 2.25
    iota: float = 0.0  # budget x context interaction at r = 2.25
    b: float = 0.07  # data-curve slope, model M
    b_spread: float = 0.008
    tau: float = 0.004  # between-language SD of each channel
    sigma_delta: float = 0.004  # SD of a paired (level - reference) contrast across seeds
    size_gap: float = 0.10  # model S is this much worse
    size_scale: float = 1.5  # model S's channels are this many times model M's
    shuffle_effect: float = 0.04  # positive control 2
    swap_effect: float = 0.0  # negative control 1: true effect of merge identity
    levels: tuple[float, ...] = (1.0, 1.5, 2.25)
    fractions: tuple[float, ...] = (0.125, 0.25, 0.5)
    seeds: tuple[int, ...] = (0, 1, 2)
    models: tuple[str, ...] = ("M", "S")
    langs: tuple[str, ...] = tuple(LANGS)
    swap_lang: str = "rus_Cyrl"
    swap_level: float = 1.5
    extra: dict = field(default_factory=dict)

    def channels(self, r: float, b_i: float, scale: float = 1.0) -> dict:
        x = math.log(r) / math.log(2.25)
        return {"residual": scale * self.rho * x, "budget": self.kappa * b_i * math.log(r),
                "context": scale * self.gamma * x, "interaction": scale * self.iota * x}


def simulate(truth: Truth, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    sig_e = truth.sigma_delta / math.sqrt(2)

    # language-level random effects, shared by seeds and models
    lang_fx = {
        lang: {"b": truth.b + rng.normal(0, truth.b_spread),
               "u": {lvl: rng.normal(0, truth.tau, size=3) for lvl in truth.levels}}
        for lang in truth.langs
    }

    for model in truth.models:
        scale = 1.0 if model == "M" else truth.size_scale
        gap = 0.0 if model == "M" else truth.size_gap
        for lang in truth.langs:
            fx = lang_fx[lang]
            b_i = fx["b"] * (1.0 if model == "M" else 0.85)
            beta = BASE_BPB[lang] + gap
            for s in truth.seeds:
                unit = {"lang": lang, "model": model, "seed": s, "tokenizer": "main"}

                def endpoint(level: float, name: str, cells: dict[str, float], fraction: float,
                             tokenizer: str = "main") -> None:
                    noise = rng.normal(0, sig_e)  # one trained model
                    for ctx, val in cells.items():
                        # at r = 1 both protocols are the same window, so the same number
                        bpb = val + noise + (rng.normal(0, 2e-5) if level != 1.0 else 0.0)
                        for eval_set in ("heldout", "flores"):
                            bps = BYTES_PER_SENTENCE[lang]
                            v = bpb if eval_set == "heldout" else bpb + 0.03  # FLORES is off-distribution
                            row = unit | {
                                "level": level, "tokenizer": tokenizer, "endpoint": name,
                                "budget": budget_of(name, level), "ctx": ctx, "eval_set": eval_set,
                                "control": "", "bpb": v, "n_bytes": 2_000_000 if eval_set == "heldout" else int(1012 * bps),
                                "n_sentences": "" if eval_set == "heldout" else 1012,
                                "bits_per_sentence": "" if eval_set == "heldout" else v * bps,
                                "content_fraction": fraction, "r_measured": level * (1 + rng.normal(0, 0.003)),
                                "endpoint_id": f"{lang}-{model}-{s}-{level}-{tokenizer}-{name}",
                            }
                            rows.append(row)
                            if eval_set == "heldout" and tokenizer == "main":
                                rows.append(row | {"control": "shuffle", "bpb": v + truth.shuffle_effect * (1 + 0.2 * rng.normal())})
                                rows.append(row | {"control": "stride", "bpb": v + rng.normal(0, 1e-4)})

                # level 1.0: full run (both protocols coincide) and the data curve
                endpoint(1.0, "final", {"tok": beta, "byte": beta}, 1.0)
                for f in truth.fractions:
                    val = beta + b_i * math.log(1 / f)
                    endpoint(1.0, curve_endpoint(f), {"tok": val, "byte": val}, f)
                for r in truth.levels:
                    if r == 1.0:
                        continue
                    ch = truth.channels(r, b_i, scale)
                    u = fx["u"][r]
                    rho, phb, phc = ch["residual"] + u[0], ch["budget"] + u[1], ch["context"] + u[2]
                    io = ch["interaction"]
                    D = {"bb": rho, "tb": rho + phb - io / 2, "bt": rho + phc - io / 2, "tt": rho + phb + phc}
                    endpoint(r, "final", {"byte": beta + D["bb"], "tok": beta + D["bt"]}, 1.0)
                    endpoint(r, "deployed", {"byte": beta + D["tb"], "tok": beta + D["tt"]}, 1.0 / r)
                    if lang == truth.swap_lang and r == truth.swap_level:
                        endpoint(r, "final", {"byte": beta + D["bb"] + truth.swap_effect,
                                              "tok": beta + D["bt"] + truth.swap_effect}, 1.0, tokenizer="swap")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/simulated_runs.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="Truth overrides, e.g. rho=0.03 tau=0.002")
    args = ap.parse_args()
    t = Truth()
    for kv in args.set:
        k, _, v = kv.partition("=")
        setattr(t, k, type(getattr(t, k))(float(v)))
    from pathlib import Path

    out = Path(args.out)
    out.unlink(missing_ok=True)
    append_rows(out, simulate(t, args.seed))
    print(f"wrote {out} ({sum(1 for _ in open(out)) - 1} rows); truth: {asdict(t)}")


if __name__ == "__main__":
    main()
