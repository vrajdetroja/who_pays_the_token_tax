"""Power by simulation, using the analysis's own pooling code (PREREGISTRATION.md §6).

    python -m recount.analysis.power --reps 2000

Rather than a normal approximation, this simulates the quantities the preregistered
tests actually compute, under the generative model of
`simulate.py` reduced to the two cells H1 needs. Per (language i, seed s):

    D_bb = rho        + u_i + e_r_final - e_ref
    D_tt = rho + phi  + v_i + e_r_dep   - e_ref       phi = budget + context channels

with u_i, v_i ~ N(0, tau^2) correlated across the two cells (corr `lang_corr`; the same
language tends to be hard in both), and each trained endpoint's seed noise
e ~ N(0, sigma_delta^2 / 2), so a paired contrast has SD sigma_delta and the shared
reference correlates the two cells. Reported per scenario:

    tost_pass@rho=0       P(residual declared equivalent to 0 within +/-SESOI), true residual 0
    falsify@S=0.3         P(H1 falsified) when content accounting removes only 30%
    falsify@S=0.5         P(H1 falsified) at the boundary: the test's size (nominal 0.025)
    support@S=0.8         P(L's interval entirely above 0) when accounting removes 80%
    coverage              P(95% interval for D_bb covers rho)
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from .stats import hksj, sesoi


def _pooled(y: np.ndarray, level: float) -> tuple[float, float, float]:
    """HKSJ over languages for y of shape (langs, seeds), pooled within-language variance."""
    k, n = y.shape
    s2 = y.var(axis=1, ddof=1).mean()  # equal n: pooled variance is the mean variance
    p = hksj(y.mean(axis=1), np.full(k, s2 / n), level)
    return p.estimate, p.lo, p.hi


def one_rep(rng, *, rho, phi, tau, sigma_delta, langs, seeds, lang_corr, bound, share_threshold):
    cov = tau**2 * np.array([[1.0, lang_corr], [lang_corr, 1.0]])
    u = rng.multivariate_normal([0.0, 0.0], cov, size=langs)
    se = sigma_delta / np.sqrt(2)
    ref, fin, dep = (rng.normal(0, se, size=(langs, seeds)) for _ in range(3))
    d_bb = rho + u[:, :1] + fin - ref
    d_tt = rho + phi + u[:, 1:] + dep - ref
    L = (1 - share_threshold) * d_tt - d_bb
    _, l_lo, l_hi = _pooled(L, 0.95)
    _, b_lo90, b_hi90 = _pooled(d_bb, 0.90)
    _, b_lo, b_hi = _pooled(d_bb, 0.95)
    return {"falsified": l_hi < 0, "supported": l_lo > 0, "tost": (-bound < b_lo90) and (b_hi90 < bound),
            "covers": b_lo <= rho <= b_hi}


def scenario(rng, reps: int, *, deployed: float, share: float, **kw) -> dict:
    rho = (1 - share) * deployed
    out = [one_rep(rng, rho=rho, phi=deployed - rho, **kw) for _ in range(reps)]
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deployed", type=float, default=0.08, help="true D(tok, tok) at r = 2.25 (BPB)")
    ap.add_argument("--b", type=float, default=0.07, help="data-curve slope, sets the SESOI")
    ap.add_argument("--langs", type=int, default=5)
    ap.add_argument("--sigma", type=float, nargs="+", default=[0.002, 0.004, 0.006])
    ap.add_argument("--tau", type=float, nargs="+", default=[0.002, 0.004, 0.008])
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--lang-corr", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="CSV path")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    bound = sesoi(args.b)
    rows = []
    for sigma, tau, n in itertools.product(args.sigma, args.tau, args.seeds):
        kw = dict(tau=tau, sigma_delta=sigma, langs=args.langs, seeds=n, lang_corr=args.lang_corr,
                  bound=bound, share_threshold=0.5)
        row = {"sigma_delta": sigma, "tau": tau, "seeds": n}
        row["tost_pass@rho=0"] = scenario(rng, args.reps, deployed=args.deployed, share=1.0, **kw)["tost"]
        row["falsify@S=0.3"] = scenario(rng, args.reps, deployed=args.deployed, share=0.3, **kw)["falsified"]
        s5 = scenario(rng, args.reps, deployed=args.deployed, share=0.5, **kw)
        row["falsify@S=0.5"] = s5["falsified"]
        row["support@S=0.8"] = scenario(rng, args.reps, deployed=args.deployed, share=0.8, **kw)["supported"]
        row["coverage"] = s5["covers"]
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"SESOI = {bound:.4f} BPB (b = {args.b}); true D(tok,tok) = {args.deployed}; {args.reps} reps per cell")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.out:
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
