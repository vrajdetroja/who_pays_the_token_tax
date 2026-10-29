"""Interval estimates for the preregistered analysis (PREREGISTRATION.md §7).

Languages are the units of the confirmatory analysis. Each contributes the mean of a
quantity over its seeds; `hksj` pools those means with a random-effects meta-analysis and
Hartung-Knapp-Sidik-Jonkman intervals (IntHout, Ioannidis & Borm, 2014), which keep their
coverage with few, heterogeneous units. Five languages give 4 degrees of freedom.

Within-language variances come from 3 seeds, so each rests on 2 degrees of freedom: its
95% range runs from 0.025x to 3.7x the true variance. By default they are pooled across
languages (`within="pooled"`) so no language's weight hinges on a lucky trio of seeds;
per-language variances are a sensitivity analysis.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Pooled:
    """A pooled estimate with its interval at `level`."""

    estimate: float
    se: float
    df: int
    tau2: float
    lo: float
    hi: float
    level: float
    k: int

    def at(self, level: float) -> tuple[float, float]:
        t = stats.t.ppf(0.5 + level / 2, self.df)
        return self.estimate - t * self.se, self.estimate + t * self.se

    def to_dict(self) -> dict:
        return asdict(self)


def t_interval(x, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and t-interval of a small sample (per-language results: descriptive only)."""
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    if len(x) < 2:
        return m, float("nan"), float("nan")
    half = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - half, m + half


def hksj(y, v, level: float = 0.95, modified: bool = False) -> Pooled:
    """Random-effects pooled mean of `y` (within-unit variances `v`), HKSJ interval.

    tau^2 is the DerSimonian-Laird estimate, as in IntHout et al. (2014). `modified=True`
    floors the HKSJ scale factor at 1 (Knapp & Hartung's ad hoc correction), which can only
    widen the interval; it is reported as a sensitivity analysis.
    """
    y, v = np.asarray(y, dtype=float), np.asarray(v, dtype=float)
    k = len(y)
    if k < 2:
        raise ValueError("need at least two units to pool")
    if np.any(v <= 0):
        raise ValueError("within-unit variances must be positive")
    w = 1.0 / v
    y_fixed = (w * y).sum() / w.sum()
    Q = (w * (y - y_fixed) ** 2).sum()
    C = w.sum() - (w**2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C)
    ws = 1.0 / (v + tau2)
    mu = (ws * y).sum() / ws.sum()
    q = (ws * (y - mu) ** 2).sum() / (k - 1)
    if modified:
        q = max(q, 1.0)
    se = float(np.sqrt(q / ws.sum()))
    t = stats.t.ppf(0.5 + level / 2, k - 1)
    return Pooled(float(mu), se, k - 1, float(tau2), float(mu - t * se), float(mu + t * se), level, k)


def language_summaries(df: pd.DataFrame, col: str, unit: str = "lang") -> pd.DataFrame:
    """Per-language mean, sample variance, and seed count of `col`."""
    g = df.groupby(unit)[col]
    out = pd.DataFrame({"mean": g.mean(), "var": g.var(ddof=1), "n": g.count()})
    if (out["n"] < 2).any():
        raise ValueError(f"{col}: every {unit} needs at least two seeds")
    return out


def within_variances(summ: pd.DataFrame, within: str = "pooled") -> np.ndarray:
    """Variance of each language's mean: s_i^2 / n_i, or the pooled s^2 / n_i."""
    if within == "per_language":
        return (summ["var"] / summ["n"]).to_numpy()
    if within == "pooled":
        dof = summ["n"] - 1
        s2 = float((dof * summ["var"]).sum() / dof.sum())
        if s2 <= 0:
            raise ValueError("pooled within-language variance is zero")
        return (s2 / summ["n"]).to_numpy()
    raise ValueError(f"unknown within={within!r}")


def pool(df: pd.DataFrame, col: str, level: float = 0.95, within: str = "pooled",
         modified: bool = False) -> Pooled:
    """Pool a per-(language, seed) quantity over languages."""
    summ = language_summaries(df, col)
    return hksj(summ["mean"].to_numpy(), within_variances(summ, within), level, modified)


def tost(df: pd.DataFrame, col: str, bound: float, alpha: float = 0.05, within: str = "pooled") -> dict:
    """Two one-sided tests against +/-bound: equivalent iff the (1 - 2 alpha) interval is inside."""
    p = pool(df, col, level=1 - 2 * alpha, within=within)
    return {"equivalent": bool(-bound < p.lo and p.hi < bound), "bound": bound,
            "estimate": p.estimate, "lo": p.lo, "hi": p.hi, "level": p.level}


def invert_ratio_test(df: pd.DataFrame, num: str, den: str, level: float = 0.95,
                      within: str = "pooled", grid: np.ndarray | None = None) -> dict:
    """Confidence set for S = 1 - mean(num) / mean(den), by inverting the pooled test of
    L(s) = (1 - s) * den - num = 0 (Fieller's construction, with HKSJ pooling).

    L(s) is formed per (language, seed) before pooling, so the correlation between `num`
    and `den` (same reference, same seed) is carried exactly; its per-language mean and
    variance follow in closed form from the two columns' moments. The set can be unbounded
    when `den` is not clearly nonzero; `bounded` says whether it reached the grid's edges.
    """
    grid = np.linspace(-2.0, 3.0, 5001) if grid is None else grid
    g = df.groupby("lang")
    n = g[num].count().to_numpy()
    m_num, m_den = g[num].mean().to_numpy(), g[den].mean().to_numpy()
    v_num, v_den = g[num].var(ddof=1).to_numpy(), g[den].var(ddof=1).to_numpy()
    cov = g[[num, den]].cov().xs(num, level=1)[den].to_numpy()
    accepted = np.zeros(len(grid), dtype=bool)
    for i, s in enumerate(grid):
        a = 1.0 - s
        mean = a * m_den - m_num
        var = a * a * v_den + v_num - 2 * a * cov
        summ = pd.DataFrame({"mean": mean, "var": var, "n": n})
        p = hksj(mean, within_variances(summ, within), level)
        accepted[i] = p.lo <= 0.0 <= p.hi
    if not accepted.any():
        return {"lo": float("nan"), "hi": float("nan"), "bounded": True, "contiguous": True, "level": level}
    idx = np.flatnonzero(accepted)
    contiguous = bool(np.all(np.diff(idx) == 1))
    bounded = bool(idx[0] > 0 and idx[-1] < len(grid) - 1)
    return {"lo": float(grid[idx[0]]), "hi": float(grid[idx[-1]]), "bounded": bounded,
            "contiguous": contiguous, "level": level}


def bootstrap_ratio(df: pd.DataFrame, num: str, den: str, draws: int = 10_000, seed: int = 0,
                    level: float = 0.95, within: str = "pooled") -> dict:
    """A parametric bootstrap for S = 1 - mean(num) / mean(den).

    Draws the two pooled means from a bivariate t (df = k - 1) centred on the HKSJ
    estimates, with their SEs and the correlation of the per-language means.
    """
    pn, pd_ = pool(df, num, within=within), pool(df, den, within=within)
    means = df.groupby("lang")[[num, den]].mean()
    rho = float(np.corrcoef(means[num], means[den])[0, 1]) if len(means) > 2 else 0.0
    cov = np.array([[pn.se**2, rho * pn.se * pd_.se], [rho * pn.se * pd_.se, pd_.se**2]])
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0.0, 0.0], cov, size=draws)
    chi = rng.chisquare(pn.df, size=draws) / pn.df
    x = np.array([pn.estimate, pd_.estimate]) + z / np.sqrt(chi)[:, None]
    s = 1.0 - x[:, 0] / x[:, 1]
    lo, hi = np.quantile(s, [0.5 - level / 2, 0.5 + level / 2])
    return {"lo": float(lo), "hi": float(hi), "rho": rho, "draws": draws, "level": level}


def fit_log_slope(fraction, bpb) -> dict:
    """OLS of bpb on ln(fraction): b is BPB gained per e-fold of content lost (b > 0 when
    more text helps). Also returns the quadratic curvature, to judge 'roughly linear'."""
    x, y = np.log(np.asarray(fraction, dtype=float)), np.asarray(bpb, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    r2 = 1 - (resid**2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-300)
    curv = float(np.polyfit(x, y, 2)[0]) if len(x) >= 3 else float("nan")
    return {"b": float(-slope), "intercept": float(intercept), "r2": float(r2), "curvature": curv}


def text_lost(effect, b: float):
    """Exchange rate: a BPB effect expressed as the fraction of training text whose loss
    would cost the same, from the data curve bpb = a - b ln(fraction)."""
    return 1.0 - np.exp(-np.asarray(effect, dtype=float) / b)


def sesoi(b: float, text_fraction: float = 0.10) -> float:
    """Smallest effect of interest: the BPB cost of discarding `text_fraction` of the text."""
    return float(b * np.log(1.0 / (1.0 - text_fraction)))


def apply(df: pd.DataFrame, fn: Callable[[pd.DataFrame], dict], by: list[str]) -> pd.DataFrame:
    """Small groupby-apply that returns one row per group (pandas' version warns)."""
    rows = []
    for key, g in df.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(by, key)) | fn(g))
    return pd.DataFrame(rows)
