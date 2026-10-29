"""Figure 1 (the finding) and Figure 2 (the mechanism), regenerated from a runs table.

    python -m recount.analysis.figures --runs results/all_runs.csv --out results/figures

Figure 1: BPB penalty against the language's full-vocabulary reference, by dose r (log
axis), pooled over languages with 95% HKSJ bands, for the as-deployed cells
D(tok, tok) and the content-matched cells D(byte, byte); thin lines are languages. The
exchange rate into "training text lost" is a direct label at r = 2.25, not a second axis.

Figure 2: at r = 2.25, each language's D(tok, tok) split into content-matched residual,
budget channel, and context channel, models M and S side by side on one shared scale. A
tick on each bar marks the data-curve prediction for the budget channel, b ln r.

Colors follow the entity across both figures, in the reference palette's fixed order:
content-matched / residual (slot 1), as-deployed / budget channel (slot 2), context
channel (slot 3). Those three slots are documented to pass the all-pairs colour-vision
checks in light mode. Text is always ink, never a series color.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import runs_table  # noqa: E402
from . import decompose as D  # noqa: E402
from .stats import pool, text_lost  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NAMES = {"eus_Latn": "Basque", "rus_Cyrl": "Russian", "arb_Arab": "Arabic", "cmn_Hani": "Chinese", "tam_Taml": "Tamil"}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)


def _banner(fig, simulated: bool) -> None:
    if simulated:
        fig.text(0.995, 0.995, "SIMULATED DATA: layout mockup, not a result", ha="right", va="top",
                 fontsize=8, color=INK2, fontweight="bold")


def figure1(runs: pd.DataFrame, model: str = "M", simulated: bool = False):
    c = D.cells(runs)
    c = c[c["model"] == model]
    curve = D.data_curve(runs)
    b = pool(curve[curve["model"] == model], "b").estimate
    levels = [1.0] + sorted(c["level"].unique())
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    pooled = {}
    for col, color, label in (("D_tt", ORANGE, "As deployed: token budget, token context"),
                              ("D_bb", BLUE, "Content-matched: byte budget, byte context")):
        est, lo, hi = [0.0], [0.0], [0.0]
        for lv in levels[1:]:
            p = pool(c[c["level"] == lv], col)
            est.append(p.estimate), lo.append(p.lo), hi.append(p.hi)
        pooled[col] = (np.array(est), np.array(lo), np.array(hi))
        for lang, g in c.groupby("lang"):  # thin context lines: one per language
            ys = [0.0] + [g[g["level"] == lv][col].mean() for lv in levels[1:]]
            ax.plot(levels, ys, color=color, alpha=0.3, linewidth=1, solid_capstyle="round", zorder=2)
        ax.fill_between(levels, lo, hi, color=color, alpha=0.10, linewidth=0, zorder=1)
        ax.plot(levels, est, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                zorder=4, label=label)
        ax.scatter(levels, est, s=40, color=color, edgecolor=SURFACE, linewidth=2, zorder=5)
    ax.set_xscale("log")
    ax.set_xticks(levels, [f"{lv:g}×" for lv in levels])
    ax.minorticks_off()
    ax.set_xlim(0.93, max(levels) * 1.32)
    ax.axhline(0, color=AXIS, linewidth=1, zorder=1)
    ax.set_xlabel("Tokens per byte, relative to the language's full tokenizer (log scale)", color=INK2, fontsize=9)
    ax.set_ylabel("Bits-per-byte penalty vs. full tokenizer", color=INK2, fontsize=9)
    r = max(levels)
    tt, bb = pooled["D_tt"][0][-1], pooled["D_bb"][0][-1]
    for y, name in ((tt, "as deployed"), (bb, "content-matched")):  # exchange rate as direct labels
        ax.annotate(f"{y:+.3f} bpb\n≈ {100 * text_lost(y, b):.0f}% of text lost", (r, y), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    share = 1 - bb / tt
    ax.annotate("", xy=(r * 1.0, bb), xytext=(r * 1.0, tt),
                arrowprops=dict(arrowstyle="<->", color=INK2, linewidth=1, shrinkA=6, shrinkB=6))
    ax.text(r * 0.965, (tt + bb) / 2, f"{100 * share:.0f}% removed\nby counting\nin content", ha="right",
            va="center", fontsize=8, color=INK2)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    for h in leg.legend_handles:
        h.set_linewidth(2)
    ax.set_title(f"Figure 1. The tokenization tax, pooled over {c['lang'].nunique()} languages (model {model})",
                 loc="left", fontsize=10, color=INK)
    _banner(fig, simulated)
    fig.tight_layout()
    return fig


def figure2(runs: pd.DataFrame, level: float = 2.25, simulated: bool = False):
    c = D.cells(runs)
    c = c[c["level"] == level]
    curve = D.data_curve(runs)
    models = [m for m in ("M", "S") if m in set(c["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models) + 0.6, 3.9), dpi=200, sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)
    parts = (("residual", BLUE, "Content-matched residual"), ("phi_budget", ORANGE, "Budget channel"),
             ("phi_context", AQUA, "Context channel"))
    for ax, model in zip(axes, models):
        _style(ax)
        cm = c[c["model"] == model]
        means = cm.groupby("lang")[[p[0] for p in parts]].mean()
        langs = sorted(means.index, key=lambda lg: list(NAMES).index(lg) if lg in NAMES else len(NAMES))
        rows = [(NAMES.get(lg, lg), means.loc[lg]) for lg in langs]
        rows.append(("Pooled", pd.Series({p[0]: pool(cm, p[0]).estimate for p in parts})))
        b_lang = curve[curve["model"] == model].groupby("lang")["b"].mean()
        b_pool = pool(curve[curve["model"] == model], "b").estimate
        x = np.arange(len(rows))
        width = 0.5
        for i, (name, vals) in enumerate(rows):
            base = 0.0
            for key, color, _ in parts:  # waterfall stack: each segment starts where the last ended
                v = float(vals[key])
                ax.bar(i, v, width, bottom=base, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
                base += v
            bl = b_pool if name == "Pooled" else b_lang.get(langs[i], b_pool)
            pred = float(vals["residual"]) + bl * math.log(level)  # residual + predicted budget channel
            ax.plot([i - width / 2 - 0.06, i + width / 2 + 0.06], [pred, pred], color=INK, linewidth=1.5, zorder=4)
        ax.axhline(0, color=AXIS, linewidth=1, zorder=2)
        ax.set_xticks(x, [n for n, _ in rows], rotation=0, fontsize=8)
        ax.axvline(len(rows) - 1.5, color=GRID, linewidth=1, zorder=1)
        ax.set_title(f"Model {model}", loc="left", fontsize=9, color=INK)
    axes[0].set_ylabel(f"Bits-per-byte penalty at r = {level:g}", color=INK2, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for _, col, _ in parts]
    handles.append(plt.Line2D([0], [0], color=INK, linewidth=1.5))
    labels = [lab for _, _, lab in parts] + ["Data-curve prediction, b·ln r (budget channel's top)"]
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8, labelcolor=INK,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Figure 2. Where the r = {level:g} penalty comes from", x=0.02, ha="left", fontsize=10, color=INK)
    _banner(fig, simulated)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/all_runs.csv")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--simulated", action="store_true", help="stamp the figures as mockups (default: if 'simulated' is in --runs)")
    args = ap.parse_args()
    runs = runs_table.load(args.runs)
    simulated = args.simulated or "simulated" in Path(args.runs).name
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_simulated" if simulated else ""
    for name, fig in (("figure1", figure1(runs, simulated=simulated)), ("figure2", figure2(runs, simulated=simulated))):
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}{suffix}.{ext}", facecolor=SURFACE, bbox_inches="tight")
        print(f"wrote {out / name}{suffix}.png/.pdf")


if __name__ == "__main__":
    main()
