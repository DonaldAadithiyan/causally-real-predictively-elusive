"""Publication figures for the mechanistic-conformal paper.

Produces three vector PDFs + the small CSV each was built from, so every number is
checkable against RESULTS_FINAL.md / H7A_RESULT.md / SLCP_RESULT.md directly.

  fig_h1c_enrichment.pdf        Fig 2  — from v2_confirm/v2_phase1_seed{0,1,2}/AGG_h1c.csv
  fig_taskg_coverage_width.pdf  Fig 3  — from v2_h6_topup/AGG_h6.csv
  fig_slcp_ratios.pdf           Fig 4  — from runs/slcp_gate/SLCP_gate.csv

No baked-in titles (caption supplies them). Single-column width (~8cm). One
colourblind-safe palette (Okabe-Ito), one colour-to-condition map reused across figures.
Flat style, no 3D / gradient / shadow.

Run:  python figures/make_figures.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = HERE
os.makedirs(OUT, exist_ok=True)

# ---- shared style ---------------------------------------------------------------
COLWIDTH_IN = 3.15         # ~8 cm single column
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,      # embed TrueType, editable text
    "ps.fonttype": 42,
})

# Okabe-Ito, colourblind-safe. One mapping reused everywhere a named condition appears.
OKABE = {
    "black":  "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "blue":   "#0072B2",
    "vermil": "#D55E00",
    "purple": "#CC79A7",
    "grey":   "#999999",
}
COND_COLOR = {
    # H1c criteria
    "MVDF":        OKABE["blue"],
    "MVDF-single": OKABE["orange"],
    "correlated":  OKABE["green"],
    # Task G / SLCP modes
    "naive_matched":   OKABE["vermil"],
    "naive_no_scorer": OKABE["purple"],
    "dense":       OKABE["black"],
    "mvdf":        OKABE["blue"],
    "random_k":    OKABE["skyblue"],
    "pca_k":       OKABE["yellow"],
    "dense+mvdf":  OKABE["grey"],
    # SLCP controls
    "temporal_adj":     OKABE["vermil"],
    "rand_gauss_dense": OKABE["grey"],
}


def _save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, format="pdf")
    plt.close(fig)
    print(f"wrote {p}")


# ==============================================================================
# Figure 2 — H1c enrichment
# ==============================================================================
def fig_h1c():
    rows = []
    for seed in (0, 1, 2):
        df = pd.read_csv(os.path.join(ROOT, "v2_confirm", f"v2_phase1_seed{seed}", "AGG_h1c.csv"))
        for _, r in df.iterrows():
            ds = r["run"].replace("_pre_head", "")
            rows.append(dict(dataset=ds, seed=seed,
                             MVDF=r["enrichment_v2"],
                             **{"MVDF-single": r["enrichment_v1"]},
                             correlated=r["enrichment_corr"]))
    long = pd.DataFrame(rows)
    long.to_csv(os.path.join(OUT, "fig_h1c_enrichment.csv"), index=False)

    datasets = ["ETTh1", "ETTh2", "electricity", "traffic"]
    crits = ["MVDF", "MVDF-single", "correlated"]
    x = np.arange(len(datasets))
    w = 0.26

    fig, ax = plt.subplots(figsize=(COLWIDTH_IN, 2.4))
    for j, c in enumerate(crits):
        means = [long[long.dataset == d][c].mean() for d in datasets]
        ax.bar(x + (j - 1) * w, means, w, color=COND_COLOR[c], label=c,
               edgecolor="white", linewidth=0.4, zorder=2)
        # per-seed dots
        for i, d in enumerate(datasets):
            vals = long[long.dataset == d][c].values
            ax.scatter(np.full_like(vals, x[i] + (j - 1) * w, dtype=float), vals,
                       s=7, color="black", zorder=3, linewidths=0)

    ax.axhline(1.0, ls="--", lw=0.8, color="black", zorder=1)
    ax.text(2.35, 1.12, "base rate (no enrichment)", ha="left", va="bottom",
            fontsize=6, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Type-B enrichment (×)")
    ax.set_xlim(-0.6, len(datasets) - 0.4)
    ax.set_ylim(0, max(5.2, long[crits].values.max() * 1.15))
    ax.legend(frameon=False, ncol=3, loc="upper center", handlelength=1.2,
              columnspacing=1.0, borderaxespad=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_h1c_enrichment.pdf")


# ==============================================================================
# Figure 3 — Task G coverage-width scatter (per-run)
# ==============================================================================
def fig_taskg():
    df_all = pd.read_csv(os.path.join(ROOT, "v2_h6_topup", "AGG_h6.csv"))
    modes = ["naive_matched", "naive_no_scorer", "dense", "mvdf", "correlated",
             "random_k", "pca_k", "dense+mvdf"]
    df_all = df_all[df_all["mode"].isin(modes)]
    # full CSV keeps all 4 datasets for checking; figure drops traffic (directional-only,
    # Task A) so the ETT/electricity cluster the result is about is legible.
    df_all[["dataset", "seed", "mode", "shift_coverage", "shift_width"]].to_csv(
        os.path.join(OUT, "fig_taskg_coverage_width.csv"), index=False)
    df = df_all[df_all["dataset"] != "traffic"]

    fig, ax = plt.subplots(figsize=(COLWIDTH_IN, 2.9))
    for m in modes:
        g = df[df["mode"] == m]
        is_ref = (m == "naive_matched")
        ax.scatter(g["shift_coverage"], g["shift_width"],
                   s=(48 if is_ref else 20),
                   marker=("D" if is_ref else "o"),
                   color=COND_COLOR[m],
                   alpha=(1.0 if is_ref else 0.8),
                   edgecolor=("black" if is_ref else "none"),
                   linewidth=(0.6 if is_ref else 0),
                   label=(m + "  (reference)" if is_ref else m),
                   zorder=(4 if is_ref else 3))

    ax.axvline(0.90, ls=":", lw=0.7, color=OKABE["grey"], zorder=1)
    ax.set_xlabel("shift coverage  (higher is better; dotted = nominal 0.90)")
    ax.set_ylabel("shift interval width  (lower is better)")
    # legend below the plot so the point cloud isn't compressed
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, handlelength=1.0, columnspacing=0.9, labelspacing=0.3,
              handletextpad=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_taskg_coverage_width.pdf")


# ==============================================================================
# Figure 4 — SLCP local-variance ratios
# ==============================================================================
def fig_slcp():
    df = pd.read_csv(os.path.join(ROOT, "runs", "slcp_gate", "SLCP_gate.csv"))
    order = ["temporal_adj", "random_k", "correlated", "mvdf", "pca_k", "rand_gauss_dense"]
    df = df[df["space"].isin(order)]
    df[["dataset", "seed", "space", "ratio_vs_dense"]].to_csv(
        os.path.join(OUT, "fig_slcp_ratios.csv"), index=False)

    fig, ax = plt.subplots(figsize=(COLWIDTH_IN, 2.5))
    for i, sp in enumerate(order):
        vals = df[df["space"] == sp]["ratio_vs_dense"].values
        mean = vals.mean()
        highlight = (sp == "temporal_adj")
        ax.bar(i, mean, 0.62,
               color=COND_COLOR.get(sp, OKABE["grey"]),
               hatch=("//" if highlight else None),
               edgecolor=("black" if highlight else "white"),
               linewidth=(0.7 if highlight else 0.4), zorder=2)
        # per-cell spread
        ax.scatter(np.full_like(vals, i, dtype=float), vals, s=8, color="black",
                   zorder=3, linewidths=0)
        ax.plot([i, i], [vals.min(), vals.max()], color="black", lw=0.7, zorder=3)

    ax.axhline(0.80, ls="--", lw=0.8, color="black", zorder=1)
    ax.text(-0.35, 0.80, "gate 0.80", ha="left", va="center", fontsize=6,
            color="black", backgroundcolor="white")
    ax.axhline(1.0, ls=":", lw=0.6, color=OKABE["grey"], zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("local var($S_t$) ratio vs dense")
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(0, max(1.5, df["ratio_vs_dense"].max() * 1.08))
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_slcp_ratios.pdf")


if __name__ == "__main__":
    fig_h1c()
    fig_taskg()
    fig_slcp()
    print("\nCSVs written alongside each PDF for number-checking.")
