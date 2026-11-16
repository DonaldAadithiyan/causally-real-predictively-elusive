"""Publication figures for the mechanistic-conformal paper — all five, regenerated.

Produces five vector PDFs + the small CSV each was built from, so every number is
checkable against the source .md files directly:

  fig_h1c_enrichment.pdf       Fig 1  <- RESULTS_FINAL.md, H1c section
                                        (v2_confirm/v2_phase1_seed{0,1,2}/AGG_h1c.csv)
  fig_phase3_dissociation.pdf  Fig 2  <- results/v2/phase3_real/*/rj_vs_predictive.csv
  fig_oracle_width.pdf         Fig 3  <- results/v2/phase4_oracle/PHASE4_RESULT.md
                                        (results/v2/phase4_oracle/AGG_phase4.csv or per-cell table)
  fig_taskg_coverage_width.pdf Fig 4  <- RESULTS_FINAL.md, H6 / Task G
                                        (v2_h6_topup/AGG_h6.csv)
  fig_slcp_ratios.pdf          Fig 5  <- SLCP_RESULT.md temporal-exclusion re-test
                                        (results/v2/slcp_gate/SLCP_gate_temporal_excluded.csv)

No baked-in titles (the paper's \\caption supplies them). Axis labels + legends only.
One colourblind-safe palette (Okabe-Ito), one colour-to-condition map reused across
figures. Flat style: no 3D / gradient / shadow. Legible at single-column width (~8 cm).

Run:  python figures/make_paper_figures.py
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

# ---- shared style ------------------------------------------------------------
COLWIDTH_IN = 3.15  # ~8 cm single column
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
    "pdf.fonttype": 42,  # embed TrueType, editable text
    "ps.fonttype": 42,
})

# Okabe-Ito, colourblind-safe. One mapping reused everywhere a named condition appears.
OKABE = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermil":  "#D55E00",
    "purple":  "#CC79A7",
    "grey":    "#999999",
}

# One colour-to-condition map, reused in every figure a named condition appears in
# more than once.
COND_COLOR = {
    # H1c criteria (Fig 1)
    "MVDF":            OKABE["blue"],
    "MVDF-single":     OKABE["orange"],
    "correlated":      OKABE["green"],
    # Task G / SLCP modes (Fig 4, Fig 5)
    "naive_matched":   OKABE["vermil"],
    "naive_no_scorer": OKABE["purple"],
    "dense":           OKABE["black"],
    "mvdf":            OKABE["blue"],
    "correlated_mode": OKABE["green"],
    "random_k":        OKABE["skyblue"],
    "pca_k":           OKABE["yellow"],
    "dense+mvdf":      OKABE["grey"],
}
# lowercase "correlated" as a Task-G/SLCP mode shares the H1c "correlated" green
COND_COLOR["correlated"] = OKABE["green"]

# Evidence-level accents (Fig 2 only plots an L2 quantity against an L1 quantity).
# This is the convention the pipeline diagram (separate MD) should match:
#   L2  (causal specificity, R_j / dose-response) -> reddish purple
#   L1  (natural predictive utility, held-out R^2) -> blue
L2_ACCENT = OKABE["purple"]
L1_ACCENT = OKABE["blue"]

# Feature-type palette for Fig 2 point colours (A/B/C/D).
TYPE_COLOR = {
    "A": OKABE["vermil"],
    "B": OKABE["blue"],
    "C": OKABE["green"],
    "D": OKABE["orange"],
}

ORACLE_HATCH = "///"  # oracle bars (Fig 3) must look unlike every solid bar elsewhere


def _save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, format="pdf")
    plt.close(fig)
    print(f"wrote {p}")


def _write_csv(df, name):
    p = os.path.join(OUT, name)
    df.to_csv(p, index=False)
    print(f"wrote {p}")


# ===========================================================================
# Figure 1 -- H1c type-B enrichment
#   source: RESULTS_FINAL.md H1c  ->  n_ge_1  MVDF 12/12  MVDF-single 4/12  corr 2/12
#           min/max/mean  MVDF 2.00/4.62/3.02  single 0.00/1.85/0.69  corr 0.00/1.20/0.64
# ===========================================================================
def fig_h1c():
    rows = []
    for seed in (0, 1, 2):
        f = os.path.join(ROOT, "results/v2/v2_confirm", f"v2_phase1_seed{seed}", "AGG_h1c.csv")
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            ds = str(r["run"]).replace("_pre_head", "")
            rows.append(dict(
                dataset=ds, seed=seed,
                MVDF=r["enrichment_v2"],
                **{"MVDF-single": r["enrichment_v1"]},
                correlated=r["enrichment_corr"],
            ))
    long = pd.DataFrame(rows)
    _write_csv(long, "fig_h1c_enrichment.csv")

    # sanity print against RESULTS_FINAL.md
    for c in ("MVDF", "MVDF-single", "correlated"):
        v = long[c].values
        print(f"  [check] {c:12s} >=1: {int((v >= 1.0).sum())}/{len(v)}  "
              f"min {v.min():.2f} max {v.max():.2f} mean {v.mean():.2f}")

    datasets = ["ETTh1", "ETTh2", "electricity", "traffic"]
    crits = ["MVDF", "MVDF-single", "correlated"]
    x = np.arange(len(datasets))
    w = 0.26

    fig, ax = plt.subplots(figsize=(COLWIDTH_IN, 2.5))
    for j, c in enumerate(crits):
        means = [long[long.dataset == d][c].mean() for d in datasets]
        ax.bar(x + (j - 1) * w, means, w, color=COND_COLOR[c], label=c,
               edgecolor="white", linewidth=0.4, zorder=2)
        for i, d in enumerate(datasets):
            vals = long[long.dataset == d][c].values
            ax.scatter(np.full_like(vals, x[i] + (j - 1) * w, dtype=float), vals,
                       s=7, color="black", zorder=3, linewidths=0)

    ax.axhline(1.0, ls="--", lw=0.8, color="black", zorder=1)
    ax.text(len(datasets) - 0.55, 1.10, "no enrichment", ha="right", va="bottom",
            fontsize=6, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(r"Type-B enrichment ($\times$)")
    ax.set_xlim(-0.6, len(datasets) - 0.4)
    ax.set_ylim(0, max(5.2, long[crits].values.max() * 1.15))
    ax.legend(frameon=False, ncol=3, loc="upper center", handlelength=1.2,
              columnspacing=1.0, borderaxespad=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_h1c_enrichment.pdf")


# ===========================================================================
# Figure 2 -- Phase 3 causal / predictive dissociation
#   source: results/v2/phase3_real/<ds>_seed<s>/rj_vs_predictive.csv
#   x = R_j (L2 causal specificity), y = best held-out natural predictive R^2 (L1)
#   rho in [-0.10, +0.13]  (exact per-cell: -0.1008 .. +0.1331)
# ===========================================================================
def fig_phase3():
    from scipy.stats import spearmanr
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    frames = []
    for ds in ("ETTh1", "ETTh2"):
        for s in (0, 1, 2):
            f = os.path.join(ROOT, "results", "v2", "phase3_real", f"{ds}_seed{s}",
                             "rj_vs_predictive.csv")
            d = pd.read_csv(f)
            d["dataset"] = ds
            d["seed"] = s
            frames.append(d)
    allf = pd.concat(frames, ignore_index=True)
    keep = ["dataset", "seed", "feature", "type", "R_j",
            "best_natural_predictive_r2", "in_mvdf"]
    _write_csv(allf[keep], "fig_phase3_dissociation.csv")

    # per-cell + pooled rank correlation (report in caption text, not on the image)
    for ds in ("ETTh1", "ETTh2"):
        for s in (0, 1, 2):
            g = allf[(allf.dataset == ds) & (allf.seed == s)]
            rho, _ = spearmanr(g["R_j"], g["best_natural_predictive_r2"])
            print(f"  [check] {ds} seed{s}  rho={rho:+.4f}  n={len(g)}")
    rho_all, _ = spearmanr(allf["R_j"], allf["best_natural_predictive_r2"])
    print(f"  [check] pooled rho={rho_all:+.4f}")

    # Side-by-side: the two datasets share the plot's horizontal reading order
    # (ETTh1 left, ETTh2 right) and share y-axis label space; ranges differ enough
    # that each panel keeps its own y-scale. The legend lives in a dedicated strip
    # above both axes (via a 3rd, short "legend row" in the gridspec) so it can
    # never sit on top of a point cloud, at any panel's data range.
    fig = plt.figure(figsize=(COLWIDTH_IN * 2.05, 2.85))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.16, 1], hspace=0.06, wspace=0.30)
    leg_ax = fig.add_subplot(gs[0, :])
    leg_ax.axis("off")
    axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    handles = None
    for ax, ds in zip(axes, ("ETTh1", "ETTh2")):
        g = allf[allf.dataset == ds]
        for t in ("A", "B", "C", "D"):
            gt = g[g["type"] == t]
            if len(gt):
                h = ax.scatter(gt["R_j"], gt["best_natural_predictive_r2"],
                               s=9, color=TYPE_COLOR[t], alpha=0.8, linewidths=0,
                               label=f"type {t}")
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        # thin, light guide line only -- the point of the figure is the absence of trend
        if g["R_j"].nunique() > 2:
            b, a = np.polyfit(g["R_j"], g["best_natural_predictive_r2"], 1)
            xs = np.linspace(g["R_j"].min(), g["R_j"].max(), 50)
            ax.plot(xs, a + b * xs, color=OKABE["grey"], lw=0.6, ls="-", zorder=1)
        rho, _ = spearmanr(g["R_j"], g["best_natural_predictive_r2"])
        ax.set_title(f"{ds}   " + r"rank $\rho$" + f" = {rho:+.2f}",
                     fontsize=7, pad=4)
        ax.axvline(0.0, color="black", lw=0.4, zorder=0)
        ax.set_xlabel(r"causal specificity $R_j$  (L2)", color=L2_ACCENT, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=6)
        # few, non-overlapping x ticks with a single shared x1e-3 offset label,
        # instead of 7 signed 4-decimal labels crammed edge-to-edge.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 2))
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.get_offset_text().set_fontsize(6)
    axes[0].set_ylabel(r"best natural pred. $R^2$ (L1)", color=L1_ACCENT, fontsize=7)

    leg_ax.legend(handles, labels, loc="center", ncol=4, frameon=False,
                  handlelength=1.0, labelspacing=0.2, columnspacing=1.1,
                  handletextpad=0.3, borderaxespad=0.0)
    _save(fig, "fig_phase3_dissociation.pdf")


# ===========================================================================
# Figure 3 -- oracle vs naive_matched interval width (retrospective ceiling)
#   source: results/v2/phase4_oracle/PHASE4_RESULT.md per-dataset summary
#     ETTh1  ID  naive 1.7461  oracle 1.3174  (-24.5%)
#     ETTh1  shift naive 1.8441 oracle 1.3758  (-25.4%)
#     ETTh2  ID  naive 3.3819  oracle 2.3936  (-29.2%)
#     ETTh2  shift naive 3.0796 oracle 2.3235  (-24.5%)
# ===========================================================================
def fig_oracle():
    # prefer a machine-readable source if present; else use the frozen per-seed table
    cand = [os.path.join(ROOT, "results", "v2", "phase4_oracle", n)
            for n in ("AGG_oracle.csv", "AGG_phase4.csv", "phase4_oracle.csv")]
    src = next((c for c in cand if os.path.exists(c)), None)

    if src is not None:
        raw = pd.read_csv(src)
        print(f"  [info] fig_oracle from {os.path.basename(src)}")
        # normalise column names we might see
        def pick(*names):
            for n in names:
                if n in raw.columns:
                    return n
            return None
        rows = []
        for ds in ("ETTh1", "ETTh2"):
            sub = raw[raw[pick("dataset", "ds")] == ds]
            for blk, ncol, ocol in [
                ("ID",    pick("naive_id_width"),    pick("oracle_id_width")),
                ("shift", pick("naive_shift_width"), pick("oracle_shift_width")),
            ]:
                rows.append(dict(dataset=ds, window=blk,
                                 naive_matched=float(sub[ncol].mean()),
                                 oracle=float(sub[ocol].mean())))
        tab = pd.DataFrame(rows)
    else:
        print("  [info] fig_oracle from PHASE4_RESULT.md per-dataset summary "
              "(no machine-readable AGG found)")
        tab = pd.DataFrame([
            dict(dataset="ETTh1", window="ID",    naive_matched=1.7461, oracle=1.3174),
            dict(dataset="ETTh1", window="shift", naive_matched=1.8441, oracle=1.3758),
            dict(dataset="ETTh2", window="ID",    naive_matched=3.3819, oracle=2.3936),
            dict(dataset="ETTh2", window="shift", naive_matched=3.0796, oracle=2.3235),
        ])

    tab["pct_reduction"] = (1.0 - tab["oracle"] / tab["naive_matched"]) * 100.0
    # PHASE4_RESULT.md's "Precondition check + reading" quotes the reduction as
    # (W_naive - W_oracle)/W_naive = 24.5 / 25.4 / 29.2 / 24.5 %. Our exact-mean
    # ratios (24.55 / 25.39 / 29.22 / 24.55) agree with those to <0.05pp; the label
    # is pinned to the source file's published value so the figure text matches it
    # verbatim, and the exact ratio is kept in the CSV for checking.
    SRC_PCT = {("ETTh1", "ID"): 24.5, ("ETTh1", "shift"): 25.4,
               ("ETTh2", "ID"): 29.2, ("ETTh2", "shift"): 24.5}
    tab["pct_label"] = [SRC_PCT[(r.dataset, r.window)] for _, r in tab.iterrows()]
    assert (tab["pct_reduction"] - tab["pct_label"]).abs().max() < 0.1, \
        "exact oracle reduction drifted >0.1pp from PHASE4_RESULT.md — investigate"
    _write_csv(tab, "fig_oracle_width.csv")
    for _, r in tab.iterrows():
        print(f"  [check] {r.dataset}-{r.window:5s}  naive {r.naive_matched:.3f}  "
              f"oracle {r.oracle:.3f}  (-{r.pct_label:.1f}%  [exact {r.pct_reduction:.2f}])")

    labels = [f"{r.dataset}\n{r.window}" for _, r in tab.iterrows()]
    x = np.arange(len(tab))
    w = 0.36
    ymax = tab[["naive_matched", "oracle"]].values.max()

    fig, ax = plt.subplots(figsize=(COLWIDTH_IN, 2.7))
    ax.bar(x - w / 2, tab["naive_matched"], w, color=COND_COLOR["naive_matched"],
           edgecolor="white", linewidth=0.4, label="naive_matched", zorder=2)
    ax.bar(x + w / 2, tab["oracle"], w, facecolor="white",
           edgecolor=OKABE["grey"], linewidth=0.8, hatch=ORACLE_HATCH,
           label="oracle (non-deployable ceiling)", zorder=2)

    for i, r in tab.reset_index(drop=True).iterrows():
        top = max(r.naive_matched, r.oracle)
        ax.text(i, top + 0.06 * ymax, f"−{r.pct_label:.1f}%", ha="center",
                va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"interval width in $S_t$ units")
    ax.set_ylim(0, ymax * 1.42)  # headroom for annotations + legend
    ax.legend(frameon=False, loc="upper center", ncol=1, handlelength=1.4,
              bbox_to_anchor=(0.5, 1.02), borderaxespad=0.0)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_oracle_width.pdf")


# ===========================================================================
# Figure 4 -- Task G shift coverage vs shift width, all seven modes
#   source: RESULTS_FINAL.md H6 / Task G  (v2_h6_topup/AGG_h6.csv, per-seed)
#   aggregate (4 datasets x 3 seeds) matches RESULTS_FINAL exactly:
#     naive_matched   0.901 / 3.345      naive_no_scorer 0.903 / 3.331
#     dense 0.896/4.354  mvdf 0.870/3.806  correlated 0.890/3.894
#     random_k 0.881/3.944  pca_k 0.894/4.363
# ===========================================================================
def fig_taskg():
    src = os.path.join(HERE, "fig_taskg_coverage_width.csv")
    if not os.path.exists(src):
        src = os.path.join(ROOT, "results/v2/v2_h6_topup", "AGG_h6.csv")
    df = pd.read_csv(src)

    modes = ["naive_matched", "naive_no_scorer", "dense", "mvdf", "correlated",
             "random_k", "pca_k"]
    df = df[df["mode"].isin(modes)].copy()

    # CSV keeps every per-run row for all 4 datasets + the mode-means (the numbers
    # that match RESULTS_FINAL's Task G table); the plot below focuses the view.
    per_run = df[["dataset", "seed", "mode", "shift_coverage", "shift_width"]].copy()
    means = (per_run.groupby("mode")[["shift_coverage", "shift_width"]]
             .mean().reindex(modes).reset_index())
    means["kind"] = "mean_4ds"
    out = pd.concat([per_run.assign(kind="run"), means], ignore_index=True)
    _write_csv(out, "fig_taskg_coverage_width.csv")
    for _, r in means.iterrows():
        print(f"  [check] {r['mode']:16s} shift_cov {r.shift_coverage:.3f}  "
              f"shift_width {r.shift_width:.3f}")

    # traffic's shift width (~8) is off-scale for the ETT/electricity cluster this
    # result is about (Task A: traffic is directional-only); excluded from the
    # plotted spread (kept in the CSV for checking).
    view = per_run[per_run["dataset"] != "traffic"]

    # Per-run dots at this size/opacity were unreadable (7 modes x 9 points,
    # low-alpha, overlapping in a narrow coverage band). Show spread instead as a
    # short, capped vertical whisker (width range) per mode -- the x-spread across
    # runs is much smaller than the y-spread here and added little, so only width
    # (the "lower is better" axis, and the one the paper's claims are about) gets
    # a whisker; this keeps the plot from turning into a thicket of crossing lines.
    fig, ax = plt.subplots(figsize=(COLWIDTH_IN, 3.0))
    for m in modes:
        g = view[view["mode"] == m]
        is_ref = (m == "naive_matched")
        mr = means[means["mode"] == m].iloc[0]
        ylo, yhi = g["shift_width"].min(), g["shift_width"].max()
        ax.errorbar(mr.shift_coverage, mr.shift_width,
                    yerr=[[mr.shift_width - ylo], [yhi - mr.shift_width]],
                    fmt="none", ecolor=COND_COLOR[m], elinewidth=1.0,
                    capsize=2.5, capthick=1.0, alpha=0.55, zorder=2)
        ax.scatter(mr.shift_coverage, mr.shift_width,
                   s=(170 if is_ref else 55),
                   marker=("*" if is_ref else "o"),
                   color=COND_COLOR[m],
                   edgecolor="black", linewidth=(0.9 if is_ref else 0.6),
                   label=(m + "  (reference)" if is_ref else m),
                   zorder=5 if is_ref else 4)

    ax.axvline(0.90, ls=":", lw=0.7, color=OKABE["grey"], zorder=1)
    ax.set_xlabel("shift coverage  (dotted = nominal 0.90)")
    ax.set_ylabel("shift width  (lower is better)")
    # large marker = 4-dataset mean; vertical whisker = min-max shift-width range
    # across the 3 ETTh1/ETTh2/electricity runs (traffic off-scale, see caption).
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=3, handlelength=1.0, columnspacing=0.9, labelspacing=0.3,
              handletextpad=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_taskg_coverage_width.pdf")


# ===========================================================================
# Figure 5 -- SLCP local-variance ratios, temporal-exclusion corrected re-test
#   source: SLCP_RESULT.md section 2 + results/v2/slcp_gate/SLCP_gate_temporal_excluded.csv
#     mean ratio vs dense:  mvdf 0.943   correlated 0.805   random_k 0.730
#     cells clearing the pre-committed gate (ratio<0.80 AND CI-excl-1 vs BOTH
#     dense and pca_k), of 6:  mvdf 2   correlated 2   random_k 4
#     (matches results/v2/slcp_gate/SLCP_gate_decision.json temporal_exclusion_check)
# ===========================================================================
def fig_slcp():
    f = os.path.join(ROOT, "results", "v2", "slcp_gate", "SLCP_gate_temporal_excluded.csv")
    df = pd.read_csv(f)
    spaces = ["mvdf", "correlated", "random_k"]
    df = df[df["space"].isin(spaces)].copy()

    def both_pass(row):
        return bool(row["vs_dense_lt80_and_ci_excl_1"]) and \
               bool(row["vs_pca_lt80_and_ci_excl_1"])
    df["gate_pass"] = df.apply(both_pass, axis=1)

    summ = (df.groupby("space")
              .agg(mean_ratio=("ratio_vs_dense", "mean"),
                   cells_pass=("gate_pass", "sum"),
                   n_cells=("gate_pass", "size"))
              .reindex(spaces).reset_index())
    _write_csv(df[["dataset", "seed", "space", "ratio_vs_dense", "gate_pass"]],
               "fig_slcp_ratios.csv")
    for _, r in summ.iterrows():
        print(f"  [check] {r['space']:11s} mean ratio {r.mean_ratio:.3f}  "
              f"cells clearing gate {int(r.cells_pass)}/{int(r.n_cells)}")

    x = np.arange(len(spaces))
    tallest = df["ratio_vs_dense"].max()
    ytop = max(1.15, tallest * 1.30)
    fig, ax = plt.subplots(figsize=(COLWIDTH_IN, 2.9))
    for i, sp in enumerate(spaces):
        vals = df[df["space"] == sp]["ratio_vs_dense"].values
        mean = vals.mean()
        callout = (sp == "random_k")  # the space that beats mvdf -- call it out
        ax.bar(i, mean, 0.6,
               color=COND_COLOR[sp],
               edgecolor=("black" if callout else "white"),
               linewidth=(1.3 if callout else 0.4),
               zorder=2)
        ax.scatter(np.full_like(vals, i, dtype=float), vals, s=9, color="black",
                   zorder=3, linewidths=0)
        ax.plot([i, i], [vals.min(), vals.max()], color="black", lw=0.7, zorder=3)
        cp = int(summ[summ["space"] == sp]["cells_pass"].iloc[0])
        ax.text(i, max(vals.max(), mean) + 0.035, f"{cp}/6 cells\nclear gate",
                ha="center", va="bottom", fontsize=6, linespacing=1.05)

    ax.axhline(0.80, ls="--", lw=0.9, color="black", zorder=1)
    ax.axhline(1.0, ls=":", lw=0.6, color=OKABE["grey"], zorder=1)
    # "gate 0.80" just above the dashed line, out in the right-hand gutter (no bar there)
    ax.text(len(spaces) - 0.05, 0.82, "gate 0.80", ha="left", va="bottom",
            fontsize=6, color="black")
    # callout parked in the right gutter at mid-height, straight arrow into the
    # random_k bar body -- clear of every bar and every per-bar label
    rk_mean = float(summ[summ.space == "random_k"]["mean_ratio"].iloc[0])
    ax.annotate("random_k\noutperforms\nmvdf",
                xy=(2.32, rk_mean * 0.6), xytext=(3.02, rk_mean * 0.62),
                fontsize=6, ha="center", va="center", linespacing=1.1,
                arrowprops=dict(arrowstyle="->", lw=0.6, color="black"))
    ax.set_xticks(x)
    ax.set_xticklabels(spaces)
    ax.set_ylabel(r"local var($S_t$) ratio vs dense")
    ax.set_xlim(-0.55, len(spaces) + 1.15)
    ax.set_ylim(0, ytop)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_slcp_ratios.pdf")


if __name__ == "__main__":
    fig_h1c()
    fig_phase3()
    fig_oracle()
    fig_taskg()
    fig_slcp()
    print("\nAll five PDFs + checkable CSVs written to", OUT)
