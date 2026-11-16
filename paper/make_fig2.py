"""Figure 2 of the camera-ready paper, regenerated from the committed result CSVs.

(a) Phase-3 best held-out R^2 (marginal battery) per run;
(b) Phase-4 oracle width reduction, in-distribution and shifted, per run.

Sources: results/camera_ready/E2/AGG_E2_phase3.csv (marg_best_r2) and
results/camera_ready/E2/AGG_E2_phase4.csv (ratio_id, ratio_shift); 30 runs
(ETTh1/ETTh2 x 10 seeds, electricity/traffic x 5). Styling is identical to the plotting
code used for the paper (paper/fig2_reference_hardcoded.py): colours #2a78d6 / #eb6834,
3.35 x 3.05 in, serif 7 pt, jitter RNG seed 0.

    python -m paper.make_fig2 [--out paper/figures/fig_dissociation_col.pdf]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mechconf.paths import REPO_ROOT  # noqa: E402

E2 = os.path.join(REPO_ROOT, "results", "camera_ready", "E2")
DATASETS = ["ETTh1", "ETTh2", "electricity", "traffic"]


def load_values():
    p3 = pd.read_csv(os.path.join(E2, "AGG_E2_phase3.csv")).sort_values(["dataset", "seed"])
    p4 = pd.read_csv(os.path.join(E2, "AGG_E2_phase4.csv")).sort_values(["dataset", "seed"])
    r2 = {d: p3[p3.dataset == d].marg_best_r2.tolist() for d in DATASETS}
    oid = {d: p4[p4.dataset == d].ratio_id.tolist() for d in DATASETS}
    osh = {d: p4[p4.dataset == d].ratio_shift.tolist() for d in DATASETS}
    return r2, oid, osh


def draw(r2, oid, osh, out):
    ds = list(r2); y = np.arange(len(ds))[::-1]
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    ink = "#0b0b0b"; mute = "#52514e"; B = "#2a78d6"; O = "#eb6834"
    fig, ax = plt.subplots(2, 1, figsize=(3.35, 3.05), gridspec_kw={"hspace": 0.75})
    rng = np.random.default_rng(0)
    for i, d in enumerate(ds):
        v = np.array(r2[d])
        ax[0].scatter(v, y[i] + rng.uniform(-.12, .12, len(v)), s=14, color=B, edgecolor="white", linewidth=0.5, zorder=3)
        ax[0].plot([np.median(v)] * 2, [y[i] - .28, y[i] + .28], color=ink, lw=1.2, zorder=4)
        a = np.array(oid[d]) * 100; b = np.array(osh[d]) * 100
        ax[1].scatter(a, y[i] + 0.14 + rng.uniform(-.06, .06, len(a)), s=14, color=B, edgecolor="white", linewidth=0.5,
                      zorder=3, label="in-distribution" if i == 0 else None)
        ax[1].scatter(b, y[i] - 0.14 + rng.uniform(-.06, .06, len(b)), s=14, marker="s", facecolor="white", edgecolor=O,
                      linewidth=1.0, zorder=3, label="shifted" if i == 0 else None)
    ax[0].axvline(0, color=mute, lw=0.8, ls=(0, (3, 2)))
    ax[0].text(-0.03, y[0] + 0.38, "constant\npredictor", ha="right", va="top", fontsize=6.5, color=mute)
    ax[0].set_xlim(-1.45, 0.12); ax[1].set_xlim(0, 45)
    ax[0].set_xlabel("best held-out $R^2$ (one dot per run; bar = median)")
    ax[1].set_xlabel("oracle width reduction vs. constant band (%)")
    [a_.set_yticks(y) for a_ in ax]; [a_.set_yticklabels(ds) for a_ in ax]; ax[1].set_ylim(-0.6, 3.6)
    ax[0].set_title("(a) natural predictive information", fontsize=7.5, loc="left")
    ax[1].set_title("(b) available difficulty variation (upper bound)", fontsize=7.5, loc="left")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False); a_.grid(axis="x", color="#e6e5e0", lw=0.5); a_.set_axisbelow(True)
        a_.tick_params(colors=ink, length=2)
    ax[1].legend(frameon=False, loc="upper left", fontsize=6.5, handletextpad=0.2, borderaxespad=0.1)
    ax[0].set_ylim(-0.6, 3.6)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})  # byte-reproducible
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "paper", "figures", "fig_dissociation_col.pdf"))
    a = ap.parse_args()
    print("wrote", draw(*load_values(), a.out))


if __name__ == "__main__":
    main()
