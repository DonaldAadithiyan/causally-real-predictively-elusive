"""Plots (spec §10.5): local coverage / width over deployment time per condition,
across the shift boundary; the Delta_j distribution; the RQ1 rank scatter."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _roll(x, w=100):
    x = np.asarray(x, float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def coverage_width_over_time(results: dict, shift_start: int, alpha: float, out_path: str, title=""):
    """results: {condition_name: runner_result_dict}"""
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for name, r in results.items():
        ax[0].plot(_roll(r["covered"]), label=name, lw=1.3)
        ax[1].plot(_roll(r["width"]), label=name, lw=1.3)
    ax[0].axhline(1 - alpha, color="k", ls="--", lw=0.8, label="target")
    for a in ax:
        a.axvline(shift_start, color="crimson", ls=":", lw=1.2)
    ax[0].set_ylabel("rolling coverage"); ax[1].set_ylabel("rolling width")
    ax[1].set_xlabel("deployment step (rolling window)")
    ax[0].set_title(title); ax[0].legend(fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def delta_j_distribution(abs_delta: np.ndarray, out_path: str):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].hist(abs_delta, bins=50, color="steelblue")
    ax[0].set_title("|Delta_j| distribution"); ax[0].set_xlabel("|Delta_j|")
    sd = np.sort(abs_delta)[::-1]
    ax[1].plot(sd, marker=".", ms=3)
    ax[1].set_yscale("log"); ax[1].set_title("|Delta_j| ranked (log y)")
    ax[1].set_xlabel("rank")
    if len(sd) > 1 and np.median(sd) > 0:
        ax[1].text(0.5, 0.9, f"max/median = {sd[0] / np.median(sd):.1f}x",
                   transform=ax[1].transAxes, fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def rq1_scatter(abs_corr: np.ndarray, abs_delta: np.ndarray, out_path: str, rho: float):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(abs_corr, abs_delta, s=12, alpha=0.6)
    ax.set_xlabel("|Corr(z_j, |e|)|  (screening)")
    ax.set_ylabel("|Delta_j|  (ablation magnitude)")
    ax.set_title(f"RQ1: correlation vs ablation ranking  (Spearman rho = {rho:.2f})")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
