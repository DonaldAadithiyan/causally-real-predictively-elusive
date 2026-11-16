"""Correlation screening (spec §5.4).

    s_j = Corr(z_j(x_t), |e_t|)   over the calibration set

Rank features by |s_j|. A feature is a 'correlated candidate' if it is active often
enough for the correlation to mean anything (min activation rate) — a feature that
fires on 3 windows can show |corr| ~ 1 by chance.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def screen(z: np.ndarray, score: np.ndarray, min_active_frac: float = 0.02) -> dict:
    """
    z     : (N, d_sae)  sparse codes on the screening set
    score : (N,)        nonconformity score |e|

    Returns dict with per-feature arrays (length d_sae):
      pearson, spearman, active_frac, eligible (bool), rank_by_abs_pearson
    """
    N, d = z.shape
    active_frac = (z != 0).mean(axis=0)
    eligible = active_frac >= min_active_frac

    pear = np.zeros(d)
    spear = np.zeros(d)
    s = score.astype(np.float64)
    for j in range(d):
        zj = z[:, j].astype(np.float64)
        if not eligible[j] or np.std(zj) < 1e-12:
            continue
        pear[j] = np.corrcoef(zj, s)[0, 1]
        spear[j] = stats.spearmanr(zj, s).statistic

    order = np.argsort(-np.abs(pear))
    rank = np.empty(d, dtype=int)
    rank[order] = np.arange(d)
    return dict(
        pearson=pear, spearman=spear, active_frac=active_frac,
        eligible=eligible, rank_by_abs_pearson=rank, order=order,
    )


def top_k_correlated(screen_result: dict, k: int) -> np.ndarray:
    """Indices of the top-k eligible features by |pearson|."""
    pear = np.abs(screen_result["pearson"]).copy()
    pear[~screen_result["eligible"]] = -np.inf
    return np.argsort(-pear)[:k]
