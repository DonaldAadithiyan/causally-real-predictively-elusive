"""Evaluation metrics (spec §6.4).

coverage (overall + worst-step), mean interval width, MACE (rolling-window approx),
Winkler score, timesteps-per-bin. Each reported in-distribution and as a Delta
(degradation) under the pre-registered shift.
"""
from __future__ import annotations

import numpy as np


def coverage(covered_all_steps: np.ndarray) -> float:
    """Fraction of windows fully covered across the whole horizon."""
    return float(np.mean(covered_all_steps))


def stepwise_coverage(lower: np.ndarray, upper: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(H,) coverage per horizon step."""
    inside = (y >= lower) & (y <= upper)
    return inside.mean(axis=0)


def worst_step_coverage(lower, upper, y) -> float:
    return float(stepwise_coverage(lower, upper, y).min())


def mean_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))


def winkler_score(lower: np.ndarray, upper: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> float:
    """Mean Winkler / interval score over all (window, step) pairs. Lower is better."""
    width = upper - lower
    below = y < lower
    above = y > upper
    penalty = np.where(below, (2 / alpha) * (lower - y), 0.0) + \
              np.where(above, (2 / alpha) * (y - upper), 0.0)
    return float(np.mean(width + penalty))


def mace(covered_all_steps: np.ndarray, alpha: float = 0.1, window: int = 100) -> float:
    """Mean Absolute Coverage Error via a rolling local-coverage window (FFDCI's
    local approximation). |rolling_coverage - (1 - alpha)| averaged over time."""
    target = 1 - alpha
    c = covered_all_steps.astype(float)
    if len(c) < window:
        return float(abs(c.mean() - target))
    kernel = np.ones(window) / window
    roll = np.convolve(c, kernel, mode="valid")
    return float(np.mean(np.abs(roll - target)))


def summarize(result: dict, y: np.ndarray, alpha: float = 0.1, mask: np.ndarray | None = None) -> dict:
    """result: dict with lower/upper/covered/width from a calibration runner.
    y        : (N, H) target-channel truth
    mask     : optional bool (N,) to restrict to a sub-window (e.g. shift block)
    """
    lo, up, cov = result["lower"], result["upper"], result["covered"]
    if mask is not None:
        lo, up, cov, y = lo[mask], up[mask], cov[mask], y[mask]
    return dict(
        n=int(len(cov)),
        coverage=coverage(cov),
        worst_step_coverage=worst_step_coverage(lo, up, y),
        mean_width=mean_width(lo, up),
        winkler=winkler_score(lo, up, y, alpha),
        mace=mace(cov, alpha),
    )


def degradation(in_dist: dict, shift: dict) -> dict:
    """Delta = in_distribution - shift  for the 'good direction' quantities, so a
    positive number always means 'got worse under shift'."""
    return dict(
        d_coverage=in_dist["coverage"] - shift["coverage"],          # coverage dropped
        d_worst_step_coverage=in_dist["worst_step_coverage"] - shift["worst_step_coverage"],
        d_width=shift["mean_width"] - in_dist["mean_width"],          # width grew
        d_winkler=shift["winkler"] - in_dist["winkler"],             # winkler grew
        d_mace=shift["mace"] - in_dist["mace"],                      # mace grew
    )
