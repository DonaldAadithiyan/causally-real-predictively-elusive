"""Adaptive Conformal Inference primitives (Gibbs & Candes 2021; FFDCI eq. 11).

Global ACI online update:
    alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)
    err_t = 1{ y_t not in C_t }

Interval half-width at time t is the empirical (1 - alpha_t) quantile of the
calibration nonconformity scores, plus (in FFDCI / Variant B) an additive correction
that the update actually moves. We follow the FFDCI additive-correction form: a base
quantile q from the calibration set, plus a scalar a_t updated online.
"""
from __future__ import annotations

import numpy as np


def empirical_quantile(scores: np.ndarray, level: float) -> float:
    """(level)-quantile with the standard conformal +1 correction."""
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(scores)
    if n == 0:
        return 0.0
    rank = int(np.ceil((n + 1) * level))
    rank = min(max(rank, 1), n)
    return float(scores[rank - 1])


class GlobalACI:
    """Single online correction term over the whole stream (FFDCI Variant A tail)."""

    def __init__(self, base_q: float, alpha_target: float = 0.1, gamma: float = 0.01,
                 clip: float | None = None):
        self.base_q = float(base_q)
        self.alpha_target = alpha_target
        self.gamma = gamma
        self.a = 0.0
        self.alpha = alpha_target
        self.clip = clip
        self.history = []

    def half_width(self, extra: float = 0.0) -> float:
        hw = self.base_q + self.a + extra
        if self.clip is not None:
            hw = float(np.clip(hw, 0.0, self.clip))
        return max(hw, 0.0)

    def update(self, covered: bool):
        err = 0.0 if covered else 1.0
        self.alpha = self.alpha + self.gamma * (self.alpha_target - err)
        # keep the correction moving in the intuitive direction: undercover -> widen
        self.a = self.a + self.gamma * (err - self.alpha_target) * max(self.base_q, 1e-6)
        self.history.append(dict(alpha=self.alpha, a=self.a, covered=covered))
