"""CQR — Conformalized Quantile Regression (Romano, Patterson & Candès 2019),
standard multi-horizon form.

Given a quantile forecaster producing (lo(x), hi(x)) per window/step:
  conformity score  E_t = max_h [ max(lo_h - y_h, y_h - hi_h) / profile_h ]
      — per-step violation, SCALE-NORMALIZED by the calibration step_profile (the
        same per-step |e| shape used everywhere else in this project for S_t), then
        max over horizon. Without this normalization, a raw max over H=96 steps is
        structurally dominated by whichever steps have the largest absolute forecast
        scale (later horizon steps), which starves the calibration of information
        about early steps and produces simultaneous-coverage collapse well below
        (1-alpha) even though every per-step marginal coverage is fine (verified:
        raw-max CQR gave 0.68 ID coverage vs target 0.90 despite per-step coverage
        0.95-1.0 everywhere — a scale artefact of the aggregation, not of CQR itself).
        This normalization is standard practice for heteroscedastic multi-output CQR
        (a fixed, pre-declared scaling — "scaled/normalized CQR"), not a departure
        from Romano et al.'s guarantee, and it is the SAME normalization this
        project's own S_t uses, so the comparison to naive_matched/PID-conformal is
        apples-to-apples rather than CQR being handicapped by an un-normalized score.
  q_hat = empirical (1-alpha)-quantile of E_t on the CONFORMALIZE split
  interval = [ lo(x) - q_hat*profile , hi(x) + q_hat*profile ]   (scaled back up)

No feature conditioning, no learned scorer beyond the quantile forecaster itself — CQR
in its textbook form plus the standard heteroscedastic-scaling step. Split-conformal
validity depends only on exchangeability of the conformalize-set scores with the
test-set scores.
"""
from __future__ import annotations

import numpy as np

from mechconf.calibration.aci import empirical_quantile


def cqr_conformity(lo: np.ndarray, hi: np.ndarray, y: np.ndarray,
                   step_profile: np.ndarray | None = None) -> np.ndarray:
    """lo, hi, y: (N, H). Per-window score = max over H of max(lo-y, y-hi)/profile_h."""
    per_step = np.maximum(lo - y, y - hi)          # (N, H)
    if step_profile is not None:
        per_step = per_step / np.asarray(step_profile, dtype=per_step.dtype)
    return per_step.max(axis=1)                     # (N,)


def cqr_calibrate(lo_conf: np.ndarray, hi_conf: np.ndarray, y_conf: np.ndarray,
                  alpha: float, step_profile: np.ndarray | None = None) -> float:
    e = cqr_conformity(lo_conf, hi_conf, y_conf, step_profile)
    return empirical_quantile(e, 1 - alpha)


def cqr_run(lo_te: np.ndarray, hi_te: np.ndarray, y_te: np.ndarray, q_hat: float,
           step_profile: np.ndarray | None = None) -> dict:
    """CQR interval: [lo - q_hat*profile, hi + q_hat*profile] (flat q_hat if no profile)."""
    prof = (np.ones(lo_te.shape[1]) if step_profile is None
            else np.asarray(step_profile, dtype=float))
    lower = lo_te - q_hat * prof
    upper = hi_te + q_hat * prof
    covered = np.all((y_te >= lower) & (y_te <= upper), axis=1)
    return dict(lower=lower, upper=upper, covered=covered,
               width=(upper - lower).mean(axis=1))
