"""PID-conformal (Angelopoulos, Candès & Tibshirani 2023, "Conformal PID Control for
Time Series"), standard form.

The paper's online quantile-tracking update (their eq. 5 / Algorithm 1, P+I terms; we
omit their optional Scorecaster term — a *learned* residual-quantile predictor — because
including it would reintroduce exactly the representation-conditioning question this
project is trying to keep OUT of the baseline. PID-conformal without the Scorecaster is
still the paper's own P+I "vanilla" variant, a citable, standard baseline on its own):

  q_t = q_{t-1} + eta * ( err_{t-1} - alpha )                         (integral, I)
      + Kp * ( err_{t-1} - alpha )                                     (proportional, P)

  err_t = 1{ y_t not covered by [yhat_t - q_t, yhat_t + q_t] }

No feature conditioning, no learned scorer, no representation input beyond the point
forecaster's own residual stream — raw residuals only, exactly like `naive_matched`
but with a PID (not a single-gain) online update, per the paper's own control-theoretic
form. Multi-horizon: same max-over-H nonconformity convention as everywhere else in this
project (`S_t = max_h |e_h| / profile_h`), so `q_t` is a scalar half-width multiplier on
the step_profile shape, consistent with `naive_matched`'s interval construction — the
only axis that varies between them is the ONLINE UPDATE RULE (PID vs single-gain ACI).
"""
from __future__ import annotations

import numpy as np


class PIDConformal:
    """Angelopoulos et al. 2023, P+I terms (no Scorecaster). q_t tracked online as a
    half-width multiplier on the (project-standard) max-over-H score."""

    def __init__(self, q0: float, alpha_target: float = 0.10, Kp: float = 1.0,
                eta: float = 0.01, clip: float | None = None):
        self.q = float(q0)
        self.alpha_target = alpha_target
        self.Kp = Kp
        self.eta = eta
        self.clip = clip
        self._prev_err = None  # err_{t-1} - alpha, for the P term

    def half_width(self) -> float:
        q = self.q
        if self.clip is not None:
            q = float(np.clip(q, 0.0, self.clip))
        return max(q, 0.0)

    def update(self, covered: bool):
        err = 0.0 if covered else 1.0
        integrand = err - self.alpha_target
        # integral term (accumulates via q itself, standard PID-conformal form)
        self.q = self.q + self.eta * integrand
        # proportional term: nudge q toward compensating the LATEST miss/hit directly
        if self._prev_err is not None:
            self.q = self.q + self.Kp * self.eta * (integrand - self._prev_err)
        self._prev_err = integrand


def pid_conformal_run(q0: float, yhat_te: np.ndarray, y_te: np.ndarray,
                      step_profile: np.ndarray, alpha: float,
                      Kp: float = 1.0, eta: float = 0.03, clip_mult: float = 20.0) -> dict:
    """Same interface/shape as calibration/difficulty_scorer.py::naive_no_scorer_run,
    so it drops into the exact same eval/consolidate pipeline. q0 = the (1-alpha)
    empirical quantile of the conformalize-set scores (same seeding convention as every
    other mode in this project)."""
    pid = PIDConformal(q0=q0, alpha_target=alpha, Kp=Kp, eta=eta,
                       clip=clip_mult * max(q0, 1e-6))
    N, H = yhat_te.shape
    prof = step_profile if step_profile is not None else np.ones(H)
    lo = np.empty((N, H)); hi = np.empty((N, H)); cov = np.empty(N, dtype=bool)
    for t in range(N):
        hw = pid.half_width()
        hw_vec = hw * prof
        lo[t], hi[t] = yhat_te[t] - hw_vec, yhat_te[t] + hw_vec
        ae = np.abs(y_te[t] - yhat_te[t])
        cov[t] = bool(np.max(ae / prof) <= hw)
        pid.update(bool(cov[t]))
    return dict(lower=lo, upper=hi, covered=cov, width=(hi - lo).mean(axis=1))
