"""Variant B — Mondrian bins on a chosen difficulty feature + per-bin ACI correction
(spec §5.5, the paper's calibration-level contribution).

Given a scalar difficulty signal z_t = a_{j*}(h_t) (the selected feature's activation,
or an aggregate over a small feature set):

  1. Partition into K bins by calibration-set quantiles (balanced).
  2. Each bin gets its own static base quantile q_k of |e| over its calibration points.
  3. Each bin gets its own online ACI correction a_{k,t}, updated ONLY on steps whose
     z_t falls in bin k (spec §5.5 pseudocode; FFDCI eq. 11 scoped per bin).
  4. Interval at t:  C_t = yhat_t +/- ( q_{k(t)} + a_{k(t),t} ).

Static Mondrian bins are NOT shift-robust (spec §5.6); the per-bin ACI term is what
restores approximate bin-conditional validity under shift. We also expose a
`static_only` mode to demonstrate that failure.

`predict_horizon_interval` builds a per-step interval for the whole horizon by scaling
the scalar half-width with the calibration set's per-step error profile — so the
interval is genuinely multi-step, not a single number broadcast.
"""
from __future__ import annotations

import numpy as np

from .aci import empirical_quantile


class MondrianPerBinACI:
    def __init__(
        self,
        n_bins: int = 3,
        alpha_target: float = 0.1,
        gamma: float = 0.05,
        static_only: bool = False,
        clip_mult: float = 20.0,
        burn_in: int = 0,
    ):
        self.K = n_bins
        self.alpha_target = alpha_target
        self.gamma = gamma
        self.static_only = static_only
        self.clip_mult = clip_mult
        self.burn_in = burn_in           # steps before a bin's coverage counts in metrics
        self.fitted = False

    # ---- fit on calibration set --------------------------------------------------
    def fit(self, z_cal: np.ndarray, score_cal: np.ndarray, perstep_abs_err_cal: np.ndarray | None = None):
        """
        z_cal              : (Ncal,)  difficulty signal on calibration windows
        score_cal          : (Ncal,)  scalar nonconformity score |e|
        perstep_abs_err_cal: (Ncal, H) optional per-step |e| for horizon shaping
        """
        z_cal = np.asarray(z_cal, dtype=np.float64)
        qs = np.linspace(0, 1, self.K + 1)[1:-1]
        self.edges = np.quantile(z_cal, qs) if self.K > 1 else np.array([])
        bins = self._assign(z_cal)

        self.q = np.zeros(self.K)
        self.n_per_bin = np.zeros(self.K, dtype=int)
        self.global_q = empirical_quantile(score_cal, 1 - self.alpha_target)
        for k in range(self.K):
            m = bins == k
            self.n_per_bin[k] = int(m.sum())
            self.q[k] = (
                empirical_quantile(score_cal[m], 1 - self.alpha_target)
                if m.sum() >= 5 else self.global_q
            )
        self.clip = self.clip_mult * self.global_q

        if perstep_abs_err_cal is not None:
            profile = perstep_abs_err_cal.mean(axis=0)
            self.step_profile = profile / (profile.mean() + 1e-12)   # (H,), mean 1
        else:
            self.step_profile = None

        # online state
        self.a = np.zeros(self.K)
        self.alpha = np.full(self.K, self.alpha_target)
        self._bin_seen = np.zeros(self.K, dtype=int)
        self.trace = []
        self.fitted = True
        return self

    def _assign(self, z: np.ndarray) -> np.ndarray:
        if self.K == 1:
            return np.zeros(len(z), dtype=int)
        return np.digitize(z, self.edges).astype(int)

    # ---- online deployment -----------------------------------------------------
    def half_width(self, z_t: float) -> tuple[float, int]:
        k = int(self._assign(np.array([z_t]))[0])
        hw = self.q[k] + (0.0 if self.static_only else self.a[k])
        hw = float(np.clip(hw, 0.0, self.clip))
        return hw, k

    def step(self, z_t: float, yhat_t: np.ndarray, y_t: np.ndarray):
        """Return (lower, upper) for the full horizon and update the bin's ACI term.

        The half-width `hw` multiplies a fixed per-step band shape (`step_profile`).
        A window is COVERED iff  max_h |e_h| / profile_h <= hw  — the exact quantity
        the calibration quantile q_k targets, so in-distribution coverage lands at
        1-alpha by construction.
        """
        hw, k = self.half_width(z_t)
        prof = self.step_profile if self.step_profile is not None else np.ones_like(yhat_t)
        hw_vec = hw * prof
        lower, upper = yhat_t - hw_vec, yhat_t + hw_vec

        ae = np.abs(y_t - yhat_t)
        window_score = float(np.max(ae / prof))
        covered = window_score <= hw

        if not self.static_only:
            err = 0.0 if covered else 1.0
            self.alpha[k] += self.gamma * (self.alpha_target - err)
            # move the correction in units of the bin's own scale
            self.a[k] += self.gamma * (err - self.alpha_target) * max(self.q[k], 1e-6)
            self.a[k] = float(np.clip(self.a[k], -0.9 * self.q[k], self.clip))

        self._bin_seen[k] += 1
        self.trace.append(dict(k=k, hw=hw, covered=covered, a_k=float(self.a[k]),
                               alpha_k=float(self.alpha[k]),
                               counts=self._bin_seen[k]))
        return lower, upper, covered, k

    def run(self, z: np.ndarray, yhat: np.ndarray, y: np.ndarray) -> dict:
        """Full online pass over a test stream.

        z    : (N,)
        yhat : (N, H)  target channel
        y    : (N, H)
        """
        assert self.fitted
        N, H = yhat.shape
        lo = np.empty((N, H)); hi = np.empty((N, H))
        cov = np.empty(N, dtype=bool); binid = np.empty(N, dtype=int)
        for t in range(N):
            l, u, c, k = self.step(float(z[t]), yhat[t], y[t])
            lo[t], hi[t], cov[t], binid[t] = l, u, c, k
        width = (hi - lo).mean(axis=1)
        return dict(lower=lo, upper=hi, covered=cov, bin=binid, width=width,
                    n_per_bin_cal=self.n_per_bin.copy(),
                    timesteps_per_bin=np.bincount(binid, minlength=self.K))
