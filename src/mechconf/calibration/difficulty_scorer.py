"""V2 spec §4.8, §10 — continuous difficulty scorer `g(z)` under three input conditions.

  g_dense(h_t)                      FFDCI as-is: full raw pre-head token
  g_MVDF(z_t[MVDF_v2_set])          only the mechanistically-validated difficulty subspace
  g_dense+MVDF(h_t, z_t[MVDF_v2])   dense token augmented with the validated subspace

All three: SAME pinball-loss MLP architecture, SAME training budget / early-stopping,
SAME global-ACI wrapper (calibration/ffdci_head.py). Feature set is the ONLY manipulated
variable — the architecture diff is logged as zero (spec §10 acceptance).

The sharpest test in the paper (H6): does g_MVDF or g_dense+MVDF achieve significantly
narrower mean width than g_dense at matched coverage, ID and under shift?
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from .ffdci_head import QuantileMLP, train_quantile_mlp, FFDCIHeadACI
from .aci import empirical_quantile


def build_features(mode: str, hidden: np.ndarray, z: np.ndarray, mvdf_idx: np.ndarray,
                   *, corr_idx: np.ndarray | None = None, rand_idx: np.ndarray | None = None,
                   pca_model=None, k: int | None = None) -> np.ndarray:
    """hidden: (N, d_model) raw pre-head token. z: (N, d_sae) SAE codes.

    Modes (H6 four-way decomposition, spec §4.8 + review):
      dense        FFDCI as-is: the full raw pre-head vector
      mvdf         only the MVDF_v2 SAE-feature subspace  (k dims)
      dense+mvdf   dense vector augmented with the MVDF subspace
      correlated   the top-k correlation-selected SAE features  (k dims)  <- sharpest test:
                   does the whole dose-response/R_j machinery beat the cheap baseline?
      random_k     k random SAE features                         (k dims) <- dimensionality control
      pca_k        top-k PCA components of the dense vector       (k dims) <- sensible-compression control
    """
    if mode == "dense":
        return hidden.astype(np.float32)
    if mode == "mvdf":
        return z[:, mvdf_idx].astype(np.float32)
    if mode == "dense+mvdf":
        return np.concatenate([hidden.astype(np.float32), z[:, mvdf_idx].astype(np.float32)], axis=1)
    if mode == "correlated":
        assert corr_idx is not None
        return z[:, corr_idx].astype(np.float32)
    if mode == "random_k":
        assert rand_idx is not None
        return z[:, rand_idx].astype(np.float32)
    if mode == "pca_k":
        assert pca_model is not None
        return pca_model.transform(hidden.astype(np.float64)).astype(np.float32)
    raise ValueError(mode)


class DifficultyScorer:
    """One (mode) scorer + a SPLIT-CONFORMAL wrapper (proper coverage) with an online
    global-ACI correction on top.

    Coverage was ~85-87% (not 90%) when base_q was the (1-alpha) quantile of the
    MLP-TRAINING scores — a noisy underestimate on a small calibration set. Fixed:
    `fit_conformalize` sets the additive conformal quantile from a HELD-OUT
    conformalization set's residuals  r = score - pred_q(feats)  , so ID coverage is
    ~1-alpha by split-conformal construction, and ACI only adapts it under shift.
    """

    def __init__(self, mode, hidden_dim, mlp_hidden=64):
        self.mode = mode
        self.mlp_hidden = mlp_hidden
        self.model: QuantileMLP | None = None
        self.arch_signature = None
        self._q_resid = 0.0

    def fit(self, feats_train: np.ndarray, score_train: np.ndarray, alpha: float, seed: int,
            epochs: int = 200, lr: float = 1e-3):
        """Train the quantile MLP on (feats_train, score_train)."""
        self.model = train_quantile_mlp(feats_train, score_train, alpha_target=alpha,
                                        epochs=epochs, lr=lr, seed=seed)
        self.arch_signature = dict(in_dim=feats_train.shape[1], hidden=self.mlp_hidden,
                                   epochs=epochs, lr=lr)
        self._alpha = alpha
        return self

    def conformalize(self, feats_conf: np.ndarray, score_conf: np.ndarray):
        """Set the additive conformal quantile from a HELD-OUT set's residuals
        r = score - pred_q(feats).  Split-conformal: P(score <= pred_q + q_resid) >= 1-alpha."""
        import torch
        with torch.no_grad():
            pq = self.model(torch.from_numpy(feats_conf.astype("float32"))).numpy()
        resid = score_conf - pq
        self._q_resid = empirical_quantile(resid, 1 - self._alpha)
        return self

    def run(self, feats_te: np.ndarray, yhat_te: np.ndarray, y_te: np.ndarray,
            step_profile: np.ndarray, alpha: float, gamma: float = 0.03,
            clip_mult: float = 20.0) -> dict:
        # split-conformal: half-width = q_resid + pred_q(x), with the online ACI term
        # on top. center_pred=False so raw pred_q is added (not a deviation).
        runner = FFDCIHeadACI(self.model, base_q=max(self._q_resid, 1e-6),
                              alpha_target=alpha, gamma=gamma, clip_mult=clip_mult,
                              step_profile=step_profile, center_pred=False)
        return runner.run(feats_te, yhat_te, y_te)


def naive_no_scorer_run(score_conf: np.ndarray, yhat_te: np.ndarray, y_te: np.ndarray,
                        step_profile: np.ndarray, alpha: float, gamma: float = 0.03) -> dict:
    """Task G — global split-conformal with a CONSTANT (unconditioned) half-width.
    No learned scorer, no feature input. Half-width = (1-alpha) empirical quantile of
    the conformalization-set scores, adapted online by a global-ACI term. Answers how
    much of the dense->mvdf gap is 'any adaptive scorer beats none' vs feature choice.
    """
    from .aci import GlobalACI
    q = empirical_quantile(score_conf, 1 - alpha)
    aci = GlobalACI(base_q=max(q, 1e-6), alpha_target=alpha, gamma=gamma,
                    clip=20.0 * max(q, 1e-6))
    N, H = yhat_te.shape
    prof = step_profile if step_profile is not None else np.ones(H)
    lo = np.empty((N, H)); hi = np.empty((N, H)); cov = np.empty(N, dtype=bool)
    for t in range(N):
        hw = aci.half_width()
        hw_vec = hw * prof
        lo[t], hi[t] = yhat_te[t] - hw_vec, yhat_te[t] + hw_vec
        ae = np.abs(y_te[t] - yhat_te[t])
        cov[t] = bool(np.max(ae / prof) <= hw)
        aci.update(bool(cov[t]))
    return dict(lower=lo, upper=hi, covered=cov, width=(hi - lo).mean(axis=1))


def paired_width_bootstrap(width_a: np.ndarray, width_b: np.ndarray, n_boot: int = 2000,
                           seed: int = 0) -> dict:
    """Paired bootstrap CI on mean(width_a) - mean(width_b), per-window paired.
    Returns the CI and P(a < b)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(width_a) - np.asarray(width_b)
    n = len(d)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return dict(
        mean_diff=float(d.mean()),
        ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        p_a_lt_b=float((boots < 0).mean()),
        significant=bool(np.percentile(boots, 97.5) < 0 or np.percentile(boots, 2.5) > 0),
    )
