"""Round 2, WP0 — corrected evaluation path. NEW functions only; originals untouched.

Corrections relative to the original code paths:
  C1 delayed feedback   online updates (ACI, PID) at time t use only windows whose full
                        horizon is observed: miss indicator of window t enters the update
                        at t + H (`delay`), never earlier.
                        replaces: calibration/difficulty_scorer.py::naive_no_scorer_run,
                        calibration/ffdci_head.py::FFDCIHeadACI.run,
                        baselines/pid_conformal.py::pid_conformal_run (immediate update).
  C2 no half-width cap  the scorer's ACI clip `20*max(q_resid,1e-6)` on the TOTAL half-width
                        is gone (clip=None by default), q_resid is used signed, and the ACI
                        step is scaled by the median calibration half-width, not by q_resid.
                        replaces: calibration/difficulty_scorer.py::DifficultyScorer.run.
  C3 one profile        profile[h] is computed once on calibration data (E_select, which
                        precedes every evaluation block) and applied unchanged to every
                        block. replaces: validation/errors.py::collect's per-block profile.
Everything is expressed in S_t units: window t is covered iff S_t <= half_width_t, which
is exactly the joint-over-horizon interval check (verified in round 1).
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.round2.common import conf_q, profile_of, score_with


# ------------------------------------------------------------------ C3
def calib_profile(abs_err_cal):
    return profile_of(abs_err_cal)


def scores(abs_err, prof):
    return score_with(abs_err, prof)


# ------------------------------------------------------------------ C1 / C2 online runners
def online(base_hw, S, alpha, gamma=0.03, delay=96, scale=None, kind="aci", Kp=1.0,
           clip=None):
    """Delayed-feedback online correction on top of a per-window base half-width.

    kind="aci": the repo's GlobalACI additive form a <- a + gamma*(err-alpha)*scale.
    kind="pid": the repo's PID-conformal P+I form on an additive offset
                q <- q + eta*(err-alpha) + Kp*eta*((err-alpha) - prev)   (eta = gamma).
    kind="static": no update.
    The update applied before issuing window t uses err_{t-delay} only.
    Returns dict(covered, hw, offset)."""
    base_hw = np.broadcast_to(np.asarray(base_hw, float), S.shape).astype(float)
    n = len(S)
    if scale is None:
        scale = float(np.median(base_hw))
    off = 0.0
    prev = None
    cov = np.empty(n, bool); hw = np.empty(n); offs = np.empty(n)
    for t in range(n):
        if kind != "static" and t - delay >= 0:
            err = 0.0 if cov[t - delay] else 1.0
            integ = err - alpha
            if kind == "aci":
                off += gamma * integ * scale
            elif kind == "pid":
                off += gamma * integ
                if prev is not None:
                    off += Kp * gamma * (integ - prev)
                prev = integ
        h = base_hw[t] + off
        if clip is not None:
            h = min(h, clip)
        h = max(h, 0.0)
        hw[t] = h; offs[t] = off
        cov[t] = S[t] <= h
    return dict(covered=cov, hw=hw, offset=offs)


# ------------------------------------------------------------------ methods (base half-widths)
def naive_base(S_cal, alpha):
    return conf_q(S_cal, 1 - alpha)


class Scorer:
    """Learned-scorer (DifficultyScorer's model + split-conformal step), corrected:
    signed q_resid, no cap. Training uses the original train_quantile_mlp unchanged."""

    def __init__(self, seed, alpha=0.10, epochs=200, lr=1e-3):
        self.seed, self.alpha, self.epochs, self.lr = seed, alpha, epochs, lr

    def fit(self, F, S):
        from mechconf.calibration.ffdci_head import train_quantile_mlp
        self.m = train_quantile_mlp(F.astype(np.float32), S.astype(np.float32),
                                    alpha_target=self.alpha, epochs=self.epochs, lr=self.lr,
                                    seed=self.seed)
        return self

    @torch.no_grad()
    def predict(self, F):
        return self.m(torch.from_numpy(F.astype(np.float32))).numpy().astype(float)

    def conformalize(self, F, S, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        self.resid_cal = S - self.predict(F)
        self.q_resid = conf_q(self.resid_cal, 1 - alpha)
        return self

    def base(self, F, alpha=None):
        q = self.q_resid if alpha is None else conf_q(self.resid_cal, 1 - alpha)
        return self.predict(F) + q


def cqr_scores(lo, hi, y, prof):
    return (np.maximum(lo - y, y - hi) / prof[None, :]).max(1)


def cqr_width(lo, hi, hw_cqr, prof):
    """mean interval width in normalised-y units for CQR with additive hw*prof."""
    return (hi - lo).mean(1) + 2 * hw_cqr * prof.mean()


# ------------------------------------------------------------------ protocols
def chrono_split(n, fracs=(0.5, 0.3, 0.2), emb=192):
    c1, c2 = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
    fit = np.arange(0, c1)
    cal = np.arange(c1 + emb, c2)
    te = np.arange(c2 + emb, n)
    return fit, cal, te


def random_block_split(n, rng, n_blocks=10, roles=(5, 3, 2), emb=192):
    """Random contiguous-block assignment of fit/cal/test roles. Test blocks are kept whole;
    fit/cal windows within `emb` of any test window are dropped, and fit/cal windows within
    emb/2 of a fit<->cal boundary are dropped on both sides (gap = emb everywhere)."""
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    # test region = roles[2] ADJACENT blocks at a random position (keeps the embargo loss
    # to one region); fit/cal roles shuffled over the remaining blocks.
    start = int(rng.integers(0, n_blocks - roles[2] + 1))
    lab = np.empty(n_blocks, int)
    lab[start:start + roles[2]] = 2
    R = np.r_[0:start, start + roles[2]:n_blocks]        # non-test blocks in time order
    cs = int(rng.integers(0, len(R) - roles[1] + 1))       # cal = roles[1] consecutive of R
    lab[R] = 0
    lab[R[cs:cs + roles[1]]] = 1
    role = np.empty(n, int)
    for b in range(n_blocks):
        role[edges[b]:edges[b + 1]] = lab[b]
    idx = np.arange(n)
    te = role == 2
    keep = np.ones(n, bool)
    tpos = idx[te]
    # distance to nearest test window
    d = np.full(n, np.inf)
    if len(tpos):
        j = np.searchsorted(tpos, idx)
        left = np.where(j > 0, idx - tpos[np.clip(j - 1, 0, len(tpos) - 1)], np.inf)
        right = np.where(j < len(tpos), tpos[np.clip(j, 0, len(tpos) - 1)] - idx, np.inf)
        d = np.minimum(np.abs(left), np.abs(right))
    keep &= te | (d > emb)
    half = emb // 2
    for b in range(1, n_blocks):
        if {lab[b], lab[b - 1]} == {0, 1}:
            e = edges[b]
            keep[max(0, e - half):min(n, e + half)] = False
    return idx[(role == 0) & keep], idx[(role == 1) & keep], idx[te]
