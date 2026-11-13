"""Shared loading + statistical machinery for the paper-126 follow-up (brief §5)."""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARR = os.environ.get("P126_ARR", os.path.join(HERE, "runs", "p126", "arrays"))
OUT = os.path.join(HERE, "runs", "p126")
ALPHA = 0.10
REAL = [(d, s) for d in ("ETTh1", "ETTh2") for s in (0, 1, 2)]
SYN = [("synth", s) for s in (0, 1, 2)]


# ------------------------------------------------------------------ loading
def profile_of(abs_err):
    p = abs_err.mean(0)
    return p / (p.mean() + 1e-12)


def score_with(abs_err, prof):
    return (abs_err / prof[None, :]).max(1)


class Run:
    """All test windows of one (dataset, seed). Index t = window index in the test split
    (stride 1), forecast origin = end_idx[t]; S_t observable at end_idx[t] + H."""

    def __init__(self, ds, seed):
        tag = f"{ds}_seed{seed}"
        self.ds, self.seed, self.tag = ds, seed, tag
        a = np.load(os.path.join(ARR, f"{tag}.npz"))
        self.a = a
        for k in a.files:
            setattr(self, k, a[k])
        self.L, self.H = int(a["L"]), int(a["H"])
        self.abs_err = np.abs(self.y - self.yhat).astype(np.float64)
        self.idx_sel = np.where(self.m_select)[0]
        self.idx_id = np.where(self.m_id)[0]
        self.idx_sh = np.where(self.m_shift)[0]
        # calibration profile: E_select (the paper's "calibration set", cal_sel in phase 3)
        self.prof_sel = profile_of(self.abs_err[self.idx_sel])
        self.S = score_with(self.abs_err, self.prof_sel)          # consistent S_t, all windows

    def z_cols(self, cols):
        cols = np.asarray(cols, int)
        out = np.zeros((len(self.z_idx), len(cols)), np.float64)
        for j, c in enumerate(cols):
            hit = self.z_idx == c
            out[:, j] = (self.z_val * hit).sum(1)
        return out

    def S_local(self, idx):
        """The ORIGINAL code's S_t for a block: validation.errors.collect() normalises by the
        profile of whatever block it is handed."""
        ae = self.abs_err[idx]
        return score_with(ae, profile_of(ae)), profile_of(ae)


def load(ds, seed):
    return Run(ds, seed)


# ------------------------------------------------------------------ statistics
def block_bootstrap(n, block, n_boot=2000, seed=0):
    """Moving-block bootstrap index sets (n_boot, n)."""
    rng = np.random.default_rng(seed)
    block = max(1, min(block, n))
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return idx


def boot_ci(stat_fn, n, block, n_boot=2000, seed=0):
    """Percentile 95% CI of stat_fn(idx) over moving-block resamples. Returns (lo, hi, n_blocks)."""
    idxs = block_bootstrap(n, block, n_boot, seed)
    vals = np.array([stat_fn(ix) for ix in idxs], float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, n / block
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), n / block


def circ_null(stat_fn, x, y, min_shift, n_perm=1000, seed=0):
    """Circular-shift null: stat_fn(roll(x, k), y) for random k in [min_shift, n-min_shift].
    Returns (null array, two-sided p for |stat|). nan if segment too short."""
    n = len(x)
    if n - 2 * min_shift < 10:
        return np.array([]), np.nan
    rng = np.random.default_rng(seed)
    ks = rng.integers(min_shift, n - min_shift, size=n_perm)
    null = np.array([stat_fn(np.roll(x, k), y) for k in ks], float)
    obs = stat_fn(x, y)
    p = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + len(null))
    return null, float(p)


def acf(x, maxlag):
    x = np.asarray(x, float) - np.mean(x)
    v = np.dot(x, x) / len(x)
    if v <= 0:
        return np.zeros(maxlag + 1)
    return np.array([1.0] + [np.dot(x[:-k], x[k:]) / len(x) / v for k in range(1, maxlag + 1)])


def n_eff(c):
    """n / (1 + 2 sum rho_k), sum truncated at the first non-positive autocorrelation
    (initial positive sequence). For a 0/1 coverage series."""
    c = np.asarray(c, float)
    n = len(c)
    if n < 3 or np.var(c) == 0:
        return float(n)
    r = acf(c, min(n - 2, n // 2))
    s = 0.0
    for k in range(1, len(r)):
        if r[k] <= 0:
            break
        s += r[k]
    return float(n / (1 + 2 * s))


def spearman(a, b):
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return 0.0
    return float(stats.spearmanr(a, b).statistic)


def kendall(a, b):
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return 0.0
    return float(stats.kendalltau(a, b, variant="b").statistic)


def auroc(score, label):
    """Tie-aware (Mann-Whitney with average ranks) AUROC."""
    label = np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def conf_q(scores, level):
    """Finite-sample split-conformal quantile: ceil((n+1)level)-th order statistic, +inf if
    that rank exceeds n."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    r = int(np.ceil((n + 1) * level))
    if r > n:
        return np.inf
    return float(s[max(r, 1) - 1])


def matched_width(hw_fn, S_test, target_cov, grid=None):
    """hw_fn(alpha) -> half-widths on test. Find alpha with empirical coverage closest to
    target_cov (linear interp on the coverage/width curve). Returns (width, alpha, cov)."""
    grid = np.linspace(0.005, 0.40, 159) if grid is None else grid
    covs, ws = [], []
    for al in grid:
        hw = hw_fn(al)
        covs.append(np.mean(S_test <= hw))
        ws.append(np.mean(2 * np.minimum(hw, 1e9)))
    covs, ws = np.array(covs), np.array(ws)
    # coverage decreasing in alpha; interpolate width at target coverage
    o = np.argsort(covs)
    c_s, w_s, a_s = covs[o], ws[o], grid[o]
    if target_cov <= c_s[0]:
        return float(w_s[0]), float(a_s[0]), float(c_s[0])
    if target_cov >= c_s[-1]:
        return float(w_s[-1]), float(a_s[-1]), float(c_s[-1])
    w = float(np.interp(target_cov, c_s, w_s))
    a = float(np.interp(target_cov, c_s, a_s))
    return w, a, float(target_cov)


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2, default=float)
