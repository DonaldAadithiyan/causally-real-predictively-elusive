"""E3 ex-ante feature sets. Window index t <-> origin end_idx[t] (stride 1).

Window t' forecasts target times end_idx[t'] + 1 + h, h = 0..H-1. A feature for row t may
use a realised error e_{t',h} only if end_idx[t'] + 1 + h <= end_idx[t].
"""
from __future__ import annotations

import numpy as np

F1_LAGS = (96, 97, 120, 144, 192)
F1_ROLL = (24, 96, 336)
F1P_LAGS = (1, 24, 48, 72)
F1P_ONESTEP = 24
MIN_T = 96 + 336            # first row with every lag defined


def f1_lagged_full(S, rows):
    """S_{t-d} for d in F1_LAGS + rolling mean/max/median of S over origins [t-96-k, t-96]."""
    out, names = [], []
    for d in F1_LAGS:
        out.append(S[rows - d]); names.append(f"S_lag{d}")
    for k in F1_ROLL:
        win = np.stack([S[r - 96 - k: r - 96 + 1] for r in rows])
        out += [win.mean(1), win.max(1), np.median(win, 1)]
        names += [f"S_roll{k}_mean", f"S_roll{k}_max", f"S_roll{k}_median"]
    return np.column_stack(out), names


def f1p_partial(abs_err, prof, rows):
    """Observed prefixes of in-flight windows + realised one-step-ahead errors."""
    norm = abs_err / prof[None, :]
    out, names = [], []
    for m in F1P_LAGS:
        # window t-m: steps h < m have target time end_idx[t]-m+1+h <= end_idx[t]
        out.append(norm[rows - m, :m].max(1)); names.append(f"Spartial_lag{m}")
    for i in range(1, F1P_ONESTEP + 1):
        out.append(norm[rows - i, 0]); names.append(f"onestep_err_lag{i}")
    return np.column_stack(out), names


def _spec_entropy(x):
    """x: (n, L, V). Normalised spectral entropy per channel (DC removed)."""
    xc = x - x.mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(xc, axis=1)) ** 2
    P = P[:, 1:, :]
    s = P.sum(1, keepdims=True)
    p = np.where(s > 0, P / np.where(s > 0, s, 1), 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0), axis=1)
    return h / np.log(P.shape[1])


def f2_input_stats(X, rows):
    x = X[rows].astype(np.float64)                     # (n, L, V)
    L = x.shape[1]
    dx = np.diff(x, axis=1)
    tt = np.arange(L) - (L - 1) / 2
    slope = (x * tt[None, :, None]).sum(1) / (tt ** 2).sum()
    xc = x - x.mean(1, keepdims=True)
    v = (xc ** 2).sum(1)
    ac1 = np.where(v > 0, (xc[:, 1:] * xc[:, :-1]).sum(1) / np.where(v > 0, v, 1), 0)
    stats = dict(
        std=x.std(1), range=x.max(1) - x.min(1), mean_abs_diff=np.abs(dx).mean(1),
        std_diff=dx.std(1), slope=slope, last_minus_mean=x[:, -1] - x.mean(1),
        acf1=ac1, spec_entropy=_spec_entropy(x),
        revin_scale=x.std(1, ddof=1) + 1e-6,          # torch .std() is unbiased: the model's scale
    )
    out, names = [], []
    for k, v_ in stats.items():
        out += [v_.mean(1), v_.max(1)]
        names += [f"{k}_chmean", f"{k}_chmax"]
    return np.column_stack(out), names


def f3_forecast_stats(X, yhat, tv, rows):
    yh = yhat[rows].astype(np.float64)
    xl = X[rows, -1, tv].astype(np.float64)
    out = [yh.std(1), yh.max(1) - yh.min(1), np.abs(np.diff(yh, axis=1)).mean(1),
           np.abs(yh - xl[:, None]).mean(1)]
    return np.column_stack(out), ["yhat_std", "yhat_range", "yhat_mean_abs_diff",
                                  "yhat_mean_dist_last_input"]


def build_exante(run, rows, abs_err=None, S=None):
    """F1, F1p, F2, F3 for the given rows. abs_err / S may be overridden (leakage test)."""
    abs_err = run.abs_err if abs_err is None else abs_err
    S = run.S if S is None else S
    rows = np.asarray(rows)
    assert rows.min() >= MIN_T, "lag features undefined before MIN_T"
    f1, n1 = f1_lagged_full(S, rows)
    f1p, n1p = f1p_partial(abs_err, run.prof_sel, rows)
    f2, n2 = f2_input_stats(run.X, rows)
    f3, n3 = f3_forecast_stats(run.X, run.yhat, run.tv, rows)
    return {"F1": (f1, n1), "F1p": (f1p, n1p), "F2": (f2, n2), "F3": (f3, n3)}


def target_times_used(run, row):
    """Max target time each lagged feature touches for row `row` (for the unit test)."""
    e = run.end_idx
    used = []
    for d in F1_LAGS:
        used.append(e[row - d] + run.H)                       # whole window t-d
    for k in F1_ROLL:
        used.append(e[row - 96] + run.H)                      # latest origin in the roll
    for m in F1P_LAGS:
        used.append(e[row - m] + 1 + (m - 1))
    for i in range(1, F1P_ONESTEP + 1):
        used.append(e[row - i] + 1)
    return np.array(used)
