"""Phase 1 synthetic positive control — the frozen data-generating process.

Implements exactly (phase1-synthetic-positive-control-task.md §1,
causal-predictive-dissociation-experiments.md §1.1):

    x_{t+1} = a_{z_t} * x_t + eps_t,        eps_t ~ N(0, 1)
    y_{t+1} = f(x_t)  + sigma(z_t) * eta_t, eta_t ~ N(0, 1)
    sigma(0) = sigma_easy,  sigma(1) = sigma_hard,  sigma_hard >> sigma_easy
    z_t in {0, 1}: two-state Markov chain with real persistence

CRITICAL: z_t is NEVER an input to the forecaster. It is inferable only from the
recent dynamics of x_t (the regime-dependent AR coefficient a_{z_t}). The windowing
code here returns ONLY x. z_t is returned separately, for the §5 ground-truth check
and for nothing that touches the forecaster's inputs.

All parameters come from configs/synthetic_dgp_config.json, loaded and frozen. This
module does not expose knobs to tune them post-hoc.
"""
from __future__ import annotations

import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.normpath(os.path.join(_HERE, "..", "configs", "synthetic_dgp_config.json"))


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _f_apply(x: np.ndarray, expr: str) -> np.ndarray:
    """Evaluate the frozen mild nonlinearity. Only the pre-declared form is allowed."""
    if expr.replace(" ", "") == "0.5*x+0.1*x**2":
        return 0.5 * x + 0.1 * x ** 2
    raise ValueError(f"f expression {expr!r} is not the frozen form; refusing to run.")


def _markov_regimes(n: int, p_stay_0: float, p_stay_1: float, rng: np.random.Generator,
                    z0: int = 0) -> np.ndarray:
    """Two-state Markov chain, length n. p_stay_k = P(z_{t+1}=k | z_t=k)."""
    z = np.empty(n, dtype=np.int8)
    z[0] = z0
    u = rng.random(n)
    for t in range(1, n):
        if z[t - 1] == 0:
            z[t] = 0 if u[t] < p_stay_0 else 1
        else:
            z[t] = 1 if u[t] < p_stay_1 else 0
    return z


def generate_series(n_timesteps: int, cfg: dict, rng: np.random.Generator,
                    p_stay_0: float | None = None, p_stay_1: float | None = None,
                    burn_in: int | None = None, mu_z_override: float | None = None) -> dict:
    """Generate one contiguous synthetic series.

    Returns dict(x=(T,), y=(T,), z=(T,)) aligned so that y[t] is the target to be
    predicted from information up to and including x[t]. (x[t+1], y[t+1]) depend on
    (x[t], z[t]) per the DGP. Burn-in steps are discarded.

    y[t] = mu_z * 1[z[t]==1] + f(x[t]) + sigma(z[t]) * eta[t]

    The mu_z mean-channel (pre-run correction 3): a variance-only regime gives an
    MSE-optimal point forecaster ZERO incentive to encode z (E[y|x,z] would be
    z-independent). mu_z makes E[y|x,z] depend on z, so the forecaster must infer
    the regime to minimize loss. Difficulty still varies through sigma(z).
    """
    d = cfg["dgp"]
    p0 = d["markov_p_stay_0"] if p_stay_0 is None else p_stay_0
    p1 = d["markov_p_stay_1"] if p_stay_1 is None else p_stay_1
    bi = d["burn_in"] if burn_in is None else burn_in
    mu_z = (d.get("mu_z") or 0.0) if mu_z_override is None else mu_z_override
    a = np.array([d["a_0"], d["a_1"]], dtype=np.float64)
    sig = np.array([d["sigma_easy"], d["sigma_hard"]], dtype=np.float64)

    T = n_timesteps + bi
    z = _markov_regimes(T, p0, p1, rng)
    eps = rng.normal(0.0, d["eps_sd"], size=T)
    eta = rng.normal(0.0, d["eta_sd"], size=T)

    x = np.empty(T, dtype=np.float64)
    x[0] = rng.normal(0.0, 1.0)
    for t in range(1, T):
        x[t] = a[z[t - 1]] * x[t - 1] + eps[t]

    fx = _f_apply(x, d["f"])
    y = mu_z * (z == 1).astype(np.float64) + fx + sig[z] * eta

    sl = slice(bi, None)
    return dict(x=x[sl].astype(np.float32), y=y[sl].astype(np.float32),
                z=z[sl].astype(np.int8))


def make_windows_xy(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    lookback: int, horizon: int):
    """Sliding windows. INPUT is x only. Target is the next-horizon y block.

    X : (N, lookback, 1)   past x   (the forecaster sees this and ONLY this)
    Y : (N, horizon, 1)    future y
    Zwin : (N,)            regime z at the window's last input step (for the §5
                           ground-truth check; NOT fed to the model)
    end_idx : (N,)         index of the last input step into the series
    """
    T = len(x)
    starts = np.arange(0, T - lookback - horizon + 1)
    X = np.stack([x[s:s + lookback] for s in starts])[:, :, None].astype(np.float32)
    Y = np.stack([y[s + lookback:s + lookback + horizon] for s in starts])[:, :, None].astype(np.float32)
    end_idx = starts + lookback - 1
    Zwin = z[end_idx].astype(np.int8)
    # also the mean regime over the horizon being predicted (the thing difficulty
    # actually tracks): fraction of hard steps in the target block
    Zhard_frac = np.stack([z[s + lookback:s + lookback + horizon].mean() for s in starts]).astype(np.float32)
    return X, Y, Zwin, Zhard_frac, end_idx


def build_split_series(cfg: dict, seed: int, mu_z_override: float | None = None) -> dict:
    """Generate the full train/val/test series with the pre-registered split scheme
    and the E_eval_shift regime-base-rate shift. Returns raw (unnormalized) arrays;
    the caller z-scores using train stats, exactly as data/datasets.get_splits does.
    """
    d = cfg["dgp"]
    sp = cfg["splits"]
    sh = cfg["shift_construction"]
    n_total = d["n_timesteps"]
    mu_z = (d.get("mu_z") or 0.0) if mu_z_override is None else mu_z_override
    rng = np.random.default_rng(seed)

    n_tr = int(sp["train_frac"] * n_total)
    n_va = int(sp["val_frac"] * n_total)
    n_te = n_total - n_tr - n_va

    # train + val + first two-thirds of test: frozen 0.97/0.97 chain, one contiguous
    # generation so persistence carries across the boundaries.
    n_pre_shift = n_tr + n_va + int(round(n_te * 2 / 3))
    pre = generate_series(n_pre_shift, cfg, rng, mu_z_override=mu_z)

    # final third of test: shifted regime base rate (more hard regime), continuing
    # from the last pre-shift x value so the AR process is continuous.
    n_shift = n_te - int(round(n_te * 2 / 3))
    rng_sh = np.random.default_rng(seed + 999)
    z_sh = _markov_regimes(n_shift + 1, sh["shift_block_p_stay_0"],
                           sh["shift_block_p_stay_1"], rng_sh,
                           z0=int(pre["z"][-1]))
    a = np.array([d["a_0"], d["a_1"]], dtype=np.float64)
    sig = np.array([d["sigma_easy"], d["sigma_hard"]], dtype=np.float64)
    eps = rng_sh.normal(0.0, d["eps_sd"], size=n_shift + 1)
    eta = rng_sh.normal(0.0, d["eta_sd"], size=n_shift + 1)
    x_sh = np.empty(n_shift + 1, dtype=np.float64)
    x_sh[0] = float(pre["x"][-1])
    for t in range(1, n_shift + 1):
        x_sh[t] = a[z_sh[t - 1]] * x_sh[t - 1] + eps[t]
    fx_sh = _f_apply(x_sh, d["f"])
    y_sh = mu_z * (z_sh == 1).astype(np.float64) + fx_sh + sig[z_sh] * eta
    x_sh, y_sh, z_sh = x_sh[1:].astype(np.float32), y_sh[1:].astype(np.float32), z_sh[1:].astype(np.int8)

    x = np.concatenate([pre["x"], x_sh])
    y = np.concatenate([pre["y"], y_sh])
    z = np.concatenate([pre["z"], z_sh])

    raw_tr_x, raw_va_x, raw_te_x = x[:n_tr], x[n_tr:n_tr + n_va], x[n_tr + n_va:]
    raw_tr_y, raw_va_y, raw_te_y = y[:n_tr], y[n_tr:n_tr + n_va], y[n_tr + n_va:]
    z_tr, z_va, z_te = z[:n_tr], z[n_tr:n_tr + n_va], z[n_tr + n_va:]

    return dict(
        train=dict(x=raw_tr_x, y=raw_tr_y, z=z_tr),
        val=dict(x=raw_va_x, y=raw_va_y, z=z_va),
        test=dict(x=raw_te_x, y=raw_te_y, z=z_te),
        hard_frac=dict(
            train=float((z_tr == 1).mean()),
            val=float((z_va == 1).mean()),
            test=float((z_te == 1).mean()),
            test_first_two_thirds=float((z_te[:int(round(len(z_te) * 2 / 3))] == 1).mean()),
            test_final_third=float((z_te[int(round(len(z_te) * 2 / 3)):] == 1).mean()),
        ),
    )
