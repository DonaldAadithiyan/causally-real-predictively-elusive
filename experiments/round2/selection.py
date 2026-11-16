"""Round 2: frozen MVDF selection (reproduced by calling the UNCHANGED project functions)
plus a vectorised dose-response replica used for nulls / split-half / out-of-sample work.

Frozen functions called unchanged: validation.correlation_screen.screen,
validation.dose_response.sweep_pool, validation.difficulty_specificity.{residualize,
feature_type_table, select_mvdf_v2, enrichment}.

`fast_dose` reproduces sweep_pool's per-feature curves exactly for the pre_head tap:
patched forecast = head(unstd(decode(z with z_j -> lam*z_j))) * std + mean, and
decode is linear, so the patched token = unstd(rec + (lam-1)*z_j*d_j) with d_j the unit
decoder column. Replacing d_j by another unit direction gives the matched random-direction
null with the same activation pattern and magnitudes.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from mechconf.validation.correlation_screen import screen
from mechconf.validation.dose_response import sweep_pool, LAMBDAS
from mechconf.validation.difficulty_specificity import (residualize, feature_type_table,
                                               select_mvdf_v2, enrichment)
from mechconf.sae.train_sae import encode_all

LAM = np.array(LAMBDAS, float)


# ------------------------------------------------------------------ model loading
def load_real_models(weights_dir, tag, n_vars, cfg):
    from mechconf.forecasters.itransformer import ITransformer
    from mechconf.sae.topk_sae import TopKSAE
    fc = cfg["forecaster"]
    m = ITransformer(n_vars=n_vars, lookback=cfg["lookback"], horizon=cfg["horizon"],
                     d_model=fc["d_model"], n_layers=fc["n_layers"], n_heads=fc["n_heads"],
                     d_ff=fc["d_ff"], dropout=fc["dropout"])
    m.load_state_dict(torch.load(os.path.join(weights_dir, f"{tag}_forecaster.pt")))
    m.eval()
    s = TopKSAE(fc["d_model"], expansion=cfg["sae"]["expansion"], k=cfg["sae"]["k"])
    s.load_state_dict(torch.load(os.path.join(weights_dir, f"{tag}_sae.pt")))
    s.eval()
    return m, s


# ------------------------------------------------------------------ frozen selection
def frozen_selection(model, sae, X, Y_full, hid, S_sel, prof, tvar, rng_seed, pool_size,
                     wcap, k, lambdas=LAMBDAS, resid="isotonic", u_thr=0.0, verbose=False):
    """Exactly the selection block of phase3_real_predictability.run_one /
    phase1_synthetic_control.run_seed (same rng consumption order: pool, then wsel)."""
    z_sel = encode_all(sae, hid)
    scr = screen(z_sel, S_sel, min_active_frac=0.02)
    elig = np.where(scr["eligible"])[0]
    rng = np.random.default_rng(rng_seed)
    n_pool = min(pool_size, len(elig))
    pool = np.sort(rng.choice(elig, size=n_pool, replace=False))
    cap = min(wcap, len(X))
    wsel = np.sort(rng.choice(len(X), size=cap, replace=False))
    dose = sweep_pool(model, sae, X[wsel], Y_full[wsel], hid[wsel], tvar, "pre_head", pool,
                      prof, lambdas=lambdas, verbose=verbose)
    ftab = feature_type_table(dose["P_j"], dose["U_j"], feature_indices=pool)
    mv = select_mvdf_v2(dose, u_threshold=u_thr, k=k, residualize_method=resid)
    mvdf = np.asarray(mv["selected"], int)
    corr = pool[np.argsort(-np.abs(scr["pearson"][pool]))[:k]]
    return dict(pool=pool, wsel=wsel, elig=elig, dose=dose, ftab=ftab, mvdf=mvdf, corr=corr,
                R=mv["R_j"], gate1=mv["gate1_pass"], z_sel=z_sel, scr=scr, rng=rng,
                enr=enrichment(mvdf, ftab, pool), enr_corr=enrichment(corr, ftab, pool))


# ------------------------------------------------------------------ fast dose replica
@torch.no_grad()
def fast_dose(model, sae, X, y_t, z, tvar, j, prof, dirs=None, lambdas=LAMBDAS,
              per_window=False):
    """Dose curves for feature j over windows where z[:, j] != 0.
    X (N,L,V) inputs, y_t (N,H) target, z (N,d_sae) codes of those windows.
    dirs: (D, d_model) unit directions in SAE-standardised space (None -> decoder column).
    Returns D_pred (D, nl), D_unc (D, nl) [and per-window (D,nl,n_a) arrays], n_active."""
    a = z[:, j] != 0
    n_a = int(a.sum())
    if dirs is None:
        dirs = sae.W_dec[:, j].detach().numpy()[None, :]
    D = len(dirs)
    nl = len(lambdas)
    if n_a < 5:
        nan = np.full((D, nl), np.nan)
        return (nan, nan, None, None, n_a) if per_window else (nan, nan, n_a)
    za = torch.from_numpy(z[a].astype(np.float32))
    rec = sae.decode(za)                                         # (n_a, d) std space
    zj = za[:, j][:, None]                                       # (n_a, 1)
    dirs_t = torch.from_numpy(np.asarray(dirs, np.float32))      # (D, d)
    lam = torch.tensor(lambdas, dtype=torch.float32)
    # (D, nl, n_a, d)
    h = rec[None, None] + (lam[None, :, None, None] - 1) * zj[None, None] * dirs_t[:, None, None, :]
    h = sae.unstandardize(h)
    xb = torch.from_numpy(X[a][:, :, tvar].astype(np.float32))   # (n_a, L)
    mu = xb.mean(1)                                              # matches forward(): per-variate
    sd = xb.std(1) + 1e-6
    yp = model.head(h)                                           # (D, nl, n_a, H)
    yp = yp * sd[None, None, :, None] + mu[None, None, :, None]
    e = yp.numpy() - y_t[a][None, None].astype(np.float32)
    mse = (e ** 2).mean(-1)                                      # (D, nl, n_a)
    sc = (np.abs(e) / np.asarray(prof, np.float64)[None, None, None, :]).max(-1)
    if per_window:
        return mse.mean(-1), sc.mean(-1), mse, sc, n_a
    return mse.mean(-1), sc.mean(-1), n_a


def slope(lams, v):
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return np.nan, np.nan
    A = np.vstack([np.asarray(lams)[ok], np.ones(ok.sum())]).T
    coef, res, *_ = np.linalg.lstsq(A, v[ok], rcond=None)
    pred = A @ coef
    ss = np.sum((v[ok] - v[ok].mean()) ** 2)
    r2 = 1 - np.sum((v[ok] - pred) ** 2) / ss if ss > 0 else np.nan
    return float(coef[0]), float(r2)


def monotone(v):
    d = np.diff(np.asarray(v, float))
    return bool(np.all(d >= -1e-9) or np.all(d <= 1e-9))


def type_B_enrichment(P, U, R, unc_curves_ok, pool, k, u_thr=0.0):
    """Unchanged selection logic applied to arbitrary (P, U) values: gate1 = monotone unc
    curve & finite R, rank by R (isotonic residual of U on P), top-k; type table by median
    split; enrichment of type B. Mirrors select_mvdf_v2 + feature_type_table + enrichment."""
    dose = dict(P_j=P, U_j=U, feature_indices=pool,
                unc_curves={int(j): ({0: 0.0, 1: 1.0, 2: 2.0} if ok else {0: 0.0, 1: 1.0, 2: 0.0})
                            for j, ok in zip(pool, unc_curves_ok)})
    mv = select_mvdf_v2(dose, u_threshold=u_thr, k=k, residualize_method="isotonic")
    ft = feature_type_table(P, U, feature_indices=pool)
    e = enrichment(np.asarray(mv["selected"], int), ft, pool)
    return float(e.get("enrichment_ratio", np.nan)), np.asarray(mv["selected"], int)
