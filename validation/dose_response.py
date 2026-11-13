"""V2 spec §4.3, §10 — the UNIFIED dose-response protocol.

For candidate feature j and lambda in Lambda = [0, 0.25, 0.5, 0.75, 1.0, 1.25], a
SINGLE patched forward pass per lambda yields BOTH:

  D_j^pred(lambda) = E[ L( f(x; z_j -> lambda * a_j(h_t)), y ) ]   effect on the forecast
  D_j^unc(lambda)  = E[ S_t( x; z_j -> lambda * a_j(h_t) ) ]       effect on the deployed score

  P_j = slope of a monotone/linear fit of D_j^pred over lambda   (interventional prediction effect;
        generalizes v1's Delta_pred[j] = D_j^pred(0)'s ablation, i.e. lambda=0 special case)
  U_j = slope of a monotone/linear fit of D_j^unc over lambda    (interventional uncertainty effect;
        v1's S_j, renamed to avoid colliding with S_t)

Both curves come from the SAME perturbation and the SAME instances, so P_j and U_j are
directly comparable — which is what makes the residualization in difficulty_specificity.py
a genuine causal-specificity measure (spec revision note).

Implementation constraint (spec §10): ONE sweep, not two. `compute_dose_response`
returns both dicts from one loop over lambda.
"""
from __future__ import annotations

import numpy as np
import torch

from forecasters.itransformer import ITransformer
from sae.topk_sae import TopKSAE

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]


@torch.no_grad()
def _sae_codes(sae: TopKSAE, hidden_tap: np.ndarray, device="cpu", bs=4096):
    zs = []
    for i in range(0, len(hidden_tap), bs):
        xb = torch.from_numpy(hidden_tap[i:i + bs].astype(np.float32)).to(device)
        zs.append(sae.encode(sae.standardize(xb)).cpu().numpy())
    return np.concatenate(zs, 0)


@torch.no_grad()
def _patched_pred_and_score(
    model: ITransformer, X: np.ndarray, Y: np.ndarray, patched_tap_token: np.ndarray,
    target_var: int, tap: str, step_profile: np.ndarray, device: str, bs: int = 256,
):
    """One patched forward pass. Returns (mse_per_window, score_per_window), both (N,)."""
    N = len(X)
    mse = np.empty(N, np.float32); sc = np.empty(N, np.float32)
    prof = np.asarray(step_profile, np.float64)
    for i in range(0, N, bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        mean = xb.mean(1, keepdim=True); std = xb.std(1, keepdim=True) + 1e-6
        taps = model.encode_all_taps((xb - mean) / std)
        h = taps[tap].clone()
        h[:, target_var, :] = torch.from_numpy(patched_tap_token[i:i + bs]).to(device)
        y = model(xb, patch=(tap, h))
        e = y.cpu().numpy()[:, :, target_var] - Y[i:i + bs][:, :, target_var]
        ae = np.abs(e)
        mse[i:i + bs] = (e ** 2).mean(1)
        sc[i:i + bs] = (ae / prof[None, :]).max(1)
    return mse, sc


@torch.no_grad()
def _decode_scaled(sae: TopKSAE, z: np.ndarray, j: int, lam: float, device="cpu", bs=4096):
    """Decode with feature j set to lam * (its current value), in raw tap space."""
    out = []
    for i in range(0, len(z), bs):
        zb = torch.from_numpy(z[i:i + bs].astype(np.float32)).to(device).clone()
        zb[:, j] = zb[:, j] * lam
        out.append(sae.unstandardize(sae.decode(zb)).cpu().numpy())
    return np.concatenate(out, 0)


def compute_dose_response(
    model: ITransformer,
    sae: TopKSAE,
    X: np.ndarray,
    Y: np.ndarray,
    hidden_tap: np.ndarray,
    target_var: int,
    tap: str,
    feature_idx: int,
    step_profile: np.ndarray,
    lambdas=LAMBDAS,
    device: str = "cpu",
    z_cache: np.ndarray | None = None,
) -> tuple[dict, dict, int]:
    """Returns (D_pred: {lambda: mean_mse}, D_unc: {lambda: mean_score}, n_active).
    Averages over windows where the feature is active at lambda=1 (its natural state)."""
    z = z_cache if z_cache is not None else _sae_codes(sae, hidden_tap, device)
    active = z[:, feature_idx] != 0
    n_active = int(active.sum())
    D_pred, D_unc = {}, {}
    if n_active < 5:
        for lam in lambdas:
            D_pred[lam] = float("nan"); D_unc[lam] = float("nan")
        return D_pred, D_unc, n_active
    Xa, Ya = X[active], Y[active]
    za = z[active]
    for lam in lambdas:
        h_lam = _decode_scaled_from(za, sae, feature_idx, lam, device)
        mse, sc = _patched_pred_and_score(model, Xa, Ya, h_lam, target_var, tap, step_profile, device)
        D_pred[lam] = float(mse.mean())
        D_unc[lam] = float(sc.mean())
    return D_pred, D_unc, n_active


@torch.no_grad()
def _decode_scaled_from(z_subset: np.ndarray, sae: TopKSAE, j: int, lam: float, device, bs=4096):
    out = []
    for i in range(0, len(z_subset), bs):
        zb = torch.from_numpy(z_subset[i:i + bs].astype(np.float32)).to(device).clone()
        zb[:, j] = zb[:, j] * lam
        out.append(sae.unstandardize(sae.decode(zb)).cpu().numpy())
    return np.concatenate(out, 0)


def intervention_sensitivity(curve: dict) -> float:
    """Slope of a linear fit of curve values over lambda. NaN-safe."""
    lam = np.array([k for k in curve], dtype=float)
    val = np.array([curve[k] for k in curve], dtype=float)
    ok = np.isfinite(val)
    if ok.sum() < 3:
        return float("nan")
    A = np.vstack([lam[ok], np.ones(ok.sum())]).T
    slope, _ = np.linalg.lstsq(A, val[ok], rcond=None)[0]
    return float(slope)


def monotonicity_fraction(curves: dict[int, dict]) -> float:
    """Fraction of curves that are monotone (non-decreasing OR non-increasing) in lambda."""
    if not curves:
        return float("nan")
    mono = 0
    for c in curves.values():
        v = np.array([c[k] for k in sorted(c)], dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < 3:
            continue
        d = np.diff(v)
        if np.all(d >= -1e-9) or np.all(d <= 1e-9):
            mono += 1
    return mono / len(curves)


def sweep_pool(
    model, sae, X, Y, hidden_tap, target_var, tap, feature_indices, step_profile,
    lambdas=LAMBDAS, device="cpu", verbose=True,
) -> dict:
    """Run compute_dose_response for every feature in `feature_indices`.
    Returns dict with arrays P_j, U_j, n_active and the raw curves (for plots + the
    monotonicity smoke test / spec §10 acceptance)."""
    z_cache = _sae_codes(sae, hidden_tap, device)
    F = len(feature_indices)
    P = np.full(F, np.nan); U = np.full(F, np.nan); nact = np.zeros(F, int)
    pred_curves, unc_curves = {}, {}
    for t, j in enumerate(feature_indices):
        j = int(j)
        Dp, Du, na = compute_dose_response(model, sae, X, Y, hidden_tap, target_var, tap, j,
                                           step_profile, lambdas, device, z_cache=z_cache)
        P[t] = intervention_sensitivity(Dp)
        U[t] = intervention_sensitivity(Du)
        nact[t] = na
        pred_curves[j] = Dp; unc_curves[j] = Du
        if verbose and t % max(1, F // 10) == 0:
            print(f"[dose] {t+1}/{F} feat {j}  P_j={P[t]:+.4f}  U_j={U[t]:+.4f}  n_active={na}")
    return dict(
        P_j=P, U_j=U, n_active=nact, feature_indices=np.asarray(feature_indices),
        pred_curves=pred_curves, unc_curves=unc_curves,
        mono_frac_pred=monotonicity_fraction(pred_curves),
        mono_frac_unc=monotonicity_fraction(unc_curves),
    )
