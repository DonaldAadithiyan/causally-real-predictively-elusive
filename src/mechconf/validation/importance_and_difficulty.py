"""V2 spec §4.2, §10 — the single-point interventional prediction effect and the
observational difficulty diagnostic.

  delta_pred[j]  (== P_j at lambda=0, == v1's Delta_j, == MVDF_v1's selection key):
      E[ L(f(x; z_j -> 0), y) - L(f(x), y) ]  over windows where feature j is active.
      Computed here for MVDF_v1 backward-compat and as a smoke-test anchor for the
      full dose-response curve.

  C_j  (observational diagnostic ONLY, v1's Delta_difficulty renamed):
      Cov( a_j(h_t), S_t )  over windows where a_j > 0.
      Feeds the `correlated_set` baseline and the H1/H1b divergence tables.
      NEVER feeds R_j (spec §4.4, §15). A provenance tag marks it.
"""
from __future__ import annotations

import numpy as np
import torch

from mechconf.forecasters.itransformer import ITransformer
from mechconf.sae.topk_sae import TopKSAE
from mechconf.validation.errors import score_with_patched_hidden


@torch.no_grad()
def _codes_and_recon_at_tap(sae: TopKSAE, hidden_tap: np.ndarray, device="cpu", bs=4096):
    """hidden_tap: (N, d_model) target-variate token AT the SAE's tap layer.
    Returns (z, h_recon_raw) with h_recon in raw (un-standardized) tap space."""
    zs, hr = [], []
    for i in range(0, len(hidden_tap), bs):
        xb = torch.from_numpy(hidden_tap[i:i + bs].astype(np.float32)).to(device)
        z = sae.encode(sae.standardize(xb))
        hr.append(sae.unstandardize(sae.decode(z)).cpu().numpy())
        zs.append(z.cpu().numpy())
    return np.concatenate(zs, 0), np.concatenate(hr, 0)


def compute_c_j(activations: np.ndarray, scores: np.ndarray, active_only: bool = True) -> np.ndarray:
    """C_j = Cov(a_j, S_t) per feature. activations: (N, d_sae), scores: (N,)."""
    N, d = activations.shape
    s = scores.astype(np.float64)
    out = np.zeros(d)
    for j in range(d):
        a = activations[:, j].astype(np.float64)
        m = a > 0 if active_only else np.ones(N, bool)
        if m.sum() < 5:
            continue
        out[j] = np.cov(a[m], s[m])[0, 1]
    return out


def compute_delta_pred_single(
    model: ITransformer,
    sae: TopKSAE,
    X: np.ndarray,
    Y: np.ndarray,
    hidden_tap: np.ndarray,
    target_var: int,
    tap: str,
    feature_indices: np.ndarray,
    step_profile: np.ndarray | None = None,
    device: str = "cpu",
    loss: str = "score",
) -> dict:
    """Single-point (lambda=0) ablation effect on the PREDICTION loss, per feature,
    over active windows. `loss='score'` uses S_t (== v1 Delta_j); `loss='mse'` uses
    horizon MSE of the target channel.

    Patches the SAE reconstruction (feature zeroed) back in AT `tap` and continues
    the forward pass — so this works at any probed layer, not just pre_head.
    """
    z, h_recon = _codes_and_recon_at_tap(sae, hidden_tap, device)
    base = _patched_loss(model, X, Y, h_recon, target_var, tap, step_profile, device, loss)

    F = len(feature_indices)
    delta = np.zeros(F); n_active = np.zeros(F, dtype=int)
    for t, j in enumerate(feature_indices):
        j = int(j)
        zz = z.copy(); zz[:, j] = 0.0
        with torch.no_grad():
            h_abl = sae.unstandardize(sae.decode(torch.from_numpy(zz.astype(np.float32)).to(device))).cpu().numpy()
        abl = _patched_loss(model, X, Y, h_abl, target_var, tap, step_profile, device, loss)
        act = z[:, j] != 0
        n_active[t] = int(act.sum())
        if n_active[t] >= 5:
            delta[t] = float(np.mean(abl[act] - base[act]))
    return dict(delta_pred=delta, n_active=n_active, feature_indices=np.asarray(feature_indices),
               provenance="interventional_single_point")


@torch.no_grad()
def _patched_loss(model, X, Y, patched_tap_token, target_var, tap, step_profile, device, loss):
    """Patch `patched_tap_token` (N, d_model) as the target-variate token AT `tap`,
    continue forward, return per-window loss array (N,)."""
    if tap in ("pre_head", "head_input"):
        return score_with_patched_hidden(model, X, Y, patched_tap_token, target_var,
                                         step_profile=step_profile, device=device) \
            if loss == "score" else _mse_with_patched_hidden(model, X, Y, patched_tap_token, target_var, device)
    # general tap: need the full (B,V,d) representation at `tap`, replace target col, continue
    N = len(X); out = np.empty(N, dtype=np.float32)
    prof = None if step_profile is None else np.asarray(step_profile, np.float64)
    bs = 256
    for i in range(0, N, bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        mean = xb.mean(1, keepdim=True); std = xb.std(1, keepdim=True) + 1e-6
        taps = model.encode_all_taps((xb - mean) / std)
        h = taps[tap].clone()
        h[:, target_var, :] = torch.from_numpy(patched_tap_token[i:i + bs]).to(device)
        y = model(xb, patch=(tap, h))
        yb = Y[i:i + bs]
        ae = np.abs(y.cpu().numpy()[:, :, target_var] - yb[:, :, target_var])
        if loss == "score":
            out[i:i + bs] = ae.max(1) if prof is None else (ae / prof[None, :]).max(1)
        else:
            out[i:i + bs] = (ae ** 2).mean(1)
    return out


@torch.no_grad()
def _mse_with_patched_hidden(model, X, Y, patched, target_var, device, bs=512):
    N = len(X); out = np.empty(N, np.float32)
    for i in range(0, N, bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        mean = xb.mean(1, keepdim=True); std = xb.std(1, keepdim=True) + 1e-6
        h = model.encode((xb - mean) / std).clone()
        h[:, target_var, :] = torch.from_numpy(patched[i:i + bs]).to(device)
        y = model(xb, h_override=h)
        ae = y.cpu().numpy()[:, :, target_var] - Y[i:i + bs][:, :, target_var]
        out[i:i + bs] = (ae ** 2).mean(1)
    return out


def rank_divergence(delta_pred: np.ndarray, c_j: np.ndarray, feature_indices: np.ndarray,
                    top_frac: float = 0.25) -> dict:
    """H1b: does the P_j (interventional) ranking diverge from the C_j (observational)
    ranking? Reported for every probed layer regardless of pool size."""
    from scipy import stats
    a, b = np.abs(delta_pred), np.abs(c_j)
    ok = (a > 0) & (b > 0)
    if ok.sum() < 5:
        return {"n": int(ok.sum()), "note": "too few features"}
    rho = stats.spearmanr(a[ok], b[ok]).statistic
    n_top = max(1, int(round(top_frac * ok.sum())))
    top_p = set(feature_indices[ok][np.argsort(-a[ok])[:n_top]].tolist())
    top_c = set(feature_indices[ok][np.argsort(-b[ok])[:n_top]].tolist())
    return dict(n=int(ok.sum()), spearman_rho=float(rho),
                top_set_jaccard=len(top_p & top_c) / len(top_p | top_c),
                interpretation="LOW rho -> interventional and observational rankings differ (H1b)"
                if abs(rho) < 0.6 else "HIGH rho -> rankings mostly agree")
