"""Per-instance forecast error and the hidden-state tensor the SAE/validation
layers consume.

The conformal layer needs, for every calibration/test window t:
  - yhat_t, y_t                        (horizon, V)
  - a scalar nonconformity score s_t   -> we use mean absolute error over the
    horizon and over the target channel(s); spec uses |e|, FFDCI uses per-step/per-dim
  - the hidden token h_t used to define difficulty features

iTransformer's hidden state is (V, d_model) per window. We reduce it to a single
d_model vector per window by taking the TARGET channel's token ('OT' for ETT, i.e.
the last channel — the standard univariate-target LTSF convention), because the
difficulty feature must be a per-window scalar-driving representation. `target_var`
is configurable.
"""
from __future__ import annotations

import numpy as np
import torch

from forecasters.itransformer import ITransformer


@torch.no_grad()
def collect(
    model: ITransformer,
    X: np.ndarray,
    Y: np.ndarray,
    target_var: int,
    batch_size: int = 512,
    device: str = "cpu",
):
    """Return dict:
      yhat   (N, H, V)
      y      (N, H, V)
      hidden (N, d_model)          target-variate token
      score  (N,)                  max_h |e_h| / profile_h  over the horizon (target chan)
      score_mean (N,)              mean_h |e_h|  (diagnostic only)
      step_profile (H,)            calibration per-step |e| shape, mean 1 (set by caller
                                   via set_step_profile; here it is the local mean shape)

    The nonconformity score is the SMALLEST scalar half-width multiplier that would
    have covered the ENTIRE horizon for that window, given a fixed per-step band
    shape. Calibrating the (1-alpha) quantile of this score therefore targets
    whole-window coverage 1-alpha directly and is consistent with an "all H steps
    inside" coverage check (the previous mean-|e| score was not, which drove the
    60-72% in-distribution undercoverage).
    """
    model.eval()
    N = len(X)
    yhat = np.empty_like(Y)
    hidden = np.empty((N, model.d_model), dtype=np.float32)
    for i in range(0, N, batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        pred, h = model(xb, return_hidden=True)
        yhat[i : i + batch_size] = pred.cpu().numpy()
        hidden[i : i + batch_size] = h[:, target_var, :].cpu().numpy()
    abs_err = np.abs(yhat[:, :, target_var] - Y[:, :, target_var])     # (N, H)
    profile = abs_err.mean(axis=0)
    profile = profile / (profile.mean() + 1e-12)                      # (H,), mean 1
    score = (abs_err / profile[None, :]).max(axis=1)                  # (N,)
    return dict(
        yhat=yhat, y=Y, hidden=hidden,
        score=score.astype(np.float32),
        score_mean=abs_err.mean(axis=1).astype(np.float32),
        step_profile=profile.astype(np.float32),
        abs_err=abs_err.astype(np.float32),
    )


@torch.no_grad()
def score_with_patched_hidden(
    model: ITransformer,
    X: np.ndarray,
    Y: np.ndarray,
    patched_target_token: np.ndarray,
    target_var: int,
    step_profile: np.ndarray | None = None,
    batch_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """Recompute the nonconformity score after replacing the target-variate hidden
    token with `patched_target_token` (N, d_model) and running it through the head.

    Uses the SAME score definition as `collect`: max_h |e_h| / profile_h. Pass the
    calibration `step_profile`; if None a flat profile is used. Only the target token
    is replaced; other variate tokens keep their original values.
    """
    model.eval()
    N = len(X)
    out = np.empty(N, dtype=np.float32)
    prof = None if step_profile is None else np.asarray(step_profile, dtype=np.float64)
    for i in range(0, N, batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        mean = xb.mean(dim=1, keepdim=True)
        std = xb.std(dim=1, keepdim=True) + 1e-6
        h = model.encode((xb - mean) / std)
        h = h.clone()
        h[:, target_var, :] = torch.from_numpy(
            patched_target_token[i : i + batch_size]
        ).to(device)
        y = model(xb, h_override=h)
        yb = Y[i : i + batch_size]
        ae = np.abs(y.cpu().numpy()[:, :, target_var] - yb[:, :, target_var])   # (b, H)
        if prof is None:
            out[i : i + batch_size] = ae.max(axis=1)
        else:
            out[i : i + batch_size] = (ae / prof[None, :]).max(axis=1)
    return out
