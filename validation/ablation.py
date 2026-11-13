"""Causal ablation protocol (spec §5.4, ported from Dissecting Chronos).

For SAE feature j, on a held-out ablation-evaluation set:
  1. encode hidden token -> sparse code z
  2. zero the code:  z'_j = 0   (all other codes kept)
  3. decode:  h_hat = W_dec z' + b_dec
  4. patch h_hat back into the forward pass (replace target-variate token), recompute
     the nonconformity score
  5. Delta_j = mean( |e(patched)| - |e(baseline-reconstruction)| )

The baseline is the SAE *reconstruction* (decode of the full code), NOT the raw
hidden state, so Delta_j isolates the effect of removing feature j from the
reconstruction rather than conflating it with the SAE's reconstruction error.

Rank by |Delta_j| (magnitude), NOT by Delta_j > 0 — Dissecting Chronos found ~100%
of features give strictly positive Delta under a binary test, so presence/absence is
uninformative (spec §3.2, §4.2). We report the full distribution.
"""
from __future__ import annotations

import numpy as np
import torch

from forecasters.itransformer import ITransformer
from sae.topk_sae import TopKSAE
from .errors import score_with_patched_hidden


@torch.no_grad()
def _codes_and_recon(sae: TopKSAE, hidden: np.ndarray, device: str = "cpu", bs: int = 4096):
    """Return (z, h_recon) for every row: sparse codes and the full-code decode,
    with h_recon in RAW hidden space (the SAE standardizes internally)."""
    zs, hr = [], []
    for i in range(0, len(hidden), bs):
        xb = torch.from_numpy(hidden[i : i + bs].astype(np.float32)).to(device)
        z = sae.encode(sae.standardize(xb))
        hr.append(sae.unstandardize(sae.decode(z)).cpu().numpy())
        zs.append(z.cpu().numpy())
    return np.concatenate(zs, 0), np.concatenate(hr, 0)


@torch.no_grad()
def _decode_with_feature_zeroed(sae: TopKSAE, z: np.ndarray, j: int, device: str = "cpu", bs: int = 4096):
    """Decode with feature j zeroed, returned in RAW hidden space."""
    out = []
    for i in range(0, len(z), bs):
        zb = torch.from_numpy(z[i : i + bs].astype(np.float32)).to(device).clone()
        zb[:, j] = 0.0
        out.append(sae.unstandardize(sae.decode(zb)).cpu().numpy())
    return np.concatenate(out, 0)


def ablate_features(
    model: ITransformer,
    sae: TopKSAE,
    X: np.ndarray,
    Y: np.ndarray,
    hidden: np.ndarray,
    target_var: int,
    feature_indices: np.ndarray,
    step_profile: np.ndarray | None = None,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Compute Delta_j for each j in `feature_indices`.

    Delta_j is measured ONLY over windows where feature j is active in the SAE code.
    Zeroing a feature that is already off is a no-op, so including inactive windows
    just dilutes the estimate toward zero and makes every feature look causally inert
    (this was the bug that produced a ~0.5 positive-rate instead of the ~1.0 the
    Dissecting-Chronos protocol reports, spec §3.2).

    Returns dict (all length len(feature_indices)):
      delta          signed mean score change over ACTIVE windows  (spec's Delta_j)
      abs_delta      |delta|                                       (MVDF ranking key)
      delta_all      signed mean change over ALL windows           (diagnostic)
      mean_abs_change per-window mean |change| over active windows  (magnitude, noise-robust)
      active_frac    fraction of eval windows where j was active
      n_active       count of active windows
      feature_indices, order (argsort by -abs_delta)
    """
    z, h_recon = _codes_and_recon(sae, hidden, device)
    base_score = score_with_patched_hidden(model, X, Y, h_recon, target_var,
                                           step_profile=step_profile, device=device)

    F = len(feature_indices)
    delta = np.zeros(F)
    delta_all = np.zeros(F)
    mean_abs_change = np.zeros(F)
    active = np.zeros(F)
    n_active = np.zeros(F, dtype=int)
    for t, j in enumerate(feature_indices):
        j = int(j)
        h_abl = _decode_with_feature_zeroed(sae, z, j, device)
        abl_score = score_with_patched_hidden(model, X, Y, h_abl, target_var,
                                              step_profile=step_profile, device=device)
        change = abl_score - base_score
        act = z[:, j] != 0
        n_active[t] = int(act.sum())
        active[t] = float(act.mean())
        delta_all[t] = float(np.mean(change))
        if n_active[t] >= 5:
            delta[t] = float(np.mean(change[act]))
            mean_abs_change[t] = float(np.mean(np.abs(change[act])))
        if verbose and (t % max(1, F // 10) == 0):
            print(f"[ablation] {t+1}/{F}  feat {j}  Delta_active={delta[t]:+.5f}  "
                  f"|change|={mean_abs_change[t]:.5f}  n_active={n_active[t]}")

    order = np.argsort(-np.abs(delta))
    return dict(
        delta=delta, abs_delta=np.abs(delta), delta_all=delta_all,
        mean_abs_change=mean_abs_change, active_frac=active, n_active=n_active,
        feature_indices=np.asarray(feature_indices), order=order,
    )


def select_mvdf(ablation_result: dict, screen_result: dict, k: int, require_correlated_pool: int | None = None) -> np.ndarray:
    """MVDF selection (spec §4.2): top-k by |Delta_j|, restricted to features that
    also passed correlation screening.

    If `require_correlated_pool` is set, the ablation candidate pool is first
    narrowed to the top-`require_correlated_pool` correlated features (this is the
    'validate the correlated set' path). For the RQ1 diagnostic the pool is NOT
    pre-filtered — see rq1.py.
    """
    feats = ablation_result["feature_indices"]
    absd = ablation_result["abs_delta"]

    corr_rank = screen_result["rank_by_abs_pearson"]
    eligible = screen_result["eligible"]

    mask = eligible[feats]
    if require_correlated_pool is not None:
        mask &= corr_rank[feats] < require_correlated_pool

    cand = np.where(mask)[0]
    cand = cand[np.argsort(-absd[cand])]
    return feats[cand[:k]]
