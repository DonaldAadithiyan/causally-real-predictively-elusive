"""Phase 0 — representation-location search (v2 spec §4.5, §10, §11).

v1/Run 5 fixed the SAE extraction point at the pre-head layer and found difficulty
only weakly represented there (max|corr(hidden PC, |e|)| = 0.09-0.22). This module
treats *where in the network* difficulty lives as an open empirical question: for
each tap in `ITransformer.TAP_NAMES` it computes a cheap first-pass signal
(max |corr(activation dimension, nonconformity score)|) so Phase 1's expensive
dose-response sweep only runs at the top layers.

Go criterion (PRE-DECLARED in configs/v2_phase0.yaml before running):
  a layer "clears" if its cheap signal exceeds `go_threshold` on >= `go_min_datasets`
  datasets. `select_winning_layers` returns the cleared layers, or an explicit
  NO_GO sentinel (not the best-of-a-bad-set) so the caller cannot silently proceed.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch

from forecasters.itransformer import ITransformer

NO_GO = "__NO_GO__"


@dataclasses.dataclass
class LayerSignal:
    tap: str
    max_abs_corr: float          # max over activation dims of |corr(a_d, S_t)|
    max_abs_corr_pc: float       # same but over the top-20 PCs of the activation
    mean_abs_corr_top10: float   # mean |corr| of the 10 most correlated dims
    n_dims: int


@torch.no_grad()
def extract_all_taps(
    model: ITransformer,
    X: np.ndarray,
    target_var: int,
    batch_size: int = 256,
    device: str = "cpu",
) -> dict:
    """Return {tap_name: (N, d_model)} — the target-variate token at every tap."""
    model.eval()
    acc: dict[str, list] = {t: [] for t in model.TAP_NAMES}
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        mean = xb.mean(dim=1, keepdim=True)
        std = xb.std(dim=1, keepdim=True) + 1e-6
        taps = model.encode_all_taps((xb - mean) / std)
        for t, h in taps.items():
            acc[t].append(h[:, target_var, :].cpu().numpy())
    return {t: np.concatenate(v, 0) for t, v in acc.items()}


def _abs_corr(feat: np.ndarray, score: np.ndarray) -> np.ndarray:
    """|corr| of every column of `feat` (N,D) with `score` (N,). Zero for dead cols."""
    s = (score - score.mean()) / (score.std() + 1e-12)
    F = feat - feat.mean(0, keepdims=True)
    sd = F.std(0) + 1e-12
    return np.abs((F * s[:, None]).mean(0) / sd)


def cheap_signal(feat: np.ndarray, score: np.ndarray) -> LayerSignal:
    from sklearn.decomposition import PCA

    ac = _abs_corr(feat, score)
    npc = min(20, feat.shape[1], max(2, feat.shape[0] // 50))
    try:
        pcs = PCA(n_components=npc).fit_transform(feat)
        ac_pc = _abs_corr(pcs, score)
        max_pc = float(ac_pc.max())
    except Exception:
        max_pc = float("nan")
    top10 = np.sort(ac)[::-1][:10]
    return LayerSignal(
        tap="", max_abs_corr=float(ac.max()), max_abs_corr_pc=max_pc,
        mean_abs_corr_top10=float(top10.mean()), n_dims=feat.shape[1],
    )


def sweep_dataset(
    model: ITransformer, X: np.ndarray, score: np.ndarray, target_var: int,
    device: str = "cpu",
) -> dict[str, LayerSignal]:
    taps = extract_all_taps(model, X, target_var, device=device)
    out = {}
    for t, feat in taps.items():
        sig = cheap_signal(feat, score)
        sig.tap = t
        out[t] = sig
    return out


def select_winning_layers(
    signals_by_dataset: dict[str, dict[str, LayerSignal]],
    go_threshold: float,
    go_min_datasets: int,
    metric: str = "max_abs_corr_pc",
) -> list[str] | str:
    """signals_by_dataset: {dataset: {tap: LayerSignal}}.
    Returns the list of taps that clear `go_threshold` on >= `go_min_datasets`
    datasets by `metric`, sorted best-first; or NO_GO if none clear."""
    taps = next(iter(signals_by_dataset.values())).keys()
    cleared = []
    for tap in taps:
        vals = [getattr(sig[tap], metric) for sig in signals_by_dataset.values()]
        n_clear = sum(v >= go_threshold for v in np.nan_to_num(vals))
        if n_clear >= go_min_datasets:
            cleared.append((tap, float(np.nanmean(vals))))
    if not cleared:
        return NO_GO
    return [t for t, _ in sorted(cleared, key=lambda kv: -kv[1])]
