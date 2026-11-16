"""Variant A — FFDCI-style pinball-loss MLP quantile regressor + global ACI
(spec §5.5). Used for the interpretability-cost ablation (Table 5): feed the SAME
k-feature set into an opaque MLP instead of Mondrian bins, holding feature selection
fixed, and report the sharpness difference.

The MLP maps the k selected SAE-feature activations -> a predicted (1 - alpha) error
quantile. A single global ACI correction is added on top (FFDCI Algorithm 1 form).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .aci import GlobalACI


class QuantileMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1), nn.Softplus(),          # non-negative quantile
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _pinball(pred, target, q):
    d = target - pred
    return torch.mean(torch.maximum(q * d, (q - 1) * d))


def train_quantile_mlp(
    feat_cal: np.ndarray, score_cal: np.ndarray, alpha_target: float = 0.1,
    epochs: int = 200, lr: float = 1e-3, seed: int = 0, verbose: bool = False,
) -> QuantileMLP:
    torch.manual_seed(seed)
    x = torch.from_numpy(feat_cal.astype(np.float32))
    y = torch.from_numpy(score_cal.astype(np.float32))
    model = QuantileMLP(feat_cal.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    q = 1 - alpha_target
    for ep in range(epochs):
        opt.zero_grad()
        loss = _pinball(model(x), y, q)
        loss.backward()
        opt.step()
        if verbose and ep % 50 == 0:
            print(f"[qmlp] ep {ep} pinball {loss.item():.4f}")
    model.eval()
    return model


class FFDCIHeadACI:
    """Variant A online runner."""

    def __init__(self, model: QuantileMLP, base_q: float, alpha_target=0.1, gamma=0.01,
                 clip_mult: float = 20.0, step_profile: np.ndarray | None = None,
                 center_pred: bool = True):
        self.model = model
        self.base_q = float(base_q)
        # center_pred=True (default): the MLP output is a per-instance DEVIATION from
        #   the pool mean, base_q carries the level (original Variant A).
        # center_pred=False: split-conformal mode — base_q is the (1-alpha) quantile
        #   of (score - pred_q) on a held-out conformalization set, and the raw pred_q
        #   is added per instance. Coverage ~1-alpha by construction; ACI adapts under shift.
        self.center_pred = center_pred
        self.aci = GlobalACI(base_q=self.base_q, alpha_target=alpha_target, gamma=gamma,
                             clip=clip_mult * max(self.base_q, 1e-6))
        self._mlp_mean = None
        self.step_profile = step_profile

    @torch.no_grad()
    def run(self, feat: np.ndarray, yhat: np.ndarray, y: np.ndarray) -> dict:
        N, H = yhat.shape
        pred_q = self.model(torch.from_numpy(feat.astype(np.float32))).numpy()
        self._mlp_mean = float(pred_q.mean())
        prof = self.step_profile if self.step_profile is not None else np.ones(H)
        lo = np.empty((N, H)); hi = np.empty((N, H)); cov = np.empty(N, dtype=bool)
        for t in range(N):
            extra = (float(pred_q[t]) - self._mlp_mean) if self.center_pred else float(pred_q[t])
            hw = self.aci.half_width(extra=extra)
            hw_vec = hw * prof
            lo[t], hi[t] = yhat[t] - hw_vec, yhat[t] + hw_vec
            ae = np.abs(y[t] - yhat[t])
            cov[t] = bool(np.max(ae / prof) <= hw)
            self.aci.update(bool(cov[t]))
        return dict(lower=lo, upper=hi, covered=cov, width=(hi - lo).mean(axis=1))
