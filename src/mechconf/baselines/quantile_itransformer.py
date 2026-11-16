"""Quantile-regression iTransformer for CQR (Romano, Patterson & Candès 2019).

CQR needs a base model that predicts the conditional quantiles q_lo(x), q_hi(x)
directly — a point forecaster's residual score is not a valid CQR input. This is a
literal CQR baseline: SAME trunk architecture as the project's point forecaster
(embed -> 3 encoder layers -> norm), but the head is replaced with two heads
(nn.Linear(d_model, horizon) each) and the model is trained with pinball loss at
alpha/2 and 1-alpha/2 instead of MSE. No SAE, no MVDF, no representation-conditioning —
this is CQR in its standard, citable form; the *only* thing shared with the rest of the
project is the choice of trunk (which CQR's proof does not constrain), not the method.

Trained on the SAME data split the project's forecaster uses (E_train / E_val), same
lookback/horizon, same RevIN-style normalize/denormalize. A fresh model per
(dataset, seed) — deterministic via `experiments._seed.derive_seed`.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mechconf.forecasters.itransformer import ITransformer


class QuantileITransformer(nn.Module):
    """Same trunk as ITransformer; two horizon-shaped heads for q_lo / q_hi."""

    def __init__(self, n_vars: int, lookback: int = 96, horizon: int = 96,
                d_model: int = 256, n_layers: int = 3, n_heads: int = 8,
                d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.trunk = ITransformer(n_vars=n_vars, lookback=lookback, horizon=horizon,
                                  d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                                  d_ff=d_ff, dropout=dropout)
        self.head_lo = nn.Linear(d_model, horizon)
        self.head_hi = nn.Linear(d_model, horizon)
        self.horizon = horizon

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        xn = (x - mean) / std
        h = self.trunk.encode(xn)                 # (B, V, d_model)
        lo = self.head_lo(h).permute(0, 2, 1)      # (B, horizon, V)
        hi = self.head_hi(h).permute(0, 2, 1)
        lo = lo * std + mean
        hi = hi * std + mean
        return lo, hi


def _pinball(pred: torch.Tensor, target: torch.Tensor, q: float) -> torch.Tensor:
    d = target - pred
    return torch.mean(torch.maximum(q * d, (q - 1) * d))


@dataclass
class QuantileTrainConfig:
    lookback: int = 96
    horizon: int = 96
    d_model: int = 256
    n_layers: int = 3
    n_heads: int = 8
    d_ff: int = 512
    dropout: float = 0.1
    lr: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 20
    patience: int = 5
    lr_decay: float = 0.8
    warmup_epochs: int = 3
    seed: int = 0
    device: str = "cpu"
    alpha: float = 0.10   # q_lo = alpha/2, q_hi = 1 - alpha/2


def _loader(X, Y, bs, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=shuffle)


def train_quantile_itransformer(train_w, val_w, n_vars: int, cfg: QuantileTrainConfig,
                                verbose: bool = False):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    Xtr, Ytr, _ = train_w
    Xva, Yva, _ = val_w
    q_lo, q_hi = cfg.alpha / 2, 1 - cfg.alpha / 2

    model = QuantileITransformer(n_vars=n_vars, lookback=cfg.lookback, horizon=cfg.horizon,
                                 d_model=cfg.d_model, n_layers=cfg.n_layers,
                                 n_heads=cfg.n_heads, d_ff=cfg.d_ff, dropout=cfg.dropout
                                 ).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def lr_lambda(ep):
        return 1.0 if ep < cfg.warmup_epochs else cfg.lr_decay ** (ep - cfg.warmup_epochs + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    dl_tr = _loader(Xtr, Ytr, cfg.batch_size, True)
    dl_va = _loader(Xva, Yva, cfg.batch_size, False)

    def val_loss():
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                lo, hi = model(xb)
                l = _pinball(lo, yb, q_lo) + _pinball(hi, yb, q_hi)
                tot += l.item() * len(xb)
        return tot / len(Xva)

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(cfg.max_epochs):
        model.train()
        t0 = time.time()
        for xb, yb in dl_tr:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            lo, hi = model(xb)
            loss = _pinball(lo, yb, q_lo) + _pinball(hi, yb, q_hi)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        val = val_loss()
        if verbose:
            print(f"[cqr-fc] epoch {ep+1:02d}  val_pinball={val:.5f}  "
                  f"({time.time()-t0:.1f}s)")
        if val < best_val - 1e-6:
            best_val, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                if verbose:
                    print(f"[cqr-fc] early stop at epoch {ep+1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_val


@torch.no_grad()
def predict_quantiles(model: QuantileITransformer, X: np.ndarray, batch_size: int = 512,
                      device: str = "cpu"):
    """Returns (lo, hi), each (N, horizon, V)."""
    los, his = [], []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        lo, hi = model(xb)
        los.append(lo.cpu().numpy())
        his.append(hi.cpu().numpy())
    return np.concatenate(los, 0), np.concatenate(his, 0)
