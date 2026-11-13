"""Train iTransformer with the standard LTSF protocol (MSE, Adam, early stop on val).

FFDCI reuses public checkpoints; we train our own so the pipeline is self-contained
and seed-controlled. Config is fixed, not searched — finding the best point forecaster
is explicitly not the point (spec §5.2).
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .itransformer import ITransformer


@dataclass
class TrainConfig:
    lookback: int = 96
    horizon: int = 96
    d_model: int = 256
    n_layers: int = 3
    n_heads: int = 8
    d_ff: int = 512
    dropout: float = 0.1
    lr: float = 1e-4              # iTransformer reference LR for ETT (3e-4 overshoots)
    batch_size: int = 32         # reference uses 32 for ETT
    max_epochs: int = 20
    patience: int = 5
    lr_decay: float = 0.8        # gentle per-epoch decay
    warmup_epochs: int = 3       # hold LR flat before decaying
    seed: int = 0
    device: str = "cpu"


def _loader(X, Y, bs, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=shuffle)


def train_itransformer(train_w, val_w, n_vars: int, cfg: TrainConfig, verbose: bool = True):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    Xtr, Ytr, _ = train_w
    Xva, Yva, _ = val_w

    model = ITransformer(
        n_vars=n_vars, lookback=cfg.lookback, horizon=cfg.horizon, d_model=cfg.d_model,
        n_layers=cfg.n_layers, n_heads=cfg.n_heads, d_ff=cfg.d_ff, dropout=cfg.dropout,
    ).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def lr_lambda(ep):
        return 1.0 if ep < cfg.warmup_epochs else cfg.lr_decay ** (ep - cfg.warmup_epochs + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    lossfn = torch.nn.MSELoss()

    dl_tr = _loader(Xtr, Ytr, cfg.batch_size, True)
    dl_va = _loader(Xva, Yva, cfg.batch_size, False)

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(cfg.max_epochs):
        model.train()
        t0 = time.time()
        for xb, yb in dl_tr:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            pred = model(xb)
            loss = lossfn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vs = [lossfn(model(xb.to(cfg.device)), yb.to(cfg.device)).item() * len(xb)
                  for xb, yb in dl_va]
            val = sum(vs) / len(Xva)
        if verbose:
            lr_now = opt.param_groups[0]["lr"]
            print(f"[itr] epoch {ep+1:02d}  val_mse={val:.5f}  lr={lr_now:.2e}  ({time.time()-t0:.1f}s)")
        if val < best_val - 1e-6:
            best_val, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                if verbose:
                    print(f"[itr] early stop at epoch {ep+1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_val


@torch.no_grad()
def predict(model, X: np.ndarray, batch_size: int = 512, device: str = "cpu") -> np.ndarray:
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out, 0)
