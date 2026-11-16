"""Train the TimeMixer forecaster — same LTSF protocol as train_itransformer
(MSE, Adam, early-stop on val). Config fixed, not searched (spec §5.2 discipline).
Mirrors train_itransformer's TrainConfig field names so the phase drivers can pass the
same cfg dict.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .timemixer import TimeMixer


@dataclass
class TMTrainConfig:
    lookback: int = 96
    horizon: int = 96
    d_model: int = 256
    n_scales: int = 3
    mix_hidden: int = 128
    dropout: float = 0.1
    ma_kernel: int = 25
    lr: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 20
    patience: int = 5
    lr_decay: float = 0.8
    warmup_epochs: int = 3
    seed: int = 0
    device: str = "cpu"


def _loader(X, Y, bs, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=shuffle)


def train_timemixer(train_w, val_w, n_vars: int, cfg: TMTrainConfig, verbose: bool = True):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    Xtr, Ytr, _ = train_w
    Xva, Yva, _ = val_w

    model = TimeMixer(
        n_vars=n_vars, lookback=cfg.lookback, horizon=cfg.horizon, d_model=cfg.d_model,
        n_scales=cfg.n_scales, mix_hidden=cfg.mix_hidden, dropout=cfg.dropout,
        ma_kernel=cfg.ma_kernel,
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
            print(f"[tmx] epoch {ep+1:02d}  val_mse={val:.5f}  lr={lr_now:.2e}  ({time.time()-t0:.1f}s)")
        if val < best_val - 1e-6:
            best_val, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                if verbose:
                    print(f"[tmx] early stop at epoch {ep+1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_val
