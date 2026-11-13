"""Briefly train the SyntheticForecaster so its hidden state is a real learned
representation (not a hand-built basis) for the SAE to decompose.

Loss = point MSE
     + lambda_mech * MSE(aux_mech(mech_block), difficulty)
     + lambda_spur * MSE(aux_spur(spur_block), spurious_covariate)

The auxiliary terms only *shape* the representation; they are dropped at inference.
After each step the spurious block is re-zeroed out of the head so the head genuinely
does not depend on it (ablation ground truth).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .synthetic_forecaster import SyntheticForecaster


def train_forecaster(
    X: np.ndarray,
    y: np.ndarray,
    aux: dict,
    hidden_dim: int = 256,
    mech_dim: int = 8,
    spur_dim: int = 8,
    epochs: int = 8,
    batch_size: int = 256,
    lr: float = 1e-3,
    lambda_mech: float = 1.0,
    lambda_spur: float = 1.0,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = True,
) -> SyntheticForecaster:
    torch.manual_seed(seed)
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
        torch.from_numpy(aux["difficulty"]),
        torch.from_numpy(aux["spurious"]),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = SyntheticForecaster(
        lookback=X.shape[1], hidden_dim=hidden_dim, mech_dim=mech_dim, spur_dim=spur_dim
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        tot, n = 0.0, 0
        for xb, yb, db, sb in dl:
            xb, yb, db, sb = xb.to(device), yb.to(device), db.to(device), sb.to(device)
            yhat, h = model(xb)
            mech = model.aux_mech(h[:, : model.mech_dim]).squeeze(-1)
            spur = model.aux_spur(
                h[:, model.mech_dim : model.mech_dim + model.spur_dim]
            ).squeeze(-1)
            loss = (
                torch.mean((yhat - yb) ** 2)
                + lambda_mech * torch.mean((mech - db) ** 2)
                + lambda_spur * torch.mean((spur - sb) ** 2)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.freeze_spurious_head()
            tot += loss.item() * len(xb)
            n += len(xb)
        if verbose:
            print(f"[forecaster] epoch {ep + 1}/{epochs}  loss={tot / max(n, 1):.4f}")
    model.eval()
    return model


@torch.no_grad()
def forecaster_outputs(model: SyntheticForecaster, X: np.ndarray, device: str = "cpu"):
    """Return (yhat, hidden) as numpy for a batch of windows."""
    xb = torch.from_numpy(X).to(device)
    yhat, h = model(xb)
    return yhat.cpu().numpy(), h.cpu().numpy()
