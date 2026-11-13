"""Train a TopK SAE on forecaster hidden states (spec §5.3).

Trained ONLY on the training-set hidden states, disjoint from the conformal
calibration set, so nothing leaks into the coverage guarantee.

Stability (learned the hard way on traffic/weather where the first version diverged
to FVU 0.5-0.95): hidden states are standardized before training; dead-feature
resampling is capped per event and stopped after half of training (Templeton et al.
do the same); gradients are clipped; LR is flat with a short warmup, not cosine.
`encode_all` applies the same standardization so downstream codes are consistent.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .topk_sae import TopKSAE


def _resample_dead(sae: TopKSAE, dead_idx: torch.Tensor, x_batch: torch.Tensor, cap: int):
    """Templeton-style dead-feature resampling, capped: re-point at most `cap` dead
    encoder rows at high-reconstruction-loss inputs."""
    if dead_idx.numel() == 0:
        return
    if dead_idx.numel() > cap:
        perm = torch.randperm(dead_idx.numel(), device=dead_idx.device)[:cap]
        dead_idx = dead_idx[perm]
    with torch.no_grad():
        x_hat, _ = sae(x_batch)
        loss_per = ((x_hat - x_batch) ** 2).mean(-1) + 1e-12
        probs = (loss_per / loss_per.sum()).cpu().numpy()
        pick = np.random.choice(len(x_batch), size=dead_idx.numel(), replace=True, p=probs)
        picks = x_batch[pick]
        picks = picks / (picks.norm(dim=-1, keepdim=True) + 1e-8)
        # scale new encoder rows to the median norm of the ALIVE rows so the resample
        # does not inject an outlier-magnitude direction into a converged SAE.
        alive = torch.ones(sae.d_sae, dtype=torch.bool, device=dead_idx.device)
        alive[dead_idx] = False
        med_norm = sae.W_enc.data[alive].norm(dim=1).median() if alive.any() else torch.tensor(0.1)
        sae.W_enc.data[dead_idx] = picks * (0.2 * med_norm)
        sae.W_dec.data[:, dead_idx] = picks.t()
        sae.b_enc.data[dead_idx] = 0.0
        sae._normalize_decoder()


def train_sae(
    hidden: np.ndarray,
    expansion: int = 8,
    k: int = 64,
    steps: int = 8000,
    batch_size: int = 512,
    lr: float = 1e-4,
    resample_every: int = 2000,
    dead_after: int = 1000,
    resample_until_frac: float = 0.5,
    resample_cap: int = 64,
    warmup_steps: int = 200,
    grad_clip: float = 1.0,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = True,
) -> TopKSAE:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # standardize hidden states (per-dim) for cross-dataset stability
    mu = hidden.mean(0, keepdims=True).astype(np.float32)
    sd = hidden.std(0, keepdims=True).astype(np.float32) + 1e-6
    Hn = ((hidden.astype(np.float32) - mu) / sd)

    d_model = Hn.shape[1]
    ds = TensorDataset(torch.from_numpy(Hn))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    it = iter(dl)

    sae = TopKSAE(d_model, expansion=expansion, k=k).to(device)
    with torch.no_grad():                                   # fill standardization buffers
        sae._input_mean.copy_(torch.from_numpy(mu.squeeze(0)).to(device))
        sae._input_std.copy_(torch.from_numpy(sd.squeeze(0)).to(device))
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    resample_until = int(resample_until_frac * steps)
    last_fired = torch.zeros(sae.d_sae, dtype=torch.long, device=device)
    best = None

    for step in range(steps):
        for g in opt.param_groups:                          # linear warmup
            g["lr"] = lr * min(1.0, (step + 1) / max(1, warmup_steps))
        try:
            (xb,) = next(it)
        except StopIteration:
            it = iter(dl); (xb,) = next(it)
        xb = xb.to(device)
        x_hat, z = sae(xb)
        loss = torch.mean((x_hat - xb) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), grad_clip)
        opt.step()
        sae._normalize_decoder()

        fired = (z != 0).any(0)
        last_fired[fired] = step
        if step > 0 and step % resample_every == 0 and step < resample_until:
            dead_idx = torch.nonzero((step - last_fired) > dead_after, as_tuple=False).squeeze(-1)
            _resample_dead(sae, dead_idx, xb, resample_cap)
            if verbose:
                print(f"[sae] step {step}  mse={loss.item():.5f}  dead={int(dead_idx.numel())}"
                      f"  (resampled<= {resample_cap})")
        elif verbose and step > 0 and step % resample_every == 0:
            n_dead = int(((step - last_fired) > dead_after).sum())
            print(f"[sae] step {step}  mse={loss.item():.5f}  dead={n_dead}  (resample off)")

    sae.eval()
    if verbose:
        with torch.no_grad():
            xb = torch.from_numpy(Hn).to(device)
            xh, zz = sae(xb)
            fvu = (torch.mean((xh - xb) ** 2) / (torch.var(xb) + 1e-12)).item()
        print(f"[sae] final  FVU={fvu:.4f}  l0={float((zz != 0).float().sum(-1).mean()):.1f}"
              f"  dead={int((sae.d_sae - (zz != 0).any(0).sum()).item())}")
    return sae


@torch.no_grad()
def encode_all(sae: TopKSAE, hidden: np.ndarray, device: str = "cpu", batch_size: int = 4096) -> np.ndarray:
    """Sparse codes z for every row of `hidden` (raw hidden space in; SAE
    standardizes internally)."""
    out = []
    for i in range(0, len(hidden), batch_size):
        xb = torch.from_numpy(hidden[i : i + batch_size].astype(np.float32)).to(device)
        out.append(sae.encode(sae.standardize(xb)).cpu().numpy())
    return np.concatenate(out, 0)
