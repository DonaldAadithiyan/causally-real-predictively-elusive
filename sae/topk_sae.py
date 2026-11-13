"""TopK sparse autoencoder, Dissecting-Chronos recipe (spec §5.3).

    z = TopK(W_enc (x - b_dec) + b_enc, k)
    x_hat = W_dec z + b_dec

- d_sae = expansion * d_model  (default 8x)
- k retained activations per forward pass (default 64)
- MSE reconstruction loss
- periodic dead-feature resampling (Templeton et al. 2024 style)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, d_model: int, expansion: int = 8, k: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_sae = expansion * d_model
        self.k = k

        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.empty(self.d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_model, self.d_sae))

        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        # tie decoder to encoder at init, then unit-normalize decoder columns
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
            self._normalize_decoder()

        # per-dim standardization stats of the space the SAE operates in.
        # train_sae fills these; identity until then. Registered as buffers so they
        # move with .to(device) and survive state_dict save/load.
        self.register_buffer("_input_mean", torch.zeros(d_model))
        self.register_buffer("_input_std", torch.ones(d_model))

    def _normalize_decoder(self):
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True) + 1e-8)

    # ---- standardization helpers (raw hidden space <-> SAE space) -----------------
    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._input_mean) / self._input_std

    def unstandardize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._input_std + self._input_mean

    @torch.no_grad()
    def reconstruct_raw(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Full encode->decode in raw hidden space (standardize in, unstandardize out)."""
        z = self.encode(self.standardize(x_raw))
        return self.unstandardize(self.decode(z))

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-activation (dense) codes, no TopK. Shape (B, d_sae)."""
        return F.linear(x - self.b_dec, self.W_enc, self.b_enc)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse codes after TopK + ReLU. Shape (B, d_sae)."""
        pre = self.encode_pre(x)
        act = F.relu(pre)
        if self.k >= self.d_sae:
            return act
        topv, topi = torch.topk(act, self.k, dim=-1)
        z = torch.zeros_like(act)
        z.scatter_(-1, topi, topv)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return F.linear(z, self.W_dec, self.b_dec)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    # ---- diagnostics ----------------------------------------------------------------
    @torch.no_grad()
    def reconstruction_stats(self, x_raw: torch.Tensor) -> dict:
        """Stats in RAW hidden space (standardize in, unstandardize out)."""
        xs = self.standardize(x_raw)
        z = self.encode(xs)
        x_hat = self.unstandardize(self.decode(z))
        mse = torch.mean((x_hat - x_raw) ** 2).item()
        var = torch.var(x_raw, unbiased=False).item()
        # also report FVU in the standardized space the SAE actually optimizes
        xh_s = self.decode(z)
        fvu_s = (torch.mean((xh_s - xs) ** 2) / (torch.var(xs) + 1e-12)).item()
        return {
            "mse_raw": mse,
            "fvu_raw": mse / (var + 1e-12),
            "fvu_standardized": fvu_s,
            "l0": (z != 0).float().sum(-1).mean().item(),
        }
