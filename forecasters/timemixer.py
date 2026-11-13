"""TimeMixer-style forecaster — the non-attention architecture for the generalization
gate (Extended wrap-up §3).

The project's forecaster has been iTransformer (attention) throughout. This is a
deliberately different design philosophy: **MLP-based, channel-independent,
multi-resolution mixing** (Wang et al. 2024, "TimeMixer: Decomposable Multiscale Mixing
for Time Series Forecasting"). Not a line-for-line reimplementation — a faithful, minimal
version with the SAME representation interface the SAE / dose-response pipeline needs:

  - a per-variate `pre_head` token of shape (B, V, d_model) that the SAE decomposes;
  - `encode` / `encode_all_taps` filling a tap dict;
  - `continue_from_tap(patched, tap)` -> pre-head rep, so dose-response can patch an
    activation and continue;
  - `forward(x, h_override=, return_hidden=, patch=(tap,tensor))` with the same per-window
    RevIN de-normalization iTransformer uses.

Kept faithful to TimeMixer's core ideas: (1) channel independence — each variate is
processed by the SAME weights; (2) multiscale — the lookback is downsampled to several
resolutions, each mixed by an MLP, then combined; (3) a decomposition (moving-average
trend / seasonal residual) at each scale, mixed separately (Past-Decomposable-Mixing).
The final per-variate representation is the flattened, projected multiscale summary —
this is `pre_head`.

TAP_NAMES here map TimeMixer's stages onto the spec's probe set. Only `pre_head` is used
by dose-response (Phase 0 found no privileged layer; the arch gate re-uses `pre_head` by
design — no layer sweep on a new architecture, that's a scope violation).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _moving_avg(x: torch.Tensor, k: int) -> torch.Tensor:
    """Trend via centered moving average along the time axis. x: (B, C, T)."""
    if k <= 1:
        return x
    pad = k // 2
    xp = F.pad(x, (pad, pad), mode="replicate")
    return F.avg_pool1d(xp, kernel_size=k, stride=1)[..., :x.shape[-1]]


class _ScaleMixer(nn.Module):
    """One resolution: decompose into trend + seasonal, mix each with its own MLP."""

    def __init__(self, length: int, hidden: int, dropout: float, ma_kernel: int):
        super().__init__()
        self.ma_kernel = ma_kernel
        self.trend_mlp = nn.Sequential(
            nn.Linear(length, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, length),
        )
        self.seasonal_mlp = nn.Sequential(
            nn.Linear(length, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, length),
        )
        self.norm = nn.LayerNorm(length)

    def forward(self, x):                                   # x: (B*C, length)
        trend = _moving_avg(x.unsqueeze(1), self.ma_kernel).squeeze(1)
        seasonal = x - trend
        y = self.trend_mlp(trend) + self.seasonal_mlp(seasonal)
        return self.norm(x + y)


class TimeMixer(nn.Module):
    # stages, forward-pass order. Only `pre_head` feeds the SAE (spec §4.5 mapping).
    TAP_NAMES = ["embed", "scale0", "scale1", "scale2", "fused", "pre_head", "head_input"]

    def __init__(
        self,
        n_vars: int,
        lookback: int = 96,
        horizon: int = 96,
        d_model: int = 256,
        n_scales: int = 3,
        mix_hidden: int = 128,
        dropout: float = 0.1,
        ma_kernel: int = 25,
        **_ignore,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.lookback = lookback
        self.horizon = horizon
        self.d_model = d_model
        self.n_scales = n_scales

        # scale s covers length L // 2**s (downsample by average pooling)
        self.scale_lengths = [max(4, lookback // (2 ** s)) for s in range(n_scales)]
        self.mixers = nn.ModuleList([
            _ScaleMixer(L, mix_hidden, dropout, max(3, ma_kernel // (2 ** s) | 1))
            for s, L in enumerate(self.scale_lengths)
        ])
        # each scale -> a d_model//n_scales chunk; concat -> d_model  (the pre_head token)
        chunk = d_model // n_scales
        self.scale_proj = nn.ModuleList([nn.Linear(L, chunk) for L in self.scale_lengths])
        self.fuse = nn.Linear(chunk * n_scales, d_model)
        self.pre_head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon)             # per-variate token -> horizon

        self.enc_dropout = nn.Dropout(dropout)

    # ---- representation ---------------------------------------------------------
    def _downsample(self, x_ct: torch.Tensor, target_len: int) -> torch.Tensor:
        """x_ct: (B*C, L_in) -> (B*C, target_len) via adaptive avg pool."""
        if x_ct.shape[-1] == target_len:
            return x_ct
        return F.adaptive_avg_pool1d(x_ct.unsqueeze(1), target_len).squeeze(1)

    def encode(self, x: torch.Tensor, taps: Optional[dict] = None) -> torch.Tensor:
        """x: (B, lookback, V) RevIN-normalized -> pre_head (B, V, d_model)."""
        B, L, V = x.shape
        xc = x.permute(0, 2, 1).reshape(B * V, L)           # (B*V, L) — channel-independent
        xc = self.enc_dropout(xc)
        if taps is not None:
            taps["embed"] = xc.reshape(B, V, L)             # (B,V,L) — not d_model, informational

        chunks = []
        for s in range(self.n_scales):
            xs = self._downsample(xc, self.scale_lengths[s])
            xs = self.mixers[s](xs)                         # (B*V, L_s)
            if taps is not None:
                taps[f"scale{s}"] = xs.reshape(B, V, -1)
            chunks.append(self.scale_proj[s](xs))           # (B*V, chunk)

        fused = torch.cat(chunks, dim=-1)                   # (B*V, chunk*n_scales)
        if taps is not None:
            taps["fused"] = fused.reshape(B, V, -1)
        h = self.pre_head_norm(self.fuse(fused))            # (B*V, d_model)
        h = h.reshape(B, V, self.d_model)
        if taps is not None:
            taps["pre_head"] = h
            taps["head_input"] = h
        return h

    @torch.no_grad()
    def encode_all_taps(self, x_norm: torch.Tensor) -> dict:
        taps: dict = {}
        self.encode(x_norm, taps=taps)
        # only return keys that exist and are (B,V,*) — pre_head/head_input are (B,V,d_model)
        return {k: taps[k] for k in self.TAP_NAMES if k in taps}

    def continue_from_tap(self, patched: torch.Tensor, tap: str) -> torch.Tensor:
        """Run from `patched` (a rep AT `tap`) to the pre-head rep. dose-response only
        ever passes tap='pre_head' (arch gate re-uses the established tap by design)."""
        if tap in ("pre_head", "head_input"):
            return patched                                  # already the final rep
        # earlier taps would require re-flattening + re-mixing; not needed for the gate.
        raise ValueError(
            f"TimeMixer.continue_from_tap only supports 'pre_head'/'head_input' "
            f"(got {tap!r}); the architecture gate re-uses the established SAE tap and "
            f"does not sweep layers on a new architecture.")

    def forward(
        self,
        x: torch.Tensor,
        h_override: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        patch: Optional[tuple] = None,
    ):
        mean = x.mean(dim=1, keepdim=True)                  # (B,1,V) — per-window RevIN
        std = x.std(dim=1, keepdim=True) + 1e-6
        xn = (x - mean) / std

        if patch is not None:
            tap_name, tensor = patch
            h = self.continue_from_tap(tensor, tap_name)
        elif h_override is not None:
            h = h_override
        else:
            h = self.encode(xn)                             # (B, V, d_model)
        y = self.head(h)                                    # (B, V, horizon)
        y = y.permute(0, 2, 1)                              # (B, horizon, V)
        y = y * std + mean
        if return_hidden:
            return y, h
        return y
