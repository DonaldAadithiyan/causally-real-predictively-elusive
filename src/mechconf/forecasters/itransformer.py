"""iTransformer (Liu et al., ICLR 2024) — the FFDCI backbone (spec §5.2).

Inverted transformer: each *variate*'s whole lookback series is embedded as one token,
attention runs across variates, a shared MLP head maps each variate token to its
horizon forecast.

Config follows FFDCI's fixed setup (not hyperparameter-searched):
  lookback 96 -> horizon 96, d_model 256, 3 encoder layers.

Tap point for the SAE / FFDCI head (spec §5.2): the per-variate token representation
straight out of the encoder stack, just before the projection head. `forward` can
return it, and can also run with an externally supplied hidden tensor (`h_override`)
so the ablation protocol (spec §5.4) can zero an SAE feature, decode, and patch the
reconstructed hidden state back through the head.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class _EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, taps: dict | None = None, attn_key=None, ffn_key=None):
        """x: (B, V, d_model). If `taps` is given, store the post-attention rep under
        `attn_key` and the layer output under `ffn_key`."""
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + self.drop(a))
        if taps is not None and attn_key is not None:
            taps[attn_key] = x
        x = self.ln2(x + self.drop(self.ff(x)))
        if taps is not None and ffn_key is not None:
            taps[ffn_key] = x
        return x

    def forward_from_ffn(self, attn_out):
        """Continue a layer from its stored `attn_out` tap through the FFN sublayer."""
        return self.ln2(attn_out + self.drop(self.ff(attn_out)))


class ITransformer(nn.Module):
    def __init__(
        self,
        n_vars: int,
        lookback: int = 96,
        horizon: int = 96,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.lookback = lookback
        self.horizon = horizon
        self.d_model = d_model

        self.embed = nn.Linear(lookback, d_model)          # per-variate series -> token
        self.enc_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [_EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon)            # token -> horizon forecast

    # ---- tap-point registry ------------------------------------------------------
    # Names map the v2 spec §4.5 probe set onto this 3-layer iTransformer.
    # Order is forward-pass order; each name is a point where a (B,V,d_model) tensor
    # can be read and (for dose_response) patched with forward-continuation.
    TAP_NAMES = [
        "embed",       # after embed + enc_dropout (before any encoder layer)
        "layer1",      # after encoder layer 1 (ffn_out of layer 1)
        "layer2",      # after encoder layer 2
        "layer3_attn", # attn_out of the final encoder layer (post attn+residual+ln1)
        "layer3_ffn",  # ffn_out of the final encoder layer (== pre-norm final rep)
        "pre_head",    # after self.norm  — Run 5's location (the SAE/ablation tap)
        "head_input",  # residual stream entering the head (identical tensor to pre_head
                       # here, kept as a named alias for the spec's "residual_stream")
    ]

    # ---- representation -------------------------------------------------------------
    def encode(self, x: torch.Tensor, taps: Optional[dict] = None) -> torch.Tensor:
        """x: (B, lookback, V) already non-stationary-normalized -> hidden (B, V, d_model).
        If `taps` (a dict) is passed, it is filled with every TAP_NAMES activation."""
        xv = x.permute(0, 2, 1)                            # (B, V, lookback)
        h = self.enc_dropout(self.embed(xv))
        if taps is not None:
            taps["embed"] = h
        for i, layer in enumerate(self.layers):
            is_last = (i == len(self.layers) - 1)
            if is_last:
                h = layer(h, taps=taps, attn_key="layer3_attn", ffn_key="layer3_ffn")
            else:
                h = layer(h)
                if taps is not None:
                    taps[f"layer{i+1}"] = h
        h_norm = self.norm(h)
        if taps is not None:
            taps["pre_head"] = h_norm
            taps["head_input"] = h_norm
        return h_norm

    @torch.no_grad()
    def encode_all_taps(self, x_norm: torch.Tensor) -> dict:
        """Return {tap_name: (B,V,d_model)} for every TAP_NAMES point."""
        taps: dict = {}
        self.encode(x_norm, taps=taps)
        return {k: taps[k] for k in self.TAP_NAMES}

    def continue_from_tap(self, patched: torch.Tensor, tap: str) -> torch.Tensor:
        """Run the network from `patched` (a representation AT `tap`) to the final
        pre-head representation self.norm(...). Used by dose_response to measure the
        downstream effect of perturbing an activation at an arbitrary layer."""
        h = patched
        li = {"embed": 0, "layer1": 1, "layer2": 2}.get(tap)
        if li is not None:
            for layer in self.layers[li:]:
                h = layer(h)
            return self.norm(h)
        if tap == "layer3_attn":
            h = self.layers[-1].forward_from_ffn(h)
            return self.norm(h)
        if tap == "layer3_ffn":
            return self.norm(h)
        if tap in ("pre_head", "head_input"):
            return h                                       # already the final rep
        raise ValueError(f"unknown tap {tap!r}; valid: {self.TAP_NAMES}")

    def forward(
        self,
        x: torch.Tensor,
        h_override: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        patch: Optional[tuple] = None,
    ):
        """
        h_override : replace the FINAL pre-head representation (Run 5 behaviour).
        patch=(tap_name, tensor) : replace the representation at `tap_name` with
            `tensor`, then continue the network from there. Mutually exclusive with
            h_override. Returns the same de-normalized forecast either way.
        """
        mean = x.mean(dim=1, keepdim=True)                 # (B,1,V)
        std = x.std(dim=1, keepdim=True) + 1e-6
        xn = (x - mean) / std

        if patch is not None:
            tap_name, tensor = patch
            h = self.continue_from_tap(tensor, tap_name)
        elif h_override is not None:
            h = h_override
        else:
            h = self.encode(xn)
        y = self.head(h)                                   # (B, V, horizon)
        y = y.permute(0, 2, 1)                             # (B, horizon, V)
        y = y * std + mean
        if return_hidden:
            return y, h
        return y


@torch.no_grad()
def collect_hidden(model: ITransformer, X: torch.Tensor, batch_size: int = 512, device: str = "cpu"):
    """Run the encoder over a stack of windows, return per-variate hidden tokens
    flattened to (N * V, d_model) plus the (N,) -> which-variate bookkeeping.

    The SAE trains on individual variate tokens (spec: SAE decomposes the hidden
    representation; each variate token is one d_model vector).
    """
    model.eval()
    hs = []
    for i in range(0, len(X), batch_size):
        xb = X[i : i + batch_size].to(device)
        mean = xb.mean(dim=1, keepdim=True)
        std = xb.std(dim=1, keepdim=True) + 1e-6
        h = model.encode((xb - mean) / std)               # (b, V, d_model)
        hs.append(h.cpu())
    H = torch.cat(hs, 0)                                   # (N, V, d_model)
    return H
