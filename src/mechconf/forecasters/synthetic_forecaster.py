"""Synthetic base forecaster + synthetic time series generator.

Purpose (spec §7): give the SAE -> ablation -> MVDF -> Mondrian-per-bin -> shift-eval
pipeline something to run against, with a *known* injected difficulty mechanism and a
*pre-registered* distribution shift, so H1/H2/H3 can be exercised end to end before a
real forecaster (iTransformer/FFDCI stack) is dropped in.

Design of the synthetic world
-----------------------------
There is a latent "difficulty" process d_t in [0, 1]. It genuinely drives forecast
error: the target's aleatoric noise scale is sigma(d_t). The forecaster's hidden state
h_t contains:

  - a "mechanistic" coordinate that the prediction head actually reads and that also
    tracks d_t  -> ablating it changes the error (large |Delta_j|), and it correlates
    with |e|.  This is what a real MVDF should look like.

  - a "spurious" coordinate that correlates with d_t (hence with |e|) *in the
    calibration distribution only*, because it is a function of a covariate whose
    relationship to d_t is a property of the pre-shift regime. The head does NOT read
    it -> ablating it barely moves the error (small |Delta_j|). Under the shift its
    correlation with |e| collapses. This is the "correlated but not mechanistic"
    feature H1 predicts and H3 exploits.

  - noise coordinates.

The forecaster is a small trained MLP: input window -> hidden h_t -> linear head ->
point prediction. We train it briefly so the hidden state is a real learned
representation the SAE can decompose, not a hand-built basis.

Nothing here is a scientific claim. It is a controlled scaffold whose ground truth we
know, used to check the machinery computes what the spec says it should.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------------------
# Synthetic time series
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class SeriesConfig:
    n_timesteps: int = 12000
    lookback: int = 96
    horizon: int = 1          # MVE uses 1-step; spec's 96-step is a later scale-up
    seed: int = 0

    # difficulty process
    diff_ar: float = 0.95     # AR(1) coefficient of the latent difficulty process
    diff_noise: float = 0.15

    # error scale as a function of difficulty d in [0,1]
    sigma_floor: float = 0.05
    sigma_span: float = 1.2   # sigma(d) = sigma_floor + sigma_span * d

    # signal
    signal_freqs: tuple = (1 / 24.0, 1 / 168.0)
    signal_amp: float = 1.0

    # spurious covariate: c_t = rho * d_t + noise  in the PRE-shift regime.
    # Under shift this coupling is broken (see shift/shift_windows.py which drives the
    # `regime` argument to generate()).
    spurious_rho_preshift: float = 0.9
    spurious_rho_postshift: float = 0.0
    spurious_noise: float = 0.3


def _ar1(n: int, phi: float, noise: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise * rng.standard_normal()
    return x


def generate_series(cfg: SeriesConfig, regime: str = "preshift") -> dict:
    """Return dict with raw arrays. `regime` selects the spurious-coupling strength.

    Keys: y (target), signal, difficulty (d_t in [0,1]), spurious (c_t), sigma (per-t
    aleatoric scale). All length cfg.n_timesteps.
    """
    rng = np.random.default_rng(cfg.seed if regime == "preshift" else cfg.seed + 10_000)
    n = cfg.n_timesteps
    t = np.arange(n)

    # deterministic-ish signal
    signal = cfg.signal_amp * sum(np.sin(2 * np.pi * f * t + i) for i, f in enumerate(cfg.signal_freqs))

    # latent difficulty in [0,1] via squashed AR(1)
    raw_d = _ar1(n, cfg.diff_ar, cfg.diff_noise, rng)
    d = 1.0 / (1.0 + np.exp(-2.0 * (raw_d - raw_d.mean()) / (raw_d.std() + 1e-8)))

    # spurious covariate coupled to d only in the pre-shift regime
    rho = cfg.spurious_rho_preshift if regime == "preshift" else cfg.spurious_rho_postshift
    spurious = rho * (d - d.mean()) + cfg.spurious_noise * rng.standard_normal(n)

    sigma = cfg.sigma_floor + cfg.sigma_span * d
    noise = sigma * rng.standard_normal(n)
    y = signal + noise
    return dict(y=y, signal=signal, difficulty=d, spurious=spurious, sigma=sigma)


def make_windows(series: dict, cfg: SeriesConfig):
    """Slice into (X, y_target, aux) supervised windows.

    X       : (N, lookback)   past target values
    y_true  : (N,)            value at t (horizon=1) -- the thing to forecast
    aux     : dict of per-window latent quantities aligned to the prediction index,
              used only for diagnostics / injecting the mechanism, never fed to the model
    """
    y = series["y"]
    L, H = cfg.lookback, cfg.horizon
    n = len(y)
    idx = np.arange(L, n - H + 1)
    X = np.stack([y[i - L:i] for i in idx]).astype(np.float32)
    y_true = y[idx + H - 1].astype(np.float32)
    aux = {
        "difficulty": series["difficulty"][idx + H - 1].astype(np.float32),
        "spurious": series["spurious"][idx + H - 1].astype(np.float32),
        "sigma": series["sigma"][idx + H - 1].astype(np.float32),
        "signal": series["signal"][idx + H - 1].astype(np.float32),
        "pred_index": idx + H - 1,
    }
    return X, y_true, aux


# --------------------------------------------------------------------------------------
# Synthetic forecaster
# --------------------------------------------------------------------------------------
class SyntheticForecaster(nn.Module):
    """MLP: window -> hidden h_t -> linear head -> point prediction.

    The hidden state is deliberately structured so the pipeline has ground truth:

      dims [0:mech_dim]              -> "mechanistic block": head reads these AND they
                                        are driven toward the difficulty signal during
                                        training via an auxiliary target.
      dims [mech_dim:mech_dim+spur]  -> "spurious block": head does NOT read these
                                        (head weights forced to ~0 there); they are
                                        driven toward the spurious covariate.
      remaining dims                 -> free capacity / noise.

    `hidden_dim` matches the spec's d_model=256 default so the SAE (8x -> 2048) is sized
    as in Dissecting Chronos.
    """

    def __init__(
        self,
        lookback: int = 96,
        hidden_dim: int = 256,
        mech_dim: int = 8,
        spur_dim: int = 8,
        horizon: int = 1,
    ):
        super().__init__()
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.mech_dim = mech_dim
        self.spur_dim = spur_dim
        self.horizon = horizon

        self.encoder = nn.Sequential(
            nn.Linear(lookback, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, horizon)

        # auxiliary probes used ONLY during training to shape the representation
        self.aux_mech = nn.Linear(mech_dim, 1)   # predicts difficulty from mech block
        self.aux_spur = nn.Linear(spur_dim, 1)   # predicts spurious covariate from spur block

        # zero the head's weights on the spurious block and keep them zero
        with torch.no_grad():
            self.head.weight[:, mech_dim:mech_dim + spur_dim] = 0.0
        self._spur_slice = slice(mech_dim, mech_dim + spur_dim)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(x))

    def forward(self, x: torch.Tensor, h_override: Optional[torch.Tensor] = None):
        h = self.hidden(x) if h_override is None else h_override
        y = self.head(h)
        if self.horizon == 1:
            y = y.squeeze(-1)
        return y, h

    def freeze_spurious_head(self):
        """Call after every optimizer step: the head must never learn to read the
        spurious block, or the ablation ground truth breaks."""
        with torch.no_grad():
            self.head.weight[:, self._spur_slice] = 0.0
