"""The E3 hard leakage rule: no ex-ante feature for window t may use a target with timestamp > t.

Runs on a synthetic Run-like object (always), and on the real ETTh1/traffic arrays when the
release assets are present (see artifacts/MANIFEST.sha256)."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.camera_ready import features as F
from experiments.camera_ready.test_leakage import check


def fake_run(n=1300, H=96, L=96, V=3, seed=0):
    g = np.random.default_rng(seed)
    y = g.normal(size=(n, H)).astype(np.float32)
    yhat = (y + g.normal(scale=0.5, size=(n, H))).astype(np.float32)
    ae = np.abs(y - yhat).astype(np.float64)
    prof = ae[:400].mean(0); prof = prof / prof.mean()
    return SimpleNamespace(tag="synthetic", X=g.normal(size=(n, L, V)).astype(np.float32), y=y, yhat=yhat,
                           abs_err=ae, prof_sel=prof, S=(ae / prof).max(1), tv=V - 1, H=H,
                           end_idx=np.arange(n) + L - 1)


def test_leakage_synthetic():
    r = fake_run()
    rows = np.array([F.MIN_T, F.MIN_T + 7, 900, 1299])
    assert check(r, rows, np.random.default_rng(0))


def test_leakage_negative_control(monkeypatch):
    r = fake_run()
    monkeypatch.setattr(F, "F1_LAGS", (95,) + F.F1_LAGS)        # S_{t-95} uses target time t+1
    with pytest.raises(AssertionError):
        check(r, np.array([F.MIN_T + 10]), np.random.default_rng(0))


@pytest.mark.parametrize("ds", ["ETTh1", "traffic"])
def test_leakage_real(ds):
    from experiments.camera_ready.common import arr_path, Run
    if not os.path.exists(arr_path(f"{ds}_seed0")):
        pytest.skip("real per-window arrays not downloaded (release assets)")
    run = Run(ds, 0)
    rows = np.unique(np.concatenate([run.idx_id[:3], run.idx_sh[-2:]]))
    assert check(run, rows[rows >= F.MIN_T], np.random.default_rng(0))
