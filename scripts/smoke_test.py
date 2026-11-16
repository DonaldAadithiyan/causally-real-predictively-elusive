"""End-to-end smoke test on CPU (< 10 min): ETTh1, one seed, a deliberately tiny configuration.

Exercises every stage with the same code paths as the paper:
  forecaster -> SAE -> dose-response pool + MVDF selection (unchanged frozen_selection)
  -> per-window arrays -> Phase 3 battery -> Phase 4 oracle -> learned scorer vs naive_matched
  -> one E3 cell (feature sets F0, F2) incl. the leakage check.
It checks that the pipeline RUNS and produces finite outputs, NOT the paper's numbers
(those are checked by tests/test_paper_numbers.py against the committed results).

    python scripts/smoke_test.py [--out smoke_out]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="smoke_out")
args = ap.parse_args()
OUT = os.path.abspath(args.out)
ARR = os.path.join(OUT, "arrays")
os.makedirs(ARR, exist_ok=True)
os.environ["CR_ARR_DIR"] = ARR          # experiments.camera_ready.common reads arrays/meta from here

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
from mechconf.paths import REPO_ROOT  # noqa: E402

sys.path.insert(0, REPO_ROOT)
from mechconf.data.datasets import get_splits, make_windows  # noqa: E402
from mechconf.data.shift_windows import three_way_env_masks  # noqa: E402
from mechconf.forecasters.train_itransformer import TrainConfig, train_itransformer  # noqa: E402
from mechconf.representation.layer_sweep import extract_all_taps  # noqa: E402
from mechconf.sae.train_sae import train_sae, encode_all  # noqa: E402
from experiments.round2.regen_arrays import _sparse, _predict  # noqa: E402
from experiments.round2.selection import frozen_selection  # noqa: E402

TINY = dict(lookback=96, horizon=96, d_model=64, n_layers=3, n_heads=4, d_ff=128, epochs=1,
            sae_expansion=8, sae_k=16, sae_steps=300, pool=16, wcap=256, k=6)


def stage(name, t0):
    print(f"[smoke] {name:<34s} ok  ({time.time() - t0:5.0f}s)", flush=True)


def main():
    t0 = time.time()
    ds, seed = "ETTh1", 0
    np.random.seed(seed); torch.manual_seed(seed)
    L, H = TINY["lookback"], TINY["horizon"]
    sd = get_splits(ds, max_channels=48)
    tv = len(sd.channels) - 1
    Xtr, Ytr, _ = make_windows(sd.train, L, H)
    Xva, Yva, _ = make_windows(sd.val, L, H)
    Xte, Yte, end_te = make_windows(sd.test, L, H)
    cfg = TrainConfig(lookback=L, horizon=H, d_model=TINY["d_model"], n_layers=TINY["n_layers"],
                      n_heads=TINY["n_heads"], d_ff=TINY["d_ff"], dropout=0.1, lr=1e-4, batch_size=64,
                      max_epochs=TINY["epochs"], patience=1, lr_decay=0.8, warmup_epochs=0,
                      seed=seed, device="cpu")
    model, val_mse = train_itransformer((Xtr, Ytr, None), (Xva, Yva, None), n_vars=len(sd.channels),
                                        cfg=cfg, verbose=False)
    model.eval(); stage("forecaster", t0)
    tr_act = extract_all_taps(model, Xtr, tv)["pre_head"]
    sae = train_sae(tr_act, expansion=TINY["sae_expansion"], k=TINY["sae_k"], steps=TINY["sae_steps"],
                    batch_size=256, lr=1e-3, resample_every=10 ** 9, resample_until_frac=0.0,
                    resample_cap=0, device="cpu", seed=seed, verbose=False)
    sae.eval(); stage("SAE", t0)
    yhat = _predict(model, Xte)
    act = extract_all_taps(model, Xte, tv)["pre_head"]
    z = encode_all(sae, act)
    env = three_way_env_masks(end_te, len(sd.test))
    sel = np.where(env["E_select"])[0]
    ae = np.abs(yhat[:, :, tv] - Yte[:, :, tv]).astype(np.float64)
    prof = ae[sel].mean(0); prof = prof / prof.mean()
    fs = frozen_selection(model, sae, Xte[sel], Yte[sel], act[sel], (ae[sel] / prof).max(1), prof, tv,
                          seed, TINY["pool"], TINY["wcap"], TINY["k"])
    assert len(fs["mvdf"]) == TINY["k"] and np.isfinite(fs["dose"]["U_j"]).any()
    stage("dose-response + MVDF selection", t0)
    zi, zv = _sparse(z)
    np.savez_compressed(os.path.join(ARR, f"{ds}_seed{seed}.npz"),
                        X=Xte.astype(np.float32), y=Yte[:, :, tv].astype(np.float32),
                        yhat=yhat[:, :, tv].astype(np.float32), hidden=act.astype(np.float32),
                        end_idx=end_te.astype(np.int32), origin_ts=np.full(len(end_te), -1, np.int64),
                        m_select=env["E_select"], m_id=env["E_eval_ID"], m_shift=env["E_eval_shift"],
                        target_var=np.int32(tv), L=np.int32(L), H=np.int32(H), z_idx=zi, z_val=zv,
                        d_sae=np.int32(z.shape[1]), mvdf=fs["mvdf"], correlated=fs["corr"])
    json.dump(dict(tag=f"{ds}_seed{seed}", val_mse=float(val_mse), sae_fvu=float("nan"),
                   mvdf=[int(x) for x in fs["mvdf"]], correlated=[int(x) for x in fs["corr"]]),
              open(os.path.join(ARR, f"{ds}_seed{seed}_meta.json"), "w"))
    stage("per-window arrays", t0)

    from experiments.camera_ready.common import Run
    from experiments.camera_ready.phase34 import phase3, phase4
    run = Run(ds, seed)
    p3, _ = phase3(run); assert np.isfinite(p3["marg_best_r2"])
    stage(f"Phase 3 battery (best R2 {p3['marg_best_r2']:+.2f})", t0)
    p4, _ = phase4(run); assert np.isfinite(p4["ratio_id"])
    stage(f"Phase 4 oracle (gap {100 * p4['ratio_id']:.0f}%)", t0)

    from experiments.camera_ready.e2_scorer import run_cell as scorer_cell
    sc = scorer_cell(ds, seed); assert all(np.isfinite(r["width"]) for r in sc)
    stage("learned scorer vs naive_matched", t0)

    from experiments.camera_ready import e3_exante
    pr = e3_exante.run_cell(ds, seed, os.path.join(OUT, "E3"), blocks=(96,), sets=["F0", "F2"])
    assert len(pr) and pr.R2.notna().any()
    stage("E3 cell (F0, F2) + leakage check", t0)
    total = time.time() - t0
    print(f"[smoke] PASS in {total:.0f}s", flush=True)
    if total > 600:
        print("[smoke] WARNING: slower than the 10-minute target", flush=True)


if __name__ == "__main__":
    main()
