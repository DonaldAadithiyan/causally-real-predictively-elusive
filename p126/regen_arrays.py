"""Paper-126 follow-up, step 1: regenerate frozen artifacts and persist per-window arrays.

No forecaster/SAE weights or per-window arrays were ever cached in this repo (only
summary CSV/JSON). This script re-trains with the ORIGINAL configs and seeds:

  real (ETTh1/ETTh2): experiments/phase3_real_predictability.py CFG, cpu,
      seed = derive_seed(seed, ds); forecaster -> SAE (seed ds_seed);
      quantile forecaster for CQR exactly as experiments/baseline_comparison.py
      (seed ds_seed + 31337).
  synthetic: experiments/phase1_synthetic_control.py run_seed() prefix, cpu, seed=seed.

Reproduction is checked against the saved forecaster_val_mse / sae_fvu
(runs/phase3_real/AGG_phase3.csv). The MVDF set is NOT re-derived: the saved indices in
runs/phase3_real/<ds>_seed<s>/mvdf_selection.json (and runs/phase1_synthetic/seed<s>/)
are frozen input and are only valid if the SAE reproduces; the check is recorded.

Output: runs/p126/arrays/<tag>.npz  (all test windows; masks for E_select/E_eval_ID/shift)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from data.datasets import get_splits, make_windows, load_dataframe  # noqa: E402
from experiments._seed import derive_seed  # noqa: E402
from forecasters.train_itransformer import TrainConfig, train_itransformer  # noqa: E402
from representation.layer_sweep import extract_all_taps  # noqa: E402
from sae.train_sae import train_sae, encode_all  # noqa: E402
from shift.shift_windows import three_way_env_masks  # noqa: E402
from experiments.phase3_real_predictability import CFG as P3CFG  # noqa: E402

OUT = os.environ.get("P126_ARR_OUT", os.path.join(HERE, "runs", "p126", "arrays"))
WEIGHTS = os.environ.get("P126_WEIGHTS")   # round 2: if set, persist state_dicts here


def _save_w(tag, **mods):
    if not WEIGHTS:
        return
    os.makedirs(WEIGHTS, exist_ok=True)
    for k, m in mods.items():
        torch.save(m.state_dict(), os.path.join(WEIGHTS, f"{tag}_{k}.pt"))
torch.set_num_threads(int(os.environ.get("P126_THREADS", "4")))


@torch.no_grad()
def _predict(model, X, bs=512):
    model.eval()
    yh = []
    for i in range(0, len(X), bs):
        p = model(torch.from_numpy(X[i:i + bs]))
        yh.append(p.numpy())
    return np.concatenate(yh, 0)


def _sparse(z):
    """TopK codes -> (idx int16, val float32), k nonzeros per row (pads with -1)."""
    k = int((z != 0).sum(1).max())
    idx = np.argsort(-np.abs(z), axis=1)[:, :k]
    val = np.take_along_axis(z, idx, 1).astype(np.float32)
    idx = np.where(val != 0, idx, -1).astype(np.int16)
    return idx, val


def run_real(ds, seed, with_quantile=True):
    tag = f"{ds}_seed{seed}"
    ds_seed = derive_seed(seed, ds)
    np.random.seed(ds_seed); torch.manual_seed(ds_seed)
    L, H, tvar = P3CFG["lookback"], P3CFG["horizon"], P3CFG["target_var"]
    fc, sae_c = P3CFG["forecaster"], P3CFG["sae"]
    sd = get_splits(ds, max_channels=48)
    tv = tvar % len(sd.channels)
    Xtr, Ytr, _ = make_windows(sd.train, L, H)
    Xva, Yva, _ = make_windows(sd.val, L, H)
    Xte, Yte, end_te = make_windows(sd.test, L, H)
    tcfg = TrainConfig(lookback=L, horizon=H, d_model=fc["d_model"], n_layers=fc["n_layers"],
                       n_heads=fc["n_heads"], d_ff=fc["d_ff"], dropout=fc["dropout"], lr=fc["lr"],
                       batch_size=fc["batch_size"], max_epochs=fc["max_epochs"],
                       patience=fc["patience"], lr_decay=fc["lr_decay"],
                       warmup_epochs=fc["warmup_epochs"], seed=ds_seed, device="cpu")
    model, val_mse = train_itransformer((Xtr, Ytr, None), (Xva, Yva, None),
                                        n_vars=len(sd.channels), cfg=tcfg, verbose=False)
    tr_act = extract_all_taps(model, Xtr, tv)[P3CFG["tap"]]
    sae = train_sae(tr_act, expansion=sae_c["expansion"], k=sae_c["k"], steps=sae_c["steps"],
                    batch_size=sae_c["batch_size"], lr=sae_c["lr"],
                    resample_every=sae_c["resample_every"],
                    resample_until_frac=sae_c["resample_until_frac"],
                    resample_cap=sae_c["resample_cap"], device="cpu", seed=ds_seed, verbose=False)
    with torch.no_grad():
        xb = sae.standardize(torch.from_numpy(tr_act.astype(np.float32)))
        xh, _ = sae(xb)
        fvu = float((torch.mean((xh - xb) ** 2) / (torch.var(xb) + 1e-12)).item())

    yhat = _predict(model, Xte)                              # (N, H, V)
    act = extract_all_taps(model, Xte, tv)[P3CFG["tap"]]     # (N, 256) pre-head target token
    z = encode_all(sae, act)
    zi, zv = _sparse(z)
    env = three_way_env_masks(end_te, len(sd.test))

    # timestamps of each window's forecast origin (last input step)
    df = load_dataframe(ds)
    n = len(df); tr_end, va_end = 12 * 30 * 24, 16 * 30 * 24
    dates = pd.to_datetime(df.iloc[va_end:, 0]).values
    origin_ts = dates[end_te].astype("datetime64[s]").astype(np.int64)

    saved = pd.read_csv(os.path.join(HERE, "runs/phase3_real/AGG_phase3.csv"))
    r = saved[(saved.dataset == ds) & (saved.seed == seed)].iloc[0]
    sel = json.load(open(os.path.join(HERE, f"runs/phase3_real/{tag}/mvdf_selection.json")))
    repro = dict(val_mse=float(val_mse), saved_val_mse=float(r.forecaster_val_mse),
                 fvu=fvu, saved_fvu=float(r.sae_fvu),
                 fingerprint=sd.fingerprint(), saved_fingerprint=r.data_fingerprint,
                 val_mse_match=bool(abs(val_mse - r.forecaster_val_mse) < 1e-6),
                 fvu_match=bool(abs(fvu - r.sae_fvu) < 1e-6))

    extra = {}
    if with_quantile:
        from baselines.quantile_itransformer import (QuantileTrainConfig,
                                                     train_quantile_itransformer,
                                                     predict_quantiles)
        qcfg = QuantileTrainConfig(lookback=L, horizon=H, d_model=fc["d_model"],
                                   n_layers=fc["n_layers"], n_heads=fc["n_heads"],
                                   d_ff=fc["d_ff"], dropout=fc["dropout"], lr=fc["lr"],
                                   batch_size=fc["batch_size"], max_epochs=fc["max_epochs"],
                                   patience=fc["patience"], lr_decay=fc["lr_decay"],
                                   warmup_epochs=fc["warmup_epochs"], seed=ds_seed + 31337,
                                   device="cpu", alpha=0.10)
        qm, _ = train_quantile_itransformer((Xtr, Ytr, None), (Xva, Yva, None),
                                            n_vars=len(sd.channels), cfg=qcfg, verbose=False)
        lo, hi = predict_quantiles(qm, Xte, device="cpu")
        extra = dict(q_lo=lo[:, :, tv].astype(np.float32), q_hi=hi[:, :, tv].astype(np.float32))

    _save_w(tag, forecaster=model, sae=sae, **({"quantile": qm} if with_quantile else {}))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(
        os.path.join(OUT, f"{tag}.npz"),
        X=Xte.astype(np.float32),                     # (N, L, V) normalized inputs, all channels
        y=Yte[:, :, tv].astype(np.float32), yhat=yhat[:, :, tv].astype(np.float32),
        hidden=act.astype(np.float32), z_idx=zi, z_val=zv, d_sae=np.int32(z.shape[1]),
        end_idx=end_te.astype(np.int32), origin_ts=origin_ts,
        m_select=env["E_select"], m_id=env["E_eval_ID"], m_shift=env["E_eval_shift"],
        mvdf=np.array(sel["MVDF_v2"]), correlated=np.array(sel["correlated"]),
        target_var=np.int32(tv), L=np.int32(L), H=np.int32(H), **extra)
    json.dump(repro, open(os.path.join(OUT, f"{tag}_repro.json"), "w"), indent=2)
    print(f"[{tag}] val_mse {val_mse:.10f} (saved {r.forecaster_val_mse:.10f})  "
          f"fvu {fvu:.8f} (saved {r.sae_fvu:.8f})  match={repro['val_mse_match']},{repro['fvu_match']}",
          flush=True)


def run_synth(seed):
    from data.synthetic_dgp import load_config, build_split_series, make_windows_xy
    cfg = load_config()
    tag = f"synth_seed{seed}"
    np.random.seed(seed); torch.manual_seed(seed)
    w, fcp, saep = cfg["windowing"], cfg["forecaster"], cfg["sae"]
    L, H, tv = w["lookback"], w["horizon"], w["target_var"]
    S = build_split_series(cfg, seed)
    mean = S["train"]["x"].mean(); std = S["train"]["x"].std() + 1e-8
    ymean = S["train"]["y"].mean(); ystd = S["train"]["y"].std() + 1e-8
    nz = lambda sp: dict(x=(sp["x"] - mean) / std, y=(sp["y"] - ymean) / ystd, z=sp["z"])
    tr, va, te = nz(S["train"]), nz(S["val"]), nz(S["test"])
    Xtr, Ytr, *_ = make_windows_xy(tr["x"], tr["y"], tr["z"], L, H)
    Xva, Yva, *_ = make_windows_xy(va["x"], va["y"], va["z"], L, H)
    Xte, Yte, Zwin, Zhard, end_te = make_windows_xy(te["x"], te["y"], te["z"], L, H)
    tcfg = TrainConfig(lookback=L, horizon=H, d_model=fcp["d_model"], n_layers=fcp["n_layers"],
                       n_heads=fcp["n_heads"], d_ff=fcp["d_ff"], dropout=fcp["dropout"],
                       lr=fcp["lr"], batch_size=fcp["batch_size"], max_epochs=fcp["max_epochs"],
                       patience=fcp["patience"], lr_decay=fcp["lr_decay"],
                       warmup_epochs=fcp["warmup_epochs"], seed=seed, device="cpu")
    model, val_mse = train_itransformer((Xtr, Ytr, None), (Xva, Yva, None), n_vars=1,
                                        cfg=tcfg, verbose=False)
    tap = fcp["tap"]
    tr_act = extract_all_taps(model, Xtr, tv)[tap]
    sae = train_sae(tr_act, expansion=saep["expansion"], k=saep["k"], steps=saep["steps"],
                    batch_size=saep["batch_size"], lr=saep["lr"],
                    resample_every=saep["resample_every"],
                    resample_until_frac=saep["resample_until_frac"],
                    resample_cap=saep["resample_cap"], device="cpu", seed=seed, verbose=False)
    with torch.no_grad():
        xb = sae.standardize(torch.from_numpy(tr_act.astype(np.float32)))
        xh, _ = sae(xb)
        fvu = float((torch.mean((xh - xb) ** 2) / (torch.var(xb) + 1e-12)).item())
    yhat = _predict(model, Xte)
    act = extract_all_taps(model, Xte, tv)[tap]
    z = encode_all(sae, act)
    zi, zv = _sparse(z)
    env = three_way_env_masks(end_te, len(te["x"]))
    starts = end_te - L + 1
    # true per-step noise scale of y over the horizon, in normalized y units
    sig = np.array([cfg["dgp"]["sigma_easy"], cfg["dgp"]["sigma_hard"]]) / ystd
    zh = np.stack([te["z"][s + L:s + L + H] for s in starts]).astype(np.int8)  # (N,H)
    sel = json.load(open(os.path.join(HERE, f"runs/phase1_synthetic/seed{seed}/mvdf_selection.json")))
    _save_w(tag, forecaster=model, sae=sae)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(
        os.path.join(OUT, f"{tag}.npz"),
        X=Xte.astype(np.float32), y=Yte[:, :, tv].astype(np.float32),
        yhat=yhat[:, :, tv].astype(np.float32), hidden=act.astype(np.float32),
        z_idx=zi, z_val=zv, d_sae=np.int32(z.shape[1]), end_idx=end_te.astype(np.int32),
        m_select=env["E_select"], m_id=env["E_eval_ID"], m_shift=env["E_eval_shift"],
        mvdf=np.array(sel["MVDF_v2"]), correlated=np.array(sel["correlated"]),
        random=np.array(sel.get("random", [])), regime_win=Zwin, regime_hardfrac=Zhard,
        regime_h=zh, sigma_h=sig[zh].astype(np.float32),
        target_var=np.int32(tv), L=np.int32(L), H=np.int32(H))
    pred = json.load(open(os.path.join(HERE, f"runs/phase1_synthetic/seed{seed}/predictability.json")))
    repro = dict(val_mse=float(val_mse), fvu=fvu,
                 saved_marginal_best_r2=pred.get("marginal_best_r2"))
    json.dump(repro, open(os.path.join(OUT, f"{tag}_repro.json"), "w"), indent=2)
    print(f"[{tag}] val_mse {val_mse:.6f} fvu {fvu:.6f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["ETTh1:0"])
    ap.add_argument("--no_quantile", action="store_true")
    a = ap.parse_args()
    for w in a.which:
        t0 = time.time()
        name, s = w.split(":")
        if name == "synth":
            run_synth(int(s))
        else:
            run_real(name, int(s), with_quantile=not a.no_quantile)
        print(f"  done {w} in {time.time()-t0:.0f}s", flush=True)
