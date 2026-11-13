"""Round 2 WP6: further real datasets, run locally (the recovered Kaggle-bundle raw files are
in ~/Downloads/results (3)/mechanistic_conformal/data/raw; electricity/traffic are the
byte-exact originals per kaggle_ext/KAGGLE_README.md). Pointed at via KAGGLE_DATA_DIR so
the repo's own adapters load them unchanged.

  regen   : forecaster (+ SAE, frozen MVDF selection, CQR quantile model) with weights and
            per-window arrays saved in the same format as the ETTh runs.
  stepzero: round-1 item-B machinery + like-for-like parity + the section-13 rule.

Deviation recorded: solar.csv carries a pandas index column 'Unnamed: 0'; it is dropped
(the generic numeric coercion would otherwise keep it as a 'channel').
Device: MC_DEVICE (default cpu); new datasets are fresh runs, not reproductions, so MPS is
allowed for training. Everything after training runs on CPU."""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
HERE0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R2 = os.path.join(HERE0, "runs", "p126_round2")
os.environ.setdefault("P126_ARR", os.path.join(R2, "arrays"))
os.environ.setdefault("KAGGLE_DATA_DIR", os.path.expanduser(
    "~/Downloads/results (3)/mechanistic_conformal/data/raw"))
torch.set_num_threads(int(os.environ.get("P126_THREADS", "4")))
DEV = os.environ.get("MC_DEVICE", "cpu")

import data.datasets as dsmod  # noqa: E402
from data.datasets import make_windows  # noqa: E402
from data.fingerprint import data_fingerprint  # noqa: E402
from experiments._seed import derive_seed  # noqa: E402
from experiments.phase3_real_predictability import CFG as P3CFG  # noqa: E402
from forecasters.train_itransformer import TrainConfig, train_itransformer  # noqa: E402
from representation.layer_sweep import extract_all_taps  # noqa: E402
from sae.train_sae import train_sae, encode_all  # noqa: E402
from shift.shift_windows import three_way_env_masks  # noqa: E402
from p126.regen_arrays import _sparse, _predict  # noqa: E402
from p126.selection import frozen_selection  # noqa: E402
from p126.common import dump  # noqa: E402

ARR, WTS, OUT = os.path.join(R2, "arrays"), os.path.join(R2, "weights"), os.path.join(R2, "wp6")
for d in (ARR, WTS, OUT):
    os.makedirs(d, exist_ok=True)

_orig_load = dsmod.load_dataframe


def _load_df(name, path=None):
    df = _orig_load(name, path)
    drop = [c for c in df.columns if str(c).startswith("Unnamed")]
    return df.drop(columns=drop) if drop else df


dsmod.load_dataframe = _load_df


def split_points(name, n):
    if name in dsmod._ETT_MONTHS:
        step = dsmod._ETT_MONTHS[name]
        tr, va = 12 * 30 * step, 16 * 30 * step
        if va >= n:
            tr, va = int(0.7 * n), int(0.8 * n)
    else:
        tr, va = int(0.7 * n), int(0.8 * n)
    return tr, va


def regen(ds, seed, with_quantile=True, with_selection=True):
    tag = f"{ds}_seed{seed}"
    ds_seed = derive_seed(seed, ds)
    np.random.seed(ds_seed); torch.manual_seed(ds_seed)
    L, H, tvar = P3CFG["lookback"], P3CFG["horizon"], P3CFG["target_var"]
    fc, sae_c = P3CFG["forecaster"], P3CFG["sae"]
    sd = dsmod.get_splits(ds, max_channels=48)
    tv = tvar % len(sd.channels)
    Xtr, Ytr, _ = make_windows(sd.train, L, H)
    Xva, Yva, _ = make_windows(sd.val, L, H)
    Xte, Yte, end_te = make_windows(sd.test, L, H)
    t0 = time.time()
    tcfg = TrainConfig(lookback=L, horizon=H, d_model=fc["d_model"], n_layers=fc["n_layers"],
                       n_heads=fc["n_heads"], d_ff=fc["d_ff"], dropout=fc["dropout"], lr=fc["lr"],
                       batch_size=fc["batch_size"], max_epochs=fc["max_epochs"],
                       patience=fc["patience"], lr_decay=fc["lr_decay"],
                       warmup_epochs=fc["warmup_epochs"], seed=ds_seed, device=DEV)
    model, val_mse = train_itransformer((Xtr, Ytr, None), (Xva, Yva, None),
                                        n_vars=len(sd.channels), cfg=tcfg, verbose=False)
    model = model.cpu().eval()
    print(f"[{tag}] forecaster {time.time()-t0:.0f}s val_mse {val_mse:.4f}", flush=True)
    yhat = _predict(model, Xte)
    act = extract_all_taps(model, Xte, tv)[P3CFG["tap"]]
    env = three_way_env_masks(end_te, len(sd.test))
    df = dsmod.load_dataframe(ds)
    tr_end, va_end = split_points(ds, len(df))
    try:
        dates = pd.to_datetime(df.iloc[va_end:, 0]).values
        origin_ts = dates[end_te].astype("datetime64[s]").astype(np.int64)
    except Exception:
        origin_ts = np.full(len(end_te), -1, np.int64)
    extra, meta = {}, dict(tag=tag, val_mse=float(val_mse), n_channels=len(sd.channels),
                           fingerprint=data_fingerprint(ds, sd), device=DEV,
                           n_windows=dict(train=len(Xtr), val=len(Xva), test=len(Xte)))
    torch.save(model.state_dict(), os.path.join(WTS, f"{tag}_forecaster.pt"))
    if with_selection:
        t0 = time.time()
        tr_act = extract_all_taps(model, Xtr, tv)[P3CFG["tap"]]
        sae = train_sae(tr_act, expansion=sae_c["expansion"], k=sae_c["k"], steps=sae_c["steps"],
                        batch_size=sae_c["batch_size"], lr=sae_c["lr"],
                        resample_every=sae_c["resample_every"],
                        resample_until_frac=sae_c["resample_until_frac"],
                        resample_cap=sae_c["resample_cap"], device=DEV, seed=ds_seed, verbose=False)
        sae = sae.cpu().eval()
        with torch.no_grad():
            xb = sae.standardize(torch.from_numpy(tr_act.astype(np.float32)))
            xh, _ = sae(xb)
            meta["sae_fvu"] = float((torch.mean((xh - xb) ** 2) / (torch.var(xb) + 1e-12)).item())
        z = encode_all(sae, act)
        sel = np.where(env["E_select"])[0]
        ae = np.abs(yhat[:, :, tv] - Yte[:, :, tv]).astype(np.float64)
        prof = ae[sel].mean(0); prof = prof / prof.mean()
        fs = frozen_selection(model, sae, Xte[sel], Yte[sel], act[sel], (ae[sel] / prof).max(1),
                              prof, tv, ds_seed, P3CFG["candidate_pool_size"], P3CFG["window_cap"],
                              P3CFG["feature_budget_k"])
        zi, zv = _sparse(z)
        extra.update(z_idx=zi, z_val=zv, d_sae=np.int32(z.shape[1]), mvdf=fs["mvdf"],
                     correlated=fs["corr"])
        torch.save(sae.state_dict(), os.path.join(WTS, f"{tag}_sae.pt"))
        pd.DataFrame(dict(feature=fs["pool"], P_j=fs["dose"]["P_j"], U_j=fs["dose"]["U_j"],
                          R_j=fs["R"], n_active=fs["dose"]["n_active"],
                          in_mvdf=np.isin(fs["pool"], fs["mvdf"]),
                          type=[next((t for t, ids in fs["ftab"]["cells"].items() if int(j) in ids), "?")
                                for j in fs["pool"]])).to_csv(os.path.join(OUT, f"{tag}_candidates.csv"),
                                                              index=False)
        meta.update(mvdf=fs["mvdf"].tolist(), correlated=fs["corr"].tolist(),
                    enrichment_typeB=fs["enr"].get("enrichment_ratio"),
                    n_gate1=int(fs["gate1"].sum()))
        print(f"[{tag}] SAE+selection {time.time()-t0:.0f}s fvu {meta['sae_fvu']:.4f}", flush=True)
    if with_quantile:
        from baselines.quantile_itransformer import (QuantileTrainConfig,
                                                     train_quantile_itransformer, predict_quantiles)
        qcfg = QuantileTrainConfig(lookback=L, horizon=H, d_model=fc["d_model"],
                                   n_layers=fc["n_layers"], n_heads=fc["n_heads"], d_ff=fc["d_ff"],
                                   dropout=fc["dropout"], lr=fc["lr"], batch_size=fc["batch_size"],
                                   max_epochs=fc["max_epochs"], patience=fc["patience"],
                                   lr_decay=fc["lr_decay"], warmup_epochs=fc["warmup_epochs"],
                                   seed=ds_seed + 31337, device=DEV, alpha=0.10)
        qm, _ = train_quantile_itransformer((Xtr, Ytr, None), (Xva, Yva, None),
                                            n_vars=len(sd.channels), cfg=qcfg, verbose=False)
        qm = qm.cpu().eval()
        lo, hi = predict_quantiles(qm, Xte, device="cpu")
        extra.update(q_lo=lo[:, :, tv].astype(np.float32), q_hi=hi[:, :, tv].astype(np.float32))
        torch.save(qm.state_dict(), os.path.join(WTS, f"{tag}_quantile.pt"))
    if "z_idx" not in extra:   # forecaster-only screen: placeholders so common.Run loads
        extra.update(z_idx=np.full((len(Xte), 1), -1, np.int16), z_val=np.zeros((len(Xte), 1), np.float32),
                     d_sae=np.int32(1), mvdf=np.array([], int), correlated=np.array([], int))
    np.savez_compressed(os.path.join(ARR, f"{tag}.npz"),
                        X=Xte.astype(np.float32), y=Yte[:, :, tv].astype(np.float32),
                        yhat=yhat[:, :, tv].astype(np.float32), hidden=act.astype(np.float32),
                        end_idx=end_te.astype(np.int32), origin_ts=origin_ts,
                        m_select=env["E_select"], m_id=env["E_eval_ID"], m_shift=env["E_eval_shift"],
                        target_var=np.int32(tv), L=np.int32(L), H=np.int32(H), **extra)
    dump(meta, os.path.join(OUT, f"{tag}_meta.json"))
    return meta


def stepzero(ds, seed):
    """Round-1 item B (forecast-time predictors only) + like-for-like parity + rule 13."""
    import p126.work_b as wb
    from p126.r2_wp0 import H as _H  # noqa
    wb.BOUT = os.path.join(OUT, "stepzero"); os.makedirs(wb.BOUT, exist_ok=True)
    rows, meta = wb.run(ds, seed, embargo=True)
    tag = f"{ds}_seed{seed}"
    pd.DataFrame(rows).to_csv(os.path.join(wb.BOUT, f"{tag}_B.csv"), index=False)
    dump(meta, os.path.join(wb.BOUT, f"{tag}_meta.json"))
    return parity_rule(ds, seed, wb.BOUT)


def parity_rule(ds, seed, bdir):
    """Like-for-like (static vs static constant band on the SAME cal segment; delayed ACI vs
    delayed ACI) for the inner-fold-selected best forecast-time predictor; rule 13."""
    from p126.common import conf_q, boot_ci, spearman, matched_width
    from p126.corrected import online
    tag = f"{ds}_seed{seed}"
    z = np.load(os.path.join(bdir, f"{tag}_emb_preds.npz"))
    meta = json.load(open(os.path.join(bdir, f"{tag}_meta.json")))
    best = meta["best_nonrep"]
    S, cal = z["S"], z["cal"]
    H = 96
    res = dict(dataset=ds, seed=seed, best_predictor=best)
    for seg, te in (("ID", z["te_id"]), ("shift", z["te_sh"])):
        g = z[f"g_{best}"].astype(float)
        gc = np.where(np.isfinite(g[cal]), g[cal], np.nanmedian(g[cal]))
        gt = np.where(np.isfinite(g[te]), g[te], np.nanmedian(gc))
        S_te, S_cal = S[te], S[cal]
        if np.any(gc <= 0):
            gc, gt = gc - gc.min() + 1e-3, gt - gc.min() + 1e-3
        ratio = S_cal / gc
        for kind in ("static", "aci"):
            bfn = lambda al: online(conf_q(S_cal, 1 - al), S_te, al, delay=H, kind=kind)["hw"]
            fn = lambda al: online(conf_q(ratio, 1 - al) * gt, S_te, al, delay=H, kind=kind)["hw"]
            bhw = bfn(0.1); bcov = float(np.mean(S_te <= bhw))
            Wm, al_m, _ = matched_width(fn, S_te, bcov, grid=np.linspace(0.01, 0.4, 40))
            hw_m = fn(al_m)
            red = (np.mean(2 * bhw) - Wm) / np.mean(2 * bhw)
            lo, hi, nbk = boot_ci(lambda ix: (np.mean(bhw[ix]) - np.mean(hw_m[ix])) / np.mean(bhw[ix]),
                                  len(te), 2 * H, 2000, seed)
            slo, shi, _ = boot_ci(lambda ix: spearman(gt[ix], S_te[ix]), len(te), 2 * H, 2000, seed + 1)
            res[f"{seg}_{kind}"] = dict(red=float(red), red_ci=[lo, hi], spearman=spearman(gt, S_te),
                                        spearman_ci=[slo, shi], n_blocks=nbk, base_cov=bcov,
                                        passes=bool(lo > 0 and slo > 0))
    return res


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "regen":
        flags = dict(with_quantile=("--noq" not in sys.argv), with_selection=("--nosel" not in sys.argv))
        for spec in [a for a in sys.argv[2:] if not a.startswith("--")]:
            ds, s = spec.split(":")
            print(regen(ds, int(s), **flags), flush=True)
    elif what == "stepzero":
        res = []
        for spec in sys.argv[2:]:
            ds, s = spec.split(":")
            res.append(stepzero(ds, int(s)))
            print(json.dumps(res[-1], default=float), flush=True)
            dump(res, os.path.join(OUT, f"stepzero_{'_'.join(a.replace(':','') for a in sys.argv[2:])}.json"))
