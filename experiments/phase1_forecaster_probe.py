"""Pre-full-run check: does a properly-trained forecaster's hidden state encode the
latent regime z_t at all?

The DGP gives the forecaster a reason to infer z only through the regime-dependent AR
coefficient a_{z}: the optimal H-step point forecast of x differs between regimes, so
a loss-minimizing forecaster that tracks local autocorrelation should encode regime in
its hidden state. If a fully-trained forecaster STILL doesn't (linear-probe accuracy
~chance, hidden-state corr with z ~0), that's a DGP problem to diagnose now, not a
pipeline problem to discover mid-run.

No SAE, no dose-response — just: train the real ITransformer on the frozen config for
the full epoch budget, then probe the pre-head hidden state for z_t with a logistic
regression (held-out) and report corr of the top hidden dims / PCs with z and with the
horizon hard-fraction.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from data.synthetic_dgp import load_config, build_split_series, make_windows_xy
from forecasters.train_itransformer import TrainConfig, train_itransformer
from representation.layer_sweep import extract_all_taps
from validation.errors import collect

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(seed=0, epochs=None):
    cfg = load_config()
    w = cfg["windowing"]; fcp = cfg["forecaster"]
    L, H, tvar = w["lookback"], w["horizon"], w["target_var"]
    tap = fcp["tap"]
    np.random.seed(seed); torch.manual_seed(seed)

    S = build_split_series(cfg, seed)
    mean, std = S["train"]["x"].mean(), S["train"]["x"].std() + 1e-8
    ymean, ystd = S["train"]["y"].mean(), S["train"]["y"].std() + 1e-8

    def nm(sp):
        return dict(x=(sp["x"] - mean) / std, y=(sp["y"] - ymean) / ystd, z=sp["z"])
    tr, va, te = nm(S["train"]), nm(S["val"]), nm(S["test"])
    Xtr, Ytr, _, _, _ = make_windows_xy(tr["x"], tr["y"], tr["z"], L, H)
    Xva, Yva, _, _, _ = make_windows_xy(va["x"], va["y"], va["z"], L, H)
    Xte, Yte, Zwin, Zhard, _ = make_windows_xy(te["x"], te["y"], te["z"], L, H)

    ep = epochs or fcp["max_epochs"]
    tcfg = TrainConfig(lookback=L, horizon=H, d_model=fcp["d_model"], n_layers=fcp["n_layers"],
                       n_heads=fcp["n_heads"], d_ff=fcp["d_ff"], dropout=fcp["dropout"],
                       lr=fcp["lr"], batch_size=fcp["batch_size"], max_epochs=ep,
                       patience=fcp["patience"], lr_decay=fcp["lr_decay"],
                       warmup_epochs=fcp["warmup_epochs"], seed=seed, device="cpu")
    model, val_mse = train_itransformer((Xtr, Ytr, None), (Xva, Yva, None),
                                        n_vars=1, cfg=tcfg, verbose=True)

    # baseline: MSE of "predict the y-mean" (== 1.0 by construction after y z-scoring),
    # and MSE of the ORACLE AR-mean point forecast, to show how much signal exists.
    from data.synthetic_dgp import _f_apply
    d = cfg["dgp"]; a = np.array([d["a_0"], d["a_1"]])
    # oracle on raw scale then z-score with same y stats
    te_raw = S["test"]
    Xr, Yr, _, _, starts = make_windows_xy(te_raw["x"], te_raw["y"], te_raw["z"], L, H)
    zc = te_raw["z"]
    oracle_ae2 = []
    for i, s in enumerate(starts):
        yt = te_raw["y"][s + L: s + L + H]
        if len(yt) < H:
            continue
        cur = te_raw["x"][s + L - 1]
        zp = zc[s + L - 1: s + L - 1 + H]
        xh = np.empty(H)
        for h in range(H):
            cur = a[zp[h]] * cur
            xh[h] = cur
        yh = _f_apply(xh, d["f"])
        oracle_ae2.append(np.mean(((yh - ymean) / ystd - (yt - ymean) / ystd) ** 2))
    oracle_mse = float(np.mean(oracle_ae2))

    # forecaster test MSE
    ce = collect(model, Xte, Yte, tvar, device="cpu")
    fc_mse = float(np.mean((ce["yhat"][:, :, tvar] - Yte[:, :, tvar]) ** 2))

    # regime probe on the pre-head hidden state (target-variate token)
    hid = extract_all_taps(model, Xte, tvar, device="cpu")[tap]   # (N, d_model)
    n = len(hid); cut = int(0.6 * n)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(hid[:cut])
    clf = LogisticRegression(max_iter=500, C=1.0).fit(sc.transform(hid[:cut]), Zwin[:cut])
    acc = float(clf.score(sc.transform(hid[cut:]), Zwin[cut:]))
    base_acc = float(max(np.mean(Zwin[cut:] == 0), np.mean(Zwin[cut:] == 1)))

    # linear regression hidden -> horizon hard-fraction
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(sc.transform(hid[:cut]), Zhard[:cut])
    pr = reg.predict(sc.transform(hid[cut:]))
    ss_res = np.sum((Zhard[cut:] - pr) ** 2); ss_tot = np.sum((Zhard[cut:] - Zhard[cut:].mean()) ** 2)
    hardfrac_r2 = float(1 - ss_res / (ss_tot + 1e-12))

    # top single-dim correlations
    from sklearn.decomposition import PCA
    hp = PCA(n_components=min(20, hid.shape[1])).fit_transform(sc.transform(hid))
    corr_pc_z = float(np.max([abs(np.corrcoef(hp[:, k], Zwin)[0, 1]) for k in range(hp.shape[1])]))
    corr_pc_hf = float(np.max([abs(np.corrcoef(hp[:, k], Zhard)[0, 1]) for k in range(hp.shape[1])]))

    out = dict(
        seed=seed, epochs_run=ep, forecaster_val_mse=float(val_mse),
        forecaster_test_mse=fc_mse, oracle_ARmean_test_mse=oracle_mse,
        predict_ymean_mse=1.0,
        signal_headroom_mse=round(1.0 - oracle_mse, 4),
        forecaster_captured_frac=round((1.0 - fc_mse) / (1.0 - oracle_mse + 1e-12), 3),
        regime_probe_acc=acc, regime_probe_baseline=base_acc,
        regime_probe_gain=round(acc - base_acc, 4),
        hidden_to_hardfrac_R2=hardfrac_r2,
        max_abs_corr_PC_vs_z=corr_pc_z, max_abs_corr_PC_vs_hardfrac=corr_pc_hf,
    )
    return out


def main():
    res = [run(seed=s) for s in (0, 1, 2)]
    p = os.path.join(HERE, "runs", "phase1_synthetic", "forecaster_probe.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, "w"), indent=2)
    for r in res:
        print(json.dumps(r, indent=2))
    print(f"\nwrote {p}")
    # verdict
    gains = [r["regime_probe_gain"] for r in res]
    r2s = [r["hidden_to_hardfrac_R2"] for r in res]
    print(f"\nregime probe gain over baseline: {np.mean(gains):+.3f} (mean over seeds)")
    print(f"hidden->hardfrac R2: {np.mean(r2s):+.3f}")
    if np.mean(gains) > 0.05 or np.mean(r2s) > 0.05:
        print("VERDICT: forecaster hidden state DOES encode regime -> safe to run full Phase 1.")
    else:
        print("VERDICT: forecaster hidden state does NOT encode regime -> DGP needs a "
              "pre-run fix (the forecaster has no loss incentive to infer z). Diagnose before "
              "the full run.")


if __name__ == "__main__":
    main()
