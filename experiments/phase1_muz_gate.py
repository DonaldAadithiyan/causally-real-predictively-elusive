"""Pre-run correction 3 — blind calibration of the mu_z mean-channel.

A variance-only regime gives an MSE-optimal point forecaster ZERO incentive to
encode z. Correction 3 adds  y = mu_z*1[z==1] + f(x) + sigma(z)*eta  so E[y|x,z]
depends on z. mu_z must be calibrated — but ONLY against pre-declared manipulation
checks (validity gates on the DGP), never against anything downstream of the
forecaster (the hypothesis test). This script is blind to MVDF/R_j/conformal
recovery by construction: it stops at the forecaster hidden state + a quadrant check.

Grid: mu_z in {0.3, 0.6, 0.9} = {1x, 2x, 3x} sigma_easy. Hold everything else at the
already-frozen values.

Gates (both must clear with real margin, not barely):
  A. regime_probe_gain  = linear-probe(hidden -> z) accuracy - majority baseline  > +0.05
  B. signal_headroom     = predict-ymean MSE (=1.0) - oracle-conditional-mean MSE  > +0.03
                           (oracle conditional mean = mu_z*1[z==1] + f(AR-mean forecast of x))
  C. quadrant: the pool feature whose natural activation best tracks the horizon
     hard-fraction must land in Type B (|U_j| >= median, |P_j| < median) OR type C,
     NOT type A — otherwise the regime is a pure content signal and plain correlation
     would find it, which does not validate R_j's causal-specificity selection.

Pick the SMALLEST mu_z clearing A + B with margin AND preserving a non-Type-A
quadrant for the regime feature. Freeze that value.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from data.synthetic_dgp import load_config, build_split_series, make_windows_xy, _f_apply
from forecasters.train_itransformer import TrainConfig, train_itransformer
from representation.layer_sweep import extract_all_taps
from sae.train_sae import train_sae, encode_all
from validation.errors import collect
from validation.correlation_screen import screen
from validation.dose_response import sweep_pool
from validation.difficulty_specificity import feature_type_table

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MU_Z_GRID = [0.3, 0.6, 0.9]


def eval_muz(cfg, mu_z, seed=0):
    w = cfg["windowing"]; fcp = cfg["forecaster"]; saep = cfg["sae"]; drp = cfg["dose_response"]
    L, H, tvar = w["lookback"], w["horizon"], w["target_var"]
    tap = fcp["tap"]
    np.random.seed(seed); torch.manual_seed(seed)

    S = build_split_series(cfg, seed, mu_z_override=mu_z)
    xm, xs = S["train"]["x"].mean(), S["train"]["x"].std() + 1e-8
    ym, ys = S["train"]["y"].mean(), S["train"]["y"].std() + 1e-8

    def nm(sp):
        return dict(x=(sp["x"] - xm) / xs, y=(sp["y"] - ym) / ys, z=sp["z"])
    tr, va, te = nm(S["train"]), nm(S["val"]), nm(S["test"])
    Xtr, Ytr, _, _, _ = make_windows_xy(tr["x"], tr["y"], tr["z"], L, H)
    Xva, Yva, _, _, _ = make_windows_xy(va["x"], va["y"], va["z"], L, H)
    Xte, Yte, Zwin, Zhard, _ = make_windows_xy(te["x"], te["y"], te["z"], L, H)

    tcfg = TrainConfig(lookback=L, horizon=H, d_model=fcp["d_model"], n_layers=fcp["n_layers"],
                       n_heads=fcp["n_heads"], d_ff=fcp["d_ff"], dropout=fcp["dropout"],
                       lr=fcp["lr"], batch_size=fcp["batch_size"], max_epochs=fcp["max_epochs"],
                       patience=fcp["patience"], lr_decay=fcp["lr_decay"],
                       warmup_epochs=fcp["warmup_epochs"], seed=seed, device="cpu")
    model, val_mse = train_itransformer((Xtr, Ytr, None), (Xva, Yva, None),
                                        n_vars=1, cfg=tcfg, verbose=False)

    # --- Gate B: oracle conditional-mean MSE vs predict-ymean (=1.0) ---
    d = cfg["dgp"]; a = np.array([d["a_0"], d["a_1"]])
    te_raw = S["test"]
    _, _, _, _, starts = make_windows_xy(te_raw["x"], te_raw["y"], te_raw["z"], L, H)
    zc = te_raw["z"]
    se = []
    for s in starts:
        yt = te_raw["y"][s + L: s + L + H]
        if len(yt) < H:
            continue
        cur = te_raw["x"][s + L - 1]
        zp = zc[s + L - 1: s + L - 1 + H]
        xh = np.empty(H)
        for h in range(H):
            cur = a[zp[h]] * cur; xh[h] = cur
        ycm = mu_z * (zp == 1).astype(float) + _f_apply(xh, d["f"])   # oracle E[y|x,z]
        se.append(np.mean(((ycm - ym) / ys - (yt - ym) / ys) ** 2))
    oracle_cm_mse = float(np.mean(se))
    signal_headroom = 1.0 - oracle_cm_mse

    ce = collect(model, Xte, Yte, tvar, device="cpu")
    fc_mse = float(np.mean((ce["yhat"][:, :, tvar] - Yte[:, :, tvar]) ** 2))

    # --- Gate A: linear probe hidden -> z ---
    hid = extract_all_taps(model, Xte, tvar, device="cpu")[tap]
    n = len(hid); cut = int(0.6 * n)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(hid[:cut])
    clf = LogisticRegression(max_iter=500).fit(sc.transform(hid[:cut]), Zwin[:cut])
    acc = float(clf.score(sc.transform(hid[cut:]), Zwin[cut:]))
    base = float(max(np.mean(Zwin[cut:] == 0), np.mean(Zwin[cut:] == 1)))
    probe_gain = acc - base

    # --- Gate C: quadrant of the best regime-tracking SAE feature ---
    # (small SAE + small pool — enough to place ONE feature in the P_j x U_j plane)
    tr_act = extract_all_taps(model, Xtr, tvar, device="cpu")[tap]
    sae = train_sae(tr_act, expansion=saep["expansion"], k=saep["k"],
                    steps=min(6000, saep["steps"]), batch_size=saep["batch_size"],
                    lr=saep["lr"], resample_every=saep["resample_every"],
                    resample_until_frac=saep["resample_until_frac"],
                    resample_cap=saep["resample_cap"], device="cpu", seed=seed, verbose=False)
    # use the E_select slice for the quadrant check (same block Phase 1 selects on)
    from shift.shift_windows import three_way_env_masks
    _, _, _, _, end_te = make_windows_xy(te["x"], te["y"], te["z"], L, H)
    env = three_way_env_masks(end_te, len(te["x"]))
    sel = env["E_select"]
    Xs, Ys, _, Zhs = Xte[sel], Yte[sel], Zwin[sel], Zhard[sel]
    csel = collect(model, Xs, Ys, tvar, device="cpu")
    sact = extract_all_taps(model, Xs, tvar, device="cpu")[tap]
    z_s = encode_all(sae, sact, device="cpu")
    scr = screen(z_s, csel["score"], min_active_frac=0.02)
    elig = np.where(scr["eligible"])[0]
    # pick the eligible feature whose natural activation best tracks hard-fraction
    tr_hf = Zhs.astype(float)
    corr_hf = np.array([abs(np.corrcoef(z_s[:, j], tr_hf)[0, 1]) if np.std(z_s[:, j]) > 0 else 0
                        for j in elig])
    order = elig[np.argsort(-corr_hf)]
    pool = np.sort(order[:min(60, len(order))])
    if len(pool) < 12:
        pool = np.sort(elig[:min(60, len(elig))])
    cap = min(1024, len(Xs))
    rng = np.random.default_rng(seed)
    wsel = np.sort(rng.choice(len(Xs), size=cap, replace=False))
    dose = sweep_pool(model, sae, Xs[wsel], Ys[wsel], sact[wsel], tvar, tap, pool,
                      csel["step_profile"], lambdas=drp["lambdas"], device="cpu", verbose=False)
    ftab = feature_type_table(dose["P_j"], dose["U_j"], feature_indices=pool)
    # regime feature = the pooled feature with highest |corr(a_j, hardfrac)|
    corr_hf_pool = np.array([abs(np.corrcoef(z_s[:, int(j)], tr_hf)[0, 1])
                             if np.std(z_s[:, int(j)]) > 0 else 0 for j in pool])
    reg_feat = int(pool[int(np.argmax(corr_hf_pool))])
    reg_type = next((t for t, ids in ftab["cells"].items() if reg_feat in ids), "?")
    reg_pos = list(pool).index(reg_feat)

    return dict(
        mu_z=mu_z, seed=seed, forecaster_val_mse=float(val_mse), forecaster_test_mse=fc_mse,
        oracle_conditional_mean_mse=oracle_cm_mse, signal_headroom=round(signal_headroom, 4),
        regime_probe_acc=round(acc, 4), regime_probe_baseline=round(base, 4),
        regime_probe_gain=round(probe_gain, 4),
        gateA_probe_gain_pass=bool(probe_gain > 0.05),
        gateB_signal_headroom_pass=bool(signal_headroom > 0.03),
        regime_feature=reg_feat, regime_feature_type=reg_type,
        regime_feature_abs_corr_hardfrac=round(float(corr_hf_pool.max()), 4),
        regime_feature_P_j=round(float(dose["P_j"][reg_pos]), 5),
        regime_feature_U_j=round(float(dose["U_j"][reg_pos]), 5),
        median_abs_P=round(ftab["median_abs_P"], 5), median_abs_U=round(ftab["median_abs_U"], 5),
        gateC_not_typeA_pass=bool(reg_type in ("B", "C")),
        type_counts=ftab["counts"],
    )


def main():
    cfg = load_config()
    rows = []
    for mu in MU_Z_GRID:
        for seed in (0, 1, 2):
            print(f"--- mu_z={mu}  seed={seed} ---")
            r = eval_muz(cfg, mu, seed=seed)
            rows.append(r)
            print(json.dumps({k: r[k] for k in ("regime_probe_gain", "signal_headroom",
                  "regime_feature_type", "gateA_probe_gain_pass", "gateB_signal_headroom_pass",
                  "gateC_not_typeA_pass")}, indent=2))
    p = os.path.join(HERE, "runs", "phase1_synthetic_gates", "muz_gate.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)

    # aggregate: a mu_z passes if all 3 gates hold on >= 2/3 seeds
    summary = []
    for mu in MU_Z_GRID:
        g = [r for r in rows if r["mu_z"] == mu]
        a_ok = sum(r["gateA_probe_gain_pass"] for r in g)
        b_ok = sum(r["gateB_signal_headroom_pass"] for r in g)
        c_ok = sum(r["gateC_not_typeA_pass"] for r in g)
        mean_gain = float(np.mean([r["regime_probe_gain"] for r in g]))
        mean_head = float(np.mean([r["signal_headroom"] for r in g]))
        types = [r["regime_feature_type"] for r in g]
        summary.append(dict(mu_z=mu, gateA_pass_seeds=a_ok, gateB_pass_seeds=b_ok,
                            gateC_pass_seeds=c_ok, mean_probe_gain=round(mean_gain, 4),
                            mean_signal_headroom=round(mean_head, 4),
                            regime_feature_types=types,
                            all_gates_2of3=bool(a_ok >= 2 and b_ok >= 2 and c_ok >= 2)))
    passing = [s for s in summary if s["all_gates_2of3"]]
    chosen = min((s["mu_z"] for s in passing), default=None)
    out = dict(grid=MU_Z_GRID, per_run=rows, summary=summary,
               chosen_mu_z=chosen,
               note="Smallest mu_z clearing gates A+B+C on >=2/3 seeds. Blind to any "
                    "MVDF/R_j/conformal recovery. If chosen_mu_z is None, none of the grid "
                    "passed -> widen the grid or reconsider the DGP (do NOT proceed to the "
                    "full run).")
    json.dump(out, open(p, "w"), indent=2, default=str)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nCHOSEN mu_z = {chosen}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
