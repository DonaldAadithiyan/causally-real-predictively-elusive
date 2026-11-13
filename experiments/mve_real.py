"""Minimum Viable Experiment on REAL data (spec §7): iTransformer + one ETT dataset.

Pipeline (spec §5.1):
  data -> iTransformer -> hidden tokens -> TopK SAE -> {correlation screen, ablation}
       -> 3 matched-budget feature sets (SAE-random / SAE-correlated / MVDF)
       -> Variant B (Mondrian bins + per-bin ACI)  [+ Variant A opaque MLP for the ablation]
       -> in-distribution eval  +  pre-registered shift eval
       -> RQ1 diagnostic, H2 table, H3 table, Variant-A/B table

Calibration set = the VAL split windows (disjoint from SAE training which uses TRAIN,
and disjoint from the TEST stream). Deployment stream = the TEST split windows, run
online; the pre-registered tail block is the shift sub-window.

Run:
  python -m experiments.mve_real --config configs/mve_etth1.yaml
  python -m experiments.mve_real --config configs/mve_etth1.yaml --fast   # Mac smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml

from data.datasets import get_splits, make_windows
from forecasters.train_itransformer import TrainConfig, train_itransformer
from sae.train_sae import train_sae, encode_all
from validation.errors import collect
from validation.correlation_screen import screen, top_k_correlated
from validation.ablation import ablate_features, select_mvdf
from validation.rq1 import rq1_diagnostic
from calibration.mondrian_perbin import MondrianPerBinACI
from calibration.ffdci_head import train_quantile_mlp, FFDCIHeadACI
from calibration.aci import empirical_quantile
from shift.shift_windows import REGISTERED, split_test_windows, describe_shift
from eval import metrics, plots


# ----------------------------------------------------------------------------------
def build_feature_signal(z: np.ndarray, feat_idx: np.ndarray) -> np.ndarray:
    """Aggregate a k-feature set into one scalar difficulty signal per window:
    mean of the (standardized) activations of the selected features. Standardize so
    no single high-variance feature dominates the bin edges."""
    sub = z[:, feat_idx].astype(np.float64)
    mu, sd = sub.mean(0, keepdims=True), sub.std(0, keepdims=True) + 1e-9
    return ((sub - mu) / sd).mean(axis=1)


def run_variant_b(z_cal, score_cal, perstep_cal, z_te, yhat_te, y_te, feat_idx, cfg, static_only=False):
    sig_cal = build_feature_signal(z_cal, feat_idx)
    sig_te = build_feature_signal(z_te, feat_idx)
    m = MondrianPerBinACI(
        n_bins=cfg["validation"]["n_bins"],
        alpha_target=cfg["alpha"],
        gamma=cfg["calibration"]["gamma_perbin"],
        static_only=static_only,
        clip_mult=cfg["calibration"]["clip_mult"],
    ).fit(sig_cal, score_cal, perstep_cal)
    return m.run(sig_te, yhat_te, y_te)


def run_variant_a(z_cal, score_cal, step_profile, z_te, yhat_te, y_te, feat_idx, cfg):
    feat_cal = z_cal[:, feat_idx].astype(np.float32)
    feat_te = z_te[:, feat_idx].astype(np.float32)
    mlp = train_quantile_mlp(feat_cal, score_cal, alpha_target=cfg["alpha"], seed=cfg["seed"])
    base_q = empirical_quantile(score_cal, 1 - cfg["alpha"])
    runner = FFDCIHeadACI(mlp, base_q=base_q, alpha_target=cfg["alpha"],
                          gamma=cfg["calibration"]["gamma_global"],
                          clip_mult=cfg["calibration"]["clip_mult"], step_profile=step_profile)
    return runner.run(feat_te, yhat_te, y_te)


# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    if args.fast:
        cfg["forecaster"]["max_epochs"] = 3
        cfg["sae"]["steps"] = 3000
        cfg["validation"]["rq1_sample_size"] = 150
        cfg["validation"]["ablation_batch_cap"] = 1024

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "runs", cfg["run_name"] + ("_fast" if args.fast else ""))
    os.makedirs(out, exist_ok=True)
    seed = cfg["seed"]; np.random.seed(seed); torch.manual_seed(seed)
    dev = cfg["device"]
    alpha = cfg["alpha"]
    L, H = cfg["dataset"]["lookback"], cfg["dataset"]["horizon"]
    tvar = cfg["dataset"]["target_var"]

    log = {"config": cfg, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # ---- data ------------------------------------------------------------------
    sd = get_splits(cfg["dataset"]["name"], max_channels=cfg["dataset"].get("max_channels"))
    if len(sd.channels) == 0 or min(len(sd.train), len(sd.val), len(sd.test)) < (L + H + 10):
        raise RuntimeError(
            f"{cfg['dataset']['name']}: not enough usable data "
            f"(channels={len(sd.channels)}, "
            f"rows train/val/test={len(sd.train)}/{len(sd.val)}/{len(sd.test)}). "
            f"Check the data adapter."
        )
    tvar_abs = tvar % len(sd.channels)
    Xtr, Ytr, _ = make_windows(sd.train, L, H)
    Xva, Yva, _ = make_windows(sd.val, L, H)
    Xte, Yte, end_te = make_windows(sd.test, L, H)
    log["data_fingerprint"] = sd.fingerprint()
    log["shapes"] = dict(train=list(Xtr.shape), val=list(Xva.shape), test=list(Xte.shape))
    print(f"[data] {cfg['dataset']['name']} fp={sd.fingerprint()} "
          f"train {Xtr.shape} val {Xva.shape} test {Xte.shape}")

    # ---- pre-registered shift check -----------------------------------------
    dsname = cfg["dataset"]["name"]
    sw = REGISTERED.get(dsname) or REGISTERED["ETTh1"].__class__(
        dataset=dsname, description=f"Final 20% of the {dsname} test split.",
        rationale="Generic pre-registered tail window (no dataset-specific entry).")
    cal_target_raw = np.concatenate([sd.raw_train[:, tvar_abs], sd.raw_val[:, tvar_abs]])
    shift_desc = describe_shift(sd.raw_test[:, tvar_abs], sw.tail_frac, cal_target_raw)
    shift_desc["window"] = dict(description=sw.description, rationale=sw.rationale,
                                tail_frac=sw.tail_frac)
    log["shift_check"] = shift_desc
    print(f"[shift] {sw.description}\n        KS stat={shift_desc['ks_stat']:.3f} "
          f"p={shift_desc['ks_pvalue']:.2e}  is_shift={shift_desc['is_shift']}  "
          f"n_tail={shift_desc['n_tail']}")
    if cfg["shift"].get("require_ks_check", True) and not shift_desc["is_shift"]:
        print("[shift] WARNING: pre-registered window did not pass KS check. "
              "Reporting anyway (spec §6.3: do not swap the window).")
    in_dist_mask, shift_mask = split_test_windows(end_te, len(sd.test), sw.tail_frac)
    log["shift_n_windows"] = int(shift_mask.sum())

    # ---- forecaster --------------------------------------------------------
    fc = cfg["forecaster"]
    tcfg = TrainConfig(lookback=L, horizon=H, d_model=fc["d_model"], n_layers=fc["n_layers"],
                       n_heads=fc["n_heads"], d_ff=fc["d_ff"], dropout=fc["dropout"],
                       lr=fc["lr"], batch_size=fc["batch_size"], max_epochs=fc["max_epochs"],
                       patience=fc["patience"], lr_decay=fc.get("lr_decay", 0.8),
                       warmup_epochs=fc.get("warmup_epochs", 3), seed=seed, device=dev)
    t0 = time.time()
    model, best_val = train_itransformer((Xtr, Ytr, None), (Xva, Yva, None),
                                         n_vars=len(sd.channels), cfg=tcfg)
    log["forecaster_val_mse"] = float(best_val)
    log["forecaster_train_s"] = round(time.time() - t0, 1)

    # ---- hidden states + errors ------------------------------------------
    tr = collect(model, Xtr, Ytr, tvar_abs, device=dev)
    cal = collect(model, Xva, Yva, tvar_abs, device=dev)
    te = collect(model, Xte, Yte, tvar_abs, device=dev)
    yhat_te_t = te["yhat"][:, :, tvar_abs]
    y_te_t = Yte[:, :, tvar_abs]
    log["forecaster_test_mae_target"] = float(np.abs(yhat_te_t - y_te_t).mean())

    # per-step |e| profile on calibration for horizon-shaped intervals
    perstep_cal = np.abs(cal["yhat"][:, :, tvar_abs] - Yva[:, :, tvar_abs])
    step_profile = perstep_cal.mean(0) / (perstep_cal.mean() + 1e-12)

    # ---- first-order diagnostic: is difficulty even representable in the hidden
    #      state on this dataset? (cheap screen before the expensive ablation) ----
    from sklearn.decomposition import PCA
    _pcs = PCA(n_components=min(20, cal["hidden"].shape[1])).fit_transform(cal["hidden"])
    _hc = [abs(np.corrcoef(_pcs[:, i], cal["score"])[0, 1]) for i in range(_pcs.shape[1])]
    log["hidden_difficulty_signal"] = dict(
        max_abs_corr_hiddenPC_vs_score=float(np.max(_hc)),
        score_dispersion=float(cal["score"].std() / (cal["score"].mean() + 1e-12)),
        note="max |corr| >> 0.1 means difficulty is linearly readable pre-head; "
             "near 0.1 predicts weak H2/H3 (spec outcome B regime, aleatoric-dominated).",
    )
    print(f"[diagnostic] max|corr(hidden PC, |e|)| = {np.max(_hc):.3f}  "
          f"score dispersion = {cal['score'].std() / cal['score'].mean():.2f}")

    # ---- SAE (train split hidden only) --------------------------------
    sae_cfg = cfg["sae"]
    sae = train_sae(tr["hidden"], expansion=sae_cfg["expansion"], k=sae_cfg["k"],
                    steps=sae_cfg["steps"], batch_size=sae_cfg["batch_size"],
                    lr=sae_cfg["lr"], resample_every=sae_cfg["resample_every"],
                    resample_until_frac=sae_cfg.get("resample_until_frac", 0.5),
                    resample_cap=sae_cfg.get("resample_cap", 64),
                    device=dev, seed=seed)
    with torch.no_grad():
        rec = sae.reconstruction_stats(torch.from_numpy(tr["hidden"]).to(dev))
        rec_cal = sae.reconstruction_stats(torch.from_numpy(cal["hidden"]).to(dev))
    log["sae_reconstruction"] = rec
    log["sae_reconstruction_cal"] = rec_cal
    print(f"[sae] recon train {rec}")
    if rec["fvu_standardized"] > 0.3:
        print(f"[sae] WARNING: FVU {rec['fvu_standardized']:.3f} > 0.3 — SAE under-fit or "
              f"diverged; ablation / MVDF results on this dataset are unreliable.")

    z_cal = encode_all(sae, cal["hidden"], device=dev)
    z_te = encode_all(sae, te["hidden"], device=dev)

    # ---- correlation screening ----------------------------------------
    scr = screen(z_cal, cal["score"], min_active_frac=cfg["validation"]["min_active_frac"])
    n_eligible = int(scr["eligible"].sum())
    log["n_eligible_features"] = n_eligible
    print(f"[screen] {n_eligible}/{sae.d_sae} features eligible (active >= "
          f"{cfg['validation']['min_active_frac']})")

    # ---- ablation on a NON-correlation-prefiltered sample (RQ1) --------
    rng = np.random.default_rng(seed)
    elig = np.where(scr["eligible"])[0]
    n_samp = min(cfg["validation"]["rq1_sample_size"], len(elig))
    samp = np.sort(rng.choice(elig, size=n_samp, replace=False))

    cap = cfg["validation"]["ablation_batch_cap"]
    sel = np.sort(rng.choice(len(Xva), size=min(cap, len(Xva)), replace=False))
    t0 = time.time()
    abl = ablate_features(model, sae, Xva[sel], Yva[sel], cal["hidden"][sel], tvar_abs,
                          feature_indices=samp, step_profile=cal["step_profile"], device=dev)
    log["ablation_s"] = round(time.time() - t0, 1)
    log["ablation_n_features"] = int(n_samp)
    log["ablation_n_windows"] = int(len(sel))

    pd.DataFrame({
        "feature": abl["feature_indices"],
        "delta": abl["delta"],
        "abs_delta": abl["abs_delta"],
        "delta_all_windows": abl["delta_all"],
        "mean_abs_change": abl["mean_abs_change"],
        "active_frac": abl["active_frac"],
        "n_active": abl["n_active"],
        "abs_pearson": np.abs(scr["pearson"][abl["feature_indices"]]),
        "spearman": scr["spearman"][abl["feature_indices"]],
    }).sort_values("abs_delta", ascending=False).to_csv(
        os.path.join(out, "delta_j_distribution.csv"), index=False)

    # binary-relevance rate (spec §3.2 — expect near 100%), over features with
    # enough active windows to estimate Delta at all
    est = abl["n_active"] >= 5
    pos_rate = float(np.mean(abl["delta"][est] > 0)) if est.any() else float("nan")
    md = np.median(abl["abs_delta"]) + 1e-12
    log["ablation_positive_rate"] = pos_rate
    log["ablation_max_over_median"] = float(abl["abs_delta"].max() / md)
    print(f"[ablation] Delta>0 rate = {pos_rate:.3f}  max/median |Delta| = "
          f"{abl['abs_delta'].max() / md:.1f}x  ({log['ablation_s']}s)")

    # ---- RQ1 diagnostic -------------------------------------------------
    rq1 = rq1_diagnostic(scr, abl)
    json.dump(rq1, open(os.path.join(out, "rq1_diagnostic.json"), "w"), indent=2)
    print(f"[RQ1] Spearman rho(corr-rank, ablation-rank) = {rq1.get('spearman_rho'):.3f}  "
          f"top-set Jaccard = {rq1.get('top_set_jaccard'):.3f}")
    print(f"      {rq1.get('interpretation')}")
    plots.delta_j_distribution(abl["abs_delta"], os.path.join(out, "delta_j_distribution.png"))
    plots.rq1_scatter(np.abs(scr["pearson"][abl["feature_indices"]]), abl["abs_delta"],
                      os.path.join(out, "rq1_scatter.png"), rq1.get("spearman_rho", float("nan")))

    # ---- three matched-budget feature sets ----------------------------
    k = cfg["validation"]["feature_budget_k"]
    feat_correlated = top_k_correlated(scr, k)
    feat_mvdf = select_mvdf(abl, scr, k, require_correlated_pool=None)
    if len(feat_mvdf) < k:
        # pad from ablation ranking ignoring correlation membership if the correlated
        # pool is too small (report this)
        extra = abl["feature_indices"][abl["order"]]
        feat_mvdf = np.array(list(dict.fromkeys(list(feat_mvdf) + list(extra)))[:k])
        log["mvdf_padded_beyond_correlated_pool"] = True
    feat_random = np.sort(rng.choice(elig, size=k, replace=False))

    log["feature_sets"] = dict(
        SAE_correlated=feat_correlated.tolist(),
        MVDF=feat_mvdf.tolist(),
        SAE_random=feat_random.tolist(),
        overlap_corr_mvdf=int(len(set(feat_correlated) & set(feat_mvdf))),
    )
    print(f"[features] |corr ∩ mvdf| = {log['feature_sets']['overlap_corr_mvdf']}/{k}")

    # ---- run calibration: Variant B for all three, Variant A for MVDF -----
    conditions = {"SAE-random": feat_random, "SAE-correlated": feat_correlated, "MVDF": feat_mvdf}
    results_b, results_b_static = {}, {}
    for name, fidx in conditions.items():
        results_b[name] = run_variant_b(z_cal, cal["score"], perstep_cal,
                                        z_te, yhat_te_t, y_te_t, fidx, cfg, static_only=False)
        results_b_static[name] = run_variant_b(z_cal, cal["score"], perstep_cal,
                                               z_te, yhat_te_t, y_te_t, fidx, cfg, static_only=True)
    result_a_mvdf = run_variant_a(z_cal, cal["score"], step_profile,
                                  z_te, yhat_te_t, y_te_t, feat_mvdf, cfg)

    # ---- H2 (in-distribution) + H3 (shift) tables ---------------------
    h2_rows, h3_rows = [], []
    for name in conditions:
        idw = metrics.summarize(results_b[name], y_te_t, alpha, mask=in_dist_mask)
        shw = metrics.summarize(results_b[name], y_te_t, alpha, mask=shift_mask)
        deg = metrics.degradation(idw, shw)
        tpb = results_b[name]["timesteps_per_bin"].tolist()
        h2_rows.append(dict(condition=name, variant="B", **{f"id_{k2}": v for k2, v in idw.items()},
                            timesteps_per_bin=tpb))
        h3_rows.append(dict(condition=name, variant="B",
                            **{f"shift_{k2}": v for k2, v in shw.items()},
                            **deg))
        # static-bin (no per-bin ACI) contrast, spec §5.6
        sh_static = metrics.summarize(results_b_static[name], y_te_t, alpha, mask=shift_mask)
        h3_rows.append(dict(condition=name + " (static bins)", variant="B-static",
                            **{f"shift_{k2}": v for k2, v in sh_static.items()},
                            **metrics.degradation(
                                metrics.summarize(results_b_static[name], y_te_t, alpha, mask=in_dist_mask),
                                sh_static)))

    # Variant A on MVDF
    a_id = metrics.summarize(result_a_mvdf, y_te_t, alpha, mask=in_dist_mask)
    a_sh = metrics.summarize(result_a_mvdf, y_te_t, alpha, mask=shift_mask)
    h2_rows.append(dict(condition="MVDF", variant="A (opaque MLP)",
                        **{f"id_{k2}": v for k2, v in a_id.items()}, timesteps_per_bin=None))
    h3_rows.append(dict(condition="MVDF", variant="A (opaque MLP)",
                        **{f"shift_{k2}": v for k2, v in a_sh.items()},
                        **metrics.degradation(a_id, a_sh)))

    pd.DataFrame(h2_rows).to_csv(os.path.join(out, "h2_in_distribution.csv"), index=False)
    pd.DataFrame(h3_rows).to_csv(os.path.join(out, "h3_shift.csv"), index=False)

    # ---- H3 headline: Delta_shift = degradation_correlated - degradation_MVDF ----
    def deg_of(name):
        idw = metrics.summarize(results_b[name], y_te_t, alpha, mask=in_dist_mask)
        shw = metrics.summarize(results_b[name], y_te_t, alpha, mask=shift_mask)
        return metrics.degradation(idw, shw)
    dc, dm = deg_of("SAE-correlated"), deg_of("MVDF")
    delta_shift = {kk: float(dc[kk] - dm[kk]) for kk in dc}
    log["H3_delta_shift_correlated_minus_mvdf"] = delta_shift
    print("\n[H3] Delta_shift (correlated degradation - MVDF degradation; >0 favours MVDF):")
    for kk, vv in delta_shift.items():
        print(f"     {kk:24s} {vv:+.4f}")

    # ---- Variant A vs B table (interpretability cost, spec Table 5) -----
    pd.DataFrame([
        dict(feature_set="MVDF", variant="B (Mondrian+per-bin ACI)",
             **{f"id_{k2}": v for k2, v in metrics.summarize(results_b["MVDF"], y_te_t, alpha, mask=in_dist_mask).items()}),
        dict(feature_set="MVDF", variant="A (opaque MLP+global ACI)",
             **{f"id_{k2}": v for k2, v in a_id.items()}),
    ]).to_csv(os.path.join(out, "variant_ab.csv"), index=False)

    # ---- plots over deployment time --------------------------------
    shift_start = int(np.where(shift_mask)[0][0]) if shift_mask.any() else len(shift_mask)
    plots.coverage_width_over_time(
        {n: results_b[n] for n in conditions}, shift_start, alpha,
        os.path.join(out, "coverage_width_over_time.png"),
        title=f"{cfg['dataset']['name']}  Variant B  (shift boundary in red)")

    log["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(log, open(os.path.join(out, "run_log.json"), "w"), indent=2, default=str)
    print(f"\nAll outputs -> {out}")
    print("  rq1_diagnostic.json  h2_in_distribution.csv  h3_shift.csv  variant_ab.csv")
    print("  delta_j_distribution.{csv,png}  rq1_scatter.png  coverage_width_over_time.png  run_log.json")


if __name__ == "__main__":
    main()
