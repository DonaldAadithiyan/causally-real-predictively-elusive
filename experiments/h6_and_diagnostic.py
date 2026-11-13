"""Combined: aggregation diagnostic + H6 four-way scorer decomposition, sharing ONE
forecaster + SAE per (dataset, seed) so nothing is trained twice.

Reuses Phase 1's mvdf_v2_selection.json / candidates.csv (from a prior run) — does
NOT re-run the Phase 1 dose-response sweep or Phase 2.

Per (dataset, seed), once:
  - reload deterministic iTransformer + SAE (from ds_seed; identical to the prior run)
  - E_eval_ID -> 50% MLP-train / 30% conformalize / 20% ID-width-eval; E_eval_shift = shift
  - load MVDF_v2 / correlated feature sets; derive intervention_only (top-|U_j|),
    random_k, pca_k
  - AGGREGATION DIAGNOSTIC: signal dispersion + bins-populated for {mean,max,l2,
    active_count} x 5 conditions on E_eval_ID + E_eval_shift
  - H6: split-conformal scorer for 6 modes -> ID/shift coverage + width, paired bootstrap

Outputs runs/h6_and_diag/{AGG_diag.csv, AGG_h6.csv, AGG_h6_tests.csv, AGG_h6_by_dataset.csv}
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import yaml

from data.datasets import get_splits, make_windows
from data.fingerprint import assert_expected_data
from experiments._seed import derive_seed
from forecasters.train_itransformer import TrainConfig, train_itransformer
from representation.layer_sweep import extract_all_taps
from sae.train_sae import train_sae, encode_all
from validation.errors import collect
from shift.shift_windows import three_way_env_masks
from calibration.difficulty_scorer import DifficultyScorer, build_features, paired_width_bootstrap
from eval import metrics


TAP = "pre_head"
AGGS = ["mean", "max", "active_count", "l2"]
CONDS = ["random", "correlated", "intervention_only", "MVDF_v1", "MVDF_v2"]
MODES = ["dense", "mvdf", "dense+mvdf", "correlated", "random_k", "pca_k"]


def _agg(z_sub, how):
    if how == "mean":
        return z_sub.mean(1)
    if how == "max":
        return z_sub.max(1)
    if how == "active_count":
        return (z_sub > 0).sum(1).astype(float)
    if how == "l2":
        return np.sqrt((z_sub ** 2).sum(1))


def _bins_pop(sig, n_bins=5, min_per=5):
    edges = np.quantile(sig, np.linspace(0, 1, n_bins + 1)[1:-1])
    return int(sum(np.bincount(np.digitize(sig, edges), minlength=n_bins) >= min_per))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2_phase3_h6.yaml")
    ap.add_argument("--phase1_root_tmpl", default="runs/v2_phase1_seed{seed}")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    datasets = args.datasets or cfg["datasets"]
    seeds = args.seeds or cfg["seeds"]
    if args.fast:
        cfg["forecaster"]["max_epochs"] = 3
        cfg["sae"]["steps"] = 3000

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "runs", "h6_and_diag")
    os.makedirs(out, exist_ok=True)
    dd = cfg["dataset_defaults"]
    L, H, tvar = dd["lookback"], dd["horizon"], dd["target_var"]
    alpha = cfg["alpha"]
    dev = cfg["device"]

    diag_rows, h6_rows = [], []

    for ds in datasets:
        for seed in seeds:
            ds_seed = derive_seed(seed, ds)
            np.random.seed(ds_seed); torch.manual_seed(ds_seed)
            sd = get_splits(ds, max_channels=dd["max_channels"])
            assert_expected_data(ds, sd, strict=cfg.get("strict_fingerprint", True))
            tvar_abs = tvar % len(sd.channels)
            Xtr, Ytr, _ = make_windows(sd.train, L, H)
            Xva, Yva, _ = make_windows(sd.val, L, H)
            Xte, Yte, end_te = make_windows(sd.test, L, H)

            fc = cfg["forecaster"]
            tcfg = TrainConfig(lookback=L, horizon=H, d_model=fc["d_model"], n_layers=fc["n_layers"],
                               n_heads=fc["n_heads"], d_ff=fc["d_ff"], dropout=fc["dropout"],
                               lr=fc["lr"], batch_size=fc["batch_size"], max_epochs=fc["max_epochs"],
                               patience=fc["patience"], lr_decay=fc.get("lr_decay", 0.8),
                               warmup_epochs=fc.get("warmup_epochs", 3), seed=ds_seed, device=dev)
            model, _ = train_itransformer((Xtr, Ytr, None), (Xva, Yva, None),
                                          n_vars=len(sd.channels), cfg=tcfg, verbose=False)
            tr_act = extract_all_taps(model, Xtr, tvar_abs, device=dev)[TAP]
            sae = train_sae(tr_act, expansion=cfg["sae"]["expansion"], k=cfg["sae"]["k"],
                            steps=cfg["sae"]["steps"], batch_size=cfg["sae"]["batch_size"],
                            lr=cfg["sae"]["lr"], resample_every=cfg["sae"]["resample_every"],
                            resample_until_frac=cfg["sae"]["resample_until_frac"],
                            resample_cap=cfg["sae"]["resample_cap"], device=dev, seed=ds_seed)

            env = three_way_env_masks(end_te, len(sd.test))
            id_idx = np.where(env["E_eval_ID"])[0]
            sh_idx = np.where(env["E_eval_shift"])[0]
            n = len(id_idx); c1, c2 = int(0.5 * n), int(0.8 * n)
            tr_i, conf_i, idw_i = id_idx[:c1], id_idx[c1:c2], id_idx[c2:]

            p1 = os.path.join(here, args.phase1_root_tmpl.format(seed=seed),
                              f"{ds}_{TAP}", "mvdf_v2_selection.json")
            if not os.path.exists(p1):
                print(f"  SKIP {ds} s{seed}: no {p1}"); continue
            sel = json.load(open(p1))
            mvdf_idx = np.array(sel["MVDF_v2"], dtype=int)
            corr_idx = np.array(sel["correlated"], dtype=int)
            k = len(mvdf_idx)
            cpath = os.path.join(here, args.phase1_root_tmpl.format(seed=seed),
                                 f"{ds}_{TAP}", "candidates.csv")
            cdf = pd.read_csv(cpath).dropna(subset=["U_j"])
            sel["intervention_only"] = cdf.reindex(
                cdf.U_j.abs().sort_values(ascending=False).index).feature.astype(int).tolist()[:k]
            sel_arr = {c: np.array(sel[c], dtype=int) for c in CONDS}

            def bundle(widx):
                Xe, Ye = Xte[widx], Yte[widx]
                ce = collect(model, Xe, Ye, tvar_abs, device=dev)
                act = extract_all_taps(model, Xe, tvar_abs, device=dev)[TAP]
                z = encode_all(sae, act, device=dev)
                return dict(hidden=ce["hidden"], z=z, score=ce["score"],
                            yhat=ce["yhat"][:, :, tvar_abs], y=Ye[:, :, tvar_abs],
                            prof=ce["step_profile"])
            b_tr, b_conf, b_idw, b_sh = bundle(tr_i), bundle(conf_i), bundle(idw_i), bundle(sh_idx)
            # matched-budget calibration pool for the naive control: tr_i U conf_i (80%),
            # the SAME data the learned modes see (MLP-train + conformalize).
            score_cal80 = np.concatenate([b_tr["score"], b_conf["score"]])

            # --- AGGREGATION DIAGNOSTIC (on E_eval_ID width slice + shift) ---
            for env_tag, b in (("E_eval_ID", b_idw), ("E_eval_shift", b_sh)):
                for cond in CONDS:
                    zs = b["z"][:, sel_arr[cond]]
                    pfa = (zs > 0).mean(0)
                    for how in AGGS:
                        sig = _agg(zs, how)
                        diag_rows.append(dict(dataset=ds, seed=seed, env=env_tag, condition=cond,
                                              agg=how, std=float(np.std(sig)),
                                              bins_populated=_bins_pop(sig),
                                              median_per_feat_active=float(np.median(pfa))))

            # --- H6 (split-conformal, 6 modes) ---
            from sklearn.decomposition import PCA
            rng = np.random.default_rng(ds_seed + 777)
            elig = np.where((b_tr["z"] != 0).mean(0) >= 0.02)[0]
            rand_idx = np.sort(rng.choice(elig, size=min(k, len(elig)), replace=False))
            pca_model = PCA(n_components=k).fit(b_tr["hidden"].astype(np.float64))
            fkw = dict(corr_idx=corr_idx, rand_idx=rand_idx, pca_model=pca_model, k=k)

            for mode in MODES:
                f_tr = build_features(mode, b_tr["hidden"], b_tr["z"], mvdf_idx, **fkw)
                f_conf = build_features(mode, b_conf["hidden"], b_conf["z"], mvdf_idx, **fkw)
                f_idw = build_features(mode, b_idw["hidden"], b_idw["z"], mvdf_idx, **fkw)
                f_sh = build_features(mode, b_sh["hidden"], b_sh["z"], mvdf_idx, **fkw)
                sc = DifficultyScorer(mode, hidden_dim=f_tr.shape[1]).fit(
                    f_tr, b_tr["score"], alpha=alpha, seed=ds_seed)
                sc.conformalize(f_conf, b_conf["score"])
                r_id = sc.run(f_idw, b_idw["yhat"], b_idw["y"], b_idw["prof"], alpha,
                              gamma=cfg["calibration"]["gamma_global"])
                r_sh = sc.run(f_sh, b_sh["yhat"], b_sh["y"], b_sh["prof"], alpha,
                              gamma=cfg["calibration"]["gamma_global"])
                mi = metrics.summarize(r_id, b_idw["y"], alpha)
                ms = metrics.summarize(r_sh, b_sh["y"], alpha)
                h6_rows.append(dict(dataset=ds, seed=seed, mode=mode,
                                    id_coverage=mi["coverage"], id_width=mi["mean_width"],
                                    shift_coverage=ms["coverage"], shift_width=ms["mean_width"],
                                    _idw=r_id["width"].tolist(), _shw=r_sh["width"].tolist()))

            # Task G — naive no-scorer (constant half-width split-conformal).
            # Two budgets: `naive_no_scorer` uses conf_i only (30%, as-is);
            # `naive_matched` uses tr_i U conf_i (80%, same as the learned modes) so the
            # width/coverage comparison is calibration-budget-matched.
            from calibration.difficulty_scorer import naive_no_scorer_run
            gg = cfg["calibration"]["gamma_global"]
            for nmode, s_cal in (("naive_no_scorer", b_conf["score"]),
                                 ("naive_matched", score_cal80)):
                r_id = naive_no_scorer_run(s_cal, b_idw["yhat"], b_idw["y"], b_idw["prof"], alpha, gamma=gg)
                r_sh = naive_no_scorer_run(s_cal, b_sh["yhat"], b_sh["y"], b_sh["prof"], alpha, gamma=gg)
                mi = metrics.summarize(r_id, b_idw["y"], alpha)
                ms = metrics.summarize(r_sh, b_sh["y"], alpha)
                h6_rows.append(dict(dataset=ds, seed=seed, mode=nmode,
                                    id_coverage=mi["coverage"], id_width=mi["mean_width"],
                                    shift_coverage=ms["coverage"], shift_width=ms["mean_width"],
                                    _idw=r_id["width"].tolist(), _shw=r_sh["width"].tolist()))
            print(f"[{ds} s{seed}] diag + H6 (+ naive x2) done")

    diag = pd.DataFrame(diag_rows)
    diag.to_csv(os.path.join(out, "AGG_diag.csv"), index=False)
    h6 = pd.DataFrame(h6_rows)
    h6.drop(columns=["_idw", "_shw"]).to_csv(os.path.join(out, "AGG_h6.csv"), index=False)

    tests = []
    for (ds, seed), g in h6.groupby(["dataset", "seed"]):
        gm = {m: g[g["mode"] == m].iloc[0] for m in g["mode"].unique()}
        pairs = [(m, "dense") for m in ("mvdf", "dense+mvdf", "correlated", "random_k", "pca_k")]
        pairs += [("mvdf", "correlated"), ("mvdf", "random_k"), ("mvdf", "pca_k")]
        pairs += [("naive_no_scorer", "dense"), ("naive_no_scorer", "mvdf"),  # Task G (30% budget)
                  ("naive_matched", "dense"), ("naive_matched", "mvdf"),      # Task G (80%, matched)
                  ("naive_matched", "naive_no_scorer")]                        # budget-sensitivity
        for a_m, b_m in pairs:
            if a_m not in gm or b_m not in gm:
                continue
            for env_tag, wk in (("ID", "_idw"), ("shift", "_shw")):
                a = np.array(gm[a_m][wk]); b = np.array(gm[b_m][wk]); nn = min(len(a), len(b))
                tests.append(dict(dataset=ds, seed=seed, comparison=f"{a_m}_minus_{b_m}",
                                  env=env_tag, **paired_width_bootstrap(a[:nn], b[:nn], seed=seed)))
    pd.DataFrame(tests).to_csv(os.path.join(out, "AGG_h6_tests.csv"), index=False)
    h6.groupby(["dataset", "mode"])[["id_coverage", "id_width", "shift_coverage", "shift_width"]] \
      .agg(["mean", "std"]).round(4).to_csv(os.path.join(out, "AGG_h6_by_dataset.csv"))

    print("\n=== AGG DIAGNOSTIC: bins_populated by condition x agg ===")
    print(diag.pivot_table(index="condition", columns="agg", values="bins_populated").round(2).to_string())
    print("\n=== per-feature activation rate (median) ===")
    print(diag.groupby("condition")["median_per_feat_active"].mean().round(3).to_string())
    print("\n=== H6 coverage sanity (~0.90 ID) ===")
    print(h6.groupby("mode")[["id_coverage", "shift_coverage"]].mean().round(3).to_string())
    print("\n=== H6 per-seed shift width ===")
    print(h6.pivot_table(index=["dataset", "seed"], columns="mode", values="shift_width").round(3).to_string())
    tdf = pd.DataFrame(tests)
    summ = tdf.groupby(["comparison", "env"]).agg(
        mean_diff=("mean_diff", "mean"), p_a_lt_b=("p_a_lt_b", "mean"),
        n_sig=("significant", "sum"), n=("significant", "size")).round(4)
    print("\n=== paired bootstrap by comparison x env ===")
    print(summ.to_string())
    print(f"\nOutputs -> {out}")


if __name__ == "__main__":
    main()
