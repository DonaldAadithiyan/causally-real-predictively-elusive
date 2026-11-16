"""Phase 3 (H6 slice only) — g_MVDF vs g_dense vs g_dense+MVDF (v2 spec §4.8, §6 H6).

The single sharpest downstream test: at matched coverage, does the mechanistically-
validated feature subspace give narrower conformal intervals than FFDCI's raw dense
feature vector? Run at alpha=0.10, ID + pre-registered shift, >=3 seeds, paired
bootstrap CI on the width difference. NOT the full baseline suite (spec §8) — just
the one paired comparison that v1's causal chain says to test regardless of H3.

Env carve-out (spec §9): E_select (Phase 1 feature selection) / E_eval_ID (calibration
+ ID width) / E_eval_shift (shift width). g's calibration set = E_eval_ID; deployment
stream = E_eval_shift; ID width measured on a held-out tail of E_eval_ID.

Reads MVDF_v2 from results/v2/v2_phase1_seed<s>/<ds>_<tap>/mvdf_v2_selection.json.

Outputs results/v2/v2_phase3_h6/AGG_h6.csv:
  dataset, seed, mode, id_coverage, id_width, shift_coverage, shift_width
plus AGG_h6_tests.csv with the paired bootstrap CIs (g_MVDF - g_dense, g_dense+MVDF - g_dense).
"""
from __future__ import annotations

from mechconf.paths import REPO_ROOT  # noqa: E402
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml

from mechconf.data.datasets import get_splits, make_windows
from experiments._seed import derive_seed
from mechconf.data.fingerprint import assert_expected_data
from mechconf.forecasters.train_itransformer import TrainConfig, train_itransformer
from mechconf.representation.layer_sweep import extract_all_taps
from mechconf.sae.train_sae import train_sae, encode_all
from mechconf.validation.errors import collect
from mechconf.data.shift_windows import three_way_env_masks
from mechconf.calibration.difficulty_scorer import DifficultyScorer, build_features, paired_width_bootstrap
from mechconf.eval import metrics


def _tap_act(model, X, tvar, tap, dev):
    return extract_all_taps(model, X, tvar, device=dev)[tap]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2_phase3_h6.yaml")
    ap.add_argument("--phase1_root_tmpl", default="results/v2/v2_phase1_seed{seed}")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.datasets:
        cfg["datasets"] = args.datasets
    if args.seeds:
        cfg["seeds"] = args.seeds
    if args.fast:
        cfg["forecaster"]["max_epochs"] = 3
        cfg["sae"]["steps"] = 3000

    here = REPO_ROOT
    out = os.path.join(here, "results", "v2", cfg["run_name"] + ("_fast" if args.fast else ""))
    os.makedirs(out, exist_ok=True)
    dd = cfg["dataset_defaults"]
    L, H, tvar = dd["lookback"], dd["horizon"], dd["target_var"]
    alpha = cfg["alpha"]
    tap = cfg["fallback_tap"]

    rows = []
    for ds in cfg["datasets"]:
        for seed in cfg["seeds"]:
            ds_seed = derive_seed(seed, ds)
            np.random.seed(ds_seed); torch.manual_seed(ds_seed)
            dev = cfg["device"]
            try:
                sd = get_splits(ds, max_channels=dd["max_channels"])
            except Exception as e:
                print(f"SKIP {ds}: {e}"); continue
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

            env = three_way_env_masks(end_te, len(sd.test))
            id_idx = np.where(env["E_eval_ID"])[0]
            sh_idx = np.where(env["E_eval_shift"])[0]
            # E_eval_ID -> 50% MLP-train / 30% conformalize (sets the split-conformal
            # residual quantile -> valid ~1-alpha ID coverage) / 20% ID-width eval.
            n = len(id_idx)
            c1, c2 = int(0.5 * n), int(0.8 * n)
            train_idx, conf_idx, idw_idx = id_idx[:c1], id_idx[c1:c2], id_idx[c2:]

            tr_act = _tap_act(model, Xtr, tvar_abs, tap, dev)
            sae = train_sae(tr_act, expansion=cfg["sae"]["expansion"], k=cfg["sae"]["k"],
                            steps=cfg["sae"]["steps"], batch_size=cfg["sae"]["batch_size"],
                            lr=cfg["sae"]["lr"], resample_every=cfg["sae"]["resample_every"],
                            resample_until_frac=cfg["sae"]["resample_until_frac"],
                            resample_cap=cfg["sae"]["resample_cap"], device=dev, seed=ds_seed)

            # feature sets from the matching-seed Phase 1
            p1 = os.path.join(here, args.phase1_root_tmpl.format(seed=seed),
                              f"{ds}_{tap}", "mvdf_v2_selection.json")
            if not os.path.exists(p1):
                print(f"  [h6] no Phase1 selection at {p1}; skipping {ds} seed {seed}")
                continue
            sel = json.load(open(p1))
            mvdf_idx = np.array(sel["MVDF_v2"], dtype=int)
            corr_idx = np.array(sel["correlated"], dtype=int)   # top-k by |corr(a_j, S_t)| on E_select
            k = len(mvdf_idx)

            def bundle(widx):
                Xe, Ye = Xte[widx], Yte[widx]
                ce = collect(model, Xe, Ye, tvar_abs, device=dev)
                act = _tap_act(model, Xe, tvar_abs, tap, dev)
                z = encode_all(sae, act, device=dev)
                return dict(hidden=ce["hidden"], z=z, score=ce["score"],
                            yhat=ce["yhat"][:, :, tvar_abs], y=Ye[:, :, tvar_abs],
                            prof=ce["step_profile"])
            b_tr, b_conf, b_idw, b_sh = (bundle(train_idx), bundle(conf_idx),
                                         bundle(idw_idx), bundle(sh_idx))

            from sklearn.decomposition import PCA
            rng = np.random.default_rng(ds_seed + 777)
            active_frac = (b_tr["z"] != 0).mean(axis=0)
            elig = np.where(active_frac >= 0.02)[0]
            rand_idx = np.sort(rng.choice(elig, size=min(k, len(elig)), replace=False))
            pca_model = PCA(n_components=k).fit(b_tr["hidden"].astype(np.float64))

            fkw = dict(corr_idx=corr_idx, rand_idx=rand_idx, pca_model=pca_model, k=k)
            for mode in ["dense", "mvdf", "dense+mvdf", "correlated", "random_k", "pca_k"]:
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
                rows.append(dict(dataset=ds, seed=seed, mode=mode,
                                 id_coverage=mi["coverage"], id_width=mi["mean_width"],
                                 id_winkler=mi["winkler"],
                                 shift_coverage=ms["coverage"], shift_width=ms["mean_width"],
                                 shift_winkler=ms["winkler"],
                                 _id_widths=r_id["width"].tolist(),
                                 _sh_widths=r_sh["width"].tolist(),
                                 arch_in_dim=sc.arch_signature["in_dim"]))
                print(f"  {ds} s{seed} {mode:11s} ID cov {mi['coverage']:.3f} w {mi['mean_width']:.3f}"
                      f"  SHIFT cov {ms['coverage']:.3f} w {ms['mean_width']:.3f}")

            # Task G — naive no-scorer (constant half-width split-conformal, no MLP).
            # `naive_no_scorer` = conf block only (30%). `naive_matched` = train U conf
            # (80%, the SAME calibration data the learned modes see) -> budget-matched.
            from mechconf.calibration.difficulty_scorer import naive_no_scorer_run
            gg = cfg["calibration"]["gamma_global"]
            score_cal80 = np.concatenate([b_tr["score"], b_conf["score"]])
            for nmode, s_cal in (("naive_no_scorer", b_conf["score"]),
                                 ("naive_matched", score_cal80)):
                r_id = naive_no_scorer_run(s_cal, b_idw["yhat"], b_idw["y"], b_idw["prof"], alpha, gamma=gg)
                r_sh = naive_no_scorer_run(s_cal, b_sh["yhat"], b_sh["y"], b_sh["prof"], alpha, gamma=gg)
                mi = metrics.summarize(r_id, b_idw["y"], alpha)
                ms = metrics.summarize(r_sh, b_sh["y"], alpha)
                rows.append(dict(dataset=ds, seed=seed, mode=nmode,
                                 id_coverage=mi["coverage"], id_width=mi["mean_width"],
                                 id_winkler=mi["winkler"],
                                 shift_coverage=ms["coverage"], shift_width=ms["mean_width"],
                                 shift_winkler=ms["winkler"],
                                 _id_widths=r_id["width"].tolist(), _sh_widths=r_sh["width"].tolist(),
                                 arch_in_dim=0))
                print(f"  {ds} s{seed} {nmode:15s} ID cov {mi['coverage']:.3f} w {mi['mean_width']:.3f}"
                      f"  SHIFT cov {ms['coverage']:.3f} w {ms['mean_width']:.3f}")

    df = pd.DataFrame(rows)
    df.drop(columns=["_id_widths", "_sh_widths"]).to_csv(os.path.join(out, "AGG_h6.csv"), index=False)

    # paired bootstrap per (dataset, seed): every non-dense mode vs dense, AND the
    # sharpest test — mvdf vs correlated (does dose-response/R_j earn anything over
    # the cheap baseline?). Full per-seed rows, not just significance counts.
    tests = []
    for (ds, seed), g in df.groupby(["dataset", "seed"]):
        gm = {m: g[g["mode"] == m].iloc[0] for m in g["mode"].unique()}
        pairs = [(m, "dense") for m in ("mvdf", "dense+mvdf", "correlated", "random_k", "pca_k")]
        pairs += [("mvdf", "correlated"), ("mvdf", "random_k"), ("mvdf", "pca_k")]
        pairs += [("naive_no_scorer", "dense"), ("naive_no_scorer", "mvdf"),
                  ("naive_matched", "dense"), ("naive_matched", "mvdf"),
                  ("naive_matched", "naive_no_scorer")]
        for a_mode, b_mode in pairs:
            if a_mode not in gm or b_mode not in gm:
                continue
            for env_tag, wk in (("ID", "_id_widths"), ("shift", "_sh_widths")):
                a = np.array(gm[a_mode][wk]); b = np.array(gm[b_mode][wk])
                n = min(len(a), len(b))
                bt = paired_width_bootstrap(a[:n], b[:n], seed=seed)
                tests.append(dict(dataset=ds, seed=seed, comparison=f"{a_mode}_minus_{b_mode}",
                                  env=env_tag, a_width=float(a[:n].mean()),
                                  b_width=float(b[:n].mean()), **bt))
    tdf = pd.DataFrame(tests)
    tdf.to_csv(os.path.join(out, "AGG_h6_tests.csv"), index=False)

    if not df.empty:
        piv = df.groupby(["dataset", "mode"])[["id_coverage", "id_width", "shift_coverage",
                                               "shift_width"]].agg(["mean", "std"]).round(4)
        piv.to_csv(os.path.join(out, "AGG_h6_by_dataset.csv"))
        print("\n=== H6 per-seed width table ===")
        print(df[["dataset", "seed", "mode", "id_coverage", "id_width",
                  "shift_coverage", "shift_width"]].round(4).to_string(index=False))
        print("\n=== paired bootstrap by comparison x env (mean over seeds) ===")
        summ = tdf.groupby(["comparison", "env"]).agg(
            mean_diff=("mean_diff", "mean"), p_a_lt_b=("p_a_lt_b", "mean"),
            n_sig=("significant", "sum"), n=("significant", "size")).round(4)
        print(summ.to_string())
        print("\nFOUR-WAY H6 READ:")
        print("  mvdf < dense (shift) but mvdf ~ correlated  -> any sensible feature selection helps, not causal content")
        print("  mvdf < dense AND mvdf < correlated          -> the dose-response/R_j content specifically matters")
        print("  mvdf ~ random_k ~ pca_k < dense             -> dimensionality reduction alone explains it")
    print(f"\nOutputs -> {out}")


if __name__ == "__main__":
    main()
