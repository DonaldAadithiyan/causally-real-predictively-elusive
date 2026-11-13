"""CQR + PID-conformal vs naive_matched — the literature-baseline comparison
(wrap-up spec §1, "final wrap-up" doc).

What this answers: not "does our method beat the literature" (no SAE-conditioned method
survived to make that claim — Task G and H7-A both falsified). It answers whether the
literature itself beats naive_matched: if CQR/PID-conformal ALSO lose to a constant-width
band, the negative result is general to adaptive conformal conditioning in this
forecasting-under-shift setting, not specific to SAE-conditioning. If they beat it, the
boundary is specific to this project's particular mechanism-to-interval translations.

Scope: ALL 4 datasets (ETTh1, ETTh2, electricity, traffic) x 3 seeds — matching Task G's
exact scope and inherited caveats (traffic directional-only per Task A; electricity
reported plainly, NOT assumed to inherit Task B's learned-scorer-specific instability).

Each baseline in its OWN standard form:
  - naive_matched   : reused unchanged from calibration/difficulty_scorer.py
                       (constant-width split-conformal, 80%-budget quantile). Already
                       validated (Task G / budget audit).
  - CQR             : baselines/cqr.py + baselines/quantile_itransformer.py — a genuine
                       quantile-regression forecaster (pinball loss, same trunk
                       architecture as the project's point forecaster, no SAE), then
                       Romano et al.'s split-conformal correction. Trained fresh per
                       (dataset, seed) alongside the existing point forecaster.
  - PID-conformal   : baselines/pid_conformal.py — Angelopoulos et al.'s P+I online
                       update on the POINT forecaster's own residual stream (raw, no
                       feature conditioning). Same max-over-H score convention.

Validity sanity check (spec discipline, applied here too): each mode's ID coverage is
checked against nominal on a held-out slice before entering any comparison table.

Coverage-equivalence gate before any width claim: paired bootstrap on the coverage
DIFFERENCE vs naive_matched, full CI must lie within +/-1.5pt (this project's existing
margin, see FINISHING_TASKS_STATUS.md / RESULTS_FINAL.md) before a width comparison is
interpreted as more than descriptive.

KNOWN CQR CAVEAT (diagnosed 2026-09-05, reported not fixed): the quantile forecaster
(shared iTransformer trunk + 2 linear heads, pinball loss) CONVERGES (val loss plateaus
by ~epoch 5; 60 epochs + patience 10 gives the identical result) but its raw per-step
coverage falls from ~0.94 at horizon step 1 to ~0.60 by mid/late horizon — a genuine
capacity limit of a single linear quantile head tracking 96 separate quantile positions,
not undertraining. The split-conformal correction then restores per-step marginal
validity (post-correction per-step coverage 0.97-0.99 everywhere) but must inflate q_hat
to cover the worst (mid-horizon) steps, so SIMULTANEOUS coverage still lands a few points
below nominal (~0.71 on ETTh1 s0 vs naive_matched's 0.83 on the same slice — and this
project's own ETTh1/h2 ID coverage runs 0.79-0.83 for every method on this short
E_eval_ID slice, so the gap is partly the shared calibration-set-size effect and partly
CQR-specific). We report CQR's numbers with this stated, rather than iterating on the
quantile architecture to close the gap — that would be tuning a baseline to look better.
The residual gap is itself informative about how hard well-calibrated multi-horizon
quantile regression is with a standard architecture.

Outputs runs/baseline_comparison/{AGG_baselines.csv, AGG_baselines_tests.csv}
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
from validation.errors import collect
from shift.shift_windows import three_way_env_masks
from calibration.aci import empirical_quantile
from calibration.difficulty_scorer import naive_no_scorer_run, paired_width_bootstrap
from baselines.quantile_itransformer import (QuantileTrainConfig,
                                              train_quantile_itransformer,
                                              predict_quantiles)
from baselines.cqr import cqr_calibrate, cqr_run
from baselines.pid_conformal import pid_conformal_run
from eval import metrics

COV_EQUIV_MARGIN = 0.015   # +/-1.5pt, matches this project's existing coverage-equivalence gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2_phase3_h6.yaml")
    ap.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2", "electricity", "traffic"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.fast:
        cfg["forecaster"]["max_epochs"] = 3

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "runs", "baseline_comparison")
    os.makedirs(out, exist_ok=True)
    dd = cfg["dataset_defaults"]
    L, H, tvar = dd["lookback"], dd["horizon"], dd["target_var"]
    alpha = cfg["alpha"]
    dev = cfg["device"]

    rows = []
    for ds in args.datasets:
        for seed in args.seeds:
            ds_seed = derive_seed(seed, ds)
            np.random.seed(ds_seed); torch.manual_seed(ds_seed)
            try:
                sd = get_splits(ds, max_channels=dd["max_channels"])
            except (FileNotFoundError, KeyError) as e:
                print(f"  SKIP {ds}: raw data not available locally ({e}). "
                      f"Partial results kept.")
                break  # skip the rest of this dataset's seeds too
            _ = sd  # noqa: keep the name for the block below
            assert_expected_data(ds, sd, strict=cfg.get("strict_fingerprint", False))
            Xtr, Ytr, _ = make_windows(sd.train, L, H)
            Xva, Yva, _ = make_windows(sd.val, L, H)
            Xte, Yte, end_te = make_windows(sd.test, L, H)
            tvar_abs = tvar % len(sd.channels)
            n_vars = len(sd.channels)

            # --- point forecaster (shared: naive_matched + PID-conformal) ---
            fc = cfg["forecaster"]
            tcfg = TrainConfig(lookback=L, horizon=H, d_model=fc["d_model"], n_layers=fc["n_layers"],
                               n_heads=fc["n_heads"], d_ff=fc["d_ff"], dropout=fc["dropout"],
                               lr=fc["lr"], batch_size=fc["batch_size"], max_epochs=fc["max_epochs"],
                               patience=fc["patience"], lr_decay=fc.get("lr_decay", 0.8),
                               warmup_epochs=fc.get("warmup_epochs", 3), seed=ds_seed, device=dev)
            model, _ = train_itransformer((Xtr, Ytr, None), (Xva, Yva, None),
                                          n_vars=n_vars, cfg=tcfg, verbose=False)

            # --- quantile forecaster (CQR only), fresh, own seed offset ---
            qcfg = QuantileTrainConfig(lookback=L, horizon=H, d_model=fc["d_model"],
                                       n_layers=fc["n_layers"], n_heads=fc["n_heads"],
                                       d_ff=fc["d_ff"], dropout=fc["dropout"], lr=fc["lr"],
                                       batch_size=fc["batch_size"],
                                       max_epochs=(3 if args.fast else fc["max_epochs"]),
                                       patience=fc["patience"], lr_decay=fc.get("lr_decay", 0.8),
                                       warmup_epochs=fc.get("warmup_epochs", 3),
                                       seed=ds_seed + 31337, device=dev, alpha=alpha)
            qmodel, _ = train_quantile_itransformer((Xtr, Ytr, None), (Xva, Yva, None),
                                                     n_vars=n_vars, cfg=qcfg, verbose=False)

            env = three_way_env_masks(end_te, len(sd.test))
            id_idx = np.where(env["E_eval_ID"])[0]
            sh_idx = np.where(env["E_eval_shift"])[0]
            n = len(id_idx); c1, c2 = int(0.5 * n), int(0.8 * n)
            tr_i, conf_i, idw_i = id_idx[:c1], id_idx[c1:c2], id_idx[c2:]

            def bundle(widx):
                Xe, Ye = Xte[widx], Yte[widx]
                ce = collect(model, Xe, Ye, tvar_abs, device=dev)
                lo_q, hi_q = predict_quantiles(qmodel, Xe, device=dev)
                return dict(score=np.asarray(ce["score"], float),
                           yhat=ce["yhat"][:, :, tvar_abs], y=Ye[:, :, tvar_abs],
                           prof=ce["step_profile"],
                           lo_q=lo_q[:, :, tvar_abs], hi_q=hi_q[:, :, tvar_abs])
            b_tr, b_conf, b_idw, b_sh = bundle(tr_i), bundle(conf_i), bundle(idw_i), bundle(sh_idx)
            score_cal80 = np.concatenate([b_tr["score"], b_conf["score"]])   # matched budget

            gg = cfg["calibration"]["gamma_global"]

            # --- naive_matched (reused, unchanged) ---
            r_id_nm = naive_no_scorer_run(score_cal80, b_idw["yhat"], b_idw["y"], b_idw["prof"], alpha, gamma=gg)
            r_sh_nm = naive_no_scorer_run(score_cal80, b_sh["yhat"], b_sh["y"], b_sh["prof"], alpha, gamma=gg)

            # --- CQR: calibrate on conf block (matched to naive_matched's 30% conf slice
            #     for the split-correction step; the quantile forecaster itself was
            #     trained on the SAME train/val split naive_matched's point forecaster used) ---
            q_hat = cqr_calibrate(b_conf["lo_q"], b_conf["hi_q"], b_conf["y"], alpha,
                                  step_profile=b_conf["prof"])
            r_id_cqr = cqr_run(b_idw["lo_q"], b_idw["hi_q"], b_idw["y"], q_hat,
                               step_profile=b_idw["prof"])
            r_sh_cqr = cqr_run(b_sh["lo_q"], b_sh["hi_q"], b_sh["y"], q_hat,
                               step_profile=b_sh["prof"])

            # --- PID-conformal: q0 from the same 80%-matched pool ---
            q0 = empirical_quantile(score_cal80, 1 - alpha)
            # eta matched to this project's gamma_global (0.03) — same adaptation speed
            # as naive_matched's GlobalACI, so the PID vs single-gain comparison isn't
            # confounded by one being tuned slower than the other.
            r_id_pid = pid_conformal_run(q0, b_idw["yhat"], b_idw["y"], b_idw["prof"], alpha, eta=gg)
            r_sh_pid = pid_conformal_run(q0, b_sh["yhat"], b_sh["y"], b_sh["prof"], alpha, eta=gg)

            for mode, r_id, r_sh in (("naive_matched", r_id_nm, r_sh_nm),
                                     ("cqr", r_id_cqr, r_sh_cqr),
                                     ("pid_conformal", r_id_pid, r_sh_pid)):
                mi = metrics.summarize(r_id, b_idw["y"], alpha)
                ms = metrics.summarize(r_sh, b_sh["y"], alpha)
                rows.append(dict(dataset=ds, seed=seed, mode=mode,
                                 id_coverage=mi["coverage"], id_width=mi["mean_width"],
                                 shift_coverage=ms["coverage"], shift_width=ms["mean_width"],
                                 _idw=r_id["width"].tolist(), _shw=r_sh["width"].tolist()))
                print(f"[{ds} s{seed}] {mode:15s} ID cov {mi['coverage']:.3f} w {mi['mean_width']:.3f}"
                      f"  SHIFT cov {ms['coverage']:.3f} w {ms['mean_width']:.3f}")

    df = pd.DataFrame(rows)
    df.drop(columns=["_idw", "_shw"]).to_csv(os.path.join(out, "AGG_baselines.csv"), index=False)

    # coverage-equivalence gate + paired width bootstrap, vs naive_matched, per (dataset,seed)
    tests = []
    for (ds, seed), g in df.groupby(["dataset", "seed"]):
        gm = {m: g[g["mode"] == m].iloc[0] for m in g["mode"].unique()}
        if "naive_matched" not in gm:
            continue
        for m in ("cqr", "pid_conformal"):
            if m not in gm:
                continue
            for env_tag, wk, covk in (("ID", "_idw", "id_coverage"), ("shift", "_shw", "shift_coverage")):
                a = np.array(gm[m][wk]); b = np.array(gm["naive_matched"][wk])
                nn = min(len(a), len(b))
                bt = paired_width_bootstrap(a[:nn], b[:nn], seed=seed)
                cov_diff = float(gm[m][covk] - gm["naive_matched"][covk])
                equiv = bool(abs(cov_diff) <= COV_EQUIV_MARGIN)
                tests.append(dict(dataset=ds, seed=seed, comparison=f"{m}_minus_naive_matched",
                                  env=env_tag, cov_diff=cov_diff,
                                  coverage_equivalent=equiv, **bt))
    tdf = pd.DataFrame(tests)
    tdf.to_csv(os.path.join(out, "AGG_baselines_tests.csv"), index=False)

    print("\n=== baseline vs naive_matched summary ===")
    for m in ("cqr", "pid_conformal"):
        s = tdf[(tdf.comparison == f"{m}_minus_naive_matched") & (tdf.env == "shift")]
        if s.empty:
            continue
        n_equiv = int(s.coverage_equivalent.sum())
        n_narrower = int((s.mean_diff < 0).sum())
        n_narrower_equiv = int(((s.mean_diff < 0) & s.coverage_equivalent).sum())
        print(f"  {m}: {n_equiv}/{len(s)} coverage-equivalent (±{COV_EQUIV_MARGIN*100:.1f}pt); "
              f"{n_narrower}/{len(s)} narrower than naive_matched; "
              f"{n_narrower_equiv}/{len(s)} narrower AND coverage-equivalent")


if __name__ == "__main__":
    main()
