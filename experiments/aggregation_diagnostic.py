"""Aggregation diagnostic — is the flat MVDF_v2/correlated invariance signal a
consequence of the mean aggregation r_t = (1/k) sum_j z_tj, or of the features
themselves?

This was flagged before Run 5 ("r_t = (1/k)Sum z_tj ... one reason I wouldn't expect
it to be particularly powerful"). The Phase-2 bin-count audit is that landing as an
empirical finding.

For each (dataset, seed) and each feature-set condition, reload the deterministic
forecaster + SAE, regenerate the SAE codes z on the E_eval_ID / E_eval_shift blocks
(E_select untouched), build the aggregated signal r_t under FOUR aggregations, and
report the dispersion (std / IQR / # distinct-ish values) of each:

  mean         (1/k) sum_j z_tj                       <- the current method
  max          max_j z_tj
  active_count  sum_j 1[z_tj > 0]
  l2           sqrt(sum_j z_tj^2)

Also report, for each, how many of 5 quantile bins the signal would populate (the
quantity the invariance metric needs >= ~4 of to be non-degenerate).

If a non-mean aggregation gives real dispersion (>=4 bins) -> the mean was the bug,
re-run Phase 2 on it. If all four stay flat -> the features genuinely lack an
aggregate-level signal, and H3 is not measurable this way (a finding, not a shrug).

Outputs runs/agg_diagnostic/AGG.csv
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


def _agg(z_sub: np.ndarray, how: str) -> np.ndarray:
    if how == "mean":
        return z_sub.mean(axis=1)
    if how == "max":
        return z_sub.max(axis=1)
    if how == "active_count":
        return (z_sub > 0).sum(axis=1).astype(float)
    if how == "l2":
        return np.sqrt((z_sub ** 2).sum(axis=1))
    raise ValueError(how)


def _bins_populated(sig: np.ndarray, n_bins: int = 5, min_per_bin: int = 5) -> int:
    edges = np.quantile(sig, np.linspace(0, 1, n_bins + 1)[1:-1])
    b = np.digitize(sig, edges)
    return int(sum(np.bincount(b, minlength=n_bins) >= min_per_bin))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2_phase2.yaml")
    ap.add_argument("--phase1_root_tmpl", default="v2_confirm/v2_phase1_seed{seed}")
    ap.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2", "electricity"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.fast:
        cfg["forecaster"]["max_epochs"] = 3
        cfg["sae"]["steps"] = 3000

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "runs", "agg_diagnostic")
    os.makedirs(out, exist_ok=True)
    dd = cfg["dataset_defaults"]
    L, H, tvar = dd["lookback"], dd["horizon"], dd["target_var"]
    tap = cfg["fallback_tap"]
    conds = ["random", "correlated", "intervention_only", "MVDF_v1", "MVDF_v2"]
    aggs = ["mean", "max", "active_count", "l2"]

    rows = []
    for ds in args.datasets:
        for seed in args.seeds:
            ds_seed = derive_seed(seed, ds)
            np.random.seed(ds_seed); torch.manual_seed(ds_seed)
            sd = get_splits(ds, max_channels=dd["max_channels"])
            assert_expected_data(ds, sd, strict=False)
            Xtr, Ytr, _ = make_windows(sd.train, L, H)
            Xva, Yva, _ = make_windows(sd.val, L, H)
            Xte, Yte, end_te = make_windows(sd.test, L, H)
            fc = cfg["forecaster"]
            tcfg = TrainConfig(lookback=L, horizon=H, d_model=fc["d_model"], n_layers=fc["n_layers"],
                               n_heads=fc["n_heads"], d_ff=fc["d_ff"], dropout=fc["dropout"],
                               lr=fc["lr"], batch_size=fc["batch_size"], max_epochs=fc["max_epochs"],
                               patience=fc["patience"], lr_decay=fc.get("lr_decay", 0.8),
                               warmup_epochs=fc.get("warmup_epochs", 3), seed=ds_seed, device=cfg["device"])
            model, _ = train_itransformer((Xtr, Ytr, None), (Xva, Yva, None),
                                          n_vars=len(sd.channels), cfg=tcfg, verbose=False)
            tr_act = extract_all_taps(model, Xtr, tvar % len(sd.channels), device=cfg["device"])[tap]
            sae = train_sae(tr_act, expansion=cfg["sae"]["expansion"], k=cfg["sae"]["k"],
                            steps=cfg["sae"]["steps"], batch_size=cfg["sae"]["batch_size"],
                            lr=cfg["sae"]["lr"], resample_every=cfg["sae"]["resample_every"],
                            resample_until_frac=cfg["sae"]["resample_until_frac"],
                            resample_cap=cfg["sae"]["resample_cap"], device=cfg["device"], seed=ds_seed)

            env = three_way_env_masks(end_te, len(sd.test))
            tvar_abs = tvar % len(sd.channels)
            sel_path = os.path.join(here, args.phase1_root_tmpl.format(seed=seed),
                                    f"{ds}_{tap}", "mvdf_v2_selection.json")
            sel = json.load(open(sel_path))
            # intervention_only from candidates.csv top-|U_j|
            cpath = os.path.join(here, args.phase1_root_tmpl.format(seed=seed),
                                 f"{ds}_{tap}", "candidates.csv")
            cdf = pd.read_csv(cpath).dropna(subset=["U_j"])
            k = len(sel["MVDF_v2"])
            sel["intervention_only"] = cdf.reindex(
                cdf.U_j.abs().sort_values(ascending=False).index).feature.astype(int).tolist()[:k]

            for env_tag in ["E_eval_ID", "E_eval_shift"]:
                widx = np.where(env[env_tag])[0]
                Xe, Ye = Xte[widx], Yte[widx]
                act = extract_all_taps(model, Xe, tvar_abs, device=cfg["device"])[tap]
                z = encode_all(sae, act, device=cfg["device"])
                for cond in conds:
                    fset = np.array(sel[cond], dtype=int)
                    z_sub = z[:, fset]
                    frac_active_any = float((z_sub > 0).any(axis=1).mean())
                    per_feat_active = (z_sub > 0).mean(axis=0)
                    for how in aggs:
                        sig = _agg(z_sub, how)
                        rows.append(dict(
                            dataset=ds, seed=seed, env=env_tag, condition=cond, agg=how,
                            n_windows=len(sig),
                            std=float(np.std(sig)),
                            iqr=float(np.percentile(sig, 75) - np.percentile(sig, 25)),
                            cv=float(np.std(sig) / (np.abs(np.mean(sig)) + 1e-9)),
                            n_distinct=int(len(np.unique(np.round(sig, 6)))),
                            bins_populated=_bins_populated(sig),
                            frac_windows_any_active=frac_active_any,
                            median_per_feat_active=float(np.median(per_feat_active)),
                        ))
            print(f"[{ds} s{seed}] done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "AGG.csv"), index=False)

    print("\n=== bins_populated by condition x agg (mean over 9 dataset-seed-env, ID+shift) ===")
    piv = df.pivot_table(index="condition", columns="agg", values="bins_populated", aggfunc="mean").round(2)
    print(piv.to_string())
    print("\n=== signal std by condition x agg (mean) ===")
    piv2 = df.pivot_table(index="condition", columns="agg", values="std", aggfunc="mean")
    print(piv2.round(4).to_string())
    print("\n=== per-feature activation rate (median) by condition ===")
    print(df.groupby("condition")["median_per_feat_active"].mean().round(3).to_string())
    print("\nREAD: if any non-'mean' agg gets bins_populated ~>=4 for MVDF_v2/correlated,")
    print("      the mean was the bug -> re-run Phase 2 on that agg. If all stay ~3,")
    print("      the features lack an aggregate-level signal (H3 not measurable this way).")
    print(f"\nOutputs -> {out}/AGG.csv")


if __name__ == "__main__":
    main()
