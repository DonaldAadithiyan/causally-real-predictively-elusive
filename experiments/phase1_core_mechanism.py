"""Phase 1 driver (v2 spec §11): full mechanistic characterization at the winning
layer(s) from Phase 0.

Per (dataset, winning tap):
  1. train iTransformer, collect S_t on the E_select block (spec §9 — selection uses
     E_select only, never E_eval_*)
  2. train a TopK SAE on the TRAIN-split activations AT this tap
  3. draw >=100 unbiased candidate features (eligible by activation rate, NOT
     pre-filtered by correlation)
  4. C_j  (observational diagnostic)
  5. unified dose-response sweep -> P_j, U_j  (one intervention, both curves)
  6. feature-type table (A/B/C/D by median split on |P_j| x |U_j|)
  7. R_j = U_j - E[U_j|P_j]  (isotonic; both terms interventional)
  8. MVDF_v2 selection (|U_j| gate + top-k by R_j)
  9. H1 (corr vs P_j), H1b (C_j vs P_j), H1c (type-B enrichment of MVDF_v2 vs
     correlated_set vs MVDF_v1)

Outputs runs/v2_phase1/<dataset>_<tap>/:
  candidates.csv          per-feature: C_j, P_j, U_j, R_j, n_active, corr(a_j,S_t), type
  feature_type_table.json
  h1_tables.json          H1 / H1b / H1c
  mvdf_v2_selection.json   selected feature indices for MVDF_v2, MVDF_v1, correlated, random
  dose_curves.json         raw D_pred / D_unc curves (for plots + acceptance smoke test)
  smoke.json               acceptance checks (spec §10)
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
from experiments._seed import derive_seed
from data.fingerprint import assert_expected_data
from forecasters.train_itransformer import TrainConfig, train_itransformer
from representation.layer_sweep import extract_all_taps
from sae.train_sae import train_sae, encode_all
from validation.errors import collect
from validation.correlation_screen import screen
from validation.importance_and_difficulty import (
    compute_c_j, compute_delta_pred_single, rank_divergence,
)
from validation.dose_response import sweep_pool
from validation.difficulty_specificity import (
    residualize, feature_type_table, enrichment, select_mvdf_v2,
)
from shift.shift_windows import three_way_env_masks, describe_env_split


def tap_activations_for_split(model, X, target_var, tap, device):
    """(N, d_model) target-variate token AT `tap` for a window stack."""
    return extract_all_taps(model, X, target_var, device=device)[tap]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2_phase1.yaml")
    ap.add_argument("--phase0", default="runs/v2_phase0/layer_sweep_summary.json")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--taps", nargs="+", default=None, help="override winning taps")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.datasets:
        cfg["datasets"] = args.datasets
    if args.fast:
        cfg["forecaster"]["max_epochs"] = 3
        cfg["sae"]["steps"] = 3000
        cfg["phase1"]["candidate_pool_size"] = 40
        cfg["phase1"]["ablation_window_cap"] = 512

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_root = os.path.join(here, "runs", cfg["run_name"] + ("_fast" if args.fast else ""))
    os.makedirs(out_root, exist_ok=True)
    seed = cfg["seed"]; np.random.seed(seed); torch.manual_seed(seed)
    dev = cfg["device"]
    dd = cfg["dataset_defaults"]
    L, H, tvar = dd["lookback"], dd["horizon"], dd["target_var"]
    p1 = cfg["phase1"]

    # winning taps
    if args.taps:
        winning_taps = args.taps
        phase0_result = "OVERRIDE"
    else:
        try:
            p0 = json.load(open(os.path.join(here, args.phase0)))
            phase0_result = p0["result"]
            winning_taps = p0["winning_layers"] if p0["result"] == "GO" else [cfg["fallback_tap"]]
        except FileNotFoundError:
            print(f"[phase1] no Phase 0 summary at {args.phase0}; using fallback tap "
                  f"{cfg['fallback_tap']}")
            phase0_result = "MISSING"
            winning_taps = [cfg["fallback_tap"]]
    print(f"[phase1] Phase 0 = {phase0_result}; running taps {winning_taps}")

    master = {"config": cfg, "phase0_result": phase0_result, "winning_taps": winning_taps,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "per_run": {}}

    strict_fp = cfg.get("strict_fingerprint", True)
    for ds in cfg["datasets"]:
        # per-dataset re-seed: make ETTh1-alone == ETTh1-as-Nth-in-a-loop
        # (both phase scripts previously seeded once at the top -> non-reproducible).
        ds_seed = derive_seed(seed, ds)
        np.random.seed(ds_seed); torch.manual_seed(ds_seed)
        try:
            sd = get_splits(ds, max_channels=dd["max_channels"])
        except Exception as e:
            print(f"SKIP {ds}: {e}"); master["per_run"][ds] = {"skipped": str(e)}; continue
        fp = assert_expected_data(ds, sd, strict=strict_fp)   # refuses wrong data
        master.setdefault("data_fingerprints", {})[ds] = fp
        tvar_abs = tvar % len(sd.channels)
        Xtr, Ytr, _ = make_windows(sd.train, L, H)
        Xte, Yte, end_te = make_windows(sd.test, L, H)

        fc = cfg["forecaster"]
        tcfg = TrainConfig(lookback=L, horizon=H, d_model=fc["d_model"], n_layers=fc["n_layers"],
                           n_heads=fc["n_heads"], d_ff=fc["d_ff"], dropout=fc["dropout"],
                           lr=fc["lr"], batch_size=fc["batch_size"], max_epochs=fc["max_epochs"],
                           patience=fc["patience"], lr_decay=fc.get("lr_decay", 0.8),
                           warmup_epochs=fc.get("warmup_epochs", 3), seed=ds_seed, device=dev)
        Xva, Yva, _ = make_windows(sd.val, L, H)
        model, val_mse = train_itransformer((Xtr, Ytr, None), (Xva, Yva, None),
                                            n_vars=len(sd.channels), cfg=tcfg, verbose=True)

        # E_select block of the TEST split (selection env — spec §9)
        env = three_way_env_masks(end_te, len(sd.test))
        sel_mask = env["E_select"]
        env_desc = describe_env_split(sd.raw_test[:, tvar_abs])
        Xsel, Ysel = Xte[sel_mask], Yte[sel_mask]
        cal_sel = collect(model, Xsel, Ysel, tvar_abs, device=dev)
        prof = cal_sel["step_profile"]
        cap = min(p1["ablation_window_cap"], len(Xsel))
        rng = np.random.default_rng(ds_seed)
        wsel = np.sort(rng.choice(len(Xsel), size=cap, replace=False))

        for tap in winning_taps:
            rd = os.path.join(out_root, f"{ds}_{tap}")
            os.makedirs(rd, exist_ok=True)
            print("=" * 70, f"\n {ds}  tap={tap}\n", "=" * 70)
            t0 = time.time()

            # SAE on TRAIN activations at this tap
            tr_act = tap_activations_for_split(model, Xtr, tvar_abs, tap, dev)
            sae = train_sae(tr_act, expansion=cfg["sae"]["expansion"], k=cfg["sae"]["k"],
                            steps=cfg["sae"]["steps"], batch_size=cfg["sae"]["batch_size"],
                            lr=cfg["sae"]["lr"], resample_every=cfg["sae"]["resample_every"],
                            resample_until_frac=cfg["sae"]["resample_until_frac"],
                            resample_cap=cfg["sae"]["resample_cap"], device=dev, seed=ds_seed)

            sel_act = tap_activations_for_split(model, Xsel, tvar_abs, tap, dev)
            z_sel = encode_all(sae, sel_act, device=dev)
            scr = screen(z_sel, cal_sel["score"], min_active_frac=0.02)
            elig = np.where(scr["eligible"])[0]
            n_pool = min(p1["candidate_pool_size"], len(elig))
            pool = np.sort(rng.choice(elig, size=n_pool, replace=False))

            # C_j (observational, diagnostic)
            C = compute_c_j(z_sel[:, pool], cal_sel["score"], active_only=True)

            # single-point delta_pred for MVDF_v1 + smoke anchor
            dsingle = compute_delta_pred_single(
                model, sae, Xsel[wsel], Ysel[wsel],
                sel_act[wsel], tvar_abs, tap, pool, step_profile=prof, device=dev, loss="score")

            # unified dose-response sweep
            dose = sweep_pool(model, sae, Xsel[wsel], Ysel[wsel], sel_act[wsel], tvar_abs,
                              tap, pool, prof, lambdas=p1["lambdas"], device=dev, verbose=True)

            # feature-type table + R_j  (cells hold feature IDs, not pool positions)
            ftab = feature_type_table(dose["P_j"], dose["U_j"], feature_indices=pool)
            R = residualize(dose["P_j"], dose["U_j"], method=p1["residualize_method"],
                            provenance_p="interventional_dose_response",
                            provenance_u="interventional_dose_response")

            # MVDF_v2 selection
            k = p1["feature_budget_k"]
            mvdf2 = select_mvdf_v2(dose, u_threshold=p1["u_threshold"], k=k,
                                   residualize_method=p1["residualize_method"])
            # MVDF_v1 = top-k by |delta_pred| (Run 5's criterion, on this pool)
            dp = np.abs(dsingle["delta_pred"])
            mvdf1 = pool[np.argsort(-dp)[:k]]
            # correlated = top-k by |corr(a_j, S_t)| among the pool
            corr_pool = np.abs(scr["pearson"][pool])
            correlated = pool[np.argsort(-corr_pool)[:k]]
            random_set = np.sort(rng.choice(elig, size=k, replace=False))

            # H1 / H1b / H1c
            h1 = rank_divergence(dsingle["delta_pred"], scr["pearson"][pool], pool)  # P_j (single) vs corr
            h1b = rank_divergence(dsingle["delta_pred"], C, pool)                    # P_j vs C_j
            h1c = dict(
                feature_type_table=ftab,
                enrichment_MVDF_v2=enrichment(mvdf2["selected"], ftab, pool),
                enrichment_MVDF_v1=enrichment(mvdf1, ftab, pool),
                enrichment_correlated=enrichment(correlated, ftab, pool),
            )

            # persist
            df = pd.DataFrame({
                "feature": pool,
                "C_j": C,
                "delta_pred_single": dsingle["delta_pred"],
                "P_j": dose["P_j"], "U_j": dose["U_j"], "R_j": R,
                "n_active": dose["n_active"],
                "abs_corr_aj_St": np.abs(scr["pearson"][pool]),
                "type": [_type_of(int(fid), ftab) for fid in pool],
            }).sort_values("R_j", ascending=False)
            df.to_csv(os.path.join(rd, "candidates.csv"), index=False)
            json.dump(ftab, open(os.path.join(rd, "feature_type_table.json"), "w"), indent=2)
            json.dump(h1 | {"_which": "H1_deltaPred_vs_corr"}, open(os.path.join(rd, "h1.json"), "w"), indent=2)
            json.dump({"H1_deltaPred_vs_corr": h1, "H1b_deltaPred_vs_Cj": h1b, "H1c": h1c},
                      open(os.path.join(rd, "h1_tables.json"), "w"), indent=2, default=str)
            json.dump(dict(
                MVDF_v2=mvdf2["selected"].tolist(), MVDF_v1=mvdf1.tolist(),
                correlated=correlated.tolist(), random=random_set.tolist(),
                n_gate1=mvdf2["n_gate1"], u_threshold=mvdf2["u_threshold"],
                overlap_v2_v1=int(len(set(mvdf2["selected"].tolist()) & set(mvdf1.tolist()))),
                overlap_v2_corr=int(len(set(mvdf2["selected"].tolist()) & set(correlated.tolist()))),
            ), open(os.path.join(rd, "mvdf_v2_selection.json"), "w"), indent=2)
            json.dump(dict(pred_curves={str(k2): v for k2, v in dose["pred_curves"].items()},
                           unc_curves={str(k2): v for k2, v in dose["unc_curves"].items()}),
                      open(os.path.join(rd, "dose_curves.json"), "w"), indent=2)

            # acceptance smoke (spec §10): dose curves non-trivial for >=some features
            some_pred = sum(1 for c in dose["pred_curves"].values()
                            if np.isfinite(list(c.values())[0]) and
                            abs(list(c.values())[0] - list(c.values())[-1]) > 1e-6)
            some_unc = sum(1 for c in dose["unc_curves"].values()
                           if np.isfinite(list(c.values())[0]) and
                           abs(list(c.values())[0] - list(c.values())[-1]) > 1e-6)
            smoke = dict(
                n_pool=int(n_pool),
                dose_pred_nontrivial_count=int(some_pred),
                dose_unc_nontrivial_count=int(some_unc),
                mono_frac_pred=dose["mono_frac_pred"], mono_frac_unc=dose["mono_frac_unc"],
                PASS=bool(some_pred >= 3 and some_unc >= 3),
                sae_final_fvu=float(np.nan),  # filled below if available
            )
            json.dump(smoke, open(os.path.join(rd, "smoke.json"), "w"), indent=2)

            master["per_run"][f"{ds}_{tap}"] = dict(
                forecaster_val_mse=float(val_mse),
                seconds=round(time.time() - t0, 1),
                env_split=env_desc,
                type_counts=ftab["counts"], type_fractions=ftab["fractions"],
                H1_rho=h1.get("spearman_rho"), H1b_rho=h1b.get("spearman_rho"),
                enrichment_v2=h1c["enrichment_MVDF_v2"].get("enrichment_ratio"),
                enrichment_v1=h1c["enrichment_MVDF_v1"].get("enrichment_ratio"),
                enrichment_corr=h1c["enrichment_correlated"].get("enrichment_ratio"),
                smoke_pass=smoke["PASS"],
                mvdf_v2_vs_v1_overlap=int(len(set(mvdf2["selected"].tolist()) & set(mvdf1.tolist()))),
            )
            print(f"  type counts {ftab['counts']}  | H1 rho {h1.get('spearman_rho'):.3f}  "
                  f"H1b rho {h1b.get('spearman_rho'):.3f}")
            print(f"  MVDF_v2 type-B enrichment {h1c['enrichment_MVDF_v2'].get('enrichment_ratio')}  "
                  f"(v1 {h1c['enrichment_MVDF_v1'].get('enrichment_ratio')}, "
                  f"corr {h1c['enrichment_correlated'].get('enrichment_ratio')})")
            print(f"  smoke PASS={smoke['PASS']}  mono_frac unc={dose['mono_frac_unc']:.2f}")

    master["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(master, open(os.path.join(out_root, "run_log.json"), "w"), indent=2, default=str)

    # aggregate H1c across runs (the central qualitative claim)
    agg = []
    for key, r in master["per_run"].items():
        if "skipped" in r:
            continue
        agg.append(dict(run=key, type_B_frac=r["type_fractions"].get("B"),
                        enrichment_v2=r["enrichment_v2"], enrichment_v1=r["enrichment_v1"],
                        enrichment_corr=r["enrichment_corr"],
                        H1_rho=r["H1_rho"], H1b_rho=r["H1b_rho"], smoke_pass=r["smoke_pass"]))
    pd.DataFrame(agg).to_csv(os.path.join(out_root, "AGG_h1c.csv"), index=False)
    print(f"\nPhase 1 outputs -> {out_root}\n  AGG_h1c.csv is the H1c summary")


def _type_of(feature_id, ftab):
    for t, idxs in ftab["cells"].items():
        if feature_id in idxs:
            return t
    return "?"


if __name__ == "__main__":
    main()
