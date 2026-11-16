"""Phase 2 driver (v2 spec §11) — cross-environment invariance, the measurement
Run 5 skipped.

For each (dataset, winning tap):
  - retrain iTransformer + SAE at the tap (deterministic from seed; matches Phase 1)
  - load the 5 feature-set conditions fixed in Phase 1 (selection used E_select ONLY)
  - build the feature signal on E_eval_ID and on n_shift_subenvs sub-blocks of
    E_eval_shift  (E_select is NEVER touched here)
  - assert_no_leakage("E_select", eval_tags)  -> hard requirement, must fire on the
    deliberately-overlapping test
  - per condition: conditional_difficulty_curve per env, then invariance_score
      I  (lower = more invariant)
      spearman_stability (higher = more stable)
      conditional_error_drift

Pre-registered comparison (spec §6 H3, §7): does I(MVDF_v2) < I(correlated)?
  and separately I(MVDF_v2) vs I(MVDF_v1) (was the dose-response worth building?).

Outputs results/v2/v2_phase2/<dataset>_<tap>/invariance.json  and  results/v2/v2_phase2/AGG_invariance.csv
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
from mechconf.validation.correlation_screen import screen
from mechconf.validation.invariance import (
    assert_no_leakage, conditional_difficulty_curve, invariance_score,
    test_invariance_leakage, LeakageError,
)
from mechconf.data.shift_windows import three_way_env_masks, describe_env_split
from experiments.v1_mve.mve_real import build_feature_signal


def _tap_act(model, X, tvar, tap, dev):
    return extract_all_taps(model, X, tvar, device=dev)[tap]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2_phase2.yaml")
    ap.add_argument("--phase1_root", default="results/v2/v2_phase1")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--taps", nargs="+", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.datasets:
        cfg["datasets"] = args.datasets
    if args.fast:
        cfg["forecaster"]["max_epochs"] = 3
        cfg["sae"]["steps"] = 3000

    # spec §10 hard requirement: the leakage guard must fire on a deliberate overlap
    assert test_invariance_leakage(), "LEAKAGE GUARD DID NOT FIRE — aborting (spec §10)"
    print("[phase2] leakage guard verified (fires on deliberate overlap)")

    here = REPO_ROOT
    out_root = os.path.join(here, "results", "v2", cfg["run_name"] + ("_fast" if args.fast else ""))
    os.makedirs(out_root, exist_ok=True)
    seed = cfg["seed"]; np.random.seed(seed); torch.manual_seed(seed)
    dev = cfg["device"]; dd = cfg["dataset_defaults"]
    L, H, tvar = dd["lookback"], dd["horizon"], dd["target_var"]
    p2 = cfg["phase2"]

    p1_root = os.path.join(here, args.phase1_root)
    p1_fast = p1_root + "_fast"
    if args.fast and os.path.isdir(p1_fast):
        p1_root = p1_fast

    master = {"config": cfg, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "per_run": {}}
    agg = []

    for ds in cfg["datasets"]:
        ds_seed = derive_seed(seed, ds)          # match phase1's per-dataset seeding
        np.random.seed(ds_seed); torch.manual_seed(ds_seed)
        try:
            sd = get_splits(ds, max_channels=dd["max_channels"])
        except Exception as e:
            print(f"SKIP {ds}: {e}"); continue
        fp = assert_expected_data(ds, sd, strict=cfg.get("strict_fingerprint", True))
        master.setdefault("data_fingerprints", {})[ds] = fp
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
        model, val_mse = train_itransformer((Xtr, Ytr, None), (Xva, Yva, None),
                                            n_vars=len(sd.channels), cfg=tcfg, verbose=True)

        env = three_way_env_masks(end_te, len(sd.test))
        env_desc = describe_env_split(sd.raw_test[:, tvar_abs])

        # eval environments: E_eval_ID + shift sub-blocks. E_select is EXCLUDED.
        shift_idx = np.where(env["E_eval_shift"])[0]
        subs = np.array_split(shift_idx, p2["n_shift_subenvs"])
        eval_env_windows = {"E_eval_ID": np.where(env["E_eval_ID"])[0]}
        for si, sub in enumerate(subs):
            eval_env_windows[f"E_eval_shift_{si+1}"] = sub
        eval_tags = list(eval_env_windows)

        # HARD leakage check (spec §4.7)
        assert_no_leakage("E_select", eval_tags)

        taps = args.taps or _winning_taps(p1_root, ds, cfg["fallback_tap"])
        for tap in taps:
            rd = os.path.join(out_root, f"{ds}_{tap}")
            os.makedirs(rd, exist_ok=True)
            print("=" * 70, f"\n {ds}  tap={tap}  (invariance)\n", "=" * 70)
            t0 = time.time()

            tr_act = _tap_act(model, Xtr, tvar_abs, tap, dev)
            sae = train_sae(tr_act, expansion=cfg["sae"]["expansion"], k=cfg["sae"]["k"],
                            steps=cfg["sae"]["steps"], batch_size=cfg["sae"]["batch_size"],
                            lr=cfg["sae"]["lr"], resample_every=cfg["sae"]["resample_every"],
                            resample_until_frac=cfg["sae"]["resample_until_frac"],
                            resample_cap=cfg["sae"]["resample_cap"], device=dev, seed=ds_seed)

            sel = _load_selection(p1_root, ds, tap, cfg["fallback_tap"])
            if sel is None:
                # derive on E_select here as a fallback (still leakage-safe)
                sel = _derive_selection_on_eselect(model, sae, Xte, Yte, env["E_select"],
                                                   tvar_abs, tap, dev, p2["feature_budget_k"], ds_seed)

            # per-env: feature signal + score, per condition
            per_env = {}
            for tag, widx in eval_env_windows.items():
                Xe, Ye = Xte[widx], Yte[widx]
                ce = collect(model, Xe, Ye, tvar_abs, device=dev)
                act_e = _tap_act(model, Xe, tvar_abs, tap, dev)
                z_e = encode_all(sae, act_e, device=dev)
                per_env[tag] = dict(z=z_e, score=ce["score"])

            results = {}
            for cond in p2["conditions"]:
                fset = np.asarray(sel.get(cond, []), dtype=int)
                if len(fset) == 0:
                    results[cond] = {"note": "empty feature set"}
                    continue
                curves = {}
                for tag, d in per_env.items():
                    signal = build_feature_signal(d["z"], fset)
                    curves[tag] = conditional_difficulty_curve(signal, d["score"], p2["n_bins"])
                inv = invariance_score(curves)
                inv["curves"] = {k: v.tolist() for k, v in curves.items()}
                # split ID-only vs shift-inclusive invariance
                id_only = {k: v for k, v in curves.items() if k == "E_eval_ID"}
                shift_curves = {k: v for k, v in curves.items() if "shift" in k}
                inv["I_shift_vs_id"] = _pair_I(curves.get("E_eval_ID"),
                                               list(shift_curves.values()))
                results[cond] = inv

            json.dump(dict(dataset=ds, tap=tap, env_split=env_desc,
                           eval_tags=eval_tags, results=results,
                           forecaster_val_mse=float(val_mse)),
                      open(os.path.join(rd, "invariance.json"), "w"), indent=2, default=str)

            row = dict(dataset=ds, tap=tap)
            for cond in p2["conditions"]:
                r = results.get(cond, {})
                row[f"I_{cond}"] = r.get("I")
                row[f"stab_{cond}"] = r.get("spearman_stability")
                row[f"Ishift_{cond}"] = r.get("I_shift_vs_id")
                row[f"degenerate_{cond}"] = bool(r.get("degenerate", False))
            # the pre-registered comparisons
            row["I_MVDFv2_lt_correlated"] = _lt(results, "MVDF_v2", "correlated", "I")
            row["I_MVDFv2_lt_MVDFv1"] = _lt(results, "MVDF_v2", "MVDF_v1", "I")
            agg.append(row)
            master["per_run"][f"{ds}_{tap}"] = dict(
                seconds=round(time.time() - t0, 1),
                I_by_condition={c: results.get(c, {}).get("I") for c in p2["conditions"]},
                MVDFv2_beats_correlated_on_I=row["I_MVDFv2_lt_correlated"],
                MVDFv2_beats_MVDFv1_on_I=row["I_MVDFv2_lt_MVDFv1"],
            )
            print(f"  I by condition: "
                  + "  ".join(f"{c}={results.get(c, {}).get('I')}" for c in p2["conditions"]))
            print(f"  I(MVDF_v2) < I(correlated)? {row['I_MVDFv2_lt_correlated']}   "
                  f"I(MVDF_v2) < I(MVDF_v1)? {row['I_MVDFv2_lt_MVDFv1']}")

    pd.DataFrame(agg).to_csv(os.path.join(out_root, "AGG_invariance.csv"), index=False)
    master["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(master, open(os.path.join(out_root, "run_log.json"), "w"), indent=2, default=str)
    print(f"\nPhase 2 outputs -> {out_root}\n  AGG_invariance.csv — the flagship table")
    print("  Pre-registered read (spec §7): I(MVDF_v2) < I(correlated) across datasets → row 1/2;")
    print("  I(MVDF_v2) ≈ I(correlated) → row 3 (genuine negative on invariance, distinct from Run 5).")


def _pair_I(id_curve, shift_curves):
    if id_curve is None or not shift_curves:
        return None
    from mechconf.validation.invariance import invariance_score
    out = []
    for sc in shift_curves:
        r = invariance_score({"id": np.asarray(id_curve), "sh": np.asarray(sc)})
        if np.isfinite(r.get("I", np.nan)):
            out.append(r["I"])
    return float(np.mean(out)) if out else None


def _lt(results, a, b, key):
    va = results.get(a, {}).get(key); vb = results.get(b, {}).get(key)
    if va is None or vb is None or not np.isfinite(va) or not np.isfinite(vb):
        return None
    return bool(va < vb)


def _winning_taps(p1_root, ds, fallback):
    summ = os.path.join(os.path.dirname(p1_root), "v2_phase0", "layer_sweep_summary.json")
    try:
        p0 = json.load(open(summ))
        if p0["result"] == "GO":
            return p0["winning_layers"]
    except FileNotFoundError:
        pass
    return [fallback]


def _load_selection(p1_root, ds, tap, fallback_tap, k=12):
    """Load the 5 feature-set conditions from Phase 1 output.

    intervention_only is derived HERE from Phase 1's candidates.csv: top-k by |U_j|
    with NO R_j residualization — the diagnostic that isolates "dose-response alone,
    no specificity gate" so Phase 2 can test whether the R_j gate adds invariance
    (spec §8: `intervention_only_set`). Earlier it was collapsed to MVDF_v1 as a
    stopgap; that made the ablation impossible.
    """
    for t in (tap, fallback_tap):
        rd = os.path.join(p1_root, f"{ds}_{t}")
        p = os.path.join(rd, "mvdf_v2_selection.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        intervention_only = d.get("MVDF_v1", [])          # fallback if candidates.csv missing
        cpath = os.path.join(rd, "candidates.csv")
        if os.path.exists(cpath):
            import pandas as pd
            cdf = pd.read_csv(cpath)
            cdf = cdf.dropna(subset=["U_j"])
            cdf["abs_U"] = cdf["U_j"].abs()
            intervention_only = cdf.sort_values("abs_U", ascending=False)["feature"].astype(int).tolist()[:k]
        return {"random": d["random"], "correlated": d["correlated"],
                "MVDF_v1": d["MVDF_v1"], "MVDF_v2": d["MVDF_v2"],
                "intervention_only": intervention_only}
    return None


def _derive_selection_on_eselect(model, sae, Xte, Yte, sel_mask, tvar, tap, dev, k, seed):
    """Fallback if Phase 1 output is absent: select on E_select only (leakage-safe)."""
    from mechconf.representation.layer_sweep import extract_all_taps
    Xs, Ys = Xte[sel_mask], Yte[sel_mask]
    cs = collect(model, Xs, Ys, tvar, device=dev)
    act = extract_all_taps(model, Xs, tvar, device=dev)[tap]
    z = encode_all(sae, act, device=dev)
    scr = screen(z, cs["score"], min_active_frac=0.02)
    elig = np.where(scr["eligible"])[0]
    rng = np.random.default_rng(seed)
    corr = np.abs(scr["pearson"]); corr[~scr["eligible"]] = -np.inf
    top_corr = np.argsort(-corr)[:k]
    rand = np.sort(rng.choice(elig, size=k, replace=False))
    return {"random": rand.tolist(), "correlated": top_corr.tolist(),
            "MVDF_v1": top_corr.tolist(), "MVDF_v2": top_corr.tolist(),
            "intervention_only": top_corr.tolist()}


if __name__ == "__main__":
    main()
