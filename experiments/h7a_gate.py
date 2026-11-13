"""H7-A gate (MG-CP spec §1) — does high Type-B (cell-B) co-activation induce a
separated empirical tail-error regime?

  R_Q = Q_.90(S_t | D_t=1) / Q_.90(S_t | D_t=0)

D_t = 1  iff  A_t  is in the TOP 10% of windows, where
  A_t = (1/|B|) * sum_j 1[z_tj != 0]   = fraction of cell-B features active in window t.
The A_t threshold tau_A is the 90th percentile of A_t computed on E_SELECT ONLY, then
applied unchanged to the E_eval blocks (leakage discipline, spec §4).

  NOTE (decided 2026-09-04): the original spec assumed ~3.7% Type-B activation and a
  scarce D=1 regime. That 3.7% was the *per-feature* rate; ">= 1 of 24 active" aggregates
  to ~55-60% of windows (verified). So D_t is defined as the top-decile of the *fraction*
  active A_t, which restores a rare (~10%), sharply-separated regime and matches spec
  §2's small-n_1 / shrinkage design intent. n_1 ~ 190-200 per dataset-seed on ETT
  E_eval_ID.

Cell-B set = feature_type_table.json['cells']['B'], per (dataset, seed), fresh from that
seed's own SAE (consistent with every other per-seed metric in this project). NOT
MVDF_v2 — H7-A tests the type-B *construct*, not the k=12 selection pipeline output.
S_t = validation/errors.collect(...)['score'] = max_h |e_h| / profile_h  (frozen score).

Runs ETTh1 + ETTh2 only (Task A excludes traffic, Task B confounds electricity), seeds
0/1/2. Reloads the deterministic forecaster + SAE (identical to Phase 1/2 by ds_seed) —
no sweep, no new training config. Reports R_Q with a paired bootstrap CI, AND n_1 / n_0
/ absolute Q_.90 values so a noisy-tail-quantile artefact is visible.

PRE-COMMITTED STOP RULE (spec §1): proceed to MG-CP-1 only if the bootstrap CI on R_Q
excludes 1 in the R_Q > 1 direction, on **E_eval_ID**, on **both datasets**, **all
seeds**. E_eval_shift is computed and reported as a SECONDARY, NON-GATING readout — if
it also separates, that is a materially stronger result; if it does not, MG-CP-1 is
still built (the gate passed) but the regime signal is ID-only, which bears on §6's
shift emphasis. If E_eval_ID fails on either dataset -> stop; H7-A negative (H1c stands,
sharpened).

Outputs runs/h7a_gate/H7A.csv + H7A_decision.json
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

Q = 0.90       # calibration level everything in this project uses (alpha = 0.10)
TAU_PCTL = 90  # D_t = 1 iff A_t >= this percentile of A_t on E_select (top decile)


def q_ratio(s: np.ndarray, d: np.ndarray, q: float = Q):
    """Q_q(s|d=1) / Q_q(s|d=0), plus the pieces."""
    s1, s0 = s[d == 1], s[d == 0]
    if len(s1) == 0 or len(s0) == 0:
        return dict(R_Q=np.nan, q1=np.nan, q0=np.nan, n1=int(len(s1)), n0=int(len(s0)))
    q1 = float(np.quantile(s1, q))
    q0 = float(np.quantile(s0, q))
    return dict(R_Q=q1 / q0 if q0 > 0 else np.nan, q1=q1, q0=q0,
                n1=int(len(s1)), n0=int(len(s0)))


def boot_ci(s: np.ndarray, d: np.ndarray, n_boot: int = 5000, seed: int = 0, q: float = Q):
    """Paired bootstrap over windows: resample (s_t, d_t) pairs, recompute R_Q.
    Returns CI and P(R_Q > 1)."""
    rng = np.random.default_rng(seed)
    n = len(s)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        r = q_ratio(s[idx], d[idx], q)
        vals[b] = r["R_Q"]
    vals = vals[np.isfinite(vals)]
    return dict(
        ci_lo=float(np.percentile(vals, 2.5)),
        ci_hi=float(np.percentile(vals, 97.5)),
        p_gt1=float((vals > 1.0).mean()),
        excludes_1_above=bool(np.percentile(vals, 2.5) > 1.0),
        n_boot_valid=int(len(vals)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2_phase2.yaml")
    ap.add_argument("--phase1_root_tmpl", default="v2_confirm/v2_phase1_seed{seed}")
    ap.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--gate-env", default="E_eval_ID",
                    help="which env block the pre-committed stop rule keys on")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.fast:
        cfg["forecaster"]["max_epochs"] = 3
        cfg["sae"]["steps"] = 3000

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "runs", "h7a_gate")
    os.makedirs(out, exist_ok=True)
    dd = cfg["dataset_defaults"]
    L, H, tvar = dd["lookback"], dd["horizon"], dd["target_var"]
    tap = cfg["fallback_tap"]
    dev = cfg["device"]

    rows = []
    for ds in args.datasets:
        for seed in args.seeds:
            ds_seed = derive_seed(seed, ds)
            np.random.seed(ds_seed); torch.manual_seed(ds_seed)
            sd = get_splits(ds, max_channels=dd["max_channels"])
            assert_expected_data(ds, sd, strict=cfg.get("strict_fingerprint", False))
            Xtr, Ytr, _ = make_windows(sd.train, L, H)
            Xva, Yva, _ = make_windows(sd.val, L, H)
            Xte, Yte, end_te = make_windows(sd.test, L, H)
            tvar_abs = tvar % len(sd.channels)

            fc = cfg["forecaster"]
            tcfg = TrainConfig(lookback=L, horizon=H, d_model=fc["d_model"], n_layers=fc["n_layers"],
                               n_heads=fc["n_heads"], d_ff=fc["d_ff"], dropout=fc["dropout"],
                               lr=fc["lr"], batch_size=fc["batch_size"], max_epochs=fc["max_epochs"],
                               patience=fc["patience"], lr_decay=fc.get("lr_decay", 0.8),
                               warmup_epochs=fc.get("warmup_epochs", 3), seed=ds_seed, device=dev)
            model, _ = train_itransformer((Xtr, Ytr, None), (Xva, Yva, None),
                                          n_vars=len(sd.channels), cfg=tcfg, verbose=False)
            tr_act = extract_all_taps(model, Xtr, tvar_abs, device=dev)[tap]
            sae = train_sae(tr_act, expansion=cfg["sae"]["expansion"], k=cfg["sae"]["k"],
                            steps=cfg["sae"]["steps"], batch_size=cfg["sae"]["batch_size"],
                            lr=cfg["sae"]["lr"], resample_every=cfg["sae"]["resample_every"],
                            resample_until_frac=cfg["sae"]["resample_until_frac"],
                            resample_cap=cfg["sae"]["resample_cap"], device=dev, seed=ds_seed)

            # cell-B set for THIS seed
            ftp = os.path.join(here, args.phase1_root_tmpl.format(seed=seed),
                               f"{ds}_{tap}", "feature_type_table.json")
            cellB = np.array(json.load(open(ftp))["cells"]["B"], dtype=int)

            def A_t(widx):
                """fraction of cell-B features active per window over the given window idx."""
                act = extract_all_taps(model, Xte[widx], tvar_abs, device=dev)[tap]
                z = encode_all(sae, act, device=dev)
                return (z[:, cellB] != 0).mean(axis=1)

            env = three_way_env_masks(end_te, len(sd.test))
            # tau_A on E_SELECT only (leakage-safe)
            sel_widx = np.where(env["E_select"])[0]
            A_sel = A_t(sel_widx)
            tau_A = float(np.percentile(A_sel, TAU_PCTL))

            for env_tag in ["E_eval_ID", "E_eval_shift"]:
                widx = np.where(env[env_tag])[0]
                Xe, Ye = Xte[widx], Yte[widx]
                ce = collect(model, Xe, Ye, tvar_abs, device=dev)
                s = np.asarray(ce["score"], dtype=float)
                a = A_t(widx)
                d = (a >= tau_A).astype(int)   # D_t = 1 iff A_t in top decile (tau_A from E_select)

                r = q_ratio(s, d)
                bc = boot_ci(s, d, seed=ds_seed)
                rows.append(dict(
                    dataset=ds, seed=seed, env=env_tag, gating=(env_tag == "E_eval_ID"),
                    n_cellB=int(len(cellB)), tau_A=tau_A, tau_pctl=TAU_PCTL,
                    activation_rate=float(d.mean()),   # should land ~0.10 on E_eval_ID
                    A_sel_p50=float(np.percentile(A_sel, 50)),
                    A_eval_p50=float(np.percentile(a, 50)),
                    **r, **bc,
                ))
                tag = "GATE" if env_tag == "E_eval_ID" else "readout"
                print(f"[{ds} s{seed} {env_tag} {tag}] R_Q={r['R_Q']:.3f}  "
                      f"CI[{bc['ci_lo']:.3f},{bc['ci_hi']:.3f}]  P(>1)={bc['p_gt1']:.3f}  "
                      f"n1={r['n1']} n0={r['n0']}  D_rate={d.mean():.4f}  tau_A={tau_A:.4f}  "
                      f"Q1={r['q1']:.3f} Q0={r['q0']:.3f}"
                      f"  {'EXCL>1' if bc['excludes_1_above'] else 'incl 1'}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "H7A.csv"), index=False)

    # --- pre-committed stop rule: gate env, per dataset, ALL seeds must exclude 1 above
    gate = df[df.env == args.gate_env]
    per_ds = {}
    for ds, g in gate.groupby("dataset"):
        seeds_excl = g["excludes_1_above"].tolist()
        per_ds[ds] = dict(
            n_seeds=int(len(g)),
            seeds_exclude_1_above=[bool(x) for x in seeds_excl],
            all_seeds_pass=bool(all(seeds_excl)),
            R_Q_by_seed=[round(float(x), 4) for x in g["R_Q"]],
            activation_rate_by_seed=[round(float(x), 5) for x in g["activation_rate"]],
            n1_by_seed=[int(x) for x in g["n1"]],
        )
    proceed = bool(len(per_ds) == len(args.datasets) and all(v["all_seeds_pass"] for v in per_ds.values()))

    # secondary, NON-GATING readout on E_eval_shift
    shift = df[df.env == "E_eval_shift"]
    shift_readout = {}
    for ds, g in shift.groupby("dataset"):
        shift_readout[ds] = dict(
            R_Q_by_seed=[round(float(x), 4) for x in g["R_Q"]],
            seeds_exclude_1_above=[bool(x) for x in g["excludes_1_above"]],
            all_seeds_separate=bool(all(g["excludes_1_above"])),
            n1_by_seed=[int(x) for x in g["n1"]],
        )

    decision = dict(
        gate_env=args.gate_env,
        D_t_definition=f"A_t (frac of {int(df.n_cellB.iloc[0])}-feat cell-B active) >= p{TAU_PCTL} of A_t on E_select",
        rule="CI on R_Q excludes 1 (R_Q>1 direction) on E_eval_ID, BOTH datasets, ALL seeds",
        per_dataset_GATING=per_ds,
        E_eval_shift_SECONDARY_nongating=shift_readout,
        PROCEED_TO_MGCP1=proceed,
        note=("PROCEED to spec §2. " + (
              "E_eval_shift also separates on both datasets — lead with the shift result."
              if all(v["all_seeds_separate"] for v in shift_readout.values())
              else "E_eval_shift does NOT separate cleanly — MG-CP-1 is built, but the regime "
                   "signal is ID-only; §6's CQR/PID comparison should not emphasize the shift "
                   "condition for this method without noting that.")
              if proceed else
              "STOP — H7-A negative on E_eval_ID. H1c stands (sharpened): interventionally "
              "validated uncertainty-specific features exist, but high co-activation of the "
              "cell-B set does not induce a sufficiently separated empirical tail-error regime "
              "for binary conformal conditioning."),
    )
    json.dump(decision, open(os.path.join(out, "H7A_decision.json"), "w"), indent=2)
    print("\n" + "=" * 70)
    print(json.dumps(decision, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
