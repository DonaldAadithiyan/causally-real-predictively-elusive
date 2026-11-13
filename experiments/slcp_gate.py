"""SLCP gate (SAE-localized conformal prediction spec §1) — is there local structure in
S_t for a similarity-weighted method to exploit, BEFORE building weighted-quantile
machinery?

For a sample of E_eval_ID windows, in EACH of the 5 representation spaces
(dense / mvdf / correlated / random_k / pca_k), compute each window's 50-NN set and the
variance of S_t among those neighbors. Average across the sample.

  ratio_space = mean_local_var(S_t | space 50-NN) / mean_local_var(S_t | dense 50-NN)

PRE-COMMITTED STOP RULE (locked 2026-09-04, before running):
  base condition — for at least one of {mvdf, correlated, random_k}:
    - ratio vs dense  < 0.80  AND paired-bootstrap CI (over the window sample) excludes 1
    - ratio vs pca_k  < 0.80  AND CI excludes 1
    - BOTH hold on ALL 6 dataset-seed cells (ETTh1/ETTh2 x seed 0/1/2)

  TWO CONFOUND CONTROLS (also cheap, cached), interpretation pre-committed:
   A. temporal-adjacency neighbours: k-NN = the 50 calendar-nearest windows
      (|t_i - t_j| smallest), zero feature info. Ratio vs dense/pca_k computed the
      same way.
   B. random-Gaussian dense projection: a FIXED random 12x256 Gaussian matrix applied
      to the dense pre-head vector (not PCA, not SAE). k-NN in that 12-d space.

  DECISION:
   - if control A (temporal) alone clears <0.80  -> STOP. Premise compromised: even
     correlated/random_k "pass" is likely recovering serial correlation of S_t, not
     feature content. Write as a methodological note (local-structure claims in time
     series must control for temporal autocorrelation), NOT a falsification of SAE
     structure per se.
   - elif control B (random-Gaussian-dense) clears <0.80  -> STOP / very weak. Effect
     is generic to any linear compression, not SAE-specific.
   - elif neither control clears it AND correlated/random_k still do  -> PROCEED to §2,
     with `correlated` or `random_k` as the kernel space, NOT `mvdf`. mvdf's non-pass is
     explained by the H1/H1b divergence (causal selection != observational tracking;
     k-NN localization is observational).
   - else  -> STOP. Third data point (after Task G, H7-A) that SAE structure's downstream
     advantage does not generalize.

k-NN = 50 (~2.5% of E_eval_ID), same neighbor count in all 5 spaces. Euclidean distance
in each space's feature coordinates (features are the SAE codes / dense pre-head vector /
PCA components exactly as build_features produces them). Reloads the deterministic
forecaster + SAE per (dataset, seed) — no sweep, no new training config.

Scope: ETTh1 + ETTh2 only (traffic excluded per Task A, electricity per Task B).

Outputs runs/slcp_gate/SLCP_gate.csv + SLCP_gate_decision.json
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
from calibration.difficulty_scorer import build_features

K_NN = 50
RATIO_THRESH = 0.80
SAE_SPACES = ["mvdf", "correlated", "random_k"]
ALL_SPACES = ["dense", "mvdf", "correlated", "random_k", "pca_k",
              "temporal_adj", "rand_gauss_dense"]  # last two = confound controls
N_SAMPLE = 800   # windows sampled from E_eval_ID for the local-variance estimate
TEMPORAL_EXCLUDE = 40   # §2: exclude +/- this many windows in time from each point's NN pool


def local_var_S(feats: np.ndarray, S: np.ndarray, sample_idx: np.ndarray, k: int,
                temporal_exclude: int = 0) -> np.ndarray:
    """For each sampled window, variance of S among its k nearest neighbours (by
    Euclidean distance in `feats`), EXCLUDING itself. If temporal_exclude > 0, also
    exclude any window within +/- that many indices in time from the candidate pool
    (structurally removes the serial-correlation confound). Returns per-sample array."""
    out = np.empty(len(sample_idx))
    n = len(S)
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True) + 1e-8
    F = (feats - mu) / sd
    for i, t in enumerate(sample_idx):
        d = np.sqrt(((F - F[t]) ** 2).sum(axis=1))
        d[t] = np.inf
        if temporal_exclude > 0:
            lo, hi = max(0, t - temporal_exclude), min(n, t + temporal_exclude + 1)
            d[lo:hi] = np.inf
        nn = np.argpartition(d, k)[:k]
        out[i] = float(np.var(S[nn]))
    return out


def local_var_S_temporal(S: np.ndarray, sample_idx: np.ndarray, k: int) -> np.ndarray:
    """Control A: neighbours = the k windows closest in index (calendar time),
    zero feature information. E_eval_ID windows are contiguous in the test stream so
    index distance == time distance."""
    n = len(S)
    out = np.empty(len(sample_idx))
    for i, t in enumerate(sample_idx):
        lo = max(0, t - k // 2 - k)   # generous window, then take the k nearest by |Δidx|
        hi = min(n, t + k // 2 + k + 1)
        cand = np.arange(lo, hi)
        cand = cand[cand != t]
        nn = cand[np.argsort(np.abs(cand - t))[:k]]
        out[i] = float(np.var(S[nn]))
    return out


def paired_boot_ratio(num: np.ndarray, den: np.ndarray, n_boot: int = 5000, seed: int = 0):
    """CI on mean(num)/mean(den), paired bootstrap over the window sample."""
    rng = np.random.default_rng(seed)
    n = len(num)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = num[idx].mean() / den[idx].mean()
    return dict(
        ratio=float(num.mean() / den.mean()),
        ci_lo=float(np.percentile(vals, 2.5)),
        ci_hi=float(np.percentile(vals, 97.5)),
        p_lt_thresh=float((vals < RATIO_THRESH).mean()),
        ci_excludes_1_below=bool(np.percentile(vals, 97.5) < 1.0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2_phase2.yaml")
    ap.add_argument("--phase1_root_tmpl", default="v2_confirm/v2_phase1_seed{seed}")
    ap.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.fast:
        cfg["forecaster"]["max_epochs"] = 3
        cfg["sae"]["steps"] = 3000

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "runs", "slcp_gate")
    os.makedirs(out, exist_ok=True)
    dd = cfg["dataset_defaults"]
    L, H, tvar = dd["lookback"], dd["horizon"], dd["target_var"]
    tap = cfg["fallback_tap"]
    dev = cfg["device"]

    rows = []
    te_rows = []  # §2: temporal-exclusion-radius check
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

            sel = json.load(open(os.path.join(here, args.phase1_root_tmpl.format(seed=seed),
                                              f"{ds}_{tap}", "mvdf_v2_selection.json")))
            mvdf_idx = np.array(sel["MVDF_v2"], dtype=int)
            corr_idx = np.array(sel["correlated"], dtype=int)
            k = len(mvdf_idx)

            env = three_way_env_masks(end_te, len(sd.test))
            id_idx = np.where(env["E_eval_ID"])[0]
            sel_idx = np.where(env["E_select"])[0]

            def bundle(widx):
                Xe, Ye = Xte[widx], Yte[widx]
                ce = collect(model, Xe, Ye, tvar_abs, device=dev)
                act = extract_all_taps(model, Xe, tvar_abs, device=dev)[tap]
                z = encode_all(sae, act, device=dev)
                return dict(hidden=ce["hidden"], z=z, score=np.asarray(ce["score"], float))

            b_id = bundle(id_idx)
            b_sel = bundle(sel_idx)

            # random_k / pca_k built exactly as h6_and_diagnostic does (deterministic)
            rng = np.random.default_rng(ds_seed + 777)
            elig = np.where((b_sel["z"] != 0).mean(0) >= 0.02)[0]
            rand_idx = np.sort(rng.choice(elig, size=min(k, len(elig)), replace=False))
            from sklearn.decomposition import PCA
            pca_model = PCA(n_components=k).fit(b_sel["hidden"].astype(np.float64))
            fkw = dict(corr_idx=corr_idx, rand_idx=rand_idx, pca_model=pca_model, k=k)

            n_id = len(id_idx)
            rs = np.random.default_rng(ds_seed + 999)
            samp = rs.choice(n_id, size=min(N_SAMPLE, n_id), replace=False)

            feat_spaces = ["dense", "mvdf", "correlated", "random_k", "pca_k"]
            feats = {sp: build_features(sp, b_id["hidden"], b_id["z"], mvdf_idx, **fkw)
                     for sp in feat_spaces}
            # control B: fixed random 12x256 Gaussian projection of the dense vector
            gproj = np.random.default_rng(ds_seed + 4242).standard_normal(
                (b_id["hidden"].shape[1], k)) / np.sqrt(b_id["hidden"].shape[1])
            feats["rand_gauss_dense"] = (b_id["hidden"].astype(np.float64) @ gproj).astype(np.float32)

            lv = {sp: local_var_S(feats[sp], b_id["score"], samp, K_NN) for sp in feats}
            # control A: temporal-adjacency neighbours (no features)
            lv["temporal_adj"] = local_var_S_temporal(b_id["score"], samp, K_NN)

            for sp in ALL_SPACES:
                r_vs_dense = paired_boot_ratio(lv[sp], lv["dense"], seed=ds_seed)
                r_vs_pca = paired_boot_ratio(lv[sp], lv["pca_k"], seed=ds_seed + 1)
                rows.append(dict(
                    dataset=ds, seed=seed, space=sp,
                    mean_local_var=float(lv[sp].mean()),
                    ratio_vs_dense=r_vs_dense["ratio"],
                    ratio_vs_dense_ci=[r_vs_dense["ci_lo"], r_vs_dense["ci_hi"]],
                    vs_dense_lt80_and_ci_excl_1=bool(r_vs_dense["ratio"] < RATIO_THRESH
                                                     and r_vs_dense["ci_excludes_1_below"]),
                    ratio_vs_pca=r_vs_pca["ratio"],
                    ratio_vs_pca_ci=[r_vs_pca["ci_lo"], r_vs_pca["ci_hi"]],
                    vs_pca_lt80_and_ci_excl_1=bool(r_vs_pca["ratio"] < RATIO_THRESH
                                                   and r_vs_pca["ci_excludes_1_below"]),
                ))
                print(f"[{ds} s{seed}] {sp:11s} local_var(S)={lv[sp].mean():.4f}  "
                      f"vs dense={r_vs_dense['ratio']:.3f} CI[{r_vs_dense['ci_lo']:.3f},{r_vs_dense['ci_hi']:.3f}]  "
                      f"vs pca={r_vs_pca['ratio']:.3f} CI[{r_vs_pca['ci_lo']:.3f},{r_vs_pca['ci_hi']:.3f}]")

            # --- §2: temporal-exclusion-radius check (additive, does not touch §1's result) ---
            # exclude +/- TEMPORAL_EXCLUDE windows in time from each point's candidate NN
            # pool, then recompute local var(S_t). If nothing clears the bar once the
            # serial-correlation confound is structurally removed, that's a NUMBER behind
            # the "not enough signal once you control for time" claim, not just an argument.
            lv_te = {sp: local_var_S(feats[sp], b_id["score"], samp, K_NN,
                                     temporal_exclude=TEMPORAL_EXCLUDE)
                     for sp in feat_spaces}
            for sp in feat_spaces:
                r_vs_dense = paired_boot_ratio(lv_te[sp], lv_te["dense"], seed=ds_seed + 2)
                r_vs_pca = paired_boot_ratio(lv_te[sp], lv_te["pca_k"], seed=ds_seed + 3)
                te_rows.append(dict(
                    dataset=ds, seed=seed, space=sp, temporal_exclude=TEMPORAL_EXCLUDE,
                    mean_local_var=float(lv_te[sp].mean()),
                    ratio_vs_dense=r_vs_dense["ratio"],
                    ratio_vs_dense_ci=[r_vs_dense["ci_lo"], r_vs_dense["ci_hi"]],
                    vs_dense_lt80_and_ci_excl_1=bool(r_vs_dense["ratio"] < RATIO_THRESH
                                                     and r_vs_dense["ci_excludes_1_below"]),
                    ratio_vs_pca=r_vs_pca["ratio"],
                    ratio_vs_pca_ci=[r_vs_pca["ci_lo"], r_vs_pca["ci_hi"]],
                    vs_pca_lt80_and_ci_excl_1=bool(r_vs_pca["ratio"] < RATIO_THRESH
                                                   and r_vs_pca["ci_excludes_1_below"]),
                ))
                print(f"  [TEMPORAL-EXCLUDED +/-{TEMPORAL_EXCLUDE}] {sp:11s} "
                      f"local_var(S)={lv_te[sp].mean():.4f}  "
                      f"vs dense={r_vs_dense['ratio']:.3f}  vs pca={r_vs_pca['ratio']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "SLCP_gate.csv"), index=False)
    df_te = pd.DataFrame(te_rows)
    df_te.to_csv(os.path.join(out, "SLCP_gate_temporal_excluded.csv"), index=False)

    # pre-committed rule
    n_cells = len(args.datasets) * len(args.seeds)

    def cells_pass(sp):
        g = df[df.space == sp]
        return int((g["vs_dense_lt80_and_ci_excl_1"] & g["vs_pca_lt80_and_ci_excl_1"]).sum())

    def summ(sp):
        g = df[df.space == sp]
        return dict(cells_passing_both=cells_pass(sp), n_cells=n_cells,
                    all_cells_pass=bool(cells_pass(sp) == n_cells),
                    ratio_vs_dense_by_cell=[round(float(x), 4) for x in g["ratio_vs_dense"]],
                    ratio_vs_pca_by_cell=[round(float(x), 4) for x in g["ratio_vs_pca"]])

    per_sae = {sp: summ(sp) for sp in SAE_SPACES}
    ctrl_A = summ("temporal_adj")       # temporal-adjacency
    ctrl_B = summ("rand_gauss_dense")   # random Gaussian projection
    sae_passes = [sp for sp in SAE_SPACES if per_sae[sp]["all_cells_pass"]]

    if ctrl_A["all_cells_pass"]:
        proceed, verdict = False, (
            "STOP — CONTROL A (temporal-adjacency) clears the bar. The premise is "
            "compromised: k-NN in ANY reasonably smooth space recovers temporally-nearby "
            "windows, and S_t is serially correlated, so the apparent 'local structure' "
            "in correlated/random_k is likely serial correlation, not feature content. "
            "Write as a methodological note on local-structure claims in time series; "
            "NOT a falsification of SAE structure per se.")
    elif ctrl_B["all_cells_pass"]:
        proceed, verdict = False, (
            "STOP / very weak — CONTROL B (random Gaussian dense projection) clears the "
            "bar. The effect is generic to any 12-d linear compression of the dense "
            "vector, not SAE-specific. correlated/random_k passing is not about the SAE "
            "dictionary.")
    elif sae_passes:
        proceed, verdict = True, (
            f"PROCEED to spec §2 with kernel space in {sae_passes} (NOT mvdf). Neither "
            "confound control cleared the bar; SAE-space-specific local structure in S_t "
            "is real. mvdf's non-pass is consistent with the H1/H1b divergence (causal "
            "selection != observational tracking; k-NN localization is observational).")
    else:
        proceed, verdict = False, (
            "STOP — no SAE space clears the pre-committed bar and no control explains it "
            "away either. Third data point (after Task G, H7-A) that SAE structure's "
            "downstream advantage does not generalize. SLCP spec closed; do not build §2.")

    # --- §2 read: temporal-exclusion-radius check ---
    def summ_te(sp):
        g = df_te[df_te.space == sp]
        pc = int((g["vs_dense_lt80_and_ci_excl_1"] & g["vs_pca_lt80_and_ci_excl_1"]).sum())
        return dict(cells_passing_both=pc, n_cells=n_cells, all_cells_pass=bool(pc == n_cells),
                    ratio_vs_dense_by_cell=[round(float(x), 4) for x in g["ratio_vs_dense"]],
                    ratio_vs_pca_by_cell=[round(float(x), 4) for x in g["ratio_vs_pca"]])
    per_sae_te = {sp: summ_te(sp) for sp in SAE_SPACES}
    sae_te_passes = [sp for sp in SAE_SPACES if per_sae_te[sp]["all_cells_pass"]]
    if sae_te_passes:
        te_verdict = (f"SURVIVES temporal exclusion (+/-{TEMPORAL_EXCLUDE}): "
                      f"{sae_te_passes} still clear the bar once temporally-near windows are "
                      "structurally removed from the candidate pool. Not enough by itself to "
                      "reopen SLCP §2 this phase (stop condition applies regardless of outcome) "
                      "— report as a documented open direction: a temporally-restricted SLCP "
                      "variant may be genuinely buildable.")
    else:
        te_verdict = (f"CONFIRMS the confound (+/-{TEMPORAL_EXCLUDE}): no SAE space clears the "
                      "bar once temporally-near windows are excluded from the candidate pool. "
                      "'Not enough signal once you control for time' is now a number, not just "
                      "an argument — the §1 STOP verdict is empirically grounded.")

    decision = dict(
        k_nn=K_NN, ratio_threshold=RATIO_THRESH, n_sample=N_SAMPLE,
        rule=("base: >=1 of {mvdf,correlated,random_k} has ratio<0.80 AND CI-excl-1 vs "
              "BOTH dense and pca_k on ALL cells. Then control A (temporal) / control B "
              "(rand-gauss-dense) gate the interpretation — see verdict."),
        per_sae_space=per_sae,
        control_A_temporal_adjacency=ctrl_A,
        control_B_rand_gauss_dense=ctrl_B,
        sae_spaces_passing_base=sae_passes,
        PROCEED_TO_SLCP=proceed,
        verdict=verdict,
        temporal_exclusion_check=dict(
            temporal_exclude_radius=TEMPORAL_EXCLUDE,
            per_sae_space=per_sae_te,
            sae_spaces_surviving=sae_te_passes,
            verdict=te_verdict,
        ),
    )
    json.dump(decision, open(os.path.join(out, "SLCP_gate_decision.json"), "w"), indent=2)
    print("\n" + "=" * 70)
    print(json.dumps(decision, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
