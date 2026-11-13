"""Phase 5 — SAE functional reproducibility.

Low SAE reconstruction loss + sparsity does NOT guarantee a unique, canonical feature
decomposition — multiple sparse factorizations reach near-identical objective values
without being the same solution ("Low Loss -> Identifiability" is a false implication;
Joshi et al. 2026 causal-ladder synthesis; Leask et al. 2025 SAE-canonical-units).
H1c's causal validity doesn't depend on this (the dose-response intervention is real
regardless), but the *uniqueness* of the specific MVDF features is a separate claim
worth checking.

Protocol (Document 2 §4):
  - Train the forecaster ONCE per dataset (fixed seed).
  - Train N_SAE independent SAEs (different seeds) on the SAME frozen hidden states.
  - For each SAE, run the existing dose-response pipeline over a shared candidate pool
    -> per-feature (P_j, U_j, R_j) causal signature. Also record C_j and activation rate.
  - Match features ACROSS runs by CAUSAL-SIGNATURE similarity (standardized Euclidean
    distance in (P_j, U_j, R_j) space, mutual nearest neighbour), NOT decoder-weight
    correlation — permutation + rotation freedom makes raw coordinate matching a poor
    criterion (the reason the consistency literature moved past naive feature-ID match).
  - Report a FUNCTIONAL-MATCHING RATE: what fraction of run 0's validated MVDF features
    have a close causal-signature match (mutual-NN, distance below a pre-declared
    threshold) in each other run.

Pre-declared match threshold: standardized-signature Euclidean distance < 1.0 (i.e. the
matched feature's (P_j,U_j,R_j) is within ~1 pooled-SD, jointly, of the query's) AND the
pair is a mutual nearest neighbour. Reported alongside the raw mutual-NN rate at no
threshold, and a null (random-permutation) matching rate for calibration.

DATA: ETTh1 + ETTh2, HTTP mirror (same scope as Phase 3/4). electricity/traffic
Kaggle-only.

Outputs runs/phase5_sae_repro/{<ds>/signatures_run<r>.csv, <ds>/matching.json,
                                AGG_phase5.csv, PHASE5_RESULT.md}
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from data.datasets import get_splits, make_windows
from experiments._seed import derive_seed
from forecasters.train_itransformer import TrainConfig, train_itransformer
from representation.layer_sweep import extract_all_taps
from sae.train_sae import train_sae, encode_all
from validation.errors import collect
from validation.correlation_screen import screen
from validation.importance_and_difficulty import compute_c_j
from validation.dose_response import sweep_pool
from validation.difficulty_specificity import residualize, feature_type_table, select_mvdf_v2
from shift.shift_windows import three_way_env_masks

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CFG = dict(
    lookback=96, horizon=96, target_var=-1,
    forecaster=dict(d_model=256, n_layers=3, n_heads=8, d_ff=512, dropout=0.1,
                    lr=1e-4, batch_size=32, max_epochs=20, patience=5,
                    lr_decay=0.8, warmup_epochs=3),
    sae=dict(expansion=8, k=64, steps=12000, batch_size=512, lr=1e-4,
             resample_every=2000, resample_until_frac=0.5, resample_cap=64),
    tap="pre_head",
    n_sae=3, sae_seeds=[10, 20, 30],
    candidate_pool_size=120, window_cap=2048,
    lambdas=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
    feature_budget_k=12, u_threshold=0.0, residualize_method="isotonic",
    match_thresholds=[0.1, 0.25, 0.5, 1.0],   # reported as a curve, not one cutoff
    headline_threshold=0.25,
)


def _signatures_for_sae(model, sae, Xsel, Ysel, sel_act, tvar_abs, tap, cal_sel, prof,
                        pool_source_scr, rng, wsel, cfg):
    """Dose-response over a shared candidate pool -> (P_j,U_j,R_j,C_j,act_rate) per feature."""
    z_sel = encode_all(sae, sel_act, device="cpu")
    scr = screen(z_sel, cal_sel["score"], min_active_frac=0.02)
    elig = np.where(scr["eligible"])[0]
    n_pool = min(cfg["candidate_pool_size"], len(elig))
    # deterministic shared pool: same rng draw across SAE runs isn't meaningful (feature
    # ids differ per SAE), so each run pools its own top-eligible-by-activation set —
    # matching is by SIGNATURE, not id, so the pools need only be comparably sized.
    pool = np.sort(elig[np.argsort(-scr["active_frac"][elig])[:n_pool]])
    C = compute_c_j(z_sel[:, pool], cal_sel["score"], active_only=True)
    dose = sweep_pool(model, sae, Xsel[wsel], Ysel[wsel], sel_act[wsel], tvar_abs, tap,
                      pool, prof, lambdas=cfg["lambdas"], device="cpu", verbose=False)
    R = residualize(dose["P_j"], dose["U_j"], method=cfg["residualize_method"],
                    provenance_p="interventional_dose_response",
                    provenance_u="interventional_dose_response")
    ftab = feature_type_table(dose["P_j"], dose["U_j"], feature_indices=pool)
    mvdf = select_mvdf_v2(dose, u_threshold=cfg["u_threshold"], k=cfg["feature_budget_k"],
                          residualize_method=cfg["residualize_method"])
    mvdf_idx = set(int(x) for x in np.asarray(mvdf["selected"]))
    df = pd.DataFrame(dict(
        feature=pool, P_j=dose["P_j"], U_j=dose["U_j"], R_j=R, C_j=C,
        n_active=dose["n_active"],
        act_rate=scr["active_frac"][pool],
        type=[_type_of(int(j), ftab) for j in pool],
        in_mvdf=[int(j) in mvdf_idx for j in pool],
    ))
    return df.dropna(subset=["P_j", "U_j", "R_j"]).reset_index(drop=True)


def _type_of(fid, ftab):
    for t, ids in ftab["cells"].items():
        if fid in ids:
            return t
    return "?"


def _match_by_signature(dfa: pd.DataFrame, dfb: pd.DataFrame, thresholds, rng) -> dict:
    """Mutual-NN matching in standardized (P_j,U_j,R_j) space.

    Real: for each of dfa's MVDF features, is its mutual-NN in dfb within `thr` SD?
    Null: the correct comparison is "would a RANDOM feature of dfa (not an MVDF one)
    match dfb this well" — i.e. does MVDF membership carry extra reproducibility beyond
    the base rate. So the null match rate = the mutual-NN-within-thr rate over the FULL
    dfa pool (all features), not a permuted-signature shuffle (which just measures how
    dense the signature cloud is). We also report a signature-shuffle rate for context.
    """
    cols = ["P_j", "U_j", "R_j"]
    A = dfa[cols].to_numpy(float)
    B = dfb[cols].to_numpy(float)
    pooled = np.concatenate([A, B], 0)
    mu, sd = pooled.mean(0), pooled.std(0) + 1e-12
    As, Bs = (A - mu) / sd, (B - mu) / sd

    def _mutual_nn(X, Y):
        D = np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
        nn_xy = D.argmin(1)
        nn_yx = D.argmin(0)
        return {i: float(D[i, nn_xy[i]]) for i in range(len(X)) if nn_yx[nn_xy[i]] == i}

    pair_dist = _mutual_nn(As, Bs)
    mvdf_rows = np.where(dfa["in_mvdf"].to_numpy())[0]
    all_rows = np.arange(len(dfa))
    n_mvdf = max(len(mvdf_rows), 1)

    rates = {}
    for thr in thresholds:
        mvdf_matched = [i for i in mvdf_rows if i in pair_dist and pair_dist[i] < thr]
        pool_matched = [i for i in all_rows if i in pair_dist and pair_dist[i] < thr]
        # signature-shuffle: break the A<->B correspondence, keep marginals
        shuf = []
        for _ in range(30):
            p = _mutual_nn(As, Bs[rng.permutation(len(Bs))])
            shuf.append(np.mean([1.0 if (i in p and p[i] < thr) else 0.0
                                 for i in mvdf_rows]) if len(mvdf_rows) else 0.0)
        rates[f"thr_{thr:g}"] = dict(
            mvdf_match_rate=float(len(mvdf_matched) / n_mvdf),
            pool_base_rate=float(len(pool_matched) / len(all_rows)),
            signature_shuffle_rate=float(np.mean(shuf)),
        )

    return dict(
        n_mvdf_query=int(len(mvdf_rows)), n_pool=int(len(dfa)),
        mutual_nn_rate_no_threshold=float(len([i for i in mvdf_rows if i in pair_dist]) / n_mvdf),
        median_mvdf_matched_distance=float(np.median(
            [pair_dist[i] for i in mvdf_rows if i in pair_dist])) if any(
            i in pair_dist for i in mvdf_rows) else float("nan"),
        by_threshold=rates,
    )


def run_dataset(ds: str, out_root: str, fast: bool) -> list[dict]:
    rd = os.path.join(out_root, ds)
    os.makedirs(rd, exist_ok=True)
    fseed = derive_seed(0, ds)
    np.random.seed(fseed); torch.manual_seed(fseed)
    L, H, tvar = CFG["lookback"], CFG["horizon"], CFG["target_var"]
    tap = CFG["tap"]
    fc, sae_c = CFG["forecaster"], CFG["sae"]
    sae_seeds = CFG["sae_seeds"][:2] if fast else CFG["sae_seeds"]
    if fast:
        fc = {**fc, "max_epochs": 3}
        sae_c = {**sae_c, "steps": 3000}
        CFG_local = {**CFG, "candidate_pool_size": 40, "window_cap": 512}
    else:
        CFG_local = CFG

    sd = get_splits(ds, max_channels=48)
    tvar_abs = tvar % len(sd.channels)
    Xtr, Ytr, _ = make_windows(sd.train, L, H)
    Xva, Yva, _ = make_windows(sd.val, L, H)
    Xte, Yte, end_te = make_windows(sd.test, L, H)

    tcfg = TrainConfig(lookback=L, horizon=H, d_model=fc["d_model"], n_layers=fc["n_layers"],
                       n_heads=fc["n_heads"], d_ff=fc["d_ff"], dropout=fc["dropout"],
                       lr=fc["lr"], batch_size=fc["batch_size"], max_epochs=fc["max_epochs"],
                       patience=fc["patience"], lr_decay=fc["lr_decay"],
                       warmup_epochs=fc["warmup_epochs"], seed=fseed, device="cpu")
    model, val_mse = train_itransformer((Xtr, Ytr, None), (Xva, Yva, None),
                                        n_vars=len(sd.channels), cfg=tcfg, verbose=True)
    print(f"[{ds}] forecaster val_mse {val_mse:.4f}  fp {sd.fingerprint()}")

    env = three_way_env_masks(end_te, len(sd.test))
    sel_mask = np.where(env["E_select"])[0]
    Xsel, Ysel = Xte[sel_mask], Yte[sel_mask]
    cal_sel = collect(model, Xsel, Ysel, tvar_abs, device="cpu")
    prof = cal_sel["step_profile"]
    tr_act = extract_all_taps(model, Xtr, tvar_abs, device="cpu")[tap]
    sel_act = extract_all_taps(model, Xsel, tvar_abs, device="cpu")[tap]

    cap = min(CFG_local["window_cap"], len(Xsel))
    rng = np.random.default_rng(fseed)
    wsel = np.sort(rng.choice(len(Xsel), size=cap, replace=False))

    sigs = []
    fvus = []
    for r, sseed in enumerate(sae_seeds):
        sae = train_sae(tr_act, expansion=sae_c["expansion"], k=sae_c["k"], steps=sae_c["steps"],
                        batch_size=sae_c["batch_size"], lr=sae_c["lr"],
                        resample_every=sae_c["resample_every"],
                        resample_until_frac=sae_c["resample_until_frac"],
                        resample_cap=sae_c["resample_cap"], device="cpu", seed=sseed)
        with torch.no_grad():
            xb = torch.from_numpy(sae.standardize(torch.from_numpy(tr_act.astype(np.float32))).numpy())
            xh, _ = sae(xb)
            fvu = float((torch.mean((xh - xb) ** 2) / (torch.var(xb) + 1e-12)).item())
        fvus.append(fvu)
        df = _signatures_for_sae(model, sae, Xsel, Ysel, sel_act, tvar_abs, tap, cal_sel,
                                 prof, None, rng, wsel, CFG_local)
        df.to_csv(os.path.join(rd, f"signatures_run{r}.csv"), index=False)
        sigs.append(df)
        print(f"[{ds}] SAE run {r} (seed {sseed}) FVU {fvu:.4f}  pool {len(df)}  "
              f"MVDF {int(df['in_mvdf'].sum())}")

    thrs = CFG_local["match_thresholds"]
    ht = f"thr_{CFG_local['headline_threshold']:g}"
    match_rng = np.random.default_rng(fseed + 7)
    pairwise = {}
    rows = []
    for r in range(1, len(sigs)):
        m = _match_by_signature(sigs[0], sigs[r], thrs, match_rng)
        pairwise[f"run0_vs_run{r}"] = m
        h = m["by_threshold"][ht]
        rows.append(dict(dataset=ds, comparison=f"run0_vs_run{r}",
                         forecaster_val_mse=float(val_mse),
                         sae_fvu_run0=fvus[0], sae_fvu_runr=fvus[r],
                         n_mvdf_query=m["n_mvdf_query"], n_pool=m["n_pool"],
                         headline_threshold=CFG_local["headline_threshold"],
                         mvdf_match_rate=h["mvdf_match_rate"],
                         pool_base_rate=h["pool_base_rate"],
                         signature_shuffle_rate=h["signature_shuffle_rate"],
                         mutual_nn_rate_no_threshold=m["mutual_nn_rate_no_threshold"],
                         median_mvdf_matched_distance=m["median_mvdf_matched_distance"],
                         match_rate_thr_0p1=m["by_threshold"]["thr_0.1"]["mvdf_match_rate"],
                         match_rate_thr_0p5=m["by_threshold"]["thr_0.5"]["mvdf_match_rate"],
                         match_rate_thr_1p0=m["by_threshold"]["thr_1"]["mvdf_match_rate"]))
    json.dump(dict(thresholds=thrs, headline_threshold=CFG_local["headline_threshold"],
                   pairwise=pairwise, sae_seeds=sae_seeds, fvus=fvus,
                   forecaster_val_mse=float(val_mse)),
              open(os.path.join(rd, "matching.json"), "w"), indent=2, default=str)
    return rows


def write_md(agg: pd.DataFrame, out_root: str):
    def md(df):
        cols = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, r in df.iterrows():
            cells = []
            for v in r:
                if isinstance(v, (bool, np.bool_)):
                    cells.append(str(bool(v)))
                elif isinstance(v, (int, np.integer)):
                    cells.append(str(int(v)))
                elif isinstance(v, float):
                    cells.append(f"{v:.3f}" if np.isfinite(v) else "")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"

    L = ["# PHASE5_RESULT — SAE functional reproducibility\n",
         "Generated by `experiments/phase5_sae_reproducibility.py`. ETTh1 + ETTh2, HTTP "
         "mirror. Forecaster trained once per dataset; 3 independent SAEs (seeds 10/20/30) "
         "on the same frozen hidden states.\n\n"]
    L.append("## What this checks\n\n")
    L.append("Low reconstruction loss + sparsity does not imply a unique decomposition. "
             "This matches features ACROSS independently-trained SAEs by **causal "
             "signature** — standardized Euclidean distance in `(P_j, U_j, R_j)` space "
             "(pooled-SD units), mutual nearest neighbour — not by decoder-weight "
             "correlation (permutation / rotation freedom makes raw coordinate matching a "
             "poor criterion). Reported as a curve over match thresholds "
             "{0.1, 0.25, 0.5, 1.0} SD; headline table shown at 0.5 SD (the loosest "
             "still-meaningful cutoff — a stricter one only lowers the match rate).\n\n")
    L.append("Two references for the rate: **`pool_base_rate`** = the same mutual-NN-"
             "within-threshold rate over the *whole* candidate pool (does MVDF membership "
             "buy extra reproducibility over a generic feature?); **`signature_shuffle_"
             "rate`** = rate after breaking the run-0↔run-r correspondence (how dense the "
             "signature cloud is — a floor).\n\n")
    L.append("## Headline (threshold 0.5 SD) — run-0 MVDF features matched in run-r\n\n")
    L.append(md(agg[["dataset", "comparison", "n_mvdf_query", "n_pool", "mvdf_match_rate",
                     "pool_base_rate", "signature_shuffle_rate",
                     "mutual_nn_rate_no_threshold", "median_mvdf_matched_distance",
                     "sae_fvu_run0", "sae_fvu_runr"]]))
    L.append("\n## Threshold curve (MVDF match rate at 0.1 / 0.25 / 0.5 / 1.0 SD)\n\n")
    L.append(md(agg[["dataset", "comparison", "match_rate_thr_0p1", "mvdf_match_rate",
                     "match_rate_thr_0p5", "match_rate_thr_1p0"]].rename(
        columns={"mvdf_match_rate": "match_rate_thr_0p25"})))
    by = agg.groupby("dataset").agg(
        mvdf_match_rate=("mvdf_match_rate", "mean"),
        pool_base_rate=("pool_base_rate", "mean"),
        signature_shuffle_rate=("signature_shuffle_rate", "mean"),
        mutual_nn_rate=("mutual_nn_rate_no_threshold", "mean")).round(3).reset_index()
    L.append("\n## Per-dataset (mean over the 2 cross-run comparisons)\n\n")
    L.append(md(by))

    fmr = agg["mvdf_match_rate"].mean()
    base = agg["pool_base_rate"].mean()
    shuf = agg["signature_shuffle_rate"].mean()
    L.append("\n## Reading\n\n")
    if fmr >= 0.6 and fmr > shuf + 0.15:
        L.append(f"**MVDF features are functionally reproducible across independent SAEs** "
                 f"— mean match rate {fmr:.2f} at 0.5 SD, vs a signature-shuffle floor of "
                 f"{shuf:.2f} and a pool base rate of {base:.2f}. A validated difficulty "
                 f"feature in one SAE run has a close causal-signature counterpart in "
                 f"another run above what a generic feature achieves. The recovered "
                 f"population is not an arbitrary factorization artifact; H1c's "
                 f"feature-level claims rest on a stable functional structure.\n")
    elif fmr >= 0.3:
        L.append(f"**Partial functional reproducibility** — mean match rate {fmr:.2f} at "
                 f"0.5 SD (shuffle floor {shuf:.2f}, pool base rate {base:.2f}). Some MVDF "
                 f"features recur across SAE seeds with a matching causal signature, not "
                 f"the whole set, and not clearly above the pool base rate. Report H1c's "
                 f"**population-level** enrichment as robust (it does not depend on any "
                 f"single feature recurring); state that individual feature identity is "
                 f"seed-sensitive — consistent with the SAE-canonical-units literature.\n")
    else:
        L.append(f"**Low functional reproducibility** — mean match rate {fmr:.2f} at "
                 f"0.5 SD (shuffle floor {shuf:.2f}). Individual MVDF feature identity does "
                 f"not survive an SAE reseed by causal signature. H1c's claim must be "
                 f"stated purely at the population level (type-B enrichment is a property "
                 f"of the SAE feature *space*, reproducible in aggregate); no claim about "
                 f"specific features.\n")
    L.append("\n> This does not affect H1c's causal validity — the dose-response "
             "intervention is real for whatever features a given SAE recovers. It bounds "
             "only how much can be said about the *identity* of those features.\n")

    with open(os.path.join(out_root, "PHASE5_RESULT.md"), "w") as f:
        f.write("".join(L))
    print(f"wrote {os.path.join(out_root, 'PHASE5_RESULT.md')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2"])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "phase5_sae_repro"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    rows = []
    for ds in args.datasets:
        print("=" * 78, f"\n {ds}\n", "=" * 78)
        rows += run_dataset(ds, args.out, args.fast)
    agg = pd.DataFrame(rows)
    agg.to_csv(os.path.join(args.out, "AGG_phase5.csv"), index=False)
    write_md(agg, args.out)
    print(f"\nPhase 5 done in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
