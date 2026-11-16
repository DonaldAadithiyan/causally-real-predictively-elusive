"""Phase 3 (real data) — the natural-activation predictability battery.

The blocking piece of the extended wrap-up. It has only ever run inside Phase 1's
synthetic environment; this is the real-data version, on ETTh1 + ETTh2.

Question (an L1 / associational test, per the causal-ladder framing — Joshi et al.
2026): using only the ALREADY-VALIDATED MVDF features' *natural* (non-intervened)
activations, can any simple held-out estimator predict S_t at all? A feature found via
L2 (dose-response) evidence does not automatically carry L1 predictive information —
that is precisely what this measures.

Estimators: linear regression, isotonic regression, shallow GBM, pinball MLP — each
trained/evaluated under the same E_train / E_select / E_eval leakage discipline as
everywhere else.

Two readings, reported separately (the doc's "rarity split"):
  - MARGINAL: battery over all E_eval_ID windows.
  - ACTIVATION-CONDITIONED: restricted to windows where the MVDF bundle is active
    (>=1 selected feature non-zero). Distinguishes "no signal because rarely active"
    from "no signal despite being active".

Headline figure data: R_j (causal specificity) vs each feature's best-achieved natural
predictive utility (best single-feature held-out R^2 across the four estimators), across
the full candidate pool — dumped to per-run rj_vs_predictive.csv.

DATA NOTE: ETTh1/ETTh2 here come from the HTTP ETDataset mirror (data/datasets.py),
8640/2880/5900 split. Same split SIZES as the Kaggle bundle used for the paper's H1c
run, slightly different edge values (different fingerprint) — so this retrains Phase 1
locally to get self-consistent MVDF selections rather than reusing the Kaggle indices.
electricity/traffic raw data is Kaggle-only and unavailable locally (same scope limit
as Result 4 / BASELINE_RESULT.md).

Outputs results/v2/phase3_real/:
  <ds>_seed<s>/{mvdf_selection.json, predictability.json, rj_vs_predictive.csv, candidates.csv}
  AGG_phase3.csv
  PHASE3_RESULT.md
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

from mechconf.data.datasets import get_splits, make_windows
from experiments._seed import derive_seed
from mechconf.forecasters.train_itransformer import TrainConfig, train_itransformer
from mechconf.representation.layer_sweep import extract_all_taps
from mechconf.sae.train_sae import train_sae, encode_all
from mechconf.validation.errors import collect
from mechconf.validation.correlation_screen import screen
from mechconf.validation.importance_and_difficulty import compute_c_j, compute_delta_pred_single
from mechconf.validation.dose_response import sweep_pool
from mechconf.validation.difficulty_specificity import (residualize, feature_type_table,
                                               enrichment, select_mvdf_v2)
from mechconf.data.shift_windows import three_way_env_masks
from experiments.v2.phase1_synthetic_control import _fit_eval_estimators, _best_r2

HERE = REPO_ROOT

# Fixed real-data config — matches configs/v2_phase1.yaml (the paper's Phase 1).
CFG = dict(
    lookback=96, horizon=96, target_var=-1,
    forecaster=dict(d_model=256, n_layers=3, n_heads=8, d_ff=512, dropout=0.1,
                    lr=1e-4, batch_size=32, max_epochs=20, patience=5,
                    lr_decay=0.8, warmup_epochs=3),
    sae=dict(expansion=8, k=64, steps=12000, batch_size=512, lr=1e-4,
             resample_every=2000, resample_until_frac=0.5, resample_cap=64),
    tap="pre_head",
    candidate_pool_size=120, window_cap=2048,
    lambdas=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
    feature_budget_k=12, u_threshold=0.0, residualize_method="isotonic",
)


def run_one(ds: str, seed: int, out_root: str, fast: bool) -> dict:
    rd = os.path.join(out_root, f"{ds}_seed{seed}")
    os.makedirs(rd, exist_ok=True)
    ds_seed = derive_seed(seed, ds)
    np.random.seed(ds_seed); torch.manual_seed(ds_seed)
    dev = "cpu"
    L, H, tvar = CFG["lookback"], CFG["horizon"], CFG["target_var"]
    tap = CFG["tap"]
    fc, sae_c = CFG["forecaster"], CFG["sae"]
    if fast:
        fc = {**fc, "max_epochs": 3}
        sae_c = {**sae_c, "steps": 3000}
        pool_size, wcap = 40, 512
    else:
        pool_size, wcap = CFG["candidate_pool_size"], CFG["window_cap"]

    sd = get_splits(ds, max_channels=48)
    tvar_abs = tvar % len(sd.channels)
    Xtr, Ytr, _ = make_windows(sd.train, L, H)
    Xva, Yva, _ = make_windows(sd.val, L, H)
    Xte, Yte, end_te = make_windows(sd.test, L, H)
    print(f"[{ds} s{seed}] windows: train {len(Xtr)} val {len(Xva)} test {len(Xte)}  "
          f"fp {sd.fingerprint()}")

    tcfg = TrainConfig(lookback=L, horizon=H, d_model=fc["d_model"], n_layers=fc["n_layers"],
                       n_heads=fc["n_heads"], d_ff=fc["d_ff"], dropout=fc["dropout"],
                       lr=fc["lr"], batch_size=fc["batch_size"], max_epochs=fc["max_epochs"],
                       patience=fc["patience"], lr_decay=fc["lr_decay"],
                       warmup_epochs=fc["warmup_epochs"], seed=ds_seed, device=dev)
    model, val_mse = train_itransformer((Xtr, Ytr, None), (Xva, Yva, None),
                                        n_vars=len(sd.channels), cfg=tcfg, verbose=True)

    env = three_way_env_masks(end_te, len(sd.test))
    sel_mask = np.where(env["E_select"])[0]
    id_mask = np.where(env["E_eval_ID"])[0]
    Xsel, Ysel = Xte[sel_mask], Yte[sel_mask]
    cal_sel = collect(model, Xsel, Ysel, tvar_abs, device=dev)
    prof = cal_sel["step_profile"]

    tr_act = extract_all_taps(model, Xtr, tvar_abs, device=dev)[tap]
    sae = train_sae(tr_act, expansion=sae_c["expansion"], k=sae_c["k"], steps=sae_c["steps"],
                    batch_size=sae_c["batch_size"], lr=sae_c["lr"],
                    resample_every=sae_c["resample_every"],
                    resample_until_frac=sae_c["resample_until_frac"],
                    resample_cap=sae_c["resample_cap"], device=dev, seed=ds_seed)
    with torch.no_grad():
        xb = torch.from_numpy(sae.standardize(torch.from_numpy(tr_act.astype(np.float32))).numpy())
        xh, _ = sae(xb)
        fvu = float((torch.mean((xh - xb) ** 2) / (torch.var(xb) + 1e-12)).item())
    print(f"[{ds} s{seed}] SAE FVU {fvu:.4f}")

    sel_act = extract_all_taps(model, Xsel, tvar_abs, device=dev)[tap]
    z_sel = encode_all(sae, sel_act, device=dev)
    scr = screen(z_sel, cal_sel["score"], min_active_frac=0.02)
    elig = np.where(scr["eligible"])[0]
    rng = np.random.default_rng(ds_seed)
    n_pool = min(pool_size, len(elig))
    pool = np.sort(rng.choice(elig, size=n_pool, replace=False))

    cap = min(wcap, len(Xsel))
    wsel = np.sort(rng.choice(len(Xsel), size=cap, replace=False))
    C = compute_c_j(z_sel[:, pool], cal_sel["score"], active_only=True)
    dsingle = compute_delta_pred_single(model, sae, Xsel[wsel], Ysel[wsel], sel_act[wsel],
                                        tvar_abs, tap, pool, step_profile=prof, device=dev, loss="score")
    dose = sweep_pool(model, sae, Xsel[wsel], Ysel[wsel], sel_act[wsel], tvar_abs, tap, pool,
                      prof, lambdas=CFG["lambdas"], device=dev, verbose=True)

    ftab = feature_type_table(dose["P_j"], dose["U_j"], feature_indices=pool)
    R = residualize(dose["P_j"], dose["U_j"], method=CFG["residualize_method"],
                    provenance_p="interventional_dose_response",
                    provenance_u="interventional_dose_response")
    k = CFG["feature_budget_k"]
    mvdf = select_mvdf_v2(dose, u_threshold=CFG["u_threshold"], k=k,
                          residualize_method=CFG["residualize_method"])
    mvdf_idx = np.asarray(mvdf["selected"], dtype=int)
    corr_pool = np.abs(scr["pearson"][pool])
    correlated = pool[np.argsort(-corr_pool)[:k]]
    enr_v2 = enrichment(mvdf_idx, ftab, pool)
    enr_corr = enrichment(correlated, ftab, pool)

    json.dump(dict(MVDF_v2=mvdf_idx.tolist(), correlated=correlated.tolist(),
                   n_gate1=int(mvdf["n_gate1"]), enrichment_v2=enr_v2,
                   enrichment_corr=enr_corr, feature_type_counts=ftab["counts"],
                   data_fingerprint=sd.fingerprint(), forecaster_val_mse=float(val_mse),
                   sae_fvu=fvu),
              open(os.path.join(rd, "mvdf_selection.json"), "w"), indent=2, default=str)

    # ---- the battery on E_eval_ID -------------------------------------------------
    Xid, Yid = Xte[id_mask], Yte[id_mask]
    ce_id = collect(model, Xid, Yid, tvar_abs, device=dev)
    act_id = extract_all_taps(model, Xid, tvar_abs, device=dev)[tap]
    z_id = encode_all(sae, act_id, device=dev)
    s_id = ce_id["score"].astype(np.float64)
    npd = len(id_mask); cpt = int(0.6 * npd)

    feat_all = z_id[:, mvdf_idx].astype(np.float64)
    pred_marg = _fit_eval_estimators(feat_all[:cpt], s_id[:cpt], feat_all[cpt:], s_id[cpt:], ds_seed)

    active_any = (z_id[:, mvdf_idx] != 0).any(axis=1)
    tr_a, te_a = active_any[:cpt], active_any[cpt:]
    pred_cond = None
    if tr_a.sum() >= 40 and te_a.sum() >= 40:
        pred_cond = _fit_eval_estimators(feat_all[:cpt][tr_a], s_id[:cpt][tr_a],
                                         feat_all[cpt:][te_a], s_id[cpt:][te_a], ds_seed)
    predblock = dict(
        marginal=pred_marg, marginal_best_r2=_best_r2(pred_marg),
        activation_conditioned=pred_cond,
        activation_conditioned_best_r2=(_best_r2(pred_cond) if pred_cond else None),
        n_eval=int(npd - cpt), n_active_windows_eval=int(te_a.sum()),
        mvdf_activation_rate=float(active_any.mean()),
        mvdf_per_feature_activation_rate=float((z_id[:, mvdf_idx] != 0).mean()),
    )
    json.dump(predblock, open(os.path.join(rd, "predictability.json"), "w"), indent=2, default=str)

    # ---- R_j vs per-feature natural predictive utility (headline figure data) ----
    rows = []
    for i, j in enumerate(pool):
        col_tr = z_id[:cpt, int(j)].astype(np.float64).reshape(-1, 1)
        col_te = z_id[cpt:, int(j)].astype(np.float64).reshape(-1, 1)
        if np.std(col_tr) < 1e-12:
            best_r2_j = float("nan")
        else:
            b = _fit_eval_estimators(col_tr, s_id[:cpt], col_te, s_id[cpt:], ds_seed)
            best_r2_j = _best_r2(b)
        rows.append(dict(feature=int(j), R_j=float(R[i]), P_j=float(dose["P_j"][i]),
                         U_j=float(dose["U_j"][i]), C_j=float(C[i]),
                         abs_corr_aj_St=float(np.abs(scr["pearson"][int(j)])),
                         n_active=int(dose["n_active"][i]),
                         best_natural_predictive_r2=best_r2_j,
                         in_mvdf=bool(int(j) in set(mvdf_idx.tolist())),
                         type=_type_of(int(j), ftab)))
    dfp = pd.DataFrame(rows).sort_values("R_j", ascending=False)
    dfp.to_csv(os.path.join(rd, "rj_vs_predictive.csv"), index=False)
    dfp.to_csv(os.path.join(rd, "candidates.csv"), index=False)
    # rank corr between R_j and best natural predictive R^2 across the pool
    from scipy import stats
    ok = dfp["R_j"].notna() & dfp["best_natural_predictive_r2"].notna()
    rj_pred_rho = (float(stats.spearmanr(dfp["R_j"][ok], dfp["best_natural_predictive_r2"][ok]).statistic)
                   if ok.sum() > 5 else float("nan"))

    bm = predblock["marginal_best_r2"]
    bc = predblock["activation_conditioned_best_r2"]
    print(f"[{ds} s{seed}] marginal best R2 {bm:+.4f}  cond best R2 "
          f"{bc if bc is None else round(bc,4)}  R_j-vs-predR2 rho {rj_pred_rho:+.3f}")
    return dict(
        dataset=ds, seed=seed, data_fingerprint=sd.fingerprint(),
        forecaster_val_mse=float(val_mse), sae_fvu=fvu, n_pool=int(n_pool),
        enrichment_v2=enr_v2.get("enrichment_ratio"),
        enrichment_corr=enr_corr.get("enrichment_ratio"),
        pred_marginal_best_r2=bm, pred_cond_best_r2=bc,
        pred_marginal_linear_r2=pred_marg["linear"]["r2"],
        pred_marginal_gbm_r2=pred_marg["gbm"]["r2"],
        pred_marginal_mlp_r2=pred_marg["mlp"]["r2"],
        pred_marginal_isotonic_r2=pred_marg["isotonic"]["r2"],
        pred_cond_linear_r2=(pred_cond["linear"]["r2"] if pred_cond else None),
        pred_cond_gbm_r2=(pred_cond["gbm"]["r2"] if pred_cond else None),
        mvdf_activation_rate=predblock["mvdf_activation_rate"],
        mvdf_per_feature_activation_rate=predblock["mvdf_per_feature_activation_rate"],
        n_active_windows_eval=predblock["n_active_windows_eval"], n_eval=predblock["n_eval"],
        rj_vs_predictive_rho=rj_pred_rho,
        pool_best_natural_predictive_r2=float(np.nanmax(dfp["best_natural_predictive_r2"])),
    )


def _type_of(fid, ftab):
    for t, ids in ftab["cells"].items():
        if fid in ids:
            return t
    return "?"


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
                    cells.append(f"{v:.4f}" if np.isfinite(v) else "")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"

    L = ["# PHASE3_RESULT — real-data natural-activation predictability battery\n",
         "Generated by `experiments/phase3_real_predictability.py`. ETTh1 + ETTh2, "
         "HTTP ETDataset mirror (8640/2880/5900 split; same split sizes as the Kaggle "
         "bundle used for the paper's H1c, different edge fingerprint — Phase 1 retrained "
         "locally for self-consistent MVDF selections). electricity/traffic raw data "
         "Kaggle-only, not available locally (same scope limit as Result 4).\n\n"]
    L.append("## What this tests (causal-ladder framing)\n\n")
    L.append("An **L1 (associational)** test: do the *natural, non-intervened* activations "
             "of the L2-validated MVDF features predict `S_t` out-of-sample? A feature "
             "found via L2 (dose-response) evidence does not automatically carry L1 "
             "predictive information — Elazar et al. 2021 (amnesic probing) is the direct "
             "precedent (decodability ≠ the model's computation using it). Four held-out "
             "estimators, so 'the MLP wasn't powerful enough' is closed.\n\n")

    L.append("## Marginal battery (all E_eval_ID windows)\n\n")
    m = agg[["dataset", "seed", "pred_marginal_linear_r2", "pred_marginal_isotonic_r2",
             "pred_marginal_gbm_r2", "pred_marginal_mlp_r2", "pred_marginal_best_r2"]].copy()
    L.append(md(m))
    L.append("\n## Activation-conditioned battery (windows where ≥1 MVDF feature is active)\n\n")
    ccond = agg[["dataset", "seed", "mvdf_activation_rate", "mvdf_per_feature_activation_rate",
                 "n_active_windows_eval", "pred_cond_linear_r2", "pred_cond_gbm_r2",
                 "pred_cond_best_r2"]].copy()
    L.append(md(ccond))
    L.append("\n> **'no signal because rarely active' vs 'no signal despite being active'** — "
             "if conditioned R² ≫ marginal R², the dissociation is *causal sparsity "
             "suppressing predictive observability*; if conditioned R² stays weak, it's the "
             "stronger, more direct dissociation (doesn't predict difficulty even when firing).\n\n")

    L.append("## `R_j` vs natural predictive utility (headline figure)\n\n")
    rj = agg[["dataset", "seed", "rj_vs_predictive_rho", "pool_best_natural_predictive_r2",
              "enrichment_v2", "enrichment_corr"]].copy()
    L.append(md(rj))
    L.append("\nPer-feature `R_j` vs best-achieved natural predictive R² in "
             "`<ds>_seed<s>/rj_vs_predictive.csv`. A weak/near-zero `rj_vs_predictive_rho` "
             "is direct feature-level evidence for the causal→predictive dissociation "
             "(stronger than inferring it from Result 3a's downstream conformal failure).\n\n")

    # verdict
    mb = agg["pred_marginal_best_r2"]
    cb = agg["pred_cond_best_r2"].dropna()
    L.append("## Reading\n\n")
    strong_marg = bool((mb > 0.05).any())
    strong_cond = bool((cb > 0.05).any()) if len(cb) else False
    if not strong_marg and not strong_cond:
        L.append("**No usable natural predictive signal on ETTh1/ETTh2 — marginal or "
                 "activation-conditioned — across four estimator classes.** The "
                 "'your predictor wasn't powerful enough' objection is closed: linear, "
                 "isotonic, GBM and a pinball MLP all fail. Combined with the synthetic "
                 "Phase 1 (where the same battery recovered R² 0.26–0.48 on a *designed-in* "
                 "signal), this is direct evidence that the real ETT representation's "
                 "L2-validated difficulty features carry little to no L1 natural predictive "
                 "information — the rung-mismatch the paper diagnoses, measured directly at "
                 "the feature level, not inferred from downstream conformal failure.\n")
    elif strong_cond and not strong_marg:
        L.append("**Signal appears only when conditioned on activation.** The dissociation "
                 "here is best described as *causal sparsity suppressing predictive "
                 "observability*: the MVDF features do carry difficulty information on the "
                 "rare windows they fire, but it is diluted to near-invisibility marginally. "
                 "Report this as the specific mechanism, not a flat 'no signal'.\n")
    else:
        L.append("**Some natural predictive signal is present** (best R² above 0.05 on ≥1 "
                 "dataset). Phase 4's oracle ceiling then becomes a question of *how much "
                 "more* was left unexploited, not whether any existed. The paper's central "
                 "claim must be stated as a partial / dataset-specific rung-mismatch, not an "
                 "absolute one — see the per-dataset numbers above.\n")
    L.append("\n> L1/L2 language discipline (Joshi et al. 2026): these results are "
             "associational. Use 'predicts' / 'correlates with' / 'is decodable from', not "
             "'encodes' / 'represents' / 'the mechanism for' — those are reserved for the "
             "L2 dose-response results (`P_j`/`U_j`/`R_j`).\n")

    with open(os.path.join(out_root, "PHASE3_RESULT.md"), "w") as f:
        f.write("".join(L))
    print(f"wrote {os.path.join(out_root, 'PHASE3_RESULT.md')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "v2", "phase3_real"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    rows = []
    for ds in args.datasets:
        for s in args.seeds:
            print("=" * 78, f"\n {ds}  seed {s}\n", "=" * 78)
            rows.append(run_one(ds, s, args.out, args.fast))
    agg = pd.DataFrame(rows)
    agg.to_csv(os.path.join(args.out, "AGG_phase3.csv"), index=False)
    write_md(agg, args.out)
    print(f"\nPhase 3 real done in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
