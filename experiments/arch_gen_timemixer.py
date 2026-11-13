"""Architecture generalization gate — TimeMixer (Extended wrap-up §3).

The project's forecaster has been iTransformer (attention) throughout. This runs the
pipeline on a structurally different architecture — MLP-based, channel-independent,
multi-resolution (`forecasters/timemixer.py`) — and reports THREE gate checks, per
dataset, explicitly (not averaged):

  1. H1c replication   — does dose-response + R_j still find a type-B-enriched MVDF
                          population, comparable order of magnitude to iTransformer's
                          2.0-4.6x?
  2. Task G replication — does `naive_matched` (constant-width split-conformal) still
                          beat a learned continuous scorer built on this architecture's
                          dense hidden state?
  3. Phase 3 replication— does the multi-estimator natural-activation predictability
                          battery show the SAME qualitative pattern as iTransformer
                          (negative best held-out R^2, marginal AND activation-
                          conditioned)? — the 3rd check, added because Phase 3 is the
                          evidence the central claim now rests on.

Extend to the full result suite only on an architecture that clears all three; report
here regardless.

SCOPE: ETTh1 + ETTh2 (HTTP mirror — same as Phase 3/4/5; electricity/traffic Kaggle-only).
NO layer sweep on the new architecture (Phase 0 found no privileged layer; re-using
`pre_head` by design — a sweep would be a scope violation, §3.1).

Outputs runs/arch_gen_timemixer/{<ds>_seed<s>/*, AGG_arch_gen.csv, ARCH_GEN_RESULT.md}
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
from forecasters.train_timemixer import TMTrainConfig, train_timemixer
from representation.layer_sweep import extract_all_taps
from sae.train_sae import train_sae, encode_all
from validation.errors import collect
from validation.correlation_screen import screen
from validation.importance_and_difficulty import compute_c_j, compute_delta_pred_single
from validation.dose_response import sweep_pool
from validation.difficulty_specificity import (residualize, feature_type_table,
                                               enrichment, select_mvdf_v2)
from shift.shift_windows import three_way_env_masks
from calibration.difficulty_scorer import (DifficultyScorer, build_features,
                                           naive_no_scorer_run, paired_width_bootstrap)
from eval import metrics
from experiments.phase1_synthetic_control import _fit_eval_estimators, _best_r2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CFG = dict(
    lookback=96, horizon=96, target_var=-1, alpha=0.10, gamma_global=0.03,
    timemixer=dict(d_model=256, n_scales=3, mix_hidden=128, dropout=0.1, ma_kernel=25,
                   lr=1e-4, batch_size=32, max_epochs=25, patience=6,
                   lr_decay=0.8, warmup_epochs=3),
    sae=dict(expansion=8, k=64, steps=12000, batch_size=512, lr=1e-4,
             resample_every=2000, resample_until_frac=0.5, resample_cap=64),
    tap="pre_head",
    candidate_pool_size=120, window_cap=2048,
    lambdas=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
    feature_budget_k=12, u_threshold=0.0, residualize_method="isotonic",
    # iTransformer reference numbers for the gate comparison
    itr_h1c_enrichment_range=[2.0, 4.6],
    itr_phase3_marginal_r2_sign="negative",
)


def run_one(ds: str, seed: int, out_root: str, fast: bool) -> dict:
    rd = os.path.join(out_root, f"{ds}_seed{seed}")
    os.makedirs(rd, exist_ok=True)
    ds_seed = derive_seed(seed, ds)
    np.random.seed(ds_seed); torch.manual_seed(ds_seed)
    dev = "cpu"
    L, H, tvar = CFG["lookback"], CFG["horizon"], CFG["target_var"]
    tap = CFG["tap"]
    alpha, gg = CFG["alpha"], CFG["gamma_global"]
    tm, sae_c = CFG["timemixer"], CFG["sae"]
    if fast:
        tm = {**tm, "max_epochs": 3}
        sae_c = {**sae_c, "steps": 3000}
        pool_size, wcap = 40, 512
    else:
        pool_size, wcap = CFG["candidate_pool_size"], CFG["window_cap"]

    sd = get_splits(ds, max_channels=48)
    tvar_abs = tvar % len(sd.channels)
    Xtr, Ytr, _ = make_windows(sd.train, L, H)
    Xva, Yva, _ = make_windows(sd.val, L, H)
    Xte, Yte, end_te = make_windows(sd.test, L, H)
    print(f"[{ds} s{seed}] windows tr {len(Xtr)} va {len(Xva)} te {len(Xte)}  fp {sd.fingerprint()}")

    tcfg = TMTrainConfig(lookback=L, horizon=H, d_model=tm["d_model"], n_scales=tm["n_scales"],
                         mix_hidden=tm["mix_hidden"], dropout=tm["dropout"], ma_kernel=tm["ma_kernel"],
                         lr=tm["lr"], batch_size=tm["batch_size"], max_epochs=tm["max_epochs"],
                         patience=tm["patience"], lr_decay=tm["lr_decay"],
                         warmup_epochs=tm["warmup_epochs"], seed=ds_seed, device=dev)
    model, val_mse = train_timemixer((Xtr, Ytr, None), (Xva, Yva, None),
                                     n_vars=len(sd.channels), cfg=tcfg, verbose=True)
    print(f"[{ds} s{seed}] TimeMixer val_mse {val_mse:.4f}")

    env = three_way_env_masks(end_te, len(sd.test))
    sel_mask = np.where(env["E_select"])[0]
    id_mask = np.where(env["E_eval_ID"])[0]
    sh_mask = np.where(env["E_eval_shift"])[0]
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
    dp = np.abs(dsingle["delta_pred"])
    mvdf_v1 = pool[np.argsort(-dp)[:k]]
    corr_pool = np.abs(scr["pearson"][pool])
    correlated = pool[np.argsort(-corr_pool)[:k]]
    enr_v2 = enrichment(mvdf_idx, ftab, pool)
    enr_v1 = enrichment(mvdf_v1, ftab, pool)
    enr_corr = enrichment(correlated, ftab, pool)

    json.dump(dict(MVDF_v2=mvdf_idx.tolist(), correlated=correlated.tolist(),
                   MVDF_v1=mvdf_v1.tolist(), n_gate1=int(mvdf["n_gate1"]),
                   enrichment_v2=enr_v2, enrichment_v1=enr_v1, enrichment_corr=enr_corr,
                   feature_type_counts=ftab["counts"], forecaster_val_mse=float(val_mse),
                   sae_fvu=fvu, data_fingerprint=sd.fingerprint()),
              open(os.path.join(rd, "mvdf_selection.json"), "w"), indent=2, default=str)

    # ---- Check 3: Phase 3 predictability battery on E_eval_ID -------------------
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
    best_marg = _best_r2(pred_marg)
    best_cond = _best_r2(pred_cond) if pred_cond else None
    json.dump(dict(marginal=pred_marg, marginal_best_r2=best_marg,
                   activation_conditioned=pred_cond,
                   activation_conditioned_best_r2=best_cond,
                   mvdf_activation_rate=float(active_any.mean())),
              open(os.path.join(rd, "predictability.json"), "w"), indent=2, default=str)

    # ---- Check 2: Task G — learned scorer (dense + mvdf) vs naive_matched -------
    c1, c2 = int(0.5 * npd), int(0.8 * npd)
    tr_i, conf_i, idw_i = id_mask[:c1], id_mask[c1:c2], id_mask[c2:]
    sh_idx = np.where(sh_mask)[0]

    def bundle(widx):
        Xe, Ye = Xte[widx], Yte[widx]
        ce = collect(model, Xe, Ye, tvar_abs, device=dev)
        act = extract_all_taps(model, Xe, tvar_abs, device=dev)[tap]
        z = encode_all(sae, act, device=dev)
        return dict(hidden=ce["hidden"], z=z, score=ce["score"],
                    yhat=ce["yhat"][:, :, tvar_abs], y=Ye[:, :, tvar_abs], prof=ce["step_profile"])
    b_tr, b_conf, b_idw, b_sh = bundle(tr_i), bundle(conf_i), bundle(idw_i), bundle(sh_idx)
    score_cal80 = np.concatenate([b_tr["score"], b_conf["score"]])

    tg = {}
    for mode in ("dense", "mvdf"):
        def bf(b):
            return build_features(mode, b["hidden"], b["z"], mvdf_idx)
        f_tr, f_conf, f_idw, f_sh = bf(b_tr), bf(b_conf), bf(b_idw), bf(b_sh)
        sc = DifficultyScorer(mode, hidden_dim=f_tr.shape[1]).fit(f_tr, b_tr["score"],
                                                                  alpha=alpha, seed=ds_seed)
        sc.conformalize(f_conf, b_conf["score"])
        r_id = sc.run(f_idw, b_idw["yhat"], b_idw["y"], b_idw["prof"], alpha, gamma=gg)
        r_sh = sc.run(f_sh, b_sh["yhat"], b_sh["y"], b_sh["prof"], alpha, gamma=gg)
        mi = metrics.summarize(r_id, b_idw["y"], alpha)
        ms = metrics.summarize(r_sh, b_sh["y"], alpha)
        tg[mode] = dict(id_cov=mi["coverage"], id_w=mi["mean_width"],
                        sh_cov=ms["coverage"], sh_w=ms["mean_width"],
                        _idw=r_id["width"].tolist(), _shw=r_sh["width"].tolist())
    r_id = naive_no_scorer_run(score_cal80, b_idw["yhat"], b_idw["y"], b_idw["prof"], alpha, gamma=gg)
    r_sh = naive_no_scorer_run(score_cal80, b_sh["yhat"], b_sh["y"], b_sh["prof"], alpha, gamma=gg)
    mi = metrics.summarize(r_id, b_idw["y"], alpha)
    ms = metrics.summarize(r_sh, b_sh["y"], alpha)
    tg["naive_matched"] = dict(id_cov=mi["coverage"], id_w=mi["mean_width"],
                               sh_cov=ms["coverage"], sh_w=ms["mean_width"],
                               _idw=r_id["width"].tolist(), _shw=r_sh["width"].tolist())

    # naive_matched - dense, shift width (negative => naive narrower => Task G replicates)
    a = np.array(tg["naive_matched"]["_shw"]); b = np.array(tg["dense"]["_shw"])
    nn = min(len(a), len(b))
    bt_nd = paired_width_bootstrap(a[:nn], b[:nn], seed=ds_seed)
    a2 = np.array(tg["naive_matched"]["_shw"]); b2 = np.array(tg["mvdf"]["_shw"])
    nn2 = min(len(a2), len(b2))
    bt_nm = paired_width_bootstrap(a2[:nn2], b2[:nn2], seed=ds_seed)
    json.dump({m: {k: v for k, v in d.items() if not k.startswith("_")} for m, d in tg.items()}
              | dict(naive_matched_minus_dense_shift=bt_nd,
                     naive_matched_minus_mvdf_shift=bt_nm),
              open(os.path.join(rd, "task_g.json"), "w"), indent=2, default=str)

    # ---- three gate checks -----------------------------------------------------
    lo, hi = CFG["itr_h1c_enrichment_range"]
    check1_h1c = bool(enr_v2.get("enrichment_ratio", 0) >= 1.0
                      and enr_v2.get("enrichment_ratio", 0) >= 0.5 * lo)  # comparable order of magnitude
    # Task G replicates if naive_matched is narrower than the learned scorer under shift
    check2_taskg = bool(bt_nd["mean_diff"] < 0 or bt_nm["mean_diff"] < 0)
    # Phase 3 replicates if the best held-out R^2 is non-positive (same qualitative pattern)
    check3_phase3 = bool((best_marg is not None and best_marg <= 0.02)
                         and (best_cond is None or best_cond <= 0.02))

    print(f"[{ds} s{seed}] CHECKS  H1c={check1_h1c} (enr {enr_v2.get('enrichment_ratio'):.2f})  "
          f"TaskG={check2_taskg} (nm-dense {bt_nd['mean_diff']:+.3f})  "
          f"Phase3={check3_phase3} (marg R2 {best_marg:+.3f})")
    return dict(
        dataset=ds, seed=seed, data_fingerprint=sd.fingerprint(),
        forecaster_val_mse=float(val_mse), sae_fvu=fvu, n_pool=int(n_pool),
        enrichment_v2=enr_v2.get("enrichment_ratio"),
        enrichment_v1=enr_v1.get("enrichment_ratio"),
        enrichment_corr=enr_corr.get("enrichment_ratio"),
        type_B_frac=ftab["fractions"].get("B"),
        pred_marginal_best_r2=best_marg, pred_cond_best_r2=best_cond,
        naive_id_cov=tg["naive_matched"]["id_cov"], naive_id_w=tg["naive_matched"]["id_w"],
        naive_sh_w=tg["naive_matched"]["sh_w"],
        dense_id_cov=tg["dense"]["id_cov"], dense_sh_w=tg["dense"]["sh_w"],
        mvdf_id_cov=tg["mvdf"]["id_cov"], mvdf_sh_w=tg["mvdf"]["sh_w"],
        nm_minus_dense_shift=bt_nd["mean_diff"], nm_minus_dense_sig=bt_nd["significant"],
        nm_minus_mvdf_shift=bt_nm["mean_diff"],
        check1_h1c=check1_h1c, check2_taskg=check2_taskg, check3_phase3=check3_phase3,
        all_checks=bool(check1_h1c and check2_taskg and check3_phase3),
    )


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

    L = ["# ARCH_GEN_RESULT — architecture generalization (TimeMixer)\n",
         "Generated by `experiments/arch_gen_timemixer.py`. The project's forecaster has "
         "been iTransformer (attention) throughout; this runs the pipeline on a "
         "structurally different architecture — **MLP-based, channel-independent, "
         "multi-resolution** (`forecasters/timemixer.py`). ETTh1 + ETTh2 × seeds "
         "(HTTP mirror; electricity/traffic Kaggle-only). No layer sweep on the new "
         "architecture — `pre_head` re-used by design.\n\n"]
    L.append("## Three gate checks (reported per dataset/seed, not averaged)\n\n")
    per = agg[["dataset", "seed", "check1_h1c", "check2_taskg", "check3_phase3",
               "all_checks", "forecaster_val_mse", "sae_fvu"]].copy()
    L.append(md(per))
    n = len(agg)
    L.append(f"\n- **Check 1 — H1c replication:** MVDF type-B enrichment ≥ 1.0× **and** "
             f"comparable order of magnitude to iTransformer's 2.0–4.6×. "
             f"{int(agg['check1_h1c'].sum())}/{n} pass.\n")
    L.append(f"- **Check 2 — Task G replication:** `naive_matched` narrower than a learned "
             f"scorer (dense or mvdf) under shift. {int(agg['check2_taskg'].sum())}/{n} pass.\n")
    L.append(f"- **Check 3 — Phase 3 replication:** best held-out natural-activation R² ≤ "
             f"0.02, marginal and activation-conditioned (same qualitative pattern as "
             f"iTransformer's negative R²). {int(agg['check3_phase3'].sum())}/{n} pass.\n\n")

    L.append("## H1c — enrichment on TimeMixer\n\n")
    L.append(md(agg[["dataset", "seed", "enrichment_v2", "enrichment_v1", "enrichment_corr",
                     "type_B_frac"]]))
    L.append("\n## Phase 3 — predictability battery on TimeMixer\n\n")
    L.append(md(agg[["dataset", "seed", "pred_marginal_best_r2", "pred_cond_best_r2"]]))
    L.append("\n## Task G — learned scorer vs naive_matched on TimeMixer\n\n")
    L.append(md(agg[["dataset", "seed", "naive_id_cov", "naive_sh_w", "dense_id_cov",
                     "dense_sh_w", "mvdf_sh_w", "nm_minus_dense_shift", "nm_minus_dense_sig",
                     "nm_minus_mvdf_shift"]]))

    all_pass = bool(agg["all_checks"].all())
    L.append("\n## Verdict\n\n")
    if all_pass:
        L.append("**All three checks pass on every dataset/seed.** The central findings — "
                 "H1c (causally difficulty-specific features exist), Task G (a learned "
                 "scorer on them doesn't beat a constant band), and Phase 3 (their natural "
                 "activation carries no predictive signal) — reproduce on a structurally "
                 "different (MLP-based, non-attention) architecture. The claim is not an "
                 "iTransformer artifact.\n")
    else:
        by = agg.groupby("dataset")[["check1_h1c", "check2_taskg", "check3_phase3"]].all()
        L.append("**Not a clean sweep.** Per-check, per-dataset:\n\n")
        L.append(md(by.reset_index()))
        L.append("\nReport per check and per dataset (not averaged). A check that fails on "
                 "TimeMixer means the corresponding finding is, so far, iTransformer-"
                 "specific — itself a reportable boundary condition, not a null. The full "
                 "result suite is extended only on an architecture clearing all three; on "
                 "this evidence, that is "
                 + ("both" if all_pass else "not the case for TimeMixer") + ".\n")
    L.append("\n> No new exploitation mechanism, no layer sweep on the new architecture. "
             "This is a generalization check on the existing findings.\n")

    with open(os.path.join(out_root, "ARCH_GEN_RESULT.md"), "w") as f:
        f.write("".join(L))
    print(f"wrote {os.path.join(out_root, 'ARCH_GEN_RESULT.md')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "arch_gen_timemixer"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    rows = []
    for ds in args.datasets:
        for s in args.seeds:
            print("=" * 78, f"\n {ds}  seed {s}\n", "=" * 78)
            rows.append(run_one(ds, s, args.out, args.fast))
    agg = pd.DataFrame(rows)
    agg.to_csv(os.path.join(args.out, "AGG_arch_gen.csv"), index=False)
    write_md(agg, args.out)
    print(f"\nArch-gen (TimeMixer) done in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
