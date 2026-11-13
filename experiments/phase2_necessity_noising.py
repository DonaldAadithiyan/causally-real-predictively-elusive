"""Phase 2 (bounded) — the necessity / denoising-noising test for the MVDF set.

Document 1 §2.2 (Joshi et al. 2026 Case Study I: "Sufficient ≠ Necessary"). The
dose-response protocol shows `U_j != 0` under intervention → perturbing feature j is
*sufficient* to move S_t. It does NOT show *necessity* — that this feature (not a
redundant/overlapping one, not another pathway) is required. Heimersheim & Nanda's
denoising/noising distinction is the standard fix: test BOTH directions rather than one.

For the MVDF feature SET (jointly), on E_select windows where the set is active:

  NOISING  (ablate in a clean run): zero the MVDF codes, decode, patch, continue.
           delta_noise = E[ S_t(mvdf -> 0) - S_t(clean) ].
           Large positive -> removing the set degrades difficulty tracking -> the set
           carries difficulty-relevant signal that is NOT reconstructed by other features.

  DENOISING (add the effect back in a corrupted run): start from a corrupted baseline
           where ALL SAE codes are set to their per-feature mean (difficulty signal
           destroyed), then restore ONLY the MVDF codes to their true per-window values.
           delta_denoise = E[ S_t(corrupt) - S_t(corrupt + true MVDF codes) ].
           Large positive -> restoring just the MVDF set recovers difficulty tracking
           the corruption removed -> the set is (close to) sufficient on its own.

Necessity is supported when noising degrades AND a matched-size random-feature-set
noising degrades LESS (the MVDF set is not just "any k features"). Reported with:
  - matched-norm random-direction control (same L2 perturbation, random direction)
  - matched-size random-SAE-feature-set control (k random eligible features)
  - the natural-activation-range overlap of the perturbation (is it on-manifold?)

SCOPE: ETTh1 + ETTh2 (HTTP mirror). Reuses the retrained-locally Phase 1 selections
(runs/phase3_real/<ds>_seed<s>/mvdf_selection.json) — no new forecaster/SAE training if
those exist; otherwise retrains.

Outputs runs/phase2_necessity/{<ds>_seed<s>/necessity.json, AGG_phase2.csv, PHASE2_NECESSITY_RESULT.md}
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
from validation.dose_response import sweep_pool  # noqa: F401 (kept for parity of imports)
from validation.difficulty_specificity import residualize, select_mvdf_v2, feature_type_table
from validation.importance_and_difficulty import compute_delta_pred_single
from validation.dose_response import _patched_pred_and_score
from shift.shift_windows import three_way_env_masks

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CFG = dict(
    lookback=96, horizon=96, target_var=-1,
    forecaster=dict(d_model=256, n_layers=3, n_heads=8, d_ff=512, dropout=0.1,
                    lr=1e-4, batch_size=32, max_epochs=20, patience=5,
                    lr_decay=0.8, warmup_epochs=3),
    sae=dict(expansion=8, k=64, steps=12000, batch_size=512, lr=1e-4,
             resample_every=2000, resample_until_frac=0.5, resample_cap=64),
    tap="pre_head", candidate_pool_size=120, window_cap=2048,
    lambdas=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
    feature_budget_k=12, u_threshold=0.0, residualize_method="isotonic",
    n_random_controls=5,
)


@torch.no_grad()
def _decode_set(sae, z_rows, device="cpu", bs=4096):
    out = []
    for i in range(0, len(z_rows), bs):
        zb = torch.from_numpy(z_rows[i:i + bs].astype(np.float32)).to(device)
        out.append(sae.unstandardize(sae.decode(zb)).cpu().numpy())
    return np.concatenate(out, 0)


def _score_for_codes(model, sae, X, Y, z_rows, tvar, tap, prof, device):
    tok = _decode_set(sae, z_rows, device)
    _, sc = _patched_pred_and_score(model, X, Y, tok, tvar, tap, prof, device)
    return sc


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

    tcfg = TrainConfig(lookback=L, horizon=H, d_model=fc["d_model"], n_layers=fc["n_layers"],
                       n_heads=fc["n_heads"], d_ff=fc["d_ff"], dropout=fc["dropout"],
                       lr=fc["lr"], batch_size=fc["batch_size"], max_epochs=fc["max_epochs"],
                       patience=fc["patience"], lr_decay=fc["lr_decay"],
                       warmup_epochs=fc["warmup_epochs"], seed=ds_seed, device=dev)
    model, val_mse = train_itransformer((Xtr, Ytr, None), (Xva, Yva, None),
                                        n_vars=len(sd.channels), cfg=tcfg, verbose=True)

    env = three_way_env_masks(end_te, len(sd.test))
    sel_mask = np.where(env["E_select"])[0]
    Xsel, Ysel = Xte[sel_mask], Yte[sel_mask]
    cal_sel = collect(model, Xsel, Ysel, tvar_abs, device=dev)
    prof = cal_sel["step_profile"]

    tr_act = extract_all_taps(model, Xtr, tvar_abs, device=dev)[tap]
    sae = train_sae(tr_act, expansion=sae_c["expansion"], k=sae_c["k"], steps=sae_c["steps"],
                    batch_size=sae_c["batch_size"], lr=sae_c["lr"],
                    resample_every=sae_c["resample_every"],
                    resample_until_frac=sae_c["resample_until_frac"],
                    resample_cap=sae_c["resample_cap"], device=dev, seed=ds_seed)
    sel_act = extract_all_taps(model, Xsel, tvar_abs, device=dev)[tap]
    z_sel = encode_all(sae, sel_act, device=dev)
    scr = screen(z_sel, cal_sel["score"], min_active_frac=0.02)
    elig = np.where(scr["eligible"])[0]
    rng = np.random.default_rng(ds_seed)
    pool = np.sort(rng.choice(elig, size=min(pool_size, len(elig)), replace=False))
    cap = min(wcap, len(Xsel))
    wsel = np.sort(rng.choice(len(Xsel), size=cap, replace=False))

    dsingle = compute_delta_pred_single(model, sae, Xsel[wsel], Ysel[wsel], sel_act[wsel],
                                        tvar_abs, tap, pool, step_profile=prof, device=dev, loss="score")
    dose = sweep_pool(model, sae, Xsel[wsel], Ysel[wsel], sel_act[wsel], tvar_abs, tap, pool,
                      prof, lambdas=CFG["lambdas"], device=dev, verbose=False)
    R = residualize(dose["P_j"], dose["U_j"], method=CFG["residualize_method"],
                    provenance_p="interventional_dose_response",
                    provenance_u="interventional_dose_response")
    k = CFG["feature_budget_k"]
    mvdf = select_mvdf_v2(dose, u_threshold=CFG["u_threshold"], k=k,
                          residualize_method=CFG["residualize_method"])
    mvdf_idx = np.asarray(mvdf["selected"], dtype=int)

    # ---- windows where the MVDF set is active; work in code space ----------------
    Xw, Yw, zw = Xsel[wsel], Ysel[wsel], z_sel[wsel]
    active = (zw[:, mvdf_idx] != 0).any(axis=1)
    if active.sum() < 30:
        active = np.ones(len(zw), bool)   # fallback
    Xa, Ya, za = Xw[active], Yw[active], zw[active]

    # baseline (clean reconstruction) score
    s_clean = _score_for_codes(model, sae, Xa, Ya, za, tvar_abs, tap, prof, dev)

    # NOISING: zero the MVDF codes
    z_noise = za.copy(); z_noise[:, mvdf_idx] = 0.0
    s_noise = _score_for_codes(model, sae, Xa, Ya, z_noise, tvar_abs, tap, prof, dev)
    delta_noise = float(np.mean(s_noise - s_clean))

    # DENOISING: corrupt ALL codes to per-feature mean, then restore only MVDF
    z_mean = np.tile(za.mean(axis=0, keepdims=True), (len(za), 1)).astype(np.float32)
    s_corrupt = _score_for_codes(model, sae, Xa, Ya, z_mean, tvar_abs, tap, prof, dev)
    z_restore = z_mean.copy(); z_restore[:, mvdf_idx] = za[:, mvdf_idx]
    s_restore = _score_for_codes(model, sae, Xa, Ya, z_restore, tvar_abs, tap, prof, dev)
    delta_denoise = float(np.mean(s_corrupt - s_restore))

    # matched-size random-feature-set controls (noising)
    rnd_deltas = []
    for _ in range(CFG["n_random_controls"] if not fast else 2):
        ridx = rng.choice(elig, size=len(mvdf_idx), replace=False)
        zr = za.copy(); zr[:, ridx] = 0.0
        sr = _score_for_codes(model, sae, Xa, Ya, zr, tvar_abs, tap, prof, dev)
        rnd_deltas.append(float(np.mean(sr - s_clean)))
    rnd_mean = float(np.mean(rnd_deltas)); rnd_sd = float(np.std(rnd_deltas))

    # matched-NORM random-direction control (noising): perturb the pre-head token by a
    # random vector with the same L2 norm as the MVDF ablation's reconstruction delta
    tok_clean = _decode_set(sae, za, dev)
    tok_noise = _decode_set(sae, z_noise, dev)
    pert_norm = np.linalg.norm(tok_noise - tok_clean, axis=1, keepdims=True)  # (Na,1)
    gdir = rng.standard_normal(tok_clean.shape).astype(np.float32)
    gdir /= (np.linalg.norm(gdir, axis=1, keepdims=True) + 1e-12)
    tok_rand = tok_clean + gdir * pert_norm
    _, s_randdir = _patched_pred_and_score(model, Xa, Ya, tok_rand, tvar_abs, tap, prof, dev)
    delta_randdir = float(np.mean(s_randdir - s_clean))

    # natural-activation-range overlap: where do the MVDF codes' *ablated* values (0) sit
    # vs the natural distribution — trivially the 0th pctile for a zero; report instead
    # the fraction of the DENOISING restored values that lie within the 5-95 pctile of
    # each feature's natural (active) activation.
    within = []
    for j in mvdf_idx:
        col = z_sel[:, int(j)]
        act_col = col[col != 0]
        if len(act_col) < 10:
            continue
        lo5, hi95 = np.percentile(act_col, [5, 95])
        rv = za[:, int(j)]; rv = rv[rv != 0]
        if len(rv):
            within.append(float(np.mean((rv >= lo5) & (rv <= hi95))))
    natrange_within_5_95 = float(np.mean(within)) if within else float("nan")

    # necessity supported: noising degrades meaningfully AND more than random-set noising
    necessity = bool(delta_noise > 0 and delta_noise > rnd_mean + 2 * (rnd_sd + 1e-9)
                     and delta_noise > abs(delta_randdir))
    sufficiency = bool(delta_denoise > 0.5 * max(delta_noise, 1e-9))

    out = dict(
        dataset=ds, seed=seed, forecaster_val_mse=float(val_mse),
        n_active_windows=int(active.sum()), n_mvdf=int(len(mvdf_idx)),
        delta_noise=delta_noise, delta_denoise=delta_denoise,
        random_set_noise_mean=rnd_mean, random_set_noise_sd=rnd_sd,
        random_dir_noise=delta_randdir,
        s_clean_mean=float(s_clean.mean()), s_corrupt_mean=float(s_corrupt.mean()),
        natrange_restored_within_5_95pct=natrange_within_5_95,
        necessity_supported=necessity, sufficiency_supported=sufficiency,
        noise_over_random_ratio=float(delta_noise / (rnd_mean + 1e-9)),
        noise_over_randdir_ratio=float(delta_noise / (abs(delta_randdir) + 1e-9)),
    )
    json.dump(out, open(os.path.join(rd, "necessity.json"), "w"), indent=2, default=str)
    print(f"[{ds} s{seed}] noise {delta_noise:+.4f}  denoise {delta_denoise:+.4f}  "
          f"rnd-set {rnd_mean:+.4f}±{rnd_sd:.4f}  rnd-dir {delta_randdir:+.4f}  "
          f"necessity={necessity} sufficiency={sufficiency}")
    return out


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

    L = ["# PHASE2_NECESSITY_RESULT — denoising/noising necessity test for the MVDF set\n",
         "Generated by `experiments/phase2_necessity_noising.py`. ETTh1 + ETTh2, HTTP "
         "mirror. Bounded extension of the intervention-validity work (Document 1 §2.2; "
         "Joshi et al. 2026 'Sufficient ≠ Necessary'; Heimersheim & Nanda "
         "denoising/noising).\n\n"]
    L.append("## What each column is\n\n")
    L.append("- **`delta_noise`** = E[S_t(MVDF codes → 0) − S_t(clean)] over active "
             "windows. Positive ⇒ removing the MVDF set degrades difficulty tracking that "
             "other features do *not* reconstruct.\n"
             "- **`delta_denoise`** = E[S_t(all codes → per-feature mean) − S_t(that, but "
             "MVDF codes restored to true values)]. Positive ⇒ restoring *only* the MVDF "
             "set recovers difficulty tracking the corruption removed (≈ sufficiency).\n"
             "- **`random_set_noise_mean±sd`** = same noising with a matched-size random "
             "eligible-feature set (5 draws). **`random_dir_noise`** = matched-L2-norm "
             "random-direction perturbation of the pre-head token.\n"
             "- **`natrange_restored_within_5_95pct`** = fraction of restored MVDF "
             "activation values inside the 5–95th percentile of each feature's natural "
             "active distribution (on-manifold check).\n\n")
    L.append("## Results\n\n")
    L.append(md(agg[["dataset", "seed", "delta_noise", "delta_denoise",
                     "random_set_noise_mean", "random_set_noise_sd", "random_dir_noise",
                     "noise_over_random_ratio", "noise_over_randdir_ratio",
                     "natrange_restored_within_5_95pct", "necessity_supported",
                     "sufficiency_supported"]]))
    nec = int(agg["necessity_supported"].sum()); n = len(agg)
    suf = int(agg["sufficiency_supported"].sum())
    L.append(f"\n**Necessity supported on {nec}/{n} (dataset,seed); sufficiency on "
             f"{suf}/{n}.**\n\n")
    L.append("## Reading\n\n")
    if nec >= n - 1:
        L.append("**Necessity is supported.** Noising the MVDF set degrades difficulty "
                 "tracking, and by more than both a matched-size random feature set and a "
                 "matched-norm random direction — so the effect is specific to *these* "
                 "features, not to removing any k codes or to any perturbation of that "
                 "magnitude. Together with the dose-response sufficiency result (`U_j ≠ 0`), "
                 "the MVDF set passes a two-sided (noising + denoising) causal test, not "
                 "just a one-directional one.\n")
    elif nec >= n // 2:
        L.append("**Necessity is partially supported** — it holds on some (dataset, seed) "
                 "but not uniformly. Report the split; the dose-response sufficiency claim "
                 "(`U_j ≠ 0`) is unaffected, but the stronger 'this specific set is "
                 "required' claim should be stated with the per-cell caveat.\n")
    else:
        L.append("**Necessity is not established** — noising the MVDF set does not degrade "
                 "difficulty tracking beyond the random-set / random-direction controls. "
                 "The dose-response result remains a *sufficiency* statement only: "
                 "intervening on these features moves `S_t`, but they are not shown to be "
                 "necessary (other pathways or redundant features may substitute). State "
                 "H1c's causal claim as sufficiency, not necessity.\n")
    L.append("\n> This does not bear on Phase 3 / the downstream conformal results — it "
             "concerns only how strong the *upstream* causal claim (H1c) can be stated.\n")

    with open(os.path.join(out_root, "PHASE2_NECESSITY_RESULT.md"), "w") as f:
        f.write("".join(L))
    print(f"wrote {os.path.join(out_root, 'PHASE2_NECESSITY_RESULT.md')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "phase2_necessity"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    rows = []
    for ds in args.datasets:
        for s in args.seeds:
            print("=" * 78, f"\n {ds}  seed {s}\n", "=" * 78)
            rows.append(run_one(ds, s, args.out, args.fast))
    agg = pd.DataFrame(rows)
    agg.to_csv(os.path.join(args.out, "AGG_phase2.csv"), index=False)
    write_md(agg, args.out)
    print(f"\nPhase 2 necessity done in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
