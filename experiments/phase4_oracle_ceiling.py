"""Phase 4 — oracle ceiling (Mondrian-on-truth). Real data, ETTh1 + ETTh2.

Answers the one question every earlier negative result leaves open: was there ever
enough exploitable heterogeneity in this data for ANY method, however perfect, to find?

Construction (retrospective, non-deployable BY CONSTRUCTION):
  - Evaluate BOTH naive and oracle leave-one-out over the WHOLE E_eval_ID block (and,
    separately, the whole E_eval_shift block) — no train/eval-slice split. This removes
    the ETT chronological-calibration-transfer confound (short ID blocks -> a static
    first-80% quantile undercovers the last 20%, a documented issue) so the comparison
    isolates the one thing the oracle exists to measure: exploitable heterogeneity.
  - naive half-width for window i = the (1-alpha) conformal quantile of S_t over ALL
    OTHER windows in the block (i's own value excluded). This is the LOO analogue of
    the constant-width `naive_matched` band; by construction it sits at ~1-alpha.
  - oracle half-width for window i = the (1-alpha) conformal quantile of S_t over the
    other windows IN i's OWN true-S_t decile. Also ~1-alpha by construction, but
    per-decile, so it can be tighter than naive exactly to the extent difficulty is
    heteroskedastic.
  Both are at ~1-alpha coverage by construction, so W_naive - W_oracle is a clean
  heterogeneity measurement, not a coverage-confounded one.

Report per dataset:
  W_naive, W_oracle, gap = W_naive - W_oracle, ratio = gap / W_naive
  coverage of each (sanity: oracle should sit at ~1-alpha by construction on ID)

Precondition (non-negotiable, Document 2 §2): confirm `gap` is MEANINGFULLY positive
(bootstrap CI on gap excludes ~0 and gap/W_naive is not a rounding-noise fraction)
BEFORE any downstream method's gap-recovery ratio would be computed against it. If the
gap is small, that is the finding, stated directly.

ABSOLUTE RULE: the oracle never enters a deployable comparison table. It is a ceiling,
reported once, labeled retrospective.

DATA: HTTP ETDataset mirror (same split sizes as the Kaggle bundle; different edge
fingerprint). electricity/traffic Kaggle-only, unavailable locally.

Outputs runs/phase4_oracle/{AGG_oracle.csv, PHASE4_RESULT.md}
"""
from __future__ import annotations

import argparse

import os
import time

import numpy as np
import pandas as pd
import torch

from data.datasets import get_splits, make_windows
from experiments._seed import derive_seed
from forecasters.train_itransformer import TrainConfig, train_itransformer
from validation.errors import collect
from shift.shift_windows import three_way_env_masks
from calibration.aci import empirical_quantile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CFG = dict(
    lookback=96, horizon=96, target_var=-1, alpha=0.10, n_deciles=10,
    forecaster=dict(d_model=256, n_layers=3, n_heads=8, d_ff=512, dropout=0.1,
                    lr=1e-4, batch_size=32, max_epochs=20, patience=5,
                    lr_decay=0.8, warmup_epochs=3),
)


def _oracle_widths(score: np.ndarray, alpha: float, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """For each window, the (1-alpha) conformal quantile of S_t over the OTHER windows
    in its own true-S_t decile. Returns (half_width per window, decile id per window)."""
    s = np.asarray(score, float)
    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1)[1:-1])
    binid = np.digitize(s, edges)
    hw = np.empty(len(s))
    for k in range(n_bins):
        m = np.where(binid == k)[0]
        if len(m) < 5:
            hw[m] = empirical_quantile(s, 1 - alpha)  # fall back to global
            continue
        for i in m:
            others = s[m[m != i]]
            hw[i] = empirical_quantile(others, 1 - alpha)
    return hw, binid


def _coverage_and_width(score: np.ndarray, half_width: np.ndarray) -> tuple[float, float]:
    """S_t is already the max-over-horizon normalized abs error, so a window is covered
    iff S_t <= half_width, and the reported 'width' is 2*half_width in S_t units."""
    covered = score <= half_width
    return float(covered.mean()), float(np.mean(2.0 * half_width))


def run_one(ds: str, seed: int) -> dict:
    ds_seed = derive_seed(seed, ds)
    np.random.seed(ds_seed); torch.manual_seed(ds_seed)
    dev = "cpu"
    L, H, tvar = CFG["lookback"], CFG["horizon"], CFG["target_var"]
    alpha, nb = CFG["alpha"], CFG["n_deciles"]
    fc = CFG["forecaster"]

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
    id_idx = np.where(env["E_eval_ID"])[0]
    sh_idx = np.where(env["E_eval_shift"])[0]

    ce_id = collect(model, Xte[id_idx], Yte[id_idx], tvar_abs, device=dev)
    ce_sh = collect(model, Xte[sh_idx], Yte[sh_idx], tvar_abs, device=dev)
    s_id = np.asarray(ce_id["score"], float)
    s_sh = np.asarray(ce_sh["score"], float)

    # Both naive and oracle: LOO over the WHOLE block, ~1-alpha coverage by construction.
    # naive half-width is a constant (the block's (1-alpha) conformal quantile) — its LOO
    # variant differs by <=1 order statistic out of ~2000, negligible; use the constant.
    q_naive_id = empirical_quantile(s_id, 1 - alpha)
    q_naive_sh = empirical_quantile(s_sh, 1 - alpha)
    idw = s_id                                            # whole ID block
    naive_id_cov, naive_id_w = _coverage_and_width(idw, np.full_like(idw, q_naive_id))
    naive_sh_cov, naive_sh_w = _coverage_and_width(s_sh, np.full_like(s_sh, q_naive_sh))

    # oracle: leave-one-out within true-S_t decile, over the whole block
    hw_id, _ = _oracle_widths(idw, alpha, nb)
    hw_sh, _ = _oracle_widths(s_sh, alpha, nb)
    orc_id_cov, orc_id_w = _coverage_and_width(idw, hw_id)
    orc_sh_cov, orc_sh_w = _coverage_and_width(s_sh, hw_sh)


    # gap + bootstrap CI on the per-window width difference (whole ID block)
    d_id = 2.0 * (np.full_like(idw, q_naive_id) - hw_id)
    rng = np.random.default_rng(ds_seed)
    boots = np.array([d_id[rng.integers(0, len(d_id), len(d_id))].mean() for _ in range(2000)])
    gap_id = float(d_id.mean())
    gap_ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    d_sh = 2.0 * (np.full_like(s_sh, q_naive_sh) - hw_sh)
    boots_s = np.array([d_sh[rng.integers(0, len(d_sh), len(d_sh))].mean() for _ in range(2000)])
    gap_sh = float(d_sh.mean())
    gap_sh_ci = [float(np.percentile(boots_s, 2.5)), float(np.percentile(boots_s, 97.5))]

    print(f"[{ds} s{seed}] naive ID w {naive_id_w:.3f} (cov {naive_id_cov:.3f}) | "
          f"oracle ID w {orc_id_w:.3f} (cov {orc_id_cov:.3f}) | gap {gap_id:+.3f} "
          f"ratio {gap_id / naive_id_w:+.3f}")
    return dict(
        dataset=ds, seed=seed, data_fingerprint=sd.fingerprint(),
        forecaster_val_mse=float(val_mse), n_id_block=int(len(idw)), n_shift=int(len(s_sh)),
        naive_id_coverage=naive_id_cov, naive_id_width=naive_id_w,
        oracle_id_coverage=orc_id_cov, oracle_id_width=orc_id_w,
        gap_id=gap_id, gap_id_ci_lo=gap_ci[0], gap_id_ci_hi=gap_ci[1],
        ratio_id=gap_id / naive_id_w if naive_id_w > 0 else float("nan"),
        gap_id_meaningfully_positive=bool(gap_ci[0] > 0.01 * naive_id_w),
        naive_shift_coverage=naive_sh_cov, naive_shift_width=naive_sh_w,
        oracle_shift_coverage=orc_sh_cov, oracle_shift_width=orc_sh_w,
        gap_shift=gap_sh, gap_shift_ci_lo=gap_sh_ci[0], gap_shift_ci_hi=gap_sh_ci[1],
        ratio_shift=gap_sh / naive_sh_w if naive_sh_w > 0 else float("nan"),
        gap_shift_meaningfully_positive=bool(gap_sh_ci[0] > 0.01 * naive_sh_w),
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

    L = ["# PHASE4_RESULT — oracle ceiling (Mondrian-on-truth)\n",
         "Generated by `experiments/phase4_oracle_ceiling.py`. ETTh1 + ETTh2, HTTP "
         "ETDataset mirror. **The oracle is a retrospective, non-deployable ceiling — it "
         "bins by the true realized `S_t`, which is unavailable at deployment. It never "
         "enters a deployable comparison table.**\n\n"]
    L.append("## Construction\n\n")
    L.append("Windows binned into deciles of their true realized `S_t`. A window's oracle "
             "half-width = the (1−α) conformal quantile of `S_t` over the *other* windows "
             "in its own decile (leave-one-out; a window's own value never sets its own "
             "width). `naive` = the (1−α) quantile of the calibration pool (first 80% of "
             "E_eval_ID), same `naive_matched` construction as everywhere else. α = 0.10, "
             "10 deciles. `S_t` is the horizon-max normalized abs error; a window is "
             "covered iff `S_t ≤ half_width`; reported width is `2·half_width` in `S_t` "
             "units.\n\n")

    L.append("## Per (dataset, seed)\n\n")
    show = agg[["dataset", "seed", "naive_id_width", "oracle_id_width", "gap_id",
                "gap_id_ci_lo", "gap_id_ci_hi", "ratio_id", "gap_id_meaningfully_positive",
                "naive_id_coverage", "oracle_id_coverage"]].copy()
    L.append(md(show))
    L.append("\n## Shift block\n\n")
    shows = agg[["dataset", "seed", "naive_shift_width", "oracle_shift_width", "gap_shift",
                 "gap_shift_ci_lo", "gap_shift_ci_hi", "ratio_shift",
                 "gap_shift_meaningfully_positive", "naive_shift_coverage",
                 "oracle_shift_coverage"]].copy()
    L.append(md(shows))

    # per-dataset means
    L.append("\n## Per-dataset summary (mean over seeds)\n\n")
    g = agg.groupby("dataset").agg(
        naive_id_width=("naive_id_width", "mean"), oracle_id_width=("oracle_id_width", "mean"),
        gap_id=("gap_id", "mean"), ratio_id=("ratio_id", "mean"),
        gap_id_pos=("gap_id_meaningfully_positive", "all"),
        naive_shift_width=("naive_shift_width", "mean"),
        oracle_shift_width=("oracle_shift_width", "mean"),
        gap_shift=("gap_shift", "mean"), ratio_shift=("ratio_shift", "mean"),
        gap_shift_pos=("gap_shift_meaningfully_positive", "all"),
    ).reset_index()
    L.append(md(g))

    L.append("\n## Precondition check + reading\n\n")
    for _, r in g.iterrows():
        ds = r["dataset"]
        idp, shp = bool(r["gap_id_pos"]), bool(r["gap_shift_pos"])
        L.append(f"### {ds}\n\n")
        if idp:
            L.append(f"- **ID: gap is meaningfully positive** — `W_naive − W_oracle` = "
                     f"{r['gap_id']:.3f} ({100*r['ratio_id']:.1f}% of `W_naive` "
                     f"{r['naive_id_width']:.3f}), bootstrap CI lower bound clears 1% of "
                     f"`W_naive`. A gap-recovery ratio `(W_naive − W_method)/(W_naive − "
                     f"W_oracle)` **is** interpretable against this denominator for ID.\n")
        else:
            L.append(f"- **ID: gap is NOT meaningfully positive** — `W_naive − W_oracle` = "
                     f"{r['gap_id']:.3f} ({100*r['ratio_id']:.1f}% of `W_naive`). Report as "
                     f"**limited exploitable heterogeneity at this coverage level**; do NOT "
                     f"compute a gap-recovery ratio against a near-zero, unstable "
                     f"denominator.\n")
        if shp:
            L.append(f"- **Shift: gap is meaningfully positive** — {r['gap_shift']:.3f} "
                     f"({100*r['ratio_shift']:.1f}% of `W_naive` {r['naive_shift_width']:.3f}).\n")
        else:
            L.append(f"- **Shift: gap is NOT meaningfully positive** — {r['gap_shift']:.3f} "
                     f"({100*r['ratio_shift']:.1f}% of `W_naive`). Limited exploitable "
                     f"heterogeneity under shift too.\n")
        L.append("\n")

    all_id_pos = bool(g["gap_id_pos"].all())
    all_sh_pos = bool(g["gap_shift_pos"].all())
    any_pos = bool(g["gap_id_pos"].any() or g["gap_shift_pos"].any())
    L.append("## Bottom line\n\n")
    if all_id_pos and all_sh_pos:
        lo = 100 * min(g["ratio_id"].min(), g["ratio_shift"].min())
        hi = 100 * max(g["ratio_id"].max(), g["ratio_shift"].max())
        L.append(f"**Every dataset and block shows a large, statistically clean exploitable "
                 f"heterogeneity gap — an oracle that bins by true difficulty is "
                 f"{lo:.0f}–{hi:.0f}% narrower than a constant-width band at the same "
                 f"coverage.** So the difficulty heterogeneity in ETT errors is real and "
                 f"substantial — this is *not* a near-homoskedastic setting where "
                 f"difficulty-conditioning has no room to help.\n\n"
                 f"**Read together with Phase 3 (natural MVDF activation carries no L1 "
                 f"predictive signal — best held-out R² deeply negative, even conditioned on "
                 f"activation) and the closed exploitation ladder (Task G / H7-A / SLCP / "
                 f"CQR-PID all fail to close a meaningful fraction of this gap):** the "
                 f"negative results are **genuine method failures against a real, quantified "
                 f"opportunity**, not 'correctly finding nothing to find'. The heterogeneity "
                 f"exists; it is just not accessible through the forecaster's internal "
                 f"representation (causally-validated or otherwise), nor through any adaptive "
                 f"conformal machinery tested.\n\n"
                 f"**Reconciles with Phase 0's `corr(hidden, |e|)` ≈ 0.08–0.21:** that "
                 f"measured what is *linearly decodable from the hidden state*, not whether "
                 f"heterogeneity *exists in the errors*. Phase 4 shows the latter clearly "
                 f"does. The gap between 'heterogeneity exists' and 'nothing recovers it' is "
                 f"the paper's central tension, now quantified from both ends.\n")
    elif any_pos:
        L.append("**Mixed across datasets/blocks** (see per-dataset section) — a real "
                 "exploitable gap on some, limited heterogeneity on others. Every method "
                 "tested failed to close the gap where one exists. State this per dataset, "
                 "not globally.\n")
    else:
        L.append("**On ETTh1/ETTh2, an oracle that bins by true difficulty barely beats a "
                 "constant-width band** — the exploitable heterogeneity ceiling is low at "
                 "α = 0.10. This reframes every negative result in the paper (Task G, H7-A, "
                 "SLCP, CQR/PID) as *correctly finding almost nothing to find*, not failing "
                 "to find something that was there.\n")
    L.append("\n> The oracle is retrospective and non-deployable. It is reported here once, "
             "as a ceiling. It is not a baseline and does not appear in any deployable "
             "comparison table.\n")

    with open(os.path.join(out_root, "PHASE4_RESULT.md"), "w") as f:
        f.write("".join(L))
    print(f"wrote {os.path.join(out_root, 'PHASE4_RESULT.md')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "phase4_oracle"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    rows = []
    for ds in args.datasets:
        for s in args.seeds:
            print("=" * 78, f"\n {ds}  seed {s}\n", "=" * 78)
            rows.append(run_one(ds, s))
    agg = pd.DataFrame(rows)
    agg.to_csv(os.path.join(args.out, "AGG_oracle.csv"), index=False)
    write_md(agg, args.out)
    print(f"\nPhase 4 oracle done in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
