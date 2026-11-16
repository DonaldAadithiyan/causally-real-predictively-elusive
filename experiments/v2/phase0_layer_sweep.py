"""Phase 0 driver (v2 spec §11): cheap layer sweep across all datasets, single seed.

For each dataset: train the iTransformer (or reuse if cached), compute the
nonconformity score S_t on the calibration (VAL) split, and measure the cheap
first-pass signal max|corr(activation PC, S_t)| at every one of the 7 taps.

Outputs results/v2/v2_phase0/:
  layer_sweep.csv        (dataset, tap, max_abs_corr, max_abs_corr_pc, mean_top10)
  layer_sweep_summary.json  {winning_layers | NO_GO, per-tap mean across datasets}
  layer_sweep.png        heatmap dataset x tap
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
from mechconf.forecasters.train_itransformer import TrainConfig, train_itransformer
from mechconf.validation.errors import collect
from mechconf.representation.layer_sweep import sweep_dataset, select_winning_layers, NO_GO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2_phase0.yaml")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.datasets:
        cfg["datasets"] = args.datasets
    if args.fast:
        cfg["forecaster"]["max_epochs"] = 3

    here = REPO_ROOT
    out = os.path.join(here, "results", "v2", cfg["run_name"] + ("_fast" if args.fast else ""))
    os.makedirs(out, exist_ok=True)
    seed = cfg["seed"]; np.random.seed(seed); torch.manual_seed(seed)
    dev = cfg["device"]
    dd = cfg["dataset_defaults"]
    L, H, tvar = dd["lookback"], dd["horizon"], dd["target_var"]

    rows = []
    log = {"config": cfg, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "per_dataset": {}}

    for ds in cfg["datasets"]:
        print("=" * 70, "\n", ds, "\n", "=" * 70)
        try:
            sd = get_splits(ds, max_channels=dd["max_channels"])
        except Exception as e:
            print(f"SKIP {ds}: {e}")
            log["per_dataset"][ds] = {"skipped": str(e)}
            continue
        tvar_abs = tvar % len(sd.channels)
        Xtr, Ytr, _ = make_windows(sd.train, L, H)
        Xva, Yva, _ = make_windows(sd.val, L, H)

        fc = cfg["forecaster"]
        tcfg = TrainConfig(lookback=L, horizon=H, d_model=fc["d_model"], n_layers=fc["n_layers"],
                           n_heads=fc["n_heads"], d_ff=fc["d_ff"], dropout=fc["dropout"],
                           lr=fc["lr"], batch_size=fc["batch_size"], max_epochs=fc["max_epochs"],
                           patience=fc["patience"], lr_decay=fc.get("lr_decay", 0.8),
                           warmup_epochs=fc.get("warmup_epochs", 3), seed=seed, device=dev)
        t0 = time.time()
        model, val_mse = train_itransformer((Xtr, Ytr, None), (Xva, Yva, None),
                                            n_vars=len(sd.channels), cfg=tcfg, verbose=True)
        cal = collect(model, Xva, Yva, tvar_abs, device=dev)   # cal["score"] = max/profile S_t

        sigs = sweep_dataset(model, Xva, cal["score"], tvar_abs, device=dev)
        for tap, s in sigs.items():
            rows.append(dict(dataset=ds, tap=tap, max_abs_corr=s.max_abs_corr,
                             max_abs_corr_pc=s.max_abs_corr_pc,
                             mean_abs_corr_top10=s.mean_abs_corr_top10, n_dims=s.n_dims))
        log["per_dataset"][ds] = dict(
            forecaster_val_mse=float(val_mse), train_s=round(time.time() - t0, 1),
            signals={t: s.max_abs_corr_pc for t, s in sigs.items()},
        )
        print("  cheap signal (max|corr(PC, S_t)|) by tap:")
        for t, s in sigs.items():
            print(f"    {t:12s} {s.max_abs_corr_pc:.3f}   (raw-dim max {s.max_abs_corr:.3f})")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "layer_sweep.csv"), index=False)

    gc = cfg["go_criterion"]
    from mechconf.representation.layer_sweep import LayerSignal
    sbd = {}
    for ds, g in df.groupby("dataset"):
        sbd[ds] = {r["tap"]: LayerSignal(tap=r["tap"], max_abs_corr=r["max_abs_corr"],
                                         max_abs_corr_pc=r["max_abs_corr_pc"],
                                         mean_abs_corr_top10=r["mean_abs_corr_top10"],
                                         n_dims=int(r["n_dims"])) for _, r in g.iterrows()}
    winners = select_winning_layers(sbd, gc["go_threshold"], gc["go_min_datasets"], gc["metric"])

    per_tap_mean = df.groupby("tap")[gc["metric"]].agg(["mean", "max", "min"]).round(3).to_dict("index")
    summary = dict(
        go_threshold=gc["go_threshold"], go_min_datasets=gc["go_min_datasets"], metric=gc["metric"],
        result="NO_GO" if winners == NO_GO else "GO",
        winning_layers=None if winners == NO_GO else winners[: cfg["phase1_max_taps"]],
        all_cleared=None if winners == NO_GO else winners,
        per_tap_signal_across_datasets=per_tap_mean,
        run5_prehead_baseline="0.09-0.22 (max ~0.22 weather)",
        interpretation=(
            "NO-GO: no tap clears the threshold on enough datasets. Difficulty is not "
            "recoverable above the Run 5 pre-head ceiling at any probed depth (spec §7 "
            "row 4). Phase 1 proceeds at pre_head as a documented fallback; this null "
            "sweep is a reported result."
            if winners == NO_GO else
            f"GO: taps {winners[: cfg['phase1_max_taps']]} clear the threshold. Phase 1 "
            f"runs the full P_j/U_j/R_j suite at these taps."
        ),
    )
    json.dump(summary, open(os.path.join(out, "layer_sweep_summary.json"), "w"), indent=2)
    log["summary"] = summary
    log["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(log, open(os.path.join(out, "run_log.json"), "w"), indent=2, default=str)

    # heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        piv = df.pivot(index="dataset", columns="tap", values=gc["metric"]).reindex(
            columns=[t for t in sbd[next(iter(sbd))].keys()])
        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.35, gc["go_threshold"] * 1.2))
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        for (i, j), v in np.ndenumerate(piv.values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < gc["go_threshold"] else "black", fontsize=8)
        ax.set_title(f"Phase 0 cheap signal ({gc['metric']})  —  go_threshold={gc['go_threshold']}")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(os.path.join(out, "layer_sweep.png"), dpi=130); plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)

    print("\n" + "=" * 70)
    print("PHASE 0 RESULT:", summary["result"])
    print("  winning layers:", summary["winning_layers"])
    print("  per-tap mean signal:", json.dumps(per_tap_mean, indent=2))
    print(f"\nOutputs -> {out}")


if __name__ == "__main__":
    main()
