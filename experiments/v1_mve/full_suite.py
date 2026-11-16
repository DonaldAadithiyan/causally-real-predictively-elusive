"""Full FFDCI-style suite driver (spec §6): loop the MVE over datasets x seeds,
aggregate the H1/H2/H3 tables and the RQ1 diagnostic across runs.

Scale target (spec §7 decision rule): if Delta_shift (H3) shows up on the MVE
(iTransformer + ETTh1 + ETTh2/traffic), run this over the full 12-dataset suite;
add architectures (Leddam, SOFTS) and the FFDCI-raw reproduction.

This is a thin wrapper around experiments.v1_mve.mve_real.main for now — it shells the
single-run driver per (dataset, seed) and concatenates the CSVs. Kept minimal until
the MVE go/no-go gate is passed.
"""
from __future__ import annotations

from mechconf.paths import REPO_ROOT  # noqa: E402
import argparse
import glob
import os
import subprocess
import sys

import pandas as pd
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--base_config", default="configs/mve_etth1.yaml")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    here = REPO_ROOT
    base = yaml.safe_load(open(os.path.join(here, args.base_config)))
    tmp_dir = os.path.join(here, "configs", "_generated")
    os.makedirs(tmp_dir, exist_ok=True)

    run_dirs = []
    for ds in args.datasets:
        for seed in args.seeds:
            cfg = dict(base)
            cfg["dataset"] = dict(base["dataset"], name=ds)
            cfg["seed"] = seed
            cfg["run_name"] = f"suite_{ds}_seed{seed}"
            cpath = os.path.join(tmp_dir, f"{ds}_seed{seed}.yaml")
            yaml.safe_dump(cfg, open(cpath, "w"))
            cmd = [sys.executable, "-m", "experiments.v1_mve.mve_real", "--config", cpath]
            if args.fast:
                cmd.append("--fast")
            print("RUN", " ".join(cmd))
            subprocess.run(cmd, cwd=here, check=True)
            run_dirs.append(os.path.join(here, "results", "v2", cfg["run_name"] + ("_fast" if args.fast else "")))

    # aggregate
    for fname in ("h2_in_distribution.csv", "h3_shift.csv"):
        frames = []
        for rd in run_dirs:
            p = os.path.join(rd, fname)
            if os.path.exists(p):
                df = pd.read_csv(p)
                df.insert(0, "run", os.path.basename(rd))
                frames.append(df)
        if frames:
            agg = pd.concat(frames, ignore_index=True)
            agg.to_csv(os.path.join(here, "results", "v2", f"AGG_{fname}"), index=False)
            print(f"[agg] results/v2/AGG_{fname}  ({len(agg)} rows)")


if __name__ == "__main__":
    main()
