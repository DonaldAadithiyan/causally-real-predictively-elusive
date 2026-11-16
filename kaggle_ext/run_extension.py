"""Driver — run the 6 ICAART Section 5.3-5.11 experiments on electricity + traffic.

Mirrors the documented ETTh1/ETTh2 protocol exactly (same seeds 0/1/2, same configs).
Each block appends the two new datasets to the SAME runs/<phase>/ output dirs and
regenerates the corresponding *_RESULT.md with electricity+traffic folded in.

Run from the repo root, AFTER `bash patch_gpu.sh`:
    MC_DEVICE=cuda python run_extension.py --all
or a subset:
    MC_DEVICE=cuda python run_extension.py --only phase3 oracle

Sections -> scripts (5 jobs; see KAGGLE_README.md for why not 6):
    5.3  Phase 3 dissociation      experiments/phase3_real_predictability.py
    5.5  Oracle ceiling            experiments/phase4_oracle_ceiling.py
    5.9  CQR + PID baseline        experiments/baseline_comparison.py
    5.10 Arch-gen (TimeMixer)      experiments/arch_gen_timemixer.py
    5.11 SAE reproducibility       experiments/phase5_sae_reproducibility.py

  NOT run here:
    5.4  synthetic control  -> dataset-independent, nothing to add.
    5.6  Task G / H6        -> v2_h6_topup/AGG_h6.csv ALREADY has electricity +
         traffic (96 rows = 4 datasets x 3 seeds x 8 modes). Confirmed already in
         the paper's RESULTS_FINAL.md. If you want to regenerate it locally anyway,
         see the `taskg` job (disabled by default).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["electricity", "traffic"]
SEEDS = ["0", "1", "2"]

# name -> (module, extra args). All use `python -m experiments.<mod>`.
JOBS = {
    "phase3": ("experiments.v2.phase3_real_predictability",
               ["--datasets", *DATASETS, "--seeds", *SEEDS]),
    "oracle": ("experiments.v2.phase4_oracle_ceiling",
               ["--datasets", *DATASETS, "--seeds", *SEEDS]),
    "baseline": ("experiments.v2.baseline_comparison",
                 ["--datasets", *DATASETS, "--seeds", *SEEDS]),
    "archgen": ("experiments.v2.arch_gen_timemixer",
                ["--datasets", *DATASETS, "--seeds", *SEEDS]),
    "phase5": ("experiments.v2.phase5_sae_reproducibility",
               ["--datasets", *DATASETS]),
    # optional: Task G is already complete for all 4 datasets in v2_h6_topup/.
    # Only run this if you deliberately want a local-retrain regeneration.
    "taskg":  ("experiments.v2.phase3_scorer_h6",
               ["--datasets", *DATASETS, "--seeds", *SEEDS,
                "--phase1_root_tmpl", "results/v2/v2_confirm/v2_phase1_seed{seed}"]),
}
ORDER = ["phase3", "oracle", "baseline", "archgen", "phase5"]  # taskg excluded by default


def run_one(name: str) -> tuple[str, int, float]:
    mod, extra = JOBS[name]
    t0 = time.time()
    print(f"\n{'#'*78}\n#  {name}  ->  python -m {mod} {' '.join(extra)}\n{'#'*78}",
          flush=True)
    env = dict(os.environ)
    env.setdefault("MC_DEVICE", "cuda")
    env.setdefault("PYTHONUNBUFFERED", "1")
    p = subprocess.run([sys.executable, "-m", mod, *extra], cwd=REPO, env=env)
    dt = time.time() - t0
    print(f"[{name}] exit {p.returncode}  ({dt/60:.1f} min)", flush=True)
    return name, p.returncode, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run every job in ORDER")
    ap.add_argument("--only", nargs="+", choices=list(JOBS),
                    help="run just these jobs (in the given order)")
    ap.add_argument("--with-taskg", action="store_true",
                    help="also regenerate Task G locally (default: skip, already done)")
    args = ap.parse_args()

    if not (args.all or args.only):
        ap.error("pass --all or --only <job...>")
    jobs = list(ORDER) if args.all else list(args.only)
    if args.with_taskg and "taskg" not in jobs:
        jobs.append("taskg")

    # sanity: raw files present
    for ds in DATASETS:
        p = os.path.join(REPO, "data", "raw", f"{ds}.csv")
        if not os.path.exists(p):
            sys.exit(f"missing {p} — attach the recovered raw files first "
                     f"(see KAGGLE_README.md step 2)")

    results = []
    for j in jobs:
        results.append(run_one(j))

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    bad = 0
    for name, rc, dt in results:
        flag = "OK " if rc == 0 else "FAIL"
        if rc != 0:
            bad += 1
        print(f"  {flag}  {name:9s}  {dt/60:6.1f} min")
    print(f"\n{'all good' if bad == 0 else f'{bad} job(s) FAILED — see logs above'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
