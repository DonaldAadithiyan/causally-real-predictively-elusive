#!/usr/bin/env bash
# Re-run the camera-ready pipeline stage by stage. Wall-clock figures are those measured for the
# committed results (Apple Silicon; ETTh on CPU with 4 torch threads - the setting under which
# seeds reproduce bit-exactly; electricity/traffic forecasters and SAEs on MPS).
#
#   bash scripts/reproduce_all.sh <stage>|all
#
# Stages read and write results/ (outputs are deterministic; re-running overwrites with identical
# values). Large inputs (weights, per-window arrays) come from the release assets
# (artifacts/MANIFEST.sha256) or from the `train` / `arrays` stages.
set -euo pipefail
PY=${PY:-python}
stage=${1:-all}
ETT_SEEDS="0 1 2 3 4 5 6 7 8 9"; ET_SEEDS="0 1 2 3 4"

run() { echo "== $*"; "$@"; }

train()   { # ~6-15 min per ETTh seed (CPU), ~12-21 min per electricity/traffic seed (MPS); ~3.5 h in total
  run $PY -m experiments.camera_ready.train_seeds --datasets ETTh1 ETTh2 --seeds 3 4 5 6 7 8 9 --device cpu
  run $PY -m experiments.camera_ready.train_seeds --datasets electricity traffic --seeds 3 4 --device mps
  # seeds 0-2: experiments/round2/regen_arrays.py (ETTh) and experiments/round2/r2_wp6.py (electricity/traffic)
}
arrays()  { # ~1 min per traffic seed: rebuild traffic seed 0-2 arrays from saved weights
  run $PY -m experiments.camera_ready.arrays_from_weights --dataset traffic --seeds 0 1 2; }
phase34() { # ~1 min: Phase 3 battery + Phase 4 oracle from arrays (E1, E2)
  run $PY -m experiments.camera_ready.phase34 --datasets ETTh1 ETTh2 --seeds $ETT_SEEDS --out results/camera_ready/E2
  run $PY -m experiments.camera_ready.phase34 --datasets electricity traffic --seeds $ET_SEEDS --out results/camera_ready/E1; }
scorer()  { # ~1 min: continuous scorer vs naive_matched (E2 metric 4)
  run $PY -m experiments.camera_ready.e2_scorer --datasets ETTh1 ETTh2 --seeds $ETT_SEEDS --out results/camera_ready/E2/_sc_ett.csv
  run $PY -m experiments.camera_ready.e2_scorer --datasets electricity traffic --seeds $ET_SEEDS --out results/camera_ready/E2/_sc_et.csv
  run $PY - <<'PYEOF'
import os, pandas as pd
d = "results/camera_ready/E2/"
pd.concat([pd.read_csv(d + "_sc_ett.csv"), pd.read_csv(d + "_sc_et.csv")]).to_csv(d + "AGG_E2_scorer_vs_naive.csv", index=False)
b = pd.read_pickle(d + "_sc_ett_boot.pkl"); b.update(pd.read_pickle(d + "_sc_et_boot.pkl"))
pd.to_pickle(b, d + "AGG_E2_scorer_vs_naive_boot.pkl")
for f in ("_sc_ett.csv", "_sc_et.csv", "_sc_ett_boot.pkl", "_sc_et_boot.pkl"):
    os.remove(d + f)
PYEOF
}
e3()      { # ~5-8 min per cell, 30 cells (~3 h); ex-ante baselines + gap recovery (E3, E1)
  run $PY -m experiments.camera_ready.test_leakage
  run $PY -m experiments.camera_ready.e3_exante --datasets ETTh1 ETTh2 electricity traffic --seeds $ETT_SEEDS --out results/camera_ready/E3 --skip_done || true
  run $PY -m experiments.camera_ready.report_e3 --datasets ETTh1 ETTh2 --out results/camera_ready/E3 --prefix E3
  run $PY -m experiments.camera_ready.report_e3 --datasets electricity traffic --out results/camera_ready/E1 --prefix E1_E3; }
e5()      { # ~10 s per cell: dose-response diagnostics (E5a) and interchange (E5c); E5b is post-processing
  run $PY -m experiments.camera_ready.e5_dose a --datasets ETTh1 ETTh2 --seeds $ETT_SEEDS
  run $PY -m experiments.camera_ready.e5_dose a --datasets electricity traffic --seeds $ET_SEEDS
  run $PY -m experiments.camera_ready.e5_dose c --datasets ETTh1 ETTh2 --seeds $ETT_SEEDS
  run $PY -m experiments.camera_ready.e5_dose c --datasets electricity traffic --seeds $ET_SEEDS
  run $PY -m experiments.camera_ready.report_e5; }
e4()      { # ~4 min: R_j robustness (needs E5a outputs)
  run $PY -m experiments.camera_ready.e4_rj
  run $PY -m experiments.camera_ready.report_e4; }
e6()      { # ~10 min per TimeMixer run (CPU) + ~25 min analysis
  run $PY -m experiments.camera_ready.e6_timemixer --datasets ETTh1 ETTh2 --seeds 0 1 2
  CR_ARR_DIR=results/camera_ready/E6/arrays run $PY -m experiments.camera_ready.phase34 --datasets ETTh1 ETTh2 --seeds 0 1 2 --out results/camera_ready/E6
  CR_ARR_DIR=results/camera_ready/E6/arrays run $PY -m experiments.camera_ready.e3_exante --datasets ETTh1 ETTh2 --seeds 0 1 2 --sets F0 F1 F1p F2 F6 --out results/camera_ready/E6
  run $PY -m experiments.camera_ready.report_e3 --cells results/camera_ready/E6/cells --out results/camera_ready/E6 --datasets ETTh1 ETTh2 --prefix E6; }
f()       { # ~4 min: regime gating on 10 seeds, pooled enrichment vs selection null, shift GRR
  run $PY -m experiments.camera_ready.f_followup; }
g()       { # G2 ~2 min, G4 ~15 min, G5 instant (G1 and G3 were stopped; see results/camera_ready/G/STATUS.md)
  run $PY -m experiments.camera_ready.g_analysis g2
  run $PY -m experiments.camera_ready.g_analysis g4
  run $PY -m experiments.camera_ready.g_analysis g5; }
reports() { run $PY -m experiments.camera_ready.report_e2; run $PY -m experiments.camera_ready.report_e1; }

case "$stage" in
  all) train; arrays; phase34; scorer; e3; e5; e4; e6; f; g; reports ;;
  train|arrays|phase34|scorer|e3|e4|e5|e6|f|g|reports) $stage ;;
  *) echo "unknown stage: $stage"; exit 2 ;;
esac
