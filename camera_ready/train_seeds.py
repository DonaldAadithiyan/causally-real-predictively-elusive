"""Camera-ready E2: full pipeline retrains (forecaster -> SAE -> frozen MVDF selection).

Thin wrapper over the round-2 pipeline `p126/r2_wp6.regen` (itself a verified replica of
experiments/phase3_real_predictability.run_one; selection calls the unchanged project
functions). Only the output directories and the data search path are changed:

  weights -> runs/camera_ready/weights/<ds>_seed<s>_{forecaster,sae}.pt
  arrays  -> runs/camera_ready/arrays/<ds>_seed<s>.npz      (same format as round 2)
  meta    -> runs/camera_ready/E2/meta/<ds>_seed<s>_{meta.json,candidates.csv}

KAGGLE_DATA_DIR is pointed at an empty directory so data/raw (HTTP ETT mirror; electricity
and traffic byte-identical to the Kaggle bundle) is always the copy used.

Usage: python -m camera_ready.train_seeds --datasets ETTh1 ETTh2 --seeds 3 4 [--device cpu]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CR = os.path.join(HERE, "runs", "camera_ready")
EMPTY = os.path.join(CR, "_no_kaggle_bundle")
os.makedirs(EMPTY, exist_ok=True)
os.environ["KAGGLE_DATA_DIR"] = EMPTY


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    os.environ["MC_DEVICE"] = a.device
    os.environ["P126_THREADS"] = str(a.threads)
    sys.path.insert(0, HERE)
    import p126.r2_wp6 as w
    w.ARR = os.path.join(CR, "arrays")
    w.WTS = os.path.join(CR, "weights")
    w.OUT = os.path.join(CR, "E2", "meta")
    w.DEV = a.device
    for d in (w.ARR, w.WTS, w.OUT):
        os.makedirs(d, exist_ok=True)
    for ds in a.datasets:
        raw = os.path.join(HERE, "data", "raw", f"{ds}.csv")
        for s in a.seeds:
            tag = f"{ds}_seed{s}"
            if os.path.exists(os.path.join(w.ARR, f"{tag}.npz")):
                print(f"[{tag}] exists, skipping", flush=True)
                continue
            t0 = time.time()
            meta = w.regen(ds, s, with_quantile=False, with_selection=True)
            meta.update(raw_sha256=sha256(raw), raw_file=os.path.relpath(raw, HERE),
                        wall_clock_s=time.time() - t0, device=a.device, threads=a.threads)
            json.dump(meta, open(os.path.join(w.OUT, f"{tag}_meta.json"), "w"), indent=2,
                      default=float)
            print(f"[{tag}] done {time.time()-t0:.0f}s val_mse {meta['val_mse']:.4f} "
                  f"fvu {meta.get('sae_fvu', float('nan')):.4f} enr {meta.get('enrichment_typeB')}",
                  flush=True)


if __name__ == "__main__":
    main()
