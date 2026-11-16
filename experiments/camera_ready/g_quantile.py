"""G1 helper: CQR quantile forecasts for every run (target channel, all test windows).

Seeds with a saved round-2 quantile model are predicted from it; others are trained with
the unchanged recipe of experiments/baseline_comparison.py (QuantileTrainConfig, seed
derive_seed + 31337, forecaster hyper-parameters of phase3 CFG, alpha 0.10).
Output: results/camera_ready/G/quantile/<ds>_seed<s>.npz {q_lo, q_hi} (+ _meta.json)
  python -m experiments.camera_ready.g_quantile --datasets ETTh1 ETTh2 --seeds 0 1 ... --device cpu
"""
from __future__ import annotations

from mechconf.paths import REPO_ROOT  # noqa: E402
import argparse
import json
import os
import sys
import time

HERE = REPO_ROOT
CR = os.path.join(HERE, "results", "camera_ready")
os.environ["KAGGLE_DATA_DIR"] = os.path.join(CR, "_no_kaggle_bundle")
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import mechconf.data.datasets as dsmod  # noqa: E402
from mechconf.data.datasets import make_windows  # noqa: E402
from experiments._seed import derive_seed  # noqa: E402
from experiments.v2.phase3_real_predictability import CFG  # noqa: E402
from mechconf.baselines.quantile_itransformer import (QuantileTrainConfig, train_quantile_itransformer,  # noqa: E402
                                             predict_quantiles)

OUT = os.path.join(CR, "G", "quantile")
R2W = os.path.join(HERE, "results", "round2", "p126_round2", "weights")


def build_cfg(seed, device):
    fc = CFG["forecaster"]
    return QuantileTrainConfig(lookback=CFG["lookback"], horizon=CFG["horizon"], d_model=fc["d_model"],
                               n_layers=fc["n_layers"], n_heads=fc["n_heads"], d_ff=fc["d_ff"],
                               dropout=fc["dropout"], lr=fc["lr"], batch_size=fc["batch_size"],
                               max_epochs=fc["max_epochs"], patience=fc["patience"],
                               lr_decay=fc["lr_decay"], warmup_epochs=fc["warmup_epochs"],
                               seed=seed, device=device, alpha=0.10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(4)
    os.makedirs(OUT, exist_ok=True)
    L, H = CFG["lookback"], CFG["horizon"]
    for ds in a.datasets:
        sd = dsmod.get_splits(ds, max_channels=48)
        tv = CFG["target_var"] % len(sd.channels)
        Xtr, Ytr, _ = make_windows(sd.train, L, H)
        Xva, Yva, _ = make_windows(sd.val, L, H)
        Xte, _, _ = make_windows(sd.test, L, H)
        for s in a.seeds:
            tag = f"{ds}_seed{s}"
            if os.path.exists(os.path.join(OUT, f"{tag}.npz")):
                continue
            t0 = time.time()
            qseed = derive_seed(s, ds) + 31337
            wp = os.path.join(R2W, f"{tag}_quantile.pt")
            np.random.seed(qseed); torch.manual_seed(qseed)
            if os.path.exists(wp):
                from mechconf.baselines.quantile_itransformer import QuantileITransformer
                fc = CFG["forecaster"]
                qm = QuantileITransformer(n_vars=len(sd.channels), lookback=L, horizon=H,
                                          d_model=fc["d_model"], n_layers=fc["n_layers"],
                                          n_heads=fc["n_heads"], d_ff=fc["d_ff"], dropout=fc["dropout"])
                qm.load_state_dict(torch.load(wp, map_location="cpu"))
                src, val = "round2_saved_weights", None
            else:
                qm, val = train_quantile_itransformer((Xtr, Ytr, None), (Xva, Yva, None),
                                                      n_vars=len(sd.channels), cfg=build_cfg(qseed, a.device),
                                                      verbose=False)
                src = f"trained_{a.device}"
                os.makedirs(os.path.join(CR, "G", "quantile_weights"), exist_ok=True)
                torch.save(qm.cpu().state_dict(), os.path.join(CR, "G", "quantile_weights", f"{tag}_quantile.pt"))
            qm = qm.cpu().eval()
            lo, hi = predict_quantiles(qm, Xte, device="cpu")
            np.savez_compressed(os.path.join(OUT, f"{tag}.npz"), q_lo=lo[:, :, tv].astype(np.float32),
                                q_hi=hi[:, :, tv].astype(np.float32))
            json.dump(dict(tag=tag, source=src, seed=qseed, val=(None if val is None else float(val)),
                           seconds=time.time() - t0), open(os.path.join(OUT, f"{tag}_meta.json"), "w"), indent=2)
            print(f"[{tag}] {src} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
