"""Rebuild per-window arrays from SAVED weights (inference only, no training).

Used for traffic seeds 0-2: round 2 saved their weights and meta but the arrays are not
on disk. Output format is identical to p126/r2_wp6.regen. Reproduction check: SAE FVU on
the train-split activations must equal the stored meta value.

Usage: python -m experiments.camera_ready.arrays_from_weights --dataset traffic --seeds 0 1 2
"""
from __future__ import annotations

from mechconf.paths import REPO_ROOT  # noqa: E402
import argparse
import json
import os
import sys

HERE = REPO_ROOT
CR = os.path.join(HERE, "results", "camera_ready")
os.environ["KAGGLE_DATA_DIR"] = os.path.join(CR, "_no_kaggle_bundle")
os.makedirs(os.environ["KAGGLE_DATA_DIR"], exist_ok=True)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
import mechconf.data.datasets as dsmod  # noqa: E402
from mechconf.data.datasets import make_windows  # noqa: E402
from experiments.v2.phase3_real_predictability import CFG as P3CFG  # noqa: E402
from mechconf.representation.layer_sweep import extract_all_taps  # noqa: E402
from mechconf.sae.train_sae import encode_all  # noqa: E402
from mechconf.data.shift_windows import three_way_env_masks  # noqa: E402
from experiments.round2.regen_arrays import _sparse, _predict  # noqa: E402
from experiments.round2.selection import load_real_models  # noqa: E402
from experiments.camera_ready.common import meta as get_meta, weights_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    a = ap.parse_args()
    ds = a.dataset
    L, H = P3CFG["lookback"], P3CFG["horizon"]
    sd = dsmod.get_splits(ds, max_channels=48)
    tv = P3CFG["target_var"] % len(sd.channels)
    Xtr, _, _ = make_windows(sd.train, L, H)
    Xte, Yte, end_te = make_windows(sd.test, L, H)
    env = three_way_env_masks(end_te, len(sd.test))
    os.makedirs(os.path.join(CR, "arrays"), exist_ok=True)
    for s in a.seeds:
        tag = f"{ds}_seed{s}"
        m = get_meta(ds, s)
        model, sae = load_real_models(weights_dir(tag), tag, len(sd.channels), P3CFG)
        tr_act = extract_all_taps(model, Xtr, tv)[P3CFG["tap"]]
        with torch.no_grad():
            xb = sae.standardize(torch.from_numpy(tr_act.astype(np.float32)))
            xh, _ = sae(xb)
            fvu = float((torch.mean((xh - xb) ** 2) / (torch.var(xb) + 1e-12)).item())
        yhat = _predict(model, Xte)
        act = extract_all_taps(model, Xte, tv)[P3CFG["tap"]]
        z = encode_all(sae, act)
        zi, zv = _sparse(z)
        np.savez_compressed(
            os.path.join(CR, "arrays", f"{tag}.npz"),
            X=Xte.astype(np.float32), y=Yte[:, :, tv].astype(np.float32),
            yhat=yhat[:, :, tv].astype(np.float32), hidden=act.astype(np.float32),
            end_idx=end_te.astype(np.int32), origin_ts=np.full(len(end_te), -1, np.int64),
            m_select=env["E_select"], m_id=env["E_eval_ID"], m_shift=env["E_eval_shift"],
            target_var=np.int32(tv), L=np.int32(L), H=np.int32(H), z_idx=zi, z_val=zv,
            d_sae=np.int32(z.shape[1]), mvdf=np.array(m["mvdf"]), correlated=np.array(m["correlated"]))
        rep = dict(tag=tag, fvu=fvu, stored_fvu=m["sae_fvu"],
                   fvu_match=bool(abs(fvu - m["sae_fvu"]) < 1e-5), fingerprint=sd.fingerprint())
        json.dump(rep, open(os.path.join(CR, "arrays", f"{tag}_repro.json"), "w"), indent=2)
        print(rep, flush=True)


if __name__ == "__main__":
    main()
