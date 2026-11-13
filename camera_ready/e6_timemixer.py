"""E6: regenerate TimeMixer artifacts (reproduction-gated) and save per-window arrays.

Replays experiments/arch_gen_timemixer.run_one up to the MVDF selection (identical calls
and rng order), checks val MSE / FVU / MVDF against the stored run, and saves arrays to
runs/camera_ready/E6/arrays/<ds>_seed<s>.npz in the round-2 format.
  python -m camera_ready.e6_timemixer --datasets ETTh1 ETTh2 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CR = os.path.join(HERE, "runs", "camera_ready")
os.environ["KAGGLE_DATA_DIR"] = os.path.join(CR, "_no_kaggle_bundle")
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
from data.datasets import get_splits, make_windows  # noqa: E402
from experiments._seed import derive_seed  # noqa: E402
from experiments.arch_gen_timemixer import CFG  # noqa: E402
from forecasters.train_timemixer import TMTrainConfig, train_timemixer  # noqa: E402
from representation.layer_sweep import extract_all_taps  # noqa: E402
from sae.train_sae import train_sae, encode_all  # noqa: E402
from validation.errors import collect  # noqa: E402
from validation.correlation_screen import screen  # noqa: E402
from validation.importance_and_difficulty import compute_c_j, compute_delta_pred_single  # noqa: E402
from validation.dose_response import sweep_pool  # noqa: E402
from validation.difficulty_specificity import select_mvdf_v2  # noqa: E402
from shift.shift_windows import three_way_env_masks  # noqa: E402
from p126.regen_arrays import _sparse  # noqa: E402

OUT = os.path.join(CR, "E6", "arrays")


def regen(ds, seed):
    t0 = time.time()
    ds_seed = derive_seed(seed, ds)
    np.random.seed(ds_seed); torch.manual_seed(ds_seed)
    dev = "cpu"
    L, H, tvar = CFG["lookback"], CFG["horizon"], CFG["target_var"]
    tap, tm, sae_c = CFG["tap"], CFG["timemixer"], CFG["sae"]
    sd = get_splits(ds, max_channels=48)
    tv = tvar % len(sd.channels)
    Xtr, Ytr, _ = make_windows(sd.train, L, H)
    Xva, Yva, _ = make_windows(sd.val, L, H)
    Xte, Yte, end_te = make_windows(sd.test, L, H)
    tcfg = TMTrainConfig(lookback=L, horizon=H, d_model=tm["d_model"], n_scales=tm["n_scales"],
                         mix_hidden=tm["mix_hidden"], dropout=tm["dropout"], ma_kernel=tm["ma_kernel"],
                         lr=tm["lr"], batch_size=tm["batch_size"], max_epochs=tm["max_epochs"],
                         patience=tm["patience"], lr_decay=tm["lr_decay"],
                         warmup_epochs=tm["warmup_epochs"], seed=ds_seed, device=dev)
    model, val_mse = train_timemixer((Xtr, Ytr, None), (Xva, Yva, None),
                                     n_vars=len(sd.channels), cfg=tcfg, verbose=False)
    env = three_way_env_masks(end_te, len(sd.test))
    sel = np.where(env["E_select"])[0]
    Xsel, Ysel = Xte[sel], Yte[sel]
    cal_sel = collect(model, Xsel, Ysel, tv, device=dev)
    prof = cal_sel["step_profile"]
    tr_act = extract_all_taps(model, Xtr, tv, device=dev)[tap]
    sae = train_sae(tr_act, expansion=sae_c["expansion"], k=sae_c["k"], steps=sae_c["steps"],
                    batch_size=sae_c["batch_size"], lr=sae_c["lr"],
                    resample_every=sae_c["resample_every"],
                    resample_until_frac=sae_c["resample_until_frac"],
                    resample_cap=sae_c["resample_cap"], device=dev, seed=ds_seed, verbose=False)
    with torch.no_grad():
        xb = torch.from_numpy(sae.standardize(torch.from_numpy(tr_act.astype(np.float32))).numpy())
        xh, _ = sae(xb)
        fvu = float((torch.mean((xh - xb) ** 2) / (torch.var(xb) + 1e-12)).item())
    sel_act = extract_all_taps(model, Xsel, tv, device=dev)[tap]
    z_sel = encode_all(sae, sel_act, device=dev)
    scr = screen(z_sel, cal_sel["score"], min_active_frac=0.02)
    elig = np.where(scr["eligible"])[0]
    rng = np.random.default_rng(ds_seed)
    pool = np.sort(rng.choice(elig, size=min(CFG["candidate_pool_size"], len(elig)), replace=False))
    wsel = np.sort(rng.choice(len(Xsel), size=min(CFG["window_cap"], len(Xsel)), replace=False))
    # the original also calls these (no rng use; kept for identical state)
    compute_c_j(z_sel[:, pool], cal_sel["score"], active_only=True)
    compute_delta_pred_single(model, sae, Xsel[wsel], Ysel[wsel], sel_act[wsel], tv, tap, pool,
                              step_profile=prof, device=dev, loss="score")
    dose = sweep_pool(model, sae, Xsel[wsel], Ysel[wsel], sel_act[wsel], tv, tap, pool, prof,
                      lambdas=CFG["lambdas"], device=dev, verbose=False)
    mv = np.asarray(select_mvdf_v2(dose, u_threshold=0.0, k=12, residualize_method="isotonic")["selected"], int)
    stored = json.load(open(os.path.join(HERE, "runs/arch_gen_timemixer", f"{ds}_seed{seed}",
                                         "mvdf_selection.json")))
    agg = pd.read_csv(os.path.join(HERE, "runs/arch_gen_timemixer/AGG_arch_gen.csv"))
    r = agg[(agg.dataset == ds) & (agg.seed == seed)].iloc[0]
    rep = dict(tag=f"{ds}_seed{seed}", val_mse=float(val_mse), stored_val_mse=float(r.forecaster_val_mse),
               fvu=fvu, stored_fvu=float(r.sae_fvu),
               val_mse_match=bool(abs(val_mse - r.forecaster_val_mse) < 1e-6),
               fvu_match=bool(abs(fvu - r.sae_fvu) < 1e-6),
               mvdf=sorted(mv.tolist()), stored_mvdf=sorted(stored["MVDF_v2"]),
               mvdf_match=sorted(mv.tolist()) == sorted(stored["MVDF_v2"]),
               seconds=time.time() - t0)
    model.eval()
    with torch.no_grad():
        yh = np.concatenate([model(torch.from_numpy(Xte[i:i + 512])).numpy()
                             for i in range(0, len(Xte), 512)], 0)
    act = extract_all_taps(model, Xte, tv, device=dev)[tap]
    zi, zv = _sparse(encode_all(sae, act, device=dev))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"{ds}_seed{seed}.npz"),
                        X=Xte.astype(np.float32), y=Yte[:, :, tv].astype(np.float32),
                        yhat=yh[:, :, tv].astype(np.float32), hidden=act.astype(np.float32),
                        end_idx=end_te.astype(np.int32), origin_ts=np.full(len(end_te), -1, np.int64),
                        m_select=env["E_select"], m_id=env["E_eval_ID"], m_shift=env["E_eval_shift"],
                        target_var=np.int32(tv), L=np.int32(L), H=np.int32(H), z_idx=zi, z_val=zv,
                        d_sae=np.int32(2048), mvdf=np.array(stored["MVDF_v2"]),
                        correlated=np.array(stored["correlated"]))
    json.dump(dict(rep, mvdf_used="stored", sae_fvu=fvu, correlated=stored["correlated"],
                   enrichment_typeB=stored["enrichment_v2"].get("enrichment_ratio")),
              open(os.path.join(OUT, f"{ds}_seed{seed}_meta.json"), "w"), indent=2)
    print(rep, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    for ds in a.datasets:
        for s in a.seeds:
            if os.path.exists(os.path.join(OUT, f"{ds}_seed{s}.npz")):
                continue
            regen(ds, s)


if __name__ == "__main__":
    main()
