"""Round 2 WP2: synthetic control on the corrected path + dial experiment.

Settings are named synthR<ratio>P<p_stay>; the original frozen DGP is ratio 5.0, p 0.995
(its seeds 0-2 are `synth_seed<s>` from regen_arrays). The DGP config file is NOT edited:
overrides are applied to an in-memory copy (sigma_hard = ratio * sigma_easy, both
markov_p_stay = p). Everything else (mu_z, f, shift construction, forecaster/SAE/selection
settings) is unchanged. Outputs runs/p126_round2/wp2/ and arrays/weights under round2/."""
from __future__ import annotations

import copy
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
HERE0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R2 = os.path.join(HERE0, "runs", "p126_round2")
os.environ.setdefault("P126_ARR", os.path.join(R2, "arrays"))
torch.set_num_threads(4)

from p126.common import Run, HERE, dump  # noqa: E402
from p126.selection import frozen_selection, load_real_models  # noqa: E402
from data.synthetic_dgp import load_config, build_split_series, make_windows_xy  # noqa: E402
from forecasters.train_itransformer import TrainConfig, train_itransformer  # noqa: E402
from representation.layer_sweep import extract_all_taps  # noqa: E402
from sae.train_sae import train_sae, encode_all  # noqa: E402
from shift.shift_windows import three_way_env_masks  # noqa: E402
from p126.regen_arrays import _sparse, _predict  # noqa: E402

OUT = os.path.join(R2, "wp2")
ARR = os.path.join(R2, "arrays")
WTS = os.path.join(R2, "weights")


def synth_cfg(ratio=None, p=None):
    cfg = copy.deepcopy(load_config())
    if ratio is not None:
        cfg["dgp"]["sigma_hard"] = ratio * cfg["dgp"]["sigma_easy"]
    if p is not None:
        cfg["dgp"]["markov_p_stay_0"] = p
        cfg["dgp"]["markov_p_stay_1"] = p
    return cfg


def model_cfg(cfg):
    w = cfg["windowing"]
    return dict(lookback=w["lookback"], horizon=w["horizon"], forecaster=cfg["forecaster"],
                sae=cfg["sae"])


def name_of(ratio, p):
    return f"synthR{ratio}P{p}"


def make_setting(ratio, p, seed):
    """Train forecaster + SAE on the overridden DGP, run the frozen selection, save all."""
    ds = name_of(ratio, p)
    tag = f"{ds}_seed{seed}"
    cfg = synth_cfg(ratio, p)
    np.random.seed(seed); torch.manual_seed(seed)
    w, fcp, saep, drp = cfg["windowing"], cfg["forecaster"], cfg["sae"], cfg["dose_response"]
    L, H, tv = w["lookback"], w["horizon"], w["target_var"]
    S = build_split_series(cfg, seed)
    mean = S["train"]["x"].mean(); std = S["train"]["x"].std() + 1e-8
    ymean = S["train"]["y"].mean(); ystd = S["train"]["y"].std() + 1e-8
    nz = lambda sp: dict(x=(sp["x"] - mean) / std, y=(sp["y"] - ymean) / ystd, z=sp["z"])
    tr, va, te = nz(S["train"]), nz(S["val"]), nz(S["test"])
    Xtr, Ytr, *_ = make_windows_xy(tr["x"], tr["y"], tr["z"], L, H)
    Xva, Yva, *_ = make_windows_xy(va["x"], va["y"], va["z"], L, H)
    Xte, Yte, Zwin, Zhard, end_te = make_windows_xy(te["x"], te["y"], te["z"], L, H)
    tcfg = TrainConfig(lookback=L, horizon=H, d_model=fcp["d_model"], n_layers=fcp["n_layers"],
                       n_heads=fcp["n_heads"], d_ff=fcp["d_ff"], dropout=fcp["dropout"],
                       lr=fcp["lr"], batch_size=fcp["batch_size"], max_epochs=fcp["max_epochs"],
                       patience=fcp["patience"], lr_decay=fcp["lr_decay"],
                       warmup_epochs=fcp["warmup_epochs"], seed=seed, device="cpu")
    model, val_mse = train_itransformer((Xtr, Ytr, None), (Xva, Yva, None), n_vars=1,
                                        cfg=tcfg, verbose=False)
    tr_act = extract_all_taps(model, Xtr, tv)[fcp["tap"]]
    sae = train_sae(tr_act, expansion=saep["expansion"], k=saep["k"], steps=saep["steps"],
                    batch_size=saep["batch_size"], lr=saep["lr"],
                    resample_every=saep["resample_every"],
                    resample_until_frac=saep["resample_until_frac"],
                    resample_cap=saep["resample_cap"], device="cpu", seed=seed, verbose=False)
    yhat = _predict(model, Xte)
    act = extract_all_taps(model, Xte, tv)[fcp["tap"]]
    z = encode_all(sae, act)
    env = three_way_env_masks(end_te, len(te["x"]))
    starts = end_te - L + 1
    sig = np.array([cfg["dgp"]["sigma_easy"], cfg["dgp"]["sigma_hard"]]) / ystd
    zh = np.stack([te["z"][s + L:s + L + H] for s in starts]).astype(np.int8)
    # frozen selection (same rng order as run_seed: pool, wsel, then random set)
    sel = np.where(env["E_select"])[0]
    ae = np.abs(yhat[:, :, tv] - Yte[:, :, tv]).astype(np.float64)
    prof = ae[sel].mean(0); prof = prof / prof.mean()
    S_sel = (ae[sel] / prof[None]).max(1)
    fs = frozen_selection(model, sae, Xte[sel], Yte[sel], act[sel], S_sel, prof, tv, seed,
                          drp["candidate_pool_size"], drp["window_cap"], drp["feature_budget_k"])
    rand = np.sort(fs["rng"].choice(fs["elig"], size=min(drp["feature_budget_k"], len(fs["elig"])),
                                    replace=False))
    zi, zv = _sparse(z)
    torch.save(model.state_dict(), os.path.join(WTS, f"{tag}_forecaster.pt"))
    torch.save(sae.state_dict(), os.path.join(WTS, f"{tag}_sae.pt"))
    np.savez_compressed(
        os.path.join(ARR, f"{tag}.npz"),
        X=Xte.astype(np.float32), y=Yte[:, :, tv].astype(np.float32),
        yhat=yhat[:, :, tv].astype(np.float32), hidden=act.astype(np.float32),
        z_idx=zi, z_val=zv, d_sae=np.int32(z.shape[1]), end_idx=end_te.astype(np.int32),
        m_select=env["E_select"], m_id=env["E_eval_ID"], m_shift=env["E_eval_shift"],
        mvdf=fs["mvdf"], correlated=fs["corr"], random=rand,
        regime_win=Zwin, regime_hardfrac=Zhard, regime_h=zh, sigma_h=sig[zh].astype(np.float32),
        target_var=np.int32(tv), L=np.int32(L), H=np.int32(H))
    ft = fs["ftab"]
    meta = dict(ds=ds, seed=seed, ratio=ratio, p_stay=p, val_mse=float(val_mse),
                mvdf=fs["mvdf"].tolist(), correlated=fs["corr"].tolist(), random=rand.tolist(),
                enrichment_typeB=fs["enr"].get("enrichment_ratio"),
                enrichment_corr=fs["enr_corr"].get("enrichment_ratio"),
                type_counts=ft["counts"], n_gate1=int(fs["gate1"].sum()),
                hard_frac=S["hard_frac"])
    dump(meta, os.path.join(OUT, "settings", f"{tag}.json"))
    return meta


def verify_original(seed):
    """Original frozen DGP: re-run the frozen selection on the regenerated synth arrays +
    weights and compare with runs/phase1_synthetic/seed<s>/mvdf_selection.json."""
    cfg = synth_cfg()
    r = Run("synth", seed)
    m, sae = load_real_models(WTS, r.tag, 1, model_cfg(cfg))
    sel = r.idx_sel
    Yf = r.y[:, :, None]
    drp = cfg["dose_response"]
    fs = frozen_selection(m, sae, r.X[sel], Yf[sel], r.hidden[sel], r.S[sel], r.prof_sel, 0, seed,
                          drp["candidate_pool_size"], drp["window_cap"], drp["feature_budget_k"])
    saved = json.load(open(os.path.join(HERE, f"runs/phase1_synthetic/seed{seed}/mvdf_selection.json")))
    return dict(seed=seed, mvdf_match=sorted(fs["mvdf"].tolist()) == sorted(saved["MVDF_v2"]),
                corr_match=sorted(fs["corr"].tolist()) == sorted(saved["correlated"]),
                enrichment=fs["enr"].get("enrichment_ratio"),
                saved_enrichment=(saved["enrichment_v2"].get("enrichment_ratio")
                                  if isinstance(saved["enrichment_v2"], dict) else None))


def analyze(ds, seed):
    """Item A + item B machinery (round 1, unchanged) on one synthetic setting."""
    import p126.work_a as wa
    import p126.work_b as wb
    wa.AOUT = os.path.join(OUT, "A"); os.makedirs(wa.AOUT, exist_ok=True)
    wb.BOUT = os.path.join(OUT, "B"); os.makedirs(wb.BOUT, exist_ok=True)
    summ, dec, rank, cv = wa.run(ds, seed)
    rowsB, metaB = wb.run(ds, seed, embargo=True)
    tag = f"{ds}_seed{seed}"
    pd.DataFrame(dec).to_csv(os.path.join(wa.AOUT, f"{tag}_r2.csv"), index=False)
    pd.DataFrame(rank).to_csv(os.path.join(wa.AOUT, f"{tag}_rank.csv"), index=False)
    pd.DataFrame(cv).to_csv(os.path.join(wa.AOUT, f"{tag}_cv.csv"), index=False)
    pd.DataFrame(rowsB).to_csv(os.path.join(wb.BOUT, f"{tag}_B.csv"), index=False)
    dump(dict(summary=summ, metaB=metaB), os.path.join(wa.AOUT, f"{tag}_summary.json"))
    return summ, dec, rank, cv, rowsB, metaB


def link(ds, seed, n_random=20):
    """Downstream link on the corrected path: learned scorers vs naive bands, P1/P2/P3."""
    from p126.r2_wp1 import method_bases, eval_set
    from p126.corrected import chrono_split, random_block_split
    from p126.work_a import feature_sets
    r = Run(ds, seed)
    H, L = r.H, r.L
    emb = L + H
    id_ix = r.idx_id; n = len(id_ix)
    f, c, t = chrono_split(n, emb=emb)
    fit, cal, te_id = id_ix[f], id_ix[c], id_ix[t]
    fs = feature_sets(r, fit, seed)
    bases = method_bases(r, fs, fit, cal, seed)
    rows = []
    for seg, te in (("ID", te_id), ("shift", r.idx_sh)):
        for kind, proto in (("static", "P1_chrono_static"), ("aci", "P2_chrono_delayed_aci")):
            for m, d in eval_set(bases, r.S[te], te, kind, H).items():
                rows.append(dict(dataset=ds, seed=seed, protocol=proto, segment=seg, method=m,
                                 n_test=len(te), n_blocks=len(te) / (2 * H), **d))
    rng = np.random.default_rng(seed)
    acc = []
    for rep in range(n_random):
        f, c, t = random_block_split(n, rng, emb=emb)
        if len(t) < 50 or len(c) < 150:
            continue
        fit_r, cal_r, te_r = id_ix[f], id_ix[c], id_ix[t]
        b_r = method_bases(r, feature_sets(r, fit_r, seed), fit_r, cal_r, seed + rep)
        for m, d in eval_set(b_r, r.S[te_r], te_r, "static", H).items():
            acc.append(dict(rep=rep, method=m, **d))
    a = pd.DataFrame(acc)
    for m, g in a.groupby("method"):
        rows.append(dict(dataset=ds, seed=seed, protocol="P3_random_blocks", segment="ID", method=m,
                         cov=g["cov"].mean(), width=g["width"].mean(),
                         red_vs_naive_matched=g["red_vs_naive_matched"].mean(),
                         red_q05=g["red_vs_naive_matched"].quantile(.05),
                         red_q95=g["red_vs_naive_matched"].quantile(.95), reps=len(g)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"link_{ds}_seed{seed}.csv"), index=False)
    return df


def wp3_synth(ds, seed):
    from p126.r2_wp3 import run_wp3
    cfg = synth_cfg()
    r = Run(ds, seed)
    m, sae = load_real_models(WTS, r.tag, 1, model_cfg(cfg))
    drp = cfg["dose_response"]
    saved = list(r.mvdf)
    return run_wp3(r, m, sae, seed, drp["candidate_pool_size"], drp["window_cap"],
                   drp["feature_budget_k"], saved, os.path.join(R2, "wp3", r.tag))


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "settings"), exist_ok=True)
    os.makedirs(WTS, exist_ok=True)
    what = sys.argv[1]
    if what == "verify":
        res = [verify_original(s) for s in (0, 1, 2)]
        dump(res, os.path.join(OUT, "original_selection_reproduction.json")); print(res)
    elif what == "make":
        for spec in sys.argv[2:]:
            ratio, p, seed = spec.split(":")
            t0 = time.time()
            print(make_setting(float(ratio), float(p), int(seed)), flush=True)
            print("  made", spec, f"{time.time()-t0:.0f}s", flush=True)
    elif what == "link":
        for spec in sys.argv[2:]:
            ds, seed = spec.split(":"); print(link(ds, int(seed)).round(3).to_string(), flush=True)
    elif what == "wp3":
        for spec in sys.argv[2:]:
            ds, seed = spec.split(":"); print(wp3_synth(ds, int(seed)), flush=True)
    elif what == "analyze_light":
        # dial settings: MVDF is the only representation analysed (true ceiling, oracle,
        # forecast-time predictors and MVDF rank metrics are what the dial needs)
        import p126.work_a as wa
        _fs = wa.feature_sets
        wa.feature_sets = lambda r, fit_ix, seed: {"mvdf": _fs(r, fit_ix, seed)["mvdf"]}
        for spec in sys.argv[2:]:
            ds, seed = spec.split(":")
            t0 = time.time()
            analyze(ds, int(seed))
            print("  analyzed(light)", spec, f"{time.time()-t0:.0f}s", flush=True)
    elif what == "analyze":
        for spec in sys.argv[2:]:
            ds, seed = spec.split(":")
            t0 = time.time()
            analyze(ds, int(seed))
            print("  analyzed", spec, f"{time.time()-t0:.0f}s", flush=True)
