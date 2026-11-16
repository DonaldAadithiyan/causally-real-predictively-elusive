"""Round 2 WP6 post-processing per (dataset, seed) once arrays/weights exist:
step zero, WP1 tables, round-1 items A and C (C includes the scorer-cap diagnosis), WP3."""
from __future__ import annotations

from mechconf.paths import REPO_ROOT  # noqa: E402
import json
import os
import sys
import warnings

sys.path.insert(0, REPO_ROOT)
warnings.filterwarnings("ignore")
HERE0 = REPO_ROOT
R2 = os.path.join(HERE0, "results", "round2", "p126_round2")
os.environ.setdefault("P126_ARR", os.path.join(R2, "arrays"))

import pandas as pd  # noqa: E402
import experiments.round2.r2_wp6  # noqa: E402,F401  (dataset loader patch + KAGGLE_DATA_DIR)
from experiments.round2.common import dump  # noqa: E402

OUT = os.path.join(R2, "wp6")


def post(ds, seed, parts):
    tag = f"{ds}_seed{seed}"
    if "stepzero" in parts:
        from experiments.round2.r2_wp6 import stepzero
        dump(stepzero(ds, seed), os.path.join(OUT, "stepzero", f"{tag}_rule.json"))
        print(tag, "stepzero done", flush=True)
    if "wp1" in parts:
        from experiments.round2.r2_wp1 import run as wp1run
        a, b, g, l = wp1run(ds, seed)
        d = os.path.join(OUT, "wp1"); os.makedirs(d, exist_ok=True)
        for nm, x in (("continuous_scoring", a), ("standard_baselines", b),
                      ("regime_gating", g), ("localization", l)):
            pd.DataFrame(x).to_csv(os.path.join(d, f"{tag}_{nm}.csv"), index=False)
        print(tag, "wp1 done", flush=True)
    if "A" in parts:
        import experiments.round2.work_a as wa
        wa.AOUT = os.path.join(OUT, "A"); os.makedirs(wa.AOUT, exist_ok=True)
        summ, dec, rank, cv = wa.run(ds, seed)
        for nm, x in (("r2", dec), ("rank", rank), ("cv", cv)):
            pd.DataFrame(x).to_csv(os.path.join(wa.AOUT, f"{tag}_{nm}.csv"), index=False)
        dump(summ, os.path.join(wa.AOUT, f"{tag}_summary.json"))
        print(tag, "A done", flush=True)
    if "C" in parts:
        import experiments.round2.work_c as wc
        wc.COUT = os.path.join(OUT, "C"); os.makedirs(wc.COUT, exist_ok=True)
        o, rows = wc.run(ds, seed)
        dump(o, os.path.join(wc.COUT, f"{tag}_per_run.json"))
        pd.DataFrame(rows).to_csv(os.path.join(wc.COUT, f"{tag}_coverage_decomposition.csv"), index=False)
        print(tag, "C done", flush=True)
    if "wp3" in parts:
        from experiments.round2.r2_wp3 import main_real
        main_real([(ds, seed)])
        print(tag, "wp3 done", flush=True)


if __name__ == "__main__":
    parts = sys.argv[1].split(",")
    for spec in sys.argv[2:]:
        ds, s = spec.split(":")
        post(ds, int(s), parts)
