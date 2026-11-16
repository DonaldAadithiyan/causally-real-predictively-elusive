"""One-off, mechanical migration from the working layout to the public-repo layout.

Kept in the repository so the restructure is auditable. It (1) moves files with `git mv`
(history follows them), (2) rewrites import paths, (3) replaces __file__-relative repo-root
computations with `mechconf.paths.REPO_ROOT`, and (4) rewrites relative result-directory
strings (`runs/...` -> `results/...`). It changes no experiment logic. Run once from the repo
root: `python scripts/dev/migrate_layout.py`.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.getcwd()


def sh(*a):
    subprocess.run(a, check=True)


def mv(src, dst):
    if not os.path.exists(src):
        return
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    sh("git", "mv", src, dst)


# ------------------------------------------------------------------ 1. moves
PKG = "src/mechconf"
for sub in ("data", "forecasters", "sae", "representation", "validation", "calibration",
            "baselines", "eval"):
    for f in sorted(os.listdir(sub)):
        if f.endswith(".py"):
            mv(f"{sub}/{f}", f"{PKG}/{sub}/{f}")
mv("shift/shift_windows.py", f"{PKG}/data/shift_windows.py")
if os.path.exists("shift/__init__.py"):
    sh("git", "rm", "-q", "shift/__init__.py")
if os.path.exists("__init__.py"):
    sh("git", "rm", "-q", "__init__.py")

V1 = {"mve_synthetic.py", "mve_real.py", "full_suite.py"}
for f in sorted(os.listdir("experiments")):
    if f.endswith(".py") and f not in ("__init__.py", "_seed.py"):
        mv(f"experiments/{f}", f"experiments/{'v1_mve' if f in V1 else 'v2'}/{f}")
for f in sorted(os.listdir("p126")):
    if f.endswith(".py"):
        mv(f"p126/{f}", f"experiments/round2/{f}")
for f in sorted(os.listdir("camera_ready")):
    if f.endswith(".py"):
        mv(f"camera_ready/{f}", f"experiments/camera_ready/{f}")

RES = {"runs/camera_ready": "results/camera_ready",
       "runs/p126_round2": "results/round2/p126_round2",
       "runs/p126": "results/round2/p126",
       "mve_runs": "results/v1/mve_runs",
       "v2_runs": "results/v2/v2_runs",
       "v2_confirm": "results/v2/v2_confirm",
       "v2_h6_topup": "results/v2/v2_h6_topup",
       "figures": "results/v2/figures"}
for s, d in RES.items():
    mv(s, d)
for f in sorted(os.listdir("runs")):
    mv(f"runs/{f}", f"results/v2/{f}")
for f in sorted(os.listdir(".")):
    if f.startswith("runs_") and f.endswith(".log"):
        mv(f, f"results/v2/logs/{f}")
for f in ("BASELINE_RESULT.md", "FINISHING_TASKS_STATUS.md", "H7A_RESULT.md", "NOTES.md",
          "PAPER_FRAME.md", "PROJECT_REPORT.md", "RESULTS_FINAL.md", "SLCP_RESULT.md",
          "V2_CONFIRM_REPORT.md", "V2_FINAL_REPORT.md", "V2_PHASES012_REPORT.md"):
    mv(f, f"docs/reports/{f}")
mv("README.md", "docs/reports/README_working_repo.md")
for d in ("experiments/v1_mve", "experiments/v2", "experiments/round2", "experiments/camera_ready",
          PKG, *(f"{PKG}/{s}" for s in ("data", "forecasters", "sae", "representation",
                                        "validation", "calibration", "baselines", "eval"))):
    p = os.path.join(d, "__init__.py")
    if not os.path.exists(p):
        open(p, "w").close()

# ------------------------------------------------------------------ 2-4. rewrites
MODS = ["data", "forecasters", "sae", "representation", "validation", "calibration",
        "baselines", "eval"]
EXP_V1 = [f[:-3] for f in V1]
EXP_V2 = sorted(f[:-3] for f in os.listdir("experiments/v2") if f.endswith(".py") and f != "__init__.py")


def rewrite_imports(src):
    src = re.sub(r"(?m)^(\s*(?:from|import)\s+)shift\.shift_windows\b", r"\1mechconf.data.shift_windows", src)
    for m in MODS:
        src = re.sub(rf"(?m)^(\s*(?:from|import)\s+){m}(\.|\s)", rf"\1mechconf.{m}\2", src)
    for m in EXP_V1:
        src = re.sub(rf"\bexperiments\.{m}\b", f"experiments.v1_mve.{m}", src)
    for m in EXP_V2:
        src = re.sub(rf"\bexperiments\.{m}\b", f"experiments.v2.{m}", src)
    src = re.sub(r"(?m)^(\s*(?:from|import)\s+)p126(\.|\s)", r"\1experiments.round2\2", src)
    src = re.sub(r"(?m)^(\s*(?:from|import)\s+)camera_ready(\.|\s)", r"\1experiments.camera_ready\2", src)
    # `python -m camera_ready.x` / `-m p126.x` / `-m experiments.x` in docstrings and messages
    src = re.sub(r"-m camera_ready\.", "-m experiments.camera_ready.", src)
    src = re.sub(r"-m p126\.", "-m experiments.round2.", src)
    return src


ROOT_EXPR = r"os\.path\.dirname\(\s*os\.path\.dirname\(\s*os\.path\.abspath\(__file__\)\s*\)\s*\)"


def rewrite_root(src, path):
    if not re.search(ROOT_EXPR, src) and "_HERE" not in src:
        return src
    src2 = re.sub(ROOT_EXPR, "REPO_ROOT", src)
    if path.endswith("mechconf/data/datasets.py"):
        src2 = src2.replace('RAW_DIR = os.path.join(_HERE, "raw")', 'RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")')
    if path.endswith("mechconf/data/synthetic_dgp.py"):
        src2 = src2.replace('os.path.join(_HERE, "..", "configs", "synthetic_dgp_config.json")',
                            'os.path.join(REPO_ROOT, "configs", "synthetic_dgp_config.json")')
    if path.endswith("mechconf/data/fingerprint.py"):
        src2 = src2.replace('os.path.join(_HERE, "..", "configs", "data_fingerprints.json")',
                            'os.path.join(REPO_ROOT, "configs", "data_fingerprints.json")')
    if src2 != src and "from mechconf.paths import REPO_ROOT" not in src2:
        # insert after the __future__ import (or the module docstring) so it is a top-level import
        m = re.search(r"(?m)^from __future__ import annotations\s*\n", src2)
        if m:
            i = m.end()
        else:
            m = re.match(r'\s*(""".*?"""|\'\'\'.*?\'\'\')\s*\n', src2, re.S)
            i = m.end() if m else 0
        src2 = src2[:i] + "from mechconf.paths import REPO_ROOT  # noqa: E402\n" + src2[i:]
    return src2


def rewrite_results(src):
    # path-join tokens
    src = re.sub(r"""(["'])runs\1(\s*,\s*)(["'])camera_ready\3""", r'\1results\1\2\3camera_ready\3', src)
    src = re.sub(r"""(["'])runs\1(\s*,\s*)(["'])p126_round2\3""", r'\1results\1\2\1round2\1\2\3p126_round2\3', src)
    src = re.sub(r"""(["'])runs\1(\s*,\s*)(["'])p126\3""", r'\1results\1\2\1round2\1\2\3p126\3', src)
    src = re.sub(r"""(["'])runs\1(\s*,)""", r'\1results\1\2 \1v2\1,', src)
    src = re.sub(r"""(["'])runs\1(\s*\))""", r'\1results\1, \1v2\1\2', src)
    # slash paths inside strings / docstrings
    src = re.sub(r"\bruns/camera_ready", "results/camera_ready", src)
    src = re.sub(r"\bruns/p126_round2", "results/round2/p126_round2", src)
    src = re.sub(r"\bruns/p126\b", "results/round2/p126", src)
    src = re.sub(r"\bruns/(?=[A-Za-z_{])", "results/v2/", src)
    for d, n in (("v2_runs", "results/v2/v2_runs"), ("v2_confirm", "results/v2/v2_confirm"),
                 ("v2_h6_topup", "results/v2/v2_h6_topup"), ("mve_runs", "results/v1/mve_runs")):
        src = re.sub(rf"""(["']){d}(?=[/"'])""", rf"\1{n}", src)
    return src


changed = 0
for dp, dn, fn in os.walk(ROOT):
    if "/.git" in dp or "/scripts/dev" in dp:
        continue
    for f in fn:
        if not f.endswith(".py"):
            continue
        p = os.path.join(dp, f)
        src = open(p).read()
        new = rewrite_results(rewrite_root(rewrite_imports(src), p))
        if new != src:
            open(p, "w").write(new)
            changed += 1
for p in [os.path.join("configs", f) for f in os.listdir("configs") if f.endswith((".yaml", ".yml"))]:
    s = open(p).read(); n = rewrite_results(s)
    if n != s:
        open(p, "w").write(n); changed += 1
print(f"rewrote {changed} files")
