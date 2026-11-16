"""End-to-end smoke run (tiny config, CPU). Needs data/raw/ETTh1.csv (scripts/download_data.py)."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from mechconf.paths import REPO_ROOT


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO_ROOT, "data", "raw", "ETTh1.csv")),
                    reason="run scripts/download_data.py first")
def test_smoke(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "CR_ARR_DIR"}
    p = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "smoke_test.py"), "--out", str(tmp_path)],
                       cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "[smoke] PASS" in p.stdout
