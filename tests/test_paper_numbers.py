"""Every number printed in the paper must be recomputable from the committed results/."""
from __future__ import annotations

import os
import re

import pytest
import yaml

from mechconf.paths import REPO_ROOT
from paper import paper_numbers as pn

ENTRIES = [e for grp in yaml.safe_load(open(os.path.join(REPO_ROOT, "paper", "paper_numbers.yaml"))).values()
           for e in grp]


def norm(s: str) -> str:
    s = str(s)
    for a, b in (("\\%", "%"), ("$", ""), ("\\", ""), ("{", ""), ("}", ""), ("−", "-"), ("–", "-"), ("--", "-")):
        s = s.replace(a, b)
    return re.sub(r"\s+", "", s)


@pytest.mark.parametrize("e", ENTRIES, ids=[e["id"] for e in ENTRIES])
def test_paper_number(e):
    assert e["fn"] in pn.REGISTRY, f"no function {e['fn']}"
    got = pn.REGISTRY[e["fn"]](*e.get("args", []))
    want = e.get("expect_numeric", e["printed"])
    if "known_mismatch" in e:
        pytest.xfail(f"KNOWN MISMATCH (printed {e['printed']!r}, computed {got!r}): {e['known_mismatch']}") \
            if norm(got) != norm(want) else pytest.fail(f"known_mismatch resolved for {e['id']}; update the YAML")
    assert norm(got) == norm(want), f"printed {e['printed']!r} but results give {got!r} ({e['where']})"


def test_config_constants():
    from experiments.v2.phase3_real_predictability import CFG
    from experiments.camera_ready.common import ALPHA, BLOCK
    from mechconf.validation.dose_response import LAMBDAS
    fc, sae = CFG["forecaster"], CFG["sae"]
    assert (fc["d_model"], fc["n_layers"], CFG["lookback"], CFG["horizon"]) == (256, 3, 96, 96)
    assert 256 * sae["expansion"] == 2048 and sae["k"] == 64
    assert LAMBDAS == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    assert CFG["candidate_pool_size"] == 120 and CFG["feature_budget_k"] == 12
    assert ALPHA == 0.10 and BLOCK == 96
