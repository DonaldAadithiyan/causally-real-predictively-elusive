"""Data provenance guard.

Added after a Phase 2 discrepancy: a local re-run loaded ETTh1 from the HTTP CSV
(8640 train rows) instead of the Kaggle `ETTh.csv` bundle split (17420 rows) — same
seed, totally different `I` values, because the whole chain (forecaster -> SAE ->
features) is downstream of the data. This turns "we noticed and traced it" into "the
pipeline refuses to run on the wrong data".

Design: channels are checked EXACTLY against `CANONICAL`. Row counts and an edge-hash
are recorded to `configs/data_fingerprints.json` on the first canonical run and then
compared verbatim on every subsequent run. So the guard is self-calibrating from the
first real Kaggle run and does not depend on hand-entered row counts.
"""
from __future__ import annotations

from mechconf.paths import REPO_ROOT  # noqa: E402
import hashlib
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_MANIFEST = os.path.join(REPO_ROOT, "configs", "data_fingerprints.json")

# channels after data/datasets.py adapters + max_channels=48.
# ETT: exact (7). Capped datasets: `get_splits` keeps the (max_channels-1) most
# variable channels PLUS the target, so the final count is max_channels if the target
# was outside the top-(k-1), else max_channels-1. Allow +/-1 for those.
CANONICAL_CHANNELS = {
    "ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7,
    "electricity": 48, "weather": 21, "traffic": 48, "solar": 48,
}
CHANNEL_EXACT = {"ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather"}  # rest allow +/-1
ROW_TOL_FRAC = 0.02   # +/- 2% on total rows once a manifest exists


def data_fingerprint(name: str, splitdata) -> dict:
    n_rows = len(splitdata.raw_train) + len(splitdata.raw_val) + len(splitdata.raw_test)
    h = hashlib.sha256()
    for a in (splitdata.raw_train[:64], splitdata.raw_test[-64:]):
        h.update(a.tobytes())
    return dict(
        dataset=name, rows=int(n_rows), channels=len(splitdata.channels),
        train_rows=int(len(splitdata.raw_train)), val_rows=int(len(splitdata.raw_val)),
        test_rows=int(len(splitdata.raw_test)), edge_hash=h.hexdigest()[:16],
    )


def _load_manifest() -> dict:
    try:
        return json.load(open(os.path.normpath(_MANIFEST)))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_manifest(m: dict) -> None:
    os.makedirs(os.path.dirname(os.path.normpath(_MANIFEST)), exist_ok=True)
    json.dump(m, open(os.path.normpath(_MANIFEST), "w"), indent=2, sort_keys=True)


def assert_expected_data(name: str, splitdata, strict: bool = True, record: bool = True) -> dict:
    """Check `name`'s resolved shape. Channels must match CANONICAL_CHANNELS exactly.
    Rows/edge-hash are compared to configs/data_fingerprints.json if present, else
    recorded there (first canonical run). Returns the fingerprint for the run log."""
    fp = data_fingerprint(name, splitdata)

    exp_ch = CANONICAL_CHANNELS.get(name)
    if exp_ch is not None:
        tol = 0 if name in CHANNEL_EXACT else 1
        if abs(fp["channels"] - exp_ch) > tol:
            msg = (f"DATA MISMATCH {name!r}: {fp['channels']} channels, expected "
                   f"{exp_ch}+/-{tol} (uom220001c bundle). Wrong Kaggle input, or an "
                   f"adapter change.")
            if strict:
                raise RuntimeError(msg)
            print("[fingerprint] WARNING: " + msg)
            return fp

    man = _load_manifest()
    if name in man:
        prev = man[name]
        row_ok = abs(fp["rows"] - prev["rows"]) <= ROW_TOL_FRAC * prev["rows"]
        hash_ok = fp["edge_hash"] == prev["edge_hash"]
        if row_ok and hash_ok:
            print(f"[fingerprint] {name}: OK ({fp['rows']} rows, {fp['channels']} ch, "
                  f"hash {fp['edge_hash']})")
            return fp
        msg = (f"DATA MISMATCH {name!r}: {fp['rows']} rows / hash {fp['edge_hash']} vs "
               f"recorded {prev['rows']} rows / hash {prev['edge_hash']} "
               f"(configs/data_fingerprints.json). Different data source -> non-comparable "
               f"results. Fix the Kaggle input, or delete the manifest entry if the change "
               f"is intentional.")
        if strict:
            raise RuntimeError(msg)
        print("[fingerprint] WARNING: " + msg)
        return fp

    # first sighting: record it
    if record:
        man[name] = dict(rows=fp["rows"], channels=fp["channels"],
                         train_rows=fp["train_rows"], val_rows=fp["val_rows"],
                         test_rows=fp["test_rows"], edge_hash=fp["edge_hash"])
        _save_manifest(man)
        print(f"[fingerprint] {name}: recorded canonical fingerprint {fp['edge_hash']} "
              f"({fp['rows']} rows, {fp['channels']} ch)")
    return fp
