"""Download the large artifacts (weights, per-window arrays, per-window parquet) from the GitHub
release and verify every file against artifacts/MANIFEST.sha256.

They are needed only to re-run analyses (E3-G, Phase 3/4 from arrays, dose-response). Regenerating
the paper's tables, figures and numbers needs only the committed CSVs.

    python scripts/fetch_artifacts.py                     # download all bundles, extract, verify
    python scripts/fetch_artifacts.py --bundles camera_ready
    python scripts/fetch_artifacts.py --verify            # verify files already in place
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get("ARTIFACT_REPO", "DonaldAadithiyan/causally-real-predictively-elusive")
TAG = os.environ.get("ARTIFACT_TAG", "v1.0-camera-ready")
BUNDLES = {"camera_ready": "artifacts_camera_ready.tar",          # results/camera_ready/**   ~0.62 GB
           "round2": "artifacts_round2_p126_round2.tar",          # results/round2/p126_round2/**  ~1.22 GB
           "round1": "artifacts_round2_p126.tar"}                 # results/round2/p126/**    ~0.16 GB


def manifest():
    rows = []
    for line in open(os.path.join(ROOT, "artifacts", "MANIFEST.sha256")):
        if line.startswith("#") or not line.strip():
            continue
        h, size, path = line.split(maxsplit=2)
        rows.append((h, int(size), path.strip()))
    return rows


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify(prefixes=None):
    bad = missing = ok = 0
    for h, size, path in manifest():
        if prefixes and not any(path.startswith(p) for p in prefixes):
            continue
        p = os.path.join(ROOT, path)
        if not os.path.exists(p):
            missing += 1
        elif sha256(p) != h:
            bad += 1; print(f"[MISMATCH] {path}")
        else:
            ok += 1
    print(f"[verify] ok {ok}, missing {missing}, mismatched {bad}")
    return bad == 0 and missing == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", nargs="+", default=list(BUNDLES), choices=list(BUNDLES))
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    prefixes = {"camera_ready": "results/camera_ready/", "round2": "results/round2/p126_round2/",
                "round1": "results/round2/p126/"}
    sel = [prefixes[b] for b in a.bundles]
    if not a.verify:
        for b in a.bundles:
            url = f"https://github.com/{REPO}/releases/download/{TAG}/{BUNDLES[b]}"
            dst = os.path.join(ROOT, "artifacts", BUNDLES[b])
            print(f"[download] {url}")
            urllib.request.urlretrieve(url, dst)
            with tarfile.open(dst) as t:
                members = [m for m in t.getmembers() if m.name.startswith("results/") and ".." not in m.name]
                t.extractall(ROOT, members=members)
            os.remove(dst)
    sys.exit(0 if verify(sel) else 1)


if __name__ == "__main__":
    main()
