"""Real time-series benchmark datasets.

Two supported sources:

1. ETT over HTTP  — the canonical zhouhaoyi/ETDataset mirror gives
   ETTh1/ETTh2/ETTm1/ETTm2.csv in the standard `date + 7 channels` layout.

2. A Kaggle-attached bundle  — the "unified multivariate TS forecasting" bundle
   (uom220001c / limpidcloud style) ships raw / concatenated files with
   non-standard names and formats:
       ETTh.csv            14 cols  = ETTh1 (cols 0-6) + ETTh2 (cols 7-13), hourly
       ETTm.csv            14 cols  = ETTm1 + ETTm2, 15-min
       LD2011_2014.csv     raw UCI electricity, 370 clients, 15-min, ';'-delim, ',' decimal
       mpi_roof.csv        raw Jena/Beutenberg weather, ~20 vars, 10-min
       PeMS.csv            traffic, 862 sensors, hourly
       solar_Alabama.csv   137 plants, 5-min
       exchange.csv        8 series, daily
       QPS.csv             10 apps, 1-min
   Per-file adapters below normalize each to a plain (T, C) float matrix and a
   1-hour resample where the native interval is finer (matches the bundle's own
   "unified interval" protocol). Splits then follow the standard LTSF scheme
   (ETT: 12/4/4 months; everything else 70/10/20 by time). These will NOT be
   byte-identical to FFDCI's published splits, which is acceptable for the MVE —
   what matters is a real forecaster on real regime-structured data.

Standard protocol applied after adaptation:
  - split by TIME, no shuffling
  - z-score normalization fit on TRAIN ONLY
  - sliding windows: input length L, predict next H
"""
from __future__ import annotations

from mechconf.paths import REPO_ROOT  # noqa: E402
import dataclasses
import hashlib
import os
import urllib.request
from typing import Callable, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")

_ETT_BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
DATASET_URLS = {
    "ETTh1": _ETT_BASE + "ETTh1.csv",
    "ETTh2": _ETT_BASE + "ETTh2.csv",
    "ETTm1": _ETT_BASE + "ETTm1.csv",
    "ETTm2": _ETT_BASE + "ETTm2.csv",
}

# Where Kaggle mounts attached datasets.
KAGGLE_DATA_DIR = os.environ.get("KAGGLE_DATA_DIR", "/kaggle/input")

_ETT_MONTHS = {"ETTh1": 24, "ETTh2": 24, "ETTm1": 24 * 4, "ETTm2": 24 * 4}


# ----------------------------------------------------------------------------------
# file discovery
# ----------------------------------------------------------------------------------
def _find_file(*basenames: str, search_raw: bool = True) -> Optional[str]:
    """First existing file matching any of `basenames` (case-insensitive). Kaggle
    input is searched before the local RAW_DIR cache so an attached bundle always
    wins over a stale materialized copy."""
    wanted = {b.lower() for b in basenames}
    bases = [KAGGLE_DATA_DIR] + ([RAW_DIR] if search_raw else [])
    for base in bases:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if f.lower() in wanted:
                    return os.path.join(root, f)
    return None


# ----------------------------------------------------------------------------------
# per-dataset raw -> (DataFrame indexed by datetime, list[str] channels)
# each returns hourly-resampled data
# ----------------------------------------------------------------------------------
def _load_ett_split(which: str) -> pd.DataFrame:
    """which in {ETTh1,ETTh2,ETTm1,ETTm2}. Prefer the canonical single file; if only
    the bundle's concatenated ETTh.csv / ETTm.csv is available, split it."""
    concat_name = "ETTh.csv" if which.startswith("ETTh") else "ETTm.csv"
    concat_p = _find_file(concat_name, search_raw=False)          # bundle only

    # canonical single file: accept from Kaggle input always; from RAW_DIR only when
    # no bundle concat is present (so an attached bundle wins over a stale cache).
    single_kaggle = _find_file(f"{which}.csv", search_raw=False)
    single_cache = _find_file(f"{which}.csv") if concat_p is None else None
    p = single_kaggle or single_cache
    if p is not None:
        df = pd.read_csv(p)
        return df.rename(columns={df.columns[0]: "date"})

    p = concat_p
    if p is None:
        raise FileNotFoundError(f"{which}: neither {which}.csv nor {concat_name} found")
    df = pd.read_csv(p)
    first = df.columns[0]
    has_date = first.lower() in ("date", "time")
    cols = list(df.columns[1:]) if has_date else list(df.columns)
    if len(cols) < 14:
        # already a single 7-channel table mislabelled -> just use it
        if has_date:
            return df.rename(columns={first: "date"})
        df.insert(0, "date", pd.RangeIndex(len(df)))
        return df
    block = cols[:7] if which.endswith("1") else cols[7:14]
    out = df[block].copy()
    out.columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    out.insert(0, "date", df[first] if has_date else pd.RangeIndex(len(df)))
    return out


def _sniff_and_read(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Read a CSV whose delimiter (',' ';' '\\t') and decimal (',' '.') are unknown.
    Picks the (sep, decimal) combo that yields the most numeric columns."""
    best, best_score = None, -1
    for sep in (";", ",", "\t"):
        for dec in (",", "."):
            if sep == dec:
                continue
            try:
                df = pd.read_csv(path, sep=sep, decimal=dec, nrows=nrows or 2000,
                                 engine="python")
            except Exception:
                continue
            if df.shape[1] < 2:
                continue
            num = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
            score = (num.notna().mean().mean()) * df.shape[1]
            if score > best_score:
                best, best_score, best_kw = df, score, dict(sep=sep, decimal=dec)
    if best is None:
        return pd.read_csv(path, nrows=nrows)
    if nrows is None:
        return pd.read_csv(path, engine="python", **best_kw)
    return best


def _load_electricity() -> pd.DataFrame:
    """Standard preprocessed electricity.csv if present; else raw LD2011_2014 dump
    (delimiter/decimal sniffed), 15-min, ~370 clients, resampled to hourly means."""
    p = _find_file("electricity.csv")
    if p is not None:
        df = pd.read_csv(p)
        return df.rename(columns={df.columns[0]: "date"})

    p = _find_file("LD2011_2014.csv", "LD2011_2014.txt")
    if p is None:
        raise FileNotFoundError("electricity: no electricity.csv or LD2011_2014.*")

    df = _sniff_and_read(p)
    tcol = df.columns[0]
    ts = pd.to_datetime(df[tcol].astype(str), errors="coerce")
    ok = ts.notna()
    if ok.mean() < 0.5:
        raise ValueError(
            f"electricity: could not parse timestamps in {tcol!r}; "
            f"sample values {df[tcol].head(3).tolist()}"
        )
    vals = df.loc[ok].drop(columns=[tcol]).apply(pd.to_numeric, errors="coerce")
    vals = vals.loc[:, vals.notna().mean() > 0.5].astype("float32")
    if vals.shape[1] == 0:
        raise ValueError("electricity: no numeric client columns after parse")
    vals.index = pd.DatetimeIndex(ts[ok].values)
    hourly = vals.resample("1h").mean().ffill().bfill()
    hourly.index.name = "date"
    hourly = hourly.reset_index()
    hourly.columns = ["date"] + [f"c{i}" for i in range(hourly.shape[1] - 1)]
    if len(hourly) > 35000:                       # keep the most recent ~4 years hourly
        hourly = hourly.iloc[-35000:].reset_index(drop=True)
    print(f"[data] electricity raw parse -> {len(hourly)} hourly rows, "
          f"{hourly.shape[1]-1} clients")
    return hourly


def _load_weather() -> pd.DataFrame:
    """Standard weather.csv if present; else raw mpi_roof.csv (Jena, 10-min, ~20 vars)
    resampled to hourly means."""
    p = _find_file("weather.csv")
    if p is not None:
        df = pd.read_csv(p)
        return df.rename(columns={df.columns[0]: "date"})

    p = _find_file("mpi_roof.csv", "mpi_saale.csv", "jena_climate.csv")
    if p is None:
        raise FileNotFoundError("weather: no weather.csv or mpi_roof.csv")
    df = pd.read_csv(p)
    tcol = df.columns[0]
    dt = pd.to_datetime(df[tcol], errors="coerce", dayfirst=True)
    if dt.isna().mean() > 0.5:                       # try month-first / ISO
        dt = pd.to_datetime(df[tcol], errors="coerce")
    df = df.assign(**{tcol: dt}).dropna(subset=[tcol]).set_index(tcol)
    num = df.apply(pd.to_numeric, errors="coerce").astype("float32")
    hourly = num.resample("1h").mean().ffill().bfill().reset_index()
    hourly = hourly.rename(columns={hourly.columns[0]: "date"})
    return hourly


def _load_traffic() -> pd.DataFrame:
    p = _find_file("traffic.csv")
    if p is not None:
        df = pd.read_csv(p)
        return df.rename(columns={df.columns[0]: "date"})
    p = _find_file("PeMS.csv", "PEMS.csv", "pems.csv")
    if p is None:
        raise FileNotFoundError("traffic: no traffic.csv or PeMS.csv")
    df = pd.read_csv(p)
    first = df.columns[0]
    if first.lower() in ("date", "time"):
        return df.rename(columns={first: "date"})
    df.insert(0, "date", pd.RangeIndex(len(df)))
    return df


def _load_generic(*names: str) -> Callable[[], pd.DataFrame]:
    def _inner():
        p = _find_file(*names)
        if p is None:
            raise FileNotFoundError(f"none of {names} found")
        df = pd.read_csv(p)
        first = df.columns[0]
        if first.lower() in ("date", "time"):
            return df.rename(columns={first: "date"})
        return pd.concat([pd.Series(pd.RangeIndex(len(df)), name="date"), df], axis=1)
    return _inner


LOADERS: dict[str, Callable[[], pd.DataFrame]] = {
    "ETTh1": lambda: _load_ett_split("ETTh1"),
    "ETTh2": lambda: _load_ett_split("ETTh2"),
    "ETTm1": lambda: _load_ett_split("ETTm1"),
    "ETTm2": lambda: _load_ett_split("ETTm2"),
    "electricity": _load_electricity,
    "weather": _load_weather,
    "traffic": _load_traffic,
    "solar": _load_generic("solar.csv", "solar_Alabama.csv", "solar_AL.csv"),
    "exchange": _load_generic("exchange.csv", "exchange_rate.csv"),
    "QPS": _load_generic("QPS.csv"),
}


# ----------------------------------------------------------------------------------
def download(name: str, force: bool = False) -> str:
    """Materialize `<name>.csv` in RAW_DIR (from HTTP for ETT, else via LOADERS from
    a Kaggle-attached file). Returns the path. Raises if the source isn't available."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{name}.csv")
    if os.path.exists(path) and not force:
        return path

    # 1. Kaggle / local adapter
    if name in LOADERS:
        try:
            df = LOADERS[name]()
            df.to_csv(path, index=False)
            print(f"[data] {name} <- adapter ({len(df)} rows, {df.shape[1]-1} channels) -> {path}")
            return path
        except FileNotFoundError as e:
            if name not in DATASET_URLS:
                raise KeyError(
                    f"{name}: adapter found no source ({e}) and no HTTP URL. "
                    f"Attach a dataset containing one of the expected files."
                ) from e

    # 2. HTTP (ETT only)
    if name in DATASET_URLS:
        url = DATASET_URLS[name]
        print(f"[data] downloading {name} <- {url}")
        urllib.request.urlretrieve(url, path)
        print(f"[data] saved {path}  ({os.path.getsize(path)} bytes)")
        return path

    raise KeyError(f"{name}: no adapter and no URL. Known: {sorted(set(LOADERS) | set(DATASET_URLS))}")


def load_dataframe(name: str, path: Optional[str] = None) -> pd.DataFrame:
    path = path or download(name)
    df = pd.read_csv(path)
    if df.columns[0].lower() not in ("date", "time"):
        df.insert(0, "date", pd.RangeIndex(len(df)))
    return df


@dataclasses.dataclass
class SplitData:
    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    raw_train: np.ndarray
    raw_val: np.ndarray
    raw_test: np.ndarray
    channels: list

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for a in (self.train, self.val, self.test):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()[:16]


def get_splits(name: str, path: Optional[str] = None, max_channels: Optional[int] = None) -> SplitData:
    """`max_channels`: if set and the dataset has more, keep the `max_channels` most
    variable channels (by train-portion std) plus always the last column (the
    conventional univariate target). Keeps 862-sensor traffic / 370-client
    electricity tractable for the per-variate-token SAE without changing the
    difficulty question."""
    df = load_dataframe(name, path)
    # coerce every non-'date' column to numeric; drop any that are entirely
    # non-numeric (e.g. a stray datetime string column the adapter passed through).
    raw = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    numeric_ok = raw.notna().mean(axis=0) > 0.5
    raw = raw.loc[:, numeric_ok]
    channels = list(raw.columns)
    data = raw.to_numpy(dtype=np.float64)
    # drop all-NaN columns, forward/zero fill the rest
    keep = ~np.all(np.isnan(data), axis=0)
    data, channels = data[:, keep], [c for c, k in zip(channels, keep) if k]
    if np.isnan(data).any():
        df2 = pd.DataFrame(data).ffill().bfill().fillna(0.0)
        data = df2.to_numpy(dtype=np.float64)
    n = len(data)

    if max_channels is not None and data.shape[1] > max_channels:
        tr_cut = int(0.7 * n)
        stds = data[:tr_cut].std(0)
        keep_idx = set(np.argsort(-stds)[: max_channels - 1].tolist())
        keep_idx.add(data.shape[1] - 1)                       # keep the target column
        keep_idx = sorted(keep_idx)
        data = data[:, keep_idx]
        channels = [channels[i] for i in keep_idx]

    if name in _ETT_MONTHS:
        step = _ETT_MONTHS[name]
        tr_end, va_end = 12 * 30 * step, 16 * 30 * step
        if va_end >= n:                     # bundle ETT is shorter than the full 20 months
            tr_end, va_end = int(0.7 * n), int(0.8 * n)
    else:
        tr_end, va_end = int(0.7 * n), int(0.8 * n)

    raw_tr, raw_va, raw_te = data[:tr_end], data[tr_end:va_end], data[va_end:]
    mean, std = raw_tr.mean(0), raw_tr.std(0) + 1e-8
    return SplitData(
        name=name,
        train=(raw_tr - mean) / std,
        val=(raw_va - mean) / std,
        test=(raw_te - mean) / std,
        mean=mean, std=std,
        raw_train=raw_tr, raw_val=raw_va, raw_test=raw_te,
        channels=channels,
    )


def make_windows(series: np.ndarray, lookback: int, horizon: int, stride: int = 1):
    T = len(series)
    starts = np.arange(0, T - lookback - horizon + 1, stride)
    X = np.stack([series[s : s + lookback] for s in starts]).astype(np.float32)
    Y = np.stack([series[s + lookback : s + lookback + horizon] for s in starts]).astype(np.float32)
    end_idx = starts + lookback - 1
    return X, Y, end_idx
