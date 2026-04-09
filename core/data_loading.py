"""
data_loading.py
Utilities to ingest time series data from CSV uploads or file paths.
Handles datetime parsing, sorting, missing value handling, and date filtering.
"""

from __future__ import annotations

from collections import Counter
from io import BytesIO, StringIO
from typing import Literal, Optional

import pandas as pd

MissingStrategy = Literal["drop", "ffill", "bfill", "interpolate"]


def load_csv(
    file,
    date_col: str,
    target_col: str,
    missing: MissingStrategy = "ffill",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a CSV (path, buffer, or uploaded file), parse the date column, and
    return a cleaned dataframe with only the date and target columns.
    """
    df = _read_file(file)
    if date_col not in df.columns or target_col not in df.columns:
        raise ValueError("Date or target column not found in uploaded data.")

    df = df[[date_col, target_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    df = _handle_missing(df, target_col, strategy=missing)
    if start_date:
        start_dt = pd.to_datetime(start_date, errors="coerce")
        if pd.isna(start_dt):
            raise ValueError("Invalid start date format. Use YYYY-MM-DD.")
        df = df[df[date_col] >= start_dt]
    if end_date:
        end_dt = pd.to_datetime(end_date, errors="coerce")
        if pd.isna(end_dt):
            raise ValueError("Invalid end date format. Use YYYY-MM-DD.")
        df = df[df[date_col] <= end_dt]
    if df.empty:
        raise ValueError("No data available after filtering/cleaning.")
    return df.reset_index(drop=True)


def infer_frequency(df: pd.DataFrame, date_col: str) -> Optional[str]:
    """Attempt to infer pandas offset alias frequency from the date column."""
    freq = pd.infer_freq(df[date_col])
    return freq


def infer_frequency_per_series(df: pd.DataFrame, date_col: str) -> Optional[str]:
    """
    Infer a consensus frequency when multiple series are present.
    Chooses the most common inferred freq across tickers; returns None if unavailable.
    """
    if "unique_id" not in df.columns:
        return infer_frequency(df, date_col)
    freqs = []
    for _, g in df.groupby("unique_id"):
        freq = pd.infer_freq(g.sort_values(date_col)[date_col])
        if freq:
            freqs.append(freq)
    if not freqs:
        return None
    return Counter(freqs).most_common(1)[0][0]


def train_test_split(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/test segments based on the forecast horizon."""
    if horizon <= 0:
        raise ValueError("Horizon must be positive.")
    if len(df) <= horizon:
        raise ValueError("Series too short for the requested horizon.")
    train = df.iloc[:-horizon].copy()
    test = df.iloc[-horizon:].copy()
    return train, test


def temporal_split(
    df: pd.DataFrame,
    horizon: int,
    val_ratio: float = 0.2,
    test_ratio: float | None = None,
    purge: Optional[int] = None,
    embargo: Optional[int] = None,
    date_col: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based train/val/test split with optional purge and embargo around the test window.
    - Test = last max(`horizon`, test_ratio * len) points if test_ratio provided, else last `horizon`.
    - Embargo removes a buffer before test to reduce leakage.
    - Validation carved from the remaining tail using val_ratio.
    If a 'unique_id' column exists, the split is applied per series and concatenated.
    """
    if horizon <= 0:
        raise ValueError("Horizon must be positive.")
    if len(df) <= horizon + 2:
        raise ValueError("Series too short for temporal split.")
    purge = purge if purge is not None else max(1, horizon // 2)
    embargo = embargo if embargo is not None else max(1, horizon // 4)

    date_col = date_col or df.columns[0]
    # Safety: ensure unique_id exists if a ticker column is present
    if "unique_id" not in df.columns:
        if "ticker" in df.columns:
            df = df.copy()
            df["unique_id"] = df["ticker"].astype(str)
        elif "symbol" in df.columns:
            df = df.copy()
            df["unique_id"] = df["symbol"].astype(str)

    def _split_single(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        g_sorted = g.sort_values(date_col).reset_index(drop=True)
        test_len = horizon
        if test_ratio is not None and test_ratio > 0:
            test_len = max(horizon, int(len(g_sorted) * test_ratio))
        test_local = g_sorted.iloc[-test_len:].copy()
        cutoff = len(g_sorted) - test_len
        embargo_start = max(0, cutoff - embargo)
        pre_test = g_sorted.iloc[:embargo_start]
        if pre_test.empty:
            raise ValueError("Not enough data after applying embargo buffer.")
        val_size = max(1, int(len(pre_test) * val_ratio))
        val_size = min(val_size, max(1, len(pre_test) - 1))
        train_local = pre_test.iloc[: -val_size].copy()
        val_local = pre_test.iloc[-val_size:].copy()
        if len(train_local) > purge:
            train_local = train_local.iloc[:-purge]
        if train_local.empty or val_local.empty or test_local.empty:
            raise ValueError("Not enough data remaining after validation, purge, and embargo settings.")
        return train_local, val_local, test_local

    if "unique_id" in df.columns:
        trains, vals, tests = [], [], []
        for _, g in df.groupby("unique_id"):
            t, v, te = _split_single(g)
            trains.append(t)
            vals.append(v)
            tests.append(te)
        return pd.concat(trains, ignore_index=True), pd.concat(vals, ignore_index=True), pd.concat(tests, ignore_index=True)
    else:
        return _split_single(df)


def _read_file(file) -> pd.DataFrame:
    """Internal helper to read a CSV from multiple input types."""
    if isinstance(file, (str, bytes)):
        return pd.read_csv(file)
    if isinstance(file, BytesIO):
        file.seek(0)
        return pd.read_csv(file)
    if hasattr(file, "read"):
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        return pd.read_csv(StringIO(content))
    raise ValueError("Unsupported file type for CSV loading.")


def _handle_missing(df: pd.DataFrame, target_col: str, strategy: MissingStrategy) -> pd.DataFrame:
    """Apply a missing value strategy to the target column."""
    out = df.copy()
    if strategy == "drop":
        return out.dropna(subset=[target_col])
    if strategy == "ffill":
        if "unique_id" in out.columns:
            out[target_col] = out.groupby("unique_id")[target_col].ffill()
            return out
        out[target_col] = out[target_col].ffill()
        return out
    if strategy == "bfill":
        if "unique_id" in out.columns:
            out[target_col] = out.groupby("unique_id")[target_col].bfill()
            return out
        out[target_col] = out[target_col].bfill()
        return out
    if strategy == "interpolate":
        if "unique_id" in out.columns:
            frames = []
            for _, g in out.groupby("unique_id"):
                g = g.copy()
                g[target_col] = g[target_col].interpolate().bfill().ffill()
                frames.append(g)
            return pd.concat(frames, ignore_index=True)
        out[target_col] = out[target_col].interpolate().bfill().ffill()
        return out
    raise ValueError(f"Unknown missing data strategy: {strategy}")


def prepare_multiseries_frame(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    ticker_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Standardize a raw price dataframe into a clean long format:
      - coercing dates/numerics
      - assigning a 'unique_id' column (one per ticker, or 'series' if none)
      - sorting by ticker then date
      - removing duplicate timestamps per ticker
    """
    frame = df.copy()
    # Auto-detect ticker column if not provided.
    detected_ticker = ticker_col
    lower_cols = {c.lower(): c for c in frame.columns}
    if detected_ticker is None:
        for cand in ["ticker", "symbol", "asset", "code", "name"]:
            if cand in lower_cols:
                detected_ticker = lower_cols[cand]
                break
    else:
        # Allow case-insensitive match when a hint was provided
        if detected_ticker not in frame.columns and detected_ticker.lower() in lower_cols:
            detected_ticker = lower_cols[detected_ticker.lower()]
    has_ticker = detected_ticker in frame.columns if detected_ticker else False
    if has_ticker and detected_ticker != "ticker":
        frame = frame.rename(columns={detected_ticker: "ticker"})
        detected_ticker = "ticker"
    # Validate required columns
    missing_cols = [c for c in [date_col, target_col] if c not in frame.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s): {missing_cols}")

    # Keep only the necessary columns to avoid plotting/model surprises.
    keep_cols = [date_col, target_col]
    if has_ticker:
        keep_cols.append("ticker")
    elif "unique_id" in frame.columns:
        keep_cols.append("unique_id")
    frame = frame[keep_cols].copy()
    # If a ticker column exists but was not retained (e.g., misnamed), attempt to keep it.
    if "ticker" not in frame.columns and has_ticker and detected_ticker and detected_ticker in df.columns:
        frame["ticker"] = df[detected_ticker].values
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce").dt.tz_localize(None)
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    frame = frame.dropna(subset=[date_col, target_col])
    if has_ticker:
        frame["unique_id"] = frame["ticker"].astype(str)
    elif "unique_id" in frame.columns:
        frame["unique_id"] = frame["unique_id"].astype(str)
    else:
        frame["unique_id"] = "series"
    frame["unique_id"] = frame["unique_id"].fillna("series").astype(str)
    # Sort per ticker and enforce one observation per timestamp per ticker.
    frame = (
        frame.sort_values(["unique_id", date_col])
        .drop_duplicates(subset=["unique_id", date_col], keep="last")
        .reset_index(drop=True)
    )
    return frame
