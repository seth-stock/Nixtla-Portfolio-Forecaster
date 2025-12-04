"""
models_mlforecast.py
MLForecast wrapper leveraging tree-based regressors with lag/rolling features.
"""

from __future__ import annotations

from typing import Iterable, List
from contextlib import nullcontext

import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
import os


def _set_thread_env(n_jobs: int):
    """Force common BLAS/OMP env vars so RF respects the requested cores."""
    threads = str(max(1, int(n_jobs)))
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT", "SKLEARN_NUM_THREADS"]:
        os.environ[var] = threads


def _thread_context(n_jobs: int | None):
    """
    Create a context manager that constrains thread usage for BLAS/OpenMP-heavy ops.
    Ensures the Streamlit slider truly caps CPU usage during RF training/backtests.
    """
    if n_jobs is None:
        return nullcontext()
    _set_thread_env(n_jobs)
    return threadpool_limits(limits=int(max(1, n_jobs)))


def _prepare_data(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    renamed = df.rename(columns={date_col: "ds", target_col: "y"})
    if "unique_id" in renamed.columns:
        renamed["unique_id"] = renamed["unique_id"].astype(str)
    elif "ticker" in renamed.columns:
        renamed = renamed.assign(unique_id=renamed["ticker"].astype(str))
    else:
        renamed = renamed.assign(unique_id="series")
    return renamed[["unique_id", "ds", "y"]]


def _regularize_frequency(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Ensure each unique_id has a complete date range at the desired freq, interpolate y to avoid flatlining."""
    if not freq:
        return df
    frames = []
    for uid, g in df.groupby("unique_id"):
        g = g.sort_values("ds")
        full_idx = pd.date_range(start=g["ds"].min(), end=g["ds"].max(), freq=freq)
        g = g.set_index("ds").reindex(full_idx)
        g["unique_id"] = uid
        g["y"] = g["y"].interpolate().bfill().ffill()
        g = g.dropna(subset=["y"]).reset_index().rename(columns={"index": "ds"})
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def _differenced_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Difference the target per series to reduce trend/level bias (helps RF avoid flat lines).
    Returns differenced df and last observed level per uid.
    """
    last_levels = (
        df.sort_values("ds")
        .groupby("unique_id")
        .tail(1)[["unique_id", "y"]]
        .rename(columns={"y": "last_y"})
        .reset_index(drop=True)
    )
    frames = []
    for uid, g in df.groupby("unique_id"):
        g = g.sort_values("ds")
        g["y"] = g["y"].diff()
        g = g.dropna(subset=["y"])
        frames.append(g)
    diff_df = pd.concat(frames, ignore_index=True)
    return diff_df, last_levels


def fit_and_forecast(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    horizon: int,
    freq: str,
    lags: Iterable[int] = (1, 2, 3, 7, 14),
    windows: Iterable[int] = (7, 14, 28),
    model_name: str = "RandomForest",
    rf_params: dict | None = None,
    n_jobs: int | None = None,
    use_diff: bool = True,
) -> pd.DataFrame:
    """Fit an MLForecast model with standard lag and rolling window features."""
    rf_params = rf_params or {}
    # Ensure n_jobs is honored; default to provided value else all cores.
    explicit_n_jobs = n_jobs if n_jobs is not None else rf_params.pop("n_jobs", None)
    lag_list = list(lags)
    window_list = list(windows)
    prepared = _prepare_data(df, date_col, target_col)
    results = []
    with _thread_context(explicit_n_jobs):
        for uid, g in prepared.groupby("unique_id"):
            g_reg = _regularize_frequency(g, freq)
            last_levels = None
            if use_diff:
                g_reg, last_levels = _differenced_target(g_reg)
            rf_params_local = {**rf_params, "n_jobs": int(explicit_n_jobs) if explicit_n_jobs is not None else -1}
            regressor = RandomForestRegressor(random_state=42, **rf_params_local)
            ml = MLForecast(
                models=[regressor],
                freq=freq,
                lags=lag_list,
                lag_transforms={w: [RollingMean(window_size=w)] for w in window_list},
            )
            ml.fit(g_reg)
            fcst = ml.predict(horizon)
            if "unique_id" not in fcst.columns:
                fcst["unique_id"] = uid
            forecast_cols = [c for c in fcst.columns if c not in ["unique_id", "ds"]]
            if not forecast_cols:
                continue
            fcst = fcst.rename(columns={forecast_cols[0]: "forecast"})
            if use_diff and last_levels is not None:
                fcst = fcst.merge(last_levels, on="unique_id", how="left")
                fcst["forecast"] = fcst.groupby("unique_id")["forecast"].cumsum() + fcst["last_y"]
                fcst = fcst.drop(columns=["last_y"])
            fcst["model"] = model_name
            results.append(fcst[["ds", "unique_id", "model", "forecast"]])
    if not results:
        return pd.DataFrame(columns=["ds", "unique_id", "model", "forecast"])
    return pd.concat(results, ignore_index=True)


def backtest(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    horizon: int,
    freq: str,
    n_windows: int = 2,
    lags: Iterable[int] = (1, 2, 3, 7, 14),
    windows: Iterable[int] = (7, 14, 28),
    model_name: str = "RandomForest",
    rf_params: dict | None = None,
    n_jobs: int | None = None,
    use_diff: bool = True,
) -> pd.DataFrame:
    """Perform rolling backtesting for MLForecast."""
    rf_params = rf_params or {}
    explicit_n_jobs = n_jobs if n_jobs is not None else rf_params.pop("n_jobs", None)
    prepared = _prepare_data(df, date_col, target_col)
    lag_list = list(lags)
    window_list = list(windows)
    cv_results = []
    with _thread_context(explicit_n_jobs):
        for uid, g in prepared.groupby("unique_id"):
            g_reg = _regularize_frequency(g, freq)
            base_lookup = g_reg.sort_values("ds").set_index("ds")["y"]
            g_use = g_reg
            if use_diff:
                g_use, _ = _differenced_target(g_reg)
            rf_params_local = {**rf_params, "n_jobs": int(explicit_n_jobs) if explicit_n_jobs is not None else -1}
            regressor = RandomForestRegressor(random_state=42, **rf_params_local)
            ml = MLForecast(
                models=[regressor],
                freq=freq,
                lags=lag_list,
                lag_transforms={w: [RollingMean(window_size=w)] for w in window_list},
            )
            try:
                cv = ml.cross_validation(h=horizon, n_windows=n_windows, step_size=1, df=g_use)
            except ValueError:
                continue
            if "unique_id" not in cv.columns:
                cv["unique_id"] = uid
            forecast_cols = [c for c in cv.columns if c not in ["unique_id", "ds", "cutoff", "y"]]
            if not forecast_cols:
                continue
            cv = cv.rename(columns={forecast_cols[0]: "forecast"})
            if use_diff:
                restored = []
                for cutoff, grp in cv.groupby("cutoff"):
                    try:
                        base = base_lookup.loc[cutoff]
                    except KeyError:
                        restored.append(grp)
                        continue
                    g_sorted = grp.sort_values("ds").copy()
                    g_sorted["forecast"] = g_sorted["forecast"].cumsum() + base
                    restored.append(g_sorted)
                if restored:
                    cv = pd.concat(restored, ignore_index=True)
            cv["model"] = model_name
            cv_results.append(cv[["ds", "unique_id", "cutoff", "model", "forecast", "y"]])
    if not cv_results:
        return pd.DataFrame(columns=["ds", "unique_id", "cutoff", "model", "forecast", "y"])
    return pd.concat(cv_results, ignore_index=True)
