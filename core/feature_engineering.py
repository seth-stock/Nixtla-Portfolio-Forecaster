"""
feature_engineering.py
Feature generation helpers for time series forecasting.
Supports lag features, rolling statistics, and calendar/date-derived fields.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

import pandas as pd


def add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Augment dataframe with calendar features derived from the datetime column."""
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["dayofweek"] = dt.dt.dayofweek
    df["quarter"] = dt.dt.quarter
    return df


def add_lag_features(df: pd.DataFrame, target_col: str, lags: Iterable[int]) -> pd.DataFrame:
    """Create lagged target features for ML forecasting models."""
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, target_col: str, windows: Iterable[int], funcs: Iterable[str] = ("mean", "std")
) -> pd.DataFrame:
    """Add rolling statistics over specified window sizes."""
    df = df.copy()
    for window in windows:
        roll = df[target_col].rolling(window)
        for func in funcs:
            df[f"roll_{func}_{window}"] = getattr(roll, func)()
    return df


def make_ml_features(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    lags: Iterable[int] = (1, 7, 14),
    windows: Iterable[int] = (7, 14),
) -> pd.DataFrame:
    """Convenience wrapper to build common ML features with lag and calendar info."""
    df = add_lag_features(df, target_col, lags)
    df = add_rolling_features(df, target_col, windows)
    df = add_calendar_features(df, date_col)
    df = df.dropna()
    return df


def compute_returns(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Compute simple and log returns for price series."""
    df = df.copy()
    df["returns"] = df[price_col].pct_change().fillna(0.0)
    df["log_returns"] = np.log(df[price_col]).diff().fillna(0.0)
    return df
