"""
evaluation.py
Metric utilities for forecasting comparisons and backtesting summaries.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

# Robust import: fall back to numpy-based metrics if sklearn (or its deps) fail to load.
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error  # type: ignore
    _USE_SKLEARN = True
except Exception:
    _USE_SKLEARN = False

    def mean_squared_error(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.mean((y_true - y_pred) ** 2)

    def mean_absolute_error(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.mean(np.abs(y_true - y_pred))


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Compute RMSE, MAE, and MAPE metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float((np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)).mean()) * 100)
    smape = float(
        100
        * np.mean(
            2.0
            * np.abs(y_pred - y_true)
            / (np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8))
        )
    )
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "sMAPE": smape}


def summarize_backtests(cv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate cross-validation results by model.
    Expects columns: ['ds','model','forecast','y'] plus metadata.
    """
    records = []
    cv_df = cv_df[cv_df["model"].astype(str).str.lower() != "index"]
    group_keys = ["model"] + (["unique_id"] if "unique_id" in cv_df.columns else [])
    for keys, group in cv_df.groupby(group_keys):
        clean = group.dropna(subset=["y", "forecast"])
        if clean.empty:
            continue
        metrics = compute_metrics(clean["y"], clean["forecast"])
        if isinstance(keys, tuple):
            metrics["model"] = keys[0]
            if len(keys) > 1:
                metrics["unique_id"] = keys[1]
        else:
            metrics["model"] = keys
        records.append(metrics)
    summary = pd.DataFrame(records)
    ordered_cols = ["model"] + (["unique_id"] if "unique_id" in summary.columns else []) + ["RMSE", "MAE", "MAPE", "sMAPE"]
    return summary[ordered_cols].sort_values("RMSE")


def evaluate_holdout(test_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate forecast vs holdout test segment.
    forecast_df expected columns: ['ds','model','forecast'] and optionally 'unique_id'.
    test_df expected to contain ['ds','y'] (after renaming) and optionally 'unique_id'.
    """
    join_cols = ["ds", "unique_id"] if "unique_id" in forecast_df.columns and "unique_id" in test_df.columns else ["ds"]
    forecast_df = forecast_df[forecast_df["model"].astype(str).str.lower() != "index"]
    merged = forecast_df.merge(test_df[join_cols + ["y"]] if "unique_id" in test_df.columns else test_df[["ds", "y"]], on=join_cols, how="left")
    records = []
    group_keys = ["model"] + (["unique_id"] if "unique_id" in merged.columns else [])
    for keys, group in merged.groupby(group_keys):
        clean = group.dropna(subset=["y", "forecast"])
        if clean.empty:
            continue
        metrics = compute_metrics(clean["y"], clean["forecast"])
        if isinstance(keys, tuple):
            metrics["model"] = keys[0]
            if len(keys) > 1:
                metrics["unique_id"] = keys[1]
        else:
            metrics["model"] = keys
        records.append(metrics)
    if not records:
        return pd.DataFrame(columns=["model", "RMSE", "MAE", "MAPE", "sMAPE"])
    return pd.DataFrame(records).sort_values("RMSE")
