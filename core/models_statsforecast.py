"""
models_statsforecast.py
Wrappers around Nixtla's statsforecast classical forecasting models.
Provides simple fit/forecast and cross-validation utilities.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def _lazy_imports():
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive
    except Exception as e:
        raise ImportError(
            "statsforecast and its binary dependencies are required for classical models. "
            f"Original error: {e}"
        ) from e
    model_registry = {
        "AutoARIMA": AutoARIMA,
        "AutoETS": AutoETS,
        "SeasonalNaive": SeasonalNaive,
    }
    return StatsForecast, SeasonalNaive, model_registry


def _prepare_data(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    """Format dataframe for StatsForecast (unique_id, ds, y)."""
    renamed = df.rename(columns={date_col: "ds", target_col: "y"})
    if "unique_id" in renamed.columns:
        renamed["unique_id"] = renamed["unique_id"].astype(str)
    elif "ticker" in renamed.columns:
        renamed = renamed.assign(unique_id=renamed["ticker"].astype(str))
    else:
        renamed = renamed.assign(unique_id="series")
    return renamed[["unique_id", "ds", "y"]]


def fit_and_forecast(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    horizon: int,
    freq: str,
    models: List[str],
    model_params: Dict[str, Dict] | None = None,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Fit selected statsforecast models and return forecast dataframe."""
    StatsForecast, SeasonalNaive, model_registry = _lazy_imports()
    model_params = model_params or {}
    season_len = model_params.get("SeasonalNaive", {}).get("season_length", 1)
    sf_df = _prepare_data(df, date_col, target_col)
    model_objs = [model_registry[m](**model_params.get(m, {})) for m in models if m in model_registry]
    if not model_objs:
        raise ValueError("No valid statsforecast models selected.")
    sf_jobs = max(1, int(n_jobs)) if n_jobs is not None else 1
    sf = StatsForecast(models=model_objs, freq=freq, n_jobs=sf_jobs, fallback_model=SeasonalNaive(season_length=season_len))
    fcst = sf.forecast(df=sf_df, h=horizon)
    fcst = fcst.reset_index(drop=True)
    fcst = fcst.melt(id_vars=["ds", "unique_id"], var_name="model", value_name="forecast")
    return fcst


def backtest(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    horizon: int,
    freq: str,
    models: List[str],
    n_windows: int = 2,
    model_params: Dict[str, Dict] | None = None,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Run rolling-origin evaluation for selected statsforecast models."""
    StatsForecast, SeasonalNaive, model_registry = _lazy_imports()
    model_params = model_params or {}
    season_len = model_params.get("SeasonalNaive", {}).get("season_length", 1)
    sf_df = _prepare_data(df, date_col, target_col)
    model_objs = [model_registry[m](**model_params.get(m, {})) for m in models if m in model_registry]
    if not model_objs:
        return pd.DataFrame(columns=["ds", "unique_id", "cutoff", "y", "model", "forecast"])

    # Guard against series that are too short for the requested CV settings.
    lengths = sf_df.groupby("unique_id").size()
    min_len = lengths.min() if not lengths.empty else 0
    # Require at least (n_windows + 1) * h points; shrink h if needed.
    safe_h = horizon
    if min_len:
        safe_h = min(horizon, max(1, min_len // (n_windows + 1)))
    # Drop series that still cannot support CV
    keep_ids = lengths[lengths >= safe_h * (n_windows + 1)].index if min_len else []
    sf_df_use = sf_df[sf_df["unique_id"].isin(keep_ids)] if len(keep_ids) else sf_df
    if sf_df_use.empty or safe_h < 1:
        return pd.DataFrame(columns=["ds", "unique_id", "cutoff", "y", "model", "forecast"])

    sf_jobs = max(1, int(n_jobs)) if n_jobs is not None else 1
    sf = StatsForecast(models=model_objs, freq=freq, n_jobs=sf_jobs, fallback_model=SeasonalNaive(season_length=season_len))
    cv = sf.cross_validation(df=sf_df_use, h=safe_h, n_windows=n_windows, step_size=safe_h)
    cv = cv.reset_index(drop=True)
    cv = cv.melt(
        id_vars=["ds", "unique_id", "cutoff", "y"],
        var_name="model",
        value_name="forecast",
    )
    return cv
