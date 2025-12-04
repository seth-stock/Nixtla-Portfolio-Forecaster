"""
Forecasting page
Streamlit UI for uploading time series data, configuring Nixtla models,
running forecasts/backtests, and visualizing results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
import os

import pandas as pd
import plotly.express as px
import streamlit as st
import psutil

from core import data_loading
from core import alpaca_data
from core import evaluation
from core import models_mlforecast, models_neuralforecast, models_statsforecast
from core.config import default_config, load_config, save_config


def sidebar_controls():
    loaded_cfg = st.session_state.get("fc_loaded_cfg", {})

    st.sidebar.header("Data & Preprocessing")
    data_source_default = loaded_cfg.get("data_source", "Upload CSV")
    data_source = st.sidebar.selectbox(
        "Data source",
        ["Upload CSV", "Alpaca Daily", "Alpaca Minute"],
        index=["Upload CSV", "Alpaca Daily", "Alpaca Minute"].index(data_source_default) if data_source_default in ["Upload CSV", "Alpaca Daily", "Alpaca Minute"] else 0,
        key="fc_data_source",
    )
    uploaded = None
    tickers = ""
    start_date_fetch = ""
    end_date_fetch = ""
    alpaca_api_key = ""
    alpaca_api_secret = ""
    device_choice = st.sidebar.selectbox("Device", ["cuda", "cpu"], index=0)
    cpu_threads = st.sidebar.slider("CPU threads (used for RF/CPU ops)", 1, 32, 16)
    if data_source == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="fc_upload")
    else:
        tickers = st.sidebar.text_input("Tickers (comma-separated)", value=loaded_cfg.get("tickers", st.session_state.get("fc_tickers", "AAPL,MSFT,GOOG")), key="fc_tickers")
        start_date_fetch = st.sidebar.text_input("Start date (YYYY-MM-DD)", value=loaded_cfg.get("start_date_fetch", st.session_state.get("fc_start_fetch", "")), key="fc_start_fetch")
        end_date_fetch = st.sidebar.text_input("End date (YYYY-MM-DD)", value=loaded_cfg.get("end_date_fetch", st.session_state.get("fc_end_fetch", "")), key="fc_end_fetch")
        alpaca_api_key = st.sidebar.text_input("Alpaca API Key", type="password", value=loaded_cfg.get("alpaca_api_key", st.session_state.get("fc_alpaca_key", "")), key="fc_alpaca_key")
        alpaca_api_secret = st.sidebar.text_input("Alpaca API Secret", type="password", value=loaded_cfg.get("alpaca_api_secret", st.session_state.get("fc_alpaca_secret", "")), key="fc_alpaca_secret")
    missing_strategy = st.sidebar.selectbox("Missing value handling", ["ffill", "bfill", "interpolate", "drop"], index=["ffill","bfill","interpolate","drop"].index(loaded_cfg.get("preprocess", {}).get("missing", st.session_state.get("fc_missing","ffill"))) if loaded_cfg.get("preprocess", {}).get("missing", None) in ["ffill","bfill","interpolate","drop"] else 0, key="fc_missing")
    start_date = st.sidebar.text_input("Filter start date (YYYY-MM-DD)", value=loaded_cfg.get("start_date", st.session_state.get("fc_start_filter", "")), key="fc_start_filter")
    end_date = st.sidebar.text_input("Filter end date (YYYY-MM-DD)", value=loaded_cfg.get("end_date", st.session_state.get("fc_end_filter", "")), key="fc_end_filter")

    st.sidebar.header("Forecast Settings")
    horizon = st.sidebar.slider("Forecast horizon (steps)", min_value=4, max_value=60, value=loaded_cfg.get("horizon", st.session_state.get("fc_horizon", 12)), step=1, key="fc_horizon")
    test_ratio = st.sidebar.slider(
        "Test size (fraction of history)",
        min_value=0.05,
        max_value=0.5,
        value=float(loaded_cfg.get("test_ratio", st.session_state.get("fc_test_ratio", 0.2))),
        step=0.05,
        key="fc_test_ratio",
    )
    val_ratio = st.sidebar.slider(
        "Validation size (fraction of pre-test)",
        min_value=0.05,
        max_value=0.5,
        value=float(loaded_cfg.get("val_ratio", st.session_state.get("fc_val_ratio", 0.2))),
        step=0.05,
        key="fc_val_ratio",
    )
    purge_steps = st.sidebar.number_input(
        "Purge gap before val/test (steps)",
        min_value=0,
        max_value=200,
        value=int(loaded_cfg.get("purge_steps", st.session_state.get("fc_purge", 0))),
        step=1,
        key="fc_purge_steps",
    )
    embargo_steps = st.sidebar.number_input(
        "Embargo gap before test (steps)",
        min_value=0,
        max_value=200,
        value=int(loaded_cfg.get("embargo_steps", st.session_state.get("fc_embargo", 0))),
        step=1,
        key="fc_embargo_steps",
    )
    freq_options = ["Auto-detect", "D", "W", "M", "Q"]
    freq_default = loaded_cfg.get("frequency", st.session_state.get("fc_freq", "Auto-detect"))
    freq_choice = st.sidebar.selectbox("Frequency", freq_options, index=freq_options.index(freq_default) if freq_default in freq_options else 0, key="fc_freq")
    lookback_days = st.sidebar.number_input(
        "Lookback filter (days, 0 = full history)",
        min_value=0,
        max_value=3650,
        value=int(st.session_state.get("fc_lookback_days", loaded_cfg.get("lookback_days", 0))),
        step=30,
        key="fc_lookback_days",
    )
    lookback_steps = st.sidebar.number_input(
        "Lookback window (steps for fitting, 0 = full)",
        min_value=0,
        max_value=5000,
        value=int(st.session_state.get("fc_lookback_steps", loaded_cfg.get("lookback_steps", 0))),
        step=10,
        key="fc_lookback_steps",
    )
    forecast_mode = st.sidebar.selectbox(
        "Forecast mode",
        ["multi-output (direct)", "multi-step recursive", "single-step"],
        index=0,
        key="fc_forecast_mode",
    )

    st.sidebar.header("Model Selection")
    stats_models = []
    for name in ["AutoARIMA", "AutoETS", "SeasonalNaive"]:
        default_sf = loaded_cfg.get("models", {}).get("statsforecast", meta.get("stats_models", []))
        if st.sidebar.checkbox(f"StatsForecast: {name}", value=name in default_sf if default_sf is not None else st.session_state.get(f"fc_sf_{name}", name in ["AutoARIMA", "AutoETS"]), key=f"fc_sf_{name}"):
            stats_models.append(name)
    ml_models = []
    default_ml = loaded_cfg.get("models", {}).get("mlforecast", meta.get("ml_models", []))
    if st.sidebar.checkbox("MLForecast: RandomForest", value=("RandomForest" in default_ml) if default_ml is not None else st.session_state.get("fc_ml_rf", True), key="fc_ml_rf"):
        ml_models.append("RandomForest")
    neural_models = []
    default_neural = loaded_cfg.get("models", {}).get("neuralforecast", meta.get("neural_models", []))
    for name, default in [("RNN", False), ("LSTM", False), ("S5", False)]:
        if st.sidebar.checkbox(f"NeuralForecast: {name}", value=(name in default_neural) if default_neural is not None else st.session_state.get(f"fc_neural_{name}", default), key=f"fc_neural_{name}"):
            neural_models.append(name)

    st.sidebar.header("Backtesting")
    n_windows = st.sidebar.slider("Rolling windows", min_value=1, max_value=5, value=st.session_state.get("fc_n_windows", 2), key="fc_n_windows")

    st.sidebar.header("Hyperparameters")
    hp_loaded = loaded_cfg.get("hyperparams", {})
    with st.sidebar.expander("StatsForecast"):
        sf_season = st.number_input("Season length (SeasonalNaive fallback)", min_value=1, value=hp_loaded.get("stats", {}).get("SeasonalNaive", {}).get("season_length", 1), step=1, key="sf_season_len")
    with st.sidebar.expander("MLForecast (RandomForest)"):
        rf_estimators = st.number_input("RF n_estimators", min_value=50, max_value=2000, value=hp_loaded.get("ml", {}).get("rf_params", {}).get("n_estimators", 200), step=50, key="rf_estimators")
        rf_max_depth = st.number_input("RF max_depth (0=auto)", min_value=0, max_value=500, value=0 if hp_loaded.get("ml", {}).get("rf_params", {}).get("max_depth", None) is None else hp_loaded.get("ml", {}).get("rf_params", {}).get("max_depth", 0), step=1, key="rf_max_depth")
        rf_use_diff = st.checkbox("Use differencing (RF only)", value=hp_loaded.get("ml", {}).get("use_diff", True), key="rf_use_diff")
    with st.sidebar.expander("Neural (RNN)"):
        st.caption("Warning: Large values may increase runtime and GPU memory usage.")
        rnn_input = st.number_input("RNN input size", min_value=1, value=hp_loaded.get("neural", {}).get("rnn_input", 24), step=1, key="nn_input_rnn")
        rnn_steps = st.number_input("RNN epochs", min_value=10, value=hp_loaded.get("neural", {}).get("rnn_steps", 200), step=10, key="nn_steps_rnn")
        rnn_lr = st.number_input("RNN learning rate", value=hp_loaded.get("neural", {}).get("rnn_lr", 1e-3), format="%.5f", key="nn_lr_rnn")
        rnn_batch = st.number_input("RNN batch size", min_value=8, value=hp_loaded.get("neural", {}).get("rnn_batch", 32), step=8, key="nn_batch_rnn")
        rnn_hidden = st.number_input("RNN hidden dim", min_value=8, value=hp_loaded.get("neural", {}).get("rnn_hidden", 128), step=8, key="nn_hidden_rnn")
        rnn_depth = st.number_input("RNN depth (layers)", min_value=1, value=hp_loaded.get("neural", {}).get("rnn_depth", 2), step=1, key="nn_depth_rnn")
        rnn_head = st.number_input("RNN head size", min_value=1, value=hp_loaded.get("neural", {}).get("rnn_head", 32), step=1, key="nn_head_rnn")
    with st.sidebar.expander("Neural (LSTM)"):
        st.caption("Warning: Large values may increase runtime and GPU memory usage.")
        lstm_input = st.number_input("LSTM input size", min_value=1, value=hp_loaded.get("neural", {}).get("lstm_input", 24), step=1, key="nn_input_lstm")
        lstm_steps = st.number_input("LSTM epochs", min_value=10, value=hp_loaded.get("neural", {}).get("lstm_steps", 200), step=10, key="nn_steps_lstm")
        lstm_lr = st.number_input("LSTM learning rate", value=hp_loaded.get("neural", {}).get("lstm_lr", 1e-3), format="%.5f", key="nn_lr_lstm")
        lstm_batch = st.number_input("LSTM batch size", min_value=8, value=hp_loaded.get("neural", {}).get("lstm_batch", 32), step=8, key="nn_batch_lstm")
        lstm_hidden = st.number_input("LSTM hidden dim", min_value=8, value=hp_loaded.get("neural", {}).get("lstm_hidden", 128), step=8, key="nn_hidden_lstm")
        lstm_depth = st.number_input("LSTM depth (layers)", min_value=1, value=hp_loaded.get("neural", {}).get("lstm_depth", 2), step=1, key="nn_depth_lstm")
        lstm_head = st.number_input("LSTM head size", min_value=1, value=hp_loaded.get("neural", {}).get("lstm_head", 32), step=1, key="nn_head_lstm")
    with st.sidebar.expander("Neural (S5)"):
        st.caption("Warning: Large values may increase runtime and GPU memory usage.")
        s5_d_model = st.number_input("S5 d_model", min_value=8, value=hp_loaded.get("neural", {}).get("s5_d_model", 32), step=8, key="s5_d_model")
        s5_epochs = st.number_input("S5 epochs", min_value=10, value=hp_loaded.get("neural", {}).get("s5_epochs", 50), step=10, key="s5_epochs")
        s5_lr = st.number_input("S5 learning rate", value=hp_loaded.get("neural", {}).get("s5_lr", 1e-3), format="%.5f", key="s5_lr")
        s5_depth = st.number_input("S5 depth (layers)", min_value=1, value=hp_loaded.get("neural", {}).get("s5_depth", 2), step=1, key="s5_depth")
        s5_head = st.number_input("S5 head size", min_value=1, value=hp_loaded.get("neural", {}).get("s5_head", 32), step=1, key="s5_head")

    st.sidebar.header("Config persistence")
    config_path = st.sidebar.text_input("Config path", value=st.session_state.get("fc_config_path", "configs/models_config.json"), key="fc_config_path")
    load_btn = st.sidebar.button("Load config", key="fc_load")
    save_btn = st.sidebar.button("Save current config", key="fc_save")

    return {
        "uploaded": uploaded,
        "data_source": data_source,
        "tickers": tickers,
        "start_date_fetch": start_date_fetch or None,
        "end_date_fetch": end_date_fetch or None,
        "alpaca_api_key": alpaca_api_key,
        "alpaca_api_secret": alpaca_api_secret,
        "device_choice": device_choice,
        "cpu_threads": cpu_threads,
        "missing_strategy": missing_strategy,
        "start_date": start_date or None,
        "end_date": end_date or None,
        "horizon": horizon,
        "freq_choice": None if freq_choice == "Auto-detect" else freq_choice,
        "lookback_days": lookback_days,
        "lookback_steps": lookback_steps,
        "forecast_mode": forecast_mode,
        "test_ratio": float(test_ratio),
        "val_ratio": float(val_ratio),
        "purge_steps": int(purge_steps),
        "embargo_steps": int(embargo_steps),
        "stats_models": stats_models,
        "ml_models": ml_models,
        "neural_models": neural_models,
        "hyperparams": {
            "stats": {"SeasonalNaive": {"season_length": sf_season}},
            "ml": {"rf_params": {"n_estimators": int(rf_estimators), **({"max_depth": None} if rf_max_depth == 0 else {"max_depth": int(rf_max_depth)})}},
            "ml_use_diff": rf_use_diff,
            "neural": {
                "rnn_input": int(rnn_input),
                "rnn_steps": int(rnn_steps),
                "rnn_lr": float(rnn_lr),
                "rnn_batch": int(rnn_batch),
                "rnn_hidden": int(rnn_hidden),
                "rnn_depth": int(rnn_depth),
                "rnn_head": int(rnn_head),
                "lstm_input": int(lstm_input),
                "lstm_steps": int(lstm_steps),
                "lstm_lr": float(lstm_lr),
                "lstm_batch": int(lstm_batch),
                "lstm_hidden": int(lstm_hidden),
                "lstm_depth": int(lstm_depth),
                "lstm_head": int(lstm_head),
                "s5_d_model": int(s5_d_model),
                "s5_epochs": int(s5_epochs),
                "s5_lr": float(s5_lr),
                "s5_depth": int(s5_depth),
                "s5_head": int(s5_head),
            },
        },
        "n_windows": n_windows,
        "config_path": config_path,
        "load_btn": load_btn,
        "save_btn": save_btn,
    }


def resource_monitor():
    with st.sidebar.expander("Resource Monitor", expanded=False):
        # Faster sampling (~30 Hz) per user request
        cpu = psutil.cpu_percent(interval=0.03)
        mem = psutil.virtual_memory()
        st.write(f"CPU: {cpu:.1f}%")
        st.write(f"RAM: {mem.percent:.1f}% ({(mem.used/1e9):.2f} GB / {(mem.total/1e9):.2f} GB)")
        # GPU (best effort)
        try:
            import pynvml

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(h)
            st.write(f"GPU: {util.gpu}%")
            st.write(f"VRAM: {meminfo.used/1e9:.2f} GB / {meminfo.total/1e9:.2f} GB")
        except Exception:
            st.write("GPU info unavailable")


def show_config_actions(settings, meta):
    if settings["load_btn"]:
        cfg = load_config(settings["config_path"])
        if cfg:
            # Persist loaded config so the next rerun repopulates widget defaults.
            st.session_state["fc_loaded_cfg"] = cfg
            st.session_state["fc_loaded_msg"] = f"Loaded configuration from {settings['config_path']}"
            st.session_state["fc_cfg_applied"] = False
            st.rerun()
        else:
            st.warning("Config file not found; using current settings.")

    if settings["save_btn"]:
        cfg = {
            "horizon": meta["horizon"],
            "frequency": meta["freq_choice"],
            "models": {
                "statsforecast": meta["stats_models"],
                "mlforecast": meta["ml_models"],
                "neuralforecast": meta["neural_models"],
            },
            "preprocess": {"missing": meta["missing_strategy"]},
            "backtest": {"windows": meta["n_windows"]},
            "lookback_days": settings.get("lookback_days", 0),
            "lookback_steps": settings.get("lookback_steps", 0),
            "forecast_mode": settings.get("forecast_mode"),
            "test_ratio": settings.get("test_ratio"),
            "val_ratio": settings.get("val_ratio"),
            "purge_steps": settings.get("purge_steps"),
            "embargo_steps": settings.get("embargo_steps"),
            "hyperparams": settings.get("hyperparams", {}),
            "data_source": settings.get("data_source"),
            "tickers": settings.get("tickers"),
            "start_date_fetch": settings.get("start_date_fetch"),
            "end_date_fetch": settings.get("end_date_fetch"),
            "alpaca_api_key": settings.get("alpaca_api_key"),
            "alpaca_api_secret": settings.get("alpaca_api_secret"),
            "device_choice": settings.get("device_choice"),
            "cpu_threads": settings.get("cpu_threads"),
        }
        save_config(cfg, settings["config_path"])
        st.success(f"Saved configuration to {settings['config_path']}")


def run_forecasts(df: pd.DataFrame, settings, meta_state) -> dict:
    """Execute selected models and collect forecasts and backtests."""
    results = {"forecast": [], "cv": []}
    params = settings.get("hyperparams", {})
    # Derive a safe CV horizon based on available history (per series if multi-uid).
    if "unique_id" in df.columns:
        min_len = df.groupby("unique_id").size().min()
    else:
        min_len = len(df)
    denom = settings.get("n_windows", 1) + 1
    cv_h = settings["horizon"]
    if min_len and denom > 0:
        cv_h = min(cv_h, max(1, min_len // denom))

    # StatsForecast
    if settings["stats_models"]:
        sf_fcst = models_statsforecast.fit_and_forecast(
            df,
            meta_state["date_col"],
            meta_state["target_col"],
            settings["horizon"],
            meta_state["freq"],
            settings["stats_models"],
            model_params=params.get("stats"),
            n_jobs=settings.get("cpu_threads"),
        )
        results["forecast"].append(sf_fcst)
        try:
            sf_cv = models_statsforecast.backtest(
                df,
                meta_state["date_col"],
                meta_state["target_col"],
                cv_h,
                meta_state["freq"],
                settings["stats_models"],
                settings["n_windows"],
                model_params=params.get("stats"),
                n_jobs=settings.get("cpu_threads"),
            )
            if not sf_cv.empty:
                results["cv"].append(sf_cv)
        except Exception as e:
            st.warning(f"StatsForecast backtest skipped: {e}")

    # MLForecast
    if settings["ml_models"]:
        try:
            ml_fcst = models_mlforecast.fit_and_forecast(
                df,
                meta_state["date_col"],
                meta_state["target_col"],
                settings["horizon"],
                meta_state["freq"],
                rf_params=params.get("ml", {}).get("rf_params"),
                n_jobs=settings.get("cpu_threads"),
                use_diff=params.get("ml_use_diff", True),
            )
            results["forecast"].append(ml_fcst)
            try:
                ml_cv = models_mlforecast.backtest(
                    df,
                    meta_state["date_col"],
                    meta_state["target_col"],
                    cv_h,
                    meta_state["freq"],
                    settings["n_windows"],
                    rf_params=params.get("ml", {}).get("rf_params"),
                    n_jobs=settings.get("cpu_threads"),
                    use_diff=params.get("ml_use_diff", True),
                )
                if not ml_cv.empty:
                    results["cv"].append(ml_cv)
                else:
                    st.warning("MLForecast backtest skipped (insufficient/irregular history).")
            except Exception as e:
                st.warning(f"MLForecast backtest skipped: {e}")
        except Exception as e:
            st.warning(f"MLForecast skipped: {e}")

    # NeuralForecast
    if settings["neural_models"]:
        try:
            nf_fcst = models_neuralforecast.fit_and_forecast(
                df,
                meta_state["date_col"],
                meta_state["target_col"],
                settings["horizon"],
                meta_state["freq"],
                settings["neural_models"],
                model_params={**params.get("neural", {}), "device_preference": settings.get("device_choice"), "context_length": settings.get("lookback_steps", 0)},
            )
            results["forecast"].append(nf_fcst)
            nf_cv = models_neuralforecast.backtest(
                df,
                meta_state["date_col"],
                meta_state["target_col"],
                cv_h,
                meta_state["freq"],
                settings["neural_models"],
                settings["n_windows"],
                model_params={**params.get("neural", {}), "device_preference": settings.get("device_choice"), "context_length": settings.get("lookback_steps", 0)},
            )
            results["cv"].append(nf_cv)
        except ImportError as e:
            st.warning(f"Neural models skipped: {e}")
        except Exception as e:
            st.warning(f"Neural models failed: {e}")
    return results

def plot_forecasts(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, forecasts: pd.DataFrame, meta_state, cv_preds: pd.DataFrame | None = None):
    # Align column names
    train_l = train.rename(columns={meta_state["date_col"]: "ds", meta_state["target_col"]: "y"})
    val_l = val.rename(columns={meta_state["date_col"]: "ds", meta_state["target_col"]: "y"})
    test_l = test.rename(columns={meta_state["date_col"]: "ds", meta_state["target_col"]: "y"})
    if "unique_id" in train.columns:
        train_l["unique_id"] = train["unique_id"].values
    if "unique_id" in val.columns:
        val_l["unique_id"] = val["unique_id"].values
    if "unique_id" in test.columns:
        test_l["unique_id"] = test["unique_id"].values
    base = pd.concat([train_l.assign(segment="train"), val_l.assign(segment="val"), test_l.assign(segment="test")])
    # Prepare holdout forecasts (forced to test segment aligned to test dates)
    holdout_fc = forecasts.copy()
    holdout_fc = holdout_fc[holdout_fc["model"].astype(str).str.lower() != "index"]
    if "segment" not in holdout_fc.columns:
        holdout_fc["segment"] = "test_pred"
    if "unique_id" in holdout_fc.columns and not test_l.empty:
        aligned_frames = []
        for uid, sub_fc in holdout_fc.groupby("unique_id"):
            test_ds = test_l[test_l["unique_id"] == uid]["ds"]
            if sub_fc.empty or test_ds.empty:
                continue
            aligned = sub_fc.sort_values("ds").tail(len(test_ds)).copy()
            aligned["ds"] = list(test_ds)
            aligned["segment"] = "test_pred"
            aligned_frames.append(aligned)
        if aligned_frames:
            holdout_fc = pd.concat(aligned_frames, ignore_index=True)
    # Prepare CV preds with per-uid segmentation
    cv_fc = pd.DataFrame()
    if cv_preds is not None and not cv_preds.empty:
        cv_clean = cv_preds.copy()
        cv_clean = cv_clean[cv_clean["model"].astype(str).str.lower() != "index"]
        cv_parts = []
        if "unique_id" in cv_clean.columns:
            for uid, g in cv_clean.groupby("unique_id"):
                val_uid = val_l[val_l["unique_id"] == uid] if "unique_id" in val_l.columns else val_l
                test_uid = test_l[test_l["unique_id"] == uid] if "unique_id" in test_l.columns else test_l
                val_start = val_uid["ds"].min() if not val_uid.empty else pd.Timestamp.max
                test_start = test_uid["ds"].min() if not test_uid.empty else pd.Timestamp.max

                def _seg(dt):
                    if dt < val_start:
                        return "train_pred"
                    if dt < test_start:
                        return "val_pred"
                    return "test_pred"

                g = g.copy()
                g["segment"] = g["ds"].apply(_seg)
                cv_parts.append(g)
        else:
            val_start = val_l["ds"].min() if not val_l.empty else pd.Timestamp.max
            test_start = test_l["ds"].min() if not test_l.empty else pd.Timestamp.max

            def _seg(dt):
                if dt < val_start:
                    return "train_pred"
                if dt < test_start:
                    return "val_pred"
                return "test_pred"

            cv_clean["segment"] = cv_clean["ds"].apply(_seg)
            cv_parts.append(cv_clean)
        if cv_parts:
            cv_fc = pd.concat(cv_parts, ignore_index=True)

    uid_col = "unique_id" if ("unique_id" in holdout_fc.columns or "unique_id" in cv_fc.columns) else None
    if uid_col:
        uid_values = sorted(set(holdout_fc.get(uid_col, pd.Series()).unique()).union(set(cv_fc.get(uid_col, pd.Series()).unique())))
        for uid in uid_values:
            sub_base = base[base["unique_id"] == uid] if "unique_id" in base.columns else base.copy()
            sub_base = sub_base.sort_values("ds")
            if sub_base.empty:
                continue
            title = f"Actual vs Forecasts - {uid}"
            fig = px.line(
                sub_base,
                x="ds",
                y="y",
                color="segment",
                title=title,
                labels={"y": meta_state["target_col"], "ds": meta_state["date_col"]},
            )
            sub_holdout = holdout_fc[holdout_fc[uid_col] == uid] if not holdout_fc.empty else pd.DataFrame()
            sub_cv = cv_fc[cv_fc[uid_col] == uid] if not cv_fc.empty else pd.DataFrame()
            for model, g in sub_holdout.groupby("model"):
                g_sorted = g.sort_values("ds")
                fig.add_scatter(
                    x=g_sorted["ds"],
                    y=g_sorted["forecast"],
                    mode="lines",
                    name=f"{model} (test)",
                    connectgaps=False,
                )
            for (model, seg), g in sub_cv.groupby(["model", "segment"]):
                g_sorted = g.sort_values("ds")
                fig.add_scatter(
                    x=g_sorted["ds"],
                    y=g_sorted["forecast"],
                    mode="lines",
                    name=f"{model} ({seg})",
                    line=dict(dash="dot"),
                    connectgaps=False,
                    showlegend=True,
                )
            if not val_l.empty:
                v_start = sub_base[sub_base["segment"] == "val"]["ds"].min()
                if pd.notna(v_start):
                    fig.add_vline(x=v_start, line_dash="dot", line_color="gray")
            if not test_l.empty:
                t_start = sub_base[sub_base["segment"] == "test"]["ds"].min()
                if pd.notna(t_start):
                    fig.add_vline(x=t_start, line_dash="dash", line_color="red")
            st.plotly_chart(fig, width="stretch")
    else:
        base = base.sort_values("ds")
        fig = px.line(
            base,
            x="ds",
            y="y",
            color="segment",
            title="Actual vs Forecasts",
            labels={"y": meta_state["target_col"], "ds": meta_state["date_col"]},
        )
        for model, g in holdout_fc.groupby("model"):
            g_sorted = g.sort_values("ds")
            fig.add_scatter(x=g_sorted["ds"], y=g_sorted["forecast"], mode="lines", name=f"{model} (test)", connectgaps=False)
        for (model, seg), g in cv_fc.groupby(["model", "segment"]):
            g_sorted = g.sort_values("ds")
            fig.add_scatter(x=g_sorted["ds"], y=g_sorted["forecast"], mode="lines", name=f"{model} ({seg})", line=dict(dash="dot"), connectgaps=False)
        if not val_l.empty:
            fig.add_vline(x=val_l["ds"].min(), line_dash="dot", line_color="gray")
        if not test_l.empty:
            fig.add_vline(x=test_l["ds"].min(), line_dash="dash", line_color="red")
        st.plotly_chart(fig, width="stretch")


def render_page():
    st.title("Forecasting")
    st.write("Upload a time series CSV and compare Nixtla model forecasts with automated backtesting.")

    # Apply loaded configuration to widget defaults before rendering controls.
    cfg = st.session_state.get("fc_loaded_cfg")
    if cfg and not st.session_state.get("fc_cfg_applied", False):
        models_cfg = cfg.get("models", {})
        hyper = cfg.get("hyperparams", {})
        st.session_state["fc_data_source"] = cfg.get("data_source", "Upload CSV")
        st.session_state["fc_tickers"] = cfg.get("tickers", "")
        st.session_state["fc_start_fetch"] = cfg.get("start_date_fetch", "")
        st.session_state["fc_end_fetch"] = cfg.get("end_date_fetch", "")
        st.session_state["fc_alpaca_key"] = cfg.get("alpaca_api_key", "")
        st.session_state["fc_alpaca_secret"] = cfg.get("alpaca_api_secret", "")
        st.session_state["fc_missing"] = cfg.get("preprocess", {}).get("missing", "ffill")
        st.session_state["fc_start_filter"] = cfg.get("start_date", "")
        st.session_state["fc_end_filter"] = cfg.get("end_date", "")
        st.session_state["fc_horizon"] = cfg.get("horizon", 12)
        st.session_state["fc_test_ratio"] = cfg.get("test_ratio", 0.2)
        st.session_state["fc_val_ratio"] = cfg.get("val_ratio", 0.2)
        st.session_state["fc_purge"] = cfg.get("purge_steps", 0)
        st.session_state["fc_embargo"] = cfg.get("embargo_steps", 0)
        st.session_state["fc_freq"] = cfg.get("frequency", "Auto-detect")
        st.session_state["fc_n_windows"] = cfg.get("backtest", {}).get("windows", 2)
        st.session_state["fc_lookback_days"] = cfg.get("lookback_days", 0)
        st.session_state["fc_lookback_steps"] = cfg.get("lookback_steps", 0)
        st.session_state["fc_forecast_mode"] = cfg.get("forecast_mode", "multi-output (direct)")
        # Model selections
        st.session_state["fc_sf_AutoARIMA"] = "AutoARIMA" in models_cfg.get("statsforecast", [])
        st.session_state["fc_sf_AutoETS"] = "AutoETS" in models_cfg.get("statsforecast", [])
        st.session_state["fc_sf_SeasonalNaive"] = "SeasonalNaive" in models_cfg.get("statsforecast", [])
        st.session_state["fc_ml_rf"] = "RandomForest" in models_cfg.get("mlforecast", [])
        st.session_state["fc_neural_RNN"] = "RNN" in models_cfg.get("neuralforecast", [])
        st.session_state["fc_neural_LSTM"] = "LSTM" in models_cfg.get("neuralforecast", [])
        st.session_state["fc_neural_S5"] = "S5" in models_cfg.get("neuralforecast", [])
        # Hyperparameters
        st.session_state["sf_season_len"] = hyper.get("stats", {}).get("SeasonalNaive", {}).get("season_length", 1)
        rf_params = hyper.get("ml", {}).get("rf_params", {})
        st.session_state["rf_estimators"] = rf_params.get("n_estimators", 200)
        st.session_state["rf_max_depth"] = rf_params.get("max_depth", 0 if rf_params.get("max_depth", None) is None else rf_params.get("max_depth", 0))
        st.session_state["nn_input_rnn"] = hyper.get("neural", {}).get("rnn_input", 24)
        st.session_state["nn_steps_rnn"] = hyper.get("neural", {}).get("rnn_steps", 200)
        st.session_state["nn_lr_rnn"] = hyper.get("neural", {}).get("rnn_lr", 1e-3)
        st.session_state["nn_batch_rnn"] = hyper.get("neural", {}).get("rnn_batch", 32)
        st.session_state["nn_hidden_rnn"] = hyper.get("neural", {}).get("rnn_hidden", 128)
        st.session_state["nn_depth_rnn"] = hyper.get("neural", {}).get("rnn_depth", 2)
        st.session_state["nn_head_rnn"] = hyper.get("neural", {}).get("rnn_head", 32)
        st.session_state["nn_input_lstm"] = hyper.get("neural", {}).get("lstm_input", 24)
        st.session_state["nn_steps_lstm"] = hyper.get("neural", {}).get("lstm_steps", 200)
        st.session_state["nn_lr_lstm"] = hyper.get("neural", {}).get("lstm_lr", 1e-3)
        st.session_state["nn_batch_lstm"] = hyper.get("neural", {}).get("lstm_batch", 32)
        st.session_state["nn_hidden_lstm"] = hyper.get("neural", {}).get("lstm_hidden", 128)
        st.session_state["nn_depth_lstm"] = hyper.get("neural", {}).get("lstm_depth", 2)
        st.session_state["nn_head_lstm"] = hyper.get("neural", {}).get("lstm_head", 32)
        st.session_state["s5_d_model"] = hyper.get("neural", {}).get("s5_d_model", 32)
        st.session_state["s5_epochs"] = hyper.get("neural", {}).get("s5_epochs", 50)
        st.session_state["s5_lr"] = hyper.get("neural", {}).get("s5_lr", 1e-3)
        st.session_state["s5_depth"] = hyper.get("neural", {}).get("s5_depth", 2)
        st.session_state["s5_head"] = hyper.get("neural", {}).get("s5_head", 32)
        st.session_state["fc_cfg_applied"] = True

    settings = sidebar_controls()
    resource_monitor()
    st.caption("File size limits removed (set very high).")

    # Placeholder metadata object stored in session_state
    meta.setdefault("freq", None)
    meta["horizon"] = settings["horizon"]
    meta["missing_strategy"] = settings["missing_strategy"]
    meta["stats_models"] = settings["stats_models"]
    meta["ml_models"] = settings["ml_models"]
    meta["neural_models"] = settings["neural_models"]
    meta["n_windows"] = settings["n_windows"]
    meta["freq_choice"] = settings["freq_choice"]

    show_config_actions(settings, meta)

    st.subheader("1) Load Data & Select Columns")
    df_loaded = None
    cols = []
    default_date = "timestamp"
    default_target = "close"
    if settings["data_source"] == "Upload CSV":
        if settings["uploaded"] is None:
            st.info("Upload a CSV with at least one datetime column and one numeric target column.")
            return
        try:
            df_loaded = pd.read_csv(settings["uploaded"])
            preview_df = df_loaded.head()
            cols = list(preview_df.columns)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return
    else:
        st.info("Alpaca data will be fetched when you click **Run Forecasts**.")
        cols = ["timestamp", "close", "symbol"]
    date_col = st.selectbox("Date column", cols, index=0 if cols else None)
    target_col = st.selectbox("Target column", cols, index=1 if len(cols) > 1 else 0)
    ticker_col_choice = None
    if settings["data_source"] == "Upload CSV":
        ticker_options = ["<none>"] + cols
        default_ticker_idx = ticker_options.index("ticker") if "ticker" in cols else (ticker_options.index("symbol") if "symbol" in cols else 0)
        ticker_col_choice = st.selectbox("Ticker column (for multi-series)", ticker_options, index=default_ticker_idx)
        if ticker_col_choice == "<none>":
            ticker_col_choice = None
    meta["date_col"] = date_col
    meta["target_col"] = target_col

    if st.button("Run Forecasts", type="primary"):
        with st.spinner("Processing data and training models..."):
            try:
                if settings["data_source"] == "Upload CSV":
                    # Keep full frame so ticker column is preserved; cleaning happens later.
                    df = data_loading._read_file(settings["uploaded"])
                    if settings["start_date"]:
                        df = df[pd.to_datetime(df[date_col]) >= pd.to_datetime(settings["start_date"])]
                    if settings["end_date"]:
                        df = df[pd.to_datetime(df[date_col]) <= pd.to_datetime(settings["end_date"])]
                else:
                    # Use already-loaded Alpaca data; basic cleaning and filtering
                    if settings["alpaca_api_key"] and settings["alpaca_api_secret"]:
                        os.environ["ALPACA_API_KEY"] = settings["alpaca_api_key"]
                        os.environ["ALPACA_API_SECRET"] = settings["alpaca_api_secret"]
                    symbols = [s.strip().upper() for s in settings["tickers"].split(",") if s.strip()]
                    if settings["data_source"] == "Alpaca Daily":
                        raw = alpaca_data.fetch_daily_bars(symbols, start=settings["start_date_fetch"], end=settings["end_date_fetch"])
                    else:
                        raw = alpaca_data.fetch_intraday_bars(symbols, start=settings["start_date_fetch"], end=settings["end_date_fetch"])
                    if raw.empty:
                        st.error("No data returned from Alpaca; check credentials, tickers, or date range.")
                        return
                    df = raw.rename(columns={"timestamp": date_col, "close": target_col, "symbol": "ticker"} if "symbol" in raw.columns else {"timestamp": date_col, "close": target_col})
                    df[date_col] = pd.to_datetime(df[date_col])
                    if settings["start_date"]:
                        df = df[df[date_col] >= settings["start_date"]]
                    if settings["end_date"]:
                        df = df[df[date_col] <= settings["end_date"]]
            except Exception as e:
                st.error(f"Data loading error: {e}")
                return

            # Device / thread preferences
            if settings["device_choice"] == "cpu":
                try:
                    import torch
                    torch.set_num_threads(settings["cpu_threads"])
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                except Exception:
                    pass
            else:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        st.warning("CUDA selected but torch reports no GPU available; falling back to CPU.")
                except Exception:
                    st.warning("CUDA selected but torch failed to import; falling back to CPU.")

            # Clean and standardize the time series (per ticker if present)
            try:
                ticker_hint = ticker_col_choice or ("ticker" if "ticker" in df.columns else ("symbol" if "symbol" in df.columns else None))
                df = data_loading.prepare_multiseries_frame(df, date_col=date_col, target_col=target_col, ticker_col=ticker_hint)
                # Ensure unique_id exists and is sourced from ticker/symbol
                if "unique_id" not in df.columns:
                    if "ticker" in df.columns:
                        df["unique_id"] = df["ticker"].astype(str)
                    elif "symbol" in df.columns:
                        df["unique_id"] = df["symbol"].astype(str)
                    else:
                        df["unique_id"] = "series"
                df["unique_id"] = df["unique_id"].fillna("series").astype(str)
                # Apply missing strategy on the clean frame (upload path already handled most cases)
                df = data_loading._handle_missing(df, target_col, strategy=settings["missing_strategy"])
            except Exception as e:
                st.error(f"Failed to standardize time series data: {e}")
                return

            # Final safety: if ticker column exists but unique_id missing, create it
            if "unique_id" not in df.columns and "ticker" in df.columns:
                df["unique_id"] = df["ticker"].astype(str)
            if "unique_id" not in df.columns:
                df["unique_id"] = "series"
            df["unique_id"] = df["unique_id"].fillna("series").astype(str)

            st.success(f"Loaded {len(df)} rows.")
            ticker_list = sorted(df["unique_id"].astype(str).unique().tolist()) if "unique_id" in df.columns else []
            st.write(f"Tickers detected: {ticker_list}")
            # Show a few rows per ticker so multi-ticker uploads are obvious
            preview = df.groupby("unique_id").head(3) if "unique_id" in df.columns else df.head()
            st.dataframe(preview)

            freq = settings["freq_choice"] or data_loading.infer_frequency_per_series(df, date_col) or "D"
            meta["freq"] = freq
            st.write(f"Detected/selected frequency: **{freq}**")

            try:
                # Apply lookback only when explicitly set
                if settings.get("lookback_days", 0) > 0:
                    cutoff = df[date_col].max() - pd.Timedelta(days=int(settings["lookback_days"]))
                    df = df[df[date_col] >= cutoff]
                if settings.get("lookback_steps", 0) > 0:
                    k = int(settings["lookback_steps"])
                    trimmed = []
                    for _, g in df.groupby("unique_id"):
                        trimmed.append(g.sort_values(date_col).tail(k + settings["horizon"]))
                    if trimmed:
                        df = pd.concat(trimmed, ignore_index=True)
                # Ensure per-ticker sorting before splitting with purge/embargo
                df = df.sort_values(["unique_id", date_col])
                train_df, val_df, test_df = data_loading.temporal_split(
                    df,
                    horizon=settings["horizon"],
                    val_ratio=float(settings.get("val_ratio", 0.2)),
                    test_ratio=float(settings.get("test_ratio", 0.2)),
                    purge=settings.get("purge_steps") if settings.get("purge_steps", 0) > 0 else settings["horizon"] // 2,
                    embargo=settings.get("embargo_steps") if settings.get("embargo_steps", 0) > 0 else max(1, settings["horizon"] // 4),
                    date_col=date_col,
                )
            except Exception as e:
                st.error(f"Train/val/test split error: {e}")
                return

            try:
                # Fit on train+val to respect purge/embargo gap to test
                fit_df = pd.concat([train_df, val_df]).sort_values(meta["date_col"])
                # Use per-ticker test length (max) as forecast horizon to align with holdout
                if "unique_id" in test_df.columns and not test_df.empty:
                    test_h = int(test_df.groupby("unique_id").size().max())
                else:
                    test_h = len(test_df)
                settings_with_test_h = settings.copy()
                settings_with_test_h["horizon"] = test_h
                results = run_forecasts(fit_df, settings_with_test_h, meta)
            except Exception as e:
                import traceback

                st.error(f"Forecasting failed: {e}")
                st.code(traceback.format_exc())
                return
            if not results["forecast"]:
                st.warning("No models selected.")
                return

            forecast_df = pd.concat(results["forecast"]).dropna(subset=["forecast"])
            forecast_df["segment"] = "test_pred"
            forecast_df = forecast_df[forecast_df["model"].astype(str).str.lower() != "index"]
            # Align forecast horizon to actual test dates per ticker to avoid index drift
            if "unique_id" in forecast_df.columns and "unique_id" in test_df.columns:
                aligned_frames = []
                for uid, g in forecast_df.groupby("unique_id"):
                    test_uid = test_df[test_df["unique_id"] == uid].sort_values(date_col)
                    g = g.sort_values("ds").tail(len(test_uid))
                    g["ds"] = test_uid[date_col].values[: len(g)]
                    aligned_frames.append(g)
                forecast_df = pd.concat(aligned_frames, ignore_index=True)
            else:
                test_sorted = test_df.sort_values(date_col)
                forecast_df = forecast_df.sort_values("ds").tail(len(test_sorted))
                forecast_df["ds"] = test_sorted[date_col].values[: len(forecast_df)]
            test_eval_df = test_df.rename(columns={date_col: "ds", target_col: "y"})
            if "unique_id" in test_df.columns:
                test_eval_df["unique_id"] = test_df["unique_id"]
            metrics = evaluation.evaluate_holdout(test_eval_df, forecast_df)

            st.subheader("Evaluation - Holdout")
            if metrics.empty:
                st.warning("No valid pairs to score (check for NaNs or misaligned dates).")
            else:
                st.dataframe(metrics)
                if "unique_id" in metrics.columns:
                    st.caption("Metrics shown per model and ticker.")

            cv_df = pd.concat(results["cv"], ignore_index=True) if results["cv"] else pd.DataFrame()
            plot_forecasts(train_df, val_df, test_df, forecast_df, meta, cv_preds=cv_df if not cv_df.empty else None)

            if not cv_df.empty:
                cv_summary = evaluation.summarize_backtests(cv_df)
                st.subheader("Backtest Summary")
                if cv_summary.empty:
                    st.warning("Backtest summary empty (insufficient overlap).")
                else:
                    st.dataframe(cv_summary)
            st.caption("Metrics averaged across rolling windows where available.")

            # Top-3 models per RMSE (use backtest summary if available else holdout metrics)
            ranking_source = cv_summary if not cv_df.empty else metrics
            if ranking_source is not None and not ranking_source.empty:
                if "unique_id" in ranking_source.columns:
                    top_models_map = {
                        uid: grp.sort_values("RMSE").head(3)["model"].tolist()
                        for uid, grp in ranking_source.groupby("unique_id")
                    }
                else:
                    top_models_map = {"__all__": ranking_source.sort_values("RMSE").head(3)["model"].tolist()}
                st.subheader("Top 3 Models - Predicted vs Actual")
                parts = []
                for frame, seg in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
                    temp = frame.rename(columns={date_col: "ds", target_col: "y"}).assign(segment=seg)
                    if "unique_id" in frame.columns:
                        temp["unique_id"] = frame["unique_id"].values
                    parts.append(temp)
                base_actual = pd.concat(parts, ignore_index=True)
                for uid, sub_base in (base_actual.groupby("unique_id") if "unique_id" in base_actual.columns else [("series", base_actual)]):
                    chosen_models = top_models_map.get(uid, top_models_map.get("__all__", []))
                    if not chosen_models:
                        continue
                    fig = px.line(sub_base, x="ds", y="y", color="segment", title=f"Top Models - {uid}")
                    # forecasts and cv_preds filtered to top models
                    all_fc_top = forecast_df[forecast_df["model"].isin(chosen_models)]
                    if "unique_id" in all_fc_top.columns:
                        all_fc_top = all_fc_top[all_fc_top["unique_id"] == uid]
                    for model, g in all_fc_top.groupby("model"):
                        fig.add_scatter(x=g["ds"], y=g["forecast"], mode="lines", name=f"{model} (test)", connectgaps=False)
                    if not cv_df.empty:
                        cv_top = cv_df[cv_df["model"].isin(chosen_models)]
                        if "unique_id" in cv_top.columns:
                            cv_top = cv_top[cv_top["unique_id"] == uid]
                        if not cv_top.empty:
                            val_start = sub_base[sub_base["segment"] == "val"]["ds"].min() if not sub_base[sub_base["segment"] == "val"].empty else pd.Timestamp.max
                            test_start = sub_base[sub_base["segment"] == "test"]["ds"].min() if not sub_base[sub_base["segment"] == "test"].empty else pd.Timestamp.max

                            def _seg(dt):
                                if dt < val_start:
                                    return "train_pred"
                                if dt < test_start:
                                    return "val_pred"
                                return "test_pred"

                            cv_top = cv_top.copy()
                            cv_top["segment"] = cv_top["ds"].apply(_seg)
                            for (model, seg), g in cv_top.groupby(["model", "segment"]):
                                fig.add_scatter(
                                    x=g["ds"],
                                    y=g["forecast"],
                                    mode="lines",
                                    name=f"{model} ({seg})",
                                    line=dict(dash="dot"),
                                    connectgaps=False,
                                )
                    st.plotly_chart(fig, width="stretch")


# Session metadata
if "meta" not in st.session_state:
    st.session_state.meta = default_config()

meta = st.session_state.meta

if __name__ == "__main__":
    render_page()
