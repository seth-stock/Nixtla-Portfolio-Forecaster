"""
Portfolio Optimizer page
Combines validated Nixtla forecasts with a mean-variance baseline and an
optional RL policy.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st


def _get_plotly_express():
    import plotly.express as px

    return px


def _get_psutil():
    import psutil

    return psutil


def _get_alpaca_helpers():
    from core.alpaca_data import (
        configure_alpaca_credentials,
        fetch_daily_bars,
        fetch_intraday_bars,
    )

    return configure_alpaca_credentials, fetch_daily_bars, fetch_intraday_bars


def _get_portfolio_helpers():
    from core.portfolio_rl import (
        format_price_frame,
        load_prices_from_files,
        optimize_portfolio,
        optimize_portfolio_inference,
    )

    return format_price_frame, load_prices_from_files, optimize_portfolio, optimize_portfolio_inference


def resource_monitor():
    if not st.sidebar.checkbox("Enable live resource monitor", value=False, key="po_enable_resource_monitor"):
        return
    psutil = _get_psutil()
    with st.sidebar.expander("Resource Monitor", expanded=True):
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        st.write(f"CPU: {cpu:.1f}%")
        st.write(f"RAM: {mem.percent:.1f}% ({(mem.used/1e9):.2f} GB / {(mem.total/1e9):.2f} GB)")
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


def sidebar_controls():
    st.sidebar.header("Data Input")
    data_source = st.sidebar.radio(
        "Source",
        ["Upload CSV/Files", "Fetch via Alpaca (daily)", "Fetch via Alpaca (minute)"],
    )
    upload_mode = None
    uploaded_single = None
    uploaded_multi = None
    files_have_ticker = True
    date_col = "ds"
    price_col = "close"
    ticker_col = "ticker"
    alpaca_tickers = "AAPL,MSFT,GOOG"
    alpaca_start = pd.to_datetime("2022-01-01").date()
    alpaca_end = pd.to_datetime("today").date()
    alpaca_api_key = st.session_state.get("po_alpaca_key", os.getenv("ALPACA_API_KEY", ""))
    alpaca_api_secret = st.session_state.get("po_alpaca_secret", os.getenv("ALPACA_API_SECRET", ""))

    if data_source == "Upload CSV/Files":
        upload_mode = st.sidebar.radio(
            "Upload type",
            ["Single CSV (all tickers)", "Multiple CSV files (one per ticker)"],
        )
        if upload_mode == "Single CSV (all tickers)":
            uploaded_single = st.sidebar.file_uploader(
                "Upload multi-asset price CSV",
                type=["csv"],
                accept_multiple_files=False,
            )
            files_have_ticker = True
        else:
            uploaded_multi = st.sidebar.file_uploader(
                "Upload multiple ticker CSV files",
                type=["csv"],
                accept_multiple_files=True,
            )
            files_have_ticker = st.sidebar.checkbox("Files already contain ticker column", value=False)
        date_col = st.sidebar.text_input("Date column", value="ds")
        price_col = st.sidebar.text_input("Price column", value="close")
        ticker_col = st.sidebar.text_input("Ticker column", value="ticker")
    else:
        alpaca_tickers = st.sidebar.text_input("Alpaca tickers (comma-separated)", value="AAPL,MSFT,GOOG")
        alpaca_start = st.sidebar.date_input("Start date", value=alpaca_start)
        alpaca_end = st.sidebar.date_input("End date", value=alpaca_end)
        st.sidebar.caption("Enter Alpaca credentials for this session, or leave the environment defaults in place.")
        alpaca_api_key = st.sidebar.text_input("Alpaca API Key", type="password", value=alpaca_api_key, key="po_alpaca_key")
        alpaca_api_secret = st.sidebar.text_input("Alpaca API Secret", type="password", value=alpaca_api_secret, key="po_alpaca_secret")

    st.sidebar.header("Forecast Settings")
    horizon = st.sidebar.slider("Forecast horizon (steps)", 1, 120, 20)
    resample_rule = st.sidebar.text_input("Resample rule (e.g. 1D, 1H, 10min)", value="1D")
    with st.sidebar.expander("Forecast Model Params"):
        sf_season = st.number_input("Season length (SeasonalNaive fallback)", min_value=1, value=5, step=1)
        rf_estimators = st.number_input("RF n_estimators", min_value=50, max_value=2000, value=300, step=50)
        rf_max_depth = st.number_input("RF max_depth (0=auto)", min_value=0, max_value=100, value=8, step=1)

    st.sidebar.header("Optimizer Mode")
    mode = st.sidebar.radio("Mode", ["Use pre-trained checkpoint (inference)", "Train new checkpoint locally"])
    checkpoint_path = st.sidebar.text_input("Checkpoint path", value="models/portfolio_policy.pt")
    rl_mode = st.sidebar.selectbox("Policy type", ["Continuous Weights", "Graph Trading"], index=0)

    st.sidebar.header("RL Settings")
    top_k = st.sidebar.slider("Number of holdings (k)", 1, 10, 3)
    episodes = st.sidebar.slider("Training episodes", 10, 200, 60, step=10)
    lr = st.sidebar.number_input("Learning rate", value=0.001, format="%.4f")
    risk_aversion = st.sidebar.number_input("Risk aversion penalty", value=0.02, format="%.4f")
    hidden_dim = st.sidebar.number_input("Policy hidden dim", min_value=16, max_value=256, value=64, step=16)
    rl_patience = st.sidebar.number_input("Early stop patience", min_value=0, max_value=200, value=20, step=5)
    rl_min_delta = st.sidebar.number_input("Early stop min delta", value=1e-4, format="%.5f")
    device_choice = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)
    cpu_threads = st.sidebar.slider("CPU threads", 1, 32, 8)

    return {
        "data_source": data_source,
        "upload_mode": upload_mode,
        "uploaded_single": uploaded_single,
        "uploaded_multi": uploaded_multi,
        "files_have_ticker": files_have_ticker,
        "date_col": date_col,
        "price_col": price_col,
        "ticker_col": ticker_col,
        "alpaca_tickers": alpaca_tickers,
        "alpaca_start": alpaca_start,
        "alpaca_end": alpaca_end,
        "alpaca_api_key": alpaca_api_key,
        "alpaca_api_secret": alpaca_api_secret,
        "horizon": horizon,
        "resample_rule": resample_rule,
        "mode": mode,
        "checkpoint_path": checkpoint_path,
        "top_k": top_k,
        "episodes": episodes,
        "lr": float(lr),
        "risk_aversion": float(risk_aversion),
        "device": device_choice,
        "cpu_threads": cpu_threads,
        "rl_mode": rl_mode,
        "rl_hidden_dim": int(hidden_dim),
        "rl_patience": int(rl_patience) if rl_patience > 0 else None,
        "rl_min_delta": float(rl_min_delta),
        "forecast_params": {
            "stats": {"SeasonalNaive": {"season_length": int(sf_season)}},
            "ml": {
                "rf_params": {
                    "n_estimators": int(rf_estimators),
                    "max_depth": None if int(rf_max_depth) == 0 else int(rf_max_depth),
                }
            },
        },
    }


def plot_rewards(rewards: List[float]):
    px = _get_plotly_express()
    df = pd.DataFrame({"episode": range(1, len(rewards) + 1), "reward": rewards})
    fig = px.line(df, x="episode", y="reward", title="RL Training Reward per Episode")
    st.plotly_chart(fig, use_container_width=True)


def plot_prices(prices: pd.DataFrame):
    px = _get_plotly_express()
    fig = px.line(prices, x="ds", y="close", color="ticker", title="Historical Prices")
    st.plotly_chart(fig, use_container_width=True)


def plot_weights(weights_df: pd.DataFrame, title: str):
    px = _get_plotly_express()
    fig = px.bar(weights_df, x="asset", y="weight", color="asset", title=title)
    st.plotly_chart(fig, use_container_width=True)


def plot_matrix(matrix: pd.DataFrame, title: str):
    px = _get_plotly_express()
    fig = px.imshow(matrix, title=title, color_continuous_scale="RdBu_r")
    st.plotly_chart(fig, use_container_width=True)


def _format_alpaca_prices(raw: pd.DataFrame) -> pd.DataFrame:
    format_price_frame, _, _, _ = _get_portfolio_helpers()
    renamed = raw.rename(columns={"timestamp": "ds", "symbol": "ticker"})
    return format_price_frame(renamed, date_col="ds", price_col="close", ticker_col="ticker")


def _render_metric_group(title: str, metrics: dict):
    st.markdown(f"**{title}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Horizon Return", f"{metrics['expected_horizon_return']:.4f}")
    c2.metric("Expected Volatility", f"{metrics['expected_volatility']:.4f}")
    c3.metric("Effective Assets", f"{metrics['effective_n_assets']:.2f}")


def render_page():
    st.title("Portfolio Optimizer")
    st.write(
        "Generate validated asset-return forecasts, compare them to a deterministic mean-variance baseline, "
        "and optionally train or reuse an RL allocator."
    )
    st.info("CPU is the default path. RL will run on CPU when torch is installed; otherwise the baseline optimizer still works.")

    settings = sidebar_controls()
    resource_monitor()

    if settings["mode"].startswith("Use pre-trained") and not Path(settings["checkpoint_path"]).exists():
        st.warning(f"Checkpoint not found at {settings['checkpoint_path']}. Inference mode will fail until a checkpoint exists.")

    if st.button("Run Optimizer", type="primary"):
        with st.spinner("Preparing data and running the optimizer..."):
            try:
                format_price_frame, load_prices_from_files, optimize_portfolio, optimize_portfolio_inference = _get_portfolio_helpers()
                if settings["data_source"] == "Upload CSV/Files":
                    if settings["upload_mode"] == "Single CSV (all tickers)":
                        if not settings["uploaded_single"]:
                            st.error("Please upload a multi-asset price CSV.")
                            return
                        df_raw = pd.read_csv(settings["uploaded_single"])
                        prices = format_price_frame(
                            df_raw,
                            date_col=settings["date_col"],
                            price_col=settings["price_col"],
                            ticker_col=settings["ticker_col"],
                        )
                    else:
                        if not settings["uploaded_multi"]:
                            st.error("Upload at least one ticker CSV in multi-file mode.")
                            return
                        prices = load_prices_from_files(
                            settings["uploaded_multi"],
                            date_col=settings["date_col"],
                            price_col=settings["price_col"],
                            ticker_col=settings["ticker_col"] if settings["files_have_ticker"] else None,
                        )
                else:
                    symbols = [t.strip().upper() for t in settings["alpaca_tickers"].split(",") if t.strip()]
                    if not symbols:
                        st.error("Provide at least one ticker for Alpaca fetch.")
                        return
                    configure_alpaca_credentials, fetch_daily_bars, fetch_intraday_bars = _get_alpaca_helpers()
                    configure_alpaca_credentials(
                        api_key=settings["alpaca_api_key"],
                        api_secret=settings["alpaca_api_secret"],
                        persist_env=True,
                    )
                    if settings["data_source"] == "Fetch via Alpaca (daily)":
                        raw = fetch_daily_bars(symbols, start=settings["alpaca_start"], end=settings["alpaca_end"])
                    else:
                        raw = fetch_intraday_bars(symbols, start=settings["alpaca_start"], end=settings["alpaca_end"])
                    if raw.empty:
                        st.error("No data returned from Alpaca for the requested range.")
                        return
                    prices = _format_alpaca_prices(raw)
            except Exception as e:
                st.error(f"Failed to load price data: {e}")
                return

            st.success(f"Loaded {len(prices)} rows across {prices['ticker'].nunique()} tickers.")
            coverage_days = max(1, (prices["ds"].max() - prices["ds"].min()).days)
            if coverage_days < 365:
                st.warning("Data spans less than ~1 year. Forecast validation and optimizer stability will be limited.")
            st.dataframe(prices.head())
            plot_prices(prices)

            try:
                if settings["mode"].startswith("Use pre-trained"):
                    result = optimize_portfolio_inference(
                        prices=prices,
                        checkpoint_path=settings["checkpoint_path"],
                        horizon=settings["horizon"],
                        risk_aversion=settings["risk_aversion"],
                        date_col="ds",
                        price_col="close",
                        ticker_col="ticker",
                        resample_rule=settings["resample_rule"],
                        device=settings["device"],
                        rl_mode="graph" if settings["rl_mode"] == "Graph Trading" else "weights",
                        forecast_params=settings["forecast_params"],
                        cpu_threads=settings["cpu_threads"],
                    )
                    st.success("Checkpoint loaded and inference completed.")
                else:
                    result = optimize_portfolio(
                        prices=prices,
                        horizon=settings["horizon"],
                        top_k=settings["top_k"],
                        episodes=settings["episodes"],
                        lr=settings["lr"],
                        risk_aversion=settings["risk_aversion"],
                        checkpoint_path=settings["checkpoint_path"],
                        date_col="ds",
                        price_col="close",
                        ticker_col="ticker",
                        resample_rule=settings["resample_rule"],
                        device=settings["device"],
                        rl_mode="graph" if settings["rl_mode"] == "Graph Trading" else "weights",
                        forecast_params=settings["forecast_params"],
                        rl_hidden_dim=settings["rl_hidden_dim"],
                        cpu_threads=settings["cpu_threads"],
                        rl_patience=settings["rl_patience"],
                        rl_min_delta=settings["rl_min_delta"],
                    )
                    st.success("Optimization completed.")
            except ImportError as e:
                st.error(f"RL dependencies are unavailable: {e}")
                return
            except Exception as e:
                st.error(f"Optimization failed: {e}")
                return

            if result.get("missing_assets"):
                st.warning(f"Missing assets in current data were zero-weighted during inference: {', '.join(result['missing_assets'])}")

            st.caption(f"Optimizer engine used: `{result['optimizer_engine']}`")

            st.subheader("Forecast Summary")
            st.dataframe(
                result["forecast_summary"][
                    ["unique_id", "model", "expected_return", "mean_forecast_return", "forecast_volatility", "validation_rmse"]
                ]
            )

            st.subheader("Portfolio Metrics")
            left, right = st.columns(2)
            with left:
                _render_metric_group("Recommended Portfolio", result["portfolio_metrics"])
            with right:
                _render_metric_group("Mean-Variance Baseline", result["baseline_metrics"])

            st.subheader("Portfolio Weights")
            tab1, tab2 = st.tabs(["Recommended", "Baseline"])
            with tab1:
                st.dataframe(result["portfolio"])
                plot_weights(result["portfolio"], "Recommended Portfolio Weights")
            with tab2:
                st.dataframe(result["baseline_portfolio"])
                plot_weights(result["baseline_portfolio"], "Mean-Variance Baseline Weights")

            if result.get("covariance_matrix") is not None:
                st.subheader("Covariance Matrix")
                plot_matrix(result["covariance_matrix"], "Historical Return Covariance")

            if result.get("top_portfolios") is not None:
                st.subheader("Top Graph Rollouts")
                top_rollouts = result["top_portfolios"].copy()
                top_rollouts["weights"] = top_rollouts["weights"].apply(lambda w: np.round(w, 4))
                st.dataframe(top_rollouts)

            reward_history = result.get("reward_history", [])
            if reward_history:
                st.subheader("Training Rewards")
                plot_rewards(reward_history)

            if result.get("action_log") is not None and not result["action_log"].empty:
                st.subheader("Action Log")
                st.dataframe(result["action_log"].head(50))

            if result.get("checkpoint_meta"):
                with st.expander("Checkpoint Metadata"):
                    st.json(result["checkpoint_meta"])


if __name__ == "__main__":
    render_page()
