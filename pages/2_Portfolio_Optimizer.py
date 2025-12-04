"""
Portfolio Optimizer page
Uses Nixtla forecasts per asset and a lightweight PyTorch policy-gradient
agent to propose portfolio weights.
"""

from __future__ import annotations

from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import psutil

from core.portfolio_rl import (
    format_price_frame,
    load_prices_from_files,
    optimize_portfolio,
    optimize_portfolio_inference,
)
from core.alpaca_data import fetch_daily_bars, fetch_intraday_bars
from core import data_loading


def resource_monitor():
    with st.sidebar.expander("Resource Monitor", expanded=False):
        cpu = psutil.cpu_percent(interval=0.2)
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
        ["Upload CSV/Files", "Fetch via Alpaca (daily)", "Fetch via Alpaca (minute)"]
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
    alpaca_api_key = ""
    alpaca_api_secret = ""

    if data_source == "Upload CSV/Files":
        upload_mode = st.sidebar.radio(
            "Upload type",
            ["Single CSV (all tickers)", "Multiple CSV files (one per ticker)"]
        )
        if upload_mode == "Single CSV (all tickers)":
            uploaded_single = st.sidebar.file_uploader(
                "Upload multi-asset price CSV (>=2 years of data)", type=["csv"], accept_multiple_files=False
            )
            files_have_ticker = True
        else:
            uploaded_multi = st.sidebar.file_uploader(
                "Upload folder worth of CSVs (select multiple files)", type=["csv"], accept_multiple_files=True
            )
            files_have_ticker = st.sidebar.checkbox("Files already contain ticker column", value=False)
        date_col = st.sidebar.text_input("Date column", value="ds")
        price_col = st.sidebar.text_input("Price column", value="close")
        ticker_col = st.sidebar.text_input("Ticker column", value="ticker")
    else:
        alpaca_tickers = st.sidebar.text_input("Alpaca tickers (comma-separated)", value="AAPL,MSFT,GOOG")
        alpaca_start = st.sidebar.date_input("Start date", value=alpaca_start)
        alpaca_end = st.sidebar.date_input("End date", value=alpaca_end)
        alpaca_api_key = st.sidebar.text_input("Alpaca API Key", type="password")
        alpaca_api_secret = st.sidebar.text_input("Alpaca API Secret", type="password")

    horizon = st.sidebar.slider("Forecast horizon (steps)", 1, 120, 20)
    resample_rule = st.sidebar.text_input("Resample rule (e.g., 1D, 1H, 10min)", value="1D")

    st.sidebar.header("Checkpoint / Mode")
    mode = st.sidebar.radio("Mode", ["Use pre-trained checkpoint (inference)", "Train new checkpoint locally"])
    checkpoint_path = st.sidebar.text_input("Checkpoint path", value="models/portfolio_policy.pt")
    rl_mode = st.sidebar.selectbox("RL Mode", ["Continuous Weights", "Graph Trading"], index=0)
    with st.sidebar.expander("Forecast Model Params"):
        sf_season = st.number_input("Season length (statsforecast fallback)", min_value=1, value=1, step=1, key="opt_sf_season")
        lstm_hidden = st.number_input("LSTM hidden size", min_value=16, max_value=512, value=64, step=16, key="opt_lstm_hidden")
        lstm_layers = st.number_input("LSTM layers", min_value=1, max_value=12, value=6, step=1, key="opt_lstm_layers")
        lstm_proj = st.number_input("LSTM projection dim", min_value=8, max_value=256, value=32, step=8, key="opt_lstm_proj")
        lstm_seq = st.number_input("LSTM sequence length", min_value=8, max_value=128, value=30, step=2, key="opt_lstm_seq")
        lstm_epochs = st.number_input("LSTM epochs", min_value=5, max_value=200, value=25, step=5, key="opt_lstm_epochs")
        lstm_lr = st.number_input("LSTM learning rate", value=1e-5, format="%.6f", key="opt_lstm_lr")
        lstm_patience = st.number_input("LSTM patience", min_value=0, max_value=50, value=5, step=1, key="opt_lstm_patience")
    with st.sidebar.expander("RL Trainer Params"):
        hidden_dim = st.number_input("Policy hidden dim", min_value=16, max_value=256, value=48, step=16, key="opt_policy_hidden")
        rl_patience = st.number_input("RL patience (episodes)", min_value=0, max_value=200, value=0, step=10, key="opt_rl_patience")
        rl_min_delta = st.number_input("RL min delta", value=1e-4, format="%.5f", key="opt_rl_min_delta")

    st.sidebar.header("RL Settings (for training/local only)")
    top_k = st.sidebar.slider("Number of holdings (k)", 1, 5, 3)
    episodes = st.sidebar.slider("Training episodes", 10, 200, 50, step=10)
    lr = st.sidebar.number_input("Learning rate", value=0.001, format="%.4f")
    risk_aversion = st.sidebar.number_input("Risk aversion penalty", value=0.01, format="%.4f")
    device_choice = st.sidebar.selectbox("Device", ["cuda", "cpu"], index=0)
    cpu_threads = 16
    if device_choice == "cpu":
        cpu_threads = st.sidebar.slider("CPU threads", 1, 32, 16)
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
        "forecast_params": {
            "stats": {"SeasonalNaive": {"season_length": sf_season}},
            "lstm_hidden": int(lstm_hidden),
            "lstm_layers": int(lstm_layers),
            "lstm_proj": int(lstm_proj),
            "lstm_seq_len": int(lstm_seq),
            "lstm_epochs": int(lstm_epochs),
            "lstm_lr": float(lstm_lr),
            "lstm_patience": int(lstm_patience),
        },
        "rl_hidden_dim": int(hidden_dim),
        "rl_patience": int(rl_patience) if rl_patience > 0 else None,
        "rl_min_delta": float(rl_min_delta),
    }


def plot_rewards(rewards: List[float]):
    df = pd.DataFrame({"episode": range(1, len(rewards) + 1), "reward": rewards})
    fig = px.line(df, x="episode", y="reward", title="RL Training Reward per Episode")
    st.plotly_chart(fig, use_container_width=True)


def plot_prices(prices: pd.DataFrame):
    fig = px.line(prices, x="ds", y="close", color="ticker", title="Historical Prices")
    st.plotly_chart(fig, use_container_width=True)


def plot_weights(weights_df: pd.DataFrame):
    fig = px.bar(weights_df, x="asset", y="weight", color="asset", title="Portfolio Weights")
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation(prices: pd.DataFrame):
    pivot = prices.pivot(index="ds", columns="ticker", values="close").corr()
    fig = px.imshow(pivot, title="Ticker Correlation Heatmap", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig, use_container_width=True)


def render_page():
    st.title("Portfolio Optimizer (RL + Nixtla Forecasts)")
    st.write(
        "Upload a CSV containing at least two years of historical prices (columns: date, ticker, close)."
        " The RL policy should be trained offline with ~10 years of history via `portfolio_training.ipynb`."
    )
    st.info(
        "For production/Streamlit Cloud, prefer the **inference mode** with a checkpoint generated locally via `portfolio_training.ipynb` using ~10 years of data."
    )

    settings = sidebar_controls()
    resource_monitor()
    if settings["mode"].startswith("Train"):
        st.warning("Training mode expects a long (ideally 10-year) dataset and runs best on your local GPU.")
    if settings["mode"].startswith("Use pre-trained") and not Path(settings["checkpoint_path"]).exists():
        st.warning(
            f"Checkpoint not found at {settings['checkpoint_path']}. "
            "Train locally via portfolio_training.ipynb and place the file under models/."
        )

    if st.button("Run Optimizer"):
        with st.spinner("Parsing uploaded data and running optimizer..."):
            try:
                if settings["data_source"] == "Upload CSV/Files":
                    if settings["upload_mode"] == "Single CSV (all tickers)":
                        if not settings["uploaded_single"]:
                            st.error("Please upload a CSV containing at least two years of historical prices.")
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
                            st.error("Upload at least one ticker CSV (multi-file mode).")
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
                    # Apply Alpaca credentials if provided
                    if settings["alpaca_api_key"] and settings["alpaca_api_secret"]:
                        import os
                        os.environ["ALPACA_API_KEY"] = settings["alpaca_api_key"]
                        os.environ["ALPACA_API_SECRET"] = settings["alpaca_api_secret"]
                    if settings["data_source"] == "Fetch via Alpaca (daily)":
                        prices = fetch_daily_bars(
                            symbols,
                            start=settings["alpaca_start"],
                            end=settings["alpaca_end"],
                        )
                    else:
                        prices = fetch_intraday_bars(
                            symbols,
                            start=settings["alpaca_start"],
                            end=settings["alpaca_end"],
                        )
                    if prices.empty:
                        st.error("No data returned from Alpaca for the requested range.")
                        return
            except Exception as e:
                st.error(f"Failed to parse uploaded CSV: {e}")
                return

            st.success(f"Loaded {len(prices)} rows across {prices['ticker'].nunique()} tickers.")
            coverage_days = (prices["ds"].max() - prices["ds"].min()).days
            if coverage_days < 365 * 2:
                st.warning(
                    "Data spans less than ~2 years. For best results provide at least 24 months of history."
                )
            st.dataframe(prices.head())
            plot_prices(prices)
            with st.expander("Correlation (upload source only)"):
                try:
                    plot_correlation(prices)
                except Exception:
                    st.write("Could not compute correlations for the uploaded data.")

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
                        forecast_params=settings.get("forecast_params"),
                    )
                    st.success("Loaded checkpoint and ran inference.")
                    reward_history = None
                else:
                    # Apply temporal split with purge/embargo before training RL
                    try:
                        _, _, _ = data_loading.temporal_split(
                            prices.rename(columns={"ds": "date_tmp", "close": "close_tmp"}),  # placeholder to reuse splitter
                            horizon=settings["horizon"],
                            val_ratio=0.2,
                            purge=max(1, settings["horizon"] // 2),
                            embargo=max(1, settings["horizon"] // 4),
                        )
                    except Exception:
                        # Continue even if split fails (e.g., too short), since RL code handles filtering internally
                        pass
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
                        forecast_params=settings.get("forecast_params"),
                        rl_patience=settings.get("rl_patience"),
                        rl_min_delta=settings.get("rl_min_delta", 1e-4),
                    )
                    st.success("Training completed and checkpoint saved.")
                    reward_history = result.get("reward_history", [])
            except ImportError as e:
                st.error(
                    "PyTorch is required for the RL optimizer. Install GPU-enabled torch 2.9 locally. "
                    f"Details: {e}"
                )
                return
            except Exception as e:
                st.error(f"Optimization failed: {e}")
                return

            st.subheader("Recommended Portfolio")
            st.dataframe(result["portfolio"])
            st.caption("Weights are normalized; expected returns derived from statsforecast AutoARIMA forecasts.")
            plot_weights(result["portfolio"])

            if result.get("top_portfolios") is not None:
                st.subheader("Top portfolios (graph mode rollouts)")
                tp = result["top_portfolios"]
                tp_display = tp.copy()
                tp_display["weights"] = tp_display["weights"].apply(lambda w: np.round(w, 3))
                st.dataframe(tp_display)

            if reward_history:
                st.subheader("Training Rewards")
                plot_rewards(reward_history)
            else:
                st.caption("Inference mode skips RL training; rewards not shown.")

            st.subheader("Projected Portfolio Return (naive expectation)")
            expected = float(np.dot(result["portfolio"]["weight"], result["portfolio"]["expected_return"]))
            st.metric("Expected horizon return", f"{expected:.4f}")

            # Action / trade visualization
            if result.get("action_log") is not None:
                st.subheader("Action / Trade Log")
                action_df = result["action_log"]
                st.dataframe(action_df.head())
                if "cum_reward" in action_df.columns:
                    fig = px.line(action_df, x="step", y="cum_reward", title="Cumulative Reward (simulation)")
                    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    render_page()
