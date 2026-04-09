# Nixtla Portfolio Forecaster

Streamlit application for time-series forecasting and portfolio construction using Nixtla's `StatsForecast`, `MLForecast`, and `NeuralForecast` tooling, plus an optional lightweight RL allocator for allocation experiments.

## GitHub Scope

This GitHub-facing refresh is for the Streamlit source app only. The Windows desktop build that is under local development is intentionally out of scope for this README and the root `requirements.txt`.

## What It Does

- Upload single-series or multi-asset CSV data.
- Pull daily or intraday market data from Alpaca.
- Compare statistical, tree-based, and neural forecasting models on time-based holdouts.
- Review forecast metrics, charts, and backtest outputs in Streamlit.
- Build portfolio allocations from validated asset-return forecasts.
- Reuse or retrain an RL checkpoint for allocation experiments.

## Environment

These instructions are based on the Anaconda data-science environment available on this machine:

- Python 3.12
- Anaconda env family: `datascience312`

If your registered Jupyter or Conda environment is named slightly differently, use the matching interpreter from your local `datascience` workflow.

## Quick Start

```bash
conda activate datascience312
cd /d C:\Nixtla-Portfolio-Forecaster
pip install -r requirements.txt
streamlit run streamlit_app.py
```

You can also launch the app with `launch_streamlit.bat`.

## Repository Layout

- `streamlit_app.py`: source entrypoint for the Streamlit app
- `pages/1_Forecasting.py`: forecasting workflow UI
- `pages/2_Portfolio_Optimizer.py`: optimizer workflow UI
- `core/`: forecasting wrappers, evaluation helpers, data loading, and portfolio logic
- `configs/models_config.json`: saved example configuration
- `portfolio_training.ipynb`: supporting notebook for portfolio training experiments
- `PORTFOLIO_TRAINING_GUIDE.md`: training notes and workflow details

## Notes

- Alpaca credentials can be entered in the app or provided through `ALPACA_API_KEY` and `ALPACA_API_SECRET`.
- The neural and RL paths require a working PyTorch install.
- GPU usage is optional. The Streamlit app can run in CPU-only mode.
