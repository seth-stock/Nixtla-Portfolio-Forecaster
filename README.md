# Advanced Nixtla Forecasting App

Streamlit app for time-series forecasting and portfolio construction using Nixtla libraries.

## Quick Start
1. Use Python 3.12.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the app:
```bash
streamlit run app.py
```

The default install path is CPU-safe. If you want CUDA acceleration, replace the default `torch` wheel with the matching GPU build for your system after installing the requirements.

## Forecasting Page
- Upload a CSV or fetch price data from Alpaca.
- Choose the date, target, and optional ticker columns.
- Compare `StatsForecast`, `MLForecast`, and `NeuralForecast` models on a time-based split.
- Save and reload settings from `configs/models_config.json`.

Current model coverage:
- `StatsForecast`: `AutoARIMA`, `AutoETS`, `SeasonalNaive`
- `MLForecast`: `RandomForest`
- `NeuralForecast`: `RNN`, `LSTM`, optional `S5`

Methodology:
- Data is standardized per series into `unique_id`, `ds`, `y`.
- Holdout evaluation uses time-ordered train/validation/test segments with purge and embargo buffers.
- Backtests use rolling-origin cross-validation.
- Direct vs recursive forecasting is wired through the ML and neural model wrappers.

## Portfolio Optimizer
- Accepts uploaded multi-asset CSVs or Alpaca price history.
- Builds per-asset return forecasts with validated Nixtla models.
- Produces a deterministic mean-variance baseline portfolio.
- Optionally trains or reuses an RL policy for weight allocation or graph-style buy/hold/sell actions.

Outputs:
- Forecast summary per asset
- Recommended portfolio weights
- Mean-variance baseline weights
- Covariance matrix
- RL reward trace and action log when RL is used

Notes:
- CPU is the default runtime path.
- RL training requires `torch`. If `torch` is unavailable, the training path falls back to the deterministic baseline.
- Inference from a saved RL checkpoint still requires `torch`.

## Project Structure
- `app.py`: Streamlit landing page
- `pages/1_Forecasting.py`: forecasting interface
- `pages/2_Portfolio_Optimizer.py`: portfolio interface
- `core/data_loading.py`: ingestion, frequency inference, temporal splits
- `core/models_statsforecast.py`: classical models
- `core/models_mlforecast.py`: tree-based models
- `core/models_neuralforecast.py`: neural recurrent models
- `core/portfolio_rl.py`: portfolio forecasting, mean-variance baseline, RL policies
- `configs/models_config.json`: example saved settings
