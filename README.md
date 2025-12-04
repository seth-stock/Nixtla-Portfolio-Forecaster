# Advanced Nixtla Forecasting App (Track 2)

Production-style Streamlit app built on Nixtla’s **statsforecast**, **mlforecast**, and **neuralforecast** plus an RL portfolio optimizer. Designed for Track 2 requirements: modular code, multi-model coverage, backtesting, metrics, config save/load, and deployment readiness.

## Quick Start
1) Python 3.12 (GPU optional). If you need CUDA, install the matching torch build separately.  
2) Install deps (preferably in conda environment):
```bash
pip install -r requirements.txt
```
3) Run locally(from the forecastapp folder):
```bash
streamlit run app.py
```
4) Open http://localhost:8501.

## How to Use (Forecasting Page)
- Upload a CSV (or fetch via Alpaca). Select **date column**, **target column**, and **ticker column** (creates `unique_id` per ticker).  
- Choose missing-value strategy (drop/ffill/bfill/interpolate), date filters, horizon, frequency (auto/default), purge/embargo, and lookbacks.  
- Pick models: StatsForecast (AutoARIMA, AutoETS, SeasonalNaive), MLForecast (RandomForest), NeuralForecast (RNN/LSTM/S5).  
- Click **Run Forecasts** to:
  - Clean/sort per ticker, infer frequency, apply temporal train/val/test split with purge+embargo.  
  - Fit per-ticker models, run rolling backtests, and generate holdout forecasts.  
  - Show metrics (RMSE/MAE/MAPE/sMAPE), backtest summary, actual vs forecasts, and top-3 model overlays per ticker.  
- Save/load settings via `configs/models_config.json`.

## How to Use (Portfolio Optimizer - structure is ready - needs to be re-trained for continuous and agentic mode)
- Upload multi-ticker prices (single CSV or many files) or fetch via Alpaca.  
- Provide horizon, resample rule, and RL checkpoint path.  
- Inference mode loads saved policy; training mode (local GPU) can fine-tune. Outputs weights, top rollouts, and reward traces.
- One variation simply proposes weights within a set number of holdings. Another variation acts on buy sell hold actions in a graph environment at each timestep.

## Architecture (Modular Nixtla)
- `core/data_loading.py` — CSV/Alpaca ingest, per-ticker cleaning, frequency inference, temporal split with purge/embargo.  
- `core/models_statsforecast.py` — AutoARIMA/AutoETS/SeasonalNaive fit/forecast + backtesting.  
- `core/models_mlforecast.py` — RandomForest with lags/rolling features, differencing, per-ticker threading, backtests.  
- `core/models_neuralforecast.py` — RNN/LSTM/S5 wrappers with device selection.  
- `core/evaluation.py` — RMSE/MAE/MAPE/sMAPE for holdout + CV summaries.  
- `core/config.py` — JSON save/load for reproducible runs.  
- `pages/1_Forecasting.py` — Streamlit UI for Track 2 forecasting.  
- `pages/2_Portfolio_Optimizer.py` — RL portfolio UI.  
- `requirements.txt` — All Streamlit + Nixtla dependencies.

## Models & Coverage (Track 2)
- **statsforecast**: AutoARIMA, AutoETS, SeasonalNaive (≥2 classical models).  
- **mlforecast**: RandomForestRegressor (tree-based ML with lag/rolling feats).  
- **neuralforecast**: RNN and LSTM (deep learning) plus optional S5.  


## Data Handling & Splits
- Date parsing, sorting, and per-ticker `unique_id` assignment (selectable ticker column on CSV uploads).  
- Missing handling: drop/ffill/bfill/interpolate.  
- Optional date filters and lookback windows.  
- Temporal train/val/test split with purge and embargo; backtests use rolling windows.

## Evaluation & Visualization
- Metrics: RMSE, MAE, MAPE, sMAPE for holdout; backtest summaries aggregated per model (and per ticker).  
- Plots: Actual vs Forecast per ticker with split markers; CV segments labeled; Top-3 model overlays.  
- Tables: Holdout metrics and rolling backtest summary.

## Deployment
- Streamlit Cloud ready: set entrypoint to `app.py`; Python 3.12.  
- Set `ALPACA_API_KEY` / `ALPACA_API_SECRET` in the configurable menus if using Alpaca.  
- Optional GPU locally; cloud will run CPU.

## Notes
- Streamlit cloud deployment will not occur until multi-core cpu support is enabled for model training. Until then, it's a cuda-only feature stack. 
- Increase epochs/trees/context lengths for higher fidelity; defaults are speed-focused.  
- Ensure enough history for selected horizon and backtest windows.  
- RL training is optional and best done locally with GPU and long histories.
