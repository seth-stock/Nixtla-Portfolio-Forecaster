# Portfolio Training Guide

This guide covers how to train the portfolio optimizer checkpoints from `portfolio_training.ipynb`.

## 1. Start the Environment

Open Anaconda Prompt and run:

```bash
conda activate forecastapp
cd /d C:\Nixtla-Portfolio-Forecaster
pip install -r requirements.txt
jupyter lab
```

Open `portfolio_training.ipynb` in Jupyter Lab.

## 2. Choose a Data Source

The notebook supports two paths:

- `csv`: preferred for real training runs with a long local history.
- `alpaca`: useful for smaller experiments or quick retraining.

Edit the configuration cell near the top of the notebook:

```python
DATA_SOURCE = 'csv'      # or 'alpaca'
DATA_FILE = ROOT / 'data' / 'bulk_prices' / 'ten_year_prices2.csv'
DATE_COL = 'Date'
PRICE_COL = 'Close'
TICKER_COL = 'Ticker'
```

If you use Alpaca, configure:

```python
ALPACA_TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA']
ALPACA_START = pd.Timestamp.today() - pd.DateOffset(months=6)
ALPACA_END = pd.Timestamp.today()
ALPACA_API_KEY = ''
ALPACA_API_SECRET = ''
```

You can either:

- paste your Alpaca key and secret into those two notebook variables for local use, or
- leave them blank and set `ALPACA_API_KEY` / `ALPACA_API_SECRET` in the shell before launching Jupyter.

Do not commit live credentials.

## 3. Set Training Parameters

The main controls are in the same config cell:

- `TRAIN_SPECS`: defines the resample cadence and checkpoint names.
- `TOP_K`: number of holdings in the final portfolio.
- `EPISODES`: RL training episodes.
- `LR`: optimizer learning rate.
- `RISK_AVERSION`: volatility penalty.
- `DEVICE`: defaults to `cpu`; only switch to `cuda` after confirming your CUDA stack works.
- `CPU_THREADS`: CPU worker count for training and forecast helpers.
- `RL_HIDDEN_DIM`: policy network width.
- `RL_PATIENCE` and `RL_MIN_DELTA`: early stopping controls.

Default checkpoints created by the notebook:

- `models/portfolio_policy_10m.pt`
- `models/portfolio_policy_60m.pt`
- `models/portfolio_policy_1440m.pt`
- `models/portfolio_policy_10m_graph.pt`
- `models/portfolio_policy_60m_graph.pt`
- `models/portfolio_policy_1440m_graph.pt`

## 4. Run the Notebook

Run the cells in order from top to bottom.

The notebook flow is:

1. Import dependencies and project modules.
2. Set the training configuration.
3. Define helper functions.
4. Load and normalize prices.
5. Train continuous-weight checkpoints.
6. Inspect one continuous-weight inference result.
7. Train graph-policy checkpoints.
8. Inspect one graph-policy inference result.
9. Save the combined summary CSV.

## 5. Outputs

The notebook writes:

- model checkpoints into `models/`
- a training summary into `data/training_runs/portfolio_training_summary.csv`

Each training job also runs an inference round-trip immediately after saving the checkpoint. That verifies the checkpoint can be reloaded by the application.

## 6. Use the Checkpoints in the App

Start the desktop app:

```bash
python app.py
```

Then open the Portfolio Optimizer screen and:

- choose inference mode
- point `Checkpoint path` at one of the saved `.pt` files
- choose `Continuous Weights` for normal `.pt` checkpoints
- choose `Graph Trading` for `_graph.pt` checkpoints

## 7. Troubleshooting

- If local CSV loading fails, check `DATA_FILE`, `DATE_COL`, `PRICE_COL`, and `TICKER_COL`.
- If Alpaca loading fails, verify the key, secret, ticker list, and date window.
- If training falls back to `HistoricalMean`, your Nixtla runtime dependencies are not fully healthy in this environment.
- If you want GPU training, switch `DEVICE` to `cuda` only after confirming `torch.cuda.is_available()` is `True` and the CUDA runtime is actually stable.
- If repeated Alpaca fetches are slow, keep the cache file so later notebook runs reuse the downloaded data.
