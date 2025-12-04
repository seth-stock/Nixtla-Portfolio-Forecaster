"""
portfolio_rl.py
Portfolio optimization utilities combining Nixtla forecasts with a simple
policy-gradient reinforcement learning agent implemented in PyTorch.
Torch is imported lazily to avoid failing at module import time if the
environment has an incompatible PyTorch install.
"""

from __future__ import annotations

import logging
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
_TORCH_IMPORT_ERROR: Optional[Exception] = None
_TORCH_CACHE: Dict[str, Any] = {}
_SF_CACHE: Dict[str, Any] = {}


def _lazy_torch() -> Dict[str, Any]:
    """
    Import torch/nn/optim on-demand with a clear error if unavailable.
    Keep imports minimal to avoid circular import issues seen in some environments.
    """
    global _TORCH_IMPORT_ERROR
    if _TORCH_CACHE:
        return _TORCH_CACHE
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.optim as optim  # type: ignore
    except Exception as e:  # pragma: no cover - informative fallback
        _TORCH_IMPORT_ERROR = e
        raise ImportError(
            "PyTorch is required for the RL optimizer. Install GPU-enabled torch 2.9 "
            f"(original import error: {e})"
        )

    # Prefer CUDA for acceleration if available.
    device = "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
    try:
        torch.set_default_device(device)
    except Exception:
        pass
    try:
        torch.set_num_threads(16)
    except Exception:
        pass
    _TORCH_CACHE.update({"torch": torch, "nn": nn, "optim": optim, "device": device})
    return _TORCH_CACHE


def _lazy_statsforecast() -> Dict[str, Any]:
    """Import StatsForecast models on demand with clearer errors (pyarrow on Windows)."""
    if _SF_CACHE:
        return _SF_CACHE
    try:
        from statsforecast import StatsForecast  # type: ignore
        from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive  # type: ignore
    except Exception as e:
        raise ImportError(
            "StatsForecast (and its dependencies like pyarrow) is required. "
            "Install with `pip install statsforecast pyarrow fugue`."
        ) from e
    _SF_CACHE.update(
        {"StatsForecast": StatsForecast, "AutoARIMA": AutoARIMA, "AutoETS": AutoETS, "SeasonalNaive": SeasonalNaive}
    )
    return _SF_CACHE


def format_price_frame(
    df: pd.DataFrame,
    date_col: str = "ds",
    price_col: str = "close",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Validate and standardize a price dataframe provided by the user."""
    required = {date_col, price_col, ticker_col}
    if not required.issubset(df.columns):
        raise ValueError(f"Price data must include columns: {required}")
    formatted = df[[date_col, ticker_col, price_col]].copy()
    formatted = formatted.rename(columns={date_col: "ds", ticker_col: "ticker", price_col: "close"})
    formatted["ds"] = pd.to_datetime(formatted["ds"], errors="coerce")
    formatted = formatted.dropna(subset=["ds", "close", "ticker"])
    formatted = formatted.sort_values(["ticker", "ds"]).reset_index(drop=True)
    return formatted


def clean_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop infs, remove duplicates, forward-fill missing close."""
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
    df = df.drop_duplicates(subset=["ticker", "ds"])
    df["close"] = df.groupby("ticker")["close"].ffill()
    df = df.dropna(subset=["close"])
    return df


def resample_prices(prices: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample price data to a new frequency rule (e.g., '10min','1H','1D','M','Q').
    Uses last close in each period.
    """
    prices = prices.copy()
    prices["ds"] = pd.to_datetime(prices["ds"])
    frames = []
    for ticker, grp in prices.groupby("ticker"):
        grp = grp.set_index("ds").sort_index()
        res = grp["close"].resample(rule).last().dropna()
        frames.append(pd.DataFrame({"ds": res.index, "ticker": ticker, "close": res.values}))
    if frames:
        return pd.concat(frames, ignore_index=True).sort_values(["ticker", "ds"])
    return prices


def load_prices_from_files(
    files: List[Any],
    date_col: str = "Date",
    price_col: str = "Close",
    ticker_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Combine multiple CSV files (one per ticker) into a single standardized dataframe.
    If ticker_col is None, infer ticker names from filenames.
    """
    frames = []
    for idx, file in enumerate(files):
        name = getattr(file, "name", None)
        if name is None and isinstance(file, (str, Path)):
            name = Path(file).name
        ticker_name = Path(name).stem.upper() if name else f"TICKER_{idx}"
        df = pd.read_csv(file)
        if ticker_col is None or ticker_col not in df.columns:
            df = df.copy()
            df["__ticker"] = ticker_name
            use_ticker_col = "__ticker"
        else:
            use_ticker_col = ticker_col
        formatted = format_price_frame(df, date_col=date_col, price_col=price_col, ticker_col=use_ticker_col)
        frames.append(formatted)
    if not frames:
        raise ValueError("No CSV files provided for price loading.")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "ds"]).reset_index(drop=True)
    return combined


def load_prices_from_directory(
    directory: str | Path,
    pattern: str = "*.csv",
    date_col: str = "Date",
    price_col: str = "Close",
    ticker_col: Optional[str] = None,
) -> pd.DataFrame:
    """Load and combine all CSVs in a directory as price data."""
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = sorted(dir_path.glob(pattern))
    if not files:
        raise ValueError(f"No CSV files matching {pattern} found in {directory}")
    return load_prices_from_files(files, date_col=date_col, price_col=price_col, ticker_col=ticker_col)


def prepare_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute percentage returns per ticker."""
    prices = prices.sort_values(["ticker", "ds"])
    prices["y"] = prices.groupby("ticker")["close"].pct_change().fillna(0.0)
    returns = prices.rename(columns={"ticker": "unique_id"})
    return returns[["unique_id", "ds", "y"]]


def forecast_asset_returns(
    returns_df: pd.DataFrame,
    horizon: int,
    freq: str,
    device: Optional[str] = None,
    forecast_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Use StatsForecast (AutoARIMA, AutoETS), PyCaret compare_models, NeuralProphet, and LSTM
    to forecast returns per asset.
    Train/val/test split per asset (with purge/embargo), RMSE-based model selection.
    Returns dataframe with expected return per asset and the winning model name.
    """
    try:
        sf_mod = _lazy_statsforecast()
    except Exception as e:
        raise RuntimeError(
            "StatsForecast/pyarrow are required to forecast asset returns. Install with "
            "`pip install statsforecast pyarrow fugue`. Original error: {}".format(e)
        ) from e
    StatsForecast = sf_mod["StatsForecast"]
    AutoARIMA = sf_mod["AutoARIMA"]
    AutoETS = sf_mod["AutoETS"]
    SeasonalNaive = sf_mod["SeasonalNaive"]
    def _lazy_neuralprophet():
        try:
            from neuralprophet import NeuralProphet  # type: ignore
            return NeuralProphet
        except Exception as e:
            raise ImportError("neuralprophet is required; install neuralprophet to enable.") from e

    def _lazy_s5():
        try:
            from s5_pytorch import S5Layer  # type: ignore
            return S5Layer
        except Exception as e:
            raise ImportError("s5-pytorch is required; install s5-pytorch to enable.") from e

    forecast_params = forecast_params or {}
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    nn = torch_mod["nn"]
    optim = torch_mod["optim"]
    # Prefer provided device; otherwise use cached default (CUDA if available).
    torch_device = torch_mod.get("device", "cpu")
    device = device or ("cuda" if torch_device == "cuda" and torch.cuda.is_available() else torch_device)

    freq = freq or "D"
    results = []
    purge = max(1, horizon // 2)
    embargo = max(1, horizon // 4)
    lstm_hidden = int(forecast_params.get("lstm_hidden", 64))
    lstm_layers = int(forecast_params.get("lstm_layers", 6))
    lstm_proj = int(forecast_params.get("lstm_proj", 32))
    lstm_seq_len = int(forecast_params.get("lstm_seq_len", 30))
    lstm_epochs = int(forecast_params.get("lstm_epochs", 25))
    lstm_lr = float(forecast_params.get("lstm_lr", 1e-5))

    def _strip_tz(series: pd.Series) -> pd.Series:
        """Ensure datetime series is tz-naive to avoid merge/type issues."""
        return pd.to_datetime(series).dt.tz_localize(None)

    class LSTMForecaster(nn.Module):
        def __init__(self, input_dim: int, horizon: int):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, lstm_hidden, num_layers=lstm_layers, batch_first=True)
            self.proj = nn.Sequential(
                nn.Linear(lstm_hidden, lstm_proj),
                nn.ReLU(),
                nn.Linear(lstm_proj, horizon),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.proj(last)

    for uid, group in returns_df.groupby("unique_id"):
        g = group.sort_values("ds").reset_index(drop=True)
        if len(g) <= horizon * 3:
            continue
        test = g.iloc[-horizon:]
        val = g.iloc[-(2 * horizon + embargo): -horizon]
        train = g.iloc[: -(2 * horizon + embargo + purge)]
        if len(train) <= horizon or len(val) < 1:
            continue

        # StatsForecast models
        sf_df = train.rename(columns={"ds": "ds", "y": "y"}).assign(unique_id=uid)
        sf = StatsForecast(
            models=[AutoARIMA(season_length=1), AutoETS(season_length=1)],
            freq=freq,
            n_jobs=16,
            fallback_model=SeasonalNaive(season_length=1),
        )
        sf.fit(sf_df)
        sf_fcst_val = sf.forecast(df=sf_df, h=len(val)).reset_index()
        sf_fcst_val = sf_fcst_val.melt(id_vars=["ds", "unique_id"], var_name="model", value_name="forecast")
        sf_fcst_val["ds"] = _strip_tz(sf_fcst_val["ds"])
        sf_fcst = sf.forecast(df=pd.concat([sf_df, val.rename(columns={"ds": "ds", "y": "y"}).assign(unique_id=uid)]), h=horizon).reset_index()
        sf_fcst = sf_fcst.melt(id_vars=["ds", "unique_id"], var_name="model", value_name="forecast")
        sf_fcst["ds"] = _strip_tz(sf_fcst["ds"])

        # LSTM model (simple supervised sequence forecast)
        series = train["y"].astype(float).values
        seq_len = lstm_seq_len
        X, y_target = [], []
        for i in range(len(series) - seq_len - horizon + 1):
            X.append(series[i : i + seq_len])
            y_target.append(series[i + seq_len : i + seq_len + horizon])
        X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1).to(device)
        y_target = torch.tensor(np.array(y_target), dtype=torch.float32).to(device)
        lstm_model = LSTMForecaster(input_dim=1, horizon=horizon).to(device)
        optimizer = optim.Adam(lstm_model.parameters(), lr=lstm_lr)
        loss_fn = nn.MSELoss()
        epochs = lstm_epochs
        best_loss = float("inf")
        patience = int(forecast_params.get("lstm_patience", 5))
        patience_ctr = 0
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = lstm_model(X)
            loss = loss_fn(pred, y_target)
            loss.backward()
            optimizer.step()
            curr = float(loss.detach().cpu())
            if curr + 1e-6 < best_loss:
                best_loss = curr
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break
        with torch.no_grad():
            last_seq = torch.tensor(series[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            lstm_pred = lstm_model(last_seq).detach().cpu().numpy().flatten()
        val_series = g["y"].astype(float).values[: -(horizon + embargo)]
        val_len = len(val)
        if len(val_series) >= seq_len:
            with torch.no_grad():
                val_last_seq = torch.tensor(val_series[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                raw_val_pred = lstm_model(val_last_seq).detach().cpu().numpy().flatten()
                # Ensure the validation forecast length matches the validation window
                if len(raw_val_pred) < val_len:
                    val_pred = np.resize(raw_val_pred, val_len)
                else:
                    val_pred = raw_val_pred[:val_len]
        else:
            val_pred = np.full(val_len, np.nan)
        lstm_df = pd.DataFrame({
            "ds": pd.date_range(start=test["ds"].min(), periods=horizon, freq=freq),
            "unique_id": uid,
            "model": "LSTM",
            "forecast": lstm_pred,
        })
        lstm_df["ds"] = _strip_tz(lstm_df["ds"])
        lstm_val_df = pd.DataFrame({
            "ds": val["ds"].values,
            "unique_id": uid,
            "model": "LSTM",
            "forecast": val_pred,
        })
        lstm_val_df["ds"] = _strip_tz(lstm_val_df["ds"])

        # NeuralProphet (optional)
        np_ok = True
        try:
            NeuralProphet = _lazy_neuralprophet()
            np_model = NeuralProphet(n_forecasts=1, n_lags=0)
            np_model.fit(train.rename(columns={"ds": "ds", "y": "y"}), freq=freq)
            np_val_fc = np_model.make_future_dataframe(train.rename(columns={"ds": "ds", "y": "y"}), periods=len(val))
            np_val_pred = np_model.predict(np_val_fc)
            np_val_df = pd.DataFrame({
                "ds": np_val_pred["ds"].values[-len(val):],
                "unique_id": uid,
                "model": "NeuralProphet",
                "forecast": np_val_pred[[c for c in np_val_pred.columns if c.startswith("yhat")]].mean(axis=1).values[-len(val):],
            })
            np_val_df["ds"] = _strip_tz(np_val_df["ds"])
            np_test_fc = np_model.make_future_dataframe(train.rename(columns={"ds": "ds", "y": "y"}), periods=horizon)
            np_test_pred = np_model.predict(np_test_fc)
            np_test_df = pd.DataFrame({
                "ds": np_test_pred["ds"].values[-horizon:],
                "unique_id": uid,
                "model": "NeuralProphet",
                "forecast": np_test_pred[[c for c in np_test_pred.columns if c.startswith("yhat")]].mean(axis=1).values[-horizon:],
            })
            np_test_df["ds"] = _strip_tz(np_test_df["ds"])
        except Exception:
            np_ok = False
            np_val_df = pd.DataFrame(columns=["ds", "unique_id", "model", "forecast"])
            np_test_df = pd.DataFrame(columns=["ds", "unique_id", "model", "forecast"])

        # S5 (optional)
        s5_ok = False
        try:
            S5Layer = _lazy_s5()
            class S5Wrapper(nn.Module):
                def __init__(self, input_dim: int, horizon: int, seq_len: int):
                    super().__init__()
                    self.s5 = S5Layer(d_model=32, l_max=seq_len)
                    self.proj = nn.Linear(32, horizon)

                def forward(self, x):
                    # x shape: (batch, seq_len, 1)
                    # expand to d_model
                    x = x.repeat(1, 1, 32)
                    out = self.s5(x)[0][:, -1, :]
                    return self.proj(out)

            X_s, y_s = [], []
            for i in range(len(series) - seq_len - horizon + 1):
                X_s.append(series[i : i + seq_len])
                y_s.append(series[i + seq_len : i + seq_len + horizon])
            X_s = torch.tensor(np.array(X_s), dtype=torch.float32).unsqueeze(-1).to(device)
            y_s = torch.tensor(np.array(y_s), dtype=torch.float32).to(device)
            s5_model = S5Wrapper(input_dim=16, horizon=horizon, seq_len=seq_len).to(device)
            opt_s = optim.Adam(s5_model.parameters(), lr=1e-5)
            loss_fn_s = nn.MSELoss()
            for _ in range(10):
                opt_s.zero_grad()
                pred = s5_model(X_s)
                loss_s = loss_fn_s(pred, y_s)
                loss_s.backward()
                opt_s.step()
            with torch.no_grad():
                last_seq_s = torch.tensor(series[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                s_pred = s5_model(last_seq_s).detach().cpu().numpy().flatten()
            s5_df = pd.DataFrame({
                "ds": pd.date_range(start=test["ds"].min(), periods=horizon, freq=freq),
                "unique_id": uid,
                "model": "S5",
                "forecast": s_pred,
            })
            s5_df["ds"] = _strip_tz(s5_df["ds"])
            s5_val_df = pd.DataFrame({
                "ds": val["ds"].values,
                "unique_id": uid,
                "model": "S5",
                "forecast": s_pred[: len(val)],
            })
            s5_val_df["ds"] = _strip_tz(s5_val_df["ds"])
        except Exception:
            s5_ok = False
            s5_df = pd.DataFrame(columns=["ds", "unique_id", "model", "forecast"])
            s5_val_df = pd.DataFrame(columns=["ds", "unique_id", "model", "forecast"])

        frames = [sf_fcst_val, lstm_val_df, np_val_df, s5_val_df]
        frames = [f for f in frames if not f.empty]
        if not frames:
            continue
        combined_val = pd.concat(frames, ignore_index=True)
        # Align datetime types (drop tz) before merge to avoid dtype mismatch
        combined_val["ds"] = _strip_tz(combined_val["ds"])
        val_aligned = val.copy()
        val_aligned["ds"] = _strip_tz(val_aligned["ds"])
        combined_val = combined_val.merge(val_aligned.rename(columns={"ds": "ds", "y": "actual"}), on="ds", how="left")

        best_model = None
        best_rmse = np.inf
        for model_name, grp in combined_val.groupby("model"):
            if grp.empty:
                continue
            rmse = float(np.sqrt(np.mean((grp["forecast"] - grp["actual"]) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model_name

        if best_model is None:
            continue
        if best_model == "LSTM":
            test_fc = lstm_df
        elif best_model == "NeuralProphet" and np_ok:
            test_fc = np_test_df
        elif best_model == "S5" and s5_ok:
            test_fc = s5_df
        else:
            test_fc = sf_fcst[sf_fcst["model"] == best_model]

        expected_return = float(test_fc["forecast"].mean()) if not test_fc.empty else 0.0
        results.append({"unique_id": uid, "expected_return": expected_return, "model": best_model})
        # free per-asset tensors
        try:
            del X, y_target, lstm_model
        except Exception:
            pass
        try:
            del X_s, y_s, s5_model
        except Exception:
            pass
        if str(device).startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    return pd.DataFrame(results)


def infer_freq_from_prices(prices: pd.DataFrame) -> str:
    """Infer pandas frequency code from price data dates."""
    freq = pd.infer_freq(prices["ds"].sort_values())
    return freq or "D"


@dataclass
class Transition:
    state: Any
    action_logprob: Any
    reward: float


class PortfolioEnv:
    """
    Minimal portfolio environment.
    State = concatenated forecasted expected returns and current weights.
    Action = logits mapped to portfolio weights via softmax.
    Reward = realized portfolio return - risk_aversion * weight variance.
    """

    def __init__(
        self,
        returns_matrix: np.ndarray,
        forecast_vector: np.ndarray,
        risk_aversion: float = 0.01,
    ):
        self.returns_matrix = returns_matrix
        self.forecast_vector = forecast_vector
        self.t = 0
        self.n_assets = returns_matrix.shape[1]
        self.risk_aversion = risk_aversion
        self.weights = np.ones(self.n_assets) / self.n_assets

    def reset(self) -> np.ndarray:
        self.t = 0
        self.weights = np.ones(self.n_assets) / self.n_assets
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return np.concatenate([self.forecast_vector, self.weights], axis=0)

    def step(self, action_logits: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        weights = softmax_np(action_logits)
        realized = self.returns_matrix[self.t]
        reward = float(np.dot(realized, weights) - self.risk_aversion * np.var(weights))
        self.weights = weights
        self.t += 1
        done = self.t >= len(self.returns_matrix)
        next_state = self._get_state()
        return next_state, reward, done


def build_correlation_graph(returns_df: pd.DataFrame) -> np.ndarray:
    """Build a simple correlation-based adjacency matrix between tickers."""
    pivot = returns_df.pivot(index="ds", columns="unique_id", values="y").fillna(0.0)
    corr = pivot.corr().fillna(0.0).values
    return corr


class GraphTradingEnv:
    """
    Graph-aware trading environment with per-asset buy/hold/sell decisions.
    Action logits are shaped (n_assets, 3) corresponding to [sell, hold, buy].
    """

    def __init__(
        self,
        returns_matrix: np.ndarray,
        forecast_vector: np.ndarray,
        adjacency: np.ndarray,
        risk_aversion: float = 0.01,
        trade_step: float = 0.05,
    ):
        self.returns_matrix = returns_matrix
        self.forecast_vector = forecast_vector
        self.adjacency = adjacency
        self.t = 0
        self.n_assets = returns_matrix.shape[1]
        self.risk_aversion = risk_aversion
        self.trade_step = trade_step
        self.weights = np.ones(self.n_assets) / self.n_assets

    def reset(self) -> np.ndarray:
        self.t = 0
        self.weights = np.ones(self.n_assets) / self.n_assets
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return np.concatenate([self.forecast_vector, self.weights], axis=0)

    def step(self, action_logits: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        # action_logits expected shape: (n_assets, 3) or flat
        logits = np.asarray(action_logits).reshape(self.n_assets, 3)
        # Row-wise softmax
        exps = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exps / exps.sum(axis=1, keepdims=True)
        actions = np.argmax(probs, axis=1)  # 0 sell, 1 hold, 2 buy
        # adjust weights
        deltas = np.where(actions == 2, self.trade_step, 0) - np.where(actions == 0, self.trade_step, 0)
        self.weights = np.clip(self.weights + deltas, 0, None)
        if self.weights.sum() == 0:
            self.weights = np.ones(self.n_assets) / self.n_assets
        else:
            self.weights = self.weights / self.weights.sum()

        realized = self.returns_matrix[self.t]
        # graph regularization: encourage diversification using adjacency
        diversification_penalty = float(np.mean(self.weights @ self.adjacency @ self.weights.T))
        reward = float(np.dot(realized, self.weights) - self.risk_aversion * np.var(self.weights) - 0.01 * diversification_penalty)
        self.t += 1
        done = self.t >= len(self.returns_matrix)
        next_state = self._get_state()
        return next_state, reward, done


def softmax_np(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x).reshape(-1)
    ex = np.exp(arr - np.max(arr))
    return ex / np.sum(ex)


def build_policy_network(input_dim: int, hidden_dim: int, n_assets: int):
    """Construct a simple MLP policy."""
    torch_mod = _lazy_torch()
    nn = torch_mod["nn"]
    return nn.Sequential(
        nn.Linear(input_dim, 96),
        nn.ReLU(),
        nn.Linear(96, 96),
        nn.ReLU(),
        nn.Linear(96, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 48),
        nn.ReLU(),
        nn.Linear(48, n_assets),
    )


def select_action(policy, state, device: str) -> Tuple[np.ndarray, Any]:
    """Sample an action and return logits for the environment plus log_prob for training."""
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    logits = policy(state.to(device))
    # If logits are 1D, treat as weight logits; if 2D, flatten for graph env
    if logits.dim() == 2:
        flat_logits = logits
    else:
        flat_logits = logits.unsqueeze(0)
    dist = torch.distributions.Categorical(logits=flat_logits.view(-1))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return flat_logits.detach().cpu().numpy(), log_prob


def train_policy_gradient(
    env: PortfolioEnv,
    episodes: int = 500,
    lr: float = 1e-5,
    gamma: float = 0.99,
    hidden_dim: int = 48,
    device: Optional[str] = None,
    patience: Optional[int] = None,
    min_delta: float = 1e-4,
) -> Dict[str, Any]:
    """
    Train a REINFORCE agent to propose portfolio weights.
    Returns metrics history containing episodic rewards and policy metadata.
    """
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    optim = torch_mod["optim"]
    base_device = torch_mod.get("device", "cpu")
    device = device or ("cuda" if base_device == "cuda" and torch.cuda.is_available() else base_device)
    input_dim = len(env.forecast_vector) + env.n_assets
    policy = build_policy_network(input_dim, hidden_dim, env.n_assets).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    reward_history: List[float] = []
    best_reward = -np.inf
    patience_ctr = 0

    for _ in range(episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32).to(device)
        transitions: List[Transition] = []
        done = False
        while not done:
            action_logits, log_prob = select_action(policy, state, device)
            next_state_np, reward, done = env.step(action_logits)
            transitions.append(Transition(state=state, action_logprob=log_prob, reward=reward))
            state = torch.tensor(next_state_np, dtype=torch.float32).to(device)

        # Compute returns and policy loss
        returns: List[float] = []
        G = 0.0
        for t in reversed(transitions):
            G = t.reward + gamma * G
            returns.insert(0, G)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        policy_loss = []
        for R, trans in zip(returns_tensor, transitions):
            policy_loss.append(-trans.action_logprob * R)
        loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_reward = float(np.mean([t.reward for t in transitions]))
        reward_history.append(avg_reward)
        if patience is not None:
            if avg_reward > best_reward + min_delta:
                best_reward = avg_reward
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

    metadata = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "n_assets": env.n_assets,
    }
    return {"policy": policy, "rewards": reward_history, "device": device, "metadata": metadata}


def train_policy_gradient_graph(
    env: GraphTradingEnv,
    episodes: int = 500,
    lr: float = 1e-5,
    gamma: float = 0.99,
    hidden_dim: int = 48,
    device: Optional[str] = None,
    patience: Optional[int] = None,
    min_delta: float = 1e-4,
) -> Dict[str, Any]:
    """Train REINFORCE on graph trading env with per-asset buy/hold/sell actions."""
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    nn = torch_mod["nn"]
    optim = torch_mod["optim"]
    base_device = torch_mod.get("device", "cpu")
    device = device or ("cuda" if base_device == "cuda" and torch.cuda.is_available() else base_device)
    input_dim = len(env.forecast_vector) + env.n_assets
    policy = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, env.n_assets * 3),
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    reward_history: List[float] = []
    best_reward = -np.inf
    patience_ctr = 0

    for _ in range(episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32).to(device)
        transitions: List[Transition] = []
        done = False
        while not done:
            action_logits, log_prob = select_action(policy, state, device)
            next_state_np, reward, done = env.step(action_logits.reshape(env.n_assets, 3))
            transitions.append(Transition(state=state, action_logprob=log_prob, reward=reward))
            state = torch.tensor(next_state_np, dtype=torch.float32).to(device)

        returns: List[float] = []
        G = 0.0
        for t in reversed(transitions):
            G = t.reward + gamma * G
            returns.insert(0, G)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        policy_loss = []
        for R, trans in zip(returns_tensor, transitions):
            policy_loss.append(-trans.action_logprob * R)
        loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_reward = float(np.mean([t.reward for t in transitions]))
        reward_history.append(avg_reward)
        if patience is not None:
            if avg_reward > best_reward + min_delta:
                best_reward = avg_reward
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

    metadata = {"input_dim": input_dim, "hidden_dim": hidden_dim, "n_assets": env.n_assets}
    return {"policy": policy, "rewards": reward_history, "device": device, "metadata": metadata}


def recommend_weights(policy, env: PortfolioEnv, device: str) -> np.ndarray:
    """Generate deterministic weights from the trained policy given current state."""
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    # Ensure state lives on same device as policy parameters
    policy_device = next(policy.parameters()).device
    state = torch.tensor(env.reset(), dtype=torch.float32, device=policy_device)
    with torch.no_grad():
        logits = policy(state)
        weights = torch.softmax(logits, dim=-1).cpu().numpy()
    return weights


def simulate_policy_path_weights(policy, returns_matrix: np.ndarray, risk_aversion: float = 0.01) -> pd.DataFrame:
    """Simulate a single greedy path using a weight-based policy to log weights and P&L."""
    env = PortfolioEnv(returns_matrix=returns_matrix, forecast_vector=np.zeros(returns_matrix.shape[1]), risk_aversion=risk_aversion)
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    state = torch.tensor(env.reset(), dtype=torch.float32, device=next(policy.parameters()).device)
    records = []
    step = 0
    done = False
    while not done:
        with torch.no_grad():
            logits = policy(state)
            weights = torch.softmax(logits, dim=-1).cpu().numpy()
        next_state, reward, done = env.step(weights)
        records.append({"step": step, "reward": reward, "weights": weights})
        state = torch.tensor(next_state, dtype=torch.float32, device=next(policy.parameters()).device)
        step += 1
    df = pd.DataFrame(records)
    df["cum_reward"] = df["reward"].cumsum()
    return df


def simulate_policy_path_graph(policy, env: GraphTradingEnv, device: str) -> pd.DataFrame:
    """Simulate a greedy path on the graph trading env, logging actions and P&L."""
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    state = torch.tensor(env.reset(), dtype=torch.float32, device=device)
    records = []
    step = 0
    done = False
    while not done:
        with torch.no_grad():
            logits = policy(state)
        logits_np = logits.detach().cpu().numpy().reshape(env.n_assets, 3)
        actions = np.argmax(logits_np, axis=1)
        next_state, reward, done = env.step(logits_np)
        records.append({"step": step, "reward": reward, "actions": actions})
        state = torch.tensor(next_state, dtype=torch.float32, device=device)
        step += 1
    df = pd.DataFrame(records)
    df["cum_reward"] = df["reward"].cumsum()
    return df


def simulate_policy_graph_topn(
    policy,
    env: GraphTradingEnv,
    device: str,
    runs: int = 100,
    top_n: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run multiple stochastic rollouts on the graph env, sampling actions from the policy,
    and return top-N portfolios by cumulative reward plus the best action log.
    """
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    portfolios = []
    best_log = None
    best_reward = -np.inf
    for _ in range(runs):
        env.reset()
        state = torch.tensor(env.reset(), dtype=torch.float32, device=device)
        done = False
        rewards = []
        while not done:
            logits = policy(state)
            probs = torch.softmax(logits.view(-1), dim=0)
            action_sample = torch.distributions.Categorical(probs=probs).sample()
            logits_np = logits.detach().cpu().numpy().reshape(env.n_assets, 3)
            actions = np.argmax(logits_np, axis=1)
            next_state, reward, done = env.step(logits_np)
            rewards.append(reward)
            state = torch.tensor(next_state, dtype=torch.float32, device=device)
        total_reward = float(np.sum(rewards))
        buy_mask = (actions == 2).astype(float)
        if buy_mask.sum() == 0:
            buy_mask = np.ones_like(buy_mask)
        weights = buy_mask / buy_mask.sum()
        portfolios.append({"weights": weights, "reward": total_reward})
        if total_reward > best_reward:
            best_reward = total_reward
            best_log = pd.DataFrame({"step": range(len(rewards)), "reward": rewards})
            best_log["cum_reward"] = best_log["reward"].cumsum()
    portfolios_df = pd.DataFrame(portfolios).sort_values("reward", ascending=False).head(top_n).reset_index(drop=True)
    return portfolios_df, best_log


def save_policy_checkpoint(policy, path: str, metadata: Dict[str, Any]) -> None:
    """Persist policy weights and metadata to disk for later inference."""
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": policy.state_dict(), "metadata": metadata}
    torch.save(payload, path_obj)


def load_checkpoint_payload(path: str) -> Dict[str, Any]:
    """Load raw checkpoint payload from disk."""
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    payload = torch.load(path_obj, map_location="cpu")
    return payload


def load_policy_checkpoint(path: str, input_dim: int, hidden_dim: int, n_assets: int):
    """Load policy checkpoint and rebuild the network."""
    payload = load_checkpoint_payload(path)
    policy = build_policy_network(input_dim, hidden_dim, n_assets)
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    metadata = payload.get("metadata", {})
    return policy, metadata


def optimize_portfolio(
    prices: pd.DataFrame,
    horizon: int,
    top_k: int,
    episodes: int = 500,
    lr: float = 1e-5,
    risk_aversion: float = 0.02,
    checkpoint_path: Optional[str] = None,
    date_col: str = "ds",
    price_col: str = "close",
    ticker_col: str = "ticker",
    resample_rule: Optional[str] = None,
    device: Optional[str] = None,
    rl_mode: str = "weights",
    forecast_params: Optional[Dict[str, Any]] = None,
    rl_patience: Optional[int] = None,
    rl_min_delta: float = 1e-4,
) -> Dict[str, Any]:
    """
    End-to-end pipeline: compute returns, forecast expected returns with statsforecast,
    train RL agent, and return recommended weights for top-k assets by expected return.
    """
    prices = format_price_frame(prices, date_col=date_col, price_col=price_col, ticker_col=ticker_col)
    prices = clean_price_frame(prices)
    if resample_rule:
        prices = resample_prices(prices, resample_rule)
    returns_df = prepare_returns(prices)
    freq = infer_freq_from_prices(prices) or "D"
    expected = forecast_asset_returns(returns_df, horizon=horizon, freq=freq, device=device, forecast_params=forecast_params)
    expected = expected.sort_values("expected_return", ascending=False)
    selected_assets = expected.head(top_k)["unique_id"].tolist()
    filtered_returns = returns_df[returns_df["unique_id"].isin(selected_assets)]

    pivot = filtered_returns.pivot(index="ds", columns="unique_id", values="y").fillna(0.0)
    returns_matrix = pivot.values
    forecast_vector = expected.set_index("unique_id").loc[selected_assets]["expected_return"].values

    action_log = None
    if rl_mode == "graph":
        adjacency = build_correlation_graph(filtered_returns)
        env = GraphTradingEnv(
            returns_matrix=returns_matrix,
            forecast_vector=forecast_vector,
            adjacency=adjacency,
            risk_aversion=risk_aversion,
        )
        training = train_policy_gradient_graph(env, episodes=episodes, lr=lr, device=device, patience=rl_patience, min_delta=rl_min_delta)
        # simulate stochastic rollouts to get top portfolios
        top_ports, best_log = simulate_policy_graph_topn(
            training["policy"],
            env,
            device=training["device"],
            runs=100,
            top_n=min(10, top_k if top_k > 0 else 10),
        )
        action_log = best_log
        # use best portfolio weights from top list
        weights = top_ports.iloc[0]["weights"]
    else:
        env = PortfolioEnv(returns_matrix=returns_matrix, forecast_vector=forecast_vector, risk_aversion=risk_aversion)
        training = train_policy_gradient(env, episodes=episodes, lr=lr, device=device, patience=rl_patience, min_delta=rl_min_delta)
        weights = recommend_weights(training["policy"], env, training["device"])
        action_log = simulate_policy_path_weights(training["policy"], returns_matrix, risk_aversion=risk_aversion)
    portfolio = pd.DataFrame(
        {
            "asset": selected_assets,
            "weight": weights,
            "expected_return": forecast_vector,
        }
    )
    portfolio["weight"] = portfolio["weight"] / portfolio["weight"].sum()

    # Save optional checkpoint for reuse
    if checkpoint_path:
        metadata = training["metadata"] | {"assets": selected_assets, "risk_aversion": risk_aversion}
        save_policy_checkpoint(training["policy"], checkpoint_path, metadata)

    return {
        "portfolio": portfolio,
        "reward_history": training["rewards"],
        "selected_assets": selected_assets,
        "action_log": action_log,
        "top_portfolios": top_ports if rl_mode == "graph" else None,
    }


def optimize_portfolio_inference(
    prices: pd.DataFrame,
    checkpoint_path: str,
    horizon: int,
    risk_aversion: float = 0.02,
    date_col: str = "ds",
    price_col: str = "close",
    ticker_col: str = "ticker",
    resample_rule: Optional[str] = None,
    device: Optional[str] = None,
    rl_mode: str = "weights",
    forecast_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Inference-only pipeline: load a pre-trained policy checkpoint and produce weights
    for the assets seen during training. Forecasts are recomputed for current prices.
    """
    payload = load_checkpoint_payload(checkpoint_path)
    metadata_tmp = payload.get("metadata", {})
    trained_assets: List[str] = metadata_tmp.get("assets", [])
    if not trained_assets:
        raise ValueError("Checkpoint is missing asset metadata.")

    prices = format_price_frame(prices, date_col=date_col, price_col=price_col, ticker_col=ticker_col)
    prices = clean_price_frame(prices)
    if resample_rule:
        prices = resample_prices(prices, resample_rule)
    returns_df = prepare_returns(prices)
    freq = infer_freq_from_prices(prices) or "D"
    expected = forecast_asset_returns(returns_df, horizon=horizon, freq=freq, device=device, forecast_params=forecast_params)
    expected = expected[expected["unique_id"].isin(trained_assets)]
    expected = expected.set_index("unique_id").loc[trained_assets].reset_index()

    filtered_returns = returns_df[returns_df["unique_id"].isin(trained_assets)]
    pivot = filtered_returns.pivot(index="ds", columns="unique_id", values="y").fillna(0.0)
    if pivot.empty:
        raise ValueError("No overlapping assets between checkpoint and current price data.")
    returns_matrix = pivot[trained_assets].values
    forecast_vector = expected["expected_return"].values

    # Rebuild policy with correct dimensions from metadata
    input_dim = metadata_tmp.get("input_dim", len(forecast_vector) * 2)
    hidden_dim = metadata_tmp.get("hidden_dim", 48)
    n_assets = metadata_tmp.get("n_assets", len(trained_assets))
    torch_mod = _lazy_torch()
    base_device = torch_mod.get("device", "cpu")
    device = device or ("cuda" if base_device == "cuda" and torch_mod["torch"].cuda.is_available() else base_device)
    policy = build_policy_network(input_dim, hidden_dim, n_assets).to(device)
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    metadata = metadata_tmp

    action_log = None
    if rl_mode == "graph":
        adjacency = build_correlation_graph(filtered_returns)
        env = GraphTradingEnv(
            returns_matrix=returns_matrix,
            forecast_vector=forecast_vector,
            adjacency=adjacency,
            risk_aversion=risk_aversion,
        )
        top_ports, best_log = simulate_policy_graph_topn(
            policy,
            env,
            device=device,
            runs=100,
            top_n=min(10, n_assets if n_assets > 0 else 10),
        )
        action_log = best_log
        weights = top_ports.iloc[0]["weights"]
    else:
        env = PortfolioEnv(returns_matrix=returns_matrix, forecast_vector=forecast_vector, risk_aversion=risk_aversion)
        weights = recommend_weights(policy, env, device=device)
        action_log = simulate_policy_path_weights(policy, returns_matrix, risk_aversion=risk_aversion)
    portfolio = pd.DataFrame(
        {
            "asset": trained_assets,
            "weight": weights,
            "expected_return": forecast_vector,
        }
    )
    portfolio["weight"] = portfolio["weight"] / portfolio["weight"].sum()
    return {
        "portfolio": portfolio,
        "selected_assets": trained_assets,
        "checkpoint_meta": metadata,
        "action_log": action_log,
        "top_portfolios": top_ports if rl_mode == "graph" else None,
    }
