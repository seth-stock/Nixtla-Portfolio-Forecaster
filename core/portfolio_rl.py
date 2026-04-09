"""
portfolio_rl.py
Portfolio optimization helpers combining forecasting with a lightweight
policy-gradient allocator and a deterministic mean-variance baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core import data_loading, evaluation, models_mlforecast, models_statsforecast

_TORCH_CACHE: Dict[str, Any] = {}


def _lazy_torch() -> Dict[str, Any]:
    """Import torch lazily so the rest of the app remains importable without it."""
    if _TORCH_CACHE:
        return _TORCH_CACHE
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.optim as optim  # type: ignore
    except Exception as e:
        raise ImportError(
            "PyTorch is required for RL training/inference. Install a CPU or CUDA-compatible "
            f"torch build. Original import error: {e}"
        ) from e
    _TORCH_CACHE.update({"torch": torch, "nn": nn, "optim": optim})
    return _TORCH_CACHE


def _resolve_device(requested: Optional[str] = None, cpu_threads: Optional[int] = None) -> str:
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    requested = str(requested or "cpu").lower()
    if requested == "cuda" and getattr(torch, "cuda", None) and torch.cuda.is_available():
        return "cuda"
    if cpu_threads:
        try:
            torch.set_num_threads(max(1, int(cpu_threads)))
        except Exception:
            pass
    return "cpu"


def format_price_frame(
    df: pd.DataFrame,
    date_col: str = "ds",
    price_col: str = "close",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Validate and standardize a user-provided price dataframe."""
    required = {date_col, price_col, ticker_col}
    if not required.issubset(df.columns):
        raise ValueError(f"Price data must include columns: {required}")
    formatted = df[[date_col, ticker_col, price_col]].copy()
    formatted = formatted.rename(columns={date_col: "ds", ticker_col: "ticker", price_col: "close"})
    formatted["ds"] = pd.to_datetime(formatted["ds"], errors="coerce").dt.tz_localize(None)
    formatted["close"] = pd.to_numeric(formatted["close"], errors="coerce")
    formatted["ticker"] = formatted["ticker"].astype(str)
    formatted = formatted.dropna(subset=["ds", "close", "ticker"])
    return formatted.sort_values(["ticker", "ds"]).reset_index(drop=True)


def clean_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning for price data."""
    cleaned = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"]).copy()
    cleaned = cleaned.drop_duplicates(subset=["ticker", "ds"])
    cleaned["close"] = cleaned.groupby("ticker")["close"].ffill().bfill()
    cleaned = cleaned[cleaned["close"] > 0]
    return cleaned.dropna(subset=["close"]).reset_index(drop=True)


def resample_prices(prices: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample prices to a common cadence using the last close in each bucket."""
    prices = prices.copy()
    prices["ds"] = pd.to_datetime(prices["ds"])
    frames = []
    for ticker, grp in prices.groupby("ticker"):
        grp = grp.set_index("ds").sort_index()
        res = grp["close"].resample(rule).last().dropna()
        if res.empty:
            continue
        frames.append(pd.DataFrame({"ds": res.index, "ticker": ticker, "close": res.values}))
    if not frames:
        return prices
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "ds"]).reset_index(drop=True)


def load_prices_from_files(
    files: List[Any],
    date_col: str = "Date",
    price_col: str = "Close",
    ticker_col: Optional[str] = None,
) -> pd.DataFrame:
    """Load multiple CSVs into a single long price dataframe."""
    frames = []
    for idx, file in enumerate(files):
        name = getattr(file, "name", None)
        if name is None and isinstance(file, (str, Path)):
            name = Path(file).name
        ticker_name = Path(name).stem.upper() if name else f"TICKER_{idx}"
        df = pd.read_csv(file)
        use_ticker_col = ticker_col
        if use_ticker_col is None or use_ticker_col not in df.columns:
            df = df.copy()
            df["__ticker"] = ticker_name
            use_ticker_col = "__ticker"
        frames.append(format_price_frame(df, date_col=date_col, price_col=price_col, ticker_col=use_ticker_col))
    if not frames:
        raise ValueError("No CSV files provided for price loading.")
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "ds"]).reset_index(drop=True)


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
    """Compute arithmetic returns per ticker."""
    prices = prices.sort_values(["ticker", "ds"]).copy()
    prices["y"] = prices.groupby("ticker")["close"].pct_change()
    prices["y"] = prices["y"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=-0.95)
    returns = prices.rename(columns={"ticker": "unique_id"})
    return returns[["unique_id", "ds", "y"]]


def infer_freq_from_prices(prices: pd.DataFrame) -> str:
    """Infer a consensus frequency from the available price history."""
    temp = prices.rename(columns={"ticker": "unique_id"})
    return data_loading.infer_frequency_per_series(temp[["unique_id", "ds", "close"]], "ds") or "B"


def _split_validation_frame(returns_df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    trains = []
    vals = []
    for _, g in returns_df.groupby("unique_id"):
        g = g.sort_values("ds").reset_index(drop=True)
        if len(g) < max(horizon * 4, 40):
            continue
        trains.append(g.iloc[:-horizon].copy())
        vals.append(g.iloc[-horizon:].copy())
    if not trains or not vals:
        raise ValueError("Not enough history to create a validation split for the optimizer forecasts.")
    return pd.concat(trains, ignore_index=True), pd.concat(vals, ignore_index=True)


def _align_forecasts_to_actuals(actual_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    actual_eval = actual_df.rename(columns={"y": "actual"})
    merged = forecast_df.merge(actual_eval[["unique_id", "ds", "actual"]], on=["unique_id", "ds"], how="left")
    return merged.dropna(subset=["forecast", "actual"])


def _build_candidate_forecasts(
    train_df: pd.DataFrame,
    horizon: int,
    freq: str,
    forecast_params: dict,
    cpu_threads: int,
) -> List[pd.DataFrame]:
    candidates = []
    try:
        stats_models = ["AutoARIMA", "AutoETS", "SeasonalNaive"]
        stats_fc = models_statsforecast.fit_and_forecast(
            train_df,
            "ds",
            "y",
            horizon,
            freq,
            stats_models,
            model_params=forecast_params.get("stats"),
            n_jobs=cpu_threads,
        )
        candidates.append(stats_fc)
    except Exception:
        pass
    try:
        rf_params = forecast_params.get("ml", {}).get("rf_params", {"n_estimators": 300, "max_depth": 8})
        ml_fc = models_mlforecast.fit_and_forecast(
            train_df,
            "ds",
            "y",
            horizon,
            freq,
            rf_params=rf_params,
            n_jobs=cpu_threads,
            use_diff=False,
            forecast_mode="multi-output (direct)",
        )
        candidates.append(ml_fc)
    except Exception:
        pass
    mean_rows = []
    for uid, g in train_df.groupby("unique_id"):
        g = g.sort_values("ds")
        avg_ret = float(g["y"].tail(min(20, len(g))).mean())
        future_ds = pd.date_range(start=g["ds"].max(), periods=horizon + 1, freq=freq)[1:]
        mean_rows.append(
            pd.DataFrame(
                {
                    "ds": future_ds,
                    "unique_id": uid,
                    "model": "HistoricalMean",
                    "forecast": avg_ret,
                }
            )
        )
    if mean_rows:
        candidates.append(pd.concat(mean_rows, ignore_index=True))
    return [c for c in candidates if not c.empty]


def _compute_validation_scores(val_df: pd.DataFrame, candidate_forecasts: List[pd.DataFrame]) -> pd.DataFrame:
    records = []
    for fc in candidate_forecasts:
        merged = _align_forecasts_to_actuals(val_df, fc)
        if merged.empty:
            continue
        for (uid, model), grp in merged.groupby(["unique_id", "model"]):
            metrics = evaluation.compute_metrics(grp["actual"], grp["forecast"])
            records.append(
                {
                    "unique_id": uid,
                    "model": model,
                    "validation_rmse": metrics["RMSE"],
                    "validation_mae": metrics["MAE"],
                    "validation_smape": metrics["sMAPE"],
                }
            )
    scores = pd.DataFrame(records)
    if scores.empty:
        raise ValueError("Optimizer forecast candidates could not be aligned to the validation window.")
    return scores.sort_values(["unique_id", "validation_rmse"])


def forecast_asset_returns(
    returns_df: pd.DataFrame,
    horizon: int,
    freq: str,
    device: Optional[str] = None,
    forecast_params: Optional[Dict[str, Any]] = None,
    cpu_threads: int = 8,
) -> pd.DataFrame:
    """
    Forecast next-horizon asset returns using lightweight Nixtla models and
    select the winning model per asset via a rolling validation holdout.
    """
    del device
    forecast_params = forecast_params or {}
    train_df, val_df = _split_validation_frame(returns_df, horizon)
    candidate_forecasts = _build_candidate_forecasts(train_df, horizon, freq, forecast_params, cpu_threads)
    scores = _compute_validation_scores(val_df, candidate_forecasts)
    best_models = scores.groupby("unique_id", as_index=False).first()

    full_candidates = pd.concat(
        _build_candidate_forecasts(returns_df, horizon, freq, forecast_params, cpu_threads),
        ignore_index=True,
    )
    rows = []
    for row in best_models.itertuples(index=False):
        selected = full_candidates[
            (full_candidates["unique_id"] == row.unique_id) & (full_candidates["model"] == row.model)
        ].sort_values("ds")
        if selected.empty:
            continue
        path = selected["forecast"].astype(float).head(horizon).to_numpy()
        path = np.nan_to_num(path, nan=0.0, posinf=0.0, neginf=0.0)
        path = np.clip(path, -0.95, None)
        rows.append(
            {
                "unique_id": row.unique_id,
                "model": row.model,
                "validation_rmse": float(row.validation_rmse),
                "validation_mae": float(row.validation_mae),
                "validation_smape": float(row.validation_smape),
                "mean_forecast_return": float(path.mean()),
                "expected_return": float(np.prod(1.0 + path) - 1.0),
                "forecast_volatility": float(path.std(ddof=0)),
                "forecast_path": path.tolist(),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("Optimizer could not produce any validated asset forecasts.")
    return result.sort_values("expected_return", ascending=False).reset_index(drop=True)


def build_correlation_graph(returns_df: pd.DataFrame) -> np.ndarray:
    """Build a correlation-based adjacency matrix between assets."""
    pivot = returns_df.pivot(index="ds", columns="unique_id", values="y").fillna(0.0)
    corr = pivot.corr().fillna(0.0).to_numpy()
    if corr.size == 0:
        return np.eye(max(1, pivot.shape[1]))
    return corr


def compute_covariance_matrix(returns_matrix: np.ndarray) -> np.ndarray:
    if returns_matrix.shape[0] <= 1:
        return np.eye(returns_matrix.shape[1]) * 1e-6
    if returns_matrix.shape[1] == 1:
        return np.array([[float(np.var(returns_matrix[:, 0]))]])
    cov = np.cov(returns_matrix, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.eye(returns_matrix.shape[1]) * float(cov)
    return np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)


def solve_mean_variance_weights(expected_step_returns: np.ndarray, cov: np.ndarray, risk_aversion: float) -> np.ndarray:
    """Long-only mean-variance style baseline portfolio."""
    n_assets = len(expected_step_returns)
    ridge = max(1e-6, float(risk_aversion) * 0.1)
    adjusted_cov = cov + np.eye(n_assets) * ridge
    raw = np.linalg.pinv(adjusted_cov) @ expected_step_returns
    raw = np.clip(raw, 0.0, None)
    if raw.sum() <= 0:
        return np.ones(n_assets) / n_assets
    return raw / raw.sum()


def summarize_portfolio_metrics(
    weights: np.ndarray,
    expected_cumulative_returns: np.ndarray,
    expected_step_returns: np.ndarray,
    cov: np.ndarray,
) -> Dict[str, float]:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
    expected_step = float(np.dot(weights, expected_step_returns))
    expected_horizon = float(np.dot(weights, expected_cumulative_returns))
    expected_vol = float(np.sqrt(max(weights @ cov @ weights.T, 0.0)))
    diversification = float(1.0 / np.sum(np.square(weights)))
    concentration = float(np.sum(np.square(weights)))
    return {
        "expected_step_return": expected_step,
        "expected_horizon_return": expected_horizon,
        "expected_volatility": expected_vol,
        "effective_n_assets": diversification,
        "weight_concentration": concentration,
    }


def _normalize_weight_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    if np.all(arr >= 0) and np.isfinite(arr).all():
        total = arr.sum()
        if 0.999 <= total <= 1.001:
            return arr
    arr = np.exp(arr - np.max(arr))
    total = arr.sum()
    if total <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / total


@dataclass
class Transition:
    action_logprob: Any
    reward: float


class PortfolioEnv:
    """
    State = forecast signal plus current weights.
    Action = portfolio weights on the simplex.
    Reward = realized return minus diversification and turnover penalties.
    """

    def __init__(
        self,
        returns_matrix: np.ndarray,
        forecast_vector: np.ndarray,
        risk_aversion: float = 0.01,
        turnover_penalty: float = 0.001,
    ):
        self.returns_matrix = returns_matrix
        self.forecast_vector = forecast_vector
        self.t = 0
        self.n_assets = returns_matrix.shape[1]
        self.risk_aversion = risk_aversion
        self.turnover_penalty = turnover_penalty
        self.weights = np.ones(self.n_assets) / self.n_assets

    def reset(self) -> np.ndarray:
        self.t = 0
        self.weights = np.ones(self.n_assets) / self.n_assets
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return np.concatenate([self.forecast_vector, self.weights], axis=0)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        weights = _normalize_weight_vector(action)
        realized = self.returns_matrix[self.t]
        turnover = float(np.abs(weights - self.weights).sum())
        reward = float(
            np.dot(realized, weights)
            - self.risk_aversion * np.var(weights)
            - self.turnover_penalty * turnover
        )
        self.weights = weights
        self.t += 1
        done = self.t >= len(self.returns_matrix)
        return self._get_state(), reward, done


class GraphTradingEnv:
    """
    Graph-aware environment with per-asset sell/hold/buy actions.
    """

    def __init__(
        self,
        returns_matrix: np.ndarray,
        forecast_vector: np.ndarray,
        adjacency: np.ndarray,
        risk_aversion: float = 0.01,
        trade_step: float = 0.05,
        turnover_penalty: float = 0.001,
    ):
        self.returns_matrix = returns_matrix
        self.forecast_vector = forecast_vector
        self.adjacency = adjacency
        self.risk_aversion = risk_aversion
        self.trade_step = trade_step
        self.turnover_penalty = turnover_penalty
        self.n_assets = returns_matrix.shape[1]
        self.t = 0
        self.weights = np.ones(self.n_assets) / self.n_assets

    def reset(self) -> np.ndarray:
        self.t = 0
        self.weights = np.ones(self.n_assets) / self.n_assets
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return np.concatenate([self.forecast_vector, self.weights], axis=0)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        arr = np.asarray(action)
        actions = np.argmax(arr, axis=1) if arr.ndim == 2 else arr.reshape(-1).astype(int)
        prev = self.weights.copy()
        deltas = np.where(actions == 2, self.trade_step, 0.0) - np.where(actions == 0, self.trade_step, 0.0)
        self.weights = np.clip(self.weights + deltas, 0.0, None)
        if self.weights.sum() <= 0:
            self.weights = np.ones(self.n_assets) / self.n_assets
        else:
            self.weights = self.weights / self.weights.sum()
        realized = self.returns_matrix[self.t]
        turnover = float(np.abs(self.weights - prev).sum())
        diversification_penalty = float(self.weights @ self.adjacency @ self.weights.T)
        reward = float(
            np.dot(realized, self.weights)
            - self.risk_aversion * np.var(self.weights)
            - 0.01 * diversification_penalty
            - self.turnover_penalty * turnover
        )
        self.t += 1
        done = self.t >= len(self.returns_matrix)
        return self._get_state(), reward, done


def build_policy_network(input_dim: int, hidden_dim: int, n_assets: int):
    """Construct a compact MLP for weight policies."""
    torch_mod = _lazy_torch()
    nn = torch_mod["nn"]
    hidden_dim = max(16, int(hidden_dim))
    mid_dim = max(16, hidden_dim // 2)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, mid_dim),
        nn.ReLU(),
        nn.Linear(mid_dim, n_assets),
    )


def build_graph_policy_network(input_dim: int, hidden_dim: int, n_assets: int):
    """Construct a compact MLP for graph-action policies."""
    torch_mod = _lazy_torch()
    nn = torch_mod["nn"]
    hidden_dim = max(16, int(hidden_dim))
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, n_assets * 3),
    )


def _sample_weight_action(policy, state, device: str) -> Tuple[np.ndarray, Any]:
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    logits = policy(state.to(device))
    concentration = torch.nn.functional.softplus(logits) + 1e-3
    dist = torch.distributions.Dirichlet(concentration)
    weights = dist.rsample()
    log_prob = dist.log_prob(weights)
    return weights.detach().cpu().numpy(), log_prob


def _greedy_weight_action(policy, state, device: str) -> np.ndarray:
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    with torch.no_grad():
        logits = policy(state.to(device))
        concentration = torch.nn.functional.softplus(logits) + 1e-3
        weights = concentration / concentration.sum()
    return weights.detach().cpu().numpy()


def _sample_graph_action(policy, state, n_assets: int, device: str) -> Tuple[np.ndarray, Any]:
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    logits = policy(state.to(device)).view(n_assets, 3)
    dist = torch.distributions.Categorical(logits=logits)
    actions = dist.sample()
    log_prob = dist.log_prob(actions).sum()
    return actions.detach().cpu().numpy(), log_prob


def _greedy_graph_action(policy, state, n_assets: int, device: str) -> np.ndarray:
    torch_mod = _lazy_torch()
    with torch_mod["torch"].no_grad():
        logits = policy(state.to(device)).view(n_assets, 3)
        actions = logits.argmax(dim=1)
    return actions.detach().cpu().numpy()


def train_policy_gradient(
    env: PortfolioEnv,
    episodes: int = 100,
    lr: float = 1e-3,
    gamma: float = 0.99,
    hidden_dim: int = 64,
    device: Optional[str] = None,
    cpu_threads: Optional[int] = None,
    patience: Optional[int] = None,
    min_delta: float = 1e-4,
) -> Dict[str, Any]:
    """Train a REINFORCE allocator over simplex weights."""
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    optim = torch_mod["optim"]
    resolved_device = _resolve_device(device, cpu_threads)
    input_dim = len(env.forecast_vector) + env.n_assets
    policy = build_policy_network(input_dim, hidden_dim, env.n_assets).to(resolved_device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    reward_history: List[float] = []
    best_reward = -np.inf
    patience_ctr = 0

    for _ in range(episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32, device=resolved_device)
        transitions: List[Transition] = []
        done = False
        episode_rewards: List[float] = []
        while not done:
            weights, log_prob = _sample_weight_action(policy, state, resolved_device)
            next_state_np, reward, done = env.step(weights)
            transitions.append(Transition(action_logprob=log_prob, reward=reward))
            episode_rewards.append(reward)
            state = torch.tensor(next_state_np, dtype=torch.float32, device=resolved_device)

        returns: List[float] = []
        G = 0.0
        for trans in reversed(transitions):
            G = trans.reward + gamma * G
            returns.insert(0, G)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=resolved_device)
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        loss = torch.stack([-trans.action_logprob * R for R, trans in zip(returns_tensor, transitions)]).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        reward_history.append(avg_reward)
        if patience is not None:
            if avg_reward > best_reward + min_delta:
                best_reward = avg_reward
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

    return {
        "policy": policy,
        "rewards": reward_history,
        "device": resolved_device,
        "metadata": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "n_assets": env.n_assets,
            "policy_type": "weights",
        },
    }


def train_policy_gradient_graph(
    env: GraphTradingEnv,
    episodes: int = 100,
    lr: float = 1e-3,
    gamma: float = 0.99,
    hidden_dim: int = 64,
    device: Optional[str] = None,
    cpu_threads: Optional[int] = None,
    patience: Optional[int] = None,
    min_delta: float = 1e-4,
) -> Dict[str, Any]:
    """Train REINFORCE on graph buy/hold/sell actions."""
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    optim = torch_mod["optim"]
    resolved_device = _resolve_device(device, cpu_threads)
    input_dim = len(env.forecast_vector) + env.n_assets
    policy = build_graph_policy_network(input_dim, hidden_dim, env.n_assets).to(resolved_device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    reward_history: List[float] = []
    best_reward = -np.inf
    patience_ctr = 0

    for _ in range(episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32, device=resolved_device)
        transitions: List[Transition] = []
        done = False
        episode_rewards: List[float] = []
        while not done:
            actions, log_prob = _sample_graph_action(policy, state, env.n_assets, resolved_device)
            next_state_np, reward, done = env.step(actions)
            transitions.append(Transition(action_logprob=log_prob, reward=reward))
            episode_rewards.append(reward)
            state = torch.tensor(next_state_np, dtype=torch.float32, device=resolved_device)

        returns: List[float] = []
        G = 0.0
        for trans in reversed(transitions):
            G = trans.reward + gamma * G
            returns.insert(0, G)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=resolved_device)
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        loss = torch.stack([-trans.action_logprob * R for R, trans in zip(returns_tensor, transitions)]).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        reward_history.append(avg_reward)
        if patience is not None:
            if avg_reward > best_reward + min_delta:
                best_reward = avg_reward
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

    return {
        "policy": policy,
        "rewards": reward_history,
        "device": resolved_device,
        "metadata": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "n_assets": env.n_assets,
            "policy_type": "graph",
        },
    }


def recommend_weights(policy, env: PortfolioEnv, device: str) -> np.ndarray:
    """Generate deterministic weights from a trained weight policy."""
    torch_mod = _lazy_torch()
    state = torch_mod["torch"].tensor(env.reset(), dtype=torch_mod["torch"].float32, device=device)
    return _greedy_weight_action(policy, state, device)


def simulate_policy_path_weights(
    policy,
    returns_matrix: np.ndarray,
    forecast_vector: np.ndarray,
    risk_aversion: float = 0.01,
    device: str = "cpu",
) -> pd.DataFrame:
    """Simulate a greedy path using a weight policy."""
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    env = PortfolioEnv(returns_matrix=returns_matrix, forecast_vector=forecast_vector, risk_aversion=risk_aversion)
    state = torch.tensor(env.reset(), dtype=torch.float32, device=device)
    records = []
    step = 0
    done = False
    while not done:
        weights = _greedy_weight_action(policy, state, device)
        next_state, reward, done = env.step(weights)
        records.append({"step": step, "reward": reward, "weights": np.round(weights, 4)})
        state = torch.tensor(next_state, dtype=torch.float32, device=device)
        step += 1
    df = pd.DataFrame(records)
    if not df.empty:
        df["cum_reward"] = df["reward"].cumsum()
    return df


def simulate_policy_path_graph(policy, env: GraphTradingEnv, device: str) -> pd.DataFrame:
    """Simulate a greedy path on the graph environment."""
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    state = torch.tensor(env.reset(), dtype=torch.float32, device=device)
    records = []
    step = 0
    done = False
    while not done:
        actions = _greedy_graph_action(policy, state, env.n_assets, device)
        next_state, reward, done = env.step(actions)
        records.append({"step": step, "reward": reward, "actions": actions.tolist(), "weights": np.round(env.weights, 4)})
        state = torch.tensor(next_state, dtype=torch.float32, device=device)
        step += 1
    df = pd.DataFrame(records)
    if not df.empty:
        df["cum_reward"] = df["reward"].cumsum()
    return df


def simulate_policy_graph_topn(
    policy,
    env: GraphTradingEnv,
    device: str,
    runs: int = 100,
    top_n: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run multiple stochastic graph rollouts and return the best portfolios."""
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    portfolios = []
    best_log = pd.DataFrame()
    best_reward = -np.inf
    for _ in range(runs):
        state = torch.tensor(env.reset(), dtype=torch.float32, device=device)
        done = False
        records = []
        while not done:
            actions, _ = _sample_graph_action(policy, state, env.n_assets, device)
            next_state, reward, done = env.step(actions)
            records.append({"step": len(records), "reward": reward, "actions": actions.tolist(), "weights": np.round(env.weights, 4)})
            state = torch.tensor(next_state, dtype=torch.float32, device=device)
        total_reward = float(sum(r["reward"] for r in records))
        portfolios.append({"weights": env.weights.copy(), "reward": total_reward})
        if total_reward > best_reward:
            best_reward = total_reward
            best_log = pd.DataFrame(records)
            if not best_log.empty:
                best_log["cum_reward"] = best_log["reward"].cumsum()
    portfolios_df = pd.DataFrame(portfolios).sort_values("reward", ascending=False).head(top_n).reset_index(drop=True)
    return portfolios_df, best_log


def save_policy_checkpoint(policy, path: str, metadata: Dict[str, Any]) -> None:
    """Persist policy weights and metadata to disk."""
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "metadata": metadata}, path_obj)


def load_checkpoint_payload(path: str) -> Dict[str, Any]:
    """Load a checkpoint payload from disk."""
    torch_mod = _lazy_torch()
    torch = torch_mod["torch"]
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    return torch.load(path_obj, map_location="cpu")


def load_policy_checkpoint(path: str, input_dim: int, hidden_dim: int, n_assets: int):
    """Load a weight-policy checkpoint."""
    payload = load_checkpoint_payload(path)
    policy = build_policy_network(input_dim, hidden_dim, n_assets)
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    return policy, payload.get("metadata", {})


def _assemble_portfolio_frame(
    asset_order: List[str],
    weights: np.ndarray,
    forecast_table: pd.DataFrame,
) -> pd.DataFrame:
    lookup = forecast_table.set_index("unique_id")
    rows = []
    for asset, weight in zip(asset_order, weights):
        info = lookup.loc[asset]
        rows.append(
            {
                "asset": asset,
                "weight": float(weight),
                "expected_return": float(info["expected_return"]),
                "mean_forecast_return": float(info["mean_forecast_return"]),
                "forecast_volatility": float(info["forecast_volatility"]),
                "model": str(info["model"]),
                "validation_rmse": float(info["validation_rmse"]),
            }
        )
    portfolio = pd.DataFrame(rows)
    if not portfolio.empty and portfolio["weight"].sum() > 0:
        portfolio["weight"] = portfolio["weight"] / portfolio["weight"].sum()
    return portfolio.sort_values("weight", ascending=False).reset_index(drop=True)


def optimize_portfolio(
    prices: pd.DataFrame,
    horizon: int,
    top_k: int,
    episodes: int = 100,
    lr: float = 1e-3,
    risk_aversion: float = 0.02,
    checkpoint_path: Optional[str] = None,
    date_col: str = "ds",
    price_col: str = "close",
    ticker_col: str = "ticker",
    resample_rule: Optional[str] = None,
    device: Optional[str] = None,
    rl_mode: str = "weights",
    forecast_params: Optional[Dict[str, Any]] = None,
    rl_hidden_dim: int = 64,
    cpu_threads: int = 8,
    rl_patience: Optional[int] = None,
    rl_min_delta: float = 1e-4,
) -> Dict[str, Any]:
    """
    End-to-end optimizer pipeline with validated asset forecasts, a mean-variance
    baseline, and optional RL refinement.
    """
    prices = format_price_frame(prices, date_col=date_col, price_col=price_col, ticker_col=ticker_col)
    prices = clean_price_frame(prices)
    if resample_rule:
        prices = resample_prices(prices, resample_rule)
    returns_df = prepare_returns(prices)
    freq = infer_freq_from_prices(prices)
    forecast_table = forecast_asset_returns(
        returns_df,
        horizon=horizon,
        freq=freq,
        device=device,
        forecast_params=forecast_params,
        cpu_threads=cpu_threads,
    )

    top_k = max(1, min(int(top_k), len(forecast_table)))
    selected_assets = forecast_table.head(top_k)["unique_id"].tolist()
    selected_forecasts = forecast_table.set_index("unique_id").loc[selected_assets].reset_index()
    filtered_returns = returns_df[returns_df["unique_id"].isin(selected_assets)]
    pivot = filtered_returns.pivot(index="ds", columns="unique_id", values="y").reindex(columns=selected_assets).fillna(0.0)
    if pivot.empty:
        raise ValueError("Selected assets do not have usable return history for optimization.")
    returns_matrix = pivot.to_numpy()
    mean_step_returns = selected_forecasts["mean_forecast_return"].to_numpy(dtype=float)
    expected_horizon_returns = selected_forecasts["expected_return"].to_numpy(dtype=float)
    cov = compute_covariance_matrix(returns_matrix)

    baseline_weights = solve_mean_variance_weights(mean_step_returns, cov, risk_aversion)
    baseline_portfolio = _assemble_portfolio_frame(selected_assets, baseline_weights, selected_forecasts)
    baseline_metrics = summarize_portfolio_metrics(baseline_weights, expected_horizon_returns, mean_step_returns, cov)

    reward_history: List[float] = []
    action_log = pd.DataFrame()
    top_portfolios = None
    optimizer_engine = "mean_variance"
    weights = baseline_weights.copy()
    checkpoint_meta: Dict[str, Any] = {}

    try:
        if rl_mode == "graph":
            adjacency = build_correlation_graph(filtered_returns)
            env = GraphTradingEnv(
                returns_matrix=returns_matrix,
                forecast_vector=mean_step_returns,
                adjacency=adjacency,
                risk_aversion=risk_aversion,
            )
            training = train_policy_gradient_graph(
                env,
                episodes=episodes,
                lr=lr,
                hidden_dim=rl_hidden_dim,
                device=device,
                cpu_threads=cpu_threads,
                patience=rl_patience,
                min_delta=rl_min_delta,
            )
            reward_history = training["rewards"]
            top_portfolios, action_log = simulate_policy_graph_topn(
                training["policy"],
                env,
                device=training["device"],
                runs=100,
                top_n=min(10, len(selected_assets)),
            )
            if top_portfolios is not None and not top_portfolios.empty:
                weights = np.asarray(top_portfolios.iloc[0]["weights"], dtype=float)
            checkpoint_meta = training["metadata"] | {"assets": selected_assets, "risk_aversion": risk_aversion, "rl_mode": "graph"}
            if checkpoint_path:
                save_policy_checkpoint(training["policy"], checkpoint_path, checkpoint_meta)
            optimizer_engine = "rl_graph"
        else:
            env = PortfolioEnv(returns_matrix=returns_matrix, forecast_vector=mean_step_returns, risk_aversion=risk_aversion)
            training = train_policy_gradient(
                env,
                episodes=episodes,
                lr=lr,
                hidden_dim=rl_hidden_dim,
                device=device,
                cpu_threads=cpu_threads,
                patience=rl_patience,
                min_delta=rl_min_delta,
            )
            reward_history = training["rewards"]
            weights = recommend_weights(training["policy"], env, training["device"])
            action_log = simulate_policy_path_weights(
                training["policy"],
                returns_matrix,
                forecast_vector=mean_step_returns,
                risk_aversion=risk_aversion,
                device=training["device"],
            )
            checkpoint_meta = training["metadata"] | {"assets": selected_assets, "risk_aversion": risk_aversion, "rl_mode": "weights"}
            if checkpoint_path:
                save_policy_checkpoint(training["policy"], checkpoint_path, checkpoint_meta)
            optimizer_engine = "rl_weights"
    except ImportError:
        pass

    portfolio = _assemble_portfolio_frame(selected_assets, weights, selected_forecasts)
    portfolio_metrics = summarize_portfolio_metrics(
        portfolio["weight"].to_numpy(),
        portfolio["expected_return"].to_numpy(),
        portfolio["mean_forecast_return"].to_numpy(),
        cov,
    )
    return {
        "portfolio": portfolio,
        "baseline_portfolio": baseline_portfolio,
        "forecast_summary": selected_forecasts.sort_values("expected_return", ascending=False).reset_index(drop=True),
        "reward_history": reward_history,
        "selected_assets": selected_assets,
        "action_log": action_log,
        "top_portfolios": top_portfolios,
        "portfolio_metrics": portfolio_metrics,
        "baseline_metrics": baseline_metrics,
        "optimizer_engine": optimizer_engine,
        "checkpoint_meta": checkpoint_meta,
        "covariance_matrix": pd.DataFrame(cov, index=selected_assets, columns=selected_assets),
        "missing_assets": [],
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
    cpu_threads: int = 8,
) -> Dict[str, Any]:
    """
    Inference-only pipeline using a saved policy checkpoint plus fresh asset forecasts.
    Missing assets are zero-filled and then zero-weighted before final normalization.
    """
    payload = load_checkpoint_payload(checkpoint_path)
    metadata = payload.get("metadata", {})
    trained_assets: List[str] = metadata.get("assets", [])
    if not trained_assets:
        raise ValueError("Checkpoint is missing asset metadata.")

    prices = format_price_frame(prices, date_col=date_col, price_col=price_col, ticker_col=ticker_col)
    prices = clean_price_frame(prices)
    if resample_rule:
        prices = resample_prices(prices, resample_rule)
    returns_df = prepare_returns(prices)
    freq = infer_freq_from_prices(prices)
    forecast_table = forecast_asset_returns(
        returns_df,
        horizon=horizon,
        freq=freq,
        device=device,
        forecast_params=forecast_params,
        cpu_threads=cpu_threads,
    ).set_index("unique_id")
    missing_assets = [asset for asset in trained_assets if asset not in forecast_table.index]
    for asset in missing_assets:
        forecast_table.loc[asset] = {
            "model": "Unavailable",
            "validation_rmse": np.nan,
            "validation_mae": np.nan,
            "validation_smape": np.nan,
            "mean_forecast_return": 0.0,
            "expected_return": 0.0,
            "forecast_volatility": 0.0,
            "forecast_path": [0.0] * horizon,
        }
    selected_forecasts = forecast_table.loc[trained_assets].reset_index()

    filtered_returns = returns_df[returns_df["unique_id"].isin(trained_assets)]
    pivot = filtered_returns.pivot(index="ds", columns="unique_id", values="y").reindex(columns=trained_assets).fillna(0.0)
    if pivot.empty:
        raise ValueError("No overlapping assets between checkpoint and current price data.")
    returns_matrix = pivot.to_numpy()
    mean_step_returns = selected_forecasts["mean_forecast_return"].to_numpy(dtype=float)
    expected_horizon_returns = selected_forecasts["expected_return"].to_numpy(dtype=float)
    cov = compute_covariance_matrix(returns_matrix)

    baseline_weights = solve_mean_variance_weights(mean_step_returns, cov, risk_aversion)
    baseline_portfolio = _assemble_portfolio_frame(trained_assets, baseline_weights, selected_forecasts)
    baseline_metrics = summarize_portfolio_metrics(baseline_weights, expected_horizon_returns, mean_step_returns, cov)

    input_dim = int(metadata.get("input_dim", len(trained_assets) * 2))
    hidden_dim = int(metadata.get("hidden_dim", 64))
    n_assets = int(metadata.get("n_assets", len(trained_assets)))
    policy_type = str(metadata.get("policy_type", "weights"))
    checkpoint_rl_mode = str(metadata.get("rl_mode", rl_mode))
    resolved_device = _resolve_device(device, cpu_threads)

    if policy_type == "graph" or checkpoint_rl_mode == "graph":
        policy = build_graph_policy_network(input_dim, hidden_dim, n_assets).to(resolved_device)
    else:
        policy = build_policy_network(input_dim, hidden_dim, n_assets).to(resolved_device)
    policy.load_state_dict(payload["state_dict"])
    policy.eval()

    if checkpoint_rl_mode == "graph":
        adjacency = build_correlation_graph(filtered_returns)
        env = GraphTradingEnv(
            returns_matrix=returns_matrix,
            forecast_vector=mean_step_returns,
            adjacency=adjacency,
            risk_aversion=risk_aversion,
        )
        top_portfolios, action_log = simulate_policy_graph_topn(
            policy,
            env,
            device=resolved_device,
            runs=100,
            top_n=min(10, len(trained_assets)),
        )
        weights = np.asarray(top_portfolios.iloc[0]["weights"], dtype=float) if not top_portfolios.empty else baseline_weights
        optimizer_engine = "rl_graph_inference"
    else:
        env = PortfolioEnv(returns_matrix=returns_matrix, forecast_vector=mean_step_returns, risk_aversion=risk_aversion)
        weights = recommend_weights(policy, env, device=resolved_device)
        action_log = simulate_policy_path_weights(
            policy,
            returns_matrix,
            forecast_vector=mean_step_returns,
            risk_aversion=risk_aversion,
            device=resolved_device,
        )
        top_portfolios = None
        optimizer_engine = "rl_weights_inference"

    if missing_assets:
        mask = np.array([asset not in missing_assets for asset in trained_assets], dtype=float)
        weights = weights * mask
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            fallback_weights = baseline_weights * mask
            if fallback_weights.sum() > 0:
                weights = fallback_weights / fallback_weights.sum()
            elif mask.sum() > 0:
                weights = mask / mask.sum()
            else:
                weights = baseline_weights

    portfolio = _assemble_portfolio_frame(trained_assets, weights, selected_forecasts)
    portfolio_metrics = summarize_portfolio_metrics(
        portfolio["weight"].to_numpy(),
        portfolio["expected_return"].to_numpy(),
        portfolio["mean_forecast_return"].to_numpy(),
        cov,
    )
    return {
        "portfolio": portfolio,
        "baseline_portfolio": baseline_portfolio,
        "forecast_summary": selected_forecasts.sort_values("expected_return", ascending=False).reset_index(drop=True),
        "selected_assets": trained_assets,
        "checkpoint_meta": metadata,
        "action_log": action_log,
        "top_portfolios": top_portfolios,
        "portfolio_metrics": portfolio_metrics,
        "baseline_metrics": baseline_metrics,
        "optimizer_engine": optimizer_engine,
        "covariance_matrix": pd.DataFrame(cov, index=trained_assets, columns=trained_assets),
        "missing_assets": missing_assets,
    }
