"""
models_neuralforecast.py
NeuralForecast wrapper using lightweight RNN model for univariate forecasting.
"""

from __future__ import annotations

from typing import List

import os
import pandas as pd
import numpy as np
import inspect


def _lazy_import():
    """Local import to avoid hard failure when neural dependencies are missing."""
    try:
        import torch  # type: ignore
        from neuralforecast import NeuralForecast  # type: ignore
        from neuralforecast.models import RNN  # type: ignore
        from neuralforecast.losses.pytorch import MAE  # type: ignore
    except Exception as e:
        raise ImportError(
            "neuralforecast and torch are required for neural models. "
            "Install a matching PyTorch (CPU or CUDA) and neuralforecast. "
            f"Original error: {e}"
        ) from e
    return torch, NeuralForecast, RNN, MAE


def _prepare_data(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    renamed = df.rename(columns={date_col: "ds", target_col: "y"})
    if "unique_id" in renamed.columns:
        renamed["unique_id"] = renamed["unique_id"].astype(str)
    elif "ticker" in renamed.columns:
        renamed = renamed.assign(unique_id=renamed["ticker"].astype(str))
    else:
        renamed = renamed.assign(unique_id="series")
    return renamed[["unique_id", "ds", "y"]]


def _filter_kwargs(func, params: dict) -> dict:
    """Return only the kwargs supported by the callable signature."""
    sig = inspect.signature(func)
    return {k: v for k, v in params.items() if k in sig.parameters}


def fit_and_forecast(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    horizon: int,
    freq: str,
    models: List[str],
    model_params: dict | None = None,
) -> pd.DataFrame:
    """Fit NeuralForecast models (RNN/LSTM) and optionally a light S5 wrapper; return forecasts."""
    torch, NeuralForecast, RNN, MAE = _lazy_import()
    model_params = model_params or {}
    nf_df = _prepare_data(df, date_col, target_col)
    pref = model_params.get("device_preference")
    use_cuda = pref == "cuda" and torch.cuda.is_available()
    if pref == "cuda" and not torch.cuda.is_available():
        model_params["_cuda_unavailable"] = True
    else:
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
    loss_obj = MAE()
    context_len = model_params.get("context_length", 0)

    def _model_kwargs(prefix: str) -> dict:
        return {
            "input_size": model_params.get(f"{prefix}_input", model_params.get("input_size", context_len if context_len > 0 else 24)),
            "max_steps": model_params.get(f"{prefix}_steps", model_params.get("max_steps", 200)),
            "learning_rate": model_params.get(f"{prefix}_lr", model_params.get("learning_rate", 1e-3)),
            "batch_size": model_params.get(f"{prefix}_batch", model_params.get("batch_size", 32)),
            "encoder_hidden_size": model_params.get(f"{prefix}_hidden", 128),
            "encoder_n_layers": model_params.get(f"{prefix}_depth", 2),
            "context_length": model_params.get("context_length", None),
        }

    rnn_kwargs = _model_kwargs("rnn")
    lstm_kwargs = _model_kwargs("lstm")

    def _build_rnn_kwargs(base_kwargs: dict, series_len: int) -> dict:
        """Clip batch size to series length but otherwise respect user input."""
        k = base_kwargs.copy()
        k["batch_size"] = max(1, min(k["batch_size"], series_len - max(horizon, 1)))
        filtered = _filter_kwargs(
            RNN,
            {
                "h": horizon,
                "input_size": k["input_size"],
                "max_steps": k["max_steps"],
                "scaler_type": "standard",
                "loss": loss_obj,
                "learning_rate": k["learning_rate"],
                "batch_size": k["batch_size"],
                "encoder_hidden_size": k["encoder_hidden_size"],
                "encoder_n_layers": k["encoder_n_layers"],
                "context_length": k.get("context_length"),
            },
        )
        return filtered

    fcsts = []
    for uid, g in nf_df.groupby("unique_id"):
        g = g.sort_values("ds")
        series_len = len(g)
        if series_len <= horizon + 5:
            continue
        nf_models = []
        for name in models:
            if name == "RNN":
                rnn_args = _build_rnn_kwargs(rnn_kwargs, series_len)
                nf_models.append(RNN(**rnn_args))
            if name == "LSTM":
                lstm_args = _build_rnn_kwargs(lstm_kwargs, series_len)
                nf_models.append(RNN(**lstm_args))
        if not nf_models:
            continue
        nf = NeuralForecast(models=nf_models, freq=freq, local_scaler_type="standard")
        nf.fit(g)
        fcst = nf.predict().reset_index()
        forecast_cols = [c for c in fcst.columns if c not in ["ds", "unique_id"]]
        rename_map = {}
        for i, col in enumerate(forecast_cols):
            name = models[i] if i < len(models) else col
            rename_map[col] = name
        fcst = fcst.rename(columns=rename_map)
        fcst = fcst.melt(id_vars=["ds", "unique_id"], var_name="model", value_name="forecast")
        fcsts.append(fcst)

    # Optional light S5 forecaster if requested (per-series)
    if "S5" in models:
        try:
            from s5_pytorch import S5Layer  # type: ignore
            import torch.nn as nn

            class S5Forecaster(nn.Module):
                def __init__(self, horizon: int, seq_len: int = 64, d_model: int = 32):
                    super().__init__()
                    self.seq_len = seq_len
                    self.d_model = d_model
                    self.s5 = S5Layer(d_model=d_model, l_max=seq_len)
                    self.proj = nn.Linear(d_model, horizon)

                def forward(self, x):
                    dim = self.s5.d_model if hasattr(self.s5, "d_model") else self.d_model
                    x = x.repeat(1, 1, dim)
                    out = self.s5(x)[0][:, -1, :]
                    return self.proj(out)

            device = "cuda" if use_cuda else "cpu"
            s5_frames = []
            for uid, g in nf_df.groupby("unique_id"):
                series = g["y"].values.astype(float)
                if len(series) <= 70:
                    continue
                seq_len = min(64, max(32, len(series) // 4))
                X, y = [], []
                for i in range(len(series) - seq_len - horizon + 1):
                    X.append(series[i : i + seq_len])
                    y.append(series[i + seq_len : i + seq_len + horizon])
                X_t = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1).to(device)
                y_t = torch.tensor(np.array(y), dtype=torch.float32).to(device)
                d_model = model_params.get("s5_d_model", 32)
                s5_epochs = model_params.get("s5_epochs", 50)
                model = S5Forecaster(horizon, seq_len, d_model=d_model).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=model_params.get("learning_rate", 1e-3))
                loss_fn = torch.nn.MSELoss()
                for _ in range(s5_epochs):
                    opt.zero_grad()
                    pred = model(X_t)
                    loss = loss_fn(pred, y_t)
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    last_seq = torch.tensor(series[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                    s5_pred = model(last_seq).cpu().numpy().flatten()
                s5_frames.append(
                    pd.DataFrame(
                        {
                            "ds": pd.date_range(start=g["ds"].max(), periods=horizon + 1, freq=freq)[1:],
                            "unique_id": uid,
                            "model": "S5",
                            "forecast": s5_pred,
                        }
                    )
                )
            if s5_frames:
                fcsts.append(pd.concat(s5_frames, ignore_index=True))
        except Exception:
            pass

    if not fcsts:
        return pd.DataFrame(columns=["ds", "unique_id", "model", "forecast"])
    return pd.concat(fcsts, ignore_index=True)


def backtest(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    horizon: int,
    freq: str,
    models: List[str],
    n_windows: int = 2,
    model_params: dict | None = None,
) -> pd.DataFrame:
    """Cross-validation for NeuralForecast models."""
    torch, NeuralForecast, RNN, MAE = _lazy_import()
    model_params = model_params or {}
    nf_df = _prepare_data(df, date_col, target_col)
    pref = model_params.get("device_preference")
    use_cuda = pref == "cuda" and torch.cuda.is_available()
    if pref == "cuda" and not torch.cuda.is_available():
        model_params["_cuda_unavailable"] = True
    def _model_kwargs(prefix: str) -> dict:
        return {
            "input_size": model_params.get(f"{prefix}_input", model_params.get("input_size", 24)),
            "max_steps": model_params.get(f"{prefix}_steps", model_params.get("max_steps", 200)),
            "learning_rate": model_params.get(f"{prefix}_lr", model_params.get("learning_rate", 1e-3)),
            "batch_size": model_params.get(f"{prefix}_batch", model_params.get("batch_size", 32)),
            "encoder_hidden_size": model_params.get(f"{prefix}_hidden", 128),
            "encoder_n_layers": model_params.get(f"{prefix}_depth", 2),
            "context_length": model_params.get("context_length", None),
        }

    loss_obj = MAE()
    rnn_kwargs = _model_kwargs("rnn")
    lstm_kwargs = _model_kwargs("lstm")

    def _build_rnn_kwargs(base_kwargs: dict, series_len: int) -> dict:
        k = base_kwargs.copy()
        k["batch_size"] = max(1, min(k["batch_size"], series_len - max(horizon, 1)))
        return _filter_kwargs(
            RNN,
            {
                "h": horizon,
                "input_size": k["input_size"],
                "max_steps": k["max_steps"],
                "scaler_type": "standard",
                "loss": loss_obj,
                "learning_rate": k["learning_rate"],
                "batch_size": k["batch_size"],
                "encoder_hidden_size": k["encoder_hidden_size"],
                "encoder_n_layers": k["encoder_n_layers"],
                "context_length": k.get("context_length"),
            },
        )

    cv_frames = []
    for uid, g in nf_df.groupby("unique_id"):
        g = g.sort_values("ds")
        series_len = len(g)
        if series_len <= horizon + 5:
            continue
        nf_models = []
        for name in models:
            if name == "RNN":
                nf_models.append(RNN(**_build_rnn_kwargs(rnn_kwargs, series_len)))
            if name == "LSTM":
                nf_models.append(RNN(**_build_rnn_kwargs(lstm_kwargs, series_len)))
        if not nf_models:
            continue
        nf = NeuralForecast(models=nf_models, freq=freq, local_scaler_type="standard")
        cv = nf.cross_validation(df=g, h=horizon, n_windows=n_windows, step_size=horizon).reset_index()
        forecast_cols = [c for c in cv.columns if c not in ["ds", "unique_id", "cutoff", "y"]]
        rename_map = {}
        for i, col in enumerate(forecast_cols):
            name = models[i] if i < len(models) else col
            rename_map[col] = name
        cv = cv.rename(columns=rename_map)
        cv = cv.melt(
            id_vars=["ds", "unique_id", "cutoff", "y"],
            var_name="model",
            value_name="forecast",
        )
        cv_frames.append(cv)

    if not cv_frames:
        return pd.DataFrame(columns=["ds", "unique_id", "cutoff", "y", "model", "forecast"])
    return pd.concat(cv_frames, ignore_index=True)
