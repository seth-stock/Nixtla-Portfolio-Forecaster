"""
models_neuralforecast.py
NeuralForecast wrappers for recurrent neural models.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import inspect
import numpy as np
import pandas as pd


def _lazy_import():
    """Local import to avoid hard failure when neural dependencies are missing."""
    try:
        import torch  # type: ignore
        from neuralforecast import NeuralForecast  # type: ignore
        from neuralforecast.losses.pytorch import MAE  # type: ignore
        from neuralforecast.models import LSTM, RNN  # type: ignore
    except Exception as e:
        raise ImportError(
            "neuralforecast and torch are required for neural models. "
            "Install a CPU or CUDA-compatible PyTorch build plus neuralforecast. "
            f"Original error: {e}"
        ) from e
    return torch, NeuralForecast, RNN, LSTM, MAE


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
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return params
    return {k: v for k, v in params.items() if k in sig.parameters}


def _resolve_trainer_kwargs(torch, model_params: dict) -> Tuple[bool, dict]:
    pref = str(model_params.get("device_preference", "cpu")).lower()
    cpu_threads = int(model_params.get("cpu_threads", 0) or 0)
    if pref == "cuda" and getattr(torch, "cuda", None) and torch.cuda.is_available():
        return True, {"accelerator": "gpu", "devices": 1, "enable_progress_bar": False}
    if cpu_threads > 0:
        try:
            torch.set_num_threads(cpu_threads)
        except Exception:
            pass
    return False, {"accelerator": "cpu", "devices": 1, "enable_progress_bar": False}


def _build_recurrent_kwargs(
    model_cls,
    model_name: str,
    prefix: str,
    horizon: int,
    series_len: int,
    loss_obj,
    model_params: dict,
    trainer_kwargs: dict,
) -> dict:
    input_size = int(model_params.get(f"{prefix}_input", model_params.get("input_size", 24)))
    max_steps = int(model_params.get(f"{prefix}_steps", model_params.get("max_steps", 200)))
    learning_rate = float(model_params.get(f"{prefix}_lr", model_params.get("learning_rate", 1e-3)))
    batch_size = int(model_params.get(f"{prefix}_batch", model_params.get("batch_size", 32)))
    hidden_size = int(model_params.get(f"{prefix}_hidden", model_params.get("hidden_size", 128)))
    depth = int(model_params.get(f"{prefix}_depth", model_params.get("depth", 2)))
    head_size = int(model_params.get(f"{prefix}_head", model_params.get("head_size", 32)))
    forecast_mode = model_params.get("forecast_mode", "multi-output (direct)")
    recurrent = forecast_mode != "multi-output (direct)"
    safe_batch = max(1, min(batch_size, max(1, series_len - max(horizon, 1))))
    kwargs = {
        "h": horizon,
        "input_size": max(1, input_size),
        "max_steps": max(1, max_steps),
        "scaler_type": "standard",
        "loss": loss_obj,
        "learning_rate": learning_rate,
        "batch_size": safe_batch,
        "encoder_hidden_size": max(4, hidden_size),
        "encoder_n_layers": max(1, depth),
        "decoder_hidden_size": max(4, head_size),
        "decoder_layers": max(1, min(depth, 2)),
        "recurrent": recurrent,
        "alias": model_name,
        **trainer_kwargs,
    }
    return _filter_kwargs(model_cls, kwargs)


def _build_nf_models(
    series_len: int,
    horizon: int,
    models: List[str],
    model_params: dict,
    loss_obj,
    trainer_kwargs: dict,
    RNN,
    LSTM,
):
    model_map = {"RNN": (RNN, "rnn"), "LSTM": (LSTM, "lstm")}
    built = []
    for name in models:
        if name not in model_map:
            continue
        model_cls, prefix = model_map[name]
        built.append(
            model_cls(
                **_build_recurrent_kwargs(
                    model_cls=model_cls,
                    model_name=name,
                    prefix=prefix,
                    horizon=horizon,
                    series_len=series_len,
                    loss_obj=loss_obj,
                    model_params=model_params,
                    trainer_kwargs=trainer_kwargs,
                )
            )
        )
    return built


def fit_and_forecast(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    horizon: int,
    freq: str,
    models: List[str],
    model_params: dict | None = None,
) -> pd.DataFrame:
    """Fit NeuralForecast models and return forecasts."""
    torch, NeuralForecast, RNN, LSTM, MAE = _lazy_import()
    model_params = model_params or {}
    nf_df = _prepare_data(df, date_col, target_col)
    use_cuda, trainer_kwargs = _resolve_trainer_kwargs(torch, model_params)
    loss_obj = MAE()

    fcsts = []
    for uid, g in nf_df.groupby("unique_id"):
        g = g.sort_values("ds")
        series_len = len(g)
        if series_len <= horizon + 5:
            continue
        nf_models = _build_nf_models(series_len, horizon, models, model_params, loss_obj, trainer_kwargs, RNN, LSTM)
        if nf_models:
            nf = NeuralForecast(models=nf_models, freq=freq, local_scaler_type="standard")
            nf.fit(g)
            fcst = nf.predict().reset_index()
            forecast_cols = [c for c in fcst.columns if c not in ["ds", "unique_id"]]
            if forecast_cols:
                fcsts.append(
                    fcst.melt(
                        id_vars=["ds", "unique_id"],
                        value_vars=forecast_cols,
                        var_name="model",
                        value_name="forecast",
                    )
                )

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
                    x = x.repeat(1, 1, self.d_model)
                    out = self.s5(x)[0][:, -1, :]
                    return self.proj(out)

            device = "cuda" if use_cuda else "cpu"
            s5_frames = []
            for uid, g in nf_df.groupby("unique_id"):
                series = g["y"].values.astype(float)
                if len(series) <= horizon + 32:
                    continue
                seq_len = min(64, max(16, len(series) // 4))
                X, y = [], []
                for i in range(len(series) - seq_len - horizon + 1):
                    X.append(series[i : i + seq_len])
                    y.append(series[i + seq_len : i + seq_len + horizon])
                if not X:
                    continue
                X_t = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1).to(device)
                y_t = torch.tensor(np.array(y), dtype=torch.float32).to(device)
                d_model = int(model_params.get("s5_d_model", 32))
                s5_epochs = int(model_params.get("s5_epochs", 50))
                s5_lr = float(model_params.get("s5_lr", 1e-3))
                model = S5Forecaster(horizon, seq_len, d_model=d_model).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=s5_lr)
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
    torch, NeuralForecast, RNN, LSTM, MAE = _lazy_import()
    model_params = model_params or {}
    nf_df = _prepare_data(df, date_col, target_col)
    _, trainer_kwargs = _resolve_trainer_kwargs(torch, model_params)
    loss_obj = MAE()

    cv_frames = []
    for _, g in nf_df.groupby("unique_id"):
        g = g.sort_values("ds")
        series_len = len(g)
        if series_len <= horizon + 5:
            continue
        nf_models = _build_nf_models(series_len, horizon, models, model_params, loss_obj, trainer_kwargs, RNN, LSTM)
        if not nf_models:
            continue
        nf = NeuralForecast(models=nf_models, freq=freq, local_scaler_type="standard")
        cv = nf.cross_validation(df=g, h=horizon, n_windows=n_windows, step_size=horizon).reset_index()
        forecast_cols = [c for c in cv.columns if c not in ["ds", "unique_id", "cutoff", "y"]]
        if not forecast_cols:
            continue
        cv_frames.append(
            cv.melt(
                id_vars=["ds", "unique_id", "cutoff", "y"],
                value_vars=forecast_cols,
                var_name="model",
                value_name="forecast",
            )
        )

    if not cv_frames:
        return pd.DataFrame(columns=["ds", "unique_id", "cutoff", "y", "model", "forecast"])
    return pd.concat(cv_frames, ignore_index=True)
