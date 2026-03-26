"""
alpaca_data.py
Utilities to fetch OHLCV bar data from Alpaca using alpaca-py for hourly, daily,
and monthly frequencies, and to convert to Nixtla-friendly long format.

Usage requires Alpaca credentials supplied either via:
  ALPACA_API_KEY / ALPACA_API_SECRET environment variables
  or explicit runtime configuration through configure_alpaca_credentials()

Install hint (not executed):
  pip install alpaca-py pandas
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def _normalize_datetime(value) -> datetime:
    """Convert input to datetime; accept datetime or ISO8601 string."""
    if isinstance(value, datetime):
        return value
    return pd.to_datetime(value).to_pydatetime()


def configure_alpaca_credentials(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    persist_env: bool = True,
) -> Tuple[str, str]:
    """
    Resolve Alpaca credentials from explicit values or the environment.

    This is the shared entrypoint used by the Streamlit UI and notebooks so
    callers can either type credentials into the app or rely on pre-set env
    vars.
    """
    resolved_key = (api_key or os.getenv("ALPACA_API_KEY") or "").strip()
    resolved_secret = (api_secret or os.getenv("ALPACA_API_SECRET") or "").strip()
    if not resolved_key or not resolved_secret:
        raise RuntimeError(
            "Missing Alpaca credentials. Enter an API key and secret in the app, "
            "or set ALPACA_API_KEY and ALPACA_API_SECRET."
        )
    if persist_env:
        os.environ["ALPACA_API_KEY"] = resolved_key
        os.environ["ALPACA_API_SECRET"] = resolved_secret
    return resolved_key, resolved_secret


def get_alpaca_client(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
) -> StockHistoricalDataClient:
    """
    Instantiate an Alpaca StockHistoricalDataClient using environment variables.

    Raises:
        RuntimeError: if ALPACA_API_KEY or ALPACA_API_SECRET are missing.
    """
    resolved_key, resolved_secret = configure_alpaca_credentials(
        api_key=api_key,
        api_secret=api_secret,
        persist_env=True,
    )
    return StockHistoricalDataClient(resolved_key, resolved_secret)


def fetch_stock_bars_raw(
    client: StockHistoricalDataClient,
    symbols: List[str],
    timeframe: TimeFrame,
    start,
    end,
    feed: str = "sip",
) -> pd.DataFrame:
    """
    Fetch raw OHLCV bars for multiple symbols and return standardized DataFrame.

    Args:
        client: Configured Alpaca StockHistoricalDataClient.
        symbols: Non-empty list of ticker strings.
        timeframe: Alpaca TimeFrame (e.g., TimeFrame.Hour, TimeFrame.Day).
        start: Start datetime or ISO8601 string.
        end: End datetime or ISO8601 string.
        feed: Market data feed (e.g., "sip" or "iex").

    Returns:
        DataFrame with columns: symbol, timestamp, open, high, low, close, volume.

    Raises:
        ValueError: if symbols list is empty or start >= end.
        Alpaca exceptions propagate on API errors.
    """
    if not symbols:
        raise ValueError("symbols must be a non-empty list of tickers")
    start_dt = _normalize_datetime(start)
    end_dt = _normalize_datetime(end)
    if start_dt >= end_dt:
        raise ValueError("start must be before end")

    request_params = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=start_dt,
        end=end_dt,
        feed=feed,
    )
    # API call (may raise Alpaca errors; callers can catch/log upstream)
    bars = client.get_stock_bars(request_params)
    df = bars.df
    if df.empty:
        return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
    df = df.reset_index()
    df = df.rename(columns={"timestamp": "timestamp", "symbol": "symbol"})
    df = df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def fetch_hourly_bars(symbols: List[str], start, end, feed: str = "sip") -> pd.DataFrame:
    """Fetch hourly OHLCV bars for symbols between start and end."""
    client = get_alpaca_client()
    return fetch_stock_bars_raw(client, symbols, TimeFrame.Hour, start, end, feed=feed)


def fetch_daily_bars(symbols: List[str], start, end, feed: str = "sip") -> pd.DataFrame:
    """Fetch daily OHLCV bars for symbols between start and end."""
    client = get_alpaca_client()
    return fetch_stock_bars_raw(client, symbols, TimeFrame.Day, start, end, feed=feed)


def fetch_intraday_bars(symbols: List[str], start, end, minutes: int = 1, feed: str = "sip") -> pd.DataFrame:
    """
    Fetch intraday OHLCV bars for symbols with a given minute interval.
    minutes can be 1, 5, 10, etc. depending on Alpaca allowances.
    """
    client = get_alpaca_client()
    tf = TimeFrame(amount=minutes, unit=TimeFrameUnit.Minute)
    return fetch_stock_bars_raw(client, symbols, tf, start, end, feed=feed)


def fetch_monthly_bars(symbols: List[str], start, end, feed: str = "sip") -> pd.DataFrame:
    """
    Fetch daily bars then resample to calendar month-end OHLCV.

    Aggregation per symbol:
      open = first open of month
      high = max high
      low  = min low
      close = last close
      volume = sum volume
    """
    daily = fetch_daily_bars(symbols, start, end, feed=feed)
    if daily.empty:
        return daily

    results = []
    for sym, group in daily.groupby("symbol"):
        g = group.set_index("timestamp").sort_index()
        resampled = pd.DataFrame({
            "open": g["open"].resample("M").first(),
            "high": g["high"].resample("M").max(),
            "low": g["low"].resample("M").min(),
            "close": g["close"].resample("M").last(),
            "volume": g["volume"].resample("M").sum(),
        }).dropna()
        resampled["symbol"] = sym
        resampled["timestamp"] = resampled.index
        results.append(resampled.reset_index(drop=True))

    monthly = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if not monthly.empty:
        monthly = monthly[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]
    return monthly


def to_nixtla_long(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Convert standardized OHLCV DataFrame to Nixtla-friendly long format.

    Returns columns: unique_id, ds, y
    """
    if df.empty:
        return pd.DataFrame(columns=["unique_id", "ds", "y"])
    out = df[["symbol", "timestamp", price_col]].copy()
    out = out.rename(columns={"symbol": "unique_id", "timestamp": "ds", price_col: "y"})
    out["ds"] = pd.to_datetime(out["ds"], utc=True)
    return out.sort_values(["unique_id", "ds"]).reset_index(drop=True)


if __name__ == "__main__":
    symbols = ["AAPL", "MSFT", "GOOG"]
    end = datetime.utcnow()
    start = end - timedelta(days=365)

    print("Fetching hourly bars...")
    hourly = fetch_hourly_bars(symbols, start, end)
    print(hourly.head())

    print("Fetching daily bars...")
    daily = fetch_daily_bars(symbols, start, end)
    print(daily.head())

    print("Fetching monthly bars (resampled from daily)...")
    monthly = fetch_monthly_bars(symbols, start, end)
    print(monthly.head())

    print("Nixtla long example (daily, close -> y):")
    nixtla_daily = to_nixtla_long(daily)
    print(nixtla_daily.head())
