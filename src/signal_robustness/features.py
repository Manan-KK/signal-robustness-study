"""Price validation and strictly backward-looking feature construction."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from signal_robustness.contracts import FeatureSpec


FEATURE_COLUMNS = (
    "vol_ratio",
    "rv_fast",
    "rv_slow",
    "return_5",
    "trend",
    "drawdown",
)


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy or reject an ambiguous price contract."""

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    missing = {"date", "close"} - set(prices.columns)
    if missing:
        raise ValueError(f"prices are missing columns: {', '.join(sorted(missing))}")
    frame = prices.loc[:, ["date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if frame.empty:
        raise ValueError("prices cannot be empty")
    if frame.isna().any().any():
        raise ValueError("date and close must not contain missing or invalid values")
    if frame["date"].dt.tz is not None:
        frame["date"] = frame["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    frame["date"] = frame["date"].dt.normalize()
    if frame["date"].duplicated().any():
        raise ValueError("dates must be unique")
    if not frame["date"].is_monotonic_increasing:
        raise ValueError("dates must be strictly increasing")
    close = frame["close"].to_numpy(dtype=float)
    if not np.isfinite(close).all() or (close <= 0.0).any():
        raise ValueError("close values must be finite and positive")
    return frame.reset_index(drop=True)


def build_features(
    prices: pd.DataFrame,
    spec: FeatureSpec = FeatureSpec(),
) -> pd.DataFrame:
    """Build features whose value at row *t* uses rows no later than *t*."""

    frame = validate_prices(prices)
    returns = frame["close"].pct_change(fill_method=None)
    annualizer = math.sqrt(252.0)
    rv_fast = returns.rolling(spec.fast_window, min_periods=spec.fast_window).std(
        ddof=1
    ) * annualizer
    rv_slow = returns.rolling(spec.slow_window, min_periods=spec.slow_window).std(
        ddof=1
    ) * annualizer
    rolling_high = frame["close"].rolling(
        spec.trend_window, min_periods=spec.trend_window
    ).max()
    rolling_mean = frame["close"].rolling(
        spec.trend_window, min_periods=spec.trend_window
    ).mean()

    output = frame.copy()
    output["source_position"] = np.arange(len(frame), dtype=int)
    output["vol_ratio"] = rv_fast.divide(rv_slow.where(rv_slow > 0.0))
    output["rv_fast"] = rv_fast
    output["rv_slow"] = rv_slow
    output["return_5"] = frame["close"].pct_change(5, fill_method=None)
    output["trend"] = frame["close"].divide(rolling_mean).subtract(1.0)
    output["drawdown"] = frame["close"].divide(rolling_high).subtract(1.0)
    return output
