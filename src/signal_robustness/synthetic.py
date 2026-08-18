"""Deterministic synthetic price series for tests and the public demo."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticSpec:
    days: int = 900
    seed: int = 20260818
    start: str = "2015-01-02"

    def __post_init__(self) -> None:
        if not isinstance(self.days, int) or isinstance(self.days, bool):
            raise ValueError("days must be an integer")
        if self.days < 250:
            raise ValueError("days must be at least 250")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if pd.isna(pd.to_datetime(self.start, errors="coerce")):
            raise ValueError("start must be a valid date")


def generate_synthetic_prices(spec: SyntheticSpec = SyntheticSpec()) -> pd.DataFrame:
    """Return a positive business-day close series with changing volatility.

    The generator is intentionally not calibrated to a security. It mixes a
    deterministic cycle, seeded Gaussian noise, and sparse symmetric shocks so
    the validation paths see both classes without embedding real observations.
    """

    rng = np.random.default_rng(spec.seed)
    positions = np.arange(spec.days, dtype=float)
    regime = (positions // 90).astype(int) % 4
    volatility = np.take(np.array([0.006, 0.012, 0.021, 0.009]), regime)
    drift = 0.00015 + 0.00018 * np.sin(positions / 47.0)
    cyclical = 0.0012 * np.sin(positions / 8.0)
    returns = drift + cyclical + rng.normal(0.0, volatility)

    shock_positions = np.arange(75, spec.days, 137)
    shock_signs = np.where((np.arange(len(shock_positions)) % 2) == 0, -1.0, 1.0)
    returns[shock_positions] += shock_signs * 0.045

    log_close = math.log(100.0) + np.cumsum(returns)
    close = np.exp(log_close)
    return pd.DataFrame(
        {
            "date": pd.bdate_range(spec.start, periods=spec.days),
            "close": close,
        }
    )
