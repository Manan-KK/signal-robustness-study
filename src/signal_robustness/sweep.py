"""Complete fast/slow-window sweeps over the same public evaluation contract."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from signal_robustness.contracts import EvaluationSpec, FeatureSpec, LabelSpec
from signal_robustness.evaluation import forecast_time_ordered, summarize_forecasts
from signal_robustness.labels import build_observations
from signal_robustness.models import MODEL_NAMES


SWEEP_COLUMNS = (
    "fast_window",
    "slow_window",
    "model",
    "observations",
    "positive_fraction",
    "brier_score",
    "history_base_rate_brier",
    "brier_skill",
    "roc_auc",
)


def run_window_sweep(
    prices: pd.DataFrame,
    *,
    fast_windows: Iterable[int] = (5, 10, 15),
    slow_windows: Iterable[int] = (20, 30, 40),
    label_spec: LabelSpec = LabelSpec(),
    evaluation_spec: EvaluationSpec = EvaluationSpec(),
    model: str = "ridge_logit",
) -> pd.DataFrame:
    """Evaluate every declared rectangular window pair without selection."""

    fast = _axis("fast_windows", fast_windows)
    slow = _axis("slow_windows", slow_windows)
    if max(fast) >= min(slow):
        raise ValueError("every fast window must be smaller than every slow window")
    if model not in MODEL_NAMES:
        raise ValueError(f"unknown model: {model}")
    rows = []
    for slow_window in slow:
        for fast_window in fast:
            observations = build_observations(
                prices,
                FeatureSpec(
                    fast_window=fast_window,
                    slow_window=slow_window,
                    trend_window=min(20, slow_window),
                ),
                label_spec,
            )
            forecasts = forecast_time_ordered(
                observations, evaluation_spec, model_names=(model,)
            )
            if forecasts.empty:
                raise ValueError(
                    f"insufficient resolved training rows for ({fast_window}, {slow_window})"
                )
            summary = summarize_forecasts(forecasts).iloc[0]
            row = {
                "fast_window": fast_window,
                "slow_window": slow_window,
                **summary.to_dict(),
            }
            numeric = [
                row["positive_fraction"],
                row["brier_score"],
                row["history_base_rate_brier"],
                row["brier_skill"],
                row["roc_auc"],
            ]
            if not np.isfinite(np.asarray(numeric, dtype=float)).all():
                raise ValueError("sweep produced non-finite aggregate metrics")
            rows.append(row)
    output = pd.DataFrame(rows, columns=SWEEP_COLUMNS)
    if len(output) != len(fast) * len(slow):
        raise AssertionError("sweep matrix is incomplete")
    return output


def _axis(name: str, values: Iterable[int]) -> tuple[int, ...]:
    axis = tuple(values)
    if not axis or len(axis) != len(set(axis)) or tuple(sorted(axis)) != axis:
        raise ValueError(f"{name} must be nonempty, unique, and increasing")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 2
        for value in axis
    ):
        raise ValueError(f"{name} must contain integers of at least two")
    return axis
