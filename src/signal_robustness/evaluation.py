"""Same-label, resolved-before-decision probability evaluation."""

from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np
import pandas as pd

from signal_robustness.contracts import EvaluationSpec
from signal_robustness.features import FEATURE_COLUMNS
from signal_robustness.models import MODEL_NAMES, predict_probability


FORECAST_COLUMNS = (
    "decision_date",
    "outcome_end_date",
    "label",
    "model",
    "probability",
    "history_base_rate",
    "training_observations",
    "latest_training_outcome_end",
)


def forecast_time_ordered(
    observations: pd.DataFrame,
    spec: EvaluationSpec = EvaluationSpec(),
    model_names: Iterable[str] = MODEL_NAMES,
) -> pd.DataFrame:
    """Forecast each row using only labels resolved before its decision date."""

    frame = _validate_observations(observations)
    names = tuple(model_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("model_names must be nonempty and unique")
    unknown = set(names) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"unknown models: {', '.join(sorted(unknown))}")

    rows: list[dict[str, object]] = []
    for position, decision in frame.iterrows():
        train = frame.iloc[:position]
        train = train.loc[train["outcome_end_date"].lt(decision["decision_date"])]
        train = train.tail(spec.history_window)
        if len(train) < spec.minimum_train or train["label"].nunique() < 2:
            continue
        train_x = train.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
        train_y = train["label"].to_numpy(dtype=float)
        test_x = decision.loc[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        base_rate = float(train_y.mean())
        latest = pd.Timestamp(train["outcome_end_date"].max())
        for name in names:
            rows.append(
                {
                    "decision_date": decision["decision_date"],
                    "outcome_end_date": decision["outcome_end_date"],
                    "label": int(decision["label"]),
                    "model": name,
                    "probability": predict_probability(
                        name, train_x, train_y, test_x, spec
                    ),
                    "history_base_rate": base_rate,
                    "training_observations": len(train),
                    "latest_training_outcome_end": latest,
                }
            )
    return pd.DataFrame(rows, columns=FORECAST_COLUMNS)


def summarize_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Return model-level proper-score and ranking diagnostics."""

    missing = set(FORECAST_COLUMNS) - set(forecasts.columns)
    if missing:
        raise ValueError(f"forecasts are missing columns: {', '.join(sorted(missing))}")
    if forecasts.empty:
        raise ValueError("forecasts cannot be empty")
    rows = []
    for model, group in forecasts.groupby("model", sort=True):
        labels = group["label"].to_numpy(dtype=float)
        probabilities = group["probability"].to_numpy(dtype=float)
        baseline = group["history_base_rate"].to_numpy(dtype=float)
        brier = float(np.mean((probabilities - labels) ** 2))
        baseline_brier = float(np.mean((baseline - labels) ** 2))
        skill = 1.0 - brier / baseline_brier if baseline_brier > 0.0 else math.nan
        rows.append(
            {
                "model": model,
                "observations": len(group),
                "positive_fraction": float(labels.mean()),
                "mean_probability": float(probabilities.mean()),
                "brier_score": brier,
                "history_base_rate_brier": baseline_brier,
                "brier_skill": skill,
                "roc_auc": roc_auc(labels, probabilities),
            }
        )
    return pd.DataFrame(rows)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute rank AUC with average ranks for ties."""

    y = np.asarray(labels, dtype=float)
    values = np.asarray(scores, dtype=float)
    if len(y) != len(values) or len(y) == 0:
        raise ValueError("labels and scores must have equal nonzero length")
    positive = int(np.sum(y == 1.0))
    negative = int(np.sum(y == 0.0))
    if positive == 0 or negative == 0:
        return math.nan
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[y == 1.0].sum())
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def _validate_observations(observations: pd.DataFrame) -> pd.DataFrame:
    required = {"decision_date", "outcome_end_date", "label", *FEATURE_COLUMNS}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(
            f"observations are missing columns: {', '.join(sorted(missing))}"
        )
    frame = observations.loc[:, ["decision_date", "outcome_end_date", "label", *FEATURE_COLUMNS]].copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="coerce")
    frame["outcome_end_date"] = pd.to_datetime(
        frame["outcome_end_date"], errors="coerce"
    )
    if frame.isna().any().any():
        raise ValueError("observations contain missing or invalid values")
    if not frame["decision_date"].is_monotonic_increasing:
        raise ValueError("decision dates must be increasing")
    if frame["decision_date"].duplicated().any():
        raise ValueError("decision dates must be unique")
    if frame["outcome_end_date"].le(frame["decision_date"]).any():
        raise ValueError("every outcome must end after its decision")
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("labels must be binary")
    if not np.isfinite(frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("features must be finite")
    return frame.reset_index(drop=True)
