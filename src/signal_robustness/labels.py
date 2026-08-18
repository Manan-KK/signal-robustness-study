"""Forward label construction with explicit availability dates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_robustness.contracts import FeatureSpec, LabelSpec
from signal_robustness.features import FEATURE_COLUMNS, build_features


OBSERVATION_COLUMNS = (
    "decision_date",
    "outcome_end_date",
    "label",
    *FEATURE_COLUMNS,
)


def build_observations(
    prices: pd.DataFrame,
    feature_spec: FeatureSpec = FeatureSpec(),
    label_spec: LabelSpec = LabelSpec(),
) -> pd.DataFrame:
    """Build decision rows and labels resolved at a declared future date.

    A label is one when any close in the next ``horizon`` rows reaches the
    configured return above the decision close. Regardless of when a recovery
    first occurs, the label is treated as unresolved until the full horizon ends.
    This conservative availability rule makes the training cutoff unambiguous.
    """

    features = build_features(prices, feature_spec)
    closes = features["close"].to_numpy(dtype=float)
    dates = features["date"].to_numpy()
    rows: list[dict[str, object]] = []
    last_decision = len(features) - label_spec.horizon
    ready = features.loc[
        features.loc[:, FEATURE_COLUMNS].notna().all(axis=1)
        & np.isfinite(features.loc[:, FEATURE_COLUMNS]).all(axis=1)
        & features["source_position"].lt(last_decision)
    ]
    for item in ready.iloc[:: label_spec.decision_stride].itertuples(index=False):
        position = int(item.source_position)
        forward = closes[position + 1 : position + label_spec.horizon + 1]
        maximum_return = float(np.max(forward / closes[position] - 1.0))
        row = {
            "decision_date": pd.Timestamp(dates[position]),
            "outcome_end_date": pd.Timestamp(dates[position + label_spec.horizon]),
            "label": int(maximum_return >= label_spec.recovery_return),
        }
        row.update({name: float(getattr(item, name)) for name in FEATURE_COLUMNS})
        rows.append(row)
    result = pd.DataFrame(rows, columns=OBSERVATION_COLUMNS)
    if result.empty:
        raise ValueError("price history is too short for the feature and label contracts")
    return result
