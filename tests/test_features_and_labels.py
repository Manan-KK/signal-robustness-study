from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from signal_robustness.contracts import FeatureSpec, LabelSpec
from signal_robustness.features import FEATURE_COLUMNS, build_features, validate_prices
from signal_robustness.labels import build_observations
from signal_robustness.synthetic import SyntheticSpec, generate_synthetic_prices


class FeatureAndLabelTests(unittest.TestCase):
    def test_synthetic_series_is_deterministic_positive_and_date_unique(self):
        spec = SyntheticSpec(days=300, seed=17)
        first = generate_synthetic_prices(spec)
        second = generate_synthetic_prices(spec)
        assert_frame_equal(first, second)
        self.assertTrue(first["close"].gt(0.0).all())
        self.assertTrue(first["date"].is_unique)
        self.assertTrue(first["date"].is_monotonic_increasing)

    def test_price_contract_rejects_unsorted_duplicate_and_invalid_close(self):
        valid = generate_synthetic_prices(SyntheticSpec(days=300))
        with self.assertRaisesRegex(ValueError, "increasing"):
            validate_prices(valid.iloc[::-1])
        duplicate = pd.concat([valid.iloc[:2], valid.iloc[[1]], valid.iloc[2:]])
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_prices(duplicate)
        invalid = valid.copy()
        invalid.loc[20, "close"] = np.inf
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            validate_prices(invalid)

    def test_features_before_a_cutoff_do_not_change_when_future_prices_change(self):
        prices = generate_synthetic_prices(SyntheticSpec(days=420, seed=3))
        changed = prices.copy()
        cutoff = 300
        changed.loc[cutoff:, "close"] *= np.linspace(1.2, 2.0, len(changed) - cutoff)
        spec = FeatureSpec(fast_window=8, slow_window=24, trend_window=15)
        original_features = build_features(prices, spec)
        changed_features = build_features(changed, spec)
        assert_frame_equal(
            original_features.loc[: cutoff - 1, FEATURE_COLUMNS],
            changed_features.loc[: cutoff - 1, FEATURE_COLUMNS],
        )

    def test_forward_labels_declare_the_full_horizon_end_date(self):
        prices = generate_synthetic_prices(SyntheticSpec(days=360, seed=9))
        label_spec = LabelSpec(horizon=12, recovery_return=0.015, decision_stride=3)
        observations = build_observations(
            prices,
            FeatureSpec(fast_window=6, slow_window=18, trend_window=12),
            label_spec,
        )
        date_positions = pd.Series(prices.index, index=prices["date"])
        for row in observations.head(20).itertuples(index=False):
            position = int(date_positions.loc[row.decision_date])
            self.assertEqual(
                row.outcome_end_date,
                prices.loc[position + label_spec.horizon, "date"],
            )
            forward = prices.loc[
                position + 1 : position + label_spec.horizon, "close"
            ]
            expected = int(
                (forward / prices.loc[position, "close"] - 1.0).max()
                >= label_spec.recovery_return
            )
            self.assertEqual(row.label, expected)


if __name__ == "__main__":
    unittest.main()
