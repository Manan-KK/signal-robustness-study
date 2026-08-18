from __future__ import annotations

import unittest

import numpy as np

from signal_robustness.contracts import EvaluationSpec, FeatureSpec, LabelSpec
from signal_robustness.evaluation import (
    forecast_time_ordered,
    roc_auc,
    summarize_forecasts,
)
from signal_robustness.labels import build_observations
from signal_robustness.models import MODEL_NAMES
from signal_robustness.synthetic import SyntheticSpec, generate_synthetic_prices


class TimeOrderedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prices = generate_synthetic_prices(SyntheticSpec(days=700, seed=23))
        cls.observations = build_observations(
            prices,
            FeatureSpec(fast_window=8, slow_window=24, trend_window=16),
            LabelSpec(horizon=10, recovery_return=0.02, decision_stride=4),
        )
        cls.spec = EvaluationSpec(
            minimum_train=30,
            history_window=120,
            nearest_neighbors=15,
            ridge_iterations=60,
        )

    def test_all_models_use_identical_decisions_and_resolved_training_sets(self):
        forecasts = forecast_time_ordered(self.observations, self.spec)
        self.assertFalse(forecasts.empty)
        self.assertEqual(set(forecasts["model"]), set(MODEL_NAMES))
        counts = forecasts.groupby("model").size()
        self.assertEqual(counts.nunique(), 1)
        self.assertTrue(
            forecasts["latest_training_outcome_end"]
            .lt(forecasts["decision_date"])
            .all()
        )
        per_decision = forecasts.groupby("decision_date")["training_observations"]
        self.assertTrue(per_decision.nunique().eq(1).all())
        self.assertTrue(forecasts["probability"].between(0.0, 1.0).all())

    def test_summary_reports_proper_score_baseline_and_auc(self):
        forecasts = forecast_time_ordered(self.observations, self.spec)
        summary = summarize_forecasts(forecasts)
        self.assertEqual(set(summary["model"]), set(MODEL_NAMES))
        self.assertTrue(summary["observations"].gt(0).all())
        self.assertTrue(np.isfinite(summary["brier_skill"]).all())
        self.assertTrue(summary["roc_auc"].between(0.0, 1.0).all())

    def test_unknown_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown models"):
            forecast_time_ordered(self.observations, self.spec, ("not_a_model",))

    def test_auc_uses_average_ranks_for_ties(self):
        labels = np.array([0.0, 1.0, 0.0, 1.0])
        self.assertAlmostEqual(roc_auc(labels, np.ones(4)), 0.5)
        self.assertAlmostEqual(roc_auc(labels, labels), 1.0)


if __name__ == "__main__":
    unittest.main()
