from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from signal_robustness.contracts import EvaluationSpec, LabelSpec
from signal_robustness.study import run_study
from signal_robustness.sweep import run_window_sweep
from signal_robustness.synthetic import SyntheticSpec, generate_synthetic_prices
from signal_robustness.viewer import render_viewer


class SweepViewerAndStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = generate_synthetic_prices(SyntheticSpec(days=680, seed=41))
        cls.label_spec = LabelSpec(
            horizon=10, recovery_return=0.02, decision_stride=4
        )
        cls.evaluation_spec = EvaluationSpec(
            minimum_train=30,
            history_window=100,
            nearest_neighbors=12,
            ridge_iterations=45,
        )

    def test_sweep_contains_every_declared_pair(self):
        sweep = run_window_sweep(
            self.prices,
            fast_windows=(5, 8),
            slow_windows=(20, 30),
            label_spec=self.label_spec,
            evaluation_spec=self.evaluation_spec,
        )
        self.assertEqual(len(sweep), 4)
        self.assertEqual(
            set(zip(sweep["fast_window"], sweep["slow_window"])),
            {(5, 20), (8, 20), (5, 30), (8, 30)},
        )

    def test_viewer_contains_aggregates_only_and_has_no_external_dependency(self):
        sweep = run_window_sweep(
            self.prices,
            fast_windows=(5, 8),
            slow_windows=(20, 30),
            label_spec=self.label_spec,
            evaluation_spec=self.evaluation_spec,
        )
        html = render_viewer(sweep)
        self.assertIn("Brier skill", html)
        self.assertIn("aggregate score", html)
        for forbidden in (
            "/" + "Users" + "/",
            "file:" + "//",
            "plotly",
            "PySide",
            "Corda" + "tus",
            "decision_date",
            "outcome_end_date",
        ):
            self.assertNotIn(forbidden, html)

    def test_viewer_rejects_an_incomplete_matrix(self):
        frame = pd.DataFrame(
            [
                {"fast_window": 5, "slow_window": 20, "observations": 10, "brier_skill": 0.1, "roc_auc": 0.6},
                {"fast_window": 8, "slow_window": 20, "observations": 10, "brier_skill": 0.0, "roc_auc": 0.5},
                {"fast_window": 5, "slow_window": 30, "observations": 10, "brier_skill": -0.1, "roc_auc": 0.4},
            ]
        )
        with self.assertRaisesRegex(ValueError, "complete rectangular grid"):
            render_viewer(frame)

    def test_end_to_end_study_writes_only_declared_aggregate_artifacts(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = run_study(
                self.prices,
                Path(first_dir),
                label_spec=self.label_spec,
                evaluation_spec=self.evaluation_spec,
            )
            second = run_study(
                self.prices,
                Path(second_dir),
                label_spec=self.label_spec,
                evaluation_spec=self.evaluation_spec,
            )
            self.assertEqual(
                set(first), {"model_summary", "window_sweep", "split_audit", "viewer"}
            )
            for name in first:
                self.assertEqual(first[name].read_bytes(), second[name].read_bytes())
            model_columns = set(pd.read_csv(first["model_summary"]).columns)
            self.assertNotIn("decision_date", model_columns)
            self.assertNotIn("outcome_end_date", model_columns)
            self.assertFalse(any(path.suffix in {".parquet", ".xlsx"} for path in first.values()))


if __name__ == "__main__":
    unittest.main()
