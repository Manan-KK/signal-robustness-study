from __future__ import annotations

import math
import unittest

from signal_robustness.contracts import FeatureSpec, LabelSpec
from signal_robustness.labels import build_observations
from signal_robustness.splits import (
    audit_splits,
    combinatorial_purged_splits,
    walk_forward_splits,
)
from signal_robustness.synthetic import SyntheticSpec, generate_synthetic_prices


class PurgedSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prices = generate_synthetic_prices(SyntheticSpec(days=720, seed=31))
        cls.observations = build_observations(
            prices,
            FeatureSpec(fast_window=7, slow_window=21, trend_window=14),
            LabelSpec(horizon=12, recovery_return=0.02, decision_stride=3),
        )

    def test_walk_forward_splits_are_anchored_disjoint_and_interval_clean(self):
        splits = walk_forward_splits(
            self.observations, minimum_train=40, test_size=25, embargo_rows=3
        )
        audit = audit_splits(self.observations, splits)
        self.assertGreater(len(splits), 1)
        self.assertTrue(audit["train_test_disjoint"].eq(1).all())
        self.assertTrue(audit["interval_overlap_count"].eq(0).all())
        for split in splits:
            self.assertLess(max(split.train_indices), min(split.test_indices))
            self.assertEqual(split.embargoed_rows, 3)

    def test_combinatorial_split_count_and_purge_invariants(self):
        splits = combinatorial_purged_splits(
            self.observations, groups=5, test_groups=2, embargo_rows=2
        )
        self.assertEqual(len(splits), math.comb(5, 2))
        audit = audit_splits(self.observations, splits)
        self.assertTrue(audit["train_test_disjoint"].eq(1).all())
        self.assertTrue(audit["interval_overlap_count"].eq(0).all())
        self.assertTrue(audit["purged_rows"].gt(0).any())
        self.assertTrue(audit["embargoed_rows"].gt(0).any())


if __name__ == "__main__":
    unittest.main()
