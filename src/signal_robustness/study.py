"""End-to-end aggregate study orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from signal_robustness.contracts import EvaluationSpec, FeatureSpec, LabelSpec
from signal_robustness.evaluation import forecast_time_ordered, summarize_forecasts
from signal_robustness.features import validate_prices
from signal_robustness.labels import build_observations
from signal_robustness.splits import (
    audit_splits,
    combinatorial_purged_splits,
    walk_forward_splits,
)
from signal_robustness.sweep import run_window_sweep
from signal_robustness.viewer import render_viewer


def run_study(
    prices: pd.DataFrame,
    output_dir: Path,
    *,
    feature_spec: FeatureSpec = FeatureSpec(),
    label_spec: LabelSpec = LabelSpec(),
    evaluation_spec: EvaluationSpec = EvaluationSpec(),
) -> Mapping[str, Path]:
    """Run the public contracts and persist aggregate artifacts only."""

    clean = validate_prices(prices)
    observations = build_observations(clean, feature_spec, label_spec)
    forecasts = forecast_time_ordered(observations, evaluation_spec)
    if forecasts.empty:
        raise ValueError("not enough resolved training observations for evaluation")
    model_summary = summarize_forecasts(forecasts)
    counts = model_summary["observations"]
    if counts.nunique() != 1:
        raise AssertionError("models were not evaluated on identical rows")
    if not forecasts["latest_training_outcome_end"].lt(
        forecasts["decision_date"]
    ).all():
        raise AssertionError("a forecast used an unresolved training label")

    walk_forward = walk_forward_splits(
        observations,
        minimum_train=min(40, max(20, len(observations) // 4)),
        test_size=max(10, len(observations) // 8),
        embargo_rows=2,
    )
    combinatorial = combinatorial_purged_splits(
        observations, groups=6, test_groups=2, embargo_rows=2
    )
    split_audit = pd.concat(
        [
            audit_splits(observations, walk_forward),
            audit_splits(observations, combinatorial),
        ],
        ignore_index=True,
    )
    if not split_audit["train_test_disjoint"].eq(1).all():
        raise AssertionError("a validation split overlaps train and test rows")
    if not split_audit["interval_overlap_count"].eq(0).all():
        raise AssertionError("a validation split contains an overlapping label interval")

    sweep = run_window_sweep(
        clean,
        fast_windows=(5, 10, 15),
        slow_windows=(20, 30, 40),
        label_spec=label_spec,
        evaluation_spec=evaluation_spec,
        model="ridge_logit",
    )
    html = render_viewer(sweep)

    root = Path(output_dir)
    if root.exists() and not root.is_dir():
        raise ValueError("output_dir exists and is not a directory")
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "model_summary": root / "model_summary.csv",
        "window_sweep": root / "window_sweep.csv",
        "split_audit": root / "split_audit.csv",
        "viewer": root / "viewer.html",
    }
    _write_csv(model_summary, paths["model_summary"])
    _write_csv(sweep, paths["window_sweep"])
    _write_csv(split_audit, paths["split_audit"])
    paths["viewer"].write_text(html, encoding="utf-8")
    return paths


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.17g")
