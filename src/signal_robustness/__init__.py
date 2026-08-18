"""Time-ordered signal robustness tools with synthetic-data examples."""

from signal_robustness.contracts import EvaluationSpec, FeatureSpec, LabelSpec
from signal_robustness.evaluation import forecast_time_ordered, summarize_forecasts
from signal_robustness.features import build_features, validate_prices
from signal_robustness.labels import build_observations
from signal_robustness.synthetic import SyntheticSpec, generate_synthetic_prices

__all__ = [
    "EvaluationSpec",
    "FeatureSpec",
    "LabelSpec",
    "SyntheticSpec",
    "build_features",
    "build_observations",
    "forecast_time_ordered",
    "generate_synthetic_prices",
    "summarize_forecasts",
    "validate_prices",
]
