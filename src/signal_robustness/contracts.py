"""Validated configuration contracts for the public study."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FeatureSpec:
    """Backward-looking feature windows measured in observations."""

    fast_window: int = 10
    slow_window: int = 30
    trend_window: int = 20

    def __post_init__(self) -> None:
        for name, value in (
            ("fast_window", self.fast_window),
            ("slow_window", self.slow_window),
            ("trend_window", self.trend_window),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 2:
                raise ValueError(f"{name} must be an integer of at least two")
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be smaller than slow_window")


@dataclass(frozen=True)
class LabelSpec:
    """Forward outcome definition used only for scoring and resolved training."""

    horizon: int = 10
    recovery_return: float = 0.02
    decision_stride: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.horizon, int) or isinstance(self.horizon, bool):
            raise ValueError("horizon must be an integer")
        if self.horizon < 2:
            raise ValueError("horizon must be at least two")
        if not math.isfinite(self.recovery_return) or self.recovery_return <= 0.0:
            raise ValueError("recovery_return must be finite and positive")
        if not isinstance(self.decision_stride, int) or isinstance(
            self.decision_stride, bool
        ):
            raise ValueError("decision_stride must be an integer")
        if self.decision_stride < 1:
            raise ValueError("decision_stride must be positive")


@dataclass(frozen=True)
class EvaluationSpec:
    """Time-ordered training and lightweight-model settings."""

    minimum_train: int = 60
    history_window: int = 240
    nearest_neighbors: int = 25
    ridge_alpha: float = 1.0
    ridge_iterations: int = 120
    ridge_step: float = 0.2

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_train", self.minimum_train),
            ("history_window", self.history_window),
            ("nearest_neighbors", self.nearest_neighbors),
            ("ridge_iterations", self.ridge_iterations),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.history_window < self.minimum_train:
            raise ValueError("history_window must cover minimum_train")
        if self.nearest_neighbors > self.history_window:
            raise ValueError("nearest_neighbors cannot exceed history_window")
        if not math.isfinite(self.ridge_alpha) or self.ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be finite and non-negative")
        if not math.isfinite(self.ridge_step) or self.ridge_step <= 0.0:
            raise ValueError("ridge_step must be finite and positive")
