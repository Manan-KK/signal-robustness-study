"""Small probability models implemented for auditability, not optimization."""

from __future__ import annotations

import math

import numpy as np

from signal_robustness.contracts import EvaluationSpec


MODEL_NAMES = (
    "beta_base_rate",
    "fixed_bin",
    "nearest_analog",
    "ridge_logit",
    "gaussian_nb",
)


def predict_probability(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    spec: EvaluationSpec,
) -> float:
    """Fit one declared model and return a finite probability."""

    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model: {name}")
    x, y, point = _validate_arrays(train_x, train_y, test_x)
    if name == "beta_base_rate":
        probability = (float(y.sum()) + 1.0) / (len(y) + 2.0)
    elif name == "fixed_bin":
        probability = _fixed_bin(x[:, 0], y, point[0])
    elif name == "nearest_analog":
        probability = _nearest_analog(x, y, point, spec.nearest_neighbors)
    elif name == "ridge_logit":
        probability = _ridge_logit(x, y, point, spec)
    else:
        probability = _gaussian_naive_bayes(x, y, point)
    if not math.isfinite(probability):
        raise FloatingPointError(f"{name} returned a non-finite probability")
    return float(np.clip(probability, 0.0, 1.0))


def _validate_arrays(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(train_x, dtype=float)
    y = np.asarray(train_y, dtype=float)
    point = np.asarray(test_x, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or point.ndim != 1:
        raise ValueError("model arrays have invalid dimensions")
    if len(x) != len(y) or x.shape[1] != len(point) or len(x) == 0:
        raise ValueError("model arrays have incompatible shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(point).all():
        raise ValueError("model arrays must be finite")
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("training labels must be binary")
    if len(np.unique(y)) < 2:
        raise ValueError("training labels must contain both classes")
    return x, y, point


def _standardize(
    x: np.ndarray, point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (x - mean) / scale, (point - mean) / scale


def _fixed_bin(primary: np.ndarray, y: np.ndarray, point: float) -> float:
    edges = np.unique(np.quantile(primary, [0.2, 0.4, 0.6, 0.8]))
    train_bins = np.searchsorted(edges, primary, side="right")
    point_bin = int(np.searchsorted(edges, point, side="right"))
    selected = y[train_bins == point_bin]
    if len(selected) == 0:
        selected = y
    return (float(selected.sum()) + 1.0) / (len(selected) + 2.0)


def _nearest_analog(
    x: np.ndarray,
    y: np.ndarray,
    point: np.ndarray,
    neighbors: int,
) -> float:
    standardized, target = _standardize(x, point)
    distances = np.sum((standardized - target) ** 2, axis=1)
    count = min(neighbors, len(distances))
    chosen = np.argpartition(distances, count - 1)[:count]
    return (float(y[chosen].sum()) + 1.0) / (count + 2.0)


def _ridge_logit(
    x: np.ndarray,
    y: np.ndarray,
    point: np.ndarray,
    spec: EvaluationSpec,
) -> float:
    standardized, target = _standardize(x, point)
    design = np.column_stack([np.ones(len(standardized)), standardized])
    weights = np.zeros(design.shape[1], dtype=float)
    regularizer = np.ones_like(weights)
    regularizer[0] = 0.0
    for _ in range(spec.ridge_iterations):
        scores = np.clip(design @ weights, -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-scores))
        gradient = design.T @ (probabilities - y) / len(y)
        gradient += spec.ridge_alpha * regularizer * weights / len(y)
        weights -= spec.ridge_step * gradient
    score = float(np.clip(np.r_[1.0, target] @ weights, -35.0, 35.0))
    return 1.0 / (1.0 + math.exp(-score))


def _gaussian_naive_bayes(
    x: np.ndarray,
    y: np.ndarray,
    point: np.ndarray,
) -> float:
    log_scores = []
    for label in (0.0, 1.0):
        selected = x[y == label]
        prior = (len(selected) + 1.0) / (len(y) + 2.0)
        mean = selected.mean(axis=0)
        variance = np.maximum(selected.var(axis=0, ddof=0), 1e-6)
        log_likelihood = -0.5 * np.sum(
            np.log(2.0 * math.pi * variance) + ((point - mean) ** 2) / variance
        )
        log_scores.append(math.log(prior) + float(log_likelihood))
    maximum = max(log_scores)
    scaled = np.exp(np.asarray(log_scores) - maximum)
    return float(scaled[1] / scaled.sum())
