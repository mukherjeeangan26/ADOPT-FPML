"""Accuracy thresholds and corrected Akaike information criterion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

Array = np.ndarray


def rmse_by_output(
    prediction: pd.DataFrame | Array,
    truth: pd.DataFrame | Array,
    names: Sequence[str] | None = None,
) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=float)
    true = np.asarray(truth, dtype=float)
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/truth shapes differ: {pred.shape} != {true.shape}")
    labels = list(names or getattr(truth, "columns", range(true.shape[1])))
    values = np.sqrt(np.mean((pred - true) ** 2, axis=0))
    return {str(name): float(value) for name, value in zip(labels, values)}


def mae_by_output(
    prediction: pd.DataFrame | Array,
    truth: pd.DataFrame | Array,
    names: Sequence[str] | None = None,
) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=float)
    true = np.asarray(truth, dtype=float)
    labels = list(names or getattr(truth, "columns", range(true.shape[1])))
    values = np.mean(np.abs(pred - true), axis=0)
    return {str(name): float(value) for name, value in zip(labels, values)}


def relative_rmse_by_output(
    prediction: pd.DataFrame | Array,
    truth: pd.DataFrame | Array,
    names: Sequence[str] | None = None,
) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=float)
    true = np.asarray(truth, dtype=float)
    labels = list(names or getattr(truth, "columns", range(true.shape[1])))
    scale = np.maximum(np.ptp(true, axis=0), np.maximum(np.std(true, axis=0), 1.0e-12))
    values = np.sqrt(np.mean((pred - true) ** 2, axis=0)) / scale
    return {str(name): float(value) for name, value in zip(labels, values)}


def aicc(prediction: pd.DataFrame | Array, truth: pd.DataFrame | Array, k: int) -> float:
    """Gaussian-error AICc, omitting the common additive constant.

    The effective observation count is every scalar in the prediction matrix.
    ``inf`` is returned when the small-sample correction is undefined.
    """

    pred = np.asarray(prediction, dtype=float)
    true = np.asarray(truth, dtype=float)
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/truth shapes differ: {pred.shape} != {true.shape}")
    n = int(true.size)
    if n <= k + 1:
        return float("inf")
    sse = max(float(np.sum((pred - true) ** 2)), np.finfo(float).tiny)
    return float(n * np.log(sse / n) + 2 * k + (2 * k * (k + 1)) / (n - k - 1))


class Objective(Protocol):
    def __call__(self, prediction: pd.DataFrame, truth: pd.DataFrame, k: int) -> float: ...


@dataclass(frozen=True)
class AICcObjective:
    """Default candidate objective used in the manuscript."""

    def __call__(self, prediction: pd.DataFrame, truth: pd.DataFrame, k: int) -> float:
        return aicc(prediction, truth, k)


@dataclass(frozen=True)
class MetricThreshold:
    """An output acceptance rule.

    Supported metrics are ``rmse``, ``mae``, ``max_abs`` and ``relative_rmse``.
    """

    maximum: float
    metric: str = "rmse"

    def evaluate(self, prediction: Array, truth: Array) -> tuple[bool, float]:
        error = np.asarray(prediction, dtype=float) - np.asarray(truth, dtype=float)
        if self.metric == "rmse":
            value = float(np.sqrt(np.mean(error**2)))
        elif self.metric == "mae":
            value = float(np.mean(np.abs(error)))
        elif self.metric == "max_abs":
            value = float(np.max(np.abs(error)))
        elif self.metric == "relative_rmse":
            scale = max(float(np.ptp(truth)), float(np.std(truth)), 1.0e-12)
            value = float(np.sqrt(np.mean(error**2)) / scale)
        else:
            raise ValueError(f"Unsupported threshold metric: {self.metric}")
        return value <= self.maximum, value


def normalize_thresholds(
    thresholds: Mapping[str, float | MetricThreshold], output_names: Sequence[str]
) -> dict[str, MetricThreshold]:
    missing = [name for name in output_names if name not in thresholds]
    if missing:
        raise ValueError(f"Missing thresholds for outputs: {missing}")
    return {
        name: value if isinstance(value, MetricThreshold) else MetricThreshold(float(value))
        for name, value in thresholds.items()
    }

