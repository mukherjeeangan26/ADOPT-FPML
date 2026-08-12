"""Pluggable candidate-input ranking strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class InputSelector(Protocol):
    def rank(self, candidates: pd.DataFrame, targets: pd.DataFrame) -> list[str]: ...


def _safe_abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return 0.0
    return float(abs(np.corrcoef(x, y)[0, 1]))


@dataclass(frozen=True)
class PearsonSelector:
    """Paper default: descending mean absolute Pearson correlation."""

    def rank(self, candidates: pd.DataFrame, targets: pd.DataFrame) -> list[str]:
        scores = {
            name: float(
                np.mean(
                    [_safe_abs_corr(candidates[name].to_numpy(), targets[y].to_numpy()) for y in targets]
                )
            )
            for name in candidates
        }
        return sorted(candidates.columns, key=lambda name: (-scores[name], name))


@dataclass(frozen=True)
class SpearmanSelector:
    """Rank nonlinear monotonic associations without an optional dependency."""

    def rank(self, candidates: pd.DataFrame, targets: pd.DataFrame) -> list[str]:
        return PearsonSelector().rank(candidates.rank(method="average"), targets.rank(method="average"))


@dataclass(frozen=True)
class MutualInformationSelector:
    """Mutual-information ranking; requires ``adopt-fpml[sklearn]``."""

    random_state: int = 0

    def rank(self, candidates: pd.DataFrame, targets: pd.DataFrame) -> list[str]:
        try:
            from sklearn.feature_selection import mutual_info_regression
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install adopt-fpml[sklearn] for mutual-information ranking") from exc
        X = candidates.to_numpy(dtype=float)
        scores = np.zeros(X.shape[1], dtype=float)
        for name in targets:
            scores += mutual_info_regression(X, targets[name], random_state=self.random_state)
        scores /= max(len(targets.columns), 1)
        return [
            candidates.columns[i]
            for i in sorted(range(len(scores)), key=lambda i: (-scores[i], candidates.columns[i]))
        ]

