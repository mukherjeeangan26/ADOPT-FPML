"""Stopping rules for sequential candidate-input addition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


@dataclass(frozen=True)
class PruningConfig:
    """Configure branch pruning.

    ``greedy`` reproduces the paper. ``patience`` tolerates consecutive misses;
    ``moving_window`` compares the best values in two adjacent windows.
    """

    strategy: Literal["greedy", "patience", "moving_window"] = "greedy"
    min_improvement: float = 0.0
    patience: int = 2
    window: int = 3

    def should_stop(self, objective_values: Sequence[float]) -> bool:
        values = list(objective_values)
        if len(values) < 2:
            return False
        if self.strategy == "greedy":
            return values[-2] - values[-1] <= self.min_improvement
        if self.strategy == "patience":
            misses, best = 0, values[0]
            for value in values[1:]:
                if best - value > self.min_improvement:
                    best, misses = value, 0
                else:
                    misses += 1
            return misses > self.patience
        if self.strategy == "moving_window":
            if self.window < 1:
                raise ValueError("window must be at least one")
            w = self.window
            return len(values) >= 2 * w and (
                min(values[-2 * w : -w]) - min(values[-w:]) <= self.min_improvement
            )
        raise ValueError(f"Unknown pruning strategy: {self.strategy}")

