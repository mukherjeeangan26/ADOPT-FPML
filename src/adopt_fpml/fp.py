"""Interfaces for user-supplied first-principles models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, runtime_checkable

import pandas as pd


@runtime_checkable
class FirstPrinciplesModel(Protocol):
    """Protocol implemented by any FP model used by :class:`ADOPTFPML`.

    ``parameters`` is a row-aligned DataFrame for Series II/integrated parameter
    corrections. A model without correctable parameters can ignore it.
    """

    output_names: Sequence[str]

    def predict(
        self, inputs: pd.DataFrame, parameters: pd.DataFrame | None = None
    ) -> pd.DataFrame: ...


@dataclass
class CallableFirstPrinciplesModel:
    """Wrap a plain ``function(inputs, parameters) -> DataFrame`` as an FP model."""

    function: Callable[[pd.DataFrame, pd.DataFrame | None], pd.DataFrame]
    output_names: Sequence[str]

    def predict(
        self, inputs: pd.DataFrame, parameters: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        result = self.function(inputs.copy(), parameters)
        if not isinstance(result, pd.DataFrame):
            result = pd.DataFrame(result, columns=list(self.output_names), index=inputs.index)
        missing = [name for name in self.output_names if name not in result.columns]
        if missing:
            raise ValueError(f"FP model did not return required outputs: {missing}")
        return result.loc[:, list(self.output_names)].copy()

