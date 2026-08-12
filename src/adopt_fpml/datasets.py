"""Access the small, versioned case-study datasets shipped with the package."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pandas as pd


def data_path(case_study: str, filename: str) -> Path:
    if case_study not in {"e1", "e2"}:
        raise ValueError("case_study must be 'e1' or 'e2'")
    return Path(str(files("adopt_fpml").joinpath("data", case_study, filename)))


def load_dataset(name: str) -> pd.DataFrame:
    """Load ``e1_training``, ``e1_continuous``, ``e2_training`` or ``e2_test``."""

    locations = {
        "e1_training": ("e1", "training_500.csv"),
        "e1_continuous": ("e1", "continuous_750.csv"),
        "e2_training": ("e2", "training_500.csv"),
        "e2_test": ("e2", "alternative_test_250.csv"),
    }
    try:
        case, filename = locations[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(locations)}") from exc
    return pd.read_csv(data_path(case, filename))

