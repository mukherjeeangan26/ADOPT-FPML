"""Portable, non-pickle serialization for the bundled NumPy MLP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ml import NumpyMLP


ARRAY_NAMES = ("W1", "b1", "W2", "b2", "x_mean", "x_std", "y_mean", "y_std")


def save_numpy_mlp(model: NumpyMLP, parameter_path: str | Path, metadata_path: str | Path) -> None:
    parameter_path, metadata_path = Path(parameter_path), Path(metadata_path)
    np.savez_compressed(parameter_path, **{name: getattr(model, name) for name in ARRAY_NAMES})
    metadata = {
        "format": "adopt-fpml.numpy-mlp.v1",
        "input_names": list(model.input_names),
        "output_names": list(model.output_names),
        "parameter_count": model.parameter_count,
        "train_loss": model.train_loss,
        "validation_loss": model.val_loss,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_numpy_mlp(parameter_path: str | Path, metadata_path: str | Path) -> NumpyMLP:
    """Load an NPZ+JSON model without executing arbitrary code."""

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    with np.load(parameter_path, allow_pickle=False) as archive:
        missing = [name for name in ARRAY_NAMES if name not in archive]
        if missing:
            raise ValueError(f"Portable model archive is missing arrays: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in ARRAY_NAMES}
    output_names = metadata.get("ml_output_names", metadata.get("output_names", metadata.get("target_outputs")))
    if output_names is None:
        raise ValueError("Model metadata does not define output names")
    return NumpyMLP(
        input_names=list(metadata["input_names"]),
        output_names=list(output_names),
        **arrays,
        train_loss=float(metadata.get("train_loss", float("nan"))),
        val_loss=float(metadata.get("validation_loss", float("nan"))),
    )
