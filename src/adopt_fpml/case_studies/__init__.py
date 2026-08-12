"""Reproducible synthetic case studies from the manuscript."""

from .e1 import (
    E1FirstPrinciplesModel,
    E1ParameterTargetGenerator,
    generate_e1_data,
    run_e1_pretrained,
)
from .e2 import (
    E2FirstPrinciplesModel,
    E2ParameterTargetGenerator,
    generate_e2_data,
    run_e2_pretrained,
)

__all__ = [
    "E1FirstPrinciplesModel",
    "E1ParameterTargetGenerator",
    "E2FirstPrinciplesModel",
    "E2ParameterTargetGenerator",
    "generate_e1_data",
    "generate_e2_data",
    "run_e1_pretrained",
    "run_e2_pretrained",
]
