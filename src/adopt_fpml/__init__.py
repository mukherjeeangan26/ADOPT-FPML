"""ADOPT-FPML public API."""

from .fp import CallableFirstPrinciplesModel, FirstPrinciplesModel
from .metrics import AICcObjective, MetricThreshold, aicc, rmse_by_output
from .ml import MLTrainingConfig, NumpyMLPTrainer
from .optimizer import ADOPTFPML, DiscoveryResult, SearchConfig, Structure
from .pruning import PruningConfig
from .selectors import PearsonSelector, SpearmanSelector

__all__ = [
    "ADOPTFPML",
    "AICcObjective",
    "CallableFirstPrinciplesModel",
    "DiscoveryResult",
    "FirstPrinciplesModel",
    "MLTrainingConfig",
    "MetricThreshold",
    "NumpyMLPTrainer",
    "PearsonSelector",
    "PruningConfig",
    "SearchConfig",
    "SpearmanSelector",
    "Structure",
    "aicc",
    "rmse_by_output",
]

__version__ = "0.1.0"

