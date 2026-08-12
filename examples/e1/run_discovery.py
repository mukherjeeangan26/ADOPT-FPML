"""Rerun the E1 branch-and-prune search using the generic public API."""

from adopt_fpml import ADOPTFPML, MLTrainingConfig, NumpyMLPTrainer, SearchConfig
from adopt_fpml.case_studies.e1 import (
    E1FirstPrinciplesModel,
    E1ParameterTargetGenerator,
    INPUTS,
    MEASURED_COLUMNS,
    OUTPUTS,
)
from adopt_fpml.datasets import load_dataset
from adopt_fpml.optimizer import Structure


data = load_dataset("e1_training")
truth = data[MEASURED_COLUMNS].copy()
truth.columns = OUTPUTS
fp_model = E1FirstPrinciplesModel()
result = ADOPTFPML(
    fp_model,
    ml_trainer=NumpyMLPTrainer(MLTrainingConfig(seed=7)),
    parameter_target_generator=E1ParameterTargetGenerator(fp_model),
).fit(
    data[INPUTS],
    truth,
    thresholds={"CA": 5, "CB": 2, "CC": 3, "T_out": 10},
    config=SearchConfig(
        max_stages=4,
        max_global_aicc_relative_increase=0.0,
        structures=tuple(Structure),
    ),
)
print(result.summary())
result.search_history.to_csv("e1_search_history.csv", index=False)
