"""Run E2 routes that do not require parameter-target estimation."""

from adopt_fpml import ADOPTFPML, SearchConfig
from adopt_fpml.case_studies.e2 import (
    E2FirstPrinciplesModel,
    E2ParameterTargetGenerator,
    MEASURED_COLUMNS,
    OUTPUTS,
)
from adopt_fpml.datasets import load_dataset
from adopt_fpml.optimizer import Structure


data = load_dataset("e2_training")
inputs = data[["CA_feed_measured", "CB_feed_measured", "CC_feed_measured", "T_feed_measured", "v"]].copy()
inputs.columns = ["CA_feed", "CB_feed", "CC_feed", "T_feed", "v"]
truth = data[MEASURED_COLUMNS].copy()
truth.columns = OUTPUTS
thresholds = dict(zip(OUTPUTS, [0.005, 0.01, 20, 20, 0.01, 0.05, 0.01, 0.05, 0.01, 0.05]))
fp_model = E2FirstPrinciplesModel()
result = ADOPTFPML(
    fp_model,
    parameter_target_generator=E2ParameterTargetGenerator(fp_model),
).fit(
    inputs,
    truth,
    thresholds,
    SearchConfig(
        max_stages=2,
        max_global_aicc_relative_increase=0.05,
        structures=tuple(Structure),
    ),
)
print(result.summary())
result.search_history.to_csv("e2_search_history.csv", index=False)
