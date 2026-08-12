# Defining your system

## 1. Prepare data

Inputs and measured outputs must be finite numeric pandas DataFrames with identical indexes. Every output expected from the FP model must appear in `measured_outputs`. Preprocess units and missing values before calling `fit`; the optimizer does not silently impute or resample process data.

For dynamic systems, each FP `predict` call should simulate rows in chronological order and return one prediction per row. Split training/test trajectories before discovery. Never use held-out data for threshold screening, structure selection, input selection, or fitting.

## 2. Define the FP model

Implement:

```python
class MyFPModel:
    output_names = ["outlet", "temperature"]

    def predict(self, inputs, parameters=None):
        # parameters is None for the standalone model.
        # It is row-aligned for Series II/integrated candidates.
        return predictions_dataframe
```

You may instead use `CallableFirstPrinciplesModel`. The function can call a numerical solver, a compiled simulator, or another Python package, provided it is deterministic enough for repeated candidate evaluation and returns canonical output names.

## 3. Define accuracy rules

A numeric value means maximum absolute RMSE:

```python
thresholds = {"outlet": 0.01, "temperature": 5.0}
```

Use explicit rules for other metrics:

```python
from adopt_fpml import MetricThreshold

thresholds = {
    "outlet": MetricThreshold(0.02, metric="mae"),
    "temperature": MetricThreshold(0.05, metric="relative_rmse"),
}
```

Supported metrics are `rmse`, `mae`, `max_abs`, and `relative_rmse`. Thresholds are application decisions, not tuning conveniences: derive them from measurement uncertainty, engineering tolerances, or a validation protocol and document the rationale.

## 4. Enable parameter-correction routes

Series II and integrated blocks train ML against row-wise FP parameter targets. Implement a generator with a `parameter_names` sequence and a `generate(inputs, measured_outputs, fp_predictions, target_outputs)` method. Typical implementations solve bounded parameter estimation at each row or time window.

Guard against non-identifiability and physically impossible parameters. Expensive targets are generated once per stage and reused by every applicable candidate. If the generator is omitted, Series II and integrated routes are transparently skipped.

## 5. Configure and run

```python
from adopt_fpml import ADOPTFPML, SearchConfig
from adopt_fpml.optimizer import Structure

search = SearchConfig(
    max_stages=4,
    max_global_aicc_relative_increase=0.05,
    max_inputs_per_branch=8,
    structures=(
        Structure.PARALLEL,
        Structure.SERIES_1,
        Structure.SERIES_2,
        Structure.INTEGRATED,
    ),
)

result = ADOPTFPML(
    fp_model=my_fp,
    parameter_target_generator=my_parameter_estimator,
).fit(inputs, measured_outputs, thresholds, search)
```

Keep `result.search_history`, `result.stage_summaries`, package version, random seeds, configuration, dependency lock information, and the train/test split with every reported result.

