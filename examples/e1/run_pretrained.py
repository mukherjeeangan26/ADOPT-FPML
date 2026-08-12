"""Reproduce the retained E1 architecture without retraining."""

import pandas as pd

from adopt_fpml.case_studies.e1 import MEASURED_COLUMNS, OUTPUTS, run_e1_pretrained
from adopt_fpml.datasets import load_dataset
from adopt_fpml.metrics import rmse_by_output


data = load_dataset("e1_training")
truth = data[MEASURED_COLUMNS].copy()
truth.columns = OUTPUTS
prediction = run_e1_pretrained(data)
print(pd.Series(rmse_by_output(prediction, truth), name="training_RMSE"))

