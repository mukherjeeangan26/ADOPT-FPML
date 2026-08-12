"""Reproduce the retained E2 architecture without retraining."""

import argparse

import pandas as pd

from adopt_fpml.case_studies.e2 import MEASURED_COLUMNS, OUTPUTS, run_e2_pretrained
from adopt_fpml.datasets import load_dataset
from adopt_fpml.metrics import rmse_by_output


parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["training", "test"], default="training")
args = parser.parse_args()
data = load_dataset(f"e2_{args.split}")
truth = data[MEASURED_COLUMNS].copy()
truth.columns = OUTPUTS
prediction = run_e2_pretrained(data)
print(pd.Series(rmse_by_output(prediction, truth), name=f"{args.split}_RMSE"))

