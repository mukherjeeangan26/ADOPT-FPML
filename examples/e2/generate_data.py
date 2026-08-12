"""Generate E2 data from the bundled E1 training feed."""

import argparse

from adopt_fpml.case_studies.e2 import generate_e2_data
from adopt_fpml.datasets import load_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--output", default="e2_training.csv")
args = parser.parse_args()
data = generate_e2_data(load_dataset("e1_training"))
data.to_csv(args.output, index=False)
print(f"Wrote {data.shape} to {args.output}")

