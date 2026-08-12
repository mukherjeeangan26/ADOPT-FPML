"""Generate E1 data from Appendix A."""

import argparse

from adopt_fpml.case_studies.e1 import generate_e1_data


parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, default=500)
parser.add_argument("--output", default="e1_training.csv")
args = parser.parse_args()
data = generate_e1_data(args.rows)
data.to_csv(args.output, index=False)
print(f"Wrote {data.shape} to {args.output}")

