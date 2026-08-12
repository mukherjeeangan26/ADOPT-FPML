"""Small command-line interface for data generation and reproducibility checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import __version__
from .case_studies.e1 import MEASURED_COLUMNS as E1_MEASURED
from .case_studies.e1 import OUTPUTS as E1_OUTPUTS
from .case_studies.e1 import generate_e1_data, run_e1_pretrained
from .case_studies.e2 import MEASURED_COLUMNS as E2_MEASURED
from .case_studies.e2 import OUTPUTS as E2_OUTPUTS
from .case_studies.e2 import generate_e2_data, run_e2_pretrained
from .datasets import load_dataset
from .metrics import rmse_by_output


def _write_csv(frame: pd.DataFrame, output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Wrote {len(frame)} rows to {path}")


def _reproduce(case: str, split: str) -> None:
    dataset = "e1_continuous" if case == "e1" and split == "test" else f"{case}_{split}"
    data = load_dataset(dataset)
    if case == "e1":
        prediction, measured, names = run_e1_pretrained(data), E1_MEASURED, E1_OUTPUTS
    else:
        prediction, measured, names = run_e2_pretrained(data), E2_MEASURED, E2_OUTPUTS
    if case == "e1" and split == "test":
        data = data.iloc[500:]
        prediction = prediction.iloc[500:]
    truth = data.loc[:, measured].copy()
    truth.columns = names
    print(pd.Series(rmse_by_output(prediction, truth), name="RMSE").to_string())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adopt-fpml", description="ADOPT-FPML utilities")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="generate synthetic case-study data")
    generate.add_argument("case", choices=["e1", "e2"])
    generate.add_argument("--rows", type=int, default=500, help="E1 rows; ignored for E2")
    generate.add_argument("--feed", help="E1 feed CSV required for E2 generation")
    generate.add_argument("--output", required=True)
    reproduce = subparsers.add_parser("reproduce", help="run bundled pretrained configuration")
    reproduce.add_argument("case", choices=["e1", "e2"])
    reproduce.add_argument("--split", choices=["training", "test"], default="training")
    subparsers.add_parser("doctor", help="check imports and bundled data")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        if args.case == "e1":
            frame = generate_e1_data(args.rows)
        else:
            if not args.feed:
                raise SystemExit("--feed is required for E2 generation")
            frame = generate_e2_data(pd.read_csv(args.feed))
        _write_csv(frame, args.output)
    elif args.command == "reproduce":
        _reproduce(args.case, args.split)
    elif args.command == "doctor":
        for name in ("e1_training", "e1_continuous", "e2_training", "e2_test"):
            frame = load_dataset(name)
            print(f"ok  {name}: {frame.shape[0]} rows x {frame.shape[1]} columns")
        print(f"ok  adopt-fpml {__version__}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
