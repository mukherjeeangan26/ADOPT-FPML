# ADOPT-FPML

[![CI](https://github.com/mukherjeeangan26/ADOPT-FPML/actions/workflows/ci.yml/badge.svg)](https://github.com/mukherjeeangan26/ADOPT-FPML/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/adopt-fpml.svg)](https://pypi.org/project/adopt-fpml/)
[![Python](https://img.shields.io/pypi/pyversions/adopt-fpml.svg)](https://pypi.org/project/adopt-fpml/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

**A**utomatic **D**iscovery of **OPT**imal hybrid **F**irst **P**rinciples–**M**achine **L**earning models.

ADOPT-FPML discovers a parsimonious information-flow architecture for a system that has a first-principles (FP) model and measured data. It decides which outputs already have adequate FP predictions, where ML blocks are needed, which of four FP–ML interactions to use, what the ML inputs and outputs should be, and when to stop adding complexity.

This repository contains the proposed discovery algorithm, an installable Python package, two complete synthetic case studies, data-generation code, portable pretrained model artifacts, tests, and documentation. It intentionally does **not** copy full baseline-training programs that are already available in the cited literature.

> **Publication status:** the accompanying manuscript, *On the Automatic Discovery of Optimal Hybrid First Principles – Machine Learning Models for Interconnected Systems*, by Angan Mukherjee, Nishant V. Giridhar, and Debangsu Bhattacharyya, is currently a manuscript in preparation. Replace this note and the provisional citation when a DOI and final bibliographic details become available.

## What ADOPT-FPML searches

At every stage, ADOPT-FPML evaluates four reduced-superstructure configurations:

| Configuration | Package name | Information flow | ML learns |
|---|---|---|---|
| Parallel | `parallel` | FP and ML operate in parallel | FP-output residuals |
| Series I | `series_1` | FP → ML | requested system outputs, using at least one FP output as an ML input |
| Series II | `series_2` | ML → FP | row-wise corrections to FP parameters |
| Integrated | `integrated` | ML ↔ FP | FP parameter corrections using measured/synthetic inputs |

The default algorithm follows the manuscript:

1. Run the standalone FP model and accept outputs satisfying their output-specific thresholds.
2. Rank candidate inputs by mean absolute Pearson correlation with unresolved outputs.
3. For each interaction configuration, add ranked inputs sequentially and retrain the ML block.
4. Score candidates with corrected Akaike information criterion (AICc) and greedily prune a branch at its first non-improvement.
5. Select the admissible candidate with the lowest **local** AICc within the stage.
6. Retain it only if it resolves at least one new output and its relative global-AICc increase is below `tau_AIC`.
7. Screen predicted synthetic variables for relevance, redundancy, and a cardinality cap; propagate retained variables to the next stage.
8. Stop when all outputs meet their thresholds, no admissible improvement remains, the global-AICc rule rejects a stage, or the stage limit is reached.

See [Algorithm and terminology](docs/algorithm.md) for the route variables and exact decision rules.

## Installation

ADOPT-FPML requires Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Or Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the released package:

```bash
python -m pip install --upgrade pip
python -m pip install adopt-fpml
adopt-fpml doctor
```

Install directly from a local clone for development:

```bash
git clone https://github.com/mukherjeeangan26/ADOPT-FPML.git
cd ADOPT-FPML
python -m pip install -e ".[dev]"
pytest
```

Use `python -m pip install "adopt-fpml[sklearn]"` for the optional scikit-learn estimator adapter and mutual-information selector. NumPy, pandas, and SciPy are the only required runtime dependencies.

## Quick start on a new system

Prepare two row-aligned DataFrames:

- `inputs`: measured/manipulated variables available to FP and ML models.
- `measured_outputs`: canonical target outputs, in the same row order.

Then wrap a Python function that returns FP predictions:

```python
import pandas as pd

from adopt_fpml import ADOPTFPML, CallableFirstPrinciplesModel, SearchConfig


def my_fp_model(
    inputs: pd.DataFrame,
    parameters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    gain = 2.0 if parameters is None else parameters["gain"].to_numpy()
    return pd.DataFrame(
        {
            "product": gain * inputs["feed"],
            "temperature": inputs["inlet_temperature"] + 10.0,
        },
        index=inputs.index,
    )


fp = CallableFirstPrinciplesModel(
    my_fp_model,
    output_names=["product", "temperature"],
)

result = ADOPTFPML(fp).fit(
    inputs=inputs,
    measured_outputs=measured_outputs,
    thresholds={"product": 0.05, "temperature": 2.0},  # absolute RMSE defaults
    config=SearchConfig(max_stages=3),
)

print(result.summary())
result.search_history.to_csv("search_history.csv", index=False)
```

Series II and integrated configurations need a `ParameterTargetGenerator`; without one, the optimizer records those routes as skipped and still searches parallel/Series I. The E1/E2 examples include bounded SciPy estimators, while a new system normally supplies its own parameter-estimation or inverse-model routine. See [Defining your system](docs/defining-a-system.md).

## Required choices and defaults

| Component | You provide | Default/fallback | How to replace it |
|---|---|---|---|
| FP model | `predict(inputs, parameters=None)` and output names | No universal physical model | Implement the protocol or wrap a function |
| Data | numeric, finite, row-aligned input/output DataFrames | Case-study data are included only for examples | Use your own DataFrames |
| Accuracy rules | one rule for every output | float means absolute RMSE | Use `MetricThreshold` for MAE, max error, or relative RMSE |
| ML class/solver | optional trainer | one-hidden-layer tanh NumPy MLP with Adam and early stopping | Implement `MLTrainer`, or use `SklearnTrainer` |
| Input selection | optional selector | mean absolute Pearson correlation | Spearman, mutual information, or a custom selector |
| Candidate objective | optional callable | AICc | Inject another complexity-aware objective |
| Branch pruning | search configuration | greedy first non-improvement | patience or moving-window policies |
| Stage termination | search configuration | all outputs resolved, no candidate, AICc rule, or stage cap | Change `SearchConfig` |

The supplied MLP matches the paper’s model class. It is a reproducible research default, not a claim that one architecture is universally best. If a replacement model is used with AICc, its fitted wrapper must report a meaningful `parameter_count`.

## Reproduce the case studies

Run the portable pretrained architectures without retraining:

```bash
adopt-fpml reproduce e1
adopt-fpml reproduce e1 --split test
adopt-fpml reproduce e2
adopt-fpml reproduce e2 --split test
```

Run the scripts from a clone:

```bash
python examples/e1/run_pretrained.py
python examples/e2/run_pretrained.py --split test
```

Generate E1 data from scratch:

```bash
adopt-fpml generate e1 --rows 500 --output e1_training.csv
```

Generate E2 from an E1 feed trajectory:

```bash
adopt-fpml generate e2 --feed e1_training.csv --output e2_training.csv
```

The longer discovery scripts demonstrate how to rerun structural search. ML optimization can produce small platform-level numerical variations; the portable pretrained runners are the reference path for matching reported predictions. Full details are in [E1](examples/e1/README.md), [E2](examples/e2/README.md), and [Reproducibility](docs/reproducibility.md).

### Reported retained structures

| Case | Retained stage 1 | Retained stage 2 | Standalone FP output |
|---|---|---|---|
| E1, three CSTRs | Series I: `fp_T_out`, `T_in`, `CA0`, `fp_CB`, `fp_CC`; resolves `CB`, `CC` | Parallel: `fp_T_out`, stage-1 `T_out`, stage-1 `CA`, `T_in`; resolves `CA`, `T_out` | None |
| E2, cooler/flash | Series I: `fp_x_A`, `fp_x_C`; resolves seven outputs | Integrated: `fp_F_liq`, `T_feed`; corrects `UA`, `K_A`, `K_B`, `K_C`; resolves `F_liq`, `y_A` | `x_C` |

The manuscript’s third, industrial superheater case is described in the publication but its proprietary plant data and implementation were not supplied for public distribution and are not included here.

## Repository map

```text
.
├── src/adopt_fpml/        installable algorithm, interfaces, CLI, cases, data
├── examples/              E1, E2, and a minimal custom-system example
├── docs/                  concepts, extension guides, reproducibility, release guide
├── tests/                 unit, generator, optimizer, and reference-RMSE checks
├── .github/workflows/     CI and trusted PyPI publishing
├── pyproject.toml         package metadata and dependencies
├── CITATION.cff           citation metadata
└── LICENSE                BSD 3-Clause license
```

See [Source archive audit](docs/source-archive-audit.md) for what was retained, reconstructed, consolidated, or excluded from the original working archives.

## Publication, citation, and attribution

Until the article is published, cite the software and provisional manuscript:

```bibtex
@software{mukherjee_adopt_fpml_2026,
  author = {Mukherjee, Angan and Giridhar, Nishant V. and Bhattacharyya, Debangsu},
  title = {ADOPT-FPML: Automatic Discovery of Optimal Hybrid First Principles--Machine Learning Models},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/mukherjeeangan26/ADOPT-FPML}
}
```

Update `CITATION.cff`, this block, and `pyproject.toml` when the journal citation and DOI are assigned. GitHub displays `CITATION.cff` through its “Cite this repository” control.

## Contributing and support

Bug reports, validation cases, new FP-model adapters, selectors, and carefully tested pruning extensions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Use GitHub Issues for reproducible software problems; do not include proprietary plant data or credentials.

## License and disclaimer

The code is provided under the [BSD 3-Clause License](LICENSE). Confirm that this license and all author/affiliation details are acceptable before the first public release. The software is research code supplied without warranty; users are responsible for engineering validation, safety constraints, identifiability checks, and any deployment decision.

First-time maintainers can follow the complete [GitHub and PyPI publishing guide](docs/publishing.md).
