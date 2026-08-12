# Reproducibility

## Separation of discovery and testing

E1 and E2 use 500 samples for discovery/training and 250 subsequent samples for held-out evaluation. Held-out samples are not used for FP screening, input/output selection, architecture selection, parameter fitting, AICc comparison, or threshold acceptance.

Run the portable artifacts with:

```bash
adopt-fpml reproduce e1
adopt-fpml reproduce e1 --split test
adopt-fpml reproduce e2
adopt-fpml reproduce e2 --split test
```

The E1 test trajectory is part of `e1_continuous`; see `src/adopt_fpml/data/e1/heldout_rmse_reference.csv` for the retained evaluation summary.

## Reference accuracy

The portable artifacts reproduce these manuscript-rounded RMSE values:

| E1 output | Threshold | FP train | Hybrid train | FP test | Hybrid test |
|---|---:|---:|---:|---:|---:|
| `CA` | 5.0 | 33.6 | 2.9 | 38.5 | 7.3 |
| `CB` | 2.0 | 24.3 | 1.5 | 29.9 | 2.8 |
| `CC` | 3.0 | 23.6 | 1.4 | 29.1 | 3.0 |
| `T_out` | 10.0 | 149.8 | 6.4 | 174.6 | 19.0 |

| E2 output | Threshold | FP train | Hybrid train | FP test | Hybrid test |
|---|---:|---:|---:|---:|---:|
| `F_vap` | 0.005 | 0.012 | 0.0043 | 0.013 | 0.0026 |
| `F_liq` | 0.010 | 0.012 | 0.00051 | 0.013 | 0.00079 |
| `T_vap` | 20.0 | 63.7 | 7.1 | 29.4 | 6.2 |
| `T_liq` | 20.0 | 63.7 | 8.2 | 29.4 | 7.2 |
| `y_A` | 0.010 | 0.025 | 0.0012 | 0.0062 | 0.0010 |
| `y_B` | 0.050 | 0.14 | 0.021 | 0.15 | 0.0074 |
| `y_C` | 0.010 | 0.14 | 0.0054 | 0.14 | 0.0021 |
| `x_A` | 0.050 | 0.086 | 0.016 | 0.083 | 0.013 |
| `x_B` | 0.010 | 0.058 | 0.0084 | 0.068 | 0.0031 |
| `x_C` | 0.050 | 0.035 | FP retained | 0.020 | FP retained |

The bundled runners calculate full-precision values; tables above are rounded to the precision shown in the manuscript.

## Randomness and numerical variation

- E1 input signals use seed base 123 and measurement noise seed 2026.
- E2 measurement noise uses seed 42.
- The default MLP uses seed 7.
- FP integration uses deterministic RK4; E2 temperature/root calculations use deterministic bisection.

Data generators are verified against the retained CSVs. E2 root calculations may differ at approximately `1e-8` in temperature and `1e-5` in derived duty across equivalent implementations. Retraining may vary slightly with Python/BLAS/platform versions, so reference claims should use NPZ/JSON pretrained artifacts and newly discovered structures should report their environment and seeds.

## Portable artifacts

Model parameters use compressed NumPy NPZ files with JSON metadata. Loading sets `allow_pickle=False`; no object deserialization or code execution occurs. The old working archives’ pickle models are intentionally not distributed.

For a new experiment, retain:

- immutable input and output data or a data-generation version/seed;
- exact train/validation/test boundaries;
- package and dependency versions;
- FP model constants and solver tolerances;
- thresholds and their engineering rationale;
- ML/selector/pruner/search configurations;
- complete search history and stage summaries;
- portable fitted artifacts and a prediction verification test.
