# Source archive audit

This public repository was assembled from the manuscript draft and two working archives. The audit’s purpose was to retain everything needed to understand and run ADOPT-FPML while excluding redundant, unsafe, machine-specific, or publication-output files.

## Inspected sources

| Source | Scope inspected | Main result |
|---|---|---|
| Manuscript draft | 57 pages, including algorithm, cases, Appendices A/B, tables, and code-availability note | Algorithm rules, constants, thresholds, retained configurations, and reported metrics reconciled with code/data |
| E1 working archive | 51 entries: Python, CSV, XLSX, JSON, NPZ, PKL, notebooks, and logs | Generator, discovery implementation, continuous/train/test data, portable artifacts, and reference metrics identified |
| E2 working archive | 86 entries: Python, CSV, XLSX, JSON/NDJSON, PNG, PKL, NPZ, and logs | Discovery implementation, training/test data, portable artifacts, and result summaries identified; missing generator reference found |

All 45 CSVs were schema-audited. Seven unique workbooks (35 sheets) were opened and rendered for visual/formula review. NumPy archives were loaded with `allow_pickle=False`; pickle files were structure-audited without executing their contents.

## Retained and consolidated

- A generic algorithm implementation aligned to the manuscript’s local-AICc stage selection.
- E1 and E2 FP models and complete synthetic data generators.
- Authoritative 500-row training data and held-out/continuous test data needed for reproduction.
- Only the retained E1/E2 model blocks, represented as portable NPZ parameters plus JSON metadata.
- Compact reference RMSE/stage summary CSVs.
- Examples and tests that reproduce the published rounded metrics.

Identical copies from multiple archive folders were consolidated into one canonical file. Original source scripts were refactored into package modules with user-facing protocols rather than copied as large monolithic case-specific programs.

## Reconstructed E2 generator

The E2 holdout material imports `E2_final_data_generation` and `run_final_e2_holdout`, but those modules were absent from the supplied archive. The generator in this repository was reconstructed from Appendix B and validated against the authoritative E2 CSV.

The recovered implementation uses the documented cooler/flash equations and constants, noise seed 42, and verified draw order: vapor flow, liquid flow, liquid composition, vapor composition, vapor temperature, then liquid temperature. Generated numeric values match the archive within deterministic root-solver tolerance.

## Excluded

- duplicate scripts/data/workbooks and nested duplicate ZIP files;
- Python pickles, because they are nonportable and can execute arbitrary code when loaded;
- notebook checkpoint copies;
- generated figures and workbook previews that are outputs rather than inputs;
- `.inspect.ndjson`, temporary inspection material, and large machine-generated logs;
- files containing absolute local paths or workstation-specific metadata;
- full baseline model-training code available in the cited literature;
- proprietary industrial superheater data/code, which were not in the supplied public materials.

The original archives remain the authors’ private provenance record. This repository is the minimal transparent, runnable public distribution.

