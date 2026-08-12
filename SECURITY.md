# Security policy

## Supported versions

Security fixes are provided for the latest released version.

## Reporting

Do not disclose a vulnerability in a public issue. Use GitHub’s private security-advisory feature for this repository. If it is not enabled, contact the repository owner privately.

ADOPT-FPML loads bundled models only from NPZ arrays with `allow_pickle=False` and JSON metadata. Do not replace this with untrusted pickle/joblib loading. Treat user FP models, custom trainers, and data-generation scripts as executable code and review their provenance before use.

