# Contributing

Thank you for helping improve ADOPT-FPML. Open an issue before a large change so its scientific scope and validation plan can be agreed first.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
ruff check .
pytest
```

## Pull requests

- Keep public API changes focused and documented.
- Add tests for every bug fix or new feature.
- Preserve deterministic seeds and never use held-out data for discovery.
- Report `parameter_count` correctly for any ML adapter used with AICc.
- Describe deviations from the manuscript default as extensions.
- Do not commit proprietary data, credentials, pickles, generated result folders, or machine-specific paths.
- Run linting, tests, package build, Twine checks, and reference reproduction before requesting review.

By contributing, you agree that your contribution is distributed under the repository’s BSD 3-Clause license.

