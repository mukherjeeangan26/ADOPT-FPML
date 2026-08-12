# First-time GitHub and PyPI publishing guide

This guide uses the repository’s existing CI and trusted-publishing workflow. Commands are shown for a local clone; GitHub Desktop can perform the same Git operations.

## Before anything becomes public

Review these items with the coauthors or institutional owner:

- repository name and GitHub owner (`mukherjeeangan26/ADOPT-FPML` is currently configured);
- BSD 3-Clause license choice;
- authorship order, affiliations, funding, citation text, and publication status;
- that no confidential/proprietary data, credentials, personal paths, or restricted code are present;
- package name availability on PyPI (`adopt-fpml` is only secured when the first project publication succeeds).

Run the release checks:

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python -m build
python -m twine check dist/*
```

Then create a separate clean virtual environment and install the wheel from `dist/`, run `adopt-fpml doctor`, and run all four reproduction commands.

## Create and upload the GitHub repository

### Recommended: Git command line

1. Sign in at GitHub and create a **new empty repository** named `ADOPT-FPML` under `mukherjeeangan26`. Do not ask GitHub to add a README, `.gitignore`, or license because those are already present.
2. Open a terminal inside this repository folder.
3. Run:

```bash
git init
git add .
git status
git commit -m "Initial public release of ADOPT-FPML"
git branch -M main
git remote add origin https://github.com/mukherjeeangan26/ADOPT-FPML.git
git push -u origin main
```

4. On GitHub, inspect the rendered README, repository files, Actions tab, citation display, and license detection.
5. In repository Settings → General, enable Issues and Discussions if desired. In Settings → Branches, add a `main` protection ruleset after the first push; require the CI checks before merging.

### Browser upload alternative

Create the same empty repository, choose **Add file → Upload files**, and upload the **contents** of this folder, not the surrounding ZIP or parent directory. Confirm hidden files and folders are present afterward: `.github`, `.gitignore`, and `.gitattributes`. Commit to `main`. Git or GitHub Desktop is less error-prone for later updates and releases.

## Make the first test publication

TestPyPI is separate from production PyPI and may not resolve all dependencies. Create a TestPyPI account, enable its required account security, build locally, and upload:

```bash
python -m build
python -m twine upload --repository testpypi dist/*
```

Test installation in a new environment. Install runtime dependencies from normal PyPI first, then install only this distribution from TestPyPI:

```bash
python -m pip install numpy pandas scipy
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps adopt-fpml
adopt-fpml doctor
```

TestPyPI is optional once GitHub trusted publishing is configured, but it is useful for checking the rendered project page, metadata, wheel contents, and installation flow.

## Configure trusted publishing on production PyPI

Trusted publishing lets GitHub Actions obtain a short-lived token through OpenID Connect, so no long-lived PyPI password or API token is stored in GitHub.

For a first publication, create a **pending publisher** on PyPI:

1. Sign in to PyPI.
2. Open account publishing settings and add a pending GitHub publisher.
3. Enter:
   - PyPI project name: `adopt-fpml`
   - GitHub owner: `mukherjeeangan26`
   - repository: `ADOPT-FPML`
   - workflow filename: `publish.yml`
   - environment: `release`
4. In GitHub Settings → Environments, create `release`. Add yourself as a required reviewer if you want a manual approval gate.

The included `.github/workflows/publish.yml` requests `id-token: write`, builds from the tagged commit, checks artifacts, and publishes through `pypa/gh-action-pypi-publish` only after a GitHub Release is published.

## Version and release workflow

PyPI files are immutable: a version cannot be overwritten. For every release:

1. Choose a version using semantic versioning. Start with `0.1.0`; use patch releases for compatible fixes, minor releases for new compatible features, and major releases for breaking API changes.
2. Change `version` in `pyproject.toml` and `__version__` in `src/adopt_fpml/__init__.py` to the same value.
3. Update `CHANGELOG.md` and any citation version.
4. Run all release checks and commit the changes.
5. Create an annotated tag matching the version:

```bash
git tag -a v0.1.0 -m "ADOPT-FPML 0.1.0"
git push origin main --tags
```

6. On GitHub, open Releases → Draft a new release, select `v0.1.0`, write release notes, and publish it.
7. Watch the `Publish to PyPI` workflow. Approve the `release` environment if prompted.
8. Verify `https://pypi.org/project/adopt-fpml/`, then install `adopt-fpml==0.1.0` in a brand-new environment and run `adopt-fpml doctor` plus the reproduction commands.

If the PyPI name is already taken, choose a different distribution name in `pyproject.toml` (for example an institution-qualified name). The Python import should remain `adopt_fpml`; distribution names and import names do not have to match.

## Common mistakes

- Publishing before confirming the license and public-data rights.
- Uploading the outer repository ZIP as one GitHub file.
- Forgetting `.github/workflows`, so no CI or publisher runs.
- Storing a PyPI password in source code or workflow YAML.
- Reusing a version that already exists on PyPI.
- Tagging before version metadata and changelog agree.
- Testing only an editable install rather than the built wheel.
- Committing `dist/`, `.venv/`, caches, local search results, or private datasets.

Authoritative references: the [Python Packaging User Guide tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/) and [PyPI Trusted Publishers documentation](https://docs.pypi.org/trusted-publishers/).
