# Contributing to MPASdiag

First off, thank you for considering contributing to **MPASdiag**! This package
is developed for the atmospheric science community, and contributions of all
kinds — bug reports, feature requests, documentation, and code — are very
welcome.

By participating in this project, you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Table of contents

- [Ways to contribute](#ways-to-contribute)
- [Reporting bugs](#reporting-bugs)
- [Suggesting features](#suggesting-features)
- [Development setup](#development-setup)
- [Making changes](#making-changes)
- [Code style and quality](#code-style-and-quality)
- [Testing](#testing)
- [Submitting a pull request](#submitting-a-pull-request)
- [Release process](#release-process)

## Ways to contribute

- **Report bugs** using the [bug report template](https://github.com/mrixlam/MPASdiag/issues/new?template=bug_report.yml).
- **Request features** using the [feature request template](https://github.com/mrixlam/MPASdiag/issues/new?template=feature_request.yml).
- **Improve documentation** — fixes to the README, docstrings, or examples are
  always appreciated.
- **Submit code** — bug fixes and new diagnostics via pull requests.

## Reporting bugs

Before opening an issue, please search the
[existing issues](https://github.com/mrixlam/MPASdiag/issues) to avoid
duplicates. A good bug report includes a **minimal reproducible example**, the
full traceback, and your environment details (OS, Python version, and the
versions of key dependencies such as `uxarray`, `mpi4py`, `metpy`, and
`cartopy`).

## Suggesting features

Open a feature request describing the problem you are trying to solve, the
proposed solution, and any alternatives you have considered. For larger
proposals, consider starting a
[discussion](https://github.com/mrixlam/MPASdiag/discussions) first.

## Development setup

MPASdiag depends on a number of scientific libraries (e.g. `uxarray`, `mpi4py`,
`cartopy`, `esmpy`/`xesmf`) that are most reliably installed with **conda**.

### Using conda (recommended)

```bash
# Clone your fork
git clone https://github.com/mrixlam/MPASdiag.git
cd MPASdiag

# Create and activate the development environment
conda env create -f environment.yml
conda activate mpasdiag

# The environment.yml already installs the package in editable mode (-e .)
```

### Using pip

```bash
git clone https://github.com/mrixlam/MPASdiag.git
cd MPASdiag

python -m venv venv
source venv/bin/activate

pip install -e ".[dev]"
# Note: mpi4py requires system MPI libraries (OpenMPI or MPICH).
#   macOS:        brew install open-mpi
#   Ubuntu/Debian: sudo apt-get install libopenmpi-dev
# esmpy is conda-only on some platforms; see the README for details.
```

### Install pre-commit hooks (optional but recommended)

```bash
pip install pre-commit
pre-commit install
```

This runs the linters and formatters automatically on every commit.

## Making changes

1. Create a topic branch off `main`:
   ```bash
   git checkout -b feature/short-description
   ```
2. Make your changes in focused, logical commits.
3. Add or update tests for any behavior you change.
4. Update documentation and add a `CHANGELOG.md` entry under **Unreleased**.

## Code style and quality

This project uses the following tools (configured in `pyproject.toml`). They are
also enforced in CI, so please run them before pushing:

```bash
ruff check mpasdiag/ tests/      # linting
black --check mpasdiag/ tests/   # formatting (use `black mpasdiag/ tests/` to apply)
mypy mpasdiag/                   # static type checking
```

Guidelines:

- Add type annotations to all new functions (mypy runs in strict-ish mode).
- Write clear docstrings (NumPy style) for public functions and classes.
- Keep changes consistent with the surrounding code.

## Testing

Tests live in `tests/` and use `pytest`:

```bash
# Run the full test suite with coverage
pytest tests/ --cov=mpasdiag --cov-report=term-missing

# Run tests in parallel
pytest tests/ -n auto

# Run a single test file
pytest tests/visualization/test_cross_section_coverage.py -v
```

Please make sure the full suite passes and that new code is covered by tests.

## Submitting a pull request

1. Push your branch to your fork and open a pull request against `main`.
2. Fill out the pull request template, including the checklist.
3. Ensure all CI checks (lint, type-check, tests) pass.
4. A maintainer will review your changes. Please be responsive to feedback.

By submitting a pull request, you agree that your contributions will be licensed
under the project's [MIT License](LICENSE).

## Release process

Releases are managed by the maintainers:

1. Update the version in `pyproject.toml` and `mpasdiag/__init__.py`.
2. Move the **Unreleased** section of `CHANGELOG.md` to a new version heading.
3. Commit, then tag the release: `git tag -a vX.Y.Z -m "vX.Y.Z"` and push the tag.
4. The release workflow builds the distribution and publishes it to PyPI.

Thank you for helping make MPASdiag better!
