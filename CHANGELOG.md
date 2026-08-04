# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- Hardened MPASdiag against untrusted inputs (shared/downloaded config files,
  crafted NetCDF/HDF5 data or grid files), automated/LLM agents, and hostile MPI
  co-ranks, extending the pre-release audit uniformly across the codebase.
- **Path confinement**: every user- or config-supplied path — `--grid-file`,
  `--data-dir`, `--output-dir`, `--output`, `--log-file`, `--config`, and the
  weights cache — must now resolve within the working directory (or an explicit
  `--base-dir`); paths that escape via `..` or an absolute location are refused.
  Untrusted filename components (e.g. a variable name read from a data file) are
  sanitized so they cannot inject a path separator or escape the output directory.
- **Resource limits**: generous, env-overridable dimension/size caps are now
  enforced uniformly on the raw data/grid load, the in-memory cache, the live
  regridder build (source cells and per-cell vertices), and MPI-broadcast
  metadata — not only the cached-weights load. The multiprocessing worker count
  is capped to prevent a fork-bomb from a hostile `workers` value.
- **Configuration validation**: a file-loaded configuration is fully
  re-validated after command-line overrides are merged, and numeric parameters
  (DPI, figure size, worker count, chunk size, etc.) are range-checked; unknown
  YAML keys are ignored with a warning instead of crashing.
- **Injection hardening**: file-derived text (variable names, `long_name`/`units`
  attributes) is sanitized before it appears in log/error messages or plot
  labels, mitigating log forging, indirect prompt injection when an agent drives
  the tool, and matplotlib mathtext rendering failures.
- **Supply chain**: `pip-audit` now fails CI on known-vulnerable dependencies
  (previously advisory only).

### Added
- `--base-dir <dir>` command-line option (and `MPASConfig.base_dir`) to move the
  path-confinement boundary to a trusted directory such as `/scratch` or
  `/glade`, so input/output can live outside the working directory.
- Environment overrides for the new safety limits: `MPASDIAG_MAX_CELL_VERTICES`,
  `MPASDIAG_MAX_WORKERS`, and `MPASDIAG_MAX_INPUT_FILES`.
- `file_pattern` argument on `MPAS2DProcessor`, `MPAS3DProcessor`, and
  `MPASBaseProcessor` to read custom output streams whose file names do not
  follow the built-in `diag*.nc` / `mpasout*.nc` conventions, e.g.
  `MPAS2DProcessor(grid_file=..., file_pattern="wiso.*.nc")`. Matching files are
  searched for in the data directory and then recursively beneath it; file names
  must still embed a `YYYY-MM-DD_HH.MM.SS` timestamp. See
  `examples/example_custom_file_pattern.py`.

### Changed
- **Breaking**: absolute or out-of-tree input/output paths are now rejected
  unless `--base-dir` is supplied. Workflows that read or write outside the
  current working directory must pass `--base-dir <trusted dir>`.
- Unexpected top-level errors now print a concise message by default; the full
  traceback is emitted only at `DEBUG` verbosity (`--verbose`/`--log-level DEBUG`).

## [1.0.0] - 2026-06-12

First public, production-ready release of **MPASdiag**, a Python package for
MPAS model output analysis and visualization on unstructured meshes.

### Added
- **Data processing**: readers and processors for 2D and 3D MPAS output,
  with parallel (MPI / multiprocessing) batch processing support.
- **Remapping**: pluggable regridding via KDTree (nearest/linear) and
  ESMPy/xESMF (conservative) engines.
- **2D diagnostics**: precipitation analysis, surface fields, and wind
  diagnostics (speed, barbs, arrows, streamlines).
- **3D diagnostics**: vertical soundings with Skew-T Log-P plots and
  thermodynamic indices (CAPE, CIN, SRH, bulk shear), vertical cross-sections
  (pressure and height coordinates), and integrated vapor transport (IVT/IWV).
- **Visualization**: publication-quality Cartopy-based maps with configurable
  overlays and multi-layer composites.
- **Command-line interface** (`mpasdiag`) for single-time and batch workflows.
- Comprehensive test suite with coverage reporting and CI across
  Python 3.10–3.13.

[Unreleased]: https://github.com/mrixlam/MPASdiag/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/mrixlam/MPASdiag/releases/tag/v1.0.0
