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
- **Projection centering and keyword controls**: `SurfaceMapStyle`,
  `PrecipitationRenderStyle`, and `WindPlotStyle` gain `central_longitude`,
  `central_latitude`, and `proj_kwargs` fields to control the map projection
  aesthetic. `central_longitude` enables, e.g., Pacific-centered / dateline-
  crossing maps (`central_longitude=180`) on the default `PlateCarree`
  projection; `proj_kwargs` passes any keyword straight through to the cartopy
  projection constructor (e.g. `standard_parallels`, `globe`, `satellite_height`).
  All three default to `None`, so existing output is unchanged.
- **Expanded projection whitelist**: `Robinson`, `Mollweide`, `Orthographic`,
  `NorthPolarStereo`, `SouthPolarStereo`, and `NearsidePerspective` are now
  supported in addition to `PlateCarree`, `Mercator`, and `LambertConformal`.

### Changed
- **Breaking**: absolute or out-of-tree input/output paths are now rejected
  unless `--base-dir` is supplied. Workflows that read or write outside the
  current working directory must pass `--base-dir <trusted dir>`.
- Unexpected top-level errors now print a concise message by default; the full
  traceback is emitted only at `DEBUG` verbosity (`--verbose`/`--log-level DEBUG`).
- Projection names are now validated against a whitelist and an unrecognized name
  falls back to `PlateCarree` with a warning; wind plots build their projection
  through the same shared factory as the surface and precipitation plotters, so
  they now honor auto-centering and the new centering controls consistently
  (previously wind ignored projection centering entirely).

### Fixed
- Global maps on a projected CRS (Mercator, Robinson, Mollweide, ...) now render
  the full projection domain via `set_global()` instead of a lon/lat
  `set_extent()`. Previously a global extent combined with a `central_longitude`
  near the antimeridian (e.g. a Pacific-centered `central_longitude=180`)
  collapsed the map — Mercator rendered a blank canvas and Robinson showed only a
  sliver — because both longitude bounds mapped to nearly the same projected x.
  Affected the surface, precipitation, and wind plotters.
- Globe-view projections (`Orthographic`, `NearsidePerspective`) no longer raise
  "Axis limits cannot be NaN or Inf" when a global extent is requested; the extent
  helper falls back to `set_global()` for extents a projection cannot represent.

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
