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
- **Projection centering and keyword controls**: `SurfaceMapStyle`,
  `PrecipitationRenderStyle`, and `WindPlotStyle` gain `central_longitude`,
  `central_latitude`, and `proj_kwargs` fields to control the map projection
  aesthetic. `central_longitude` enables, e.g., Pacific-centered / dateline-
  crossing maps (`central_longitude=180`) on the default `PlateCarree`
  projection; `proj_kwargs` passes any keyword straight through to the cartopy
  projection constructor (e.g. `standard_parallels`, `globe`, `satellite_height`).
  All three default to `None`. An explicit `central_longitude` outside
  [-180, 180] is wrapped into that range; a non-finite center or a
  `central_latitude` outside [-90, 90] raises `ValueError`.
- **Dateline-crossing regional maps**: a regional box that crosses ±180° can be
  requested with `lon_max > 180` (e.g. `GeographicBounds(150, 210, -30, 30)`)
  in the surface, precipitation, and wind plotters. Global-style projections
  (`PlateCarree`, `Robinson`, ...) center such a box on its midpoint
  automatically, and cells on both sides of the dateline are kept.
- **Expanded projection whitelist**: `Robinson`, `Mollweide`, `Orthographic`,
  `NorthPolarStereo`, `SouthPolarStereo`, `NearsidePerspective`,
  `AlbersEqualArea`, `AzimuthalEquidistant`, `LambertAzimuthalEqualArea`,
  `Stereographic`, `TransverseMercator`, `Gnomonic`, `EckertIV`, `EqualEarth`,
  `LambertCylindrical`, `Miller`, `Sinusoidal`, `InterruptedGoodeHomolosine`,
  and `Geostationary` are now supported in addition to `PlateCarree`,
  `Mercator`, and `LambertConformal`.

- The `overlay` subcommand is now implemented (it previously wrote nothing and
  exited 0): `precip_wind`, `temp_pressure`, `wind_temp`, `multi_level` and
  `custom` overlays are rendered with the existing surface, precipitation and
  wind plotters and honour all overlay, extent, time and output options.
- `--levels` command-line option (explicit contour/color levels, in plotted
  units), `--use-pure-xarray`, cross-section `--clim-min/--clim-max`
  (`CrossSectionStyle.clim_min/clim_max`), surface `--plot-type pcolormesh`, and
  a working `--interpolation nearest|linear|cubic`
  (`SurfaceMapStyle.interpolation`).
- `MPASConfig.from_dict` accepts YAML keys spelled like the command-line options
  (e.g. `formats`, `accumulation`, `time_range`, `figure_width`/`figure_height`)
  and validates boolean values.
- Sounding profiles carry the terrain height (`surface_height`); pass it as
  `surface_height_m` to `compute_thermodynamic_indices` so wet-bulb-zero height
  and the 0-1/0-3/0-6 km layers are measured above ground level.
- `MPASDIAG_MAX_TASKS_PER_WORKER` (default 100, 0 disables) recycles
  multiprocessing workers after that many tasks, bounding slow per-task memory
  growth in long batches.
- The multiprocessing worker count is lowered, with a warning, when the
  estimated memory per worker exceeds what is available; set
  `MPASDIAG_WORKER_MEMORY_GB` to a per-worker size in GB, or 0 to disable.

### Changed
- **Breaking**: the minimum supported Python is now **3.11** (`requires-python
  = ">=3.11"`, was `">=3.10"`). The supported and CI-verified matrix moves from
  3.10–3.13 to **3.11–3.14**. Python 3.10 users must upgrade to 3.11 or newer;
  `pip`/`uv` will refuse to install this version on 3.10 rather than fail at
  runtime. The 3.10 leg was dropped because it hangs indefinitely in conda's
  dependency solve on GitHub's macOS runners (45+ minutes, unreproducible
  off-runner), so 3.10 support could no longer be verified by CI; 3.14 was
  verified to resolve the full scientific stack before the swap. `uv.lock`,
  the classifiers, `environment.yml`, the black/mypy targets and the README
  prerequisites were regenerated/updated to match.
- **Breaking**: absolute or out-of-tree input/output paths are now rejected
  unless `--base-dir` is supplied. Workflows that read or write outside the
  current working directory must pass `--base-dir <trusted dir>`.
- Unexpected top-level errors now print a concise message by default; the full
  traceback is emitted only at `DEBUG` verbosity (`--verbose`/`--log-level DEBUG`).
- Projection names are now validated (case-insensitively) against a whitelist
  and an unrecognized name falls back to `PlateCarree` with a warning; wind plots
  build their projection through the same shared factory as the surface and
  precipitation plotters, so they now honor auto-centering and the new centering
  controls consistently. Previously `MPASWindPlotter` accepted any cartopy class
  name (case-sensitive) and never centered the map: wind maps on `Mercator` and
  `LambertConformal` are now centered on the requested domain (a
  `LambertConformal` wind map previously stayed centered on -96°, so non-US
  domains rendered as a distorted sliver), and cartopy projections outside the
  whitelist now fall back to `PlateCarree`.
- Default global maps (no centering arguments) now span the full ±180° and use
  the projection's natural global domain, so global surface and precipitation
  maps differ slightly from previous releases (e.g. longitude labels at the
  ±180° edges); global `LambertConformal` maps now show the full cone instead of
  a collapsed wedge. Default regional maps are unchanged.

- **Breaking**: storm-relative helicity (and STP/SCP, which use it) is now
  computed relative to the Bunkers right-moving storm motion, following SPC
  practice; it was previously ground-relative.
- Potential-temperature fields (`theta`, `theta_base`, `th2m`, ...) are shown in
  K instead of °C, with level presets that cover real data.
- `mpi4py` is imported only when the MPI backend is requested or an MPI launcher
  is detected. Serial and multiprocessing runs no longer initialise MPI, which on
  some MPICH installations made every command (even `--help`) exit with 143.
- Output file names now include the cross-section endpoints, the wind level for
  3D winds, and `_indices`/`_parcel` for soundings, so different runs no longer
  overwrite each other; `--output` is honoured in parallel batches.
- Relative paths are resolved against the current directory before the
  `--base-dir` confinement check; `--quiet`/`--verbose` override a YAML
  `log_level`.
- Colorbar end arrows are drawn only on the side where data exceed the levels,
  and scatter colorbars end at the last level (no extra "max + 1" bin).
- Integer and `"top"` levels of interface variables (`w`) in the wind
  diagnostics return the mass-level value, co-located with u/v.
- Density potential temperature uses the exact formula
  `theta*(1+qv/eps)/(1+qt)`.
- Importing the package no longer loads `metpy.plots` (about 10 s on the test
  machine); it is imported when a Skew-T is drawn. Every xarray/uxarray open
  now names the `netcdf4` engine, because xarray's engine guessing loads all
  backend plugins and MetPy's plugin imports `metpy.plots` as well. Start-up
  drops from ~15 s to ~5 s in the main process, every multiprocessing worker
  and every MPI rank.
- ESMF cell boundaries are built with vectorised indexing (about 10x faster on
  large meshes), and the boundaries and regridder are cached per process, so
  parallel tasks no longer rebuild them for every time step.
- Multiprocessing workers reuse the parent's loaded (lazy) dataset instead of
  re-opening and combining every data file: it is pickled once to a private
  temporary file and each worker loads it in well under a second, then reads
  only the time steps it plots. Worker start-up drops from several seconds to
  about half a second; output is unchanged. The forkserver preloads `netCDF4`
  first, because a macOS worker that loads its native libraries for the first
  time after the fork can segfault, and a worker crash now reports its exit
  code or signal.
- The precipitation, surface, wind and cross-section commands load only the
  variables they plot (plus time and vertical-coordinate helpers), the same
  selection the batch workers use, instead of all variables in the files
  (118 in a typical `diag` file). This saves seconds per run in serial,
  `--parallel` and on every MPI rank. If none of the requested variables are
  in the files, everything is loaded so the error still lists the available
  variables. Soundings and overlays still load every variable.
- On macOS, a `--parallel` batch starts the multiprocessing forkserver at the
  beginning of the run, so it imports the plotting stack while the data load
  rather than afterwards. The forkserver also preloads the modules every
  worker's first task imported (dask.array, pint, distributed,
  scipy.interpolate), about a second less per worker.
- `metpy.calc` and `metpy.units` are imported on first use (about a second per
  process, including every worker and MPI rank) instead of at package import.
- Worker thread limits now include numba's thread pool (used by uxarray),
  sized to the worker's share of the cores unless `NUMBA_NUM_THREADS` is set.
- A missing optional vertical-coordinate helper (`pressure`, `zgrid`, `fzp`,
  ...) no longer logs a "Requested variable(s) not found" warning on every
  3D load.
- Under an MPI launcher, runs that do not distribute work across ranks (no
  `--parallel`, a single time step, soundings, overlays) run on rank 0 only
  with a warning, instead of every rank repeating the run and writing the same
  files.

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
  This now also covers the base `MPASVisualizer.create_wind_plot`, which
  previously crashed for globe-view projections, global `Mercator`, and polar
  `LambertConformal` extents.
- Extents that `set_extent()` silently collapsed to a sliver on a re-centered
  projection (e.g. a near-global box -179..179 with `central_longitude=180`) are
  detected and shown with `set_global()` instead of a blank map, and
  full-longitude latitude bands (e.g. -180..180 by -30..30, or a polar cap) are
  shown at full projection width on cylindrical and pseudo-cylindrical
  projections. Previously these produced a blank map or crashed cartopy's
  gridliner (`object of type 'MultiPoint' has no len()`) in wind plots.
- Scatter plots of a box given in 0..360 longitudes (e.g. a global `0..360`
  extent, or `150..210`) no longer drop the cells stored at negative longitudes.
- Linear remapping of a global mesh (`remap_mpas_to_latlon(method="linear")`)
  no longer leaves NaN columns at the ±180° edges of a [-180, 180] grid; the
  source mesh is wrapped across the dateline before interpolation, as was
  already done for nearest-neighbor remapping onto 0..360 grids.
- Wind streamlines no longer break into a gap at ±180° on maps centered at 180°
  or on dateline-crossing regional boxes.
- Errors raised by PROJ while building a projection (e.g. an out-of-range
  parameter in `proj_kwargs`) are re-raised as `ValueError` naming the
  projection and its valid parameters, and an explicit `central_latitude` that
  the projection does not support is reported with a warning.

- Multiprocessing batches no longer hang when a worker dies (e.g. paths given
  as `pathlib.Path`, or an in-memory processor); workers inherit the log level
  and log file.
- Conservative ESMF remapping no longer produces wrong values at the 0° meridian
  and the poles; the regional validity mask no longer blanks the western edge
  column of maps of a global mesh; stale or mismatched cached weights are
  regenerated.
- Data directories whose names contain glob characters (`[`, `]`, `*`, `?`) are
  found.
- Wind plots honour `--show-background`; global streamline plots use a grid
  sized to the extent instead of a fixed 0.1° grid (which could exhaust memory).
- Unknown accumulation periods (e.g. `a05h`) are computed correctly or rejected
  instead of silently using 24 h; negative time indices are accepted
  consistently.
- Wind speed/direction/shear and the thermodynamic helpers accept dask-backed
  and plain NumPy inputs; wind-direction statistics use circular means.
- Sounding indices reject Pa/Kelvin/mismatched inputs and accept top-first
  profiles; Skew-T CIN shading is limited to the layer below the EL, the
  temperature axis fits cold profiles, LCL/LFC/EL labels are visible and barbs
  are thinned.
- Cross sections: a model-level display of a pressure/height section shows
  level indices, `max_height` is respected when choosing levels,
  `pcolormesh` honours levels and `contour` gets a colorbar; edge variables,
  coincident endpoints and out-of-range time indices raise clear errors, and
  antipodal paths are well defined.
- Surface maps honour `level_value` and show the level in the title; explicit
  `central_longitude` no longer hides a dateline-crossing extent; polar caps use
  a circular boundary; `save_plot` accepts a string format and does not double
  the extension.
- Unit metadata: `uReconstructZonal/Meridional` units, effective-radius and
  `ni` units, and overlay contours drawn in display units (e.g. q2 in g/kg).
- Multiprocessing batches of 16 or more tasks per worker silently fell back to
  serial execution in the parent process (a chunked `imap_unordered` has no
  timeout-capable `next()`); tasks are now dispatched one at a time.
- Contour and contourf maps and overlays with missing values (for example
  negative accumulations after a bucket reset) no longer fail with an
  `IndexError` in the remapper; missing cells stay missing on the map.
- Parallel (multiprocessing and MPI) surface batches now read the data the same
  way as serial runs: isobaric fields are masked below ground instead of
  showing extrapolated values, and vertex fields such as `vorticity_850hPa`
  are plotted instead of failing every step.
- Under `srun --mpi=pmix` and other launchers that export the rank and the
  size in different variables (`PMIX_RANK` with `SLURM_NTASKS`, `PALS_RANKID`
  with `PALS_NRANKS`), rank 0 no longer concludes it runs alone while the
  other ranks start MPI.
- A batch stopped with SIGTERM (for example at a scheduler's wall-time limit)
  removes its temporary processor file, and an MPI batch that fails no longer
  keeps its loaded dataset in memory.
- Sounding indices measured above ground (wet-bulb-zero height, 0-1/0-6 km
  shear, SRH) use the grid file's terrain height; MPAS output files can carry
  a `ter` filled with zeros, which put the ground at sea level everywhere.
- An installed MetPy that fails to import (for example a pint/numpy mismatch)
  falls back to the MetPy-free sounding estimates again, and Skew-T plots
  report it with the "MetPy is required" message.
- A 3D wind or variable at a pressure level outside the model's pressure range
  is an error instead of an empty plot written with exit status 0, and the
  message for an implausibly small level names both Pa and hPa.
- Cross-section start and end latitudes outside [-90, 90] are rejected.
- Longitudes given in 0..360 (for example `lonCell` in degrees) plot like the
  same data in -180..180; the ESMF remapping and the precipitation plotter
  used to drop the western hemisphere.
- A plotter reused for another mesh with the same number of cells, or with a
  different weights directory, builds a new regridder, and the mesh
  fingerprint used by the regridder and cell-boundary caches hashes the whole
  coordinate array (a large evenly spaced sample beyond 2 million values).
- When one of the requested variables is missing (for example a misspelled
  `--u-variable`), the error lists every variable in the files, not only the
  ones that were loaded.
- `sounding --max-distance-km` no longer says a point inside the domain lies
  outside it, and MetPy's "Interpolation point out of data bounds" warning no
  longer reaches the console for indices that are simply undefined.

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
