# MPASdiag - MPAS Diagnostic Analysis Package

A comprehensive Python package for analyzing and visualizing MPAS (Model for Prediction Across Scales) unstructured mesh model output data with support for both serial and parallel processing workflows. This package provides specialized tools for atmospheric model diagnostics including precipitation accumulation analysis with configurable periods, 2D surface variable visualization using scatter plots and filled contours, horizontal wind vector plotting with barbs and arrows, 3D vertical atmospheric cross-sections along arbitrary paths, thermodynamic Skew-T Log-P sounding diagrams with full MetPy index suites, and vertically integrated water vapor (IWV) and water vapor transport (IVT) diagnostics for atmospheric river detection. The toolkit features automatic unit conversion following CF conventions, professional meteorological styling, memory-efficient data processing through lazy loading and chunking strategies, and extensible architecture with modular processors and plotters. Advanced capabilities include powerful remapping from unstructured MPAS meshes to regular latitude-longitude grids using xESMF and SciPy, MPI-based parallel batch processing for time series generation, comprehensive command-line interface with YAML configuration support, and flexible vertical coordinate systems (pressure, height, model levels) for 3D analysis. The package is designed for operational meteorology applications, climate model evaluation, and research workflows requiring publication-quality visualizations from MPAS atmospheric simulations.

[![CI](https://github.com/mrixlam/MPASdiag/workflows/MPASdiag%20CI/badge.svg)](https://github.com/mrixlam/MPASdiag/actions)
[![codecov](https://codecov.io/gh/mrixlam/MPASdiag/branch/main/graph/badge.svg)](https://codecov.io/gh/mrixlam/MPASdiag)
![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-beta-yellow.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19486544.svg)](https://doi.org/10.5281/zenodo.19486544)

## Installation

### Prerequisites

- Python 3.8 or higher
- NetCDF4-compatible system libraries
- HDF5 libraries (for h5netcdf)
- MPI libraries (OpenMPI or MPICH for parallel processing)
- ESMpy and xESMF (for remapping capabilities)

### From GitHub Repository (Recommended)

```bash
# Clone the repository
git clone https://github.com/mrixlam/MPASdiag.git
cd MPASdiag

# Create a new conda environment
conda create -n mpasdiag python=3.9
conda activate mpasdiag

# Install required scientific python libraries
conda install -y -c conda-forge numpy pandas scipy xarray dask numba

# Install additional dependencies
conda install -y -c conda-forge netcdf4 h5netcdf llvmlite pyyaml psutil 

# Install testing suite dependencies
conda install -y -c conda-forge pytest pytest-cov pytest-xdist pluggy coverage execnet

# Install matplotlib and cartopy for plotting support
conda install -y -c conda-forge matplotlib cartopy

# Install UXarray (for unstructured grid support)
conda install -y -c conda-forge uxarray

# Install MPI support for parallel processing
conda install -y -c conda-forge mpi4py

# Install esmpy and xESMF for remapping capabilities
conda install -y -c conda-forge esmpy xesmf 

# Install MetPy for thermodynamic diagnostics
conda install -y -c conda-forge metpy

# Install the package
pip install -e .
```

### Using pip (Alternative)

```bash
# Clone the repository
git clone https://github.com/mrixlam/MPASdiag.git
cd MPASdiag

# Create virtual environment
python -m venv mpasdiag
source mpasdiag/bin/activate  # On Windows: mpasdiag\Scripts\activate

# Install the package
pip install -r requirements.txt
pip install -e .

# Note: mpi4py requires MPI libraries (OpenMPI or MPICH) installed on your system
# On macOS: brew install open-mpi
# On Ubuntu/Debian: sudo apt-get install libopenmpi-dev
# On RHEL/CentOS: sudo yum install openmpi-devel

# Optional: install MetPy for thermodynamic diagnostics
pip install metpy
```

## Python API Usage

The package exposes a small, focused programmatic API in the `mpasdiag` package. Below are common classes and a short example showing the typical workflow.

### Key classes

#### Processing
- `mpasdiag.processing.utils_config.MPASConfig` — configuration dataclass for CLI and programmatic runs with full YAML load/save support
- `mpasdiag.processing.processors_2d.MPAS2DProcessor` — load and extract 2D (surface) fields from MPAS output with UXarray/xarray backends
- `mpasdiag.processing.processors_3d.MPAS3DProcessor` — load and extract 3D atmospheric fields with vertical coordinate handling
- `mpasdiag.processing.remapping.MPASRemapper` — remap unstructured MPAS data to regular grids using xESMF or KDTree
- `mpasdiag.processing.parallel.MPASParallelManager` — MPI-based parallel execution manager for batch processing
- `mpasdiag.processing.data_cache.MPASDataCache` — intelligent caching for coordinates and frequently accessed data
- `mpasdiag.processing.utils_unit.UnitConverter` — CF-convention unit conversion utilities
- `mpasdiag.processing.utils_metadata.MPASFileMetadata` — variable metadata extraction and description
- `mpasdiag.processing.utils_file.FileManager` — file discovery, system info, and MPAS output pattern matching
- `mpasdiag.processing.utils_geog.MPASGeographicUtils` — coordinate transformation and geographic utilities
- `mpasdiag.processing.utils_datetime.MPASDateTimeUtils` — time dimension validation and date/time utilities

#### Diagnostics
- `mpasdiag.diagnostics.precipitation.PrecipitationDiagnostics` — compute precipitation accumulation and differencing
- `mpasdiag.diagnostics.wind.WindDiagnostics` — compute wind speed, direction, shear, and derived quantities
- `mpasdiag.diagnostics.sounding.SoundingDiagnostics` — extract vertical sounding profiles from MPAS 3D output and compute comprehensive thermodynamic indices (CAPE, CIN, LI, K-Index, STP, SCP, SRH, bulk shear, precipitable water, wet-bulb zero, and more) using MetPy
- `mpasdiag.diagnostics.moisture_transport.MoistureTransportDiagnostics` — compute vertically integrated water vapor (IWV) and vertically integrated water vapor transport (IVT) components using trapezoidal pressure-coordinate integration

#### Visualization
- `mpasdiag.visualization.precipitation.MPASPrecipitationPlotter` — create professional precipitation accumulation maps
- `mpasdiag.visualization.surface.MPASSurfacePlotter` — surface scatter/contour/pcolormesh maps with automatic unit conversion
- `mpasdiag.visualization.wind.MPASWindPlotter` — wind vector visualizations (barbs/arrows/streamlines) and overlay support
- `mpasdiag.visualization.cross_section.MPASVerticalCrossSectionPlotter` — vertical atmospheric cross sections along arbitrary paths
- `mpasdiag.visualization.skewt.MPASSkewTPlotter` — Skew-T Log-P diagrams from MPAS sounding profiles with thermodynamic overlays, parcel traces, CAPE/CIN shading, and indices table (requires MetPy)
- `mpasdiag.visualization.styling.MPASVisualizationStyle` — consistent plot styling, colorbars, branding, and save utilities

### Minimal example

#### Precipitation Analysis
```python
# Correct import paths and usage for current package layout
from mpasdiag.processing.utils_config import MPASConfig
from mpasdiag.processing.processors_2d import MPAS2DProcessor
from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter

# (1) Configure (use an existing grid file from the repository)
cfg = MPASConfig(grid_file='data/grids/x1.10242.init.nc', data_dir='./data/u240k', output_dir='./output')

# (2) Load data
proc = MPAS2DProcessor(cfg.grid_file)
proc.load_2d_data(cfg.data_dir)

# extract lon/lat from the loaded dataset
lon, lat = proc.extract_spatial_coordinates()

# (3) Compute precipitation for a time index
diag = PrecipitationDiagnostics(verbose=True)
precip = diag.compute_precipitation_difference(proc.dataset, 0, var_name='total', accum_period='a01h', data_type=proc.data_type)

# (4) Plot
plotter = MPASPrecipitationPlotter(figsize=(12, 8), dpi=300)
fig, ax = plotter.create_precipitation_map(lon, lat, precip.values,
                                           cfg.lon_min, cfg.lon_max, cfg.lat_min, cfg.lat_max,
                                           title='Total precipitation', accum_period='a01h', data_array=precip)
plotter.save_plot('./output/total_precipitation', formats=['png'])
```

#### Complex Wind Plot (Wind Speed + Barbs Overlay)
```python
from mpasdiag.processing.processors_2d import MPAS2DProcessor
from mpasdiag.visualization.surface import MPASSurfacePlotter
from mpasdiag.visualization.wind import MPASWindPlotter
import numpy as np

# (1) Load data
processor = MPAS2DProcessor(grid_file='data/grids/x1.10242.init.nc')
processor.load_2d_data('./data/u240k/diag')

# (2) Extract wind components at time index 0
u_data = processor.get_2d_variable_data('u10', 0)
v_data = processor.get_2d_variable_data('v10', 0)
lon, lat = processor.extract_2d_coordinates_for_variable('u10', u_data)

# (3) Compute wind speed
wind_speed = np.sqrt(u_data.values**2 + v_data.values**2)

# (4) Create wind speed background with filled contours
surface_plotter = MPASSurfacePlotter(figsize=(14, 11), dpi=150)
fig, ax = surface_plotter.create_surface_map(
    lon=lon, lat=lat, data=wind_speed, var_name='wind_speed',
    lon_min=-130, lon_max=-60, lat_min=20, lat_max=55,
    title='MPAS Wind Analysis - Speed with Direction Barbs',
    plot_type='contourf', colormap='YlOrRd'
)

# (5) Add wind barbs overlay showing direction
wind_plotter = MPASWindPlotter()
wind_config = {
    'u_data': u_data.values,
    'v_data': v_data.values,
    'plot_type': 'barbs',
    'subsample': -1,  # Auto-calculate optimal density
    'color': 'black',
    'grid_resolution': 0.1,
    'regrid_method': 'linear'
}
wind_plotter.add_wind_overlay(
    ax=ax, lon=lon, lat=lat, wind_config=wind_config,
    lon_min=-130, lon_max=-60, lat_min=20, lat_max=55
)

# (6) Save the multi-layer visualization
surface_plotter.save_plot('./output/wind_complex', formats=['png'])
```

#### Skew-T Log-P Sounding (Thermodynamic Profile)
```python
from mpasdiag.processing.processors_3d import MPAS3DProcessor
from mpasdiag.diagnostics.sounding import SoundingDiagnostics
from mpasdiag.visualization.skewt import MPASSkewTPlotter

# (1) Load 3D MPAS data
processor = MPAS3DProcessor(grid_file='data/grids/x1.10242.static.nc', verbose=True)
processor.load_3d_data('./data/u240k/mpasout')

# (2) Extract sounding profile at a target location (e.g., Singapore)
diag = SoundingDiagnostics(verbose=True)
profile = diag.extract_sounding_profile(processor, lon=103.82, lat=1.35, time_index=0)

# (3) Compute comprehensive thermodynamic indices (CAPE, CIN, SRH, bulk shear, etc.)
indices = diag.compute_thermodynamic_indices(
    profile['pressure'], profile['temperature'], profile['dewpoint'],
    u_wind_kt=profile.get('u_wind'), v_wind_kt=profile.get('v_wind'),
    height_m=profile.get('height')
)

# (4) Plot Skew-T Log-P diagram with parcel trace and indices table
plotter = MPASSkewTPlotter(figsize=(9, 12), dpi=150)
fig, ax = plotter.create_skewt_diagram(
    pressure=profile['pressure'], temperature=profile['temperature'],
    dewpoint=profile['dewpoint'], u_wind=profile.get('u_wind'),
    v_wind=profile.get('v_wind'), title='MPAS Sounding — Singapore',
    indices=indices, show_parcel=True
)
fig.savefig('./output/skewt_singapore.png', bbox_inches='tight', dpi=150)
```

#### Vertically Integrated Water Vapor Transport (IVT)
```python
import numpy as np
from mpasdiag.processing.processors_3d import MPAS3DProcessor
from mpasdiag.diagnostics.moisture_transport import MoistureTransportDiagnostics
from mpasdiag.visualization.surface import MPASSurfacePlotter
from mpasdiag.visualization.wind import MPASWindPlotter
from mpasdiag.processing.utils_config import MPASConfig

# (1) Load 3D MPAS data
processor = MPAS3DProcessor(grid_file='data/grids/x1.10242.static.nc')
processor.load_3d_data('./data/u240k/mpasout')
tindex = 0

# (2) Extract required 3D fields
qv       = processor.dataset['qv'].isel(Time=tindex)
u_3d     = processor.dataset['uReconstructZonal'].isel(Time=tindex)
v_3d     = processor.dataset['uReconstructMeridional'].isel(Time=tindex)
pressure = (processor.dataset['pressure_p'].isel(Time=tindex) +
            processor.dataset['pressure_base'].isel(Time=tindex))

# (3) Compute IWV, IVT_u, IVT_v, and total IVT magnitude
diag   = MoistureTransportDiagnostics(verbose=True)
result = diag.analyze_moisture_transport(qv, u_3d, v_3d, pressure)

# (4) Extract 1-D arrays and plot IVT magnitude as shading with vector overlay
ivt_mag = result['ivt']['data'].values.flatten()
ivt_u   = result['ivt_u']['data'].values.flatten()
ivt_v   = result['ivt_v']['data'].values.flatten()
lon, lat = processor.extract_2d_coordinates_for_variable('qv', result['ivt_u']['data'])

plotter = MPASSurfacePlotter(figsize=(12, 9), dpi=300)
fig, ax = plotter.create_surface_map(
    lon=lon, lat=lat, data=ivt_mag, var_name='ivt',
    lon_min=-180, lon_max=180, lat_min=-90, lat_max=90,
    plot_type='contourf', title='Vertically Integrated Water Vapor Transport (IVT)'
)
wind_plotter = MPASWindPlotter()
wind_plotter.add_wind_overlay(
    ax, lon, lat, {'u_data': ivt_u, 'v_data': ivt_v, 'plot_type': 'arrows',
                   'subsample': '-1', 'scale': 15000, 'grid_resolution': 0.1},
    lon_min=-180, lon_max=180, lat_min=-90, lat_max=90
)
plotter.save_plot('./output/ivt_map', formats=['png'])
```

If you prefer the command line, see the `CLI Examples` section below.

## CLI Examples

### Precipitation Analysis
```bash
# Basic precipitation analysis (single time)
mpasdiag precipitation \
  --grid-file grid.nc --data-dir ./data --variable total --accumulation a01h --plot-type contourf

# Batch process all time steps
mpasdiag precipitation \
  --grid-file grid.nc --data-dir ./data --variable total --batch-all --dpi 300 --figure-size 12 12

# Batch processing with parallel execution (multiprocessing backend)
mpasdiag precipitation \
  --grid-file grid.nc --data-dir ./data --variable total --batch-all --parallel --workers 4 --grid-resolution 0.1
```

### Surface Variable Analysis
```bash
# Temperature scatter plot
mpasdiag surface \
  --grid-file grid.nc --data-dir ./data \
  --variable t2m --plot-type scatter

# Pressure contour plot with custom extent
mpasdiag surface \
  --grid-file grid.nc --data-dir ./data \
  --variable surface_pressure --plot-type contour \
  --lat-min -10 --lat-max 15 --lon-min 95 --lon-max 110

# Sea level pressure with batch processing
mpasdiag surface \
  --grid-file grid.nc --data-dir ./data \
  --variable mslp --plot-type contour --batch-all

# Batch processing with parallel execution
mpasdiag surface \
  --grid-file grid.nc --data-dir ./data \
  --variable t2m --batch-all --parallel --workers 6
```

### Wind Vector Analysis  
```bash
# Wind barbs (meteorological convention)
mpasdiag wind \
  --grid-file grid.nc --data-dir ./data \
  --u-variable u10 --v-variable v10 --wind-plot-type barbs \
  --grid-resolution 0.1 --subsample -1

# Wind arrows/vectors (quiver plot)
mpasdiag wind \
  --grid-file grid.nc --data-dir ./data \
  --u-variable u10 --v-variable v10 --wind-plot-type arrows \
  --scale 300 --grid-resolution 0.1 --subsample -1 

# Wind streamlines (flow trajectories)
mpasdiag wind \
  --grid-file grid.nc --data-dir ./data \
  --u-variable u10 --v-variable v10 --wind-plot-type streamlines \
  --grid-resolution 0.5 --subsample -1

# Batch processing of wind vectors
mpasdiag wind \
  --grid-file grid.nc --data-dir ./data \
  --u-variable u10 --v-variable v10 --wind-plot-type barbs --subsample -1 --batch-all

# Parallel batch processing
mpasdiag wind \
  --grid-file grid.nc --data-dir ./data \
  --u-variable u10 --v-variable v10 --wind-plot-type arrows \
  --subsample -1 --batch-all --parallel --workers 4
```

### 3D Vertical Cross-Section Analysis
```bash
# Cross section with pressure as vertical coordinate 
mpasdiag cross \
  --grid-file grid.nc --data-dir ./data \
  --variable theta --start-lon -105.0 --start-lat 39.7 \
  --end-lon -94.6 --end-lat 39.1 --vertical-coord pressure

# Cross section with height as vertical coordinate 
mpasdiag cross \
  --grid-file grid.nc --data-dir ./data \
  --variable theta --start-lon -105.0 --start-lat 39.7 \
  --end-lon -94.6 --end-lat 39.1 --vertical-coord height

# Cross section with model level as vertical coordinate 
mpasdiag cross \
  --grid-file grid.nc --data-dir ./data \
  --variable theta --start-lon -105.0 --start-lat 39.7 \
  --end-lon -94.6 --end-lat 39.1 --vertical-coord modlev

# Wind cross-section with custom colormap
mpasdiag cross \
  --grid-file grid.nc --data-dir ./data \
  --variable uReconstructZonal --start-lon -110 --start-lat 35 \
  --end-lon -80 --end-lat 45 --colormap RdBu_r

# Custom cross-section with specific time and high resolution
mpasdiag cross \
  --grid-file grid.nc --data-dir ./data \
  --variable theta --start-lon -100 --start-lat 30 \
  --end-lon -90 --end-lat 50 --time-index 12 --max-height 10 \
  --num-points 100 --dpi 400 --formats png pdf
```

### Skew-T Log-P Sounding Analysis
```bash
# Basic sounding at a single location and time step
mpasdiag sounding \
  --grid-file grid.nc --data-dir ./data \
  --lon -105.0 --lat 39.7

# Sounding with thermodynamic indices (CAPE, CIN, SRH, STP, etc.)
mpasdiag sounding \
  --grid-file grid.nc --data-dir ./data \
  --lon -105.0 --lat 39.7 --show-indices

# Sounding with parcel profile and CAPE/CIN shading (requires MetPy)
mpasdiag sounding \
  --grid-file grid.nc --data-dir ./data \
  --lon -97.5 --lat 35.2 --show-indices --show-parcel

# Batch all time steps at a fixed sounding location
mpasdiag sounding \
  --grid-file grid.nc --data-dir ./data \
  --lon -97.5 --lat 35.2 --batch-all --show-indices
```

## Configuration Files

Create YAML configuration files for repeatable analysis. All `MPASConfig` dataclass fields can be specified in YAML. Below is an annotated reference configuration covering every supported analysis type:

```yaml
# config.yaml - Full MPASConfig reference (select fields relevant to your analysis)

# --- I/O paths ---
grid_file:   "./data/grids/x1.10242.init.nc"
data_dir:    "./data/u240k"
output_dir:  "./output"

# --- Spatial extent ---
lat_min: -4.0
lat_max:  5.0
lon_min: 98.0
lon_max: 110.0

# --- Time selection ---
time_index: 0               # Single time step (ignored when batch_mode: true)
batch_mode: false           # Set true to process all time steps
# time_start: 0             # Optional start index for a time range batch
# time_end:   12            # Optional end index for a time range batch

# --- Precipitation analysis ---
variable:           "total"   # rainc | rainnc | total (rainc + rainnc)
accumulation_period: "a01h"   # a01h | a03h | a06h | a12h | a24h
plot_type:          "scatter" # scatter | contourf
precip_units:       "mm"      # mm | cm | in
# precip_threshold: 1.0       # Optional: highlight cells above this threshold (mm)

# --- Surface variable analysis ---
# variable:          "t2m"    # Any 2D variable name in the MPAS files
# plot_type:         "contourf"  # scatter | contour | contourf | pcolormesh
# grid_resolution:   0.1      # Degrees; auto-adaptive if omitted
# interpolation_method: "linear"  # linear | cubic | nearest
# contour_levels:    15       # Number of contour levels

# --- Wind vector analysis ---
u_variable:  "u10"          # Zonal wind component variable name
v_variable:  "v10"          # Meridional wind component variable name
wind_level:  "surface"      # Label for plot annotation (e.g., "850 hPa")
wind_plot_type: "barbs"     # barbs | arrows | streamlines
subsample_factor: 0         # 0 = auto-density; N = plot every Nth vector
show_background: false      # Shade background with wind speed (contourf)
background_colormap: "viridis"
vector_color: "black"
vector_alpha: 0.8
# wind_scale: null           # Quiver scale; auto if omitted
# grid_resolution: 0.5       # Regrid resolution in degrees; no regrid if omitted
regrid_method: "linear"      # linear | nearest

# --- 3D vertical cross-section ---
# variable:      "theta"     # 3D variable name (e.g., theta, uReconstructZonal)
# start_lon:     -105.0      # Cross-section start longitude
# start_lat:      39.7       # Cross-section start latitude
# end_lon:        -94.6      # Cross-section end longitude
# end_lat:         39.1      # Cross-section end latitude
vertical_coord: "pressure"   # pressure | model_levels | height
num_points:     100          # Interpolation points along the path
# max_height:    null        # Maximum height km for vertical axis; auto if omitted
plot_style:  "contourf"      # contourf | contour | pcolormesh
extend:      "both"          # both | min | max | neither

# --- Sounding / Skew-T analysis ---
# sounding_lon:  -105.0      # Sounding location longitude
# sounding_lat:    39.7      # Sounding location latitude
show_indices: false          # Compute and display full thermodynamic index table
show_parcel:  false          # Plot lifted parcel profile with CAPE/CIN shading

# --- Visualization settings ---
colormap:        "default"       # Colormap name; "default" = auto-selected per variable
# clim_min: null                 # Manual color-scale minimum
# clim_max: null                 # Manual color-scale maximum
figure_size:     [12.0, 10.0]   # Figure width and height in inches
figure_width:    12.0
figure_height:   10.0
dpi:             400
output_formats:  ["png", "pdf"]
# title: null                    # Custom plot title; auto-generated if omitted
# output: null                   # Output filename stem; auto-generated if omitted

# --- Processing options ---
use_pure_xarray: false           # Use xarray-only backend (skip UXarray)
verbose:         true
quiet:           false
parallel:        false           # Enable multiprocessing/MPI batch dispatch
# workers: null                  # Worker count; auto (CPU count) if omitted
chunk_size:      100000          # Dask chunk size for memory-efficient loading
```

Use with unified command line:
```bash
# Use configuration file with precipitation analysis
mpasdiag precipitation --config config.yaml

# Override specific settings from the CLI
mpasdiag precipitation --config config.yaml \
  --variable total --accumulation a06h --dpi 600

# Load and save a config programmatically
from mpasdiag.processing.utils_config import MPASConfig
cfg = MPASConfig.load_from_file('config.yaml')
cfg.dpi = 600
cfg.save_to_file('config_high_res.yaml')
```

## Architecture

### File Structure
```
mpasdiag/
├── __init__.py              # Package initialization and exports
├── cli.py                   # CLI entry point (delegates to processing/cli_unified.py)
├── processing/              # Data processing and analysis modules
│   ├── __init__.py
│   ├── base.py              # Abstract base processor classes
│   ├── processors_2d.py     # 2D surface data processing (UXarray/xarray)
│   ├── processors_3d.py     # 3D atmospheric data processing with vertical coords
│   ├── remapping.py         # Grid remapping with xESMF/KDTree
│   ├── cli_unified.py       # Unified CLI (MPASUnifiedCLI + subcommand parsers)
│   ├── parallel.py          # MPI parallel execution manager
│   ├── parallel_wrappers.py # Worker functions for parallel batch dispatch
│   ├── data_cache.py        # Coordinate and data caching for performance
│   ├── constants.py         # Physical constants and configuration defaults
│   ├── utils_config.py      # MPASConfig dataclass with YAML I/O
│   ├── utils_logger.py      # Structured logging utilities
│   ├── utils_monitor.py     # Execution timing and performance profiling
│   ├── utils_validator.py   # Input data validation
│   ├── utils_unit.py        # CF-convention unit conversion
│   ├── utils_metadata.py    # Variable metadata and description extraction
│   ├── utils_file.py        # File discovery and system info
│   ├── utils_geog.py        # Geographic coordinate transformations
│   ├── utils_datetime.py    # Time dimension validation and date/time utilities
│   └── utils_parser.py      # Argument parsing helpers
├── visualization/           # Plotting and visualization tools
│   ├── __init__.py
│   ├── base_visualizer.py   # Abstract MPASVisualizer base class + UnitConverter
│   ├── precipitation.py     # Precipitation accumulation maps
│   ├── surface.py           # 2D scalar surface maps (scatter/contour/pcolormesh)
│   ├── wind.py              # Wind vector plots and overlay support
│   ├── cross_section.py     # 3D vertical cross-section plotting
│   ├── skewt.py             # Skew-T Log-P sounding diagrams (MetPy)
│   └── styling.py           # MPASVisualizationStyle: colorbars, branding, saving
└── diagnostics/             # Atmospheric diagnostic computation modules
    ├── __init__.py
    ├── precipitation.py       # Precipitation accumulation and differencing
    ├── wind.py                # Wind speed, direction, shear diagnostics
    ├── sounding.py            # Vertical sounding extraction + thermodynamic indices
    └── moisture_transport.py  # IWV and IVT computation (pressure-column integrals)

examples/
├── example_precipitation_map.py            # Precipitation accumulation map
├── example_surface_map.py                  # 2D surface variable plot
├── example_wind_map.py                     # Wind vector plot
├── example_3d_variable_slice.py            # 3D variable horizontal slice
├── example_cross_section.py                # Vertical cross-section
├── example_overlay_surface_and_wind.py     # Surface + wind barbs overlay
├── example_overlay_surface_contourf_and_contour.py  # Dual-variable contour overlay
├── example_ivt_map.py                     # IVT magnitude + vector overlay (new)
├── generate_skewt_plot.py                 # Skew-T Log-P sounding diagram (new)
├── generate_complex_weather_map.py        # Multi-panel composite map
└── benchmark.py                           # Performance benchmark script

tests/
├── conftest.py              # Centralized mock fixtures for all test modules
├── test_module_imports.py   # Package-level import and init tests
├── test_data_helpers.py     # Data helper utility tests
├── test_enhancements.py     # Enhancement and edge-case tests
├── cli/                     # CLI command and argument tests
│   ├── test_cli_arguments.py
│   ├── test_cli_analysis.py
│   ├── test_cli_batch.py
│   ├── test_cli_config.py
│   ├── test_cli_entry_point.py
│   ├── test_cli_errors.py
│   ├── test_cli_integration.py
│   ├── test_cli_logging.py
│   ├── test_cli_module.py
│   ├── test_cli_overlay.py
│   ├── test_cli_sounding.py
│   └── test_cli_utilities.py
├── diagnostics/             # Diagnostic module tests
│   ├── test_precipitation_diagnostics.py
│   ├── test_wind_diagnostics.py
│   ├── test_sounding_diagnostics.py
│   ├── test_moisture_transport_diagnostics.py
│   └── test_diagnostics_integration.py
├── processing/              # Processing module tests
│   ├── test_2d_processor.py
│   ├── test_3d_processor_*.py   # Multiple 3D processor coverage files
│   ├── test_data_cache.py
│   ├── test_parallel_*.py       # MPI and multiprocessing wrapper tests
│   ├── test_remapping_*.py      # Grid remapping tests
│   ├── test_unit_conversion.py
│   └── test_utils_*.py          # Utility module tests
├── visualization/           # Visualization module tests
│   ├── test_base_visualizer.py
│   ├── test_cross_section_*.py  # Cross-section plotter coverage files
│   ├── test_surface_*.py        # Surface plotter coverage files
│   ├── test_wind_*.py           # Wind plotter coverage files
│   ├── test_precipitation_plotter.py
│   ├── test_skewt.py
│   └── test_styling_*.py
└── integration/             # End-to-end integration tests
    ├── test_mpas_analysis.py
    └── test_example_with_real_data.py
```

### Class Hierarchy

The package follows a modular, object-oriented architecture with specialized processing, visualization, and diagnostic components:

```
mpasdiag.processing
├── Base Processing
│   ├── MPASBaseProcessor                 # Abstract base processor class
│   ├── MPAS2DProcessor                   # 2D surface field processing
│   └── MPAS3DProcessor                   # 3D atmospheric field processing
│
├── Grid Remapping
│   └── MPASRemapper                      # xESMF/KDTree remapping engine
│
├── Parallel Processing
│   ├── MPASParallelManager               # MPI coordination and load balancing
│   ├── ParallelPrecipitationProcessor    # Parallel precipitation batch workflows
│   ├── ParallelSurfaceProcessor          # Parallel surface variable batch plotting
│   ├── ParallelWindProcessor             # Parallel wind vector batch plotting
│   └── ParallelCrossSectionProcessor     # Parallel cross-section batch generation
│
├── Utilities
│   ├── MPASConfig                  # Configuration dataclass with YAML load/save
│   ├── MPASLogger                  # Structured logging system
│   ├── MPASDataCache               # Intelligent coordinate and data caching
│   ├── PerformanceMonitor          # Execution timing and profiling
│   ├── DataValidator               # Input validation and checks
│   ├── UnitConverter               # CF-compliant unit conversions
│   ├── MPASFileMetadata            # Variable metadata extraction
│   ├── MPASDateTimeUtils           # Time dimension validation and parsing
│   ├── MPASGeographicUtils         # Geographic coordinate transformations
│   ├── FileManager                 # File discovery and system info
│   └── ArgumentParser              # Shared argument parsing helpers
│
└── CLI
    └── MPASUnifiedCLI              # Unified CLI coordinator
                                    # Subcommands: precipitation, surface, wind,
                                    #   cross, sounding, overlay
                                    # Aliases: precip/rain, surf/2d, vector/winds,
                                    #   xsec/3d/vertical, skewt/profile,
                                    #   complex/multi/composite

mpasdiag.visualization
├── Base Visualization
│   └── MPASVisualizer              # Abstract base visualizer with common save/style
│
├── Specialized Plotters
│   ├── MPASPrecipitationPlotter             # Precipitation accumulation maps
│   ├── MPASSurfacePlotter                   # 2D scalar surface field visualization
│   ├── MPASWindPlotter                      # Wind vector plots (barbs/arrows/streamlines)
│   ├── MPASVerticalCrossSectionPlotter      # 3D vertical atmospheric cross-sections
│   └── MPASSkewTPlotter                     # Skew-T Log-P diagrams (requires MetPy)
│                                            #   — dry/moist adiabats, mixing lines
│                                            #   — parcel profile with CAPE/CIN shading
│                                            #   — LCL/LFC/EL level markers
│                                            #   — thermodynamic indices table
│
└── Styling
    └── MPASVisualizationStyle      # Plot styling, colorbars, branding, save helpers

mpasdiag.diagnostics
├── PrecipitationDiagnostics        # Accumulation differencing and precipitation metrics
├── WindDiagnostics                 # Wind speed, direction, shear, component analysis
├── SoundingDiagnostics             # Vertical profile extraction (KDTree nearest-cell)
│                                   # Full MetPy thermodynamic index suite:
│                                   #   SBCAPE/CIN, MLCAPE/CIN, MUCAPE/CIN, DCAPE,
│                                   #   LI, K-Index, TT, SI, CT, PW, WBZ,
│                                   #   0–1/0–6 km bulk shear, 0–1/0–3 km SRH,
│                                   #   STP, SCP, SWEAT; fallback LCL without MetPy
└── MoistureTransportDiagnostics    # Column integrals via trapezoidal rule (dask-aware)
                                    #   IWV  — vertically integrated water vapor (kg m⁻²)
                                    #   IVT_u/IVT_v — eastward/northward moisture flux
                                    #   IVT  — total moisture transport magnitude
                                    #   analyze_moisture_transport() — all-in-one method
```

## Dependencies

### Required
- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- xarray >= 0.19.0
- matplotlib >= 3.5.0
- netCDF4 >= 1.5.0
- PyYAML >= 5.4.0
- dask >= 2021.6.0
- psutil >= 5.8.0

### Optional (Recommended)
- uxarray >= 2024.01.0 (for advanced unstructured grid support)
- cartopy >= 0.20.0 (for cartographic projections)
- metpy >= 1.3.0 (for sounding/Skew-T diagnostics: `SoundingDiagnostics`, `MPASSkewTPlotter`)
- esmpy / xesmf >= 2022.01 (for xESMF-based grid remapping)
- mpi4py >= 3.0 (for MPI-based parallel batch processing)

### Development
- pytest >= 6.0 (for testing)
- pytest-cov >= 3.0 (for coverage reporting)
- pytest-xdist >= 2.0 (for parallel test execution)
- black >= 21.0 (for code formatting)
- flake8 >= 3.8 (for linting)

## Testing

MPASdiag features a comprehensive, modernized test suite with **1800+ tests** across organized sub-packages, achieving **93%+ code coverage**. Tests are organized into four domain sub-packages (`cli/`, `diagnostics/`, `processing/`, `visualization/`) plus top-level module tests and integration tests. MPASdiag uses `pytest` as the testing framework with `pytest-cov` for coverage reporting.

Key test sub-packages:
- **`tests/cli/`**: CLI command parsing, argument handling, batch mode, config loading, sounding, overlay, logging, and integration tests (12 modules)
- **`tests/diagnostics/`**: Precipitation, wind, sounding (`SoundingDiagnostics`), moisture transport (`MoistureTransportDiagnostics`), and integration tests
- **`tests/processing/`**: 2D/3D processors, data cache, parallel MPI/multiprocessing, remapping (core/edge/integration), unit conversion, and utility tests (30+ modules)
- **`tests/visualization/`**: Base visualizer, cross-section plotter (12 coverage files), surface plotter, wind plotter, precipitation plotter, **Skew-T plotter**, and styling tests
- **`tests/integration/`**: End-to-end integration tests with real MPAS data patterns

To run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with detailed coverage report
python -m pytest tests/ -v --cov=mpasdiag --cov-report=term-missing

# Run a specific domain sub-package
python -m pytest tests/diagnostics/ -v
python -m pytest tests/visualization/ -v
python -m pytest tests/processing/ -v
python -m pytest tests/cli/ -v

# Run sounding or IVT/moisture-transport tests specifically
python -m pytest tests/diagnostics/test_sounding_diagnostics.py -v
python -m pytest tests/diagnostics/test_moisture_transport_diagnostics.py -v
python -m pytest tests/cli/test_cli_sounding.py -v
python -m pytest tests/visualization/test_skewt.py -v

# Run in parallel (requires pytest-xdist)
python -m pytest tests/ -n auto
```

## Performance Tips

1. **Use pure xarray backend** for better performance with large datasets:
   ```bash
   mpasdiag precipitation \
     --grid-file grid.nc --data-dir ./data --use-pure-xarray
   ```

2. **Optimize chunking** for memory efficiency:
   ```bash
   mpasdiag precipitation \
     --grid-file grid.nc --data-dir ./data --chunk-size 50000 --parallel
   ```

3. **Process spatial subsets** to reduce memory usage:
   ```bash
   mpasdiag precipitation \
     --grid-file grid.nc --data-dir ./data --variable total \
     --lat-min -10 --lat-max 10 --lon-min 90 --lon-max 120
   ```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'uxarray'**
   ```bash
   conda install -c conda-forge uxarray
   ```

2. **Cartopy projection errors**
   ```bash
   conda install -c conda-forge cartopy
   # Or use --data-type xarray to avoid cartopy dependencies
   ```

3. **Memory errors with large datasets**
   - Use `--data-type xarray` for better memory management
   - Reduce spatial extent with `--lat-min/max --lon-min/max`
   - Process time steps individually instead of batch mode

4. **File not found errors**
   - Verify grid file and data directory paths
   - Use `mpas-validate` to check file accessibility
   - Ensure diagnostic files follow naming pattern: `diag*.nc`

5. **ImportError: MetPy is required for Skew-T diagrams**
   ```bash
   conda install -c conda-forge metpy
   # or
   pip install metpy
   ```
   Without MetPy, `SoundingDiagnostics` still works but only computes a fallback LCL estimate; `MPASSkewTPlotter` requires MetPy.

### Getting Help

1. **Get help for any CLI tool**:
   ```bash
   # Main unified CLI help
   mpasdiag --help
   
   # Specific analysis type help
   mpasdiag precipitation --help
   mpasdiag surface --help
   mpasdiag wind --help
   mpasdiag cross --help
   mpasdiag sounding --help
   ```

2. **Validate your setup**:
   ```bash
   # Test with a simple precipitation analysis
   mpasdiag precipitation \
     --grid-file your_grid.nc --data-dir your_data/ --time-index 0
   ```

3. **Check system information**:
   ```python
   # Import from the processing package
   from mpasdiag.processing import print_system_info
   print_system_info()

   # Or import directly from the module
   from mpasdiag.processing.utils_file import print_system_info
   print_system_info()

   # Or use the class method directly
   from mpasdiag.processing import FileManager
   FileManager.print_system_info()
   ```

4. **Enable verbose output for debugging**:
   ```bash
   # Verbose precipitation analysis
   mpasdiag precipitation \
     --grid-file grid.nc --data-dir ./data --variable total --verbose
   
   # Verbose cross-section analysis
   mpasdiag cross \
     --grid-file grid.nc --data-dir ./data --variable theta \
     --start-lon -105 --start-lat 40 --end-lon -95 --end-lat 40 --verbose
   ```

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Make your changes with tests
4. Run the test suite (`python -m pytest`)
5. Submit a pull request

### Development Setup

```bash
git clone https://github.com/mrixlam/MPASdiag.git
cd MPASdiag
conda env create -f environment.yml
conda activate mpasdiag
pip install -e .[dev]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{iss26mpasdiag,
  author       = {Islam, Mohammad Rubaiat and Liu, Zhiquan},
  title        = {MPASdiag: Data Processing, Visualization and Analysis Toolkit for MPAS from Native Unstructured Mesh},
  month        = apr,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19486544},
  url          = {https://doi.org/10.5281/zenodo.19486544},
}
```

## Acknowledgments

- **NCAR MMM** for MPAS model development and support
- **UXarray Team** for their support with unstructured data processing
- **Pangeo Community** for inspiration on scientific Python package design
- **Contributors** who have helped improve this package

## Contact

- **Author**: Rubaiat Islam
- **Email**: mrislam@ucar.edu
- **Institution**: Mesoscale & Microscale Meteorology Laboratory, NCAR
- **Issues**: Please report bugs and feature requests on GitHub

---

**Version**: 1.0.0  
**Last Updated**: April 12, 2026
