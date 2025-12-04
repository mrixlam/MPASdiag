# MPASdiag - MPAS Diagnostic Analysis Package

A comprehensive Python package for analyzing and visualizing MPAS (Model for Prediction Across Scales) unstructured mesh model output data with support for both serial and parallel processing workflows. This package provides specialized tools for atmospheric model diagnostics including precipitation accumulation analysis with configurable periods, 2D surface variable visualization using scatter plots and filled contours, horizontal wind vector plotting with barbs and arrows, and 3D vertical atmospheric cross-sections along arbitrary paths. The toolkit features automatic unit conversion following CF conventions, professional meteorological styling, memory-efficient data processing through lazy loading and chunking strategies, and extensible architecture with modular processors and plotters. Advanced capabilities include powerful remapping from unstructured MPAS meshes to regular latitude-longitude grids using xESMF and SciPy, MPI-based parallel batch processing for time series generation, comprehensive command-line interface with YAML configuration support, and flexible vertical coordinate systems (pressure, height, model levels) for 3D analysis. The package is designed for operational meteorology applications, climate model evaluation, and research workflows requiring publication-quality visualizations from MPAS atmospheric simulations.

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-beta-yellow.svg)

## Installation

### Prerequisites

- Python 3.8 or higher
- NetCDF4-compatible system libraries

### From GitHub Repository (Recommended)

```bash
# Clone the repository
git clone https://github.com/mrixlam/MPASdiag.git
cd MPASdiag

# Create a new conda environment
conda create -n mpas-analysis python=3.9
conda activate mpas-analysis

# Install dependencies
conda install -c conda-forge numpy pandas xarray matplotlib cartopy netcdf4 dask pyyaml psutil

# Install UXarray (for unstructured grid support)
conda install -c conda-forge uxarray

# Install the package
pip install -e .
```

### Using pip (Alternative)

```bash
# Clone the repository
git clone https://github.com/mrixlam/MPASdiag.git
cd MPASdiag

# Create virtual environment
python -m venv mpas-analysis-env
source mpas-analysis-env/bin/activate  # On Windows: mpas-analysis-env\Scripts\activate

# Install the package
pip install -r requirements.txt
pip install -e .
```

## Python API Usage

The package exposes a small, focused programmatic API in the `mpasdiag` package. Below are common classes and a short example showing the typical workflow.

### Key classes
- `mpasdiag.processing.utils_config.MPASConfig` — configuration container for CLI and programmatic runs with YAML support
- `mpasdiag.processing.processors_2d.MPAS2DProcessor` — load and extract 2D (surface) fields from MPAS output with UXarray/xarray backends
- `mpasdiag.processing.processors_3d.MPAS3DProcessor` — load and extract 3D atmospheric fields with vertical coordinate handling
- `mpasdiag.processing.remapping.MPASRemapper` — remap unstructured MPAS data to regular grids using xESMF
- `mpasdiag.processing.parallel.MPASParallelManager` — MPI-based parallel execution manager for batch processing
- `mpasdiag.processing.data_cache.MPASDataCache` — intelligent caching for coordinates and frequently accessed data
- `mpasdiag.diagnostics.precipitation.PrecipitationDiagnostics` — compute precipitation accumulation and diagnostics
- `mpasdiag.diagnostics.wind.WindDiagnostics` — compute wind speed, direction, and derived quantities
- `mpasdiag.visualization.precipitation.MPASPrecipitationPlotter` — create professional precipitation accumulation maps
- `mpasdiag.visualization.surface.MPASSurfacePlotter` — surface scatter/contour maps with automatic unit conversion
- `mpasdiag.visualization.wind.MPASWindPlotter` — wind vector visualizations (barbs/arrows/streamlines)
- `mpasdiag.visualization.cross_section.MPASVerticalCrossSectionPlotter` — vertical atmospheric cross sections

### Minimal example

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

If you prefer the command line, see the `CLI Examples` section below.

#### Precipitation Analysis
```bash
# Basic precipitation analysis (single time)
mpasdiag precipitation \
  --grid-file grid.nc --data-dir ./data --variable total --accumulation a01h --plot-type contourf

# Batch process all time steps
mpasdiag precipitation \
  --grid-file grid.nc --data-dir ./data --variable total --batch-all --dpi 300 --figure-size 12 12

# Batch processing with parallel execution (multiprocessing backend)
mpasdiag precipitation \
  --grid-file grid.nc --data-dir ./data --variable total --batch-all --parallel --workers 4
```

#### Surface Variable Analysis
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

#### Wind Vector Analysis  
```bash
# Surface wind barbs
mpasdiag wind \
  --grid-file grid.nc --data-dir ./data \
  --u-variable u10 --v-variable v10 --wind-plot-type barbs

# Batch processing of wind vectors
mpasdiag wind \
  --grid-file grid.nc --data-dir ./data \
  --u-variable u10 --v-variable v10 --batch-all

# Parallel batch processing
mpasdiag wind \
  --grid-file grid.nc --data-dir ./data \
  --u-variable u10 --v-variable v10 --batch-all --parallel
```

#### 3D Vertical Cross-Section Analysis
```bash
# Temperature cross-section from Denver to Kansas City  
mpasdiag cross \
  --grid-file grid.nc --data-dir ./data \
  --variable theta --start-lon -105.0 --start-lat 39.7 \
  --end-lon -94.6 --end-lat 39.1 --vertical-coord pressure

# Wind cross-section with custom colormap
mpasdiag cross \
  --grid-file grid.nc --data-dir ./data \
  --variable uReconstructZonal --start-lon -110 --start-lat 35 \
  --end-lon -80 --end-lat 45 --colormap RdBu_r

# Custom cross-section with specific time and high resolution
mpasdiag cross \
  --grid-file grid.nc --data-dir ./data \
  --variable theta --start-lon -100 --start-lat 30 \
  --end-lon -90 --end-lat 50 --time-index 12 \
  --num-points 100 --dpi 400 --formats png pdf
```

## Configuration Files

Create YAML configuration files for repeatable analysis:

```yaml
# config.yaml - Comprehensive MPAS Analysis Configuration
grid_file: "./data/grids/x1.10242.init.nc"
data_dir: "./data/u240k"
output_dir: "./output"

# Spatial extent (Maritime Continent)
lat_min: -4.0
lat_max: 5.0
lon_min: 98.0
lon_max: 110.0

# Precipitation analysis parameters
variable: "total"                 # total = rainc + rainnc (recommended)
accumulation_period: "a01h"       # 1-hour accumulation
batch_mode: true                  # Process all time steps

# Visualization settings
figure_width: 14.0
figure_height: 10.0
dpi: 400
output_formats: ["png", "pdf"]
colormap: "default"               # Auto-selected based on variable

# Advanced options
title: "MPAS Maritime Continent Analysis"
time_index: 0                     # Specific time step (ignored if batch_mode: true)
figure_size: [12.0, 10.0]         # Custom figure dimensions

# Processing options
use_pure_xarray: false            # Use UXarray for better unstructured grid support
verbose: true
parallel: false                   # Parallel processing
chunk_size: 100000                # Data chunk size for memory optimization
```

Use with unified command line:
```bash
# Use configuration file with precipitation analysis
mpasdiag precipitation --config config.yaml

# Override specific settings
mpasdiag precipitation --config config.yaml \
  --variable total --accumulation a06h --dpi 600
```

## Architecture

### File Structure
```
mpasdiag/
├── __init__.py              # Package initialization and exports
├── cli.py                   # Legacy CLI interface
├── processing/              # Data processing and analysis modules
│   ├── __init__.py
│   ├── base.py              # Base processor classes
│   ├── processors_2d.py     # 2D surface data processing
│   ├── processors_3d.py     # 3D atmospheric data processing
│   ├── remapping.py         # Grid remapping with xESMF/KDTree
│   ├── cli_unified.py       # Unified command-line interface
│   ├── parallel.py          # MPI parallel execution manager
│   ├── parallel_wrappers.py # Parallel processing wrappers
│   ├── data_cache.py        # Data caching for performance
│   ├── constants.py         # Physical and configuration constants
│   ├── utils_config.py      # Configuration management
│   ├── utils_logger.py      # Logging utilities
│   ├── utils_monitor.py     # Performance monitoring
│   ├── utils_validator.py   # Data validation
│   ├── utils_unit.py        # Unit conversion
│   ├── utils_metadata.py    # Metadata handling
│   ├── utils_file.py        # File operations
│   ├── utils_geog.py        # Geographic calculations
│   ├── utils_datetime.py    # Date/time utilities
│   └── utils_parser.py      # Argument parsing
├── visualization/           # Plotting and visualization tools
│   ├── __init__.py
│   ├── base_visualizer.py   # Base visualization class
│   ├── precipitation.py     # Precipitation plotting
│   ├── surface.py           # Surface variable plotting
│   ├── wind.py              # Wind vector plotting
│   ├── cross_section.py     # Vertical cross-section plotting
│   └── styling.py           # Plot styling utilities
└── diagnostics/             # Diagnostic computation modules
    ├── __init__.py
    ├── precipitation.py     # Precipitation diagnostics
    └── wind.py              # Wind diagnostics

examples/
├── generate_complex_weather_map.py         # Multi-variable composite plots
├── generate_surface_plots.py               # Surface variable examples
├── generate_vertical_cross_section.py      # 3D cross-section examples
└── remap_mpas_data.py                      # Grid remapping examples

tests/
├── test_*.py                # Comprehensive unit test suite
└── __init__.py              # Test package initialization
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
│   ├── ParallelPrecipitationProcessor    # Parallel precipitation workflows
│   ├── ParallelSurfaceProcessor          # Parallel surface plotting
│   └── ParallelCrossSectionProcessor     # Parallel cross-section generation
│
├── Utilities
│   ├── MPASConfig                  # Configuration management with YAML
│   ├── MPASLogger                  # Structured logging system
│   ├── MPASDataCache               # Intelligent data caching
│   ├── PerformanceMonitor          # Execution timing and profiling
│   ├── DataValidator               # Input validation and checks
│   ├── UnitConverter               # CF-compliant unit conversions
│   ├── MPASFileMetadata            # Variable metadata extraction
│   ├── FileManager                 # File discovery and operations
│   └── GeographicUtils             # Coordinate transformations
│
└── CLI
    └── MPASUnifiedCLI              # Command-line interface coordinator

mpasdiag.visualization
├── Base Visualization
│   └── MPASVisualizer              # Abstract base visualizer class
│
├── Specialized Plotters
│   ├── MPASPrecipitationPlotter             # Precipitation accumulation maps
│   ├── MPASSurfacePlotter                   # Surface scalar field visualization
│   ├── MPASWindPlotter                      # Wind vector plots (barbs/arrows/streamlines)
│   └── MPASVerticalCrossSectionPlotter      # 3D atmospheric cross-sections
│
└── Styling
    └── PlotStyleManager            # Consistent plot styling and branding

mpasdiag.diagnostics
├── PrecipitationDiagnostics        # Accumulation and precipitation metrics
└── WindDiagnostics                 # Wind speed, direction, and derivatives
```

## Dependencies

### Required
- numpy >= 1.20.0
- pandas >= 1.3.0
- xarray >= 0.19.0
- matplotlib >= 3.5.0
- netCDF4 >= 1.5.0
- PyYAML >= 5.4.0
- dask >= 2021.6.0
- psutil >= 5.8.0

### Optional (Recommended)
- uxarray >= 2024.01.0 (for advanced unstructured grid support)
- cartopy >= 0.20.0 (for cartographic projections)

### Development
- pytest >= 6.0 (for testing)
- black >= 21.0 (for code formatting)
- flake8 >= 3.8 (for linting)

## Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=mpas_analysis --cov-report=html

# Run specific test module
python -m pytest tests/test_data_processing.py -v

# Run tests with the included test runner
cd tests/
python __init__.py
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
   ```

2. **Validate your setup**:
   ```bash
   # Test with a simple precipitation analysis
   mpasdiag precipitation \
     --grid-file your_grid.nc --data-dir your_data/ --time-index 0
   ```

3. **Check system information**:
   ```python
   from mpas_analysis import print_system_info
   print_system_info()
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
conda activate mpas-analysis-dev
pip install -e .[dev]

# Run tests to verify installation
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{islam2025mpas,
  title={MPASdiag: Python Package for MPAS Model Output Analysis and Visualization},
  author={Islam, Rubaiat},
  year={2025},
  institution={Mesoscale \& Microscale Meteorology Laboratory, NCAR},
  url={https://github.com/mrixlam/MPASdiag}
}
```

## Acknowledgments

- **NCAR MMM Lab** for MPAS model development and support
- **UXarray Team** for unstructured grid data handling
- **Pangeo Community** for inspiration on scientific Python package design
- **Contributors** who have helped improve this package

## Contact

- **Author**: Rubaiat Islam
- **Email**: mrislam@ucar.edu
- **Institution**: Mesoscale & Microscale Meteorology Laboratory, NCAR
- **Issues**: Please report bugs and feature requests on GitHub

---

**Version**: 1.0.0  
**Last Updated**: December 4, 2025