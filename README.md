# MPASdiag - MPAS Diagnostic Analysis Package

A comprehensive Python package for analyzing and visualizing MPAS (Model for Prediction Across Scales) unstructured mesh model output data. This package provides tools for data processing, visualization, and analysis of MPAS atmospheric model simulations including **precipitation analysis**, **surface variable plotting**, and **wind vector visualization**.

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-beta-yellow.svg)

## Features

### 🌧️ Precipitation Analysis
- **Multiple Variables**: Convective, non-convective, and total precipitation
- **Accumulation Periods**: 1h, 3h, 6h, 12h, 24h analysis capabilities
- **Advanced Visualization**: Professional cartographic presentation with customizable colormaps

### 🌡️ Surface Variable Analysis  
- **Temperature & Pressure**: 2m temperature, surface pressure analysis
- **Humidity & Wind Speed**: Specific humidity, wind speed calculations
- **Flexible Plotting**: Scatter plots and interpolated contour visualizations

### 💨 Wind Vector Analysis
- **Meteorological Standards**: Wind barbs and arrow vector representations
- **Multi-Level Support**: Surface, 850mb, 500mb, 200mb atmospheric levels
- **Background Visualization**: Optional wind speed background with customizable colormaps

### 🔧 Technical Features
- **Comprehensive Data Processing**: Load and process MPAS unstructured grid data using both UXarray and xarray backends
- **Professional Visualization**: Create publication-quality maps with customizable cartographic projections
- **Batch Processing**: Efficient processing of multiple time steps with memory optimization
- **Flexible Configuration**: YAML-based configuration files and comprehensive command-line interfaces
- **Performance Monitoring**: Built-in performance tracking and optimization tools
- **Extensive Testing**: Comprehensive unit test suite with detailed examples
- **Command-Line Tools**: Multiple specialized CLI tools for different analysis types

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

## Quick Start

### Command Line Usage

The package provides specialized command-line tools for different types of analysis:

#### 🌧️ Precipitation Analysis
```bash
# Basic precipitation analysis
mpas-precip-plot --grid-file grid.nc --data-dir ./data --output-dir ./output

# Custom spatial extent (Maritime Continent example)
mpas-precip-plot --grid-file grid.nc --data-dir ./data \
  --lat-min -4 --lat-max 5 --lon-min 98 --lon-max 110 \
  --var rainnc --accum a01h --output-dir ./output

# Batch process all time steps
mpas-precip-plot --grid-file grid.nc --data-dir ./data --batch-all
```

#### 🌡️ Surface Variable Analysis
```bash
# Temperature scatter plot
mpas-surface-plot --grid-file grid.nc --data-dir ./data \
  --variable t2m --plot-type scatter

# Pressure contour plot with custom extent
mpas-surface-plot --grid-file grid.nc --data-dir ./data \
  --variable surface_pressure --plot-type contour \
  --lat-min -10 --lat-max 15 --lon-min 95 --lon-max 110

# Wind speed with custom colormap
mpas-surface-plot --grid-file grid.nc --data-dir ./data \
  --variable wspd10 --colormap plasma --time-index 12
```

#### 💨 Wind Vector Analysis  
```bash
# Surface wind barbs
mpas-wind-plot /path/to/grid.nc /path/to/data \
  --u-variable u10 --v-variable v10 --wind-plot-type barbs

# 850mb wind arrows with background speed
mpas-wind-plot /path/to/grid.nc /path/to/data \
  --u-variable u850 --v-variable v850 --wind-plot-type arrows \
  --show-background --background-colormap viridis

# Custom extent and high resolution
mpas-wind-plot /path/to/grid.nc /path/to/data \
  --u-variable u10 --v-variable v10 \
  --extent -105 -95 35 45 --subsample 3

# Batch processing all time steps
mpas-analyze --grid-file grid.nc --data-dir ./data --batch-all --output-dir ./output

# High-resolution output
mpas-analyze --grid-file grid.nc --data-dir ./data \
  --dpi 400 --formats png pdf --output-dir ./output

# Validate data before processing
mpas-validate --grid-file grid.nc --data-dir ./data
```

### Python API Usage

```python
from mpas_analysis import MPASDataProcessor, MPASVisualizer, MPASConfig

# Configure analysis parameters
config = MPASConfig(
    grid_file="maritime_continent_3km.init.nc",
    data_dir="./diagnostic_files",
    output_dir="./output",
    lat_min=-4.0, lat_max=5.0,
    lon_min=98.0, lon_max=110.0,
    variable="rainnc",
    accumulation_period="a01h"
)

# Initialize processor and load data
processor = MPASDataProcessor(config.grid_file)
dataset, data_type = processor.load_data(config.data_dir)

# Extract spatial coordinates
lon, lat = processor.extract_spatial_coordinates()

# Compute precipitation for a specific time step
time_index = 10
precip_data = processor.compute_precipitation_difference(time_index, config.variable)

# Create visualization
visualizer = MPASVisualizer(figsize=(12, 8), dpi=300)
fig, ax = visualizer.create_precipitation_map(
    lon, lat, precip_data.values,
    config.lon_min, config.lon_max,
    config.lat_min, config.lat_max,
    title="MPAS Precipitation Analysis"
)

# Save the plot
visualizer.save_plot("./output/precipitation_map", formats=["png", "pdf"])
```

## Configuration Files

Create YAML configuration files for repeatable analysis:

```yaml
# config.yaml
grid_file: "/path/to/maritime_continent_3km.init.nc"
data_dir: "/path/to/diagnostic_files"
output_dir: "./analysis_output"

# Spatial extent (Maritime Continent)
lat_min: -4.0
lat_max: 5.0
lon_min: 98.0
lon_max: 110.0

# Analysis parameters
variable: "total"  # rainc + rainnc
accumulation_period: "a01h"
batch_mode: true

# Visualization settings
figure_width: 14.0
figure_height: 10.0
dpi: 400
output_formats: ["png", "pdf"]
colormap: "default"

# Processing options
use_pure_xarray: true
verbose: true
parallel: false
```

Use with command line:
```bash
mpas-analyze --config config.yaml
```

## Supported Variables

- **rainc**: Convective precipitation
- **rainnc**: Non-convective precipitation  
- **total**: Total precipitation (rainc + rainnc)

## Supported Accumulation Periods

- **a01h**: 1-hour accumulation
- **a03h**: 3-hour accumulation
- **a06h**: 6-hour accumulation
- **a12h**: 12-hour accumulation
- **a24h**: 24-hour accumulation

## Examples

The package includes comprehensive examples for all analysis types:

- **precipitation_examples.py**: 🌧️ Complete precipitation analysis workflows
- **surface_examples.py**: 🌡️ Temperature, pressure, and surface variables  
- **wind_examples.py**: 💨 Wind vector analysis with barbs and arrows

Run examples:
```bash
cd examples/
python precipitation_examples.py  # 4 precipitation scenarios
python surface_examples.py        # 5 surface variable examples 
python wind_examples.py           # 6 wind vector examples
```

See [examples/README.md](examples/README.md) for detailed descriptions and console script alternatives.

## Sample Plots

This repository now includes a `sample_plot/` directory containing example output images generated
by the package. These are useful for quickly previewing the style and layout of the visualizations
without running the full plotting pipeline.

Contents:
- `sample_plot/precipitation/` - Example precipitation map PNGs
- `sample_plot/surface/` - Example surface variable plots (scatter/contour)
- `sample_plot/wind/` - Example wind vector visualizations (barbs/arrows)

How to view:

1. From the repository root, open the images with your preferred image viewer. For macOS Preview, run:

```bash
open sample_plot/**/*.png
```

2. Or inspect them in the terminal using `ls` and `open`:

```bash
ls -R sample_plot
open sample_plot/precipitation/*.png
```

3. To regenerate sample plots from code (requires dependencies and data):

```bash
# Run any of the example scripts which will create plots in the configured output dir
python examples/precipitation_examples.py
python examples/surface_examples.py
python examples/wind_examples.py
```

Additions like `sample_plot/` are intended to help reviewers and users quickly
see expected outputs. 

## Architecture

```
mpas_analysis/
├── data_processing.py    # Core data loading and processing
├── visualization.py      # Plotting and visualization tools
├── utils.py             # Configuration, logging, and utilities
├── cli.py               # Command-line interfaces
└── __init__.py          # Package initialization

examples/
├── precipitation_examples.py  # Comprehensive precipitation workflows
├── surface_examples.py        # Surface variable analysis examples
├── wind_examples.py           # Wind vector plotting examples
└── README.md                  # Detailed example documentation

tests/
├── test_mpas_analysis.py      # Comprehensive unit test suite
└── __init__.py               # Test package initialization
└── __init__.py              # Test runner
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
   mpas-analyze --data-type xarray
   ```

2. **Optimize chunking** for memory efficiency:
   ```python
   config = MPASConfig(chunk_size=50000, parallel=True)
   ```

3. **Process spatial subsets** to reduce memory usage:
   ```bash
   mpas-analyze --lat-min -10 --lat-max 10 --lon-min 90 --lon-max 120
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

1. **Validate your setup**:
   ```bash
   mpas-validate --grid-file your_grid.nc --data-dir your_data/
   ```

2. **Check system information**:
   ```python
   from mpas_analysis import print_system_info
   print_system_info()
   ```

3. **Enable verbose output**:
   ```bash
   mpas-analyze --verbose --grid-file grid.nc --data-dir ./data
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
**Last Updated**: October 2025