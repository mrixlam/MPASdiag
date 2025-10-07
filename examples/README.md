# MPAS Analysis Examples

This directory contains comprehensive examples demonstrating how to use the MPAS Analysis toolkit for processing and visualizing MPAS (Model for Prediction Across Scales) model output.

## 📋 Overview

The MPAS Analysis toolkit provides three main visualization capabilities:

1. **Precipitation Analysis** (`precipitation_examples.py`) - Analysis and visualization of precipitation variables
2. **Surface Variable Analysis** (`surface_examples.py`) - Temperature, pressure, humidity, and other surface variables
3. **Wind Vector Analysis** (`wind_examples.py`) - Wind barbs, arrows, and multi-level wind analysis

## 🚀 Quick Start

### Prerequisites

1. **Install the MPAS Analysis package:**
   ```bash
   cd /path/to/mpasdiag
   pip install -e .
   ```

2. **Prepare your data:**
   ```
   examples/
   ├── data/
   │   ├── grid.nc              # MPAS grid file
   │   └── diagnostics/         # Directory with diagnostic files
   │       ├── diag.2018-04-18_00.00.00.nc
   │       ├── diag.2018-04-18_01.00.00.nc
   │       └── ...
   ```

3. **Run the examples:**
   ```bash
   cd examples/
   python precipitation_examples.py
   python surface_examples.py
   python wind_examples.py
   ```

## 📊 Example Categories

### 1. Precipitation Analysis (`precipitation_examples.py`)

**Features:**
- Multiple accumulation periods (1h, 3h, 6h, 24h)
- Convective vs non-convective precipitation comparison
- High-resolution regional analysis
- Custom color scales and contour levels

**Examples:**
- Basic precipitation plot with default settings
- Multi-accumulation period comparison
- Convective vs non-convective analysis
- High-resolution precipitation mapping

**Usage:**
```bash
python precipitation_examples.py
```

**Console Script Alternative:**
```bash
# Basic precipitation plot
mpas-precip-plot --grid-file data/grid.nc --data-dir data/diagnostics/ --output-dir output/precip

# Custom spatial extent
mpas-precip-plot --grid-file data/grid.nc --data-dir data/diagnostics/ --lat-min -9.8 --lat-max 12.2 --lon-min 91 --lon-max 113 --output-dir output/precip

# Batch processing all time steps
mpas-precip-plot --grid-file data/grid.nc --data-dir data/diagnostics/ --batch-all --output-dir output/precip
```

### 2. Surface Variable Analysis (`surface_examples.py`)

**Features:**
- Temperature, pressure, humidity analysis
- Scatter plots and contour visualizations
- Unit conversions (K to °C, Pa to hPa, etc.)
- Multi-variable comparisons

**Examples:**
- 2-meter temperature analysis (scatter and contour)
- Surface pressure with custom contour levels
- Humidity analysis with automatic unit conversion
- Wind speed calculation from components
- Multi-variable comparison grid

**Usage:**
```bash
python surface_examples.py
```

**Console Script Alternative:**
```bash
# Temperature scatter plot
mpas-surface-plot --grid-file data/grid.nc --data-dir data/diagnostics/ --variable t2m --plot-type scatter --lat-min -9.8 --lat-max 12.2 --lon-min 91 --lon-max 113 --output-dir output/surface

# Pressure contour plot
mpas-surface-plot --grid-file data/grid.nc --data-dir data/diagnostics/ --variable surface_pressure --plot-type contour --output-dir output/surface --grid-resolution-deg 0.03

# Wind speed with custom colormap
mpas-surface-plot --grid-file data/grid.nc --data-dir data/diagnostics/ --variable wspd10 --colormap plasma --output-dir output/surface --batch-all
```

### 3. Wind Vector Analysis (`wind_examples.py`)

**Features:**
- Meteorological wind barbs and arrow vectors
- Multiple atmospheric levels (surface, 850mb, 500mb, etc.)
- Background wind speed visualization
- High-resolution regional analysis
- Time series evolution

**Examples:**
- Surface wind barbs
- Upper-level wind arrows (850mb)
- Wind vectors with speed background
- High-resolution regional analysis
- Multi-level wind comparison
- Time series wind evolution

**Usage:**
```bash
python wind_examples.py
```

**Console Script Alternative:**
```bash
# Surface wind barbs
mpas-wind-plot --grid_file data/grid.nc --data_dir data/diagnostics/ --u-variable u10 --v-variable v10 --wind-plot-type barbs --output-dir output/wind

# 850mb wind arrows
mpas-wind-plot --grid_file data/grid.nc --data_dir data/diagnostics/ --u-variable u850 --v-variable v850 --wind-plot-type arrows --output-dir output/wind

# Wind with background speed
mpas-wind-plot --grid_file data/grid.nc --data_dir data/diagnostics/ --u-variable u10 --v-variable v10 --show-background --background-colormap viridis --batch-all
```

## 🔧 Customization Options

### Spatial Extent
All examples support custom spatial extents:
```python
config = MPASConfig(
    lat_min=-10.0, lat_max=15.0,
    lon_min=91.0, lon_max=113.0
)
```

### Time Selection
Choose specific time indices:
```python
config = MPASConfig(
    time_index=12  # 12th time step
)
```

### Visualization Options
Customize plots with:
```python
config = MPASConfig(
    colormap="plasma",
    figure_size=(14, 10),
    dpi=300,
    output_formats=['png', 'pdf', 'svg']
)
```

### Wind-Specific Options
For wind plots:
```python
config = MPASConfig(
    wind_plot_type="barbs",        # or "arrows"
    subsample_factor=5,            # Vector density
    show_background=True,          # Speed background
    background_colormap="viridis"  # Background colors
)
```

## 📁 Output Structure

Examples create organized output directories:

```
output/
├── basic_precipitation_plot.png
├── precipitation_a01h.png
├── precipitation_a03h.png
├── high_resolution_precipitation.pdf
├── surface/
│   ├── temperature_scatter.png
│   ├── temperature_contour.png
│   ├── surface_pressure.pdf
│   └── humidity_analysis.png
└── wind/
    ├── surface_wind_barbs.pdf
    ├── 850mb_wind_arrows.pdf
    ├── wind_with_background.png
    └── time_series/
        ├── wind_evolution_00.png
        ├── wind_evolution_03.png
        └── ...
```

## 📚 API Reference

### Key Classes

**MPASDataProcessor**: Data loading and processing
```python
processor = MPASDataProcessor(grid_file="grid.nc", verbose=True)
processor.load_data(data_dir="diagnostics/", pattern="diag*.nc")
data = processor.get_variable_data("t2m", time_index=0)
u_data, v_data = processor.get_wind_components("u10", "v10", time_index=0)
```

**MPASVisualizer**: Plot creation and visualization
```python
visualizer = MPASVisualizer(figsize=(12, 10), dpi=300)
fig, ax = visualizer.create_precipitation_map(lon, lat, data, ...)
fig, ax = visualizer.create_wind_plot(lon, lat, u_data, v_data, ...)
visualizer.save_plot(output_file, formats=['png', 'pdf'])
```

**MPASConfig**: Configuration management
```python
config = MPASConfig(
    grid_file="grid.nc",
    data_dir="diagnostics/",
    output_dir="output/",
    lat_min=-10, lat_max=15,
    lon_min=91, lon_max=113
)
```

## 🛠️ Troubleshooting

### Common Issues

1. **FileNotFoundError**: Check that grid.nc and diagnostic files exist
2. **Variable not found**: Use `processor.get_available_variables()` to see available variables
3. **Memory issues**: Use smaller spatial extents or subsample data
4. **Import errors**: Ensure mpas-analysis package is installed: `pip install -e .`

### Data Requirements

**Grid File**: Must contain MPAS mesh information
- `lonCell`, `latCell`: Cell center coordinates
- Grid connectivity information

**Diagnostic Files**: Must contain requested variables
- Precipitation: `rainc`, `rainnc`, or computed `total`
- Surface: `t2m`, `surface_pressure`, `q2`, etc.
- Wind: `u10`, `v10`, `u850`, `v850`, etc.
- Time dimension: `Time` or `time`

### Performance Tips

1. **Spatial filtering**: Use smaller lat/lon extents for faster processing
2. **Time selection**: Process specific time indices rather than all data
3. **Subsampling**: Use higher subsample factors for wind vectors
4. **File patterns**: Use specific patterns to load only needed files

## 🤝 Contributing

To add new examples:

1. Follow the existing example structure
2. Include comprehensive docstrings
3. Add error handling for missing variables/files
4. Provide both programmatic and console script examples
5. Update this README with the new functionality

## 📄 License

This software is provided under the same license as the MPAS Analysis toolkit.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the main MPAS Analysis documentation
3. Check variable names using `processor.get_available_variables()`
4. Examine the console script help: `mpas-precip-plot --help`