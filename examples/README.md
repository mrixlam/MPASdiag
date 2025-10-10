# MPAS Analysis Examples

This directory contains comprehensive examples demonstrating how to use the MPAS Analysis toolkit for processing and visualizing MPAS (Model for Prediction Across Scales) model output with enhanced unit conversion and professional meteorological plotting.

## ✨ New in v1.1.0
All examples now feature:
- **Automatic Unit Conversion**: Temperature (K→°C), Pressure (Pa→hPa), Humidity (kg/kg→g/kg)
- **Enhanced Scientific Notation**: Proper formatting for extreme values (vorticity, mixing ratios)
- **Professional Meteorological Standards**: Industry-standard plotting conventions
- **Comprehensive Metadata**: Enhanced variable information and proper labeling

## 📋 Overview

The MPAS Analysis toolkit now provides comprehensive visualization capabilities:

1. **Comprehensive Demo** (`comprehensive_demo.py`) - **NEW**: Complete v1.1.0 feature demonstration
2. **Temperature & Thermodynamics** (`temperature_examples.py`) - **NEW**: Complete temperature analysis
3. **Pressure & Atmospheric Dynamics** (`pressure_examples.py`) - **NEW**: Comprehensive pressure analysis
4. **Humidity & Moisture Variables** (`humidity_examples.py`) - **NEW**: Complete moisture analysis
5. **Precipitation Analysis** (`precipitation_examples.py`) - **ENHANCED**: Updated with unit conversion
6. **Surface Variable Analysis** (`surface_examples.py`) - **ENHANCED**: Updated with unit conversion
7. **Wind Vector Analysis** (`wind_examples.py`) - **ENHANCED**: Updated with unit conversion
8. **850 hPa Composite Analysis** (`composite_850hPa_example.py`) - **NEW**: Advanced multi-variable plotting

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

## 📚 Detailed Example Descriptions

### 1. **Comprehensive Demo** (`comprehensive_demo.py`) 🚀
**NEW in v1.1.0** - Start here to see all enhancements!

**Features:**
- Complete unit conversion system demonstration
- Scientific notation formatting examples
- Before/after comparison plots
- Feature summary with practical examples
- All conversion types in action

**Key Demonstrations:**
- Temperature: K ↔ °C ↔ °F conversions
- Pressure: Pa ↔ hPa ↔ mb ↔ inHg conversions
- Humidity: kg/kg ↔ g/kg, % ↔ fraction conversions
- Wind: m/s ↔ kts ↔ mph ↔ km/h conversions
- Precipitation: m ↔ mm ↔ in ↔ cm conversions
- Scientific notation for extreme values

### 2. **Temperature & Thermodynamics** (`temperature_examples.py`) 🌡️
**NEW in v1.1.0** - Complete temperature analysis with automatic K→°C conversion

**Features:**
- 2-meter temperature with enhanced metadata
- Multi-level temperature comparisons (surface, 850hPa, 500hPa)
- Dewpoint temperature analysis
- Temperature depression calculations (T - Td)
- Unit conversion demonstrations

**Key Variables:**
- t2m, tsk, th2m (surface temperatures)
- temperature_850hPa, temperature_500hPa (upper-air)
- dewpoint_surface, dewpoint_850hPa (moisture analysis)

### 3. **Pressure & Atmospheric Dynamics** (`pressure_examples.py`) 🌪️
**NEW in v1.1.0** - Comprehensive pressure analysis with Pa→hPa conversion

**Features:**
- Mean sea level pressure analysis
- Multi-level pressure visualization
- Vorticity analysis (scientific notation demo)
- Geopotential height analysis
- Pressure gradient calculations

**Key Variables:**
- mslp, psfc (surface pressure)
- pressure_850hPa, pressure_500hPa, pressure_200hPa
- vorticity_850hPa (demonstrates scientific notation)
- geopotential_500hPa (height analysis)

### 4. **Humidity & Moisture Variables** (`humidity_examples.py`) 💧
**NEW in v1.1.0** - Complete moisture analysis with kg/kg→g/kg conversion

**Features:**
- Specific humidity with enhanced units
- Relative humidity analysis
- Multi-level humidity comparisons
- Cloud condensation visualization (qc, qi, qr, qs, qg)
- Precipitable water analysis
- Unit conversion demonstration plots

**Key Variables:**
- q2, qv (specific humidity)
- rh2m (relative humidity)
- qc, qi, qr, qs, qg (hydrometeors)
- precipw, pwat (precipitable water)

### 5. **Surface Variable Analysis** (`surface_examples.py`) 🌍
**ENHANCED in v1.1.0** - Updated with automatic unit conversion

**Features:**
- Temperature, pressure, humidity analysis
- Scatter plots and contour visualizations
- Enhanced unit conversions (K→°C, Pa→hPa, kg/kg→g/kg)
- Multi-variable comparisons
- Professional meteorological plotting

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

### 6. **Precipitation Analysis** (`precipitation_examples.py`) 🌧️
**ENHANCED in v1.1.0** - Updated with automatic unit conversion

**Features:**
- Precipitation data processing and visualization
- Different precipitation variables and accumulation types
- Enhanced contour and filled contour plotting
- Automatic unit conversion (m→mm, mm→in)
- Professional precipitation analysis

**Key Variables:**
- rainnc, rainc (accumulated precipitation)
- Different accumulation periods (1h, 3h, 6h, 24h)
- Enhanced with proper mm labeling

### 7. **Wind Vector Analysis** (`wind_examples.py`) 💨
**ENHANCED in v1.1.0** - Updated with wind speed unit conversion

**Features:**
- Meteorological wind barbs and arrow vectors
- Multiple atmospheric levels (surface, 850mb, 500mb, etc.)
- Background wind speed visualization with unit conversion
- High-resolution regional analysis
- Enhanced with m/s ↔ kts ↔ mph ↔ km/h conversions

**Examples:**
- Surface wind barbs with proper speed units
- Upper-level wind arrows (850mb) 
- Wind vectors with converted speed background
- Multi-level wind analysis with consistent units

### 8. **850 hPa Composite Analysis** (`composite_850hPa_example.py`) 🌪️
**NEW in v1.1.0** - Advanced multi-variable composite plotting

**Features:**
- Professional composite meteorological analysis
- Specific humidity shading (kg/kg→g/kg conversion)
- Wind barbs with speed conversion (m/s→kts)
- Geopotential height contours with proper labeling
- Multi-variable overlay techniques
- Publication-ready visualization

**Key Components:**
- **Specific Humidity**: Moisture transport visualization (shaded background)
- **Wind Barbs**: Atmospheric circulation patterns (red barbs)
- **Geopotential Height**: Synoptic-scale features (black contours)
- **Professional Styling**: Meteorological standards with legends

**Use Cases:**
- Moisture transport analysis
- Synoptic pattern identification
- Weather system tracking
- Operational forecasting
- Research and publication plots
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

## 🎯 Key Benefits of v1.1.0

### Automatic Unit Conversion
- **No more manual conversions**: Temperature automatically converts K→°C
- **Meteorological standards**: Pressure in hPa, humidity in g/kg
- **Consistent labeling**: All plots use proper meteorological units
- **Error reduction**: Eliminates unit conversion mistakes

### Enhanced Scientific Notation
- **Extreme value handling**: Proper formatting for vorticity (~1e-5 s⁻¹)
- **Readable tick labels**: Scientific notation for values < 1e-3 or >= 1e4
- **Professional appearance**: Publication-ready formatting

### Professional Plotting
- **Industry standards**: Follows meteorological plotting conventions
- **Enhanced metadata**: Comprehensive variable information
- **Proper colormaps**: Meteorologically appropriate color schemes
- **Publication ready**: High-quality plots suitable for reports/papers

### Comprehensive Testing
- **103 total tests**: Including 13 new unit conversion tests
- **Robust validation**: All conversion types thoroughly tested
- **Error handling**: Comprehensive error checking and warnings

## 🚀 Getting Started with New Examples

### 1. Start with Comprehensive Demo
```bash
cd examples/
python comprehensive_demo.py
```
This demonstrates all v1.1.0 features with synthetic data.

### 2. Run Specific Variable Examples
```bash
# Temperature analysis
python temperature_examples.py

# Pressure and dynamics
python pressure_examples.py

# Humidity and moisture
python humidity_examples.py
```

### 3. Try Advanced Composite Plotting
```bash
# Multi-variable composite analysis
python composite_850hPa_example.py
```
This demonstrates professional meteorological analysis with multiple overlaid variables.

### 4. Update Your Data Paths
Edit the configuration section in each script:
```python
grid_file = "/path/to/your/grid.nc"
data_dir = "/path/to/your/diagnostic/files/"
output_dir = "/path/to/output/directory/"
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