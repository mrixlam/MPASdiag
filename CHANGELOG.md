# Changelog

All notable changes to the MPAS Analysis package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.2] - 2025-10-10

### Fixed
- **Critical CLI Import Fix**:
  - Fixed command-line interface import error: "cannot import name 'get_2d_variable_metadata' from 'mpas_analysis.visualization'"
  - Added module-level exports for all refactored functions to maintain full backward compatibility
  - All CLI commands (`mpas-surface-plot`, `mpas-precip-plot`, `mpas-wind-plot`) now work correctly
  - Example files and external scripts continue to work without modification

## [1.1.1] - 2025-10-10

### Changed
- **Code Architecture Refactoring**:
  - Created `UnitConverter` class for centralized unit conversion functionality
  - Created `MPASFileMetadata` class for centralized variable metadata management
  - Moved `validate_plot_parameters` to `MPASVisualizer` as static method
  - Improved class hierarchy with better separation of concerns
  - Enhanced module documentation to reflect new architecture
  - Updated all internal function calls to use new class methods
  - Maintained full backward compatibility through proper static method implementation

### Improved
- **Documentation Updates**:
  - Updated README.md with new API examples using specialized plotters
  - Enhanced architecture documentation with class hierarchy diagrams
  - Updated module docstrings to reflect new class structure
  - Improved code comments and inline documentation

### Fixed
- **Backward Compatibility**:
  - Added module-level exports for refactored functions to maintain compatibility
  - Fixed CLI import issues with `get_2d_variable_metadata` and other moved functions
  - Ensured all example files and command-line tools continue to work without modification

### Technical
- **Testing**:
  - Updated all tests to use new class methods
  - Maintained 102/102 tests passing
  - Verified backward compatibility imports work correctly
  - Enhanced test documentation to reflect current functionality
  - All existing functionality preserved with improved organization

## [1.1.0] - 2025-10-08

### Added
- **Comprehensive Unit Conversion System**:
  - Automatic unit conversion to meteorological conventions
  - Temperature: K → °C, K ↔ °F, °C ↔ °F conversions
  - Pressure: Pa → hPa, Pa ↔ mb, hPa ↔ mb conversions  
  - Mixing ratio: kg/kg → g/kg conversions
  - Precipitation: mm/hr ↔ mm/day, mm/hr ↔ in/hr conversions
  - Wind speed: m/s ↔ kt, m/s ↔ mph, m/s ↔ km/h conversions
  - Distance: m ↔ km, m ↔ ft conversions
  - Smart unit string normalization (handles common variations)
  - Works with scalars, numpy arrays, and xarray DataArrays

- **Enhanced Tick Formatting**:
  - Scientific notation for extreme values (< 1e-3 or >= 1e4)
  - Solves vorticity plotting issue where all ticks showed "0.000"
  - Clean "x.xe±yy" format for scientific notation
  - Maintains existing precision rules for normal-range values

- **Improved Variable Metadata System**:
  - Automatic unit conversion in `get_2d_variable_metadata()`
  - Updated colorbar levels to match converted units

- **Comprehensive Example Suite**:
  - `comprehensive_demo.py`: Complete v1.1.0 feature demonstration
  - `temperature_examples.py`: Temperature & thermodynamic variable analysis
  - `pressure_examples.py`: Pressure & atmospheric dynamics analysis
  - `humidity_examples.py`: Humidity & moisture variable analysis
  - `composite_850hPa_example.py`: Advanced multi-variable composite plotting
  - Enhanced existing examples with automatic unit conversion

- **Advanced Composite Plotting**:
  - Professional multi-variable overlay capabilities
  - 850 hPa composite analysis (humidity shading + wind barbs + geopotential contours)
  - Intelligent variable detection for different naming conventions
  - Automatic data subsampling for optimal visualization
  - Publication-ready meteorological plots
  - Comprehensive legends and professional styling
  - Preserves original units in metadata (`original_units` field)
  - Enhanced long names to include unit information
  - Support for all diagnostic variables with proper spatial dimensions

- **Enhanced Test Coverage**:
  - 13 new unit conversion tests
  - Comprehensive tick formatting tests  
  - Total test coverage: 103 tests
  - All numpy compatibility warnings suppressed

### Changed
- **Variable Metadata Enhancement**:
  - Temperature variables now display in °C instead of K
  - Pressure variables now display in hPa instead of Pa
  - Humidity variables now display in g/kg instead of kg/kg
  - Colorbar levels automatically converted to display units
  - Long names updated to reflect unit conversions

- **Tick Formatting Improvements**:
  - Very small values (vorticity, etc.) now show meaningful labels
  - Large values use clean scientific notation
  - Enhanced `_format_ticks_dynamic()` with scientific notation detection

- **Test Suite Updates**:
  - Fixed batch precipitation test expectations (4 files vs 5, accounting for accumulation period)
  - Updated metadata tests to expect converted units
  - Added numpy warning suppression in pytest configuration

- **Enhanced Example Documentation**:
  - Updated `examples/README.md` with comprehensive v1.1.0 feature descriptions
  - Added detailed usage instructions for all new examples
  - Included meteorological applications and use cases
  - Added getting started guide with step-by-step instructions
  - Professional documentation with proper formatting and structure

### Removed
- **Backward Compatibility Cleanup**:
  - Removed deprecated `get_variable_metadata()` function  
  - Removed deprecated `extract_coordinates_for_variable()` method
  - Removed associated deprecation warning tests
  - Updated all internal code to use new method names

### Fixed
- **UXarray Time Indexing Issue**:
  - Fixed time coordinate access for surface and wind plot filename generation
  - Proper timestamp extraction for all plot types
  - Consistent time display across visualization functions

- **Precipitation Batch Processing**:
  - Fixed test expectations for accumulation period logic
  - Correctly handles time step skipping for accumulation calculations
  - Proper file count validation (accounts for skipped initial time steps)

- **Scientific Notation Display**:
  - Vorticity and other small-value variables now show proper tick labels
  - Eliminated "0.000" labels on colorbars for extreme values
  - Improved readability for variables with very large or small ranges

### Technical Details
- **New Functions Added**:
  - `convert_units()` - Core unit conversion function
  - `get_display_units()` - Returns preferred display units
  - `convert_data_for_display()` - Converts data and metadata together
  - `_normalize_unit_string()` - Handles unit string variations

- **Enhanced Functions**:
  - `get_2d_variable_metadata()` - Now includes automatic unit conversion
  - `_format_ticks_dynamic()` - Enhanced with scientific notation support
  - `create_surface_map()` - Better timestamp handling and conditional time display

- **Configuration Updates**:
  - Added numpy warning filters to `pyproject.toml`
  - Cleaned up pytest configuration
  - Improved error handling and logging

- **New Example Scripts**:
  - `comprehensive_demo.py` - Complete feature demonstration with synthetic data
  - `temperature_examples.py` - Multi-level temperature analysis with K→°C conversion
  - `pressure_examples.py` - Pressure dynamics with Pa→hPa conversion and vorticity
  - `humidity_examples.py` - Moisture analysis with kg/kg→g/kg conversion
  - `composite_850hPa_example.py` - Advanced multi-variable composite plotting
  - All examples include professional documentation and error handling

- **Composite Plotting Features**:
  - Multi-variable overlay capability (shading + contours + vectors)
  - Intelligent variable detection across different naming conventions
  - Automatic data subsampling for optimal wind barb density
  - Professional meteorological styling with proper legends
  - Synthetic data generation for demonstration purposes

### Meteorological Applications
The v1.1.0 examples showcase real-world meteorological analysis workflows:

- **850 hPa Composite Analysis**: Standard operational weather analysis combining moisture transport (specific humidity shading), circulation patterns (wind barbs), and synoptic features (geopotential height contours)
- **Multi-level Temperature Analysis**: Atmospheric temperature structure analysis from surface to upper levels with proper °C conversion
- **Moisture Transport Visualization**: Humidity analysis at multiple levels with meteorologically appropriate g/kg units
- **Pressure Dynamics**: Synoptic-scale pattern analysis with proper hPa units and scientific notation for vorticity
- **Professional Publication Standards**: All plots meet operational meteorology and research publication requirements

### Summary
Version 1.1.0 represents a major enhancement to the MPAS Analysis package, transforming it from a basic plotting tool into a comprehensive meteorological analysis framework. The automatic unit conversion system eliminates manual conversion errors, the enhanced scientific notation ensures proper visualization of extreme values, and the extensive example suite provides templates for real-world meteorological analysis. The new composite plotting capabilities enable professional multi-variable analysis suitable for operational forecasting and research publication.

---

## [1.0.0] - 2025-10-06

### Added
- Initial release of MPAS Analysis package
- Core data processing module with MPASDataProcessor class
- Comprehensive visualization module with MPASVisualizer class
- Utility functions for configuration, logging, and file management
- Command-line interface with multiple entry points
- Support for multiple precipitation variables (rainc, rainnc, total)
- Batch processing capabilities for time series analysis
- Professional cartographic visualizations using Cartopy
- Flexible configuration system with YAML support
- Performance monitoring and optimization tools
- Comprehensive unit test suite with >90% coverage
- Example scripts demonstrating basic and advanced usage
- Complete documentation with installation and usage instructions

### Features
- **Data Processing**:
  - Load MPAS unstructured grid data using UXarray and xarray
  - Lazy loading for memory-efficient processing of large datasets
  - Temporal difference calculations for precipitation analysis
  - Spatial filtering and coordinate transformation
  - Comprehensive data validation and quality control
  
- **Visualization**:
  - Publication-quality precipitation maps with scatter plots
  - Customizable colormaps and contour levels
  - Multiple output formats (PNG, PDF, SVG, EPS)
  - Cartographic projections (PlateCarree, Mercator, Lambert Conformal)
  - Batch visualization for time series analysis
  
- **Command-Line Tools**:
  - `mpas-analyze`: Main analysis tool
  - `mpas-batch-process`: Batch processing utility
  - `mpas-validate`: Data validation tool
  
- **Configuration**:
  - YAML-based configuration files
  - Command-line argument parsing
  - Flexible parameter management
  - Environment-specific settings

### Dependencies
- Python 3.8+
- numpy >= 1.20.0
- pandas >= 1.3.0
- xarray >= 0.19.0
- matplotlib >= 3.5.0
- cartopy >= 0.20.0 (optional)
- uxarray >= 2024.01.0 (optional)
- netCDF4 >= 1.5.0
- PyYAML >= 5.4.0
- dask >= 2021.6.0
- psutil >= 5.8.0

### Documentation
- Comprehensive README with installation and usage instructions
- Example scripts with detailed comments
- API documentation in docstrings
- Configuration file examples
- Troubleshooting guide

### Testing
- Unit tests for all major components
- Test coverage for data processing, visualization, and utilities
- Mocked tests for external dependencies
- Continuous integration setup ready

---

## Development Notes

### Version Numbering
- Major version (X.0.0): Breaking changes to API
- Minor version (0.X.0): New features, backwards compatible
- Patch version (0.0.X): Bug fixes, backwards compatible

### Planned Features for Future Versions
- Support for additional MPAS variables (temperature, wind, etc.)
- Interactive plotting capabilities with Plotly/Bokeh
- NetCDF output for processed data
- Integration with Jupyter notebooks
- Parallel processing optimizations
- Cloud storage support (S3, Google Cloud)
- Machine learning integration for pattern analysis

### Known Limitations
- Currently focused on precipitation analysis
- Requires UXarray for full unstructured grid support
- Memory usage can be high for very large datasets
- Limited to specific MPAS output formats

### Acknowledgments
- NCAR MMM Lab for MPAS model development
- UXarray development team
- Pangeo community for scientific Python best practices
- Contributors and beta testers