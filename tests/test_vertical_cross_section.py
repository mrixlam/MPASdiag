#!/usr/bin/env python3
"""
MPAS Vertical Cross-Section Module Unit Tests

This module provides comprehensive unit tests for the MPASVerticalCrossSectionPlotter class
which handles creation of vertical atmospheric cross-sections along great circle paths. These
tests validate path generation, interpolation algorithms, level selection, and input validation
using synthetic data to isolate logic from heavy data dependencies.

Tests Performed:
    - test_vertical_cross_section_plotter_initialization: Verifies plotter initialization with default and custom parameters (figsize, dpi)
    - test_great_circle_path_generation: Validates great circle path calculation between geographic points with distance computation
    - test_default_levels_generation: Tests automatic contour level generation for various data types (temperature, wind, constant, NaN)
    - test_interpolation_along_path: Validates spatial interpolation of grid data along cross-section paths
    - test_input_validation: Tests error handling for invalid processor inputs and parameter validation

Test Coverage:
    - Plotter initialization: figure size, DPI settings, default state management
    - Great circle geometry: path generation, distance calculations, endpoint validation
    - Contour level algorithms: automatic level selection for different variable types and ranges
    - Spatial interpolation: grid-to-path data mapping, handling of irregular grids
    - Input validation: type checking, error handling for invalid configurations

Testing Approach:
    Unit tests using synthetic NumPy arrays and mock data to validate computational geometry,
    interpolation algorithms, and plotting logic without requiring actual MPAS output files.
    Tests verify method signatures, return types, array shapes, and mathematical properties
    (monotonicity, bounds checking) without full rendering.

Expected Results:
    - Great circle paths connect specified endpoints with monotonically increasing distances
    - Contour levels span data range appropriately for various meteorological variables
    - Interpolated values along paths are non-trivial (not all NaN) for valid inputs
    - Invalid inputs raise appropriate ValueError exceptions with descriptive messages
    - Plotter objects initialize correctly with specified or default parameters

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import sys
import os
import numpy as np
import math
import pytest
from typing import cast, Any
from pathlib import Path

package_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_dir))

from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter


def test_vertical_cross_section_plotter_initialization() -> None:
    """
    Verify MPASVerticalCrossSectionPlotter instantiation with default and custom configuration parameters for figure dimensions and resolution. This test validates that plotter objects initialize correctly with default settings including 10×12 figure size and 100 DPI resolution suitable for vertical cross-section displays. Custom parameter testing confirms that user-specified values for figure dimensions (10×6) and DPI (150) properly override defaults. The test also verifies that figure and axes objects remain uninitialized (None) until actual plotting operations commence. This initialization validation ensures that plotter configuration works correctly before computationally expensive cross-section generation operations.

    Parameters:
        None

    Returns:
        None
    """
    plotter = MPASVerticalCrossSectionPlotter()
    
    assert plotter.figsize == (10, 12)
    assert plotter.dpi == 100  
    assert plotter.fig is None
    assert plotter.ax is None
    
    custom_plotter = MPASVerticalCrossSectionPlotter(figsize=(10, 6), dpi=150)
    assert custom_plotter.figsize == (10, 6)
    assert custom_plotter.dpi == 150


def test_great_circle_path_generation() -> None:
    """
    Validate great circle path calculation between geographic endpoints with accurate distance computation and coordinate interpolation. This test confirms that the path generation method produces 11 evenly-spaced points along the great circle connecting start (-100°W, 40°N) and end (-90°W, 40°N) coordinates. The test verifies that returned longitude, latitude, and distance arrays have correct lengths matching the requested number of points. Endpoint validation ensures that first and last coordinates match specified start/end points within numerical tolerance (0.01°). Distance array verification confirms monotonically increasing values starting from zero and ending at positive total distance reflecting the computed great circle arc length.

    Parameters:
        None

    Returns:
        None
    """
    plotter = MPASVerticalCrossSectionPlotter()
    
    start_point = (-100.0, 40.0)
    end_point = (-90.0, 40.0)
    num_points = 11
    
    lons, lats, distances = plotter._generate_great_circle_path(start_point, end_point, num_points)
    
    assert len(lons) == num_points
    assert len(lats) == num_points
    assert len(distances) == num_points
    
    assert math.isclose(lons[0], start_point[0], abs_tol=0.01)
    assert math.isclose(lats[0], start_point[1], abs_tol=0.01)
    assert math.isclose(lons[-1], end_point[0], abs_tol=0.01)
    assert math.isclose(lats[-1], end_point[1], abs_tol=0.01)
    
    assert np.all(np.diff(distances) >= 0)
    assert math.isclose(distances[0], 0.0, abs_tol=1e-6)
    assert distances[-1] > 0.0
    
    print("Great circle path generation test passed!")


def test_default_levels_generation() -> None:
    """
    Verify automatic contour level generation for diverse data types including temperature, wind, constant, and NaN arrays. This test validates that the level generation algorithm produces appropriate contour intervals spanning the data range for temperature values (250-330K) and wind data including negative values (-15 to 25 m/s). The algorithm must handle edge cases including constant-value arrays (all 5.0) generating at least one level and NaN-filled arrays producing valid level sets. Level bounds verification ensures that minimum levels don't exceed data minimum and maximum levels encompass data maximum values. This adaptive level generation supports flexible visualization across diverse meteorological variables without requiring manual level specification.

    Parameters:
        None

    Returns:
        None
    """
    plotter = MPASVerticalCrossSectionPlotter()
    
    temp_data = np.array([[250, 260, 270], [280, 290, 300], [310, 320, 330]])
    temp_levels = plotter._get_default_levels(temp_data, 'temperature')
    
    assert len(temp_levels) > 0
    assert temp_levels.min() <= temp_data.min()
    assert temp_levels.max() >= temp_data.max()
    
    wind_data = np.array([[-10, -5, 0], [5, 10, 15], [-15, 20, 25]])
    wind_levels = plotter._get_default_levels(wind_data, 'uwind')
    
    assert len(wind_levels) > 0
    assert wind_levels.min() <= wind_data.min()
    assert wind_levels.max() >= wind_data.max()
    
    constant_data = np.full((3, 3), 5.0)
    constant_levels = plotter._get_default_levels(constant_data, 'constant')
    
    assert len(constant_levels) >= 1
    
    nan_data = np.full((3, 3), np.nan)
    nan_levels = plotter._get_default_levels(nan_data, 'nan_data')
    
    assert len(nan_levels) > 0
    
    print("Default levels generation test passed!")


def test_interpolation_along_path() -> None:
    """
    Validate spatial interpolation of irregular grid data onto cross-section path points using nearest-neighbor or linear methods. This test creates synthetic grid with 5 points spanning longitude (-102° to -98°) and latitude (39° to 43°) with corresponding data values (10-50). The path consists of 3 intermediate points requiring interpolation from the surrounding grid locations. The test verifies that interpolated values have correct array length (3 points) and contain non-trivial results (not all NaN) indicating successful spatial mapping. The test gracefully handles scipy import failures by skipping when interpolation dependencies are unavailable rather than failing the test suite.

    Parameters:
        None

    Returns:
        None
    """
    plotter = MPASVerticalCrossSectionPlotter()
    
    grid_lons = np.array([-102, -101, -100, -99, -98])
    grid_lats = np.array([39, 40, 41, 42, 43])
    grid_data = np.array([10, 20, 30, 40, 50])
    
    path_lons = np.array([-101.5, -100.5, -99.5])
    path_lats = np.array([39.5, 40.5, 41.5])
    
    try:
        interpolated = plotter._interpolate_along_path(
            grid_lons, grid_lats, grid_data, path_lons, path_lats
        )
        
        assert len(interpolated) == len(path_lons)
        assert not np.all(np.isnan(interpolated))  
        
        print("Interpolation along path test passed!")
        
    except ImportError:
        print("Scipy not available, skipping interpolation test")
        pytest.skip("Scipy not available for interpolation test")


def test_input_validation() -> None:
    """
    Verify error handling for invalid processor inputs raising descriptive ValueError exceptions during cross-section creation attempts. This test confirms that the plotter correctly rejects invalid processor objects (string instead of MPAS3DProcessor) by raising ValueError with informative error messages mentioning expected type. The validation prevents cryptic downstream failures by catching type errors at the method entry point with clear diagnostic information. Error message content verification ensures that exception text references MPAS3DProcessor type helping users identify the correct input requirement. This defensive input validation approach improves code robustness and user experience by providing immediate actionable feedback for configuration errors.

    Parameters:
        None

    Returns:
        None
    """
    plotter = MPASVerticalCrossSectionPlotter()
    try:
        plotter.create_vertical_cross_section(
            mpas_3d_processor=cast(Any, "invalid"),
            var_name="temperature",
            start_point=(-100, 40),
            end_point=(-90, 40)
        )
        assert False, "Should have raised ValueError for invalid processor"
    except ValueError as e:
        assert "MPAS3DProcessor" in str(e)
        print("Input validation test passed!")


def run_all_tests() -> bool:
    """
    Execute complete test suite for MPASVerticalCrossSectionPlotter functionality with consolidated success reporting and error handling. This function coordinates execution of five test functions covering initialization, great circle geometry, level generation, interpolation, and input validation. The test runner prints formatted progress messages with separator lines for visual clarity and captures any exceptions during test execution. Success status returns as boolean True when all tests pass or False if any test raises exceptions. Exception handling includes full traceback printing for debugging failed tests while maintaining clean output format. This orchestration function enables both command-line test execution and programmatic test suite integration.

    Parameters:
        None

    Returns:
        bool: True if all tests pass successfully, False if any test fails or raises exceptions.
    """
    print("Testing MPASVerticalCrossSectionPlotter...")
    print("=" * 50)
    
    try:
        test_vertical_cross_section_plotter_initialization()
        test_great_circle_path_generation()
        test_default_levels_generation()
        test_interpolation_along_path()
        test_input_validation()
        
        print("=" * 50)
        print("All tests passed successfully! ✓")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)