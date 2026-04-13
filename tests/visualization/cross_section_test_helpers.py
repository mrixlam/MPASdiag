#!/usr/bin/env python3
"""
MPASdiag Test Suite: Shared helpers for cross-section plotter tests.

This module contains shared helper functions and constants used across multiple test modules for the MPASVerticalCrossSectionPlotter class. These helpers include functions to check proper initialization of the plotter, validate the generation of great circle paths, verify default contour level calculations, test interpolation along paths, and ensure input validation. By centralizing these common checks, we can maintain consistency across tests and reduce code duplication when testing various aspects of the cross-section plotting functionality. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import os
import math
import pytest
import numpy as np
from typing import cast, Any

from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter


TEST_DATA_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data'
)

GRID_FILE: str = os.path.join(TEST_DATA_DIR, 'grids', 'x1.10242.static.nc')
MPASOUT_DIR: str = os.path.join(TEST_DATA_DIR, 'u240k', 'mpasout')


def check_plotter_initialization() -> None:
    """
    This test verifies that the MPASVerticalCrossSectionPlotter class initializes its attributes correctly. By creating instances of the plotter with default and custom parameters, the test checks that the figsize and dpi attributes are set as expected, and that the fig and ax attributes are initialized to None. This ensures that the plotter's constructor is functioning properly and that it can be instantiated with both default and user-defined settings for figure size and resolution.

    Parameters:
        None

    Returns:
        None
    """
    plotter = MPASVerticalCrossSectionPlotter()

    assert plotter.figsize == (pytest.approx(10), pytest.approx(12))
    assert plotter.dpi == pytest.approx(100)
    assert plotter.fig is None
    assert plotter.ax is None

    custom_plotter = MPASVerticalCrossSectionPlotter(figsize=(10, 6), dpi=150)
    assert custom_plotter.figsize == (pytest.approx(10), pytest.approx(6))
    assert custom_plotter.dpi == pytest.approx(150)


def check_great_circle_path() -> None:
    """
    This test verifies that the `_generate_great_circle_path` method in the MPASVerticalCrossSectionPlotter class correctly generates a great circle path between two geographic points. By providing specific start and end coordinates along with a defined number of points, the test checks that the output longitude, latitude, and distance arrays have the expected length and values. It ensures that the first and last points of the generated path match the requested start and end coordinates within a reasonable tolerance. Additionally, it verifies that the distance array is monotonically non-decreasing and starts at zero, confirming that the path generation logic is functioning as intended for accurate cross-sectional plotting along great circle routes. 

    Parameters:
        None

    Returns:
        None
    """
    plotter = MPASVerticalCrossSectionPlotter()

    start_point = (-100.0, 40.0)
    end_point = (-90.0, 40.0)
    num_points = 11

    lons, lats, distances = plotter._generate_great_circle_path(
        start_point, end_point, num_points
    )

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


def check_default_levels() -> None:
    """
    This test verifies that the `_get_default_levels` method in the MPASVerticalCrossSectionPlotter class correctly calculates default contour levels for different types of data. By providing sample data arrays representing temperature-like, wind-like, constant, and all-NaN data, the test checks that the generated levels are appropriate for each case. For temperature and wind data, it ensures that the levels cover the range of the data values. For constant data, it verifies that at least one level is generated. For all-NaN data, it checks that a default set of levels is still produced. This ensures that the plotter can handle a variety of input data scenarios and generate meaningful contour levels for visualization. 

    Parameters:
        None

    Returns:
        None
    """
    plotter = MPASVerticalCrossSectionPlotter()

    temp_data = np.array([[250, 260, 270], [280, 290, 300], [310, 320, 330]])
    temp_levels = plotter._get_default_levels(temp_data, 'theta')
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


def check_interpolation_along_path() -> None:
    """
    This test verifies that the `_interpolate_along_path` method in the MPASVerticalCrossSectionPlotter class correctly performs interpolation of grid data along a specified path. By providing known grid coordinates, data values, and path points, the test checks that the output interpolated values have the expected length and contain valid (non-NaN) results. This ensures that the interpolation logic is functioning as intended, allowing for accurate extraction of cross-sectional data along arbitrary paths. 

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
    except ImportError:
        pytest.skip("Scipy not available for interpolation test")


def check_input_validation() -> None:
    """
    This test verifies that the `create_vertical_cross_section` method in the MPASVerticalCrossSectionPlotter class properly validates its inputs. By passing an invalid type for the `mpas_3d_processor` argument, the test checks that the method raises a ValueError with an appropriate error message, ensuring that users are informed about incorrect usage of the method and the requirement for a valid MPAS3DProcessor instance.

    Parameters:
        None

    Returns:
        None
    """
    plotter = MPASVerticalCrossSectionPlotter()
    try:
        plotter.create_vertical_cross_section(
            mpas_3d_processor=cast(Any, "invalid"),
            var_name="theta",
            start_point=(-100, 40),
            end_point=(-90, 40)
        )
        assert False, "Should have raised ValueError for invalid processor"
    except ValueError as e:
        assert "MPAS3DProcessor" in str(e)
