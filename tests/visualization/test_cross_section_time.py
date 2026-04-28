#!/usr/bin/env python3
"""
MPASdiag Test Suite: Test MPASVerticalCrossSectionPlotter Functionality

This test suite validates the core functionality of the MPASVerticalCrossSectionPlotter class, which is responsible for generating vertical cross-section visualizations from MPAS model output. The tests cover key aspects including plotter initialization with default and custom parameters, great circle path generation between geographic endpoints, automatic contour level generation for various data types, spatial interpolation of irregular grid data onto cross-section paths, and robust input validation for processor objects. Each test function asserts expected behaviors and handles edge cases to ensure the plotter operates correctly under diverse conditions. The suite also includes a comprehensive test runner that executes all tests with clear reporting of successes and failures.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import pytest
import matplotlib
matplotlib.use('Agg')


from tests.visualization.cross_section_test_helpers import (
    check_great_circle_path,
)


def test_great_circle_path_generation() -> None:
    """
    This test verifies the correctness of the great circle path generation method in the MPASVerticalCrossSectionPlotter class. The test defines a start point at (-100.0, 40.0) and an end point at (-90.0, 40.0) with a specified number of points (11) along the path. It asserts that the generated longitude and latitude arrays have the correct length, that the first and last points match the specified start and end coordinates within a reasonable tolerance, and that the distance array is monotonically increasing with the first distance being zero and the last distance being greater than zero. This ensures that the great circle path is generated correctly between the two geographic points.

    Parameters:
        None

    Returns:
        None
    """
    check_great_circle_path()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
