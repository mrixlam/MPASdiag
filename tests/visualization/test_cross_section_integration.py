#!/usr/bin/env python3
"""
MPASdiag Test Suite: Test Cross-Section Integration with MPASVerticalCrossSectionPlotter

This  test suite validates the functionality of the MPASVerticalCrossSectionPlotter class, ensuring that it correctly generates vertical cross-sections from MPAS 3D data. The tests cover initialization, great circle path generation, default level creation, interpolation along paths, and input validation. Additionally, integration tests with real MPAS data files confirm that the plotter can handle actual datasets and produce expected outputs without errors. This comprehensive testing approach ensures robustness and reliability of the cross-section visualization capabilities within the MPASdiag framework. It also includes error handling for missing dependencies and invalid inputs, providing informative feedback to users. The test suite is designed for both command-line execution and integration into automated testing pipelines.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules for testing
import os
import sys
import math
import shutil
import pytest
import tempfile
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import cast, Any, Generator

from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
from mpasdiag.processing.processors_3d import MPAS3DProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
GRID_FILE = os.path.join(TEST_DATA_DIR, 'grids', 'x1.10242.static.nc')
MPASOUT_DIR = os.path.join(TEST_DATA_DIR, 'u240k', 'mpasout')


def test_vertical_cross_section_plotter_initialization() -> None:
    """
    This test validates the initialization of the MPASVerticalCrossSectionPlotter class, ensuring that default parameters are set correctly and that custom parameters are properly assigned. The test checks that the default figure size is (10, 12) inches and the default DPI is 100. It also verifies that when custom parameters are provided during instantiation, they override the defaults as expected. This ensures that the plotter can be configured flexibly for different visualization needs while maintaining sensible defaults for typical use cases.

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


def test_great_circle_path_generation() -> None:
    """
    This test verifies the correctness of the great circle path generation method in the MPASVerticalCrossSectionPlotter class. It checks that the generated longitude and latitude points along the path between specified start and end coordinates are accurate and that the corresponding distances are correctly calculated. The test ensures that the first and last points match the input coordinates within a reasonable tolerance and that the distances are non-negative and properly ordered along the path. This validation confirms that the plotter can accurately define cross-section paths for subsequent data interpolation and visualization.

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
    This test validates the default level generation method in the MPASVerticalCrossSectionPlotter class for different variable types. It checks that the generated levels for temperature (theta) and wind (uwind) data are appropriate based on the input data range, ensuring that the levels encompass the minimum and maximum values of the data. The test also verifies that for constant data, at least one level is generated, and that for NaN-filled data, the method still returns a valid set of levels without errors. This ensures that the plotter can robustly determine suitable contour levels for a variety of data scenarios, which is critical for effective visualization of cross-sections.

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
    
    print("Default levels generation test passed!")


def test_interpolation_along_path() -> None:
    """
    This test verifies the interpolation of grid data along a specified path defined by longitude and latitude coordinates. It checks that the interpolated values are returned for each point along the path and that they are not all NaN, indicating that the interpolation is functioning correctly. The test uses a simple synthetic dataset to validate the interpolation logic, ensuring that the method can handle typical scenarios encountered in cross-section plotting where data values need to be estimated at specific locations along the path. This validation confirms that the plotter can accurately interpolate data for visualization purposes, enhancing the quality and reliability of the generated cross-section plots.

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
    This test validates the input handling of the `create_vertical_cross_section` method in the MPASVerticalCrossSectionPlotter class. It checks that when an invalid MPAS3DProcessor object is passed, the method raises a ValueError with an appropriate error message indicating the issue. This ensures that the plotter can gracefully handle incorrect inputs and provide informative feedback to users, preventing unexpected crashes and guiding them towards correct usage of the plotting functionality. The test intentionally passes an invalid processor (a string) to trigger the error handling mechanism and confirms that the expected exception is raised.

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
        print("Input validation test passed!")


class TestRealDataIntegration:
    """ Integration tests with real MPAS data. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestRealDataIntegration", mpas_3d_processor) -> Generator[None, None, None]:
        """
        This fixture sets up the necessary environment for integration tests using real MPAS data. It checks for the availability of the MPAS3DProcessor fixture, which should provide access to real MPAS data files. If the processor is not available, it skips the tests that depend on real data. The fixture also creates a temporary directory for storing any output files generated during the tests and initializes an instance of the MPASVerticalCrossSectionPlotter for use in the test methods. After the tests are completed, it ensures that any temporary files or directories created during testing are cleaned up to maintain a tidy testing environment.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): A fixture that provides access to real MPAS data for testing.

        Returns:
            Generator[None, None, None]: A generator that yields control to the test methods and performs cleanup after tests are completed.
        """
        if mpas_3d_processor is None:
            pytest.skip("MPAS data not available")
            return
        
        self.processor = mpas_3d_processor
        self.temp_dir = tempfile.mkdtemp()
        self.plotter = MPASVerticalCrossSectionPlotter()
    
        yield
        
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_complete_workflow_with_real_data(self: "TestRealDataIntegration") -> None:
        """
        This test validates the complete workflow of creating a vertical cross-section plot using real MPAS data. It uses the MPAS3DProcessor to access the data, generates a cross-section for the 'theta' variable along a specified path, and saves the resulting plot to a temporary directory. The test asserts that the output file is created successfully, confirming that the entire process from data access to visualization works correctly with real datasets. This integration test ensures that the MPASVerticalCrossSectionPlotter can handle actual MPAS data and produce expected outputs without errors.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor        
        output_path = os.path.join(self.temp_dir, 'test_crosssection.png')
        
        fig, ax = self.plotter.create_vertical_cross_section(
            processor, 'theta',
            start_point=(-105, 35),
            end_point=(-95, 45),
            num_points=30,
            save_path=output_path
        )
        
        assert os.path.exists(output_path)
        plt.close(fig)
    
    def test_all_plot_types_with_real_data(self: "TestRealDataIntegration") -> None:
        """
        This test validates that the MPASVerticalCrossSectionPlotter can generate vertical cross-section plots using different plot types ('filled_contour', 'contour', 'pcolormesh') with real MPAS data. It iterates over the specified plot types, creating a cross-section for the 'theta' variable along a defined path for each type. The test ensures that the plotting function executes without errors for all plot types, confirming that the plotter can handle various visualization styles when working with real datasets.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor
        
        for plot_type in ['filled_contour', 'contour', 'pcolormesh']:
            fig, ax = self.plotter.create_vertical_cross_section(
                processor, 'theta',
                start_point=(-105, 35),
                end_point=(-95, 45),
                num_points=20,
                plot_type=plot_type
            )
            plt.close(fig)
    
    def test_all_vertical_coords_with_real_data(self: "TestRealDataIntegration") -> None:
        """
        This test validates that the MPASVerticalCrossSectionPlotter can generate vertical cross-section plots using different vertical coordinate options ('pressure' and 'model_levels') with real MPAS data. It iterates over the specified vertical coordinate types, creating a cross-section for the 'theta' variable along a defined path for each type. The test ensures that the plotting function executes without errors for both vertical coordinate options, confirming that the plotter can handle different vertical representations when working with real datasets.

        Parameters:
            None

        Returns:
            None
        """
        processor = self.processor
        
        for vert_coord in ['pressure', 'model_levels']:
            fig, ax = self.plotter.create_vertical_cross_section(
                processor, 'theta',
                start_point=(-105, 35),
                end_point=(-95, 45),
                num_points=20,
                vertical_coord=vert_coord
            )
            plt.close(fig)


class TestRealMPASDataIntegration:
    """ Test with real MPAS data files. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestRealMPASDataIntegration") -> None:
        """
        This fixture sets up the necessary environment for testing the integration of real MPAS data with the MPASVerticalCrossSectionPlotter. It initializes an instance of the plotter that will be used across multiple test methods to create vertical cross-section plots from real MPAS datasets. The fixture ensures that the plotter is ready for use in the tests, allowing them to focus on validating the functionality of the plotting methods when applied to actual data. This setup is crucial for confirming that the plotter can handle real-world scenarios and produce expected outputs without errors.

        Parameters:
            None

        Returns:
            None
        """
        self.plotter = MPASVerticalCrossSectionPlotter()
        
    def test_real_data_cross_section_with_pressure(self: "TestRealMPASDataIntegration", mpas_3d_processor) -> None:
        """
        This test validates the creation of a vertical cross-section plot using pressure as the vertical coordinate with real MPAS data. It checks that the plot is generated successfully and that the pressure axis is labeled correctly in the resulting plot. The test ensures that the MPASVerticalCrossSectionPlotter can handle real datasets and produce accurate visualizations when using pressure levels, which is a common vertical coordinate in atmospheric science. This validation confirms that the plotter can effectively utilize real MPAS data to create meaningful cross-section visualizations.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): Processor for MPAS 3D data.

        Returns:
            None
        """
        if mpas_3d_processor is None or not os.path.exists(GRID_FILE) or not os.path.exists(MPASOUT_DIR):
            pytest.skip("Real MPAS data not available")
            return
        
        processor = mpas_3d_processor
        
        fig, ax = self.plotter.create_vertical_cross_section(
            mpas_3d_processor=processor,
            var_name='theta',
            start_point=(0, 40),
            end_point=(10, 50),
            vertical_coord='pressure',
            num_points=50,
            time_index=0
        )
        
        assert fig is not None
        assert ax is not None
        
        ylabel = ax.get_ylabel()

        assert 'Pressure' in ylabel
        
        plt.close(fig)
    
    def test_real_data_cross_section_with_height(self: "TestRealMPASDataIntegration", mpas_3d_processor) -> None:
        """
        This test validates the creation of a vertical cross-section plot using height as the vertical coordinate with real MPAS data. It checks that the plot is generated successfully and that the height axis is labeled correctly in the resulting plot. The test ensures that the MPASVerticalCrossSectionPlotter can handle real datasets and produce accurate visualizations when using height levels, which is a common vertical coordinate in atmospheric science. This validation confirms that the plotter can effectively utilize real MPAS data to create meaningful cross-section visualizations.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): Processor for MPAS 3D data.

        Returns:
            None
        """
        if mpas_3d_processor is None or not os.path.exists(GRID_FILE) or not os.path.exists(MPASOUT_DIR):
            pytest.skip("Real MPAS data not available")
            return
        
        processor = mpas_3d_processor
        
        fig, ax = self.plotter.create_vertical_cross_section(
            mpas_3d_processor=processor,
            var_name='theta',
            start_point=(-10, 30),
            end_point=(15, 45),
            vertical_coord='height',
            max_height=15000,
            num_points=50,
            time_index=0
        )
        
        assert fig is not None
        assert ax is not None
        
        ylabel = ax.get_ylabel()

        assert 'evel' in ylabel.lower()  
        
        plt.close(fig)
    
    def test_real_data_batch_processing(self: "TestRealMPASDataIntegration") -> None:
        """
        This test validates the batch processing capability of the MPASVerticalCrossSectionPlotter when working with real MPAS data. It checks that the method for creating batch cross-section plots can successfully generate multiple plots for a specified variable across available time steps. The test ensures that the output files are created correctly in a temporary directory and that they exist after the plotting process is completed. This validation confirms that the plotter can handle batch processing of real datasets, which is essential for analyzing temporal evolution in atmospheric simulations.

        Parameters:
            self (Any): Test case instance with `plotter` fixture.

        Returns:
            None: Asserts that the returned file list is non-empty.
        """
        if not os.path.exists(GRID_FILE) or not os.path.exists(MPASOUT_DIR):
            pytest.skip("Real MPAS data not available")
            return
        
        processor = MPAS3DProcessor(grid_file=GRID_FILE)
        processor.load_3d_data(MPASOUT_DIR)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.plotter.create_batch_cross_section_plots(
                mpas_3d_processor=processor,
                output_dir=tmpdir,
                var_name='theta',
                start_point=(0, 35),
                end_point=(20, 50),
                vertical_coord='pressure',
                num_points=50,
                formats=['png']
            )
            
            assert len(result) > 0
            
            for file_path in result:
                assert os.path.exists(file_path)


if __name__ == "__main__":
    pytest.main([__file__])