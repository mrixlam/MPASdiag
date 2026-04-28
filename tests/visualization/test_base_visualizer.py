#!/usr/bin/env python3
"""
MPASdiag Test Suite: Base Processor and Visualizer Tests

This module contains unit tests for the MPASBaseProcessor and MPASVisualizer classes, covering initialization, file discovery, spatial coordinate handling, dataset operations, and helper methods. The tests use pytest fixtures to set up temporary resources and mock data, ensuring that the processor and visualizer can handle various scenarios including real MPAS data when available. The tests also verify that verbose output is generated correctly and that exceptions are raised with informative messages when expected.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load standard libraries
import os
import sys
import pytest
import shutil
import tempfile
import warnings
import numpy as np
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Generator, Any
from cartopy.mpl.geoaxes import GeoAxes
from unittest.mock import patch

from mpasdiag.processing.base import MPASBaseProcessor
from mpasdiag.visualization.base_visualizer import MPASVisualizer
from tests.test_data_helpers import load_mpas_coords_from_processor, assert_expected_public_methods

warnings.filterwarnings('ignore')

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
GRID_FILE = os.path.join(TEST_DATA_DIR, 'grids', 'x1.10242.static.nc')
MPASOUT_DIR = os.path.join(TEST_DATA_DIR, 'u240k', 'mpasout')


class TestHelperMethodsAndOutput:
    """ Tests for helper methods and verbose output generation. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestHelperMethodsAndOutput', 
                     mock_mpas_mesh: Any, 
                     mock_mpas_2d_data: Any) -> Generator[None, None, None]:
        """
        This fixture sets up a temporary environment for testing helper methods and verbose output generation of the `MPASBaseProcessor`. It creates a temporary directory and grid file using the provided `mock_mpas_mesh` fixture, which contains real MPAS mesh data when available. The fixture initializes an instance of `MPASBaseProcessor` with the created grid file and assigns the provided `mock_mpas_2d_data` to an instance variable for use in the tests. After yielding control to the test methods, it performs cleanup by ensuring that the processor's dataset is reset to a known state using real MPAS data from the fixtures, allowing subsequent tests to operate on valid datasets and verify the functionality of helper methods.

        Parameters:
            mock_mpas_mesh: Fixture providing real MPAS mesh with spatial vars.
            mock_mpas_2d_data: Fixture providing real 2D diagnostic data.

        Returns:
            Generator[None, None, None]: Yields control to the test and runs cleanup steps after the test completes.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.grid_file = os.path.join(self.temp_dir, "grid.nc")
        
        mock_mpas_mesh.to_netcdf(self.grid_file)
        
        self.processor = MPASBaseProcessor(self.grid_file, verbose=False)
        assert_expected_public_methods(self.processor, 'MPASBaseProcessor')
        self.mock_mpas_2d_data = mock_mpas_2d_data
            
        yield
        
        combined_ds = xr.Dataset({
            't2m': self.mock_mpas_2d_data['t2m'].isel(Time=0)
        })
        
        result_ds = self.processor._add_spatial_coords_helper(
            combined_ds,
            dimensions_to_add=['nCells'],
            spatial_vars=['lonCell', 'latCell'],
            processor_type='2D'
        )
        
        assert 'lonCell' in result_ds.data_vars
        assert 'latCell' in result_ds.data_vars
        assert 'nCells' in result_ds.coords
    
    def test_add_spatial_coords_helper_verbose(self: 'TestHelperMethodsAndOutput') -> None:
        """
        This test verifies that the `_add_spatial_coords_helper` method correctly adds spatial coordinate variables to the dataset and produces verbose output when `processor.verbose` is True. The test uses real MPAS 2D data from the fixture to create a combined dataset and calls the helper method to add spatial coordinates. It captures the standard output and asserts that the expected messages about loading the grid file and adding spatial coordinates are present, as well as confirming that the resulting dataset contains the new coordinate variables.

        Parameters:
            None

        Returns:
            None
        """
        self.processor.verbose = True

        combined_ds = xr.Dataset({
            't2m': self.mock_mpas_2d_data['t2m'].isel(Time=0)
        })
        
        from io import StringIO

        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            result_ds = self.processor._add_spatial_coords_helper(
                combined_ds,
                dimensions_to_add=['nCells'],
                spatial_vars=['lonCell', 'latCell'],
                processor_type='2D'
            )

            output = captured_output.getvalue()
            
            assert "Grid file loaded" in output
            assert "Added spatial coordinate variable" in output
            assert 'lonCell' in result_ds.data_vars
            assert 'latCell' in result_ds.data_vars

        finally:
            sys.stdout = sys.__stdout__
    

    def test_add_spatial_coords_no_matching_dimension(self: 'TestHelperMethodsAndOutput', 
                                                      mock_mpas_mesh: Any) -> None:
        """
        This test verifies that the `_add_spatial_coords_helper` method does not add spatial coordinate variables when the specified dimensions to add do not match any dimensions in the dataset. The test creates a dataset with a dimension that is not present in the grid file and calls the helper method to attempt to add spatial coordinates. It asserts that the resulting dataset remains unchanged (i.e., no new coordinate variables are added) and that the original variable is still present, confirming that the method correctly handles cases where expected dimensions are missing.

        Parameters:
            mock_mpas_mesh: Fixture providing real MPAS mesh data.

        Returns:
            None
        """
        processor = MPASBaseProcessor(GRID_FILE, verbose=False)
        assert_expected_public_methods(processor, 'MPASBaseProcessor')
        
        if 'nEdges' in mock_mpas_mesh.dims:
            n_edges = len(mock_mpas_mesh['nEdges'])
            combined_ds = xr.Dataset({
                'var1': xr.DataArray(np.ones(n_edges), dims=['nEdges'])
            })
        else:
            combined_ds = xr.Dataset({
                'var1': xr.DataArray(np.ones(50), dims=['nEdges'])
            })
        
        result_ds = processor._add_spatial_coords_helper(
            combined_ds,
            dimensions_to_add=['nCells', 'nVertices'], 
            spatial_vars=['lonCell', 'latCell'],
            processor_type='2D'
        )
        
        assert 'var1' in result_ds.data_vars
    

class TestDataLoadingStrategies:
    """ Tests for various data loading strategies and fallback mechanisms. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestDataLoadingStrategies', 
                     mock_mpas_mesh: Any, 
                     mock_mpas_2d_data: Any) -> Generator[None, None, None]:
        """
        This fixture sets up a temporary environment for testing data loading strategies and fallback mechanisms of the `MPASBaseProcessor`. It creates a temporary directory and grid file using the provided `mock_mpas_mesh` fixture, which contains real MPAS mesh data when available. The fixture initializes an instance of `MPASBaseProcessor` with the created grid file. It also creates a test data file using the provided `mock_mpas_2d_data` fixture, which contains real 2D diagnostic data. After yielding control to the test methods, it performs cleanup by ensuring that the processor's dataset is reset to a known state using real MPAS data from the fixtures, allowing subsequent tests to operate on valid datasets and verify the functionality of data loading strategies.

        Parameters:
            mock_mpas_mesh: Fixture providing real or synthetic MPAS mesh data.
            mock_mpas_2d_data: Fixture providing real 2D diagnostic data.

        Returns:
            Generator[None, None, None]: Yields to the test and performs cleanup after completion.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.grid_file = os.path.join(self.temp_dir, "grid.nc")
        
        mock_mpas_mesh.to_netcdf(self.grid_file)        
        self.processor = MPASBaseProcessor(self.grid_file, verbose=False)
        
        self.data_file = os.path.join(self.temp_dir, "data.nc")
        mock_mpas_2d_data.to_netcdf(self.data_file)
        
        yield

        assert_expected_public_methods(self.processor, 'MPASBaseProcessor')

    
    def test_all_loading_fails_triggers_single_file_fallback(self: 'TestDataLoadingStrategies') -> None:
        """
        This test verifies that when both the primary loading method (using `uxarray`) and the multi-file fallback method (using `xarray.open_mfdataset`) fail, the `_load_data` method correctly triggers the single-file fallback mechanism. The test patches both `ux.open_dataset` and `xr.open_mfdataset` to raise exceptions, simulating failures in both loading paths. It then calls the `_load_data` method and asserts that the resulting dataset is not None and that the output contains indications of using the single-file fallback, confirming that the method correctly handles multiple loading failures and falls back to a single file load.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        
        processor = MPAS3DProcessor(GRID_FILE, verbose=True)
        assert_expected_public_methods(processor, 'MPAS3DProcessor')

        from io import StringIO
        import sys as sys_module
        captured_output = StringIO()
        sys_module.stdout = captured_output
        
        try:
            with patch('mpasdiag.processing.base.ux.open_dataset') as mock_ux:
                with patch('mpasdiag.processing.base.xr.open_mfdataset') as mock_mfd:
                    mock_ux.side_effect = Exception("UXarray failed")
                    mock_mfd.side_effect = Exception("Multi-file xarray failed")
                    
                    dataset, data_type = processor._load_data(
                        MPASOUT_DIR,
                        use_pure_xarray=False,
                        chunks={'Time': 1},
                        reference_file="",
                        data_type_label="3D"
                    )
                    
                    output = captured_output.getvalue()
                    
                    assert dataset is not None
                    assert 'single' in output.lower()
        finally:
            sys_module.stdout = sys_module.__stdout__


class TestEdgeCasesAndErrorHandling:
    """ Tests for edge cases, boundary conditions, and error handling. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestEdgeCasesAndErrorHandling', 
                     mock_mpas_mesh: Any) -> Generator[None, None, None]:
        """
        This fixture sets up a temporary environment for testing edge cases and error handling of the `MPASBaseProcessor`. It creates a temporary directory and grid file using the provided `mock_mpas_mesh` fixture, which contains real MPAS mesh data when available. The fixture initializes an instance of `MPASBaseProcessor` with the created grid file and an instance of `MPASVisualizer`. After yielding control to the test methods, it performs cleanup by calling the `normalize_longitude` method with a value that exceeds the valid range, ensuring that the method correctly normalizes it back into the expected range. This setup allows the tests to exercise edge cases related to geographic coordinate handling and ensures that cleanup steps validate the processor's behavior.

        Parameters:
            mock_mpas_mesh: Fixture providing real or synthetic MPAS mesh data.

        Returns:
            Generator[None, None, None]: Yields control to the test and runs cleanup checks after the test completes.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.grid_file = os.path.join(self.temp_dir, "grid.nc")
        
        mock_mpas_mesh.to_netcdf(self.grid_file)
        
        self.processor = MPASBaseProcessor(self.grid_file, verbose=False)
        self.visualizer = MPASVisualizer(verbose=False)
        # assert_expected_public_methods(self.processor, 'MPASBaseProcessor')
        # assert_expected_public_methods(self.visualizer, 'MPASVisualizer')
            
        yield
        
        result = self.processor.normalize_longitude(450.0)
        assert result == pytest.approx(90.0)
    
    
    def test_time_series_empty_data(self: 'TestEdgeCasesAndErrorHandling') -> None:
        """
        This test verifies that the `create_time_series_plot` method can handle empty time and value arrays without raising an exception. The test calls the method with empty lists for times and values and asserts that a valid figure object is still created, confirming that the method can gracefully handle cases where no data is available for plotting.

        Parameters:
            None

        Returns:
            None
        """
        times = []
        values = []
        
        fig, ax = self.visualizer.create_time_series_plot(times, values)
        assert fig is not None
    

    def test_histogram_single_value(self: 'TestEdgeCasesAndErrorHandling') -> None:
        """
        This test verifies that the `create_histogram` method can handle an array with a single value without raising an exception. The test creates a numpy array containing a single value and calls the method to create a histogram. It asserts that a valid figure object is created, confirming that the method can gracefully handle cases where the input data has minimal variability.

        Parameters:
            None

        Returns:
            None
        """
        data = np.ones(100)        
        fig, ax = self.visualizer.create_histogram(data)
        assert fig is not None
    

class TestVisualizerOperations:
    """ Tests for MPASVisualizer initialization, configuration, and operations. """
    
    def setup_method(self: 'TestVisualizerOperations', 
                     mock_mpas_3d_data: Any) -> None:
        """
        This method sets up the context for testing the operations of the `MPASVisualizer`. It initializes an instance of `MPASVisualizer` with specific parameters (e.g., `figsize`, `dpi`, `verbose`) and creates a temporary directory for any output files that may be generated during the tests. It also assigns the provided `mock_mpas_3d_data` to an instance variable for use in the tests. This setup allows the test methods to operate on a consistent visualizer instance and have access to mock data for testing visualization functionalities.

        Parameters:
            mock_mpas_3d_data: Fixture providing real 3D MPAS data for testing visualizations.

        Returns:
            None
        """
        self.visualizer = MPASVisualizer(figsize=(10, 14), dpi=100, verbose=False)
        # assert_expected_public_methods(self.visualizer, 'MPASVisualizer')
        self.temp_dir = tempfile.mkdtemp()
        self.mock_mpas_3d_data = mock_mpas_3d_data
    
    def teardown_method(self: 'TestVisualizerOperations') -> None:
        """
        This method performs cleanup after each test method in the `TestVisualizerOperations` class. It checks if the temporary directory created during setup exists and removes it to ensure that no temporary files are left on the filesystem after the tests complete. Additionally, it closes all matplotlib figures to free up resources and prevent any interference with subsequent tests that may involve plotting.

        Parameters:
            None

        Returns:
            None
        """
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        plt.close('all')
    
    
    def test_create_time_series_plot_numpy_datetime(self: 'TestVisualizerOperations') -> None:
        """
        This test verifies that the `create_time_series_plot` method can handle time inputs provided as numpy datetime64 objects. The test creates a list of numpy datetime64 objects and corresponding values, calls the method to create a time series plot, and asserts that a figure is created and that a line is plotted on the axes, confirming that the method can process numpy datetime inputs correctly.

        Parameters:
            None

        Returns:
            None
        """
        times = [datetime(2024, 1, 1, i) for i in range(3)]
        values = [1.0, 2.0, 3.0]
        
        fig, ax = self.visualizer.create_time_series_plot(times, values)
        
        assert fig is not None
        lines = ax.get_lines()
        assert len(lines) == pytest.approx(1)
    

    def test_create_time_series_custom_labels(self: 'TestVisualizerOperations') -> None:
        """
        This test verifies that the `create_time_series_plot` method correctly applies custom title and axis labels when provided. The test creates a list of datetime objects and corresponding values, calls the method with specific title, x-label, and y-label parameters, and asserts that the axes have the expected title and labels, confirming that the method correctly incorporates custom labeling into the plot.

        Parameters:
            None

        Returns:
            None
        """
        times = [datetime(2024, 1, 1, i) for i in range(3)]
        values = [100.0, 200.0, 300.0]
        
        _, ax = self.visualizer.create_time_series_plot(
            times, values, 
            title="Custom Title",
            ylabel="Custom Y",
            xlabel="Custom X"
        )
        
        assert ax.get_title() == "Custom Title"
        assert ax.get_ylabel() == "Custom Y"
        assert ax.get_xlabel() == "Custom X"

    def test_create_histogram_with_nan(self: 'TestVisualizerOperations') -> None:
        """
        This test verifies that the `create_histogram` method can handle input data containing NaN and infinite values by filtering them out and still creating a valid histogram. The test creates a numpy array with a mix of finite, NaN, and infinite values, calls the method to create a histogram, and asserts that a figure is created and that the histogram is plotted using only the finite values, confirming that the method correctly handles non-finite data.

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([1.0, 2.0, np.nan, 3.0, np.inf, 4.0, -np.inf, 5.0])        
        fig, ax = self.visualizer.create_histogram(data, bins=10)
        
        assert fig is not None
        patches = ax.patches
        assert len(patches) > 0
    

    def test_create_histogram_log_scale(self: 'TestVisualizerOperations') -> None:
        """
        This test verifies that the `create_histogram` method correctly applies a logarithmic scale to the y-axis when the `log_scale` parameter is set to True. The test creates a numpy array with positive values suitable for log scaling, calls the method with `log_scale=True`, and asserts that the y-axis scale of the resulting plot is set to 'log', confirming that the method correctly configures the histogram for logarithmic scaling.

        Parameters:
            None

        Returns:
            None
        """
        data = np.logspace(0, 3, 1000)  # 1 to 1000        
        _, ax = self.visualizer.create_histogram(data, log_scale=True)        
        assert ax.get_yscale() == 'log'
    

    def test_create_histogram_custom_bins(self: 'TestVisualizerOperations') -> None:
        """
        This test verifies that the `create_histogram` method correctly uses custom bin edges when the `bins` parameter is provided. The test creates a numpy array of data and defines specific bin edges, calls the method with these custom bins, and asserts that the histogram is created using the specified bins by checking the number of patches (bars) in the histogram, confirming that the method correctly applies custom binning to the histogram.

        Parameters:
            None

        Returns:
            None
        """
        data = np.linspace(-3, 3, 500)
        bins = np.linspace(-3, 3, 10)
        
        fig, _ = self.visualizer.create_histogram(data, bins=bins)
        
        assert fig is not None
    

    def test_create_histogram_empty_data(self: 'TestVisualizerOperations') -> None:
        """
        This test verifies that the `create_histogram` method can handle an empty input array without raising an exception. The test creates an empty numpy array, calls the method to create a histogram, and asserts that a figure is still created, confirming that the method can gracefully handle cases where there is no data to plot.

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([np.nan, np.inf, -np.inf])
        fig, _ = self.visualizer.create_histogram(data)
        assert fig is not None


class TestWindVisualization:
    """ Tests for wind plot creation and background overlay. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestWindVisualization') -> Generator[None, None, None]:
        """
        This fixture sets up the context for testing wind visualization functionalities of the `MPASVisualizer`. It initializes an instance of `MPASVisualizer` with verbose output disabled and creates sample wind data (longitude, latitude, u and v components) using a module-level helper function that generates realistic MPAS grid coordinates. The fixture yields control to the test methods, allowing them to use the initialized visualizer and sample data. After the tests complete, it performs cleanup by creating a wind plot using the sample data and asserting that a figure is produced and that the axis is a GeoAxes instance, confirming that the visualizer can successfully create a wind plot with the provided data.

        Parameters:
            None

        Returns:
            Generator[None, None, None]: Fixture generator for pytest.
        """
        self.visualizer = MPASVisualizer(verbose=False)   

        self.lon, self.lat, self.u, self.v = load_mpas_coords_from_processor(n=50)
            
        yield
        
        fig, ax = self.visualizer.create_wind_plot(
            self.lon, self.lat, self.u, self.v,
            -120, -80, 30, 50,
            plot_type='barbs'
        )
        
        assert fig is not None
        assert isinstance(ax, GeoAxes)
    
    def test_create_wind_plot_arrows(self: 'TestWindVisualization') -> None:
        """
        This test verifies that the `create_wind_plot` method can create a wind plot using arrows to represent wind vectors when `plot_type='arrows'`. The test uses the sample longitude, latitude, u, and v data initialized in the fixture, calls the method with the specified plot type, and asserts that a figure is created and that the axis is an instance of GeoAxes, confirming that the method can successfully generate a wind plot with arrow representations.

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = self.visualizer.create_wind_plot(
            self.lon, self.lat, self.u, self.v,
            -120, -80, 30, 50,
            plot_type='arrows'
        )
        
        assert fig is not None
        assert isinstance(ax, GeoAxes)
    
    def test_create_wind_plot_with_background(self: 'TestWindVisualization') -> None:
        """
        This test verifies that the `create_wind_plot` method can include a background wind-speed field when the `show_background` parameter is set to True. The test uses the sample longitude, latitude, u, and v data initialized in the fixture, calls the method with `show_background=True` and a specified colormap for the background, and asserts that a figure is created without errors, confirming that the method can successfully generate a wind plot with an overlaid background field.

        Parameters:
            None

        Returns:
            None
        """
        fig, _ = self.visualizer.create_wind_plot(
            self.lon, self.lat, self.u, self.v,
            -120, -80, 30, 50,
            show_background=True,
            bg_colormap='plasma'
        )
        
        assert fig is not None
    
    def test_create_wind_plot_auto_subsample(self: 'TestWindVisualization') -> None:
        """
        This test verifies that the `create_wind_plot` method can automatically determine an appropriate subsampling factor when `subsample=0` is passed. The test uses a larger set of sample longitude, latitude, u, and v data initialized in the fixture, calls the method with `subsample=0`, and asserts that a figure is created without errors, confirming that the method can successfully compute an automatic subsampling factor to manage high-density wind data for plotting.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, u, v = load_mpas_coords_from_processor(n=1000)
        
        fig, _ = self.visualizer.create_wind_plot(
            lon, lat, u, v,
            -120, -80, 30, 50,
            subsample=0  
        )
        
        assert fig is not None
    
    def test_create_wind_plot_manual_subsample(self: 'TestWindVisualization') -> None:
        """
        This test verifies that the `create_wind_plot` method can accept a manual subsampling factor when the `subsample` parameter is set to a positive integer. The test uses the sample longitude, latitude, u, and v data initialized in the fixture, calls the method with a specified subsample value (e.g., 5), and asserts that a figure is created without errors, confirming that the method can successfully apply manual subsampling to manage wind data density for plotting.

        Parameters:
            None

        Returns:
            None
        """
        fig, _ = self.visualizer.create_wind_plot(
            self.lon, self.lat, self.u, self.v,
            -120, -80, 30, 50,
            subsample=5
        )
        
        assert fig is not None
    
    def test_create_wind_plot_with_timestamp(self: 'TestWindVisualization') -> None:
        """
        This test verifies that the `create_wind_plot` method correctly incorporates a provided timestamp into the plot title when the `time_stamp` parameter is given. The test uses the sample longitude, latitude, u, and v data initialized in the fixture, calls the method with a specific datetime object for the timestamp, and asserts that the resulting plot title contains the formatted date and time information, confirming that the method can successfully include temporal context in the wind plot title.

        Parameters:
            None

        Returns:
            None
        """
        time_stamp = datetime(2024, 1, 15, 12, 0)
        
        _, ax = self.visualizer.create_wind_plot(
            self.lon, self.lat, self.u, self.v,
            -120, -80, 30, 50,
            time_stamp=time_stamp
        )
        
        title = ax.get_title()
        assert '2024-01-15' in title
        assert '12:00 UTC' in title
    
    def test_create_wind_plot_custom_title(self: 'TestWindVisualization') -> None:
        """
        This test verifies that the `create_wind_plot` method correctly uses a custom title when the `title` parameter is provided. The test uses the sample longitude, latitude, u, and v data initialized in the fixture, calls the method with a specific string for the title, and asserts that the resulting plot title matches the provided custom title, confirming that the method can successfully apply user-defined titles to the wind plot.

        Parameters:
            None

        Returns:
            None
        """
        _, ax = self.visualizer.create_wind_plot(
            self.lon, self.lat, self.u, self.v,
            -120, -80, 30, 50,
            title="Custom Wind Title"
        )
        
        assert ax.get_title() == "Custom Wind Title"
    
    def test_create_wind_plot_custom_scale(self: 'TestWindVisualization') -> None:
        """
        This test verifies that the `create_wind_plot` method can accept a custom scaling factor for arrow plots when the `scale` parameter is provided. The test uses the sample longitude, latitude, u, and v data initialized in the fixture, calls the method with a specified scale value (e.g., 50.0) and `plot_type='arrows'`, and asserts that a figure is created without errors, confirming that the method can successfully apply custom scaling to the wind vectors in the plot.

        Parameters:
            None

        Returns:
            None
        """
        fig, _ = self.visualizer.create_wind_plot(
            self.lon, self.lat, self.u, self.v,
            -120, -80, 30, 50,
            plot_type='arrows',
            scale=50.0
        )
        
        assert fig is not None
    
    
class TestVisualizerHelperMethods:
    """ Tests for static/internal helper methods in MPASVisualizer. """

    @pytest.fixture(autouse=True)
    def setup_visualizer(self: 'TestVisualizerHelperMethods') -> Generator[None, None, None]:
        """
        This fixture sets up the context for testing the static and internal helper methods of the `MPASVisualizer`. It initializes an instance of `MPASVisualizer` before yielding control to the test methods, allowing them to use this instance for testing. After the tests complete, it ensures that all matplotlib figures are closed to clean up any resources used during plotting.

        Parameters:
            None

        Returns:
            None
        """
        self.viz = MPASVisualizer()
        yield
        plt.close('all')


    def test_create_time_series_plot_with_non_datetime_times(self: 'TestVisualizerHelperMethods') -> None:
        """
        This test verifies that the `create_time_series_plot` method can handle non-datetime time values without raising an exception. The test creates a simple time series with integer time values, calls the method to create a plot, and asserts that a figure and axis are created successfully, confirming that the method can accommodate time values that are not in datetime format. 

        Parameters:
            None

        Returns:
            None
        """
        times = [0, 1, 2, 3]  # non-datetime → triggers except branch
        values = np.array([1.0, 2.0, 3.0, 4.0])
        fig, ax = self.viz.create_time_series_plot(times, values, title='T', xlabel='X',
                                                   ylabel='Y')
        assert fig is not None
        assert ax is not None

    def test_create_histogram_bins_as_ndarray(self: 'TestVisualizerHelperMethods') -> None:
        """
        This test verifies that the `create_histogram` method can accept bins specified as a numpy array. The test creates a dataset, defines bins as a numpy array, calls the method to create a histogram, and asserts that a figure is created successfully, confirming that the method can handle bins provided in this format without errors. 

        Parameters:
            None

        Returns:
            None
        """
        data = np.linspace(0, 10, 200)
        bins = np.linspace(0, 10, 6)  # numpy array → exercises tolist() branch
        fig, ax = self.viz.create_histogram(data, bins=bins)
        assert fig is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
