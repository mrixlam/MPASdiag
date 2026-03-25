#!/usr/bin/env python3
"""
MPASdiag Test Suite: Surface Map Batch Processing Tests

This module contains unit tests for the batch processing functionality of the `MPASSurfacePlotter` class, as well as tests for helper methods and the `create_surface_plot` convenience function. The tests cover scenarios such as handling datasets with and without time dimensions, ensuring proper error handling when no data is loaded, and verifying that the helper methods return expected types and values.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules
import os
import sys
import shutil
import pytest
import tempfile
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
matplotlib.use('Agg')
from typing import Generator
from unittest.mock import Mock
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from cartopy.mpl.geoaxes import GeoAxes

from mpasdiag.visualization.surface import MPASSurfacePlotter, create_surface_plot

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestBatchProcessing:
    """ Tests for batch surface map creation. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestBatchProcessing") -> Generator[None, None, None]:
        """
        This fixture sets up a temporary directory and a plotter instance for batch processing tests. It also defines a common geographic extent for the tests. After the test runs, it ensures that the temporary directory is cleaned up. 

        Parameters:
            self ("TestBatchProcessing"): Test instance which will receive fixture attributes.

        Returns:
            None: Populates `self.plotter` and `self.temp_dir` and handles teardown.
        """
        self.plotter = MPASSurfacePlotter()
        self.temp_dir = tempfile.mkdtemp()
        self.extent_bounds = (-100, -90, 30, 40)

        yield

        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_batch_no_data_loaded(self: "TestBatchProcessing") -> None:
        """
        This test verifies that the batch processing function raises a ValueError when no dataset is loaded in the processor. It creates a mock processor with a `None` dataset and asserts that the expected exception is raised when attempting to create batch surface maps.

        Parameters:
            self ("TestBatchProcessing"): Test instance which has a temporary output dir.

        Returns:
            None: Assertion expects a ValueError to be raised.
        """
        mock_processor = Mock()
        mock_processor.dataset = None
        
        with pytest.raises(ValueError) as exc_info:
            self.plotter.create_batch_surface_maps(
                mock_processor, self.temp_dir,
                -100, -90, 30, 40
            )

        assert 'No data loaded' in str(exc_info.value)
    
    def test_batch_with_time_dimension(self: "TestBatchProcessing") -> None:
        """
        This test confirms that the batch processing function correctly iterates over a dataset with a `Time` dimension and produces output files for each time step. It creates a mock processor with a dataset containing a `Time` coordinate, mocks the necessary methods to return appropriate data, and asserts that the expected number of files are created in the output directory.

        Parameters:
            self ("TestBatchProcessing"): Test instance with a temporary output directory.

        Returns:
            None: Assertions validate file creation and count.
        """
        mock_processor = Mock()
        times = pd.date_range('2025-01-01', periods=3, freq='6h')
        
        ds = xr.Dataset({
            't2m': (['Time', 'nCells'], np.random.rand(3, 50) * 300),
            'lonCell': ('nCells', np.linspace(-100, -90, 50)),
            'latCell': ('nCells', np.linspace(30, 40, 50))
        }, coords={'Time': times})
        
        mock_processor.dataset = ds
        mock_processor.get_2d_variable_data = Mock(side_effect=lambda var, idx: ds[var].isel(Time=idx))

        mock_processor.extract_2d_coordinates_for_variable = Mock(
            return_value=(ds.lonCell.values, ds.latCell.values)
        )
        
        files = self.plotter.create_batch_surface_maps(
            mock_processor, self.temp_dir,
            *self.extent_bounds,
            var_name='t2m',
            plot_type='scatter'
        )
        
        assert len(files) == 3
        assert all(os.path.exists(f) for f in files)
    
    def test_batch_without_time_dimension(self: "TestBatchProcessing") -> None:
        """
        This test checks that the batch processing function can handle datasets that do not have a `Time` dimension. It creates a mock processor with a dataset that only has spatial dimensions, mocks the necessary methods to return appropriate data, and asserts that a single output file is created in the output directory.

        Parameters:
            self ("TestBatchProcessing"): Test instance with a temporary output directory.

        Returns:
            None: Assertions validate the number of produced files.
        """
        mock_processor = Mock()
        
        ds = xr.Dataset({
            't2m': (['time', 'nCells'], np.random.rand(2, 50) * 300),
            'lonCell': ('nCells', np.linspace(-100, -90, 50)),
            'latCell': ('nCells', np.linspace(30, 40, 50))
        })
        
        mock_processor.dataset = ds
        mock_processor.get_2d_variable_data = Mock(side_effect=lambda var, idx: ds[var].isel(time=idx))

        mock_processor.extract_2d_coordinates_for_variable = Mock(
            return_value=(ds.lonCell.values, ds.latCell.values)
        )
        
        files = self.plotter.create_batch_surface_maps(
            mock_processor, self.temp_dir,
            *self.extent_bounds,
            var_name='t2m'
        )
        
        assert len(files) == 2
    
    def test_batch_with_exception(self: "TestBatchProcessing") -> None:
        """
        This test ensures that if an exception occurs during the processing of a time step, the batch function handles it gracefully and continues processing subsequent time steps. It creates a mock processor with a dataset containing a `Time` coordinate, mocks the method to raise an exception for one of the time steps, and asserts that only the successful time steps produce output files.

        Parameters:
            self ("TestBatchProcessing"): Test instance with temporary output directory.

        Returns:
            None: Assertions validate the number of files produced.
        """
        mock_processor = Mock()
        times = pd.date_range('2025-01-01', periods=2, freq='6h')
        
        ds = xr.Dataset({
            't2m': (['Time', 'nCells'], np.random.rand(2, 50) * 300),
            'lonCell': ('nCells', np.linspace(-100, -90, 50)),
            'latCell': ('nCells', np.linspace(30, 40, 50))
        }, coords={'Time': times})
        
        mock_processor.dataset = ds

        mock_processor.get_2d_variable_data = Mock(
            side_effect=[ds['t2m'].isel(Time=0), Exception("Processing failed")]
        )

        mock_processor.extract_2d_coordinates_for_variable = Mock(
            return_value=(ds.lonCell.values, ds.latCell.values)
        )
        
        files = self.plotter.create_batch_surface_maps(
            mock_processor, self.temp_dir,
            *self.extent_bounds,
            var_name='t2m'
        )
        
        assert len(files) == 1


class TestHelperMethods:
    """ Tests for helper methods. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestHelperMethods") -> None:
        """
        This fixture initializes an instance of `MPASSurfacePlotter` for use in helper method tests. It ensures that each test has access to a fresh plotter instance without needing to create it within each test method.

        Parameters:
            self ("TestHelperMethods"): Test instance which will receive `plotter`.

        Returns:
            None: Populates `self.plotter`.
        """
        self.plotter = MPASSurfacePlotter()
    
    def test_get_surface_colormap_and_levels(self: "TestHelperMethods") -> None:
        """
        This test verifies that the `get_surface_colormap_and_levels` method returns a valid colormap name and a list of contour levels for a given variable. It calls the method with a common variable name (e.g., 't2m') and asserts that the returned colormap is a string and the levels are provided as a list. This ensures that the method is correctly configured to provide plotting parameters for surface variables.

        Parameters:
            self ("TestHelperMethods"): Test instance containing the plotter.

        Returns:
            None: Assertion validates returned types.
        """
        cmap, levels = self.plotter.get_surface_colormap_and_levels('t2m')
        
        assert isinstance(cmap, str)
        assert isinstance(levels, list)
    
    def test_create_simple_scatter_plot(self: "TestHelperMethods", mpas_coordinates, mpas_surface_temp_data) -> None:
        """
        This test confirms that the `create_simple_scatter_plot` method generates a Matplotlib Figure and Axes when provided with valid longitude, latitude, and data arrays. It uses real MPAS coordinate and surface temperature data from fixtures, calls the plotting method, and asserts that the returned objects are of the expected types. This validates that the helper method can successfully create a scatter plot with real data.

        Parameters:
            self ("TestHelperMethods"): Test instance containing prepared plotter.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_surface_temp_data: Session fixture providing real surface temperature data.

        Returns:
            None: Assertions validate returned Figure and Axes.
        """
        if mpas_coordinates is None or mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
        
        lon_full, lat_full = mpas_coordinates
        lon = lon_full[:50]
        lat = lat_full[:50]
        data = mpas_surface_temp_data[:50]
        
        fig, ax = self.plotter.create_simple_scatter_plot(
            lon, lat, data,
            title="Test Plot",
            colorbar_label="Temperature [K]"
        )
        
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)
    
    def test_simple_scatter_no_valid_data(self: "TestHelperMethods") -> None:
        """
        This test checks that the `create_simple_scatter_plot` method raises a ValueError when all data points are NaN. It creates longitude and latitude arrays along with a data array filled with NaN values, then calls the plotting method and asserts that the expected exception is raised with an appropriate error message. This ensures that the method correctly handles cases where there is no valid data to plot.

        Parameters:
            self ("TestHelperMethods"): Test instance containing the plotter.

        Returns:
            None: Assertion expects a ValueError to be raised.
        """
        lon = np.array([1, 2, 3])
        lat = np.array([1, 2, 3])
        data = np.array([np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError) as exc_info:
            self.plotter.create_simple_scatter_plot(lon, lat, data)

        assert 'No valid data' in str(exc_info.value)


class TestConvenienceFunction:
    """ Tests for create_surface_plot convenience function. """
    
    def test_create_surface_plot_function(self: "TestConvenienceFunction", mpas_coordinates, mpas_surface_temp_data) -> None:
        """
        This test verifies that the `create_surface_plot` convenience function successfully creates a surface plot and returns a Matplotlib Figure and GeoAxes when provided with valid longitude, latitude, data, variable name, and extent. It uses real MPAS coordinate and surface temperature data from fixtures, calls the convenience function, and asserts that the returned objects are of the expected types. This confirms that the convenience function correctly interfaces with the underlying plotting functionality.

        Parameters:
            self ("TestConvenienceFunction"): Test caller (unused).
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_surface_temp_data: Session fixture providing real surface temperature data.

        Returns:
            None: Assertions validate returned Figure and GeoAxes types.
        """
        if mpas_coordinates is None or mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
        
        lon_full, lat_full = mpas_coordinates
        lon = lon_full[:50]
        lat = lat_full[:50]
        data = mpas_surface_temp_data[:50]
        extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
        
        fig, ax = create_surface_plot(
            lon, lat, data, 't2m', extent,
            plot_type='scatter',
            title="Test Plot"
        )
        
        assert isinstance(fig, Figure)
        assert isinstance(ax, GeoAxes)
        plt.close(fig)
    
    def test_create_surface_plot_with_kwargs(self: "TestConvenienceFunction", mpas_coordinates, mpas_surface_temp_data) -> None:
        """
        This test checks that the `create_surface_plot` convenience function can accept additional keyword arguments for customizing the plot, such as colormap and projection. It uses real MPAS coordinate and surface temperature data from fixtures, calls the convenience function with extra parameters, and asserts that the returned Figure is of the expected type. This ensures that the convenience function can handle and apply additional plotting options correctly.

        Parameters:
            self ("TestConvenienceFunction"): Test caller (unused).
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_surface_temp_data: Session fixture providing real surface temperature data.
            
        Returns:
            None: Assertion validates returned Figure type.
        """
        if mpas_coordinates is None or mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
        
        lon_full, lat_full = mpas_coordinates
        lon = lon_full[:50]
        lat = lat_full[:50]
        data = mpas_surface_temp_data[:50]
        extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
        
        fig, ax = create_surface_plot(
            lon, lat, data, 't2m', extent,
            colormap='coolwarm',
            projection='Mercator'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
