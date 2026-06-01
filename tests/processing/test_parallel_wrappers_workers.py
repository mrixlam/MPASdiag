#!/usr/bin/env python3

"""
MPASdiag Test Suite: Tests for MPASdiag parallel processing wrappers

This module contains unit tests for the parallel processing wrapper functions in `mpasdiag.processing.parallel_wrappers`. These wrappers are responsible for orchestrating the execution of diagnostic plotting tasks in parallel using `MPASParallelManager`. The tests verify that the worker functions correctly prepare data, call the appropriate plotters, and return results in the expected format. Additionally, the tests ensure proper handling of edge cases, error conditions, and performance considerations in a parallel execution context.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules for testing
import os
import pytest
import shutil
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
from typing import Generator
from unittest.mock import Mock

from mpasdiag.processing.parallel_wrappers import (
    _precipitation_worker,
    _surface_worker,
    _wind_worker,
    _cross_section_worker
)

from tests.test_data_helpers import assert_expected_public_methods

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
GRID_FILE = os.path.join(TEST_DATA_DIR, 'grids', 'x1.10242.static.nc')


_RNG = np.random.default_rng(42)


class TestPrecipitationWorker:
    """ Tests for the precipitation worker wrapper that generates precipitation maps. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestPrecipitationWorker') -> Generator[None, None, None]:
        """
        This fixture sets up a temporary directory and a mock processor with a synthetic dataset for precipitation worker tests. The dataset includes a `rainnc` variable and coordinate arrays for longitude and latitude. The mock processor is configured to return this dataset and provide coordinate extraction functionality. After the test runs, the temporary directory is cleaned up.     

        Parameters:
            None

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()        
        self.mock_processor = Mock()
        n_time, n_cells = 5, 100
        
        self.mock_dataset = xr.Dataset({
            'rainnc': xr.DataArray(
                _RNG.uniform(0, 50, (n_time, n_cells)),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=n_time, freq='h')}
            ),
            'lonCell': xr.DataArray(np.linspace(-120, -80, n_cells), dims=['nCells']),
            'latCell': xr.DataArray(np.linspace(30, 50, n_cells), dims=['nCells'])
        })
        
        self.mock_processor.dataset = self.mock_dataset
        self.mock_processor.data_type = 'UXarray'

        self.mock_processor.extract_2d_coordinates_for_variable.return_value = (
            self.mock_dataset['lonCell'].values,
            self.mock_dataset['latCell'].values
        )

        yield

        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_precipitation_worker_success(self: 'TestPrecipitationWorker') -> None:
        """
        This test verifies that the `_precipitation_worker` function successfully generates precipitation maps and returns the expected result structure. It mocks the `PrecipitationDiagnostics` to return a random precipitation difference dataset and the `MPASPrecipitationPlotter` to return mock figure and axis objects. The test asserts that the result contains the expected keys (`files`, `timings`, `time_str`, `cache_hits`) and that the `files` key contains a list, confirming that the worker function executed without errors and produced output in the correct format. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter, orig_diag = _pw.MPASPrecipitationPlotter, _pw.PrecipitationDiagnostics
        
        assert_expected_public_methods(orig_plotter, 'MPASPrecipitationPlotter')
        assert_expected_public_methods(orig_diag, 'PrecipitationDiagnostics')

        mock_diag = Mock()
        mock_precip_data = xr.DataArray(_RNG.uniform(0, 10, 100))
        mock_diag.compute_precipitation_difference.return_value = mock_precip_data
        
        mock_plotter = Mock()
        mock_fig, mock_ax = Mock(), Mock()
        mock_plotter.create_precipitation_map.return_value = (mock_fig, mock_ax)
        
        kwargs = {
            'processor': self.mock_processor,
            'cache': None,
            'output_dir': self.temp_dir,
            'lon_min': -120, 'lon_max': -80,
            'lat_min': 30, 'lat_max': 50,
            'var_name': 'rainnc',
            'accum_period': 'a01h',
            'plot_type': 'scatter',
            'grid_resolution': None,
            'file_prefix': 'test_precip',
            'formats': ['png'],
            'custom_title_template': None,
            'colormap': None,
            'levels': None
        }
        
        _pw.MPASPrecipitationPlotter = lambda *a, **kw: mock_plotter
        _pw.PrecipitationDiagnostics = lambda *a, **kw: mock_diag

        try:
            result = _precipitation_worker((2, kwargs))
            
            assert 'files' in result
            assert 'timings' in result
            assert 'time_str' in result
            assert 'cache_hits' in result
            assert isinstance(result['files'], list)
            assert result['timings']['total'] > 0
        finally:
            _pw.MPASPrecipitationPlotter, _pw.PrecipitationDiagnostics = orig_plotter, orig_diag
    
    def test_precipitation_worker_with_cache(self: 'TestPrecipitationWorker') -> None:
        """
        This test ensures that the `_precipitation_worker` function correctly utilizes a provided cache for coordinate lookups. It mocks a cache that returns precomputed longitude and latitude values, and verifies that the worker records cache hits and calls the cache's `get_coordinates` method exactly once. This confirms that the caching mechanism is integrated properly and avoids redundant coordinate extraction. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter, orig_diag = _pw.MPASPrecipitationPlotter, _pw.PrecipitationDiagnostics
        assert_expected_public_methods(orig_plotter, 'MPASPrecipitationPlotter')
        assert_expected_public_methods(orig_diag, 'PrecipitationDiagnostics')

        mock_cache = Mock()

        mock_cache.get_coordinates.return_value = (
            self.mock_dataset['lonCell'].values,
            self.mock_dataset['latCell'].values
        )
        
        mock_diag = Mock()
        mock_precip_data = xr.DataArray(_RNG.uniform(0, 10, 100))
        mock_diag.compute_precipitation_difference.return_value = mock_precip_data
        
        mock_plotter = Mock()
        mock_fig, mock_ax = Mock(), Mock()
        mock_plotter.create_precipitation_map.return_value = (mock_fig, mock_ax)
        
        kwargs = {
            'processor': self.mock_processor,
            'cache': mock_cache,
            'output_dir': self.temp_dir,
            'lon_min': -120, 'lon_max': -80,
            'lat_min': 30, 'lat_max': 50,
            'var_name': 'rainnc',
            'accum_period': 'a01h',
            'plot_type': 'scatter',
            'grid_resolution': None,
            'file_prefix': 'test_precip',
            'formats': ['png'],
            'custom_title_template': None,
            'colormap': None,
            'levels': None
        }
        
        _pw.MPASPrecipitationPlotter = lambda *a, **kw: mock_plotter
        _pw.PrecipitationDiagnostics = lambda *a, **kw: mock_diag

        try:
            result = _precipitation_worker((2, kwargs))
            
            assert result['cache_hits']['coordinates']
            mock_cache.get_coordinates.assert_called_once()
        finally:
            _pw.MPASPrecipitationPlotter, _pw.PrecipitationDiagnostics = orig_plotter, orig_diag

    def test_precipitation_worker_with_weights_dir(self: 'TestPrecipitationWorker') -> None:
        """
        This test verifies that the `_precipitation_worker` function can successfully execute when a `weights_dir` is provided in the kwargs. It mocks the necessary plotter and diagnostics classes, and ensures that the worker function completes without errors and returns a result containing the expected keys. This test confirms that the presence of a `weights_dir` does not interfere with the normal operation of the precipitation worker. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter, orig_diag = _pw.MPASPrecipitationPlotter, _pw.PrecipitationDiagnostics

        mock_diag = Mock()
        mock_precip_data = xr.DataArray(_RNG.uniform(0, 10, 100))
        mock_diag.compute_precipitation_difference.return_value = mock_precip_data

        mock_plotter = Mock()
        mock_fig, mock_ax = Mock(), Mock()
        mock_plotter.create_precipitation_map.return_value = (mock_fig, mock_ax)

        kwargs = {
            'processor': self.mock_processor,
            'cache': None,
            'output_dir': self.temp_dir,
            'lon_min': -120, 'lon_max': -80,
            'lat_min': 30, 'lat_max': 50,
            'var_name': 'rainnc',
            'accum_period': 'a01h',
            'plot_type': 'scatter',
            'grid_resolution': None,
            'file_prefix': 'test_precip_weights',
            'formats': ['png'],
            'custom_title_template': None,
            'colormap': None,
            'levels': None,
            'weights_dir': self.temp_dir,
        }

        _pw.MPASPrecipitationPlotter = lambda *a, **kw: mock_plotter
        _pw.PrecipitationDiagnostics = lambda *a, **kw: mock_diag

        try:
            result = _precipitation_worker((2, kwargs))
            assert 'files' in result
        finally:
            _pw.MPASPrecipitationPlotter, _pw.PrecipitationDiagnostics = orig_plotter, orig_diag


class TestSurfaceWorker:
    """ Tests for the surface worker wrapper that generates surface variable maps. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestSurfaceWorker') -> Generator[None, None, None]: # type: ignore
        """
        This fixture sets up a temporary directory and a mock processor with a synthetic dataset containing a `t2m` variable for surface worker tests. The dataset includes coordinate arrays for longitude and latitude. The mock processor is configured to return this dataset and provide methods for extracting spatial coordinates. After the test runs, the temporary directory is removed to clean up resources. 

        Parameters:
            None

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()        
        self.mock_processor = Mock()
        n_time, n_cells = 5, 100
        
        self.mock_dataset = xr.Dataset({
            't2m': xr.DataArray(
                _RNG.uniform(250, 310, (n_time, n_cells)),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=n_time, freq='h')}
            ),
            'lonCell': xr.DataArray(np.linspace(-120, -80, n_cells), dims=['nCells']),
            'latCell': xr.DataArray(np.linspace(30, 50, n_cells), dims=['nCells'])
        })
        
        self.mock_processor.dataset = self.mock_dataset

        self.mock_processor.extract_spatial_coordinates.return_value = (
            self.mock_dataset['lonCell'].values,
            self.mock_dataset['latCell'].values
        )
    
    def teardown_method(self: 'TestSurfaceWorker') -> None:
        """
        This method cleans up temporary resources created for surface worker tests by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment. 

        Parameters:
            None

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_surface_worker_success(self: 'TestSurfaceWorker') -> None:
        """
        This test verifies that the `_surface_worker` function successfully generates surface variable maps and returns the expected result structure. It mocks the `MPASSurfacePlotter` to return mock figure and axis objects, executes the worker with a prepared mock processor and specified kwargs, and asserts that the result contains the expected keys (`files`, `timings`, `time_str`) and that the `files` key contains a list, confirming that the worker function executed without errors and produced output in the correct format. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter = _pw.MPASSurfacePlotter
        assert_expected_public_methods(orig_plotter, 'MPASSurfacePlotter')

        mock_plotter = Mock()
        mock_fig, mock_ax = Mock(), Mock()
        mock_plotter.create_surface_map.return_value = (mock_fig, mock_ax)
        
        kwargs = {
            'processor': self.mock_processor,
            'cache': None,
            'output_dir': self.temp_dir,
            'lon_min': -120, 'lon_max': -80,
            'lat_min': 30, 'lat_max': 50,
            'var_name': 't2m',
            'plot_type': 'scatter',
            'file_prefix': 'test_surface',
            'formats': ['png'],
            'custom_title': None,
            'colormap': None,
            'levels': None
        }
        
        _pw.MPASSurfacePlotter = lambda *a, **kw: mock_plotter

        try:
            result = _surface_worker((2, kwargs))
            
            assert 'files' in result
            assert 'timings' in result
            assert 'time_str' in result
            assert isinstance(result['files'], list)
        finally:
            _pw.MPASSurfacePlotter = orig_plotter

    def test_surface_worker_with_primed_cache(self: 'TestSurfaceWorker') -> None:
        """
        This test verifies that `_surface_worker` records a cache coordinate hit when the supplied cache already has coordinates loaded for the variable. It provides a mock cache whose `get_coordinates` returns coordinates successfully, then asserts that `cache_hits['coordinates']` is True in the result and that `get_coordinates` was called exactly once. This confirms that the worker function properly utilizes the cache when available, avoiding redundant coordinate extraction from the processor. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter = _pw.MPASSurfacePlotter

        mock_plotter = Mock()
        mock_fig, mock_ax = Mock(), Mock()
        mock_plotter.create_surface_map.return_value = (mock_fig, mock_ax)

        mock_cache = Mock()
        mock_cache.get_coordinates.return_value = (
            self.mock_dataset['lonCell'].values,
            self.mock_dataset['latCell'].values,
        )

        kwargs = {
            'processor': self.mock_processor,
            'cache': mock_cache,
            'output_dir': self.temp_dir,
            'lon_min': -120, 'lon_max': -80,
            'lat_min': 30, 'lat_max': 50,
            'var_name': 't2m',
            'plot_type': 'scatter',
            'file_prefix': 'test_surface_cache',
            'formats': ['png'],
            'custom_title': None,
            'colormap': None,
            'levels': None,
        }

        _pw.MPASSurfacePlotter = lambda *a, **kw: mock_plotter

        try:
            result = _surface_worker((2, kwargs))
            assert result['cache_hits']['coordinates'] is True
            mock_cache.get_coordinates.assert_called_once()
        finally:
            _pw.MPASSurfacePlotter = orig_plotter


class TestWindWorker:
    """ Tests for the wind worker wrapper that generates wind plots. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestWindWorker') -> None:
        """
        This fixture sets up a temporary directory and a mock processor with a synthetic dataset containing `u10` and `v10` wind components for wind worker tests. The dataset includes coordinate arrays for longitude and latitude. The mock processor is configured to return this dataset and provide methods for extracting spatial coordinates. After the test runs, the temporary directory is removed to clean up resources. 

        Parameters:
            None

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()        
        self.mock_processor = Mock()
        n_time, n_cells = 5, 100
        
        self.mock_dataset = xr.Dataset({
            'u10': xr.DataArray(
                _RNG.uniform(-10, 10, (n_time, n_cells)),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=n_time, freq='h')}
            ),
            'v10': xr.DataArray(
                _RNG.uniform(-10, 10, (n_time, n_cells)),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=n_time, freq='h')}
            ),
            'lonCell': xr.DataArray(np.linspace(-120, -80, n_cells), dims=['nCells']),
            'latCell': xr.DataArray(np.linspace(30, 50, n_cells), dims=['nCells'])
        })
        
        self.mock_processor.dataset = self.mock_dataset
        self.mock_processor.get_2d_variable_data.side_effect = lambda var, idx: self.mock_dataset[var].isel(Time=idx)

        self.mock_processor.extract_2d_coordinates_for_variable.return_value = (
            self.mock_dataset['lonCell'].values,
            self.mock_dataset['latCell'].values
        )
    
    def teardown_method(self: 'TestWindWorker') -> None:
        """
        This method cleans up temporary resources created for wind worker tests by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment. 

        Parameters:
            None

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_wind_worker_success(self: 'TestWindWorker') -> None:
        """
        This test verifies that the `_wind_worker` function successfully generates wind plots and returns the expected result structure. It mocks the `MPASWindPlotter` to return mock figure and axis objects, executes the worker with a prepared mock processor and specified kwargs, and asserts that the result contains the expected keys (`files`, `timings`, `time_str`) and that the `files` key contains a list, confirming that the worker function executed without errors and produced output in the correct format. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter = _pw.MPASWindPlotter
        assert_expected_public_methods(orig_plotter, 'MPASWindPlotter')

        mock_plotter = Mock()
        mock_fig, mock_ax = Mock(), Mock()
        mock_plotter.create_wind_plot.return_value = (mock_fig, mock_ax)
        
        kwargs = {
            'processor': self.mock_processor,
            'cache': None,
            'output_dir': self.temp_dir,
            'lon_min': -120, 'lon_max': -80,
            'lat_min': 30, 'lat_max': 50,
            'u_variable': 'u10',
            'v_variable': 'v10',
            'plot_type': 'barbs',
            'subsample': 1,
            'scale': None,
            'show_background': False,
            'grid_resolution': None,
            'regrid_method': 'linear',
            'file_prefix': 'test_wind',
            'formats': ['png']
        }
        
        _pw.MPASWindPlotter = lambda *a, **kw: mock_plotter

        try:
            result = _wind_worker((2, kwargs))
            
            assert 'files' in result
            assert 'timings' in result
            assert 'time_str' in result
        finally:
            _pw.MPASWindPlotter = orig_plotter

    def test_wind_worker_with_primed_cache(self: 'TestWindWorker') -> None:
        """
        This test verifies that `_wind_worker` records a cache coordinate hit when the supplied cache already has coordinates loaded for the variable. It provides a mock cache whose `get_coordinates` returns coordinates successfully, then asserts that `cache_hits['coordinates']` is True in the result and that `get_coordinates` was called exactly once. This confirms that the worker function properly utilizes the cache when available, avoiding redundant coordinate extraction from the processor. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter = _pw.MPASWindPlotter

        mock_plotter = Mock()
        mock_fig, mock_ax = Mock(), Mock()
        mock_plotter.create_wind_plot.return_value = (mock_fig, mock_ax)

        mock_cache = Mock()
        mock_cache.get_coordinates.return_value = (
            self.mock_dataset['lonCell'].values,
            self.mock_dataset['latCell'].values,
        )

        kwargs = {
            'processor': self.mock_processor,
            'cache': mock_cache,
            'output_dir': self.temp_dir,
            'lon_min': -120, 'lon_max': -80,
            'lat_min': 30, 'lat_max': 50,
            'u_variable': 'u10',
            'v_variable': 'v10',
            'plot_type': 'barbs',
            'subsample': 1,
            'scale': None,
            'show_background': False,
            'grid_resolution': None,
            'regrid_method': 'linear',
            'file_prefix': 'test_wind_cache',
            'formats': ['png'],
        }

        _pw.MPASWindPlotter = lambda *a, **kw: mock_plotter

        try:
            result = _wind_worker((2, kwargs))
            assert result['cache_hits']['coordinates'] is True
            mock_cache.get_coordinates.assert_called_once()
        finally:
            _pw.MPASWindPlotter = orig_plotter


class TestCrossSectionWorker:
    """ Tests for the cross-section worker wrapper that generates vertical cross-section plots. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestCrossSectionWorker') -> None:
        """
        This fixture sets up a temporary directory and a mock processor with a synthetic dataset containing a `temperature` variable for cross-section worker tests. The dataset includes coordinate arrays for time, vertical levels, and cells. The mock processor is configured to return this dataset, allowing the worker function to be invoked without accessing external data files. After the test runs, the temporary directory is removed to clean up resources. 

        Parameters:
            None

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()        
        self.mock_processor = Mock()
        n_time, n_levels, n_cells = 3, 20, 100
        
        self.mock_dataset = xr.Dataset({
            'temperature': xr.DataArray(
                _RNG.uniform(250, 310, (n_time, n_levels, n_cells)),
                dims=['Time', 'nVertLevels', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=n_time, freq='h')}
            )
        })
        
        self.mock_processor.dataset = self.mock_dataset
    
    def teardown_method(self: 'TestCrossSectionWorker') -> None:
        """
        This method cleans up temporary resources created for cross-section worker tests by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment. 

        Parameters:
            None

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cross_section_worker_success(self: 'TestCrossSectionWorker') -> None:
        """
        This test verifies that the `_cross_section_worker` function successfully generates vertical cross-section plots and returns the expected result structure. It mocks the `MPASVerticalCrossSectionPlotter` to return mock figure and axis objects, executes the worker with a prepared mock processor and specified kwargs, and asserts that the result contains the expected keys (`files`, `timings`, `time_str`) and that the `files` key contains a list, confirming that the worker function executed without errors and produced output in the correct format. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter = _pw.MPASVerticalCrossSectionPlotter

        mock_plotter = Mock()
        mock_fig, mock_ax = Mock(), Mock()
        mock_plotter.create_vertical_cross_section.return_value = (mock_fig, mock_ax)
        
        kwargs = {
            'processor': self.mock_processor,
            'output_dir': self.temp_dir,
            'start_lat': 35, 'start_lon': -110,
            'end_lat': 45, 'end_lon': -90,
            'var_name': 'temperature',
            'file_prefix': 'test_xsec',
            'formats': ['png'],
            'custom_title': None,
            'colormap': None,
            'levels': None,
            'vertical_coord': 'pressure',
            'num_points': 100
        }
        
        _pw.MPASVerticalCrossSectionPlotter = lambda *a, **kw: mock_plotter

        try:
            result = _cross_section_worker((1, kwargs))
            
            assert 'files' in result
            assert 'timings' in result
            assert 'time_str' in result
        finally:
            _pw.MPASVerticalCrossSectionPlotter = orig_plotter

    def test_cross_section_worker_uppercase_format(self: 'TestCrossSectionWorker') -> None:
        """
        This test verifies that the `_cross_section_worker` function can successfully execute when the output format is specified in uppercase (e.g., 'PNG' instead of 'png'). It mocks the `MPASVerticalCrossSectionPlotter` to return mock figure and axis objects, executes the worker with a prepared mock processor and specified kwargs that include an uppercase format, and asserts that the result contains the expected keys (`files`, `timings`, `time_str`) and that the `files` key contains a list, confirming that the worker function correctly handles uppercase format specifications without errors. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter = _pw.MPASVerticalCrossSectionPlotter

        mock_plotter = Mock()
        mock_fig, mock_ax = Mock(), Mock()
        mock_plotter.create_vertical_cross_section.return_value = (mock_fig, mock_ax)

        kwargs = {
            'processor': self.mock_processor,
            'output_dir': self.temp_dir,
            'start_lat': 35, 'start_lon': -110,
            'end_lat': 45, 'end_lon': -90,
            'var_name': 'temperature',
            'file_prefix': 'test_xsec_uppercase',
            'formats': ['PNG'],
            'custom_title': None,
            'colormap': None,
            'levels': None,
            'vertical_coord': 'pressure',
            'num_points': 100,
        }

        _pw.MPASVerticalCrossSectionPlotter = lambda *a, **kw: mock_plotter

        try:
            result = _cross_section_worker((1, kwargs))
            assert 'files' in result
        finally:
            _pw.MPASVerticalCrossSectionPlotter = orig_plotter


class TestWorkerCacheKeyErrorFallback:
    """Tests for the KeyError fallback paths in worker coordinate-cache lookups."""

    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestWorkerCacheKeyErrorFallback') -> None:
        """
        This fixture sets up a temporary directory and mock processors with synthetic datasets for testing the KeyError fallback paths in worker coordinate-cache lookups. The mock processors are configured to return datasets with appropriate variables and coordinate extraction methods. This setup allows the tests to verify that when a KeyError occurs during cache coordinate retrieval, the worker functions correctly fall back to extracting coordinates from the processor, ensuring that the worker can still execute successfully even when cache entries are missing. After the test runs, the temporary directory is removed to clean up resources.

        Parameters:
            None

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()
        n_cells = 100
        lon = np.linspace(-120, -80, n_cells)
        lat = np.linspace(30, 50, n_cells)

        self.mock_processor_precip = Mock()
        ds_precip = xr.Dataset({
            'rainnc': xr.DataArray(
                _RNG.uniform(0, 50, (3, n_cells)),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=3, freq='h')}
            ),
        })
        self.mock_processor_precip.dataset = ds_precip
        self.mock_processor_precip.data_type = 'UXarray'
        self.mock_processor_precip.extract_2d_coordinates_for_variable.return_value = (lon, lat)

        self.mock_processor_surface = Mock()
        ds_surface = xr.Dataset({
            't2m': xr.DataArray(
                _RNG.uniform(260, 310, (3, n_cells)),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=3, freq='h')}
            ),
        })
        self.mock_processor_surface.dataset = ds_surface
        self.mock_processor_surface.extract_spatial_coordinates.return_value = (lon, lat)

        self.mock_processor_wind = Mock()
        self.mock_processor_wind.dataset = xr.Dataset({
            'u10': xr.DataArray(
                _RNG.uniform(-5, 5, (3, n_cells)),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=3, freq='h')}
            ),
            'v10': xr.DataArray(
                _RNG.uniform(-5, 5, (3, n_cells)),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=3, freq='h')}
            ),
        })
        self.mock_processor_wind.extract_2d_coordinates_for_variable.return_value = (lon, lat)
        self.mock_processor_wind.get_2d_variable_data.return_value = xr.DataArray(
            _RNG.uniform(-5, 5, n_cells)
        )

        self.lon, self.lat = lon, lat

    def teardown_method(self: 'TestWorkerCacheKeyErrorFallback') -> None:
        """
        This method cleans up temporary resources created for the worker cache key error fallback tests by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment.

        Parameters:
            None

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_precipitation_worker_cache_key_error_fallback(self: 'TestWorkerCacheKeyErrorFallback',) -> None:
        """
        This test verifies that the `_precipitation_worker` function correctly falls back to extracting coordinates from the processor when a KeyError occurs during cache coordinate retrieval. It mocks a cache that raises a KeyError for the 'rainnc' variable, and ensures that the worker function can still execute successfully by extracting coordinates directly from the processor. The test asserts that the result contains the expected keys and that the cache hit for coordinates is recorded as False, confirming that the fallback mechanism is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter, orig_diag = _pw.MPASPrecipitationPlotter, _pw.PrecipitationDiagnostics

        mock_cache = Mock()
        mock_cache.get_coordinates.side_effect = KeyError('rainnc')
        mock_cache.load_coordinates_from_dataset.side_effect = Exception("cache write failure")

        mock_diag = Mock()
        mock_diag.compute_precipitation_difference.return_value = xr.DataArray(
            _RNG.uniform(0, 5, 100)
        )
        mock_plotter = Mock()
        mock_plotter.create_precipitation_map.return_value = (Mock(), Mock())

        kwargs = {
            'processor': self.mock_processor_precip,
            'cache': mock_cache,
            'output_dir': self.temp_dir,
            'lon_min': -120, 'lon_max': -80,
            'lat_min': 30, 'lat_max': 50,
            'var_name': 'rainnc',
            'accum_period': 'a01h',
            'plot_type': 'scatter',
            'grid_resolution': None,
            'file_prefix': 'test_precip_keyerr',
            'formats': ['png'],
            'custom_title_template': None,
            'colormap': None,
            'levels': None,
        }

        _pw.MPASPrecipitationPlotter = lambda *a, **kw: mock_plotter
        _pw.PrecipitationDiagnostics = lambda *a, **kw: mock_diag

        try:
            result = _precipitation_worker((1, kwargs))
            assert 'files' in result
            assert result['cache_hits'].get('coordinates') is not True
        finally:
            _pw.MPASPrecipitationPlotter, _pw.PrecipitationDiagnostics = orig_plotter, orig_diag

    def test_precipitation_worker_no_time_dim_fallback(self: 'TestWorkerCacheKeyErrorFallback',) -> None:
        """
        This test verifies that the `_precipitation_worker` function can successfully execute and generate a time string when the input dataset does not contain a time dimension. It mocks a processor with a dataset that has no time dimension, and ensures that the worker function can still run without errors and produces a result with a time string that starts with 't', confirming that the worker can handle datasets without time coordinates by generating an appropriate time string for output files.

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter, orig_diag = _pw.MPASPrecipitationPlotter, _pw.PrecipitationDiagnostics

        proc = Mock()
        proc.data_type = 'UXarray'
        proc.dataset = xr.Dataset({
            'rainnc': xr.DataArray(
                np.zeros((3, 100)), dims=['step', 'nCells']
            ),
        })
        proc.extract_2d_coordinates_for_variable.return_value = (self.lon, self.lat)

        mock_diag = Mock()
        mock_diag.compute_precipitation_difference.return_value = xr.DataArray(
            np.zeros(100)
        )
        mock_plotter = Mock()
        mock_plotter.create_precipitation_map.return_value = (Mock(), Mock())

        kwargs = {
            'processor': proc,
            'cache': None,
            'output_dir': self.temp_dir,
            'lon_min': -120, 'lon_max': -80,
            'lat_min': 30, 'lat_max': 50,
            'var_name': 'rainnc',
            'accum_period': 'a01h',
            'plot_type': 'scatter',
            'grid_resolution': None,
            'file_prefix': 'test_notime',
            'formats': ['png'],
            'custom_title_template': None,
            'colormap': None,
            'levels': None,
        }

        _pw.MPASPrecipitationPlotter = lambda *a, **kw: mock_plotter
        _pw.PrecipitationDiagnostics = lambda *a, **kw: mock_diag

        try:
            result = _precipitation_worker((1, kwargs))
            assert result['time_str'].startswith('t')
        finally:
            _pw.MPASPrecipitationPlotter, _pw.PrecipitationDiagnostics = orig_plotter, orig_diag

    def test_precipitation_worker_custom_title_template(self: 'TestWorkerCacheKeyErrorFallback',) -> None:
        """
        This test verifies that the `_precipitation_worker` function can successfully execute when a custom title template is provided in the kwargs. It mocks the necessary plotter and diagnostics classes, and ensures that the worker function completes without errors and returns a result containing the expected keys. This test confirms that the presence of a `custom_title_template` does not interfere with the normal operation of the precipitation worker, allowing for customized plot titles while still generating output correctly. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter, orig_diag = _pw.MPASPrecipitationPlotter, _pw.PrecipitationDiagnostics

        mock_diag = Mock()
        mock_diag.compute_precipitation_difference.return_value = xr.DataArray(
            np.zeros(100)
        )
        mock_plotter = Mock()
        mock_plotter.create_precipitation_map.return_value = (Mock(), Mock())

        kwargs = {
            'processor': self.mock_processor_precip,
            'cache': None,
            'output_dir': self.temp_dir,
            'lon_min': -120, 'lon_max': -80,
            'lat_min': 30, 'lat_max': 50,
            'var_name': 'rainnc',
            'accum_period': 'a01h',
            'plot_type': 'scatter',
            'grid_resolution': None,
            'file_prefix': 'test_custom_title',
            'formats': ['png'],
            'custom_title_template': 'Custom {var_name} at {time_str} ({accum_period})',
            'colormap': None,
            'levels': None,
        }

        _pw.MPASPrecipitationPlotter = lambda *a, **kw: mock_plotter
        _pw.PrecipitationDiagnostics = lambda *a, **kw: mock_diag

        try:
            result = _precipitation_worker((1, kwargs))
            assert 'files' in result
        finally:
            _pw.MPASPrecipitationPlotter, _pw.PrecipitationDiagnostics = orig_plotter, orig_diag

    def test_surface_worker_cache_key_error_fallback(self: 'TestWorkerCacheKeyErrorFallback',) -> None:
        """
        This test verifies that the `_surface_worker` function correctly falls back to extracting coordinates from the processor when a KeyError occurs during cache coordinate retrieval. It mocks a cache that raises a KeyError for the 't2m' variable, and ensures that the worker function can still execute successfully by extracting coordinates directly from the processor. The test asserts that the result contains the expected keys and that the cache hit for coordinates is recorded as False, confirming that the fallback mechanism is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter = _pw.MPASSurfacePlotter

        mock_cache = Mock()
        mock_cache.get_coordinates.side_effect = KeyError('t2m')
        mock_cache.load_coordinates_from_dataset.side_effect = Exception("cache write failure")

        mock_plotter = Mock()
        mock_plotter.create_surface_map.return_value = (Mock(), Mock())

        kwargs = {
            'processor': self.mock_processor_surface,
            'cache': mock_cache,
            'output_dir': self.temp_dir,
            'lon_min': -120, 'lon_max': -80,
            'lat_min': 30, 'lat_max': 50,
            'var_name': 't2m',
            'plot_type': 'scatter',
            'file_prefix': 'test_surf_keyerr',
            'formats': ['png'],
            'custom_title': None,
            'colormap': None,
            'levels': None,
        }

        _pw.MPASSurfacePlotter = lambda *a, **kw: mock_plotter

        try:
            result = _surface_worker((1, kwargs))
            assert 'files' in result
            assert result['cache_hits'].get('coordinates') is not True
        finally:
            _pw.MPASSurfacePlotter = orig_plotter

    def test_wind_worker_cache_key_error_fallback(self: 'TestWorkerCacheKeyErrorFallback',) -> None:
        """
        This test verifies that the `_wind_worker` function correctly falls back to extracting coordinates from the processor when a KeyError occurs during cache coordinate retrieval. It mocks a cache that raises a KeyError for the 'u10' variable, and ensures that the worker function can still execute successfully by extracting coordinates directly from the processor. The test asserts that the result contains the expected keys and that the cache hit for coordinates is recorded as False, confirming that the fallback mechanism is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter = _pw.MPASWindPlotter

        mock_cache = Mock()
        mock_cache.get_coordinates.side_effect = KeyError('u10')
        mock_cache.load_coordinates_from_dataset.side_effect = Exception("cache write failure")

        mock_plotter = Mock()
        mock_plotter.create_wind_plot.return_value = (Mock(), Mock())

        kwargs = {
            'processor': self.mock_processor_wind,
            'cache': mock_cache,
            'output_dir': self.temp_dir,
            'lon_min': -120, 'lon_max': -80,
            'lat_min': 30, 'lat_max': 50,
            'u_variable': 'u10',
            'v_variable': 'v10',
            'plot_type': 'barbs',
            'subsample': 1,
            'scale': None,
            'show_background': False,
            'grid_resolution': None,
            'regrid_method': 'linear',
            'file_prefix': 'test_wind_keyerr',
            'formats': ['png'],
        }

        _pw.MPASWindPlotter = lambda *a, **kw: mock_plotter

        try:
            result = _wind_worker((1, kwargs))
            assert 'files' in result
            assert result['cache_hits'].get('coordinates') is not True
        finally:
            _pw.MPASWindPlotter = orig_plotter

    def test_wind_worker_no_time_dim_fallback(self: 'TestWorkerCacheKeyErrorFallback',) -> None:
        """
        This test verifies that the `_wind_worker` function can successfully execute and generate a time string when the input dataset does not contain a time dimension. It mocks a processor with a dataset that has no time dimension, and ensures that the worker function can still run without errors and produces a result with a time string that starts with 't', confirming that the worker can handle datasets without time coordinates by generating an appropriate time string for output files. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter = _pw.MPASWindPlotter

        n_cells = 100
        proc = Mock()
        proc.dataset = xr.Dataset({
            'u10': xr.DataArray(np.zeros((3, n_cells)), dims=['step', 'nCells']),
            'v10': xr.DataArray(np.zeros((3, n_cells)), dims=['step', 'nCells']),
        })
        proc.extract_2d_coordinates_for_variable.return_value = (self.lon, self.lat)
        proc.get_2d_variable_data.return_value = xr.DataArray(np.zeros(n_cells))

        mock_plotter = Mock()
        mock_plotter.create_wind_plot.return_value = (Mock(), Mock())

        kwargs = {
            'processor': proc,
            'cache': None,
            'output_dir': self.temp_dir,
            'lon_min': -120, 'lon_max': -80,
            'lat_min': 30, 'lat_max': 50,
            'u_variable': 'u10',
            'v_variable': 'v10',
            'plot_type': 'barbs',
            'subsample': 1,
            'scale': None,
            'show_background': False,
            'grid_resolution': None,
            'regrid_method': 'linear',
            'file_prefix': 'test_wind_notime',
            'formats': ['png'],
        }

        _pw.MPASWindPlotter = lambda *a, **kw: mock_plotter

        try:
            result = _wind_worker((1, kwargs))
            assert result['time_str'].startswith('t')
        finally:
            _pw.MPASWindPlotter = orig_plotter
