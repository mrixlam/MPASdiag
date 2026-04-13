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
import io
import pytest
import shutil
import tempfile
import builtins
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from contextlib import redirect_stdout
from typing import Any, List, Dict, Generator
from unittest.mock import Mock, MagicMock, patch

from mpasdiag.processing.parallel import ParallelStats, TaskResult, MPASParallelManager
from mpasdiag.processing.parallel_wrappers import (
    _precipitation_worker,
    _surface_worker,
    _wind_worker,
    _cross_section_worker,
    _process_parallel_results,
    ParallelPrecipitationProcessor,
    ParallelSurfaceProcessor,
    ParallelWindProcessor,
    ParallelCrossSectionProcessor,
    auto_batch_processor
)

from mpasdiag.processing.processors_3d import MPAS3DProcessor
from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter
from mpasdiag.visualization.surface import MPASSurfacePlotter
from mpasdiag.visualization.wind import MPASWindPlotter
from tests.test_data_helpers import assert_expected_public_methods

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
GRID_FILE = os.path.join(TEST_DATA_DIR, 'grids', 'x1.10242.static.nc')


class TestPrecipitationWorker:
    """ Tests for the precipitation worker wrapper that generates precipitation maps. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestPrecipitationWorker") -> Generator[None, None, None]:
        """
        This fixture sets up a temporary directory and a mock processor with a synthetic dataset for precipitation worker tests. The dataset includes a `rainnc` variable and coordinate arrays for longitude and latitude. The mock processor is configured to return this dataset and provide coordinate extraction functionality. After the test runs, the temporary directory is cleaned up.     

        Parameters:
            self ("TestPrecipitationWorker"): Test instance receiving temporary directories and mocks.

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()        
        self.mock_processor = Mock()
        n_time, n_cells = 5, 100
        
        self.mock_dataset = xr.Dataset({
            'rainnc': xr.DataArray(
                np.random.uniform(0, 50, (n_time, n_cells)),
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
    
    def test_precipitation_worker_success(self: "TestPrecipitationWorker") -> None:
        """
        This test verifies that the `_precipitation_worker` function successfully generates precipitation maps and returns the expected result structure. It mocks the `PrecipitationDiagnostics` to return a random precipitation difference dataset and the `MPASPrecipitationPlotter` to return mock figure and axis objects. The test asserts that the result contains the expected keys (`files`, `timings`, `time_str`, `cache_hits`) and that the `files` key contains a list, confirming that the worker function executed without errors and produced output in the correct format. 

        Parameters:
            self ("TestPrecipitationWorker"): Test instance with prepared fixtures.
            mock_diag_class (Any): Pytest mock for the diagnostics class constructor.
            mock_plotter_class (Any): Pytest mock for the plotter class constructor.

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter, orig_diag = _pw.MPASPrecipitationPlotter, _pw.PrecipitationDiagnostics
        
        assert_expected_public_methods(orig_plotter, 'MPASPrecipitationPlotter')
        assert_expected_public_methods(orig_diag, 'PrecipitationDiagnostics')

        mock_diag = Mock()
        mock_precip_data = xr.DataArray(np.random.uniform(0, 10, 100))
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
    
    def test_precipitation_worker_with_cache(self: "TestPrecipitationWorker") -> None:
        """
        This test ensures that the `_precipitation_worker` function correctly utilizes a provided cache for coordinate lookups. It mocks a cache that returns precomputed longitude and latitude values, and verifies that the worker records cache hits and calls the cache's `get_coordinates` method exactly once. This confirms that the caching mechanism is integrated properly and avoids redundant coordinate extraction. 

        Parameters:
            self ("TestPrecipitationWorker"): Test instance with prepared fixtures.
            mock_diag_class (Any): Pytest mock for the diagnostics class constructor.
            mock_plotter_class (Any): Pytest mock for the plotter class constructor.

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
        mock_precip_data = xr.DataArray(np.random.uniform(0, 10, 100))
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


class TestSurfaceWorker:
    """ Tests for the surface worker wrapper that generates surface variable maps. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestSurfaceWorker") -> Generator[None, None, None]: # type: ignore
        """
        This fixture sets up a temporary directory and a mock processor with a synthetic dataset containing a `t2m` variable for surface worker tests. The dataset includes coordinate arrays for longitude and latitude. The mock processor is configured to return this dataset and provide methods for extracting spatial coordinates. After the test runs, the temporary directory is removed to clean up resources. 

        Parameters:
            self ("TestSurfaceWorker"): Test instance to receive temporary resources.

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()        
        self.mock_processor = Mock()
        n_time, n_cells = 5, 100
        
        self.mock_dataset = xr.Dataset({
            't2m': xr.DataArray(
                np.random.uniform(250, 310, (n_time, n_cells)),
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
    
    def teardown_method(self: "TestSurfaceWorker") -> None:
        """
        This method cleans up temporary resources created for surface worker tests by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment. 

        Parameters:
            self ("TestSurfaceWorker"): Test instance owning temporary resources.

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_surface_worker_success(self: "TestSurfaceWorker") -> None:
        """
        This test verifies that the `_surface_worker` function successfully generates surface variable maps and returns the expected result structure. It mocks the `MPASSurfacePlotter` to return mock figure and axis objects, executes the worker with a prepared mock processor and specified kwargs, and asserts that the result contains the expected keys (`files`, `timings`, `time_str`) and that the `files` key contains a list, confirming that the worker function executed without errors and produced output in the correct format. 

        Parameters:
            self ("TestSurfaceWorker"): Test instance with a prepared mock processor.
            mock_plotter_class (Any): Pytest mock for the surface plotter constructor.

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


class TestWindWorker:
    """ Tests for the wind worker wrapper that generates wind plots. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestWindWorker") -> None:
        """
        This fixture sets up a temporary directory and a mock processor with a synthetic dataset containing `u10` and `v10` wind components for wind worker tests. The dataset includes coordinate arrays for longitude and latitude. The mock processor is configured to return this dataset and provide methods for extracting spatial coordinates. After the test runs, the temporary directory is removed to clean up resources. 

        Parameters:
            self ("TestWindWorker"): Test instance to receive created fixtures.

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()        
        self.mock_processor = Mock()
        n_time, n_cells = 5, 100
        
        self.mock_dataset = xr.Dataset({
            'u10': xr.DataArray(
                np.random.uniform(-10, 10, (n_time, n_cells)),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=n_time, freq='h')}
            ),
            'v10': xr.DataArray(
                np.random.uniform(-10, 10, (n_time, n_cells)),
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
    
    def teardown_method(self: "TestWindWorker") -> None:
        """
        This method cleans up temporary resources created for wind worker tests by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment. 

        Parameters:
            self ("TestWindWorker"): Test instance owning temporary resources.

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_wind_worker_success(self: "TestWindWorker") -> None:
        """
        This test verifies that the `_wind_worker` function successfully generates wind plots and returns the expected result structure. It mocks the `MPASWindPlotter` to return mock figure and axis objects, executes the worker with a prepared mock processor and specified kwargs, and asserts that the result contains the expected keys (`files`, `timings`, `time_str`) and that the `files` key contains a list, confirming that the worker function executed without errors and produced output in the correct format. 

        Parameters:
            self ("TestWindWorker"): Test instance with mock processor fixtures.
            mock_plotter_class (Any): Pytest mock for the wind plotter class.

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


class TestCrossSectionWorker:
    """ Tests for the cross-section worker wrapper that generates vertical cross-section plots. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestCrossSectionWorker") -> None:
        """
        This fixture sets up a temporary directory and a mock processor with a synthetic dataset containing a `temperature` variable for cross-section worker tests. The dataset includes coordinate arrays for time, vertical levels, and cells. The mock processor is configured to return this dataset, allowing the worker function to be invoked without accessing external data files. After the test runs, the temporary directory is removed to clean up resources. 

        Parameters:
            self ("TestCrossSectionWorker"): Test instance to receive mocked dataset and processor.

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()        
        self.mock_processor = Mock()
        n_time, n_levels, n_cells = 3, 20, 100
        
        self.mock_dataset = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.uniform(250, 310, (n_time, n_levels, n_cells)),
                dims=['Time', 'nVertLevels', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=n_time, freq='h')}
            )
        })
        
        self.mock_processor.dataset = self.mock_dataset
    
    def teardown_method(self: "TestCrossSectionWorker") -> None:
        """
        This method cleans up temporary resources created for cross-section worker tests by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment. 

        Parameters:
            self ("TestCrossSectionWorker"): Test instance owning the temporary directory.

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cross_section_worker_success(self: "TestCrossSectionWorker") -> None:
        """
        This test verifies that the `_cross_section_worker` function successfully generates vertical cross-section plots and returns the expected result structure. It mocks the `MPASVerticalCrossSectionPlotter` to return mock figure and axis objects, executes the worker with a prepared mock processor and specified kwargs, and asserts that the result contains the expected keys (`files`, `timings`, `time_str`) and that the `files` key contains a list, confirming that the worker function executed without errors and produced output in the correct format. 

        Parameters:
            self ("TestCrossSectionWorker"): Test instance with the prepared mock processor.
            mock_plotter_class (Any): Pytest mock for the vertical cross-section plotter.

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


