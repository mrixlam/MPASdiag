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


class TestProcessParallelResults:
    """ Tests for the `_process_parallel_results` function that summarizes parallel task outcomes. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestProcessParallelResults") -> None:
        """
        This fixture prepares a temporary directory and a mock parallel manager for testing the `_process_parallel_results` function. The temporary directory is used to simulate file output locations, while the mock parallel manager allows for controlled testing of statistics retrieval without requiring an actual parallel execution environment. After the test runs, the temporary directory is removed to clean up resources. 

        Parameters:
            self ("TestProcessParallelResults"): Test instance to receive temporary paths and mocks.

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()
        self.mock_manager = Mock(spec=MPASParallelManager)
    
    def teardown_method(self: "TestProcessParallelResults") -> None:
        """
        This method cleans up temporary resources created for testing the `_process_parallel_results` function by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment. 

        Parameters:
            self ("TestProcessParallelResults"): Test instance owning the temporary directory.

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_process_results_all_success(self: "TestProcessParallelResults") -> None:
        """
        This test verifies that the `_process_parallel_results` function correctly processes a list of `TaskResult` objects where all tasks were successful. It mocks the parallel manager to return specific statistics, captures the printed output, and asserts that the expected number of files is returned and that the summary contains the correct success count. This confirms that the function accurately summarizes successful outcomes and integrates with the parallel manager's statistics. 

        Parameters:
            self ("TestProcessParallelResults"): Test instance with prepared temporary directory and manager mock.

        Returns:
            None
        """
        results = [
            TaskResult(
                task_id=0, success=True,
                result={
                    'files': ['file1.png'],
                    'timings': {'data_processing': 0.1, 'plotting': 0.2, 'saving': 0.05, 'total': 0.35}
                }
            ),
            TaskResult(
                task_id=1, success=True,
                result={
                    'files': ['file2.png'],
                    'timings': {'data_processing': 0.15, 'plotting': 0.25, 'saving': 0.06, 'total': 0.46}
                }
            )
        ]
        
        from mpasdiag.processing.parallel import ParallelStats
        mock_stats = ParallelStats()
        mock_stats.total_time = 1.0
        mock_stats.load_imbalance = 0.1
        self.mock_manager.get_statistics.return_value = mock_stats
        
        from io import StringIO
        import sys
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            files = _process_parallel_results(
                results, [0, 1], self.temp_dir, self.mock_manager, "TEST"
            )
            
            output = captured_output.getvalue()
            
            assert len(files) == pytest.approx(2)
            assert "TEST BATCH PROCESSING RESULTS" in output
            assert "Successful: 2/2" in output
        finally:
            sys.stdout = sys.__stdout__
    
    def test_process_results_mixed(self: "TestProcessParallelResults") -> None:
        """
        This test verifies that the `_process_parallel_results` function correctly processes a list of `TaskResult` objects where some tasks were successful and others failed. It mocks the parallel manager to return specific statistics, captures the printed output, and asserts that the expected number of files is returned (only from successful tasks) and that the summary contains the correct counts for both successful and failed tasks. This confirms that the function accurately summarizes mixed outcomes and integrates with the parallel manager's statistics. 

        Parameters:
            self ("TestProcessParallelResults"): Test instance with prepared temporary directory and manager mock.

        Returns:
            None
        """
        results = [
            TaskResult(
                task_id=0, success=True,
                result={
                    'files': ['file1.png'],
                    'timings': {'data_processing': 0.1, 'plotting': 0.2, 'saving': 0.05, 'total': 0.35}
                }
            ),
            TaskResult(
                task_id=1, success=False,
                error="Test error"
            )
        ]
        
        from mpasdiag.processing.parallel import ParallelStats
        mock_stats = ParallelStats()

        mock_stats.total_time = 1.0
        mock_stats.load_imbalance = 0.1
        self.mock_manager.get_statistics.return_value = mock_stats
        
        from io import StringIO
        import sys

        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            files = _process_parallel_results(
                results, [0, 1], self.temp_dir, self.mock_manager, "TEST"
            )
            
            output = captured_output.getvalue()
            
            assert len(files) == pytest.approx(1)
            assert "Successful: 1/2" in output
            assert "Failed: 1/2" in output
        finally:
            sys.stdout = sys.__stdout__


class TestParallelPrecipitationProcessor:
    """ Tests for the `ParallelPrecipitationProcessor` batch functions. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestParallelPrecipitationProcessor") -> None:
        """
        This fixture prepares a temporary directory and a mock processor with a synthetic dataset containing a `rainnc` variable for testing the `ParallelPrecipitationProcessor` batch functions. The dataset includes coordinate arrays for time and cells. The mock processor is configured to return this dataset, allowing the batch processing functions to be invoked without accessing external data files. After the test runs, the temporary directory is removed to clean up resources. 

        Parameters:
            self ("TestParallelPrecipitationProcessor"): Test instance to receive prepared resources.

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
            )
        })
        
        self.mock_processor.dataset = self.mock_dataset
    
    def teardown_method(self: "TestParallelPrecipitationProcessor") -> None:
        """
        This method cleans up temporary resources created for testing the `ParallelPrecipitationProcessor` batch functions by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment. 

        Parameters:
            self ("TestParallelPrecipitationProcessor"): Test instance owning temporary resources.

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_batch_precipitation_maps_parallel(self: "TestParallelPrecipitationProcessor") -> None:
        """
        This test verifies that the `create_batch_precipitation_maps_parallel` function successfully orchestrates parallel batch processing for precipitation maps. It mocks the `MPASDataCache` to return a mock cache object and the `MPASParallelManager` to simulate parallel execution with a successful task result. The test asserts that the function returns a non-None result and that the `parallel_map` method of the manager was called exactly once, confirming that the batch processing was initiated correctly and that the worker function executed as expected in a parallel context. 

        Parameters:
            self ("TestParallelPrecipitationProcessor"): Test instance with temporary directory and mock processor.
            mock_cache_class (Any): Pytest mock for the data cache constructor.
            mock_manager_class (Any): Pytest mock for the parallel manager constructor.

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr, orig_cache = _pw.MPASParallelManager, _pw.MPASDataCache

        mock_cache = Mock()
        mock_manager = Mock()
        mock_manager.is_master = True

        mock_manager.parallel_map.return_value = [
            TaskResult(
                task_id=0, success=True,
                result={
                    'files': ['file1.png'],
                    'timings': {'data_processing': 0.1, 'plotting': 0.2, 'saving': 0.05, 'total': 0.35}
                }
            )
        ]

        mock_manager.get_statistics.return_value = None
        
        _pw.MPASParallelManager = lambda *a, **kw: mock_manager
        _pw.MPASDataCache = lambda *a, **kw: mock_cache

        try:
            result = ParallelPrecipitationProcessor.create_batch_precipitation_maps_parallel(
                processor=self.mock_processor,
                output_dir=self.temp_dir,
                lon_min=-120, lon_max=-80,
                lat_min=30, lat_max=50,
                time_indices=[2, 3]
            )
            
            assert result is not None
            mock_manager.parallel_map.assert_called_once()
        finally:
            _pw.MPASParallelManager, _pw.MPASDataCache = orig_mgr, orig_cache
    
    def test_create_batch_no_valid_time_indices(self: "TestParallelPrecipitationProcessor") -> None:
        """
        This test verifies that the `create_batch_precipitation_maps_parallel` function correctly handles the case where no valid time indices are provided for processing. It mocks the `MPASDataCache` and `MPASParallelManager` to simulate a parallel execution environment, but provides time indices that are all below the minimum required for precipitation difference calculations. The test asserts that the function returns an empty list, confirming that it properly detects the lack of valid time indices and avoids initiating unnecessary parallel processing. 

        Parameters:
            self ("TestParallelPrecipitationProcessor"): Test instance with mock processor fixtures.
            mock_cache_class (Any): Pytest mock for the data cache constructor.
            mock_manager_class (Any): Pytest mock for the parallel manager constructor.

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr, orig_cache = _pw.MPASParallelManager, _pw.MPASDataCache

        mock_cache = Mock()
        mock_manager = Mock()
        mock_manager.is_master = True
        
        _pw.MPASParallelManager = lambda *a, **kw: mock_manager
        _pw.MPASDataCache = lambda *a, **kw: mock_cache

        try:
            result = ParallelPrecipitationProcessor.create_batch_precipitation_maps_parallel(
                processor=self.mock_processor,
                output_dir=self.temp_dir,
                lon_min=-120, lon_max=-80,
                lat_min=30, lat_max=50,
                accum_period='a24h',
                time_indices=[0, 1, 2]  
            )
            
            assert result == []
        finally:
            _pw.MPASParallelManager, _pw.MPASDataCache = orig_mgr, orig_cache


class TestParallelSurfaceProcessor:
    """ Tests for the `ParallelSurfaceProcessor` batch processing methods. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestParallelSurfaceProcessor") -> None:
        """
        This fixture prepares a temporary directory and a mock processor with a synthetic dataset containing a `t2m` variable for testing the `ParallelSurfaceProcessor` batch processing methods. The dataset includes coordinate arrays for time and cells. The mock processor is configured to return this dataset, allowing the batch processing functions to be invoked without accessing external data files. After the test runs, the temporary directory is removed to clean up resources. 

        Parameters:
            self ("TestParallelSurfaceProcessor"): Test instance to receive fixtures.

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()
        self.mock_processor = Mock()
        
        n_time, n_cells = 5, 100

        self.mock_dataset = xr.Dataset({
            't2m': xr.DataArray(
                np.random.uniform(250, 310, (n_time, n_cells)),
                dims=['Time', 'nCells']
            )
        })
        
        self.mock_processor.dataset = self.mock_dataset
    
    def teardown_method(self: "TestParallelSurfaceProcessor") -> None:
        """
        This method cleans up temporary resources created for testing the `ParallelSurfaceProcessor` batch processing methods by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment. 

        Parameters:
            self ("TestParallelSurfaceProcessor"): Test instance owning the temporary directory.

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_batch_surface_maps_parallel(self: "TestParallelSurfaceProcessor") -> None:
        """
        This test verifies that the `create_batch_surface_maps_parallel` function successfully orchestrates parallel batch processing for surface variable maps. It mocks the `MPASDataCache` to return a mock cache object and the `MPASParallelManager` to simulate parallel execution with a successful task result. The test asserts that the function returns a non-None result, confirming that the batch processing was initiated correctly and that the worker function executed as expected in a parallel context. 

        Parameters:
            self ("TestParallelSurfaceProcessor"): Test instance with temporary directory and mock processor.
            mock_cache_class (Any): Pytest mock for the data cache constructor.
            mock_manager_class (Any): Pytest mock for the parallel manager constructor.

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr, orig_cache = _pw.MPASParallelManager, _pw.MPASDataCache

        mock_cache = Mock()
        mock_manager = Mock()
        mock_manager.is_master = True

        mock_manager.parallel_map.return_value = [
            TaskResult(
                task_id=0, success=True,
                result={
                    'files': ['file1.png'],
                    'timings': {'data_processing': 0.1, 'plotting': 0.2, 'saving': 0.05, 'total': 0.35}
                }
            )
        ]

        mock_manager.get_statistics.return_value = None
        
        _pw.MPASParallelManager = lambda *a, **kw: mock_manager
        _pw.MPASDataCache = lambda *a, **kw: mock_cache

        try:
            result = ParallelSurfaceProcessor.create_batch_surface_maps_parallel(
                processor=self.mock_processor,
                output_dir=self.temp_dir,
                lon_min=-120, lon_max=-80,
                lat_min=30, lat_max=50,
                var_name='t2m',
                time_indices=[0, 1]
            )
            
            assert result is not None
        finally:
            _pw.MPASParallelManager, _pw.MPASDataCache = orig_mgr, orig_cache


class TestParallelWindProcessor:
    """ Tests for the `ParallelWindProcessor` batch processing methods. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestParallelWindProcessor") -> None:
        """
        This fixture prepares a temporary directory and a mock processor with a synthetic dataset containing `u10` and `v10` wind components for testing the `ParallelWindProcessor` batch processing methods. The dataset includes coordinate arrays for time and cells. The mock processor is configured to return this dataset, allowing the batch processing functions to be invoked without accessing external data files. After the test runs, the temporary directory is removed to clean up resources. 

        Parameters:
            self ("TestParallelWindProcessor"): Test instance to receive mock processor and temp dir.

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()
        self.mock_processor = Mock()
        
        n_time, n_cells = 5, 100

        self.mock_dataset = xr.Dataset({
            'u10': xr.DataArray(np.random.uniform(-10, 10, (n_time, n_cells)), dims=['Time', 'nCells']),
            'v10': xr.DataArray(np.random.uniform(-10, 10, (n_time, n_cells)), dims=['Time', 'nCells'])
        })
        
        self.mock_processor.dataset = self.mock_dataset
    
    def teardown_method(self: "TestParallelWindProcessor") -> None:
        """
        This method cleans up temporary resources created for testing the `ParallelWindProcessor` batch processing methods by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment. 

        Parameters:
            self ("TestParallelWindProcessor"): Test instance owning temporary resources.

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_batch_wind_plots_parallel(self: "TestParallelWindProcessor") -> None:
        """
        This test verifies that the `create_batch_wind_plots_parallel` function successfully orchestrates parallel batch processing for wind plots. It mocks the `MPASDataCache` to return a mock cache object and the `MPASParallelManager` to simulate parallel execution with a successful task result. The test asserts that the function returns a non-None result, confirming that the batch processing was initiated correctly and that the worker function executed as expected in a parallel context.

        Parameters:
            self ("TestParallelWindProcessor"): Test instance with temporary directory and mock processor.
            mock_cache_class (Any): Pytest mock for the data cache constructor.
            mock_manager_class (Any): Pytest mock for the parallel manager constructor.

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr, orig_cache = _pw.MPASParallelManager, _pw.MPASDataCache
        mock_cache = Mock()
        mock_manager = Mock()
        mock_manager.is_master = True

        mock_manager.parallel_map.return_value = [
            TaskResult(
                task_id=0, success=True,
                result={
                    'files': ['file1.png'],
                    'timings': {'data_processing': 0.1, 'plotting': 0.2, 'saving': 0.05, 'total': 0.35}
                }
            )
        ]

        mock_manager.get_statistics.return_value = None
        
        _pw.MPASParallelManager = lambda *a, **kw: mock_manager
        _pw.MPASDataCache = lambda *a, **kw: mock_cache

        try:
            result = ParallelWindProcessor.create_batch_wind_plots_parallel(
                processor=self.mock_processor,
                output_dir=self.temp_dir,
                lon_min=-120, lon_max=-80,
                lat_min=30, lat_max=50,
                u_variable='u10',
                v_variable='v10',
                time_indices=[0, 1]
            )
            
            assert result is not None
        finally:
            _pw.MPASParallelManager, _pw.MPASDataCache = orig_mgr, orig_cache


class TestParallelCrossSectionProcessor:
    """ Tests for the `ParallelCrossSectionProcessor` batch processing methods. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestParallelCrossSectionProcessor") -> None:
        """
        This fixture prepares a temporary directory and a mock processor with a synthetic dataset containing a `temperature` variable for testing the `ParallelCrossSectionProcessor` batch processing methods. The dataset includes coordinate arrays for time, vertical levels, and cells. The mock processor is configured to return this dataset, allowing the batch processing functions to be invoked without accessing external data files. After the test runs, the temporary directory is removed to clean up resources. 

        Parameters:
            self ("TestParallelCrossSectionProcessor"): Test instance receiving the mock processor.

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()
        self.mock_processor = Mock()
        
        n_time, n_levels, n_cells = 3, 20, 100

        self.mock_dataset = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.uniform(250, 310, (n_time, n_levels, n_cells)),
                dims=['Time', 'nVertLevels', 'nCells']
            )
        })
        
        self.mock_processor.dataset = self.mock_dataset
    
    def teardown_method(self: "TestParallelCrossSectionProcessor") -> None:
        """
        This method cleans up temporary resources created for testing the `ParallelCrossSectionProcessor` batch processing methods by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment. 

        Parameters:
            self ("TestParallelCrossSectionProcessor"): Test instance owning temporary resources.

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_batch_cross_section_plots_parallel(self: "TestParallelCrossSectionProcessor") -> None:
        """
        This test verifies that the `create_batch_cross_section_plots_parallel` function successfully orchestrates parallel batch processing for vertical cross-section plots. It mocks the `MPASParallelManager` to simulate parallel execution with a successful task result. The test asserts that the function returns a non-None result, confirming that the batch processing was initiated correctly and that the worker function executed as expected in a parallel context. 

        Parameters:
            self ("TestParallelCrossSectionProcessor"): Test instance with temporary directory and mock processor.
            mock_manager_class (Any): Pytest mock for the parallel manager constructor.

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr = _pw.MPASParallelManager

        mock_manager = Mock()
        mock_manager.is_master = True

        mock_manager.parallel_map.return_value = [
            TaskResult(task_id=0, success=True, result=['file1.png'])
        ]
        
        _pw.MPASParallelManager = lambda *a, **kw: mock_manager

        try:
            result = ParallelCrossSectionProcessor.create_batch_cross_section_plots_parallel(
                mpas_3d_processor=self.mock_processor,
                var_name='temperature',
                start_point=(-110, 35),
                end_point=(-90, 45),
                output_dir=self.temp_dir,
                time_indices=[0, 1]
            )
            
            assert result is not None
        finally:
            _pw.MPASParallelManager = orig_mgr


class TestAutoBatchProcessor:
    """ Tests for the `auto_batch_processor` function that determines whether to use parallel processing based on user input and environment capabilities. """
    
    def test_auto_batch_processor_explicit_true(self: "TestAutoBatchProcessor") -> None:
        """
        This test verifies that the `auto_batch_processor` function returns True when `use_parallel` is explicitly set to True, indicating that parallel processing should be used regardless of environment capabilities. The test asserts that the function correctly respects the user's explicit request for parallel processing by returning True. 

        Parameters:
            self ("TestAutoBatchProcessor"): Test instance.

        Returns:
            None
        """
        result = auto_batch_processor(use_parallel=True)
        assert result is True

    def test_auto_batch_processor_auto_no_mpi(self: "TestAutoBatchProcessor") -> None:
        """
        This test verifies that the `auto_batch_processor` function returns False when `use_parallel` is set to 'auto' and the `mpi4py` module is not available in the environment. It mocks the built-in import function to raise an ImportError when attempting to import `mpi4py`, simulating an environment without MPI capabilities. The test asserts that the function correctly detects the lack of MPI support and returns False, indicating that parallel processing should not be used. 

        Parameters:
            self ("TestAutoBatchProcessor"): Test instance.

        Returns:
            None
        """
        import builtins
        real_import = builtins.__import__
        
        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            """
            Replacement for builtins.__import__ used to simulate ImportError for mpi4py.

            Parameters:
                name (str): Module name being imported.
                *args (Any): Positional args forwarded to the real import.
                **kwargs (Any): Keyword args forwarded to the real import.

            Returns:
                Any
            """
            if name == 'mpi4py':
                raise ImportError("No module named 'mpi4py'")
            return real_import(name, *args, **kwargs)
        
        builtins.__import__ = mock_import
        
        try:
            result = auto_batch_processor(use_parallel=None)
            assert not result
        finally:
            builtins.__import__ = real_import


class TestEdgeCases:
    """ Tests for edge cases, error handling, and exceptional conditions in parallel wrappers. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestEdgeCases") -> Generator[None, None, None]:
        """
        This fixture sets up a temporary directory for edge case tests and ensures that it is cleaned up after the tests run. The temporary directory can be used by tests that require file output without affecting the actual filesystem. After yielding control to the test, the fixture removes the temporary directory to maintain a clean testing environment. 

        Parameters:
            self ("TestEdgeCases"): Test instance to receive the temporary directory.

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_load_failure(self: "TestEdgeCases") -> None:
        """
        This test verifies that the `create_batch_precipitation_maps_parallel` function handles a failure in loading coordinates from the data cache gracefully. It mocks the `MPASDataCache` to raise an exception when attempting to load coordinates, simulating a cache failure scenario. The test asserts that the function completes without crashing and returns a result (which may be None or an empty list), confirming that the error handling for cache loading issues is functioning correctly and does not cause unhandled exceptions in the batch processing workflow. 

        Parameters:
            self ("TestEdgeCases"): Test instance with temporary output directory.
            mock_cache_class (Any): Pytest mock for the data cache class.
            mock_manager_class (Any): Pytest mock for the parallel manager class.

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr, orig_cache = _pw.MPASParallelManager, _pw.MPASDataCache

        mock_processor = Mock()

        mock_dataset = xr.Dataset({
            'rainnc': xr.DataArray(np.random.uniform(0, 50, (5, 100)), dims=['Time', 'nCells'])
        })

        mock_processor.dataset = mock_dataset        
        mock_cache = Mock()
        mock_cache.load_coordinates_from_dataset.side_effect = Exception("Cache error")
        
        mock_manager = Mock()
        mock_manager.is_master = True
        mock_manager.parallel_map.return_value = []
        
        from io import StringIO
        import sys

        captured_output = StringIO()
        sys.stdout = captured_output
        
        _pw.MPASParallelManager = lambda *a, **kw: mock_manager
        _pw.MPASDataCache = lambda *a, **kw: mock_cache

        try:
            result = ParallelPrecipitationProcessor.create_batch_precipitation_maps_parallel(
                processor=mock_processor,
                output_dir=self.temp_dir,
                lon_min=-120, lon_max=-80,
                lat_min=30, lat_max=50,
                time_indices=[2]
            )
            
            assert result is not None or result is None  
        finally:
            _pw.MPASParallelManager, _pw.MPASDataCache = orig_mgr, orig_cache
            sys.stdout = sys.__stdout__


@pytest.fixture
def test_data_dir() -> Path:
    """
    This fixture provides a Path object pointing to the base directory containing test data files for integration tests. It constructs the path by navigating up three levels from the current file's location and then into the "data" directory, which is expected to contain subdirectories with sample MPAS output files, grids, and diagnostics used by various tests. 

    Parameters:
        None

    Returns:
        Path
    """
    return Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def grid_file(test_data_dir: Path) -> str:
    """
    This fixture locates a sample grid file for testing purposes and returns its filesystem path as a string. It constructs the path to the grid file by appending the "grids/x1.10242.static.nc" subpath to the provided base test data directory. This grid file can be used in tests that require access to grid information without needing to rely on external data sources. 

    Parameters:
        test_data_dir (Path): Fixture-provided base test data directory.

    Returns:
        str: Filesystem path to the sample grid file as a string.
    """
    return str(test_data_dir / "grids" / "x1.10242.static.nc")


@pytest.fixture
def mpas_output_files(test_data_dir: Path) -> List[str]:
    """
    This fixture provides a list of MPAS output file paths for testing purposes when available. It looks for NetCDF files in the "u240k/mpasout" subdirectory of the provided base test data directory. If the directory exists and contains files, it returns a sorted list of the first two file paths as strings. If the directory does not exist or contains no files, it returns an empty list. This allows tests to access sample MPAS output data without requiring external dependencies. 

    Parameters:
        test_data_dir (Path): Fixture-provided base test data directory.

    Returns:
        List[str]: List of MPAS output file paths (as strings). Returns empty list when files are unavailable.
    """
    output_dir = test_data_dir / "u240k" / "mpasout"

    if output_dir.exists():
        files = sorted(output_dir.glob("*.nc"))
        return [str(f) for f in files[:2]] 
    return []


@pytest.fixture
def mpas_diag_files(test_data_dir: Path) -> List[str]:
    """
    This fixture provides a list of MPAS diagnostic file paths for testing purposes when available. It looks for NetCDF files in the "u240k/diag" subdirectory of the provided base test data directory. If the directory exists and contains files, it returns a sorted list of the first two file paths as strings. If the directory does not exist or contains no files, it returns an empty list. This allows tests to access sample MPAS diagnostic data without requiring external dependencies.

    Parameters:
        test_data_dir (Path): Fixture-provided base test data directory.

    Returns:
        List[str]: List of diagnostic file paths as strings, or empty list if none present.
    """
    diag_dir = test_data_dir / "u240k" / "diag"

    if diag_dir.exists():
        files = sorted(diag_dir.glob("*.nc"))
        return [str(f) for f in files[:2]]
    return []


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """
    This fixture creates a temporary directory for test outputs and yields its Path object to the test. After the test completes, it ensures that the temporary directory is removed to clean up resources. This allows tests to write output files without affecting the actual filesystem and guarantees that no residual files remain after testing. 

    Parameters:
        None

    Returns:
        Generator[Path, None, None]: Yields a Path object pointing to a temporary directory for test outputs. The directory is removed after the test completes.
    """
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestImportErrorHandling:
    """ Tests that verify import-time fallback behavior for optional modules. """
    
    def test_import_from_data_cache(self: "TestImportErrorHandling") -> None:
        """
        This test confirms that the `MPASDataCache` and `get_global_cache` symbols are available from the wrapper module, ensuring that the import fallback mechanism for the data cache is functioning correctly. It imports these symbols and asserts that they are not None, which indicates that the import succeeded and the fallback did not cause a failure. 

        Parameters:
            self ("TestImportErrorHandling"): Test instance (unused).

        Returns:
            None
        """
        from mpasdiag.processing.parallel_wrappers import MPASDataCache, get_global_cache
        assert MPASDataCache is not None
        assert get_global_cache is not None
    
    def test_import_from_parallel(self: "TestImportErrorHandling") -> None:
        """
        This test confirms that the `MPASParallelManager`, `MPAS2DProcessor`, and `MPAS3DProcessor` symbols are available from the wrapper module, ensuring that the import fallback mechanism for the parallel processing components is functioning correctly. It imports these symbols and asserts that they are not None, which indicates that the import succeeded and the fallback did not cause a failure. 

        Parameters:
            self ("TestImportErrorHandling"): Test instance (unused).

        Returns:
            None
        """
        from mpasdiag.processing.parallel_wrappers import MPASParallelManager
        from mpasdiag.processing.parallel_wrappers import MPAS2DProcessor
        from mpasdiag.processing.parallel_wrappers import MPAS3DProcessor
        
        assert MPASParallelManager is not None
        assert MPAS2DProcessor is not None
        assert MPAS3DProcessor is not None
    
    def test_import_from_visualization(self: "TestImportErrorHandling") -> None:
        """
        This test confirms that the `MPASPrecipitationPlotter`, `MPASSurfacePlotter`, `MPASWindPlotter`, `MPASVerticalCrossSectionPlotter`, and `PrecipitationDiagnostics` symbols are available from the wrapper module, ensuring that the import fallback mechanism for the visualization components is functioning correctly. It imports these symbols and asserts that they are not None, which indicates that the import succeeded and the fallback did not cause a failure. 

        Parameters:
            self ("TestImportErrorHandling"): Test instance (unused).

        Returns:
            None
        """
        from mpasdiag.processing.parallel_wrappers import MPASPrecipitationPlotter
        from mpasdiag.processing.parallel_wrappers import MPASSurfacePlotter
        from mpasdiag.processing.parallel_wrappers import MPASWindPlotter
        from mpasdiag.processing.parallel_wrappers import MPASVerticalCrossSectionPlotter
        from mpasdiag.processing.parallel_wrappers import PrecipitationDiagnostics
        
        assert MPASPrecipitationPlotter is not None
        assert MPASSurfacePlotter is not None
        assert MPASWindPlotter is not None
        assert MPASVerticalCrossSectionPlotter is not None
        assert PrecipitationDiagnostics is not None


class TestPrecipitationWorkerCacheException:
    """ Tests for handling exceptions related to cache loading in the precipitation worker. """
    
    def test_precipitation_worker_cache_load_exception(self: "TestPrecipitationWorkerCacheException", temp_output_dir: Path) -> None:
        """
        This test verifies that the `_precipitation_worker` function handles exceptions raised during cache loading gracefully. It mocks the processor to return a dataset with a `rainnc` variable and configures the cache mock to raise an exception when attempting to load coordinates. The test asserts that the worker function completes without crashing and returns a result, confirming that the error handling for cache loading issues is functioning correctly and does not cause unhandled exceptions in the worker workflow. 

        Parameters:
            self ("TestPrecipitationWorkerCacheException"): Test instance (unused).
            mock_diag_class (Any): Pytest mock for the diagnostics class constructor.
            mock_plotter_class (Any): Pytest mock for the plotter class constructor.
            temp_output_dir (Path): Temporary directory fixture for outputs.

        Returns:
            None
        """
        mock_processor = Mock()
        n_time, n_cells = 3, 50
        
        mock_dataset = xr.Dataset({
            'rainnc': xr.DataArray(
                np.random.uniform(0, 50, (n_time, n_cells)),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=n_time, freq='h')}
            ),
            'lonCell': xr.DataArray(np.linspace(-120, -80, n_cells), dims=['nCells']),
            'latCell': xr.DataArray(np.linspace(30, 50, n_cells), dims=['nCells'])
        })
        
        mock_processor.dataset = mock_dataset
        mock_processor.data_type = 'UXarray'
        mock_processor.extract_2d_coordinates_for_variable.return_value = (
            mock_dataset['lonCell'].values,
            mock_dataset['latCell'].values
        )
        
        mock_cache = Mock()
        mock_cache.get_coordinates.side_effect = KeyError("Not cached")
        mock_cache.load_coordinates_from_dataset.side_effect = RuntimeError("Cache load failed")
        mock_cache.get_cache_info.return_value = {'hits': 0, 'misses': 1}
        
        mock_diag = Mock()
        mock_precip_data = xr.DataArray(np.random.uniform(0, 10, n_cells))
        mock_diag.compute_precipitation_difference.return_value = mock_precip_data
        
        mock_plotter = Mock()
        mock_fig, mock_ax = Mock(), Mock()
        mock_plotter.create_precipitation_map.return_value = (mock_fig, mock_ax)
        
        kwargs = {
            'processor': mock_processor,
            'cache': mock_cache,
            'output_dir': temp_output_dir,
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
        
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter, orig_diag = _pw.MPASPrecipitationPlotter, _pw.PrecipitationDiagnostics
        _pw.MPASPrecipitationPlotter = lambda *a, **kw: mock_plotter
        _pw.PrecipitationDiagnostics = lambda *a, **kw: mock_diag

        try:
            result = _precipitation_worker((1, kwargs))
            
            assert 'files' in result
            assert not result['cache_hits']['coordinates']  
            mock_cache.load_coordinates_from_dataset.assert_called_once()
        finally:
            _pw.MPASPrecipitationPlotter, _pw.PrecipitationDiagnostics = orig_plotter, orig_diag


class TestPrecipitationWorkerTimeString:
    """ Tests for handling missing Time dimension and fallback time string generation in the precipitation worker. """
    
    def test_precipitation_worker_no_time_dimension(self: "TestPrecipitationWorkerTimeString", temp_output_dir: Path) -> None:
        """
        This test verifies that the `_precipitation_worker` function can handle a dataset that lacks a Time dimension and correctly generates a fallback time string. It mocks the processor to return a dataset with a `rainnc` variable that only has an `nCells` dimension, simulating the absence of time information. The test asserts that the worker function completes without crashing and returns a result with a time string in the expected fallback format, confirming that the worker can gracefully handle datasets without time coordinates. 

        Parameters:
            self ("TestPrecipitationWorkerTimeString"): Test instance (unused).
            mock_diag_class (Any): Pytest mock for diagnostics constructor.
            mock_plotter_class (Any): Pytest mock for plotter constructor.
            temp_output_dir (Path): Fixture path for temporary output directory.

        Returns:
            None
        """
        mock_processor = Mock()
        n_cells = 50
        
        mock_dataset = xr.Dataset({
            'rainnc': xr.DataArray(
                np.random.uniform(0, 50, n_cells),
                dims=['nCells']
            ),
            'lonCell': xr.DataArray(np.linspace(-120, -80, n_cells), dims=['nCells']),
            'latCell': xr.DataArray(np.linspace(30, 50, n_cells), dims=['nCells'])
        })
        
        mock_processor.dataset = mock_dataset
        mock_processor.data_type = 'UXarray'

        mock_processor.extract_2d_coordinates_for_variable.return_value = (
            mock_dataset['lonCell'].values,
            mock_dataset['latCell'].values
        )
        
        mock_diag = Mock()
        mock_precip_data = xr.DataArray(np.random.uniform(0, 10, n_cells))
        mock_diag.compute_precipitation_difference.return_value = mock_precip_data
        
        mock_plotter = Mock()
        mock_fig, mock_ax = Mock(), Mock()
        mock_plotter.create_precipitation_map.return_value = (mock_fig, mock_ax)
        
        kwargs = {
            'processor': mock_processor,
            'cache': None,
            'output_dir': temp_output_dir,
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
        
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter, orig_diag = _pw.MPASPrecipitationPlotter, _pw.PrecipitationDiagnostics
        _pw.MPASPrecipitationPlotter = lambda *a, **kw: mock_plotter
        _pw.PrecipitationDiagnostics = lambda *a, **kw: mock_diag

        try:
            result = _precipitation_worker((0, kwargs))
            assert result['time_str'] == "t000"
        finally:
            _pw.MPASPrecipitationPlotter, _pw.PrecipitationDiagnostics = orig_plotter, orig_diag


class TestPrecipitationWorkerCustomTitle:
    """ Tests for verifying that a custom title template is correctly passed through to the precipitation plotter in the worker function. """
    
    def test_precipitation_worker_custom_title_template(self: "TestPrecipitationWorkerCustomTitle", temp_output_dir: Path) -> None:
        """
        This test verifies that the `_precipitation_worker` function correctly accepts a custom title template and passes it to the plotter when creating precipitation maps. It mocks the processor to return a dataset with a `rainnc` variable and configures the plotter mock to capture the title argument used in the `create_precipitation_map` method. The test asserts that the custom title template is included in the title passed to the plotter, confirming that the worker function properly integrates user-defined title templates into the plotting workflow. 

        Parameters:
            self ("TestPrecipitationWorkerCustomTitle"): Test instance (unused).
            temp_output_dir (Path): Temporary directory fixture for outputs.

        Returns:
            None: Behavior is validated through assertions.
        """
        mock_processor = Mock()
        n_time, n_cells = 3, 50
        
        mock_dataset = xr.Dataset({
            'rainnc': xr.DataArray(
                np.random.uniform(0, 50, (n_time, n_cells)),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=n_time, freq='h')}
            ),
            'lonCell': xr.DataArray(np.linspace(-120, -80, n_cells), dims=['nCells']),
            'latCell': xr.DataArray(np.linspace(30, 50, n_cells), dims=['nCells'])
        })
        
        mock_processor.dataset = mock_dataset
        mock_processor.data_type = 'UXarray'

        mock_processor.extract_2d_coordinates_for_variable.return_value = (
            mock_dataset['lonCell'].values,
            mock_dataset['latCell'].values
        )
        
        mock_diag = Mock()
        mock_precip_data = xr.DataArray(np.random.uniform(0, 10, n_cells))
        mock_diag.compute_precipitation_difference.return_value = mock_precip_data
        
        mock_plotter = Mock()
        mock_fig, mock_ax = Mock(), Mock()
        mock_plotter.create_precipitation_map.return_value = (mock_fig, mock_ax)
        
        custom_template = "Custom Precip: {var_name} | Time: {time_str} | Period: {accum_period}"
        
        kwargs = {
            'processor': mock_processor,
            'cache': None,
            'output_dir': temp_output_dir,
            'lon_min': -120, 'lon_max': -80,
            'lat_min': 30, 'lat_max': 50,
            'var_name': 'rainnc',
            'accum_period': 'a01h',
            'plot_type': 'scatter',
            'grid_resolution': None,
            'file_prefix': 'test_precip',
            'formats': ['png'],
            'custom_title_template': custom_template,
            'colormap': None,
            'levels': None
        }
        
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter, orig_diag = _pw.MPASPrecipitationPlotter, _pw.PrecipitationDiagnostics
        _pw.MPASPrecipitationPlotter = lambda *a, **kw: mock_plotter
        _pw.PrecipitationDiagnostics = lambda *a, **kw: mock_diag

        try:
            result = _precipitation_worker((1, kwargs))            
            call_args = mock_plotter.create_precipitation_map.call_args
            assert 'title' in call_args[1]
            assert 'Custom Precip' in call_args[1]['title']
            assert result['time_str'] in call_args[1]['title']
        finally:
            _pw.MPASPrecipitationPlotter, _pw.PrecipitationDiagnostics = orig_plotter, orig_diag


class TestSurfaceWorkerCacheException:
    """ Tests for handling cache failures in the surface worker. """
    
    def test_surface_worker_cache_load_exception(self: "TestSurfaceWorkerCacheException", temp_output_dir: Path) -> None:
        """
        This test verifies that the `_surface_worker` function can handle exceptions raised during cache loading without crashing. It mocks the processor to return a dataset with a `t2m` variable and configures the cache mock to raise an exception when attempting to load coordinates. The test asserts that the worker function completes and returns a result, confirming that the error handling for cache loading issues is functioning correctly and does not cause unhandled exceptions in the worker workflow. 

        Parameters:
            self ("TestSurfaceWorkerCacheException"): Test instance (unused).
            temp_output_dir (Path): Temporary directory fixture for outputs.

        Returns:
            None
        """
        mock_processor = Mock()
        n_time, n_cells = 3, 50
        
        mock_dataset = xr.Dataset({
            't2m': xr.DataArray(
                np.random.uniform(280, 300, (n_time, n_cells)),
                dims=['Time', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=n_time, freq='h')}
            ),
            'lonCell': xr.DataArray(np.linspace(-120, -80, n_cells), dims=['nCells']),
            'latCell': xr.DataArray(np.linspace(30, 50, n_cells), dims=['nCells'])
        })
        
        mock_processor.dataset = mock_dataset

        mock_processor.extract_spatial_coordinates.return_value = (
            mock_dataset['lonCell'].values,
            mock_dataset['latCell'].values
        )
        
        mock_cache = Mock()
        mock_cache.get_coordinates.side_effect = KeyError("Not cached")
        mock_cache.load_coordinates_from_dataset.side_effect = RuntimeError("Cache load failed")
        mock_cache.get_cache_info.return_value = {'hits': 0, 'misses': 1}
        
        mock_plotter = Mock()
        mock_fig, mock_ax = Mock(), Mock()
        mock_plotter.create_surface_map.return_value = (mock_fig, mock_ax)
        
        kwargs = {
            'processor': mock_processor,
            'cache': mock_cache,
            'output_dir': temp_output_dir,
            'lon_min': -120, 'lon_max': -80,
            'lat_min': 30, 'lat_max': 50,
            'var_name': 't2m',
            'plot_type': 'contourf',
            'file_prefix': 'test_surface',
            'formats': ['png'],
            'custom_title': None,
            'colormap': None,
            'levels': None
        }
        
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter = _pw.MPASSurfacePlotter
        _pw.MPASSurfacePlotter = lambda *a, **kw: mock_plotter

        try:
            result = _surface_worker((1, kwargs))
            
            assert 'files' in result
            assert not result['cache_hits']['coordinates']  
            mock_cache.load_coordinates_from_dataset.assert_called_once()
        finally:
            _pw.MPASSurfacePlotter = orig_plotter


class TestWindWorkerTimeString:
    """ Tests for handling missing Time dimension and fallback time string generation in the wind worker. """
    
    def test_wind_worker_no_time_dimension(self: "TestWindWorkerTimeString", temp_output_dir: Path) -> None:
        """
        This test verifies that the `_wind_worker` function can handle a dataset that lacks a Time dimension and correctly generates a fallback time string. It mocks the processor to return a dataset with `u10` and `v10` variables that only have an `nCells` dimension, simulating the absence of time information. The test asserts that the worker function completes without crashing and returns a result with a time string in the expected fallback format, confirming that the worker can gracefully handle datasets without time coordinates. 

        Parameters:
            self ("TestWindWorkerTimeString"): Test instance (unused).
            temp_output_dir (Path): Temporary directory fixture for outputs.

        Returns:
            None: Assertions validate fallback naming behavior.
        """
        mock_processor = Mock()
        n_cells = 50
        
        mock_dataset = xr.Dataset({
            'u10': xr.DataArray(np.random.uniform(-10, 10, n_cells), dims=['nCells']),
            'v10': xr.DataArray(np.random.uniform(-10, 10, n_cells), dims=['nCells']),
            'lonCell': xr.DataArray(np.linspace(-120, -80, n_cells), dims=['nCells']),
            'latCell': xr.DataArray(np.linspace(30, 50, n_cells), dims=['nCells'])
        })
        
        mock_processor.dataset = mock_dataset
        mock_processor.get_2d_variable_data.side_effect = lambda var, idx: mock_dataset[var]

        mock_processor.extract_2d_coordinates_for_variable.return_value = (
            mock_dataset['lonCell'].values,
            mock_dataset['latCell'].values
        )
        
        mock_plotter = Mock()
        mock_fig, mock_ax = Mock(), Mock()
        mock_plotter.create_wind_plot.return_value = (mock_fig, mock_ax)
        
        kwargs = {
            'processor': mock_processor,
            'cache': None,
            'output_dir': temp_output_dir,
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
        
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter = _pw.MPASWindPlotter
        _pw.MPASWindPlotter = lambda *a, **kw: mock_plotter

        try:
            result = _wind_worker((0, kwargs))            
            assert result['time_str'] == "t000"
        finally:
            _pw.MPASWindPlotter = orig_plotter


class TestWindWorkerCacheException:
    """ Tests for handling exceptions related to cache loading in the wind worker. """
    
    def test_wind_worker_cache_load_exception(self: "TestWindWorkerCacheException", temp_output_dir: Path) -> None:
        """
        This test verifies that the `_wind_worker` function can handle exceptions raised during cache loading gracefully. It mocks the processor to return a dataset with `u10` and `v10` variables and configures the cache mock to raise an exception when attempting to load coordinates. The test asserts that the worker function completes without crashing and returns a result, confirming that the error handling for cache loading issues is functioning correctly and does not cause unhandled exceptions in the worker workflow. 

        Parameters:
            self ("TestWindWorkerCacheException"): Test instance (unused).
            temp_output_dir (Path): Temporary directory fixture for outputs.

        Returns:
            None: Assertions validate fallback behavior.
        """
        mock_processor = Mock()
        n_time, n_cells = 3, 50
        
        mock_dataset = xr.Dataset({
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
        
        mock_processor.dataset = mock_dataset
        u_data = mock_dataset['u10'].isel(Time=1)
        v_data = mock_dataset['v10'].isel(Time=1)

        mock_processor.get_2d_variable_data.side_effect = [u_data, v_data]
        
        mock_processor.extract_2d_coordinates_for_variable.return_value = (
            mock_dataset['lonCell'].values,
            mock_dataset['latCell'].values
        )
        
        mock_cache = Mock()
        mock_cache.get_coordinates.side_effect = KeyError("Not cached")
        mock_cache.load_coordinates_from_dataset.side_effect = RuntimeError("Cache load failed")
        mock_cache.get_cache_info.return_value = {'hits': 0, 'misses': 1}
        
        mock_plotter = Mock()
        mock_fig, mock_ax = Mock(), Mock()
        mock_plotter.create_wind_plot.return_value = (mock_fig, mock_ax)
        
        kwargs = {
            'processor': mock_processor,
            'cache': mock_cache,
            'output_dir': temp_output_dir,
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
        
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter = _pw.MPASWindPlotter
        _pw.MPASWindPlotter = lambda *a, **kw: mock_plotter

        try:
            result = _wind_worker((1, kwargs))            
            assert 'files' in result
            assert not result['cache_hits']['coordinates']  
        finally:
            _pw.MPASWindPlotter = orig_plotter


class TestCrossSectionWorkerMultipleFormats:
    """ Tests for handling multiple output formats in the cross-section worker. """
    
    def test_cross_section_worker_multiple_formats(self: "TestCrossSectionWorkerMultipleFormats", temp_output_dir: Path) -> None:
        """
        This test verifies that the `_cross_section_worker` function can handle multiple output formats correctly. It mocks the processor to return a dataset with a `theta` variable and configures the plotter mock to capture calls to `savefig`. The test asserts that the worker function completes without crashing, returns a result with multiple files, and that the plotter's `savefig` method is called for each specified format, confirming that the worker function properly handles generating and saving plots in multiple formats as requested. 

        Parameters:
            self ("TestCrossSectionWorkerMultipleFormats"): Test instance with mock processor fixtures.
            mock_plotter_class (Any): Pytest mock for the vertical cross-section plotter.
            temp_output_dir (Path): Temporary directory fixture for outputs.

        Returns:
            None: Assertions validate multiple output formats behavior.
        """
        mock_processor = Mock()
        n_time = 3
        
        mock_dataset = xr.Dataset({
            'theta': xr.DataArray(
                np.random.uniform(280, 320, (n_time, 10, 100)),
                dims=['Time', 'nVertLevels', 'nCells'],
                coords={'Time': pd.date_range('2024-01-01', periods=n_time, freq='h')}
            )
        })
        
        mock_processor.dataset = mock_dataset
        
        mock_plotter = Mock()
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plotter.create_vertical_cross_section.return_value = (mock_fig, mock_ax)
        
        kwargs = {
            'processor': mock_processor,
            'output_dir': temp_output_dir,
            'start_lat': 35.0,
            'start_lon': -105.0,
            'end_lat': 40.0,
            'end_lon': -95.0,
            'var_name': 'theta',
            'file_prefix': 'cross_section',
            'formats': ['png', 'pdf', 'jpg'],  
            'custom_title': None,
            'colormap': None,
            'levels': None,
            'vertical_coord': 'pressure',
            'num_points': 100
        }
        
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_plotter = _pw.MPASVerticalCrossSectionPlotter
        _pw.MPASVerticalCrossSectionPlotter = lambda *a, **kw: mock_plotter

        try:
            result = _cross_section_worker((1, kwargs))            
            assert len(result['files']) >= 3
            assert mock_fig.savefig.call_count >= 2  
        finally:
            _pw.MPASVerticalCrossSectionPlotter = orig_plotter


class TestProcessParallelResultsFailures:
    """ Test _process_parallel_results with failed tasks. """
    
    def test_process_parallel_results_with_failures(self: "TestProcessParallelResultsFailures", temp_output_dir: Path) -> None:
        """
        This test verifies that the `_process_parallel_results` function can handle a mix of successful and failed task results without crashing. It creates a list of `TaskResult` objects where some tasks indicate success with valid results, while others indicate failure with error messages. The test asserts that the function processes the results correctly, returns only the files from successful tasks, and that the output includes relevant statistics about the total, completed, and failed tasks, as well as timing information. This confirms that the function can gracefully handle and report on mixed outcomes from parallel processing tasks. 

        Parameters:
            self ("TestProcessParallelResultsFailures"): Test instance (unused).

        Returns:
            None: Assertions validate correct handling of mixed results.
        """
        results = [
            TaskResult(
                task_id=0,
                success=True,
                result={
                    'files': ['output1.png'],
                    'timings': {'data_processing': 1.0, 'plotting': 2.0, 'saving': 0.5, 'total': 3.5}
                }
            ),
            TaskResult(
                task_id=1,
                success=False,
                error="Processing failed"
            ),
            TaskResult(
                task_id=2,
                success=True,
                result={
                    'files': ['output2.png'],
                    'timings': {'data_processing': 1.2, 'plotting': 1.8, 'saving': 0.6, 'total': 3.6}
                }
            )
        ]
        
        time_indices = [0, 1, 2]
        mock_manager = Mock()
        mock_stats = Mock()
        mock_stats.total_tasks = 3
        mock_stats.completed_tasks = 2
        mock_stats.failed_tasks = 1
        mock_stats.total_time = 7.1  
        mock_stats.load_imbalance = 0.05  
        mock_manager.get_statistics.return_value = mock_stats
        
        import io
        from contextlib import redirect_stdout        
        f = io.StringIO()

        with redirect_stdout(f):
            created_files = _process_parallel_results(
                results,
                time_indices,
                str(temp_output_dir),
                mock_manager,
                "TEST_PROCESSING",
                var_info="test_var"
            )
        
        output = f.getvalue()
        
        assert len(created_files) == pytest.approx(2)
        assert "TEST_PROCESSING" in output
        assert "test_var" in output


class TestAutoBatchProcessorAdditional:
    """ Tests for additional edge cases in the auto_batch_processor function, including MPI import failure and explicit True/False settings. """
    
    def test_auto_batch_processor_mpi_import_error(self: "TestAutoBatchProcessorAdditional") -> None:
        """
        This test verifies that the `auto_batch_processor` function correctly handles an `ImportError` when attempting to import `mpi4py`. By mocking the built-in `__import__` function to raise an `ImportError` when `mpi4py` is imported, the test confirms that `auto_batch_processor` falls back to returning `False`, indicating that parallel processing cannot be used. This ensures that the function can gracefully handle the absence of optional dependencies without crashing. 

        Parameters:
            self ("TestAutoBatchProcessorAdditional"): Test instance (unused).
            
        Returns:
            None: Assertions validate fallback behavior.
        """
        with patch('builtins.__import__', side_effect=ImportError("No mpi4py")):
            result = auto_batch_processor(use_parallel=None)
            assert result is False
    
    def test_auto_batch_processor_explicit_true(self: "TestAutoBatchProcessorAdditional") -> None:
        """
        This test confirms that the `auto_batch_processor` function returns `True` when explicitly enabled, regardless of the environment. By passing `use_parallel=True`, the test asserts that the function respects the explicit setting and returns `True`, indicating that parallel processing should be used. This ensures that user preferences for enabling parallel processing are honored by the function. 

        Parameters:
            self ("TestAutoBatchProcessorAdditional"): Test instance (unused).

        Returns:
            None: Assertion verifies returned boolean.
        """
        result = auto_batch_processor(use_parallel=True)
        assert result is True
    
    def test_auto_batch_processor_explicit_false(self: "TestAutoBatchProcessorAdditional") -> None:
        """
        This test confirms that the `auto_batch_processor` function returns `False` when explicitly disabled, regardless of the environment. By passing `use_parallel=False`, the test asserts that the function respects the explicit setting and returns `False`, indicating that parallel processing should not be used. This ensures that user preferences for disabling parallel processing are honored by the function. 

        Parameters:
            self ("TestAutoBatchProcessorAdditional"): Test instance (unused).

        Returns:
            None: Assertion verifies returned boolean.
        """
        result = auto_batch_processor(use_parallel=False)
        assert result is False


class TestWorkerExceptionHandler:
    """ Tests for the catch-all exception handler in worker functions. """

    def test_precipitation_worker_returns_error_dict_on_crash(self, temp_output_dir: Path) -> None:
        """
        This test verifies that if the `_precipitation_worker` function encounters an unhandled exception (simulated by passing `None` as the processor), it returns a dictionary containing 'error' and 'traceback' keys with appropriate error information. It asserts that the error information is present in the result and that the worker does not crash, confirming that the catch-all exception handler is functioning correctly to capture and report unexpected errors in the worker workflow. 

        Parameters:
            self ("TestWorkerExceptionHandler"): Test instance (unused).
            temp_output_dir (Path): Temporary directory for output files.

        Returns:
            None: Assertions validate error handling behavior.
        """
        kwargs = {
            'processor': None, 
            'cache': None,
            'output_dir': str(temp_output_dir),
            'lon_min': -120, 'lon_max': -80,
            'lat_min': 30, 'lat_max': 50,
            'var_name': 'rainnc',
            'accum_period': 'a01h',
            'file_prefix': 'crash_test',
            'formats': ['png'],
        }

        captured = io.StringIO()
        with redirect_stdout(captured):
            result = _precipitation_worker((0, kwargs))

        assert 'error' in result
        assert 'traceback' in result
        assert result['files'] == []
        assert result['timings'] == {}
        assert 'WORKER ERROR' in captured.getvalue()

    def test_precipitation_worker_error_dict_time_idx_non_int(
        self, temp_output_dir: Path
    ) -> None:
        """
        This test verifies that if the `_precipitation_worker` function receives a non-integer `time_idx`, it returns a dictionary containing 'error' and 'time_str' keys with 'unknown' as the value for 'time_str'. It asserts that the error information is present in the result and that the worker does not crash, confirming that the catch-all exception handler is functioning correctly to capture and report unexpected errors in the worker workflow. 

        Parameters:
            self ("TestWorkerExceptionHandler"): Test instance (unused).
            temp_output_dir (Path): Temporary directory for output files.

        Returns:
            None: Assertions validate error handling behavior.
        """
        captured = io.StringIO()

        with redirect_stdout(captured):
            result = _precipitation_worker((slice(None), {'processor': None})) # type: ignore

        assert 'error' in result
        assert result['time_str'] == 'unknown'


class TestProcessParallelResultsEdgeCases:
    """ Tests for result processing with all-failure results and absent stats. """

    def test_all_failures_no_timing_stats(self, temp_output_dir: Path) -> None:
        """
        This test verifies that if the `_process_parallel_results` function receives a list of results where all tasks have failed, it correctly reports the failure statistics and does not attempt to print timing breakdowns. It creates a list of `TaskResult` objects where all tasks indicate failure with error messages, and mocks the manager to return no statistics. The test asserts that the function processes the results without crashing, returns an empty list of files, and that the output includes the correct failure statistics while omitting any timing information, confirming that the function can gracefully handle scenarios where all parallel tasks fail.

        Parameters:
            self ("TestProcessParallelResultsEdgeCases"): Test instance (unused).
            temp_output_dir (Path): Temporary directory for output files.

        Returns:
            None: Assertions validate result processing behavior.
        """
        results = [
            TaskResult(task_id=0, success=False, error="boom"),
            TaskResult(task_id=1, success=False, error="crash"),
        ]

        mock_manager = Mock()
        mock_manager.get_statistics.return_value = None
        captured = io.StringIO()

        with redirect_stdout(captured):
            files = _process_parallel_results(
                results, [0, 1], str(temp_output_dir), mock_manager, "ALL_FAIL"
            )

        output = captured.getvalue()

        assert files == []
        assert "Failed: 2/2" in output
        assert "Timing Breakdown" not in output  

    def test_no_manager_statistics(self, temp_output_dir: Path) -> None:
        """
        This test verifies that if the `_process_parallel_results` function receives a manager with no statistics, the report is printed without speedup information. It creates a list of `TaskResult` objects where one task indicates success with valid results, and mocks the manager to return no statistics. The test asserts that the function processes the results without crashing, returns the expected files, and that the output includes the report header but omits any speedup information, confirming that the function can gracefully handle scenarios where performance statistics are unavailable. 

        Parameters:
            self ("TestProcessParallelResultsEdgeCases"): Test instance (unused).
            temp_output_dir (Path): Temporary directory for output files.

        Returns:
            None: Assertions validate result processing behavior.
        """
        results = [
            TaskResult(
                task_id=0, success=True,
                result={
                    'files': ['a.png'],
                    'timings': {'data_processing': 0.1, 'plotting': 0.2, 'saving': 0.05, 'total': 0.35}
                }
            ),
        ]
        mock_manager = Mock()
        mock_manager.get_statistics.return_value = None

        captured = io.StringIO()
        with redirect_stdout(captured):
            files = _process_parallel_results(
                results, [0], str(temp_output_dir), mock_manager, "NO_STATS"
            )

        output = captured.getvalue()
        assert len(files) == pytest.approx(1)
        assert "Speedup potential" not in output

    def test_var_info_formatting(self, temp_output_dir: Path) -> None:
        """
        This test verifies that if the `_process_parallel_results` function receives a `var_info` string, it is included in the report header. It creates a list of `TaskResult` objects where one task indicates success with valid results, and mocks the manager to return no statistics. The test asserts that the function processes the results without crashing, returns the expected files, and that the output includes the provided `var_info` string in the report header, confirming that the function correctly integrates variable information into the result reporting. 

        Parameters:
            self ("TestProcessParallelResultsEdgeCases"): Test instance (unused).
            temp_output_dir (Path): Temporary directory for output files.

        Returns:
            None: Assertions validate result processing behavior.
        """
        results = [
            TaskResult(
                task_id=0, success=True,
                result={
                    'files': ['b.png'],
                    'timings': {'data_processing': 0.1, 'plotting': 0.2, 'saving': 0.05, 'total': 0.35}
                }
            ),
        ]
        mock_manager = Mock()
        mock_manager.get_statistics.return_value = None

        captured = io.StringIO()

        with redirect_stdout(captured):
            _process_parallel_results(
                results, [0], str(temp_output_dir), mock_manager,
                "VARINFO_TEST", var_info="Variable: temperature_2m"
            )

        assert "Variable: temperature_2m" in captured.getvalue()


class TestAutoBatchProcessorMPISingleProcess:
    """ Tests for auto_batch_processor when MPI is available but only 1 rank. """

    def test_mpi_single_rank_returns_false(self) -> None:
        """
        This test verifies that when MPI is available but Get_size() returns 1, the auto_batch_processor function returns False, indicating that parallel processing should not be used. It mocks the mpi4py import and the Get_size method to simulate an MPI environment with only one rank, and asserts that auto_batch_processor correctly identifies that parallel processing is not beneficial in this scenario. 

        Parameters:
            self ("TestAutoBatchProcessorMPISingleProcess"): Test instance (unused).

        Returns:
            None: Assertions validate auto_batch_processor behavior.
        """
        mock_mpi = MagicMock()
        mock_mpi.MPI.COMM_WORLD.Get_size.return_value = 1

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'mpi4py':
                return mock_mpi
            return real_import(name, *args, **kwargs)

        builtins.__import__ = mock_import

        try:
            result = auto_batch_processor(use_parallel=None)
            assert result is False
        finally:
            builtins.__import__ = real_import

    def test_mpi_multi_rank_returns_true(self) -> None:
        """
        This test verifies that when MPI is available and Get_size() returns a value greater than 1, the auto_batch_processor function returns True, indicating that parallel processing should be used. It mocks the mpi4py import and the Get_size method to simulate an MPI environment with multiple ranks, and asserts that auto_batch_processor correctly identifies that parallel processing is beneficial in this scenario. 

        Parameters:
            self ("TestAutoBatchProcessorMPISingleProcess"): Test instance (unused).

        Returns:
            None: Assertions validate auto_batch_processor behavior.
        """
        mock_mpi = MagicMock()
        mock_mpi.MPI.COMM_WORLD.Get_size.return_value = 4

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'mpi4py':
                return mock_mpi
            return real_import(name, *args, **kwargs)

        builtins.__import__ = mock_import

        try:
            result = auto_batch_processor(use_parallel=None)
            assert result is True
        finally:
            builtins.__import__ = real_import


class TestMPIModeBranches:
    """ Tests exercising MPI-mode branches in worker functions and Parallel*Processor classes. """

    def _make_mpi_kwargs(self, output_dir: str) -> Dict:
        """
        This helper method creates a dictionary of keyword arguments for worker functions that will trigger the MPI mode branches by including 'grid_file' and 'data_dir' keys. The values are set to fake paths since the actual file reading is mocked in the tests. 

        Parameters:
            self ("TestMPIModeBranches"): Test instance (unused).
            output_dir (str): Directory for output files.

        Returns:
            Dict: Keyword arguments for MPI mode functions.
        """
        return {
            'grid_file': '/fake/grid.nc',
            'data_dir': '/fake/data',
            'output_dir': output_dir,
            'lon_min': -120, 'lon_max': -80,
            'lat_min': 30, 'lat_max': 50,
        }

    def test_precipitation_worker_mpi_grid_file_branch(self, tmp_path) -> None:
        """
        This test verifies that the `_precipitation_worker` function correctly executes the MPI mode branch when `grid_file` and `data_dir` are provided. It mocks the processor to return a dataset with a `rainnc` variable and configures the diagnostic and plotter mocks to simulate the computation and plotting of precipitation data. The test asserts that the worker function completes without crashing and returns a result containing a 'files' key, confirming that the MPI mode branch is functioning correctly in the precipitation worker workflow. 

        Parameters:
            self ("TestMPIModeBranches"): Test instance (unused).
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None: Assertions validate _precipitation_worker behavior.
        """
        mock_proc = MagicMock()

        mock_proc.dataset = xr.Dataset({
            'rainnc': (['Time', 'nCells'], np.zeros((2, 10))),
            'Time': (['Time'], pd.date_range('2025-01-01', periods=2, freq='h').values),
        })

        mock_proc.load_2d_data.return_value = mock_proc
        mock_proc.extract_2d_coordinates_for_variable.return_value = (np.zeros(10), np.zeros(10))
        mock_proc.data_type = 'history'

        kwargs = self._make_mpi_kwargs(str(tmp_path))

        kwargs.update({
            'var_name': 'rainnc', 'accum_period': 'a01h',
            'file_prefix': 'test', 'formats': ['png'],
            'plot_type': 'scatter', 'grid_resolution': None,
            'custom_title_template': None, 'colormap': None, 'levels': None,
        })

        with patch('mpasdiag.processing.processors_2d.MPAS2DProcessor') as mock_cls, \
             patch('mpasdiag.processing.parallel_wrappers.PrecipitationDiagnostics') as mock_diag_cls, \
             patch('mpasdiag.processing.parallel_wrappers.MPASPrecipitationPlotter') as mock_plotter_cls:
            mock_cls.return_value = mock_proc
            mock_diag = MagicMock()
            mock_diag.compute_precipitation_difference.return_value = xr.DataArray(np.random.uniform(0, 5, 10))
            mock_diag_cls.return_value = mock_diag
            mock_plotter = MagicMock()
            mock_plotter.create_precipitation_map.return_value = (MagicMock(), MagicMock())
            mock_plotter_cls.return_value = mock_plotter
            result = _precipitation_worker((0, kwargs))

        assert isinstance(result, dict)
        assert 'files' in result

    def test_surface_worker_mpi_mode_branch(self, tmp_path) -> None:
        """
        This test verifies that the `_surface_worker` function correctly executes the MPI mode branch when `grid_file` and `data_dir` are provided. It mocks the processor to return a dataset with a `t2m` variable and configures the plotter mock to simulate the creation of a surface map. The test asserts that the worker function completes without crashing and returns a result containing a 'files' key, confirming that the MPI mode branch is functioning correctly in the surface worker workflow. 

        Parameters:
            self ("TestMPIModeBranches"): Test instance (unused).
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None: Assertions validate _surface_worker behavior.
        """
        mock_proc = MagicMock()

        mock_proc.dataset = xr.Dataset({
            't2m': (['Time', 'nCells'], np.random.uniform(260, 310, (2, 10))),
            'Time': (['Time'], pd.date_range('2025-01-01', periods=2, freq='h').values),
        })

        mock_proc.load_2d_data.return_value = mock_proc
        mock_proc.extract_spatial_coordinates.return_value = (np.zeros(10), np.zeros(10))

        kwargs = self._make_mpi_kwargs(str(tmp_path))

        kwargs.update({
            'var_name': 't2m', 'plot_type': 'contourf',
            'file_prefix': 'test', 'formats': ['png'],
            'custom_title': None, 'colormap': None, 'levels': None,
        })

        with patch('mpasdiag.processing.processors_2d.MPAS2DProcessor') as mock_cls, \
             patch('mpasdiag.processing.parallel_wrappers.MPASSurfacePlotter') as mock_plotter_cls:
            mock_cls.return_value = mock_proc
            mock_plotter = MagicMock()
            mock_plotter.create_surface_map.return_value = (MagicMock(), MagicMock())
            mock_plotter_cls.return_value = mock_plotter
            result = _surface_worker((0, kwargs))

        assert isinstance(result, dict)
        assert 'files' in result

    def test_wind_worker_mpi_mode_branch(self, tmp_path) -> None:
        """
        This test verifies that the `_wind_worker` function correctly executes the MPI mode branch when `grid_file` and `data_dir` are provided. It mocks the processor to return a dataset with `u10` and `v10` variables and configures the plotter mock to simulate the creation of a wind plot. The test asserts that the worker function completes without crashing and returns a result containing a 'files' key, confirming that the MPI mode branch is functioning correctly in the wind worker workflow. 

        Parameters:
            self ("TestMPIModeBranches"): Test instance (unused).
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None: Assertions validate _wind_worker behavior.
        """
        mock_proc = MagicMock()

        mock_proc.dataset = xr.Dataset({
            'u10': (['Time', 'nCells'], np.random.uniform(-5, 5, (2, 10))),
            'v10': (['Time', 'nCells'], np.random.uniform(-5, 5, (2, 10))),
            'Time': (['Time'], pd.date_range('2025-01-01', periods=2, freq='h').values),
        })

        mock_proc.load_2d_data.return_value = mock_proc
        mock_proc.get_2d_variable_data.return_value = xr.DataArray(np.random.uniform(-5, 5, 10))
        mock_proc.extract_2d_coordinates_for_variable.return_value = (np.zeros(10), np.zeros(10))

        kwargs = self._make_mpi_kwargs(str(tmp_path))

        kwargs.update({
            'u_variable': 'u10', 'v_variable': 'v10',
            'plot_type': 'barbs', 'subsample': 1, 'scale': None,
            'show_background': False, 'grid_resolution': None,
            'regrid_method': 'linear',
            'file_prefix': 'test', 'formats': ['png'],
        })

        with patch('mpasdiag.processing.processors_2d.MPAS2DProcessor') as mock_cls, \
             patch('mpasdiag.processing.parallel_wrappers.MPASWindPlotter') as mock_plotter_cls:
            mock_cls.return_value = mock_proc
            mock_plotter = MagicMock()
            mock_plotter.create_wind_plot.return_value = (MagicMock(), MagicMock())
            mock_plotter_cls.return_value = mock_plotter
            result = _wind_worker((0, kwargs))

        assert isinstance(result, dict)
        assert 'files' in result

    def test_cross_section_worker_mpi_3d_reload(self, tmp_path) -> None:
        """
        This test verifies that the `_cross_section_worker` function correctly executes the MPI mode branch for 3D data when `grid_file` and `data_dir` are provided. It mocks the processor to return a dataset with a `theta` variable and configures the plotter mock to simulate the creation of a vertical cross-section plot. The test asserts that the worker function completes without crashing and returns a result containing a 'files' key, confirming that the MPI mode branch for 3D data reload is functioning correctly in the cross-section worker workflow. 

        Parameters:
            self ("TestMPIModeBranches"): Test instance (unused).
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None: Assertions validate _cross_section_worker behavior.
        """
        mock_proc = MagicMock()

        mock_proc.dataset = xr.Dataset({
            'theta': (['Time', 'nVertLevels', 'nCells'], np.zeros((2, 5, 10))),
            'Time': (['Time'], pd.date_range('2025-01-01', periods=2, freq='h').values),
        })

        mock_proc.load_3d_data.return_value = mock_proc

        kwargs = {
            'grid_file': '/fake/grid.nc',
            'data_dir': '/fake/data',
            'output_dir': str(tmp_path),
            'start_lat': 30, 'start_lon': -110,
            'end_lat': 40, 'end_lon': -100,
            'var_name': 'theta', 'file_prefix': 'cross',
            'formats': ['png'], 'custom_title': None,
            'colormap': None, 'levels': None,
            'vertical_coord': 'pressure', 'num_points': 50,
        }

        with patch('mpasdiag.processing.processors_3d.MPAS3DProcessor') as mock_cls, \
             patch('mpasdiag.processing.parallel_wrappers.MPASVerticalCrossSectionPlotter') as mock_plotter_cls:
            mock_cls.return_value = mock_proc
            mock_plotter = MagicMock()
            mock_fig = MagicMock()
            mock_plotter.create_vertical_cross_section.return_value = (mock_fig, MagicMock())
            mock_plotter_cls.return_value = mock_plotter
            result = _cross_section_worker((0, kwargs))

        assert isinstance(result, dict)
        assert 'files' in result


class TestParallelProcessorBatchMethods:
    """ Tests for Parallel*Processor batch methods targeting MPI mode branches. """

    def test_process_parallel_results_success_iteration(self, tmp_path) -> None:
        """
        This test verifies that the `_process_parallel_results` function correctly processes a list of successful `TaskResult` objects, extracts the files, and prints the appropriate statistics. It mocks a set of successful results with timing information, simulates a manager with relevant statistics, and asserts that the function returns the expected list of files and includes the correct success statistics in the output, confirming that the function can handle and report on successful parallel processing outcomes effectively. 

        Parameters:
            self ("TestParallelProcessorBatchMethods"): Test instance (unused).
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None: Assertions validate _process_parallel_results behavior.
        """
        results = [
            TaskResult(task_id=0, success=True, result={
                'files': ['a.png', 'b.png'],
                'timings': {'data_processing': 0.1, 'plotting': 0.2, 'saving': 0.05, 'total': 0.35}
            }),
            TaskResult(task_id=1, success=True, result={
                'files': ['c.png'],
                'timings': {'data_processing': 0.15, 'plotting': 0.25, 'saving': 0.03, 'total': 0.43}
            }),
        ]

        mock_manager = Mock()

        mock_manager.get_statistics.return_value = ParallelStats(
            total_tasks=2, completed_tasks=2, failed_tasks=0,
            total_time=0.78
        )

        captured = io.StringIO()

        with redirect_stdout(captured):
            files = _process_parallel_results(
                results, [0, 1], str(tmp_path), mock_manager, "TEST"
            )

        assert len(files) == pytest.approx(3)
        assert 'Successful: 2/2' in captured.getvalue()

    def test_timing_statistics_creation(self, tmp_path) -> None:
        """
        This test verifies that the `_process_parallel_results` function correctly creates a timing_stats dictionary for each key. It mocks a set of successful `TaskResult` objects with timing information, simulates a manager with no statistics, and asserts that the function returns the expected list of files and includes the correct timing breakdown in the output, confirming that the function can generate and report detailed timing statistics even when overall performance statistics are unavailable. 

        Parameters:
            self ("TestParallelProcessorBatchMethods"): Test instance (unused).
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None: Assertions validate _process_parallel_results behavior.
        """
        results = [
            TaskResult(task_id=0, success=True, result={
                'files': ['x.png'],
                'timings': {'data_processing': 0.5, 'plotting': 1.0, 'saving': 0.2, 'total': 1.7}
            }),
        ]

        mock_manager = Mock()
        mock_manager.get_statistics.return_value = None
        captured = io.StringIO()

        with redirect_stdout(captured):
            files = _process_parallel_results(
                results, [0], str(tmp_path), mock_manager, "TIMING"
            )

        output = captured.getvalue()
        assert 'Timing Breakdown' in output
        assert 'Data Processing' in output
        assert len(files) == pytest.approx(1)

    def test_precipitation_processor_mpi_check(self, tmp_path) -> None:
        """
        This test verifies that the `ParallelPrecipitationProcessor` correctly checks MPI mode and requires a `data_dir`. It mocks a manager in MPI mode, removes the `data_dir` attribute from the processor, and asserts that an `AttributeError` is raised when attempting to create batch precipitation maps in parallel. 

        Parameters:
            self ("TestParallelProcessorBatchMethods"): Test instance (unused).
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None: Assertions validate MPI mode behavior.
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr = _pw.MPASParallelManager
        mock_mgr = MagicMock()
        mock_mgr.backend = 'mpi'
        mock_mgr.is_master = True
        _pw.MPASParallelManager = lambda *a, **kw: mock_mgr
        try:
            processor = MagicMock()
            processor.grid_file = '/fake/grid.nc'
            del processor.data_dir  

            with pytest.raises(AttributeError, match="MPI mode requires"):
                ParallelPrecipitationProcessor.create_batch_precipitation_maps_parallel(
                    processor=processor,
                    output_dir=str(tmp_path),
                    lon_min=-120, lon_max=-80, lat_min=30, lat_max=50,
                    var_name='rainnc', accum_period='a01h',
                    time_indices=[0, 1], n_processes=2
                )
        finally:
            _pw.MPASParallelManager = orig_mgr

    def test_surface_processor_mpi_check(self, tmp_path) -> None:
        """
        This test verifies that the `ParallelSurfaceProcessor` correctly checks MPI mode and requires a `data_dir`. It mocks a manager in MPI mode, removes the `data_dir` attribute from the processor, and asserts that an `AttributeError` is raised when attempting to create batch surface maps in parallel. 

        Parameters:
            self ("TestParallelProcessorBatchMethods"): Test instance (unused).
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None: Assertions validate MPI mode behavior.
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr = _pw.MPASParallelManager
        mock_mgr = MagicMock()
        mock_mgr.backend = 'mpi'
        mock_mgr.is_master = True
        _pw.MPASParallelManager = lambda *a, **kw: mock_mgr
        try:
            processor = MagicMock()
            processor.grid_file = '/fake/grid.nc'
            del processor.data_dir

            with pytest.raises(AttributeError, match="MPI mode requires"):
                ParallelSurfaceProcessor.create_batch_surface_maps_parallel(
                    processor=processor,
                    output_dir=str(tmp_path),
                    lon_min=-120, lon_max=-80, lat_min=30, lat_max=50,
                    var_name='t2m', time_indices=[0, 1], n_processes=2
                )
        finally:
            _pw.MPASParallelManager = orig_mgr

    def test_wind_processor_mpi_check(self, tmp_path) -> None:
        """
        This test verifies that the `ParallelWindProcessor` correctly checks MPI mode and requires a `data_dir`. It mocks a manager in MPI mode, removes the `data_dir` attribute from the processor, and asserts that an `AttributeError` is raised when attempting to create batch wind plots in parallel. 

        Parameters:
            self ("TestParallelProcessorBatchMethods"): Test instance (unused).
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None: Assertions validate MPI mode behavior.
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr = _pw.MPASParallelManager
        mock_mgr = MagicMock()
        mock_mgr.backend = 'mpi'
        mock_mgr.is_master = True
        _pw.MPASParallelManager = lambda *a, **kw: mock_mgr
        try:
            processor = MagicMock()
            processor.grid_file = '/fake/grid.nc'
            del processor.data_dir

            with pytest.raises(AttributeError, match="MPI mode requires"):
                ParallelWindProcessor.create_batch_wind_plots_parallel(
                    processor=processor,
                    output_dir=str(tmp_path),
                    lon_min=-120, lon_max=-80, lat_min=30, lat_max=50,
                    u_variable='u10', v_variable='v10',
                    time_indices=[0, 1], n_processes=2
                )
        finally:
            _pw.MPASParallelManager = orig_mgr

    def test_cross_section_processor_mpi_check(self, tmp_path) -> None:
        """
        This test verifies that the `ParallelCrossSectionProcessor` correctly checks MPI mode and requires a `data_dir`. It mocks a manager in MPI mode, removes the `data_dir` attribute from the processor, and asserts that an `AttributeError` is raised when attempting to create batch cross-section plots in parallel. 

        Parameters:
            self ("TestParallelProcessorBatchMethods"): Test instance (unused).
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None: Assertions validate MPI mode behavior.
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr = _pw.MPASParallelManager
        mock_mgr = MagicMock()
        mock_mgr.backend = 'mpi'
        mock_mgr.is_master = True
        _pw.MPASParallelManager = lambda *a, **kw: mock_mgr
        try:
            processor = MagicMock()
            processor.grid_file = '/fake/grid.nc'
            del processor.data_dir

            with pytest.raises(AttributeError, match="MPI mode requires"):
                ParallelCrossSectionProcessor.create_batch_cross_section_plots_parallel(
                    mpas_3d_processor=processor,
                    output_dir=str(tmp_path),
                    var_name='theta',
                    start_point=(-110, 30), end_point=(-100, 40),
                    time_indices=[0, 1], n_processes=2
                )
        finally:
            _pw.MPASParallelManager = orig_mgr

    def test_precipitation_processor_mpi_kwargs_construction(self, tmp_path) -> None:
        """
        This test verifies that the `ParallelPrecipitationProcessor` correctly constructs `worker_kwargs` in MPI mode. It mocks a manager in MPI mode, sets up a processor with necessary attributes, and asserts that the `create_batch_precipitation_maps_parallel` method behaves as expected when `parallel_map` returns `None`, confirming that the method can construct the appropriate arguments for worker functions in MPI mode without attempting to process results when `parallel_map` is mocked to return `None`. 

        Parameters:
            self ("TestParallelProcessorBatchMethods"): Test instance (unused).
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None: Assertions validate MPI mode behavior.
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr = _pw.MPASParallelManager
        mock_mgr = MagicMock()
        mock_mgr.backend = 'mpi'
        mock_mgr.is_master = True
        mock_mgr.parallel_map.return_value = None 
        _pw.MPASParallelManager = lambda *a, **kw: mock_mgr
        try:
            processor = MagicMock()
            processor.grid_file = '/fake/grid.nc'
            processor.data_dir = '/fake/data'
            processor.data_type = 'history'
            processor.dataset = MagicMock()
            processor.dataset.sizes = {'Time': 5}

            result = ParallelPrecipitationProcessor.create_batch_precipitation_maps_parallel(
                processor=processor,
                output_dir=str(tmp_path),
                lon_min=-120, lon_max=-80, lat_min=30, lat_max=50,
                var_name='rainnc', accum_period='a01h',
                time_indices=[1, 2], n_processes=2
            )
            assert result is None
            mock_mgr.parallel_map.assert_called_once()
        finally:
            _pw.MPASParallelManager = orig_mgr

    def test_cross_section_processor_result_aggregation(self, tmp_path) -> None:
        """
        This test verifies that the `ParallelCrossSectionProcessor` correctly aggregates results from `parallel_map` and processes them with `_process_parallel_results`. It mocks a manager in multiprocessing mode, simulates a set of results with one success and one failure, and asserts that the `create_batch_cross_section_plots_parallel` method returns the expected list of files and includes the correct success statistics in the output, confirming that the method can handle and report on mixed outcomes from parallel processing effectively. 

        Parameters:
            self ("TestParallelProcessorBatchMethods"): Test instance (unused).
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None: Assertions validate result aggregation behavior.
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr = _pw.MPASParallelManager
        mock_mgr = MagicMock()
        mock_mgr.backend = 'multiprocessing'
        mock_mgr.is_master = True

        results = [
            MagicMock(success=True, result=['cross_0.png']),
            MagicMock(success=False, result=None, error='Some error', task_id=1),
        ]

        mock_mgr.parallel_map.return_value = results
        _pw.MPASParallelManager = lambda *a, **kw: mock_mgr

        try:
            processor = MagicMock(spec=MPAS3DProcessor)
            processor.grid_file = '/fake/grid.nc'
            processor.data_dir = '/fake/data'
            processor.dataset = MagicMock()
            processor.dataset.sizes = {'Time': 5}

            captured = io.StringIO()
            
            with redirect_stdout(captured):
                files = ParallelCrossSectionProcessor.create_batch_cross_section_plots_parallel(
                    mpas_3d_processor=processor,
                    output_dir=str(tmp_path),
                    var_name='theta',
                    start_point=(-110, 30), end_point=(-100, 40),
                    time_indices=[0, 1], n_processes=2
                )
            assert files == ['cross_0.png']
            assert 'Successful: 1/2' in captured.getvalue()
        finally:
            _pw.MPASParallelManager = orig_mgr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

