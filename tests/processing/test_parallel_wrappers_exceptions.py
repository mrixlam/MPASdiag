#!/usr/bin/env python3
"""
MPASdiag Test Suite: Parallel Processing Wrappers

This module contains tests for the parallel processing wrapper functions in the MPASdiag package. The tests focus on verifying that the worker functions can handle exceptions gracefully, particularly those related to cache loading and missing time dimensions, without crashing the entire processing workflow. Additionally, the tests confirm that custom title templates are correctly passed to the plotters and that multiple output formats are handled properly. The test suite uses mocking to simulate various scenarios and ensure that the error handling mechanisms in the worker functions are robust.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
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

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
GRID_FILE = os.path.join(TEST_DATA_DIR, "grids", "x1.10242.static.nc")


class TestImportErrorHandling:
    """ Tests that verify import-time fallback behavior for optional modules. """
    
    def test_import_from_data_cache(self: "TestImportErrorHandling") -> None:
        """
        This test confirms that the `MPASDataCache` and `get_global_cache` symbols are available from the wrapper module, ensuring that the import fallback mechanism for the data cache is functioning correctly. It imports these symbols and asserts that they are not None, which indicates that the import succeeded and the fallback did not cause a failure. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.parallel_wrappers import MPASDataCache, get_global_cache
        data_cache = MPASDataCache()
        assert_expected_public_methods(data_cache, 'MPASDataCache')
        assert get_global_cache is not None
    
    def test_import_from_parallel(self: "TestImportErrorHandling") -> None:
        """
        This test confirms that the `MPASParallelManager`, `MPAS2DProcessor`, and `MPAS3DProcessor` symbols are available from the wrapper module, ensuring that the import fallback mechanism for the parallel processing components is functioning correctly. It imports these symbols and asserts that they are not None, which indicates that the import succeeded and the fallback did not cause a failure. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.parallel_wrappers import MPASParallelManager
        from mpasdiag.processing.parallel_wrappers import MPAS2DProcessor
        from mpasdiag.processing.parallel_wrappers import MPAS3DProcessor

        parallel_manager = MPASParallelManager()
        assert_expected_public_methods(parallel_manager, 'MPASParallelManager')

        process_2d = MPAS2DProcessor(grid_file=GRID_FILE)
        assert_expected_public_methods(process_2d, 'MPAS2DProcessor')

        process_3d = MPAS3DProcessor(grid_file=GRID_FILE)
        assert_expected_public_methods(process_3d, 'MPAS3DProcessor')
    
    def test_import_from_visualization(self: "TestImportErrorHandling") -> None:
        """
        This test confirms that the `MPASPrecipitationPlotter`, `MPASSurfacePlotter`, `MPASWindPlotter`, `MPASVerticalCrossSectionPlotter`, and `PrecipitationDiagnostics` symbols are available from the wrapper module, ensuring that the import fallback mechanism for the visualization components is functioning correctly. It imports these symbols and asserts that they are not None, which indicates that the import succeeded and the fallback did not cause a failure. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.parallel_wrappers import ParallelPrecipitationProcessor
        from mpasdiag.processing.parallel_wrappers import ParallelSurfaceProcessor
        from mpasdiag.processing.parallel_wrappers import ParallelWindProcessor
        from mpasdiag.processing.parallel_wrappers import ParallelCrossSectionProcessor
        from mpasdiag.processing.parallel_wrappers import PrecipitationDiagnostics

        precip_plotter = ParallelPrecipitationProcessor()
        assert_expected_public_methods(precip_plotter, 'ParallelPrecipitationProcessor')

        surface_plotter = ParallelSurfaceProcessor()
        assert_expected_public_methods(surface_plotter, 'ParallelSurfaceProcessor')

        wind_plotter = ParallelWindProcessor()
        assert_expected_public_methods(wind_plotter, 'ParallelWindProcessor')

        cross_section_plotter = ParallelCrossSectionProcessor()
        assert_expected_public_methods(cross_section_plotter, 'ParallelCrossSectionProcessor')  

        precip_diag = PrecipitationDiagnostics()
        assert_expected_public_methods(precip_diag, 'PrecipitationDiagnostics')      


class TestPrecipitationWorkerCacheException:
    """ Tests for handling exceptions related to cache loading in the precipitation worker. """
    
    def test_precipitation_worker_cache_load_exception(self: "TestPrecipitationWorkerCacheException", 
                                                       temp_output_dir: Path) -> None:
        """
        This test verifies that the `_precipitation_worker` function handles exceptions raised during cache loading gracefully. It mocks the processor to return a dataset with a `rainnc` variable and configures the cache mock to raise an exception when attempting to load coordinates. The test asserts that the worker function completes without crashing and returns a result, confirming that the error handling for cache loading issues is functioning correctly and does not cause unhandled exceptions in the worker workflow. 

        Parameters:
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
    
    def test_precipitation_worker_no_time_dimension(self: "TestPrecipitationWorkerTimeString", 
                                                    temp_output_dir: Path) -> None:
        """
        This test verifies that the `_precipitation_worker` function can handle a dataset that lacks a Time dimension and correctly generates a fallback time string. It mocks the processor to return a dataset with a `rainnc` variable that only has an `nCells` dimension, simulating the absence of time information. The test asserts that the worker function completes without crashing and returns a result with a time string in the expected fallback format, confirming that the worker can gracefully handle datasets without time coordinates. 

        Parameters:
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
    
    def test_precipitation_worker_custom_title_template(self: "TestPrecipitationWorkerCustomTitle", 
                                                        temp_output_dir: Path) -> None:
        """
        This test verifies that the `_precipitation_worker` function correctly accepts a custom title template and passes it to the plotter when creating precipitation maps. It mocks the processor to return a dataset with a `rainnc` variable and configures the plotter mock to capture the title argument used in the `create_precipitation_map` method. The test asserts that the custom title template is included in the title passed to the plotter, confirming that the worker function properly integrates user-defined title templates into the plotting workflow. 

        Parameters:
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
    
    def test_surface_worker_cache_load_exception(self: "TestSurfaceWorkerCacheException", 
                                                 temp_output_dir: Path) -> None:
        """
        This test verifies that the `_surface_worker` function can handle exceptions raised during cache loading without crashing. It mocks the processor to return a dataset with a `t2m` variable and configures the cache mock to raise an exception when attempting to load coordinates. The test asserts that the worker function completes and returns a result, confirming that the error handling for cache loading issues is functioning correctly and does not cause unhandled exceptions in the worker workflow. 

        Parameters:
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
    
    def test_wind_worker_no_time_dimension(self: "TestWindWorkerTimeString", 
                                           temp_output_dir: Path) -> None:
        """
        This test verifies that the `_wind_worker` function can handle a dataset that lacks a Time dimension and correctly generates a fallback time string. It mocks the processor to return a dataset with `u10` and `v10` variables that only have an `nCells` dimension, simulating the absence of time information. The test asserts that the worker function completes without crashing and returns a result with a time string in the expected fallback format, confirming that the worker can gracefully handle datasets without time coordinates. 

        Parameters:
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
    
    def test_wind_worker_cache_load_exception(self: "TestWindWorkerCacheException", 
                                              temp_output_dir: Path) -> None:
        """
        This test verifies that the `_wind_worker` function can handle exceptions raised during cache loading gracefully. It mocks the processor to return a dataset with `u10` and `v10` variables and configures the cache mock to raise an exception when attempting to load coordinates. The test asserts that the worker function completes without crashing and returns a result, confirming that the error handling for cache loading issues is functioning correctly and does not cause unhandled exceptions in the worker workflow. 

        Parameters:
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
    
    def test_cross_section_worker_multiple_formats(self: "TestCrossSectionWorkerMultipleFormats", 
                                                   temp_output_dir: Path) -> None:
        """
        This test verifies that the `_cross_section_worker` function can handle multiple output formats correctly. It mocks the processor to return a dataset with a `theta` variable and configures the plotter mock to capture calls to `savefig`. The test asserts that the worker function completes without crashing, returns a result with multiple files, and that the plotter's `savefig` method is called for each specified format, confirming that the worker function properly handles generating and saving plots in multiple formats as requested. 

        Parameters:
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
    
    def test_process_parallel_results_with_failures(self: "TestProcessParallelResultsFailures", 
                                                    temp_output_dir: Path) -> None:
        """
        This test verifies that the `_process_parallel_results` function can handle a mix of successful and failed task results without crashing. It creates a list of `TaskResult` objects where some tasks indicate success with valid results, while others indicate failure with error messages. The test asserts that the function processes the results correctly, returns only the files from successful tasks, and that the output includes relevant statistics about the total, completed, and failed tasks, as well as timing information. This confirms that the function can gracefully handle and report on mixed outcomes from parallel processing tasks. 

        Parameters:
            temp_output_dir (Path): Temporary directory fixture for outputs.

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
            None
            
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
            None

        Returns:
            None: Assertion verifies returned boolean.
        """
        result = auto_batch_processor(use_parallel=True)
        assert result is True
    
    def test_auto_batch_processor_explicit_false(self: "TestAutoBatchProcessorAdditional") -> None:
        """
        This test confirms that the `auto_batch_processor` function returns `False` when explicitly disabled, regardless of the environment. By passing `use_parallel=False`, the test asserts that the function respects the explicit setting and returns `False`, indicating that parallel processing should not be used. This ensures that user preferences for disabling parallel processing are honored by the function. 

        Parameters:
            None

        Returns:
            None: Assertion verifies returned boolean.
        """
        result = auto_batch_processor(use_parallel=False)
        assert result is False


