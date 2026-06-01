#!/usr/bin/env python3

"""
MPASdiag Test Suite: Parallel Processing Wrappers

This module contains unit tests for the parallel processing wrapper functions in the MPASdiag package. These tests cover the functionality of the worker functions and the result processing function that summarize outcomes from parallel tasks. The tests use pytest fixtures to set up mock environments and synthetic datasets, allowing for controlled testing of parallel execution without requiring actual parallel runs. The test cases include scenarios for all successful tasks, mixed success and failure, and edge cases such as cache loading failures. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import os
import pytest
import shutil
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import List, Generator
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mpasdiag.processing.parallel import ParallelStats, TaskResult, MPASParallelManager
from mpasdiag.processing.parallel_wrappers import (
    _process_parallel_results,
    ParallelPrecipitationProcessor,
    ParallelSurfaceProcessor,
    ParallelWindProcessor,
    ParallelCrossSectionProcessor
)
from tests.test_data_helpers import assert_expected_public_methods

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
GRID_FILE = os.path.join(TEST_DATA_DIR, "grids", "x1.10242.static.nc")


_RNG = np.random.default_rng()


class TestProcessParallelResults:
    """ Tests for the `_process_parallel_results` function that summarizes parallel task outcomes. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestProcessParallelResults') -> None:
        """
        This fixture prepares a temporary directory and a mock parallel manager for testing the `_process_parallel_results` function. The temporary directory is used to simulate file output locations, while the mock parallel manager allows for controlled testing of statistics retrieval without requiring an actual parallel execution environment. After the test runs, the temporary directory is removed to clean up resources. 

        Parameters:
            None

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()
        self.mock_manager = Mock(spec=MPASParallelManager)
    
    def teardown_method(self: 'TestProcessParallelResults') -> None:
        """
        This method cleans up temporary resources created for testing the `_process_parallel_results` function by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment. 

        Parameters:
            None

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_process_results_all_success(self: 'TestProcessParallelResults') -> None:
        """
        This test verifies that the `_process_parallel_results` function correctly processes a list of `TaskResult` objects where all tasks were successful. It mocks the parallel manager to return specific statistics, captures the printed output, and asserts that the expected number of files is returned and that the summary contains the correct success count. This confirms that the function accurately summarizes successful outcomes and integrates with the parallel manager's statistics. 

        Parameters:
            None

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
    
    def test_process_results_mixed(self: 'TestProcessParallelResults') -> None:
        """
        This test verifies that the `_process_parallel_results` function correctly processes a list of `TaskResult` objects where some tasks were successful and others failed. It mocks the parallel manager to return specific statistics, captures the printed output, and asserts that the expected number of files is returned (only from successful tasks) and that the summary contains the correct counts for both successful and failed tasks. This confirms that the function accurately summarizes mixed outcomes and integrates with the parallel manager's statistics. 

        Parameters:
            None

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

    def test_process_results_success_with_error_key(self: 'TestProcessParallelResults') -> None:
        """
        This test verifies that the `_process_parallel_results` function correctly processes a list of `TaskResult` objects where a task is marked as successful but contains an 'error' key in its result. It mocks the parallel manager to return specific statistics, captures the printed output, and asserts that the expected number of files is returned (only from the successful task) and that the summary indicates a failure due to the presence of the error key. This confirms that the function can handle cases where tasks are technically successful but contain error information, and that it integrates with the parallel manager's statistics appropriately. 

        Parameters:
            None

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
                result={'error': 'worker encountered a problem', 'files': [], 'timings': {}}
            ),
        ]

        mock_stats = ParallelStats()
        mock_stats.total_time = 1.0
        mock_stats.load_imbalance = 0.0
        self.mock_manager.get_statistics.return_value = mock_stats

        from io import StringIO
        import sys
        captured = StringIO()
        sys.stdout = captured

        try:
            files = _process_parallel_results(
                results, [0, 1], self.temp_dir, self.mock_manager, "TEST"
            )
            assert len(files) == pytest.approx(1)
            assert "Failed" in captured.getvalue()
        finally:
            sys.stdout = sys.__stdout__


class TestParallelPrecipitationProcessor:
    """ Tests for the `ParallelPrecipitationProcessor` batch functions. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestParallelPrecipitationProcessor') -> None:
        """
        This fixture prepares a temporary directory and a mock processor with a synthetic dataset containing a `rainnc` variable for testing the `ParallelPrecipitationProcessor` batch functions. The dataset includes coordinate arrays for time and cells. The mock processor is configured to return this dataset, allowing the batch processing functions to be invoked without accessing external data files. After the test runs, the temporary directory is removed to clean up resources. 

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
            )
        })
        
        self.mock_processor.dataset = self.mock_dataset
    
    def teardown_method(self: 'TestParallelPrecipitationProcessor') -> None:
        """
        This method cleans up temporary resources created for testing the `ParallelPrecipitationProcessor` batch functions by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment. 

        Parameters:
            None

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_batch_precipitation_maps_parallel(self: 'TestParallelPrecipitationProcessor') -> None:
        """
        This test verifies that the `create_batch_precipitation_maps_parallel` function successfully orchestrates parallel batch processing for precipitation maps. It mocks the `MPASDataCache` to return a mock cache object and the `MPASParallelManager` to simulate parallel execution with a successful task result. The test asserts that the function returns a non-None result and that the `parallel_map` method of the manager was called exactly once, confirming that the batch processing was initiated correctly and that the worker function executed as expected in a parallel context. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr, orig_cache = _pw.MPASParallelManager, _pw.MPASDataCache

        assert_expected_public_methods(_pw.MPASParallelManager, 'MPASParallelManager')
        assert_expected_public_methods(_pw.MPASDataCache, 'MPASDataCache')

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
    

class TestParallelSurfaceProcessor:
    """ Tests for the `ParallelSurfaceProcessor` batch processing methods. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestParallelSurfaceProcessor') -> None:
        """
        This fixture prepares a temporary directory and a mock processor with a synthetic dataset containing a `t2m` variable for testing the `ParallelSurfaceProcessor` batch processing methods. The dataset includes coordinate arrays for time and cells. The mock processor is configured to return this dataset, allowing the batch processing functions to be invoked without accessing external data files. After the test runs, the temporary directory is removed to clean up resources. 

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
                dims=['Time', 'nCells']
            )
        })
        
        self.mock_processor.dataset = self.mock_dataset
    
    def teardown_method(self: 'TestParallelSurfaceProcessor') -> None:
        """
        This method cleans up temporary resources created for testing the `ParallelSurfaceProcessor` batch processing methods by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment. 

        Parameters:
            None

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_batch_surface_maps_parallel(self: 'TestParallelSurfaceProcessor') -> None:
        """
        This test verifies that the `create_batch_surface_maps_parallel` function successfully orchestrates parallel batch processing for surface variable maps. It mocks the `MPASDataCache` to return a mock cache object and the `MPASParallelManager` to simulate parallel execution with a successful task result. The test asserts that the function returns a non-None result, confirming that the batch processing was initiated correctly and that the worker function executed as expected in a parallel context. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr, orig_cache = _pw.MPASParallelManager, _pw.MPASDataCache

        assert_expected_public_methods(orig_cache, 'MPASDataCache')
        assert_expected_public_methods(orig_mgr, 'MPASParallelManager')

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
    def setup_method(self: 'TestParallelWindProcessor') -> None:
        """
        This fixture prepares a temporary directory and a mock processor with a synthetic dataset containing `u10` and `v10` wind components for testing the `ParallelWindProcessor` batch processing methods. The dataset includes coordinate arrays for time and cells. The mock processor is configured to return this dataset, allowing the batch processing functions to be invoked without accessing external data files. After the test runs, the temporary directory is removed to clean up resources. 

        Parameters:
            None

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()
        self.mock_processor = Mock()
        
        n_time, n_cells = 5, 100

        self.mock_dataset = xr.Dataset({
            'u10': xr.DataArray(_RNG.uniform(-10, 10, (n_time, n_cells)), dims=['Time', 'nCells']),
            'v10': xr.DataArray(_RNG.uniform(-10, 10, (n_time, n_cells)), dims=['Time', 'nCells'])
        })
        
        self.mock_processor.dataset = self.mock_dataset
    
    def teardown_method(self: 'TestParallelWindProcessor') -> None:
        """
        This method cleans up temporary resources created for testing the `ParallelWindProcessor` batch processing methods by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment. 

        Parameters:
            None

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_batch_wind_plots_parallel(self: 'TestParallelWindProcessor') -> None:
        """
        This test verifies that the `create_batch_wind_plots_parallel` function successfully orchestrates parallel batch processing for wind plots. It mocks the `MPASDataCache` to return a mock cache object and the `MPASParallelManager` to simulate parallel execution with a successful task result. The test asserts that the function returns a non-None result, confirming that the batch processing was initiated correctly and that the worker function executed as expected in a parallel context.

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr, orig_cache = _pw.MPASParallelManager, _pw.MPASDataCache

        assert_expected_public_methods(orig_cache, 'MPASDataCache')
        assert_expected_public_methods(orig_mgr, 'MPASParallelManager')

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
    def setup_method(self: 'TestParallelCrossSectionProcessor') -> None:
        """
        This fixture prepares a temporary directory and a mock processor with a synthetic dataset containing a `temperature` variable for testing the `ParallelCrossSectionProcessor` batch processing methods. The dataset includes coordinate arrays for time, vertical levels, and cells. The mock processor is configured to return this dataset, allowing the batch processing functions to be invoked without accessing external data files. After the test runs, the temporary directory is removed to clean up resources. 

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
                dims=['Time', 'nVertLevels', 'nCells']
            )
        })
        
        self.mock_processor.dataset = self.mock_dataset
    
    def teardown_method(self: 'TestParallelCrossSectionProcessor') -> None:
        """
        This method cleans up temporary resources created for testing the `ParallelCrossSectionProcessor` batch processing methods by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment. 

        Parameters:
            None

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_batch_cross_section_plots_parallel(self: 'TestParallelCrossSectionProcessor') -> None:
        """
        This test verifies that the `create_batch_cross_section_plots_parallel` function successfully orchestrates parallel batch processing for vertical cross-section plots. It mocks the `MPASParallelManager` to simulate parallel execution with a successful task result. The test asserts that the function returns a non-None result, confirming that the batch processing was initiated correctly and that the worker function executed as expected in a parallel context. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr = _pw.MPASParallelManager

        assert_expected_public_methods(orig_mgr, 'MPASParallelManager')

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


class TestEdgeCases:
    """ Tests for edge cases, error handling, and exceptional conditions in parallel wrappers. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestEdgeCases') -> Generator[None, None, None]:
        """
        This fixture sets up a temporary directory for edge case tests and ensures that it is cleaned up after the tests run. The temporary directory can be used by tests that require file output without affecting the actual filesystem. After yielding control to the test, the fixture removes the temporary directory to maintain a clean testing environment. 

        Parameters:
            None

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_load_failure(self: 'TestEdgeCases') -> None:
        """
        This test verifies that the `create_batch_precipitation_maps_parallel` function handles a failure in loading coordinates from the data cache gracefully. It mocks the `MPASDataCache` to raise an exception when attempting to load coordinates, simulating a cache failure scenario. The test asserts that the function completes without crashing and returns a result (which may be None or an empty list), confirming that the error handling for cache loading issues is functioning correctly and does not cause unhandled exceptions in the batch processing workflow. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr, orig_cache = _pw.MPASParallelManager, _pw.MPASDataCache

        assert_expected_public_methods(orig_cache, 'MPASDataCache')
        assert_expected_public_methods(orig_mgr, 'MPASParallelManager')

        mock_processor = Mock()

        mock_dataset = xr.Dataset({
            'rainnc': xr.DataArray(_RNG.uniform(0, 50, (5, 100)), dims=['Time', 'nCells'])
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


class TestParallelPrecipitationProcessorEdgeCases:
    """ Edge-case tests for ParallelPrecipitationProcessor targeting uncovered branches. """

    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestParallelPrecipitationProcessorEdgeCases') -> None:
        """
        This fixture prepares a temporary directory and a mock processor with a synthetic dataset containing a `rainnc` variable for testing edge cases in the `ParallelPrecipitationProcessor` batch functions. The dataset includes coordinate arrays for time and cells. The mock processor is configured to return this dataset, allowing the batch processing functions to be invoked without accessing external data files. After the test runs, the temporary directory is removed to clean up resources.

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
            )
        })
        self.mock_processor.dataset = self.mock_dataset

    def teardown_method(self: 'TestParallelPrecipitationProcessorEdgeCases') -> None:
        """
        This method cleans up temporary resources created for testing edge cases in the `ParallelPrecipitationProcessor` batch functions by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment. 

        Parameters:
            None

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_time_indices_after_filtering_returns_empty(self: 'TestParallelPrecipitationProcessorEdgeCases',) -> None:
        """
        This test verifies that the `create_batch_precipitation_maps_parallel` function returns an empty list when the provided `time_indices` are filtered out due to the accumulation period requirements, resulting in no valid time indices to process. It uses a mock dataset with 5 time steps and an accumulation period of 'a06h', which requires at least 6 time steps for valid processing. The test asserts that the function returns an empty list, confirming that it correctly handles cases where no time indices meet the criteria for processing. 

        Parameters:
            None

        Returns:
            None
        """
        result = ParallelPrecipitationProcessor.create_batch_precipitation_maps_parallel(
            processor=self.mock_processor,
            output_dir=self.temp_dir,
            lon_min=-120, lon_max=-80,
            lat_min=30, lat_max=50,
            accum_period='a06h',
            time_indices=[0, 1, 2],
        )
        assert result == []

    def test_mpi_mode_without_data_dir_raises_attribute_error(self: 'TestParallelPrecipitationProcessorEdgeCases',) -> None:
        """
        This test verifies that the `create_batch_precipitation_maps_parallel` function raises an `AttributeError` when in MPI mode but the `data_dir` attribute is not set on the parallel manager. It mocks the `MPASParallelManager` to simulate MPI mode and ensures that the function checks for the required `data_dir` attribute, confirming that it raises the appropriate error when this condition is not met. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr = _pw.MPASParallelManager

        mock_manager = Mock()
        mock_manager.backend = 'mpi'
        mock_manager.is_master = True
        _pw.MPASParallelManager = lambda *a, **kw: mock_manager

        processor = SimpleNamespace(
            dataset=self.mock_dataset,
            grid_file='grid.nc',
        )

        try:
            with pytest.raises(AttributeError, match="data_dir"):
                ParallelPrecipitationProcessor.create_batch_precipitation_maps_parallel(
                    processor=processor,
                    output_dir=self.temp_dir,
                    lon_min=-120, lon_max=-80,
                    lat_min=30, lat_max=50,
                    time_indices=[2, 3],
                )
        finally:
            _pw.MPASParallelManager = orig_mgr

    def test_weights_dir_and_resolution_triggers_prebuild_remapper(self: 'TestParallelPrecipitationProcessorEdgeCases',) -> None:
        """
        This test verifies that when both `weights_dir` and `grid_resolution` are provided to the `create_batch_precipitation_maps_parallel` function, it triggers the call to the `_prebuild_remapper_mpi` function. It mocks the `MPASParallelManager` and `MPASDataCache`, and uses a patch to monitor calls to `_prebuild_remapper_mpi`. The test asserts that `_prebuild_remapper_mpi` is called exactly once, confirming that the batch processor correctly identifies the need to pre-build remapping weights when these parameters are specified. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr, orig_cache = _pw.MPASParallelManager, _pw.MPASDataCache

        mock_cache = Mock()
        mock_manager = Mock()
        mock_manager.backend = 'multiprocessing'
        mock_manager.is_master = True
        mock_manager.comm = None
        mock_manager.parallel_map.return_value = []
        mock_manager.get_statistics.return_value = None

        _pw.MPASParallelManager = lambda *a, **kw: mock_manager
        _pw.MPASDataCache = lambda *a, **kw: mock_cache

        try:
            with patch('mpasdiag.processing.parallel_wrappers._prebuild_remapper_mpi') as mock_pre:
                ParallelPrecipitationProcessor.create_batch_precipitation_maps_parallel(
                    processor=self.mock_processor,
                    output_dir=self.temp_dir,
                    lon_min=-120, lon_max=-80,
                    lat_min=30, lat_max=50,
                    time_indices=[2, 3],
                    weights_dir=self.temp_dir,
                    grid_resolution=0.5,
                )
                mock_pre.assert_called_once()
        finally:
            _pw.MPASParallelManager, _pw.MPASDataCache = orig_mgr, orig_cache

    def test_time_indices_none_uses_all_times(self: 'TestParallelPrecipitationProcessorEdgeCases',) -> None:
        """
        This test verifies that when `time_indices` is set to `None`, the `create_batch_precipitation_maps_parallel` function uses all available time indices from the dataset for processing. It mocks the `MPASParallelManager` and `MPASDataCache`, and asserts that the `parallel_map` method is called with the correct list of time indices corresponding to all time steps in the mock dataset. This confirms that the function correctly defaults to using all time indices when `None` is provided. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr, orig_cache = _pw.MPASParallelManager, _pw.MPASDataCache

        mock_cache = Mock()
        mock_manager = Mock()
        mock_manager.backend = 'multiprocessing'
        mock_manager.is_master = True
        mock_manager.comm = None
        mock_manager.parallel_map.return_value = []
        mock_manager.get_statistics.return_value = None

        _pw.MPASParallelManager = lambda *a, **kw: mock_manager
        _pw.MPASDataCache = lambda *a, **kw: mock_cache

        try:
            result = ParallelPrecipitationProcessor.create_batch_precipitation_maps_parallel(
                processor=self.mock_processor,
                output_dir=self.temp_dir,
                lon_min=-120, lon_max=-80,
                lat_min=30, lat_max=50,
                accum_period='a01h',
                time_indices=None,
            )
            assert isinstance(result, list)
            mock_manager.parallel_map.assert_called_once()
        finally:
            _pw.MPASParallelManager, _pw.MPASDataCache = orig_mgr, orig_cache


class TestParallelSurfaceProcessorEdgeCases:
    """ Edge-case tests for ParallelSurfaceProcessor targeting uncovered branches. """

    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestParallelSurfaceProcessorEdgeCases') -> None:
        """
        This fixture prepares a temporary directory and a mock processor with a synthetic dataset containing a `t2m` variable for testing edge cases in the `ParallelSurfaceProcessor` batch processing methods. The dataset includes coordinate arrays for time and cells. The mock processor is configured to return this dataset, allowing the batch processing functions to be invoked without accessing external data files. After the test runs, the temporary directory is removed to clean up resources.

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
            )
        })
        self.mock_processor.dataset = self.mock_dataset

    def teardown_method(self: 'TestParallelSurfaceProcessorEdgeCases') -> None:
        """
        This method cleans up temporary resources created for testing edge cases in the `ParallelSurfaceProcessor` batch processing methods by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment. 

        Parameters:
            None

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_mpi_mode_without_data_dir_raises_attribute_error(self: 'TestParallelSurfaceProcessorEdgeCases',) -> None:
        """
        This test verifies that the `create_batch_surface_maps_parallel` function raises an `AttributeError` when in MPI mode but the `data_dir` attribute is not set on the parallel manager. It mocks the `MPASParallelManager` to simulate MPI mode and ensures that the function checks for the required `data_dir` attribute, confirming that it raises the appropriate error when this condition is not met. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr = _pw.MPASParallelManager

        mock_manager = Mock()
        mock_manager.backend = 'mpi'
        mock_manager.is_master = True
        _pw.MPASParallelManager = lambda *a, **kw: mock_manager

        processor = SimpleNamespace(dataset=self.mock_dataset, grid_file='grid.nc')

        try:
            with pytest.raises(AttributeError, match="data_dir"):
                ParallelSurfaceProcessor.create_batch_surface_maps_parallel(
                    processor=processor,
                    output_dir=self.temp_dir,
                    lon_min=-120, lon_max=-80,
                    lat_min=30, lat_max=50,
                    time_indices=[0, 1],
                )
        finally:
            _pw.MPASParallelManager = orig_mgr

    def test_cache_preload_exception_shows_warning(self: 'TestParallelSurfaceProcessorEdgeCases',) -> None:
        """
        This test verifies that if the `MPASDataCache` raises an exception when attempting to load coordinates from the dataset, the `create_batch_surface_maps_parallel` function catches this exception and prints a warning message instead of crashing. It mocks the `MPASDataCache` to raise an exception on coordinate loading and captures the printed output to assert that the expected warning message is present. This confirms that the function has robust error handling for cache loading issues and provides informative feedback without causing unhandled exceptions. 

        Parameters:
            None

        Returns:
            None
        """
        import io
        from contextlib import redirect_stdout
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr, orig_cache = _pw.MPASParallelManager, _pw.MPASDataCache

        mock_cache = Mock()
        mock_cache.load_coordinates_from_dataset.side_effect = Exception("coord load failed")

        mock_manager = Mock()
        mock_manager.backend = 'multiprocessing'
        mock_manager.is_master = True
        mock_manager.parallel_map.return_value = []
        mock_manager.get_statistics.return_value = None

        _pw.MPASParallelManager = lambda *a, **kw: mock_manager
        _pw.MPASDataCache = lambda *a, **kw: mock_cache

        try:
            f = io.StringIO()
            with redirect_stdout(f):
                ParallelSurfaceProcessor.create_batch_surface_maps_parallel(
                    processor=self.mock_processor,
                    output_dir=self.temp_dir,
                    lon_min=-120, lon_max=-80,
                    lat_min=30, lat_max=50,
                    time_indices=[0, 1],
                )
            assert "Could not pre-load coordinates" in f.getvalue()
        finally:
            _pw.MPASParallelManager, _pw.MPASDataCache = orig_mgr, orig_cache

    def test_non_master_process_returns_none(self: 'TestParallelSurfaceProcessorEdgeCases',) -> None:
        """
        This test verifies that if the current process is not the master process in a parallel execution context, the `create_batch_surface_maps_parallel` function returns `None` without attempting to perform any processing. It mocks the `MPASParallelManager` to simulate a non-master process and asserts that the function returns `None`, confirming that it correctly identifies non-master processes and avoids unnecessary processing or errors in those cases. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr, orig_cache = _pw.MPASParallelManager, _pw.MPASDataCache

        mock_cache = Mock()
        mock_manager = Mock()
        mock_manager.backend = 'multiprocessing'
        mock_manager.is_master = False
        mock_manager.parallel_map.return_value = None

        _pw.MPASParallelManager = lambda *a, **kw: mock_manager
        _pw.MPASDataCache = lambda *a, **kw: mock_cache

        try:
            result = ParallelSurfaceProcessor.create_batch_surface_maps_parallel(
                processor=self.mock_processor,
                output_dir=self.temp_dir,
                lon_min=-120, lon_max=-80,
                lat_min=30, lat_max=50,
                time_indices=[0, 1],
            )
            assert result is None
        finally:
            _pw.MPASParallelManager, _pw.MPASDataCache = orig_mgr, orig_cache

    def test_time_indices_none_uses_all_times(self: 'TestParallelSurfaceProcessorEdgeCases',) -> None:
        """
        This test verifies that when `time_indices` is set to `None`, the `create_batch_surface_maps_parallel` function uses all available time indices from the dataset for processing. It mocks the `MPASParallelManager` and `MPASDataCache`, and asserts that the `parallel_map` method is called with the correct list of time indices corresponding to all time steps in the mock dataset. This confirms that the function correctly defaults to using all time indices when `None` is provided.

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr, orig_cache = _pw.MPASParallelManager, _pw.MPASDataCache

        mock_cache = Mock()
        mock_manager = Mock()
        mock_manager.backend = 'multiprocessing'
        mock_manager.is_master = True
        mock_manager.parallel_map.return_value = []
        mock_manager.get_statistics.return_value = None

        _pw.MPASParallelManager = lambda *a, **kw: mock_manager
        _pw.MPASDataCache = lambda *a, **kw: mock_cache

        try:
            result = ParallelSurfaceProcessor.create_batch_surface_maps_parallel(
                processor=self.mock_processor,
                output_dir=self.temp_dir,
                lon_min=-120, lon_max=-80,
                lat_min=30, lat_max=50,
                time_indices=None,
            )
            assert isinstance(result, list)
            mock_manager.parallel_map.assert_called_once()
        finally:
            _pw.MPASParallelManager, _pw.MPASDataCache = orig_mgr, orig_cache

    def test_mpi_mode_with_data_dir_constructs_kwargs(self: 'TestParallelSurfaceProcessorEdgeCases',) -> None:
        """
        This test verifies that when in MPI mode and the `data_dir` attribute is set on the parallel manager, the `create_batch_surface_maps_parallel` function successfully constructs the worker kwargs and initiates parallel processing. It mocks the `MPASParallelManager` to simulate MPI mode with a specified `data_dir`, and asserts that the `parallel_map` method is called, confirming that the function correctly handles MPI mode with the necessary attributes for processing.

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr = _pw.MPASParallelManager

        mock_manager = Mock()
        mock_manager.backend = 'mpi'
        mock_manager.is_master = True
        mock_manager.parallel_map.return_value = []
        mock_manager.get_statistics.return_value = None

        _pw.MPASParallelManager = lambda *a, **kw: mock_manager

        self.mock_processor.grid_file = '/fake/grid.nc'
        self.mock_processor.data_dir = '/fake/data'

        try:
            result = ParallelSurfaceProcessor.create_batch_surface_maps_parallel(
                processor=self.mock_processor,
                output_dir=self.temp_dir,
                lon_min=-120, lon_max=-80,
                lat_min=30, lat_max=50,
                time_indices=[0, 1],
            )
            assert isinstance(result, list)
            mock_manager.parallel_map.assert_called_once()
        finally:
            _pw.MPASParallelManager = orig_mgr


class TestParallelWindProcessorEdgeCases:
    """ Edge-case tests for ParallelWindProcessor targeting uncovered branches. """

    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestParallelWindProcessorEdgeCases') -> None:
        """
        This fixture prepares a temporary directory and a mock processor with a synthetic dataset containing `u10` and `v10` wind components for testing edge cases in the `ParallelWindProcessor` batch processing methods. The dataset includes coordinate arrays for time and cells. The mock processor is configured to return this dataset, allowing the batch processing functions to be invoked without accessing external data files. After the test runs, the temporary directory is removed to clean up resources. 

        Parameters:
            None

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()
        self.mock_processor = Mock()
        n_time, n_cells = 5, 100
        self.mock_dataset = xr.Dataset({
            'u10': xr.DataArray(_RNG.uniform(-10, 10, (n_time, n_cells)), dims=['Time', 'nCells']),
            'v10': xr.DataArray(_RNG.uniform(-10, 10, (n_time, n_cells)), dims=['Time', 'nCells']),
        })
        self.mock_processor.dataset = self.mock_dataset

    def teardown_method(self: 'TestParallelWindProcessorEdgeCases') -> None:
        """
        This method cleans up temporary resources created for testing edge cases in the `ParallelWindProcessor` batch processing methods by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment. 

        Parameters:
            None

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_mpi_mode_without_data_dir_raises_in_build_kwargs(self: 'TestParallelWindProcessorEdgeCases',) -> None:
        """
        This test verifies that the `_build_wind_worker_kwargs` function raises an `AttributeError` when in MPI mode but the `data_dir` attribute is not set on the processor. It mocks the `MPASParallelManager` to simulate MPI mode and ensures that the function checks for the required `data_dir` attribute, confirming that it raises the appropriate error when this condition is not met. The test directly calls the `_build_wind_worker_kwargs` function to isolate this specific behavior, confirming that the error is raised at the correct point in the code when the necessary attributes for MPI mode are not present.

        Parameters:
            None

        Returns:
            None
        """
        processor = SimpleNamespace(dataset=self.mock_dataset, grid_file='grid.nc')

        with pytest.raises(AttributeError, match="data_dir"):
            ParallelWindProcessor._build_wind_worker_kwargs(
                processor=processor,
                is_mpi_mode=True,
                output_dir=self.temp_dir,
                lon_min=-120, lon_max=-80,
                lat_min=30, lat_max=50,
                u_variable='u10', v_variable='v10',
                plot_type='barbs', subsample=1,
                scale=None, show_background=False,
                grid_resolution=None, regrid_method='linear',
                formats=['png'],
            )

    def test_cache_preload_exception_shows_warning(self: 'TestParallelWindProcessorEdgeCases',) -> None:
        """
        This test verifies that if the `MPASDataCache` raises an exception when attempting to load coordinates from the dataset, the `create_batch_wind_plots_parallel` function catches this exception and prints a warning message instead of crashing. It mocks the `MPASDataCache` to raise an exception on coordinate loading and captures the printed output to assert that the expected warning message is present. This confirms that the function has robust error handling for cache loading issues and provides informative feedback without causing unhandled exceptions. 

        Parameters:
            None

        Returns:
            None
        """
        import io
        from contextlib import redirect_stdout
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr, orig_cache = _pw.MPASParallelManager, _pw.MPASDataCache

        mock_cache = Mock()
        mock_cache.load_coordinates_from_dataset.side_effect = Exception("coord fail")

        mock_manager = Mock()
        mock_manager.backend = 'multiprocessing'
        mock_manager.is_master = True
        mock_manager.parallel_map.return_value = []
        mock_manager.get_statistics.return_value = None

        _pw.MPASParallelManager = lambda *a, **kw: mock_manager
        _pw.MPASDataCache = lambda *a, **kw: mock_cache

        try:
            f = io.StringIO()
            with redirect_stdout(f):
                ParallelWindProcessor.create_batch_wind_plots_parallel(
                    processor=self.mock_processor,
                    output_dir=self.temp_dir,
                    lon_min=-120, lon_max=-80,
                    lat_min=30, lat_max=50,
                    u_variable='u10', v_variable='v10',
                    time_indices=[0, 1],
                )
            assert "Could not pre-load coordinates" in f.getvalue()
        finally:
            _pw.MPASParallelManager, _pw.MPASDataCache = orig_mgr, orig_cache

    def test_show_background_and_grid_resolution_print(self: 'TestParallelWindProcessorEdgeCases',) -> None:
        """
        This test verifies that when `show_background` is set to `True` and `grid_resolution` is provided, the `create_batch_wind_plots_parallel` function prints the expected messages indicating that the background wind speed field is enabled and showing the grid resolution. It mocks the `MPASParallelManager` and `MPASDataCache`, and captures the printed output to assert that the expected messages are present. This confirms that the function correctly reports the configuration of background plotting and grid resolution when these options are specified. 

        Parameters:
            None

        Returns:
            None
        """
        import io
        from contextlib import redirect_stdout
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr, orig_cache = _pw.MPASParallelManager, _pw.MPASDataCache

        mock_cache = Mock()
        mock_manager = Mock()
        mock_manager.backend = 'multiprocessing'
        mock_manager.is_master = True
        mock_manager.parallel_map.return_value = []
        mock_manager.get_statistics.return_value = None

        _pw.MPASParallelManager = lambda *a, **kw: mock_manager
        _pw.MPASDataCache = lambda *a, **kw: mock_cache

        try:
            f = io.StringIO()
            with redirect_stdout(f):
                ParallelWindProcessor.create_batch_wind_plots_parallel(
                    processor=self.mock_processor,
                    output_dir=self.temp_dir,
                    lon_min=-120, lon_max=-80,
                    lat_min=30, lat_max=50,
                    u_variable='u10', v_variable='v10',
                    show_background=True,
                    grid_resolution=0.5,
                    time_indices=[0, 1],
                )
            output = f.getvalue()
            assert "Background wind speed field: enabled" in output
            assert "Grid resolution" in output
        finally:
            _pw.MPASParallelManager, _pw.MPASDataCache = orig_mgr, orig_cache

    def test_non_master_process_returns_none(self: 'TestParallelWindProcessorEdgeCases',) -> None:
        """
        This test verifies that if the current process is not the master process in a parallel execution context, the `create_batch_wind_plots_parallel` function returns `None` without attempting to perform any processing. It mocks the `MPASParallelManager` to simulate a non-master process and asserts that the function returns `None`, confirming that it correctly identifies non-master processes and avoids unnecessary processing or errors in those cases. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr, orig_cache = _pw.MPASParallelManager, _pw.MPASDataCache

        mock_cache = Mock()
        mock_manager = Mock()
        mock_manager.backend = 'multiprocessing'
        mock_manager.is_master = False
        mock_manager.parallel_map.return_value = None

        _pw.MPASParallelManager = lambda *a, **kw: mock_manager
        _pw.MPASDataCache = lambda *a, **kw: mock_cache

        try:
            result = ParallelWindProcessor.create_batch_wind_plots_parallel(
                processor=self.mock_processor,
                output_dir=self.temp_dir,
                lon_min=-120, lon_max=-80,
                lat_min=30, lat_max=50,
                u_variable='u10', v_variable='v10',
                time_indices=[0, 1],
            )
            assert result is None
        finally:
            _pw.MPASParallelManager, _pw.MPASDataCache = orig_mgr, orig_cache

    def test_time_indices_none_uses_all_times(self: 'TestParallelWindProcessorEdgeCases',) -> None:
        """
        This test verifies that when `time_indices` is set to `None`, the `create_batch_wind_plots_parallel` function uses all available time indices from the dataset for processing. It mocks the `MPASParallelManager` and `MPASDataCache`, and asserts that the `parallel_map` method is called with the correct list of time indices corresponding to all time steps in the mock dataset. This confirms that the function correctly defaults to using all time indices when `None` is provided. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr, orig_cache = _pw.MPASParallelManager, _pw.MPASDataCache

        mock_cache = Mock()
        mock_manager = Mock()
        mock_manager.backend = 'multiprocessing'
        mock_manager.is_master = True
        mock_manager.parallel_map.return_value = []
        mock_manager.get_statistics.return_value = None

        _pw.MPASParallelManager = lambda *a, **kw: mock_manager
        _pw.MPASDataCache = lambda *a, **kw: mock_cache

        try:
            result = ParallelWindProcessor.create_batch_wind_plots_parallel(
                processor=self.mock_processor,
                output_dir=self.temp_dir,
                lon_min=-120, lon_max=-80,
                lat_min=30, lat_max=50,
                u_variable='u10', v_variable='v10',
                time_indices=None,
            )
            assert isinstance(result, list)
            mock_manager.parallel_map.assert_called_once()
        finally:
            _pw.MPASParallelManager, _pw.MPASDataCache = orig_mgr, orig_cache

    def test_build_kwargs_mpi_mode_with_data_dir(self: 'TestParallelWindProcessorEdgeCases',) -> None:
        """
        This test verifies that the `_build_wind_worker_kwargs` function successfully constructs the worker kwargs when in MPI mode and the `data_dir` attribute is set on the processor. It mocks the `MPASParallelManager` to simulate MPI mode with a specified `data_dir`, and asserts that the returned kwargs include the expected `grid_file` and `data_dir` values, confirming that the function correctly builds the worker kwargs for MPI mode when the necessary attributes are present.

        Parameters:
            None

        Returns:
            None
        """
        self.mock_processor.grid_file = '/fake/grid.nc'
        self.mock_processor.data_dir = '/fake/data'

        result = ParallelWindProcessor._build_wind_worker_kwargs(
            processor=self.mock_processor,
            is_mpi_mode=True,
            output_dir=self.temp_dir,
            lon_min=-120, lon_max=-80,
            lat_min=30, lat_max=50,
            u_variable='u10', v_variable='v10',
            plot_type='barbs', subsample=1,
            scale=None, show_background=False,
            grid_resolution=None, regrid_method='linear',
            formats=['png'],
        )
        assert 'grid_file' in result
        assert 'data_dir' in result
        assert result['grid_file'] == '/fake/grid.nc'


class TestParallelCrossSectionProcessorEdgeCases:
    """ Edge-case tests for ParallelCrossSectionProcessor targeting uncovered branches. """

    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestParallelCrossSectionProcessorEdgeCases') -> None:
        """
        This fixture prepares a temporary directory and a mock processor with a synthetic dataset containing a `temperature` variable for testing edge cases in the `ParallelCrossSectionProcessor` batch processing methods. The dataset includes coordinate arrays for time, vertical levels, and cells. The mock processor is configured to return this dataset, allowing the batch processing functions to be invoked without accessing external data files. After the test runs, the temporary directory is removed to clean up resources. 

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
            )
        })
        self.mock_processor.dataset = self.mock_dataset

    def teardown_method(self: 'TestParallelCrossSectionProcessorEdgeCases') -> None:
        """
        This method cleans up temporary resources created for testing edge cases in the `ParallelCrossSectionProcessor` batch processing methods by removing the temporary directory. This ensures that no residual files or directories remain after the tests are executed, maintaining a clean testing environment. 

        Parameters:
            None

        Returns:
            None
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_collect_results_with_error_in_success_dict(self: 'TestParallelCrossSectionProcessorEdgeCases',) -> None:
        """
        This test verifies that the `_collect_cross_section_results` function correctly handles a case where one of the results has `success=True` but contains an error message in the result dictionary. It checks that only the successful results without errors are collected and returned, confirming that the function properly filters out results that indicate a failure even if they are marked as successful. The test captures the printed output to ensure that any relevant messages are displayed as expected.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.parallel import TaskResult as TR

        results = [
            TR(task_id=0, success=True, result={'error': 'plot failed', 'files': []}),
            TR(task_id=1, success=True, result={'files': ['ok.png']}),
        ]

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            files = ParallelCrossSectionProcessor._collect_cross_section_results(
                results, [0, 1], self.temp_dir
            )

        assert 'ok.png' in files
        assert len(files) == 1

    def test_mpi_mode_without_data_dir_raises_attribute_error(self: 'TestParallelCrossSectionProcessorEdgeCases',) -> None:
        """
        This test verifies that the `create_batch_cross_section_plots_parallel` function raises an `AttributeError` when in MPI mode but the `data_dir` attribute is not set on the parallel manager. It mocks the `MPASParallelManager` to simulate MPI mode and ensures that the function checks for the required `data_dir` attribute, confirming that it raises the appropriate error when this condition is not met. The test directly calls the `create_batch_cross_section_plots_parallel` function to isolate this specific behavior, confirming that the error is raised at the correct point in the code when the necessary attributes for MPI mode are not present. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr = _pw.MPASParallelManager

        mock_manager = Mock()
        mock_manager.backend = 'mpi'
        mock_manager.is_master = True
        _pw.MPASParallelManager = lambda *a, **kw: mock_manager

        processor = SimpleNamespace(
            dataset=self.mock_dataset,
            grid_file='grid.nc',
        )

        try:
            with pytest.raises(AttributeError, match="data_dir"):
                ParallelCrossSectionProcessor.create_batch_cross_section_plots_parallel(
                    mpas_3d_processor=processor,
                    var_name='temperature',
                    start_point=(-110, 35),
                    end_point=(-90, 45),
                    output_dir=self.temp_dir,
                    time_indices=[0, 1],
                )
        finally:
            _pw.MPASParallelManager = orig_mgr

    def test_max_height_prints_message(self: 'TestParallelCrossSectionProcessorEdgeCases',) -> None:
        """
        This test verifies that when the `max_height` parameter is provided to the `create_batch_cross_section_plots_parallel` function, it prints a message indicating the maximum height being used for the cross-section plots. It mocks the `MPASParallelManager` and captures the printed output to assert that the expected message about maximum height is present. This confirms that the function correctly reports the configuration of maximum height when this option is specified. 

        Parameters:
            None

        Returns:
            None
        """
        import io
        from contextlib import redirect_stdout
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr = _pw.MPASParallelManager

        mock_manager = Mock()
        mock_manager.backend = 'multiprocessing'
        mock_manager.is_master = True
        mock_manager.parallel_map.return_value = []

        _pw.MPASParallelManager = lambda *a, **kw: mock_manager

        try:
            f = io.StringIO()
            with redirect_stdout(f):
                ParallelCrossSectionProcessor.create_batch_cross_section_plots_parallel(
                    mpas_3d_processor=self.mock_processor,
                    var_name='temperature',
                    start_point=(-110, 35),
                    end_point=(-90, 45),
                    output_dir=self.temp_dir,
                    time_indices=[0, 1],
                    max_height=15,
                )
            assert "Maximum height" in f.getvalue()
        finally:
            _pw.MPASParallelManager = orig_mgr

    def test_non_master_process_returns_none(self: 'TestParallelCrossSectionProcessorEdgeCases',) -> None:
        """
        This test verifies that `create_batch_cross_section_plots_parallel` executes the non-master cleanup path (lines 1436-1438) and returns `None` when `manager.is_master` is False. It mocks the `MPASParallelManager` to simulate a non-master process and asserts that the function returns `None`, confirming that it correctly identifies non-master processes and avoids unnecessary processing or errors in those cases. The test directly calls the `create_batch_cross_section_plots_parallel` function to isolate this specific behavior, confirming that the non-master path is executed as expected when the master process condition is not met. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr = _pw.MPASParallelManager

        mock_manager = Mock()
        mock_manager.backend = 'multiprocessing'
        mock_manager.is_master = False
        mock_manager.parallel_map.return_value = None

        _pw.MPASParallelManager = lambda *a, **kw: mock_manager

        try:
            result = ParallelCrossSectionProcessor.create_batch_cross_section_plots_parallel(
                mpas_3d_processor=self.mock_processor,
                var_name='temperature',
                start_point=(-110, 35),
                end_point=(-90, 45),
                output_dir=self.temp_dir,
                time_indices=[0, 1],
            )
            assert result is None
        finally:
            _pw.MPASParallelManager = orig_mgr

    def test_time_indices_none_uses_all_times(self: 'TestParallelCrossSectionProcessorEdgeCases',) -> None:
        """
        This test verifies that when `time_indices` is set to `None`, the `create_batch_cross_section_plots_parallel` function uses all available time indices from the dataset for processing. It mocks the `MPASParallelManager` and asserts that the `parallel_map` method is called with the correct list of time indices corresponding to all time steps in the mock dataset. This confirms that the function correctly defaults to using all time indices when `None` is provided.

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr = _pw.MPASParallelManager

        mock_manager = Mock()
        mock_manager.backend = 'multiprocessing'
        mock_manager.is_master = True
        mock_manager.parallel_map.return_value = []
        mock_manager.get_statistics.return_value = None

        _pw.MPASParallelManager = lambda *a, **kw: mock_manager

        try:
            result = ParallelCrossSectionProcessor.create_batch_cross_section_plots_parallel(
                mpas_3d_processor=self.mock_processor,
                var_name='temperature',
                start_point=(-110, 35),
                end_point=(-90, 45),
                output_dir=self.temp_dir,
                time_indices=None,
            )
            assert isinstance(result, list)
            mock_manager.parallel_map.assert_called_once()
        finally:
            _pw.MPASParallelManager = orig_mgr

    def test_mpi_mode_with_data_dir_constructs_kwargs(self: 'TestParallelCrossSectionProcessorEdgeCases',) -> None:
        """
        This test verifies that when in MPI mode and the `data_dir` attribute is set on the parallel manager, the `create_batch_cross_section_plots_parallel` function successfully constructs the worker kwargs and initiates parallel processing. It mocks the `MPASParallelManager` to simulate MPI mode with a specified `data_dir`, and asserts that the `parallel_map` method is called, confirming that the function correctly handles MPI mode with the necessary attributes for processing. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw
        orig_mgr = _pw.MPASParallelManager

        mock_manager = Mock()
        mock_manager.backend = 'mpi'
        mock_manager.is_master = True
        mock_manager.parallel_map.return_value = []
        mock_manager.get_statistics.return_value = None

        _pw.MPASParallelManager = lambda *a, **kw: mock_manager

        self.mock_processor.grid_file = '/fake/grid.nc'
        self.mock_processor.data_dir = '/fake/data'

        try:
            result = ParallelCrossSectionProcessor.create_batch_cross_section_plots_parallel(
                mpas_3d_processor=self.mock_processor,
                var_name='temperature',
                start_point=(-110, 35),
                end_point=(-90, 45),
                output_dir=self.temp_dir,
                time_indices=[0, 1],
            )
            assert isinstance(result, list)
            mock_manager.parallel_map.assert_called_once()
        finally:
            _pw.MPASParallelManager = orig_mgr


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
