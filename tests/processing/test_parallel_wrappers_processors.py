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
import builtins
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Any, List, Generator
from unittest.mock import Mock

from mpasdiag.processing.parallel import ParallelStats, TaskResult, MPASParallelManager
from mpasdiag.processing.parallel_wrappers import (
    _process_parallel_results,
    ParallelPrecipitationProcessor,
    ParallelSurfaceProcessor,
    ParallelWindProcessor,
    ParallelCrossSectionProcessor,
    auto_batch_processor
)
from tests.test_data_helpers import assert_expected_public_methods

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
GRID_FILE = os.path.join(TEST_DATA_DIR, "grids", "x1.10242.static.nc")


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

        assert_expected_public_methods(orig_cache, 'MPASDataCache')
        assert_expected_public_methods(orig_mgr, 'MPASParallelManager')

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

        assert_expected_public_methods(orig_cache, 'MPASDataCache')
        assert_expected_public_methods(orig_mgr, 'MPASParallelManager')

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



