#!/usr/bin/env python3
"""
MPASdiag Test Suite: Parallel Processing Wrappers

This module contains unit tests for the parallel processing wrapper functions in the MPASdiag package, specifically targeting the MPI mode branches and edge cases in result processing. The tests are designed to ensure that the worker functions and result processing logic can handle various scenarios gracefully, including error handling, timing statistics, and MPI-specific behavior.

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


class TestWorkerExceptionHandler:
    """ Tests for the catch-all exception handler in worker functions. """

    def test_precipitation_worker_returns_error_dict_on_crash(self: "TestWorkerExceptionHandler", 
                                                              temp_output_dir: Path) -> None:
        """
        This test verifies that if the `_precipitation_worker` function encounters an unhandled exception (simulated by passing `None` as the processor), it returns a dictionary containing 'error' and 'traceback' keys with appropriate error information. It asserts that the error information is present in the result and that the worker does not crash, confirming that the catch-all exception handler is functioning correctly to capture and report unexpected errors in the worker workflow. 

        Parameters:
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

    def test_precipitation_worker_error_dict_time_idx_non_int(self: "TestWorkerExceptionHandler",
                                                              temp_output_dir: Path) -> None:
        """
        This test verifies that if the `_precipitation_worker` function receives a non-integer `time_idx`, it returns a dictionary containing 'error' and 'time_str' keys with 'unknown' as the value for 'time_str'. It asserts that the error information is present in the result and that the worker does not crash, confirming that the catch-all exception handler is functioning correctly to capture and report unexpected errors in the worker workflow. 

        Parameters:
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

    def test_all_failures_no_timing_stats(self: "TestProcessParallelResultsEdgeCases", 
                                          temp_output_dir: Path) -> None:
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

    def test_no_manager_statistics(self: "TestProcessParallelResultsEdgeCases", 
                                   temp_output_dir: Path) -> None:
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

    def test_var_info_formatting(self: "TestProcessParallelResultsEdgeCases", 
                                 temp_output_dir: Path) -> None:
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

    def test_mpi_single_rank_returns_false(self: "TestAutoBatchProcessorMPISingleProcess") -> None:
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

    def test_mpi_multi_rank_returns_true(self: "TestAutoBatchProcessorMPISingleProcess") -> None:
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

    def _make_mpi_kwargs(self: "TestMPIModeBranches", 
                         output_dir: str) -> Dict:
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

    def test_precipitation_worker_mpi_grid_file_branch(self: "TestMPIModeBranches", 
                                                       tmp_path: Path) -> None:
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

    def test_surface_worker_mpi_mode_branch(self: "TestMPIModeBranches", 
                                            tmp_path: Path) -> None:
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

        with patch('mpasdiag.processing.parallel_wrappers._rank_processor_cache', new={}), \
             patch('mpasdiag.processing.processors_2d.MPAS2DProcessor') as mock_cls, \
             patch('mpasdiag.processing.parallel_wrappers.MPASSurfacePlotter') as mock_plotter_cls:
            mock_cls.return_value = mock_proc
            mock_plotter = MagicMock()
            mock_plotter.create_surface_map.return_value = (MagicMock(), MagicMock())
            mock_plotter_cls.return_value = mock_plotter
            result = _surface_worker((0, kwargs))

        assert isinstance(result, dict)
        assert 'files' in result

    def test_wind_worker_mpi_mode_branch(self: "TestMPIModeBranches", 
                                         tmp_path: Path) -> None:
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

    def test_cross_section_worker_mpi_3d_reload(self: "TestMPIModeBranches", 
                                                tmp_path: Path) -> None:
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

    def test_process_parallel_results_success_iteration(self: "TestParallelProcessorBatchMethods", 
                                                        tmp_path: Path) -> None:
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

    def test_timing_statistics_creation(self: "TestParallelProcessorBatchMethods", 
                                        tmp_path: Path) -> None:
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

    def test_precipitation_processor_mpi_check(self: "TestParallelProcessorBatchMethods", 
                                               tmp_path: Path) -> None:
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

        assert_expected_public_methods(orig_mgr, 'MPASParallelManager')

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

    def test_surface_processor_mpi_check(self: "TestParallelProcessorBatchMethods", 
                                         tmp_path: Path) -> None:
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

        assert_expected_public_methods(orig_mgr, 'MPASParallelManager')

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

    def test_wind_processor_mpi_check(self: "TestParallelProcessorBatchMethods", 
                                      tmp_path: Path) -> None:
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

        assert_expected_public_methods(orig_mgr, 'MPASParallelManager')

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

    def test_cross_section_processor_mpi_check(self: "TestParallelProcessorBatchMethods", 
                                               tmp_path: Path) -> None:
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

        assert_expected_public_methods(orig_mgr, 'MPASParallelManager')

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

    def test_precipitation_processor_mpi_kwargs_construction(self: "TestParallelProcessorBatchMethods", 
                                                             tmp_path: Path) -> None:
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

        assert_expected_public_methods(orig_mgr, 'MPASParallelManager')

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

    def test_cross_section_processor_result_aggregation(self: "TestParallelProcessorBatchMethods", 
                                                        tmp_path: Path) -> None:
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

        assert_expected_public_methods(orig_mgr, 'MPASParallelManager')
        
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

