#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
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
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from contextlib import redirect_stdout
from typing import Dict, Optional
from unittest.mock import Mock, MagicMock, patch

from mpasdiag.processing.parallel import ParallelStats, TaskResult
from mpasdiag.processing.parallel_wrappers import (
    _precipitation_worker,
    _surface_worker,
    _wind_worker,
    _cross_section_worker,
    _skewt_worker,
    _seed_worker_processor_cache,
    _process_parallel_results,
    ParallelPrecipitationProcessor,
    ParallelWindProcessor,
    ParallelCrossSectionProcessor,
    ParallelSkewTProcessor,
)
from mpasdiag.processing.processors_3d import MPAS3DProcessor
from tests.test_data_helpers import assert_expected_public_methods
from mpasdiag.processing.utils_geog import GeographicBounds

TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
)
GRID_FILE = os.path.join(TEST_DATA_DIR, "grids", "x1.10242.static.nc")


_RNG = np.random.default_rng(42)


class TestWorkerExceptionHandler:
    """Tests for the catch-all exception handler in worker functions."""

    def test_precipitation_worker_returns_error_dict_on_crash(
        self: "TestWorkerExceptionHandler", temp_output_dir: Path
    ) -> None:
        """
        This test verifies that if the `_precipitation_worker` function encounters an unhandled exception (simulated by passing `None` as the processor), it returns a dictionary containing 'error' and 'traceback' keys with appropriate error information. It asserts that the error information is present in the result and that the worker does not crash, confirming that the catch-all exception handler is functioning correctly to capture and report unexpected errors in the worker workflow.

        Parameters:
            temp_output_dir (Path): Temporary directory for output files.

        Returns:
            None
        """
        kwargs = {
            "processor": None,
            "cache": None,
            "output_dir": str(temp_output_dir),
            "lon_min": -120,
            "lon_max": -80,
            "lat_min": 30,
            "lat_max": 50,
            "var_name": "rainnc",
            "accum_period": "a01h",
            "file_prefix": "crash_test",
            "formats": ["png"],
        }

        captured = io.StringIO()
        with redirect_stdout(captured):
            result = _precipitation_worker((0, kwargs))

        assert "error" in result
        assert "traceback" in result
        assert result["files"] == []
        assert result["timings"] == {}
        assert "WORKER ERROR" in captured.getvalue()


class TestProcessParallelResultsEdgeCases:
    """Tests for result processing with all-failure results and absent stats."""

    def test_no_manager_statistics(
        self: "TestProcessParallelResultsEdgeCases", temp_output_dir: Path
    ) -> None:
        """
        This test verifies that if the `_process_parallel_results` function receives a manager with no statistics, the report is printed without speedup information. It creates a list of `TaskResult` objects where one task indicates success with valid results, and mocks the manager to return no statistics. The test asserts that the function processes the results without crashing, returns the expected files, and that the output includes the report header but omits any speedup information, confirming that the function can gracefully handle scenarios where performance statistics are unavailable.

        Parameters:
            temp_output_dir (Path): Temporary directory for output files.

        Returns:
            None
        """
        results = [
            TaskResult(
                task_id=0,
                success=True,
                result={
                    "files": ["a.png"],
                    "timings": {
                        "data_processing": 0.1,
                        "plotting": 0.2,
                        "saving": 0.05,
                        "total": 0.35,
                    },
                },
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

    def test_var_info_formatting(
        self: "TestProcessParallelResultsEdgeCases", temp_output_dir: Path
    ) -> None:
        """
        This test verifies that if the `_process_parallel_results` function receives a `var_info` string, it is included in the report header. It creates a list of `TaskResult` objects where one task indicates success with valid results, and mocks the manager to return no statistics. The test asserts that the function processes the results without crashing, returns the expected files, and that the output includes the provided `var_info` string in the report header, confirming that the function correctly integrates variable information into the result reporting.

        Parameters:
            temp_output_dir (Path): Temporary directory for output files.

        Returns:
            None
        """
        results = [
            TaskResult(
                task_id=0,
                success=True,
                result={
                    "files": ["b.png"],
                    "timings": {
                        "data_processing": 0.1,
                        "plotting": 0.2,
                        "saving": 0.05,
                        "total": 0.35,
                    },
                },
            ),
        ]
        mock_manager = Mock()
        mock_manager.get_statistics.return_value = None

        captured = io.StringIO()

        with redirect_stdout(captured):
            _process_parallel_results(
                results,
                [0],
                str(temp_output_dir),
                mock_manager,
                "VARINFO_TEST",
                var_info="Variable: temperature_2m",
            )

        assert "Variable: temperature_2m" in captured.getvalue()


class TestMPIModeBranches:
    """Tests exercising MPI-mode branches in worker functions and Parallel*Processor classes."""

    def _make_mpi_kwargs(self: "TestMPIModeBranches", output_dir: str) -> Dict:
        """
        This helper method creates a dictionary of keyword arguments for worker functions that will trigger the MPI mode branches by including 'grid_file' and 'data_dir' keys. The values are set to fake paths since the actual file reading is mocked in the tests.

        Parameters:
            output_dir (str): Directory for output files.

        Returns:
            Dict: Keyword arguments for MPI mode functions.
        """
        return {
            "grid_file": "/fake/grid.nc",
            "data_dir": "/fake/data",
            "output_dir": output_dir,
            "lon_min": -120,
            "lon_max": -80,
            "lat_min": 30,
            "lat_max": 50,
        }

    def test_precipitation_worker_mpi_grid_file_branch(
        self: "TestMPIModeBranches", tmp_path: Path
    ) -> None:
        """
        This test verifies that the `_precipitation_worker` function correctly executes the MPI mode branch when `grid_file` and `data_dir` are provided. It mocks the processor to return a dataset with a `rainnc` variable and configures the diagnostic and plotter mocks to simulate the computation and plotting of precipitation data. The test asserts that the worker function completes without crashing and returns a result containing a 'files' key, confirming that the MPI mode branch is functioning correctly in the precipitation worker workflow.

        Parameters:
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None
        """
        mock_proc = MagicMock()

        mock_proc.dataset = xr.Dataset(
            {
                "rainnc": (["Time", "nCells"], np.zeros((2, 10))),
                "Time": (
                    ["Time"],
                    pd.date_range("2025-01-01", periods=2, freq="h").values,
                ),
            }
        )

        mock_proc.load_2d_data.return_value = mock_proc
        mock_proc.extract_2d_coordinates_for_variable.return_value = (
            np.zeros(10),
            np.zeros(10),
        )
        mock_proc.data_type = "history"

        kwargs = self._make_mpi_kwargs(str(tmp_path))

        kwargs.update(
            {
                "var_name": "rainnc",
                "accum_period": "a01h",
                "file_prefix": "test",
                "formats": ["png"],
                "plot_type": "scatter",
                "grid_resolution": None,
                "custom_title_template": None,
                "colormap": None,
                "levels": None,
            }
        )

        with (
            patch("mpasdiag.processing.processors_2d.MPAS2DProcessor") as mock_cls,
            patch(
                "mpasdiag.processing.parallel_wrappers.PrecipitationDiagnostics"
            ) as mock_diag_cls,
            patch(
                "mpasdiag.processing.parallel_wrappers.MPASPrecipitationPlotter"
            ) as mock_plotter_cls,
        ):
            mock_cls.return_value = mock_proc
            mock_diag = MagicMock()
            mock_diag.compute_precipitation_difference.return_value = xr.DataArray(
                _RNG.uniform(0, 5, 10)
            )
            mock_diag_cls.return_value = mock_diag
            mock_plotter = MagicMock()
            mock_plotter.create_precipitation_map.return_value = (
                MagicMock(),
                MagicMock(),
            )
            mock_plotter_cls.return_value = mock_plotter
            result = _precipitation_worker((0, kwargs))

        assert isinstance(result, dict)
        assert "files" in result

    def test_surface_worker_mpi_mode_branch(
        self: "TestMPIModeBranches", tmp_path: Path
    ) -> None:
        """
        This test verifies that the `_surface_worker` function correctly executes the MPI mode branch when `grid_file` and `data_dir` are provided. It mocks the processor to return a dataset with a `t2m` variable and configures the plotter mock to simulate the creation of a surface map. The test asserts that the worker function completes without crashing and returns a result containing a 'files' key, confirming that the MPI mode branch is functioning correctly in the surface worker workflow.

        Parameters:
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None
        """
        mock_proc = MagicMock()

        mock_proc.dataset = xr.Dataset(
            {
                "t2m": (["Time", "nCells"], _RNG.uniform(260, 310, (2, 10))),
                "Time": (
                    ["Time"],
                    pd.date_range("2025-01-01", periods=2, freq="h").values,
                ),
            }
        )

        mock_proc.load_2d_data.return_value = mock_proc
        mock_proc.extract_spatial_coordinates.return_value = (
            np.zeros(10),
            np.zeros(10),
        )

        kwargs = self._make_mpi_kwargs(str(tmp_path))

        kwargs.update(
            {
                "var_name": "t2m",
                "plot_type": "contourf",
                "file_prefix": "test",
                "formats": ["png"],
                "custom_title": None,
                "colormap": None,
                "levels": None,
            }
        )

        with (
            patch(
                "mpasdiag.processing.parallel_wrappers._rank_processor_cache", new={}
            ),
            patch("mpasdiag.processing.processors_2d.MPAS2DProcessor") as mock_cls,
            patch(
                "mpasdiag.processing.parallel_wrappers.MPASSurfacePlotter"
            ) as mock_plotter_cls,
        ):
            mock_cls.return_value = mock_proc
            mock_plotter = MagicMock()
            mock_plotter.create_surface_map.return_value = (MagicMock(), MagicMock())
            mock_plotter_cls.return_value = mock_plotter
            result = _surface_worker((0, kwargs))

        assert isinstance(result, dict)
        assert "files" in result

    def test_wind_worker_mpi_mode_branch(
        self: "TestMPIModeBranches", tmp_path: Path
    ) -> None:
        """
        This test verifies that the `_wind_worker` function correctly executes the MPI mode branch when `grid_file` and `data_dir` are provided. It mocks the processor to return a dataset with `u10` and `v10` variables and configures the plotter mock to simulate the creation of a wind plot. The test asserts that the worker function completes without crashing and returns a result containing a 'files' key, confirming that the MPI mode branch is functioning correctly in the wind worker workflow.

        Parameters:
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None
        """
        mock_proc = MagicMock()

        mock_proc.dataset = xr.Dataset(
            {
                "u10": (["Time", "nCells"], _RNG.uniform(-5, 5, (2, 10))),
                "v10": (["Time", "nCells"], _RNG.uniform(-5, 5, (2, 10))),
                "Time": (
                    ["Time"],
                    pd.date_range("2025-01-01", periods=2, freq="h").values,
                ),
            }
        )

        mock_proc.load_2d_data.return_value = mock_proc
        mock_proc.get_2d_variable_data.return_value = xr.DataArray(
            _RNG.uniform(-5, 5, 10)
        )
        mock_proc.extract_2d_coordinates_for_variable.return_value = (
            np.zeros(10),
            np.zeros(10),
        )

        kwargs = self._make_mpi_kwargs(str(tmp_path))

        kwargs.update(
            {
                "u_variable": "u10",
                "v_variable": "v10",
                "plot_type": "barbs",
                "subsample": 1,
                "scale": None,
                "show_background": False,
                "grid_resolution": None,
                "regrid_method": "linear",
                "file_prefix": "test",
                "formats": ["png"],
            }
        )

        with (
            patch("mpasdiag.processing.processors_2d.MPAS2DProcessor") as mock_cls,
            patch(
                "mpasdiag.processing.parallel_wrappers.MPASWindPlotter"
            ) as mock_plotter_cls,
        ):
            mock_cls.return_value = mock_proc
            mock_plotter = MagicMock()
            mock_plotter.create_wind_plot.return_value = (MagicMock(), MagicMock())
            mock_plotter_cls.return_value = mock_plotter
            result = _wind_worker((0, kwargs))

        assert isinstance(result, dict)
        assert "files" in result

    def test_cross_section_worker_mpi_3d_reload(
        self: "TestMPIModeBranches", tmp_path: Path
    ) -> None:
        """
        This test verifies that the `_cross_section_worker` function correctly executes the MPI mode branch for 3D data when `grid_file` and `data_dir` are provided. It mocks the processor to return a dataset with a `theta` variable and configures the plotter mock to simulate the creation of a vertical cross-section plot. The test asserts that the worker function completes without crashing and returns a result containing a 'files' key, confirming that the MPI mode branch for 3D data reload is functioning correctly in the cross-section worker workflow.

        Parameters:
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None
        """
        mock_proc = MagicMock()

        mock_proc.dataset = xr.Dataset(
            {
                "theta": (["Time", "nVertLevels", "nCells"], np.zeros((2, 5, 10))),
                "Time": (
                    ["Time"],
                    pd.date_range("2025-01-01", periods=2, freq="h").values,
                ),
            }
        )

        mock_proc.load_3d_data.return_value = mock_proc

        kwargs = {
            "grid_file": "/fake/grid.nc",
            "data_dir": "/fake/data",
            "output_dir": str(tmp_path),
            "start_lat": 30,
            "start_lon": -110,
            "end_lat": 40,
            "end_lon": -100,
            "var_name": "theta",
            "file_prefix": "cross",
            "formats": ["png"],
            "custom_title": None,
            "colormap": None,
            "levels": None,
            "vertical_coord": "pressure",
            "num_points": 50,
        }

        with (
            patch("mpasdiag.processing.processors_3d.MPAS3DProcessor") as mock_cls,
            patch(
                "mpasdiag.processing.parallel_wrappers.MPASVerticalCrossSectionPlotter"
            ) as mock_plotter_cls,
        ):
            mock_cls.return_value = mock_proc
            mock_plotter = MagicMock()
            mock_fig = MagicMock()
            mock_plotter.create_vertical_cross_section.return_value = (
                mock_fig,
                MagicMock(),
            )
            mock_plotter_cls.return_value = mock_plotter
            result = _cross_section_worker((0, kwargs))

        assert isinstance(result, dict)
        assert "files" in result

    def test_skewt_worker_mpi_mode_branch(
        self: "TestMPIModeBranches", tmp_path: Path
    ) -> None:
        """
        This test verifies that the `_skewt_worker` function correctly executes the MPI mode branch when `grid_file` and `data_dir` are provided. It mocks the processor to return a dataset with a `theta` variable and configures the diagnostic and plotter mocks to simulate the extraction of a sounding profile, computation of thermodynamic indices, and creation of a skew-T diagram. The test asserts that the worker function completes without crashing and returns a result containing a 'files' key, confirming that the MPI mode branch is functioning correctly in the skew-T worker workflow.

        Parameters:
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None
        """
        mock_proc = MagicMock()

        mock_proc.dataset = xr.Dataset(
            {
                "theta": (["Time", "nVertLevels", "nCells"], np.zeros((2, 5, 10))),
                "Time": (
                    ["Time"],
                    pd.date_range("2025-01-01", periods=2, freq="h").values,
                ),
            }
        )

        mock_proc.load_3d_data.return_value = mock_proc

        kwargs = {
            "grid_file": "/fake/grid.nc",
            "data_dir": "/fake/data",
            "output_dir": str(tmp_path),
            "lon": 103.2,
            "lat": 3.8,
            "file_prefix": "mpas_skewt",
            "formats": ["png"],
            "show_parcel": False,
        }

        profile = {
            "pressure": np.array([1000.0, 900.0, 800.0]),
            "temperature": np.array([25.0, 20.0, 15.0]),
            "dewpoint": np.array([20.0, 15.0, 10.0]),
            "u_wind": np.array([5.0, 6.0, 7.0]),
            "v_wind": np.array([1.0, 2.0, 3.0]),
            "height": np.array([0.0, 1000.0, 2000.0]),
            "station_lon": 103.2,
            "station_lat": 3.8,
        }

        with (
            patch("mpasdiag.processing.processors_3d.MPAS3DProcessor") as mock_cls,
            patch(
                "mpasdiag.processing.parallel_wrappers.SoundingDiagnostics"
            ) as mock_diag_cls,
            patch(
                "mpasdiag.processing.parallel_wrappers.MPASSkewTPlotter"
            ) as mock_plotter_cls,
        ):
            mock_cls.return_value = mock_proc
            mock_diag = MagicMock()
            mock_diag.extract_sounding_profile.return_value = profile
            mock_diag.compute_thermodynamic_indices.return_value = {"cape": 1000.0}
            mock_diag_cls.return_value = mock_diag
            mock_plotter = MagicMock()
            mock_plotter.create_skewt_diagram.return_value = (MagicMock(), MagicMock())
            mock_plotter_cls.return_value = mock_plotter
            result = _skewt_worker((0, kwargs))

        assert isinstance(result, dict)
        assert "files" in result
        assert len(result["files"]) == 1
        assert result["files"][0].endswith(".png")


class TestParallelProcessorBatchMethods:
    """Tests for Parallel*Processor batch methods targeting MPI mode branches."""

    def test_process_parallel_results_success_iteration(
        self: "TestParallelProcessorBatchMethods", tmp_path: Path
    ) -> None:
        """
        This test verifies that the `_process_parallel_results` function correctly processes a list of successful `TaskResult` objects, extracts the files, and prints the appropriate statistics. It mocks a set of successful results with timing information, simulates a manager with relevant statistics, and asserts that the function returns the expected list of files and includes the correct success statistics in the output, confirming that the function can handle and report on successful parallel processing outcomes effectively.

        Parameters:
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None
        """
        results = [
            TaskResult(
                task_id=0,
                success=True,
                result={
                    "files": ["a.png", "b.png"],
                    "timings": {
                        "data_processing": 0.1,
                        "plotting": 0.2,
                        "saving": 0.05,
                        "total": 0.35,
                    },
                },
            ),
            TaskResult(
                task_id=1,
                success=True,
                result={
                    "files": ["c.png"],
                    "timings": {
                        "data_processing": 0.15,
                        "plotting": 0.25,
                        "saving": 0.03,
                        "total": 0.43,
                    },
                },
            ),
        ]

        mock_manager = Mock()

        mock_manager.get_statistics.return_value = ParallelStats(
            total_tasks=2, completed_tasks=2, failed_tasks=0, total_time=0.78
        )

        captured = io.StringIO()

        with redirect_stdout(captured):
            files = _process_parallel_results(
                results, [0, 1], str(tmp_path), mock_manager, "TEST"
            )

        assert len(files) == pytest.approx(3)
        assert "Successful: 2/2" in captured.getvalue()

    def test_timing_statistics_creation(
        self: "TestParallelProcessorBatchMethods", tmp_path: Path
    ) -> None:
        """
        This test verifies that the `_process_parallel_results` function correctly creates a timing_stats dictionary for each key. It mocks a set of successful `TaskResult` objects with timing information, simulates a manager with no statistics, and asserts that the function returns the expected list of files and includes the correct timing breakdown in the output, confirming that the function can generate and report detailed timing statistics even when overall performance statistics are unavailable.

        Parameters:
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None
        """
        results = [
            TaskResult(
                task_id=0,
                success=True,
                result={
                    "files": ["x.png"],
                    "timings": {
                        "data_processing": 0.5,
                        "plotting": 1.0,
                        "saving": 0.2,
                        "total": 1.7,
                    },
                },
            ),
        ]

        mock_manager = Mock()
        mock_manager.get_statistics.return_value = None
        captured = io.StringIO()

        with redirect_stdout(captured):
            files = _process_parallel_results(
                results, [0], str(tmp_path), mock_manager, "TIMING"
            )

        output = captured.getvalue()
        assert "Timing Breakdown" in output
        assert "Data Processing" in output
        assert len(files) == pytest.approx(1)

    def test_precipitation_processor_mpi_kwargs_construction(
        self: "TestParallelProcessorBatchMethods", tmp_path: Path
    ) -> None:
        """
        This test verifies that the `ParallelPrecipitationProcessor` correctly constructs `worker_kwargs` in MPI mode. It mocks a manager in MPI mode, sets up a processor with necessary attributes, and asserts that the `create_batch_precipitation_maps_parallel` method behaves as expected when `parallel_map` returns `None`, confirming that the method can construct the appropriate arguments for worker functions in MPI mode without attempting to process results when `parallel_map` is mocked to return `None`.

        Parameters:
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw

        orig_mgr = _pw.MPASParallelManager

        assert_expected_public_methods(orig_mgr, "MPASParallelManager")

        mock_mgr = MagicMock()
        mock_mgr.backend = "mpi"
        mock_mgr.is_master = True
        mock_mgr.parallel_map.return_value = None
        _pw.MPASParallelManager = lambda *a, **kw: mock_mgr
        try:
            processor = MagicMock()
            processor.grid_file = "/fake/grid.nc"
            processor.data_dir = "/fake/data"
            processor.data_type = "history"
            processor.dataset = MagicMock()
            processor.dataset.sizes = {"Time": 5}

            result = (
                ParallelPrecipitationProcessor.create_batch_precipitation_maps_parallel(
                    processor=processor,
                    output_dir=str(tmp_path),
                    var_name="rainnc",
                    accum_period="a01h",
                    time_indices=[1, 2],
                    n_processes=2,
                    bounds=GeographicBounds(-120, -80, 30, 50),
                )
            )
            assert result is None
            mock_mgr.parallel_map.assert_called_once()
        finally:
            _pw.MPASParallelManager = orig_mgr

    def test_cross_section_processor_result_aggregation(
        self: "TestParallelProcessorBatchMethods", tmp_path: Path
    ) -> None:
        """
        This test verifies that the `ParallelCrossSectionProcessor` correctly aggregates results from `parallel_map` and processes them with `_process_parallel_results`. It mocks a manager in multiprocessing mode, simulates a set of results with one success and one failure, and asserts that the `create_batch_cross_section_plots_parallel` method returns the expected list of files and includes the correct success statistics in the output, confirming that the method can handle and report on mixed outcomes from parallel processing effectively.

        Parameters:
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw

        orig_mgr = _pw.MPASParallelManager

        assert_expected_public_methods(orig_mgr, "MPASParallelManager")

        mock_mgr = MagicMock()
        mock_mgr.backend = "multiprocessing"
        mock_mgr.is_master = True

        results = [
            MagicMock(success=True, result=["cross_0.png"]),
            MagicMock(success=False, result=None, error="Some error", task_id=1),
        ]

        mock_mgr.parallel_map.return_value = results
        _pw.MPASParallelManager = lambda *a, **kw: mock_mgr

        try:
            processor = MagicMock(spec=MPAS3DProcessor)
            processor.grid_file = "/fake/grid.nc"
            processor.data_dir = "/fake/data"
            processor.dataset = MagicMock()
            processor.dataset.sizes = {"Time": 5}

            captured = io.StringIO()

            with redirect_stdout(captured):
                files = ParallelCrossSectionProcessor.create_batch_cross_section_plots_parallel(
                    mpas_3d_processor=processor,
                    output_dir=str(tmp_path),
                    var_name="theta",
                    start_point=(-110, 30),
                    end_point=(-100, 40),
                    time_indices=[0, 1],
                    n_processes=2,
                )
            assert files == ["cross_0.png"]
            assert "Successful: 1/2" in captured.getvalue()
        finally:
            _pw.MPASParallelManager = orig_mgr


class TestPrebuildRemapperMPI:
    """Tests for the _prebuild_remapper_mpi function body (lines 756-770)."""

    def test_prebuild_remapper_mpi_executes(
        self: "TestPrebuildRemapperMPI", tmp_path: Path
    ) -> None:
        """
        This test verifies that the `_prebuild_remapper_mpi` function executes without error when provided with a mock processor and boundary dataset. It mocks the necessary methods to return predefined longitude and latitude arrays, and asserts that the function completes successfully and calls the remapper building method, confirming that the MPI-specific remapper prebuilding logic can run without issues when the required data is available.

        Parameters:
            None

        Returns:
            None
        """
        from unittest.mock import patch, MagicMock
        import numpy as np
        import xarray as xr
        import mpasdiag.processing.parallel_wrappers as _pw

        lon = np.linspace(-120, -80, 10)
        lat = np.linspace(30, 50, 10)
        lon_b = np.linspace(-121, -79, 11)
        lat_b = np.linspace(29, 51, 11)

        boundary_ds = xr.Dataset(
            {
                "lon_b": (["n_b"], lon_b),
                "lat_b": (["n_b"], lat_b),
            }
        )

        mock_proc = MagicMock()
        mock_proc.dataset = xr.Dataset()

        with (
            patch.object(
                _pw.MPASPrecipitationPlotter,
                "_ensure_boundary_data",
                return_value=boundary_ds,
            ),
            patch.object(
                _pw.MPASPrecipitationPlotter,
                "_extract_full_grid",
                return_value=(lon, lat),
            ),
            patch.object(
                _pw.MPASPrecipitationPlotter, "_get_or_build_remapper"
            ) as mock_build,
        ):

            _pw._prebuild_remapper_mpi(
                mock_proc, str(tmp_path), -120.0, -80.0, 30.0, 50.0, 0.5, None
            )

        mock_build.assert_called_once()


class TestAutoBatchProcessor:
    """Tests for the auto_batch_processor function."""

    def test_explicit_true_returns_true(self: "TestAutoBatchProcessor") -> None:
        """
        This test verifies that the `auto_batch_processor` function returns `True` when explicitly passed `True`, regardless of the MPI environment. It asserts that the function behaves as expected when the user explicitly requests automatic batch processing, confirming that the function respects explicit user input for enabling batch processing.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.parallel_wrappers import auto_batch_processor

        assert auto_batch_processor(True) is True

    def test_explicit_false_returns_false(self: "TestAutoBatchProcessor") -> None:
        """
        This test verifies that the `auto_batch_processor` function returns `False` when explicitly passed `False`, regardless of the MPI environment. It asserts that the function behaves as expected when the user explicitly requests to disable automatic batch processing, confirming that the function respects explicit user input for disabling batch processing.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.parallel_wrappers import auto_batch_processor

        assert auto_batch_processor(False) is False

    def test_no_mpi_returns_false(self: "TestAutoBatchProcessor") -> None:
        """
        This test verifies that the `auto_batch_processor` function returns `False` when mpi4py is not importable. It asserts that the function behaves as expected in the absence of the MPI environment, confirming that the function correctly handles the scenario where MPI is unavailable.

        Parameters:
            None

        Returns:
            None
        """
        import sys
        from mpasdiag.processing.parallel_wrappers import auto_batch_processor

        with patch.dict(sys.modules, {"mpi4py": None, "mpi4py.MPI": None}):
            result = auto_batch_processor(None)
        assert result is False

    def test_mpi_single_rank_returns_false(self: "TestAutoBatchProcessor") -> None:
        """
        This test verifies that the `auto_batch_processor` function returns `False` when mpi4py is available but `Get_size()` returns 1, indicating a single MPI rank. It asserts that the function behaves as expected in an MPI environment with only one rank, confirming that the function correctly identifies that automatic batch processing is not beneficial in a single-rank scenario.

        Parameters:
            None

        Returns:
            None
        """
        import sys
        from mpasdiag.processing.parallel_wrappers import auto_batch_processor

        mock_comm = MagicMock()
        mock_comm.Get_size.return_value = 1

        mock_mpi4py = MagicMock()
        mock_mpi4py.MPI.COMM_WORLD = mock_comm

        with patch.dict(
            sys.modules, {"mpi4py": mock_mpi4py, "mpi4py.MPI": mock_mpi4py.MPI}
        ):
            result = auto_batch_processor(None)
        assert result is False

    def test_mpi_multi_rank_returns_true(self: "TestAutoBatchProcessor") -> None:
        """
        This test verifies that the `auto_batch_processor` function returns `True` when mpi4py is available and `Get_size()` returns a value greater than 1, indicating multiple MPI ranks. It asserts that the function behaves as expected in an MPI environment with multiple ranks, confirming that the function correctly identifies that automatic batch processing can be beneficial in a multi-rank scenario.

        Parameters:
            None

        Returns:
            None
        """
        import sys
        from mpasdiag.processing.parallel_wrappers import auto_batch_processor

        mock_comm = MagicMock()
        mock_comm.Get_size.return_value = 4

        mock_mpi4py = MagicMock()
        mock_mpi4py.MPI.COMM_WORLD = mock_comm

        with patch.dict(
            sys.modules, {"mpi4py": mock_mpi4py, "mpi4py.MPI": mock_mpi4py.MPI}
        ):
            result = auto_batch_processor(None)
        assert result is True


class TestSkewTBatchProcessor:
    """Tests for ParallelSkewTProcessor.create_batch_skewt_plots_parallel across multiprocessing and MPI modes."""

    @staticmethod
    def _mock_processor(
        grid_file: Optional[str] = "/fake/grid.nc",
        data_dir: Optional[str] = "/fake/data",
        n_times: int = 2,
    ) -> MagicMock:
        """
        This helper method creates a mock MPAS3DProcessor with specified grid_file, data_dir, and number of time steps. The dataset is mocked to have a 'Time' dimension with the given size. This allows tests to simulate different processor configurations and dataset sizes without relying on actual files or data.

        Parameters:
            grid_file (Optional[str]): Path to the grid file, or None to simulate an in-memory processor.
            data_dir (Optional[str]): Path to the data directory, or None to simulate an in-memory processor.
            n_times (int): Number of time steps in the mocked dataset.

        Returns:
            MagicMock: A mock MPAS3DProcessor with the specified attributes.
        """
        processor = MagicMock(spec=MPAS3DProcessor)
        processor.grid_file = grid_file
        processor.data_dir = data_dir
        processor.dataset = MagicMock()
        processor.dataset.sizes = {"Time": n_times}
        return processor

    def test_skewt_processor_result_aggregation(
        self: "TestSkewTBatchProcessor", tmp_path: Path
    ) -> None:
        """
        This test verifies that the Skew-T batch processor aggregates worker results and returns the created files on the master process in multiprocessing mode. It mocks the parallel manager to return one successful and one failed task result, leaves formats and time indices at their defaults, and asserts that only the successful file is returned and the printed report reflects one success out of two. This exercises the default-argument handling, the grid-reload kwargs branch, and the master result-aggregation path of the method.

        Parameters:
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw

        orig_mgr = _pw.MPASParallelManager

        assert_expected_public_methods(orig_mgr, "MPASParallelManager")

        mock_mgr = MagicMock()
        mock_mgr.backend = "multiprocessing"
        mock_mgr.is_master = True
        mock_mgr.get_statistics.return_value = None

        results = [
            TaskResult(
                task_id=0,
                success=True,
                result={
                    "files": ["skewt_0.png"],
                    "timings": {
                        "data_processing": 0.1,
                        "plotting": 0.2,
                        "saving": 0.05,
                        "total": 0.35,
                    },
                },
            ),
            TaskResult(task_id=1, success=False, error="boom"),
        ]

        mock_mgr.parallel_map.return_value = results
        _pw.MPASParallelManager = lambda *a, **kw: mock_mgr

        try:
            processor = self._mock_processor()
            captured = io.StringIO()

            with redirect_stdout(captured):
                files = ParallelSkewTProcessor.create_batch_skewt_plots_parallel(
                    processor,
                    str(tmp_path),
                    lon=103.2,
                    lat=3.8,
                    n_processes=2,
                )

            assert files == ["skewt_0.png"]
            assert "Successful: 1/2" in captured.getvalue()
        finally:
            _pw.MPASParallelManager = orig_mgr

    def test_skewt_processor_in_memory_branch(
        self: "TestSkewTBatchProcessor", tmp_path: Path
    ) -> None:
        """
        This test verifies that the Skew-T batch processor falls back to passing the live processor object when it lacks the string grid_file/data_dir paths needed for reloading. It runs in multiprocessing mode so the missing paths do not raise, mocks parallel_map to return None, and asserts the method returns None while still dispatching the work, exercising the in-memory worker-kwargs branch and the no-results return path.

        Parameters:
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw

        orig_mgr = _pw.MPASParallelManager

        mock_mgr = MagicMock()
        mock_mgr.backend = "multiprocessing"
        mock_mgr.is_master = True
        mock_mgr.parallel_map.return_value = None

        _pw.MPASParallelManager = lambda *a, **kw: mock_mgr

        try:
            processor = self._mock_processor(grid_file=None, data_dir=None)
            result = ParallelSkewTProcessor.create_batch_skewt_plots_parallel(
                processor,
                str(tmp_path),
                lon=1.0,
                lat=2.0,
                time_indices=[0],
                n_processes=2,
            )
            assert result is None
            mock_mgr.parallel_map.assert_called_once()
        finally:
            _pw.MPASParallelManager = orig_mgr

    def test_skewt_processor_mpi_raises_without_data_dir(
        self: "TestSkewTBatchProcessor", tmp_path: Path
    ) -> None:
        """
        This test verifies that the Skew-T batch processor raises an informative AttributeError in MPI mode when the processor cannot be reloaded from disk (no string data_dir), because each MPI rank must be able to reconstruct the data independently.

        Parameters:
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw

        orig_mgr = _pw.MPASParallelManager

        mock_mgr = MagicMock()
        mock_mgr.backend = "mpi"
        mock_mgr.is_master = True

        _pw.MPASParallelManager = lambda *a, **kw: mock_mgr

        try:
            processor = self._mock_processor(grid_file=None, data_dir=None)
            with pytest.raises(AttributeError, match="MPI mode requires"):
                ParallelSkewTProcessor.create_batch_skewt_plots_parallel(
                    processor,
                    str(tmp_path),
                    lon=1.0,
                    lat=2.0,
                    time_indices=[0],
                )
        finally:
            _pw.MPASParallelManager = orig_mgr

    def test_skewt_processor_mpi_seed_and_evict(
        self: "TestSkewTBatchProcessor", tmp_path: Path
    ) -> None:
        """
        This test verifies that in MPI mode the Skew-T batch processor seeds the rank processor cache with the already-loaded processor before dispatching and evicts that entry afterwards, leaving the cache in its prior state. parallel_map is mocked to return None so the no-master-results return path is also exercised.

        Parameters:
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw

        orig_mgr = _pw.MPASParallelManager

        mock_mgr = MagicMock()
        mock_mgr.backend = "mpi"
        mock_mgr.is_master = True
        mock_mgr.parallel_map.return_value = None

        _pw.MPASParallelManager = lambda *a, **kw: mock_mgr
        _pw._rank_processor_cache.clear()

        try:
            processor = self._mock_processor()
            result = ParallelSkewTProcessor.create_batch_skewt_plots_parallel(
                processor,
                str(tmp_path),
                lon=103.2,
                lat=3.8,
                time_indices=[0, 1],
            )
            assert result is None
            mock_mgr.parallel_map.assert_called_once()
        finally:
            _pw.MPASParallelManager = orig_mgr

        assert _pw._rank_processor_cache == {}


class TestSeedAndInMemoryBranches:
    """Tests for the worker-cache seeding guards and the in-memory (non grid-reload) cross-section worker branch."""

    def test_seed_worker_cache_non_string_paths(
        self: "TestSeedAndInMemoryBranches",
    ) -> None:
        """
        This test verifies that the `_seed_worker_processor_cache` function returns `None` and does not attempt to seed the cache when either `grid_file` or `data_dir` is not a string path, which are required for reloading the processor on worker ranks in MPI mode. This guards against invalid cache seeding attempts when the processor object lacks the necessary attributes for disk-based reloading.

        Parameters:
            None

        Returns:
            None
        """
        processor = MagicMock()
        assert (
            _seed_worker_processor_cache(
                "2d", {"grid_file": None, "data_dir": "/d"}, processor
            )
            is None
        )
        assert _seed_worker_processor_cache("2d", {"data_dir": "/d"}, processor) is None

    def test_seed_worker_cache_missing_dataset(
        self: "TestSeedAndInMemoryBranches",
    ) -> None:
        """
        This test verifies that _seed_worker_processor_cache declines to seed (returns None) when the processor lacks a dataset attribute, which is required for seeding the cache. This guards against attempting to seed with an invalid processor object.

        Parameters:
            None

        Returns:
            None
        """
        processor = MagicMock()
        processor.dataset = None
        result = _seed_worker_processor_cache(
            "2d", {"grid_file": "/g", "data_dir": "/d"}, processor
        )
        assert result is None

    def test_cross_section_worker_in_memory_processor(
        self: "TestSeedAndInMemoryBranches", tmp_path: Path
    ) -> None:
        """
        This test verifies that the cross-section worker uses a processor passed directly in kwargs (the in-memory branch) when no grid_file/data_dir are supplied, rather than reloading from disk. It mocks the plotter and asserts the worker returns a result dictionary with a 'files' key.

        Parameters:
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None
        """
        mock_proc = MagicMock()

        mock_proc.dataset = xr.Dataset(
            {
                "theta": (["Time", "nVertLevels", "nCells"], np.zeros((2, 5, 10))),
                "Time": (
                    ["Time"],
                    pd.date_range("2025-01-01", periods=2, freq="h").values,
                ),
            }
        )

        kwargs = {
            "processor": mock_proc,
            "output_dir": str(tmp_path),
            "start_lat": 30,
            "start_lon": -110,
            "end_lat": 40,
            "end_lon": -100,
            "var_name": "theta",
            "file_prefix": "cross",
            "formats": ["png"],
            "colormap": None,
            "levels": None,
            "vertical_coord": "pressure",
            "num_points": 50,
        }

        with patch(
            "mpasdiag.processing.parallel_wrappers.MPASVerticalCrossSectionPlotter"
        ) as mock_cls:
            mock_plotter = MagicMock()
            mock_plotter.create_vertical_cross_section.return_value = (
                MagicMock(),
                MagicMock(),
            )
            mock_cls.return_value = mock_plotter
            result = _cross_section_worker((0, kwargs))

        assert isinstance(result, dict)
        assert "files" in result


class TestProcessorBranchCoverage:
    """Targeted MPI-mode tests covering default-style construction and seed/evict in the wind and cross-section batch processors."""

    def test_wind_processor_mpi_default_style_and_evict(
        self: "TestProcessorBranchCoverage", tmp_path: Path
    ) -> None:
        """
        This test verifies that the ParallelWindProcessor correctly seeds the worker cache with the provided processor in MPI mode, dispatches the work, and then evicts the cache entry afterwards. It mocks the parallel manager to simulate MPI mode and asserts that the worker function is called and the cache is cleared after execution, confirming that the seeding and eviction logic for MPI mode is functioning as intended in the wind processor workflow.

        Parameters:
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw

        orig_mgr = _pw.MPASParallelManager

        mock_mgr = MagicMock()
        mock_mgr.backend = "mpi"
        mock_mgr.is_master = True
        mock_mgr.parallel_map.return_value = None

        _pw.MPASParallelManager = lambda *a, **kw: mock_mgr
        _pw._rank_processor_cache.clear()

        try:
            processor = MagicMock()
            processor.grid_file = "/fake/grid.nc"
            processor.data_dir = "/fake/data"
            processor.dataset = MagicMock()
            processor.dataset.sizes = {"Time": 3}

            result = ParallelWindProcessor.create_batch_wind_plots_parallel(
                processor,
                str(tmp_path),
                GeographicBounds(-120, -80, 30, 50),
                u_variable="u10",
                v_variable="v10",
                time_indices=[0, 1],
                n_processes=2,
            )

            assert result is None
        finally:
            _pw.MPASParallelManager = orig_mgr

        assert _pw._rank_processor_cache == {}

    def test_cross_section_processor_mpi_seed_evict(
        self: "TestProcessorBranchCoverage", tmp_path: Path
    ) -> None:
        """
        This test verifies that the cross-section batch processor seeds the rank processor cache with the provided processor when grid_file/data_dir are present, and evicts that cache entry after dispatch, in MPI mode. parallel_map is mocked to return None so no real plotting occurs, and the test asserts that the method returns None and leaves the cache empty.

        Parameters:
            tmp_path (Path): Temporary directory for output files.

        Returns:
            None
        """
        import mpasdiag.processing.parallel_wrappers as _pw

        orig_mgr = _pw.MPASParallelManager

        mock_mgr = MagicMock()
        mock_mgr.backend = "mpi"
        mock_mgr.is_master = True
        mock_mgr.parallel_map.return_value = None

        _pw.MPASParallelManager = lambda *a, **kw: mock_mgr
        _pw._rank_processor_cache.clear()

        try:
            processor = MagicMock(spec=MPAS3DProcessor)
            processor.grid_file = "/fake/grid.nc"
            processor.data_dir = "/fake/data"
            processor.dataset = MagicMock()
            processor.dataset.sizes = {"Time": 3}

            result = (
                ParallelCrossSectionProcessor.create_batch_cross_section_plots_parallel(
                    processor,
                    var_name="theta",
                    start_point=(-110, 30),
                    end_point=(-100, 40),
                    output_dir=str(tmp_path),
                    time_indices=[0, 1],
                    n_processes=2,
                )
            )

            assert result is None
        finally:
            _pw.MPASParallelManager = orig_mgr

        assert _pw._rank_processor_cache == {}

    def test_build_precipitation_worker_kwargs_defaults(
        self: "TestProcessorBranchCoverage",
    ) -> None:
        """
        This test verifies that _build_precipitation_worker_kwargs constructs default PrecipitationMapStyle and RemapConfig objects when none are supplied, exercising the default-argument branches of the helper, and returns grid-reload kwargs for a disk-backed processor.

        Returns:
            None
        """
        processor = MagicMock()
        processor.grid_file = "/fake/grid.nc"
        processor.data_dir = "/fake/data"

        kwargs = ParallelPrecipitationProcessor._build_precipitation_worker_kwargs(
            processor,
            True,
            "/out",
            GeographicBounds(-120, -80, 30, 50),
            "rainnc",
            "a01h",
            "scatter",
            None,
            ["png"],
            style=None,
            remap_config=None,
        )

        assert "grid_file" in kwargs

    def test_build_wind_worker_kwargs_defaults(
        self: "TestProcessorBranchCoverage",
    ) -> None:
        """
        This test verifies that _build_wind_worker_kwargs constructs a default WindBatchStyle and RemapConfig when none are supplied, exercising the default-argument branches of the helper, and returns grid-reload kwargs for a disk-backed processor.

        Returns:
            None
        """
        processor = MagicMock()
        processor.grid_file = "/fake/grid.nc"
        processor.data_dir = "/fake/data"

        kwargs = ParallelWindProcessor._build_wind_worker_kwargs(
            processor,
            True,
            "/out",
            GeographicBounds(-120, -80, 30, 50),
            "u10",
            "v10",
            ["png"],
            style=None,
            remap_config=None,
        )

        assert "grid_file" in kwargs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
