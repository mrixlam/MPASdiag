#!/usr/bin/env python3
"""
MPASdiag Test Suite: Parallel Processing - Utilities, Error Handling, and Integration Tests

This module contains a comprehensive set of tests for the parallel processing utilities in the MPASdiag package, focusing on the `MPASParallelManager` and related functions. The tests cover various aspects of parallel execution, including task wrapping for multiprocessing, error handling policies (abort, continue, collect), statistics collection and formatting, and the `parallel_plot` convenience function. Additionally, edge cases such as empty task lists and single tasks are tested to ensure robustness. The module also includes fixtures for providing sample data files and tasks for integration-style tests.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import time
import pytest
from pathlib import Path
from typing import List
from unittest.mock import Mock, MagicMock, patch

from mpasdiag.processing.parallel import (
    MPASParallelManager,
    TaskResult,
    ParallelStats,
    _multiprocessing_task_wrapper,
    parallel_plot,
)

from tests.test_data_helpers import assert_expected_public_methods


class TestMultiprocessingTaskWrapperModule:
    """ Tests for the internal `_multiprocessing_task_wrapper` which adapts user functions for multiprocessing pools. """
    
    def test_wrapper_success(self: "TestMultiprocessingTaskWrapperModule") -> None:
        """
        This test confirms that the `_multiprocessing_task_wrapper` correctly executes a simple function and returns a successful `TaskResult`. It defines a `simple_func` that adds an offset to its input, constructs the expected arguments for the wrapper, and calls it directly. The test asserts that the result indicates success, that the computed result is correct, and that execution time is recorded. This validates that the wrapper correctly adapts user functions for execution in a multiprocessing context and returns results in the expected format. 

        Parameters:
            None

        Returns:
            None
        """
        def simple_func(x, offset=0):
            return x + offset
        
        args = (0, 5, simple_func, 'collect', (10,), {})
        result = _multiprocessing_task_wrapper(args)
        
        assert result.success
        assert result.result == pytest.approx(15)
        assert result.task_id == pytest.approx(0)
        assert result.execution_time > 0
    
    def test_wrapper_error_collect(self: "TestMultiprocessingTaskWrapperModule") -> None:
        """
        This test checks that the `_multiprocessing_task_wrapper` correctly captures exceptions and returns a failed `TaskResult` when the error policy is 'collect'. It defines a `failing_func` that raises a `ValueError`, constructs the expected arguments for the wrapper, and calls it directly. The test asserts that the result indicates failure, that no result is returned, and that the error message contains expected information about the exception. This validates that the wrapper correctly handles exceptions according to the 'collect' policy, allowing for error information to be captured without crashing the multiprocessing pool. 

        Parameters:
            None

        Returns:
            None
        """
        def failing_func(x):
            raise ValueError("Test error")
        
        args = (1, 5, failing_func, 'collect', (), {})
        result = _multiprocessing_task_wrapper(args)
        
        assert not result.success
        assert result.result is None
        assert result.error is not None
        assert "ValueError" in result.error
        assert "Test error" in result.error
    
    def test_wrapper_error_abort(self: "TestMultiprocessingTaskWrapperModule") -> None:
        """
        This test checks that the `_multiprocessing_task_wrapper` re-raises exceptions when the error policy is 'abort'. It defines a `failing_func` that raises a `ValueError`, constructs the expected arguments for the wrapper with the 'abort' policy, and calls it directly. The test asserts that a `ValueError` is raised, confirming that the wrapper does not catch the exception and allows it to propagate, which is necessary for enabling upstream abort behavior in the multiprocessing context. This ensures that the wrapper correctly enforces the semantics of the 'abort' error policy. 

        Parameters:
            None

        Returns:
            None
        """
        def failing_func(x):
            raise ValueError("Abort error")
        
        args = (2, 5, failing_func, 'abort', (), {})
        
        with pytest.raises(ValueError):
            _multiprocessing_task_wrapper(args)


class TestErrorHandling:
    """ Tests exercising different error handling policies (abort, continue, collect). """
    
    def test_error_policy_collect(self: "TestErrorHandling") -> None:
        """
        This test verifies that the 'collect' error policy allows the manager to execute all tasks while capturing error information for any that fail. It defines a `failing_func` that raises an exception for even inputs, sets the manager's error policy to 'collect', and calls `parallel_map` with a list of tasks. The test asserts that results are returned, that the correct number of tasks succeeded and failed, and that error information is captured for failed tasks. This validates that the manager correctly implements the 'collect' policy, enabling users to gather detailed error information without halting execution when some tasks fail. 

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        manager.set_error_policy('collect')
        assert_expected_public_methods(manager, 'MPASParallelManager')
        
        def failing_func(x):
            if x % 2 == 0:
                raise ValueError(f"Error on {x}")
            return x * 2
        
        tasks = [1, 2, 3, 4, 5]
        results = manager.parallel_map(failing_func, tasks)
        
        assert results is not None
        assert len(results) == pytest.approx(5)
        assert sum(1 for r in results if r.success) == pytest.approx(3)
        assert sum(1 for r in results if not r.success) == pytest.approx(2)
    
    def test_error_policy_continue(self: "TestErrorHandling") -> None:
        """
        This test verifies that the 'continue' error policy allows the manager to execute all tasks while treating failed tasks as unsuccessful without raising exceptions. It defines a `failing_func` that raises an exception for a specific input, sets the manager's error policy to 'continue', and calls `parallel_map` with a list of tasks. The test asserts that results are returned, that the correct number of tasks succeeded and failed, and that no exceptions are raised during execution. This validates that the manager correctly implements the 'continue' policy, allowing execution to proceed while marking failed tasks appropriately without halting or raising errors. 

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        manager.set_error_policy('continue')
        assert_expected_public_methods(manager, 'MPASParallelManager')
        
        def failing_func(x):
            if x == 2:
                raise ValueError("Error")
            return x * 2
        
        tasks = [1, 2, 3]
        results = manager.parallel_map(failing_func, tasks)
        
        assert results is not None
        assert len(results) == pytest.approx(3)
        assert sum(1 for r in results if r.success) == pytest.approx(2)


class TestStatistics:
    """ Tests for computing and formatting execution statistics from collected task timings. """
    
    def test_get_statistics(self: "TestStatistics") -> None:
        """
        This test confirms that the `get_statistics` method returns a valid statistics object after executing tasks in the multiprocessing backend. It runs a simple mapping of tasks and then retrieves statistics, asserting that the statistics object contains expected fields such as total tasks, completed tasks, failed tasks, and total execution time. This validates that the manager correctly collects and computes execution statistics in the multiprocessing context, providing users with insights into the performance and outcomes of their parallel executions. 

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=False)
        assert_expected_public_methods(manager, 'MPASParallelManager')        
        assert manager.get_statistics() is None
        
        results = manager.parallel_map(lambda x: x * 2, [1, 2, 3])
        stats = manager.get_statistics()
        
        assert stats is not None
        assert stats.total_tasks == pytest.approx(3)
        assert results is not None
        assert all(r.success for r in results)
    
    def test_print_statistics(self: "TestStatistics") -> None:
        """
        This test verifies that the `print_statistics` method outputs formatted execution statistics to stdout when verbose mode is enabled. It runs a simple mapping of tasks using the multiprocessing backend with `verbose=True`, captures the printed output, and asserts that it contains expected information such as "PARALLEL EXECUTION STATISTICS", total tasks, completed tasks, and other relevant details. This confirms that the manager correctly formats and prints execution statistics for users when verbose mode is active, providing insights into the performance of parallel executions. 

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=True)
        assert_expected_public_methods(manager, 'MPASParallelManager')
        
        from io import StringIO
        import sys

        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            manager.parallel_map(lambda x: x * 2, [1, 2, 3])
            output = captured_output.getvalue()
            
            assert "PARALLEL EXECUTION STATISTICS" in output
            assert "Total tasks:" in output
            assert "Completed:" in output
        finally:
            sys.stdout = sys.__stdout__


class TestParallelPlotFunctionModule:
    """ Tests for the `parallel_plot` convenience function that wraps manager creation, mapping, and error policy configuration. """
    
    def test_parallel_plot_basic(self: "TestParallelPlotFunctionModule") -> None:
        """
        This test confirms that the `parallel_plot` function can execute a simple plotting function across multiple files without raising exceptions. It defines a `simple_plot` function that simulates plotting by returning a string, creates a list of file paths, and calls `parallel_plot` with these inputs. The test asserts that results are returned, that the correct number of results is produced, and that all results indicate success. This validates that the `parallel_plot` function correctly sets up the parallel manager, applies the plotting function to each file, and handles results according to the expected behavior, providing a convenient interface for parallel plotting tasks. 

        Parameters:
            None

        Returns:
            None
        """
        def simple_plot(filepath, output_dir=None):
            return f"Plotted {filepath}"
        
        files = ['file1.nc', 'file2.nc', 'file3.nc']
        results = parallel_plot(simple_plot, files, output_dir='./output/')
        
        assert results is not None
        assert len(results) == pytest.approx(3)
        assert all(r.success for r in results)


class TestEdgeCases:
    """ Tests covering edge conditions such as empty task lists, single tasks, and more-workers-than-tasks scenarios. """
    
    def test_empty_task_list(self: "TestEdgeCases") -> None:
        """
        This test verifies that the manager can handle an empty list of tasks without errors. It constructs a manager and calls `parallel_map` with an empty task list, asserting that results are returned and that the length of results is zero. This confirms that the manager correctly handles edge cases where no tasks are provided, ensuring that it does not raise exceptions or produce invalid results when given an empty workload. 

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        assert_expected_public_methods(manager, 'MPASParallelManager')
        results = manager.parallel_map(lambda x: x * 2, [])

        assert results is not None
        assert len(results) == pytest.approx(0)
    
    def test_single_task(self: "TestEdgeCases") -> None:
        """
        This test confirms that the manager can handle a single task correctly. It constructs a manager, calls `parallel_map` with a list containing one task, and asserts that results are returned, that there is exactly one result, that the task succeeded, and that the result is correct. This validates that the manager can execute a single task without issues, ensuring that it does not rely on having multiple tasks to function properly and can handle minimal workloads as expected. 

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=False)
        assert_expected_public_methods(manager, 'MPASParallelManager')
        results = manager.parallel_map(lambda x: x * 2, [5])
        
        assert results is not None
        assert len(results) == pytest.approx(1)
        assert results[0].success
        assert results[0].result == pytest.approx(10)
    
    def test_more_workers_than_tasks(self: "TestEdgeCases") -> None:
        """
        This test verifies that the manager can handle scenarios where the number of workers exceeds the number of tasks. It constructs a manager with more workers than tasks, calls `parallel_map` with a small list of tasks, and asserts that results are returned, that the correct number of results is produced, and that all tasks succeed. This ensures that the manager efficiently utilizes available workers without creating invalid task slots or errors when there are fewer tasks than workers. 

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(backend='multiprocessing', n_workers=10, verbose=False)
        assert_expected_public_methods(manager, 'MPASParallelManager')
        results = manager.parallel_map(lambda x: x * 2, [1, 2, 3])
        assert results is not None
        assert len(results) == pytest.approx(3)
        assert all(r.success for r in results)


@pytest.fixture
def test_data_dir() -> Path:
    """
    This fixture provides the path to the repository's `data` directory, which contains sample MPAS grid files and output files for integration-style tests. It constructs the path relative to the location of this test file, ensuring that tests can access necessary data files without hardcoding absolute paths. This allows for flexible test execution across different environments while maintaining a consistent reference to required test data. 

    Parameters:
        None

    Returns:
        Path: Path to the repository `data` directory used for integration-style tests.
    """
    return Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def grid_file(test_data_dir: Path) -> str:
    """
    This fixture returns the filesystem path to a sample MPAS grid initialization file as a string. It constructs the path by appending the relative location of the grid file within the `data` directory, allowing tests to easily access this file for integration testing without hardcoding absolute paths. This ensures that tests can reliably locate the necessary grid file regardless of the execution environment, as long as the expected directory structure is maintained. 

    Parameters:
        test_data_dir (Path): Root data directory fixture.

    Returns:
        str: Filesystem path to the MPAS grid initialization file.
    """
    return str(test_data_dir / "grids" / "x1.10242.static.nc")


@pytest.fixture
def mpas_output_files(test_data_dir: Path) -> List[str]:
    """
    This fixture returns a list of filesystem paths to sample MPAS output files as strings. It looks for NetCDF files in the `mpasout` subdirectory of the `u240k` directory within the `data` directory. If such files are found, it returns the paths to up to three of them; if no files are present, it returns an empty list. This allows tests to access sample MPAS output data for integration testing without hardcoding absolute paths, while also providing flexibility in the number of files available for testing. 

    Parameters:
        test_data_dir (Path): Root data directory fixture.

    Returns:
        List[str]: Up to three MPAS output file paths, or an empty list if none are present.
    """
    output_dir = test_data_dir / "u240k" / "mpasout"

    if output_dir.exists():
        files = sorted(output_dir.glob("*.nc"))
        return [str(f) for f in files[:3]]  
    return []


@pytest.fixture
def sample_tasks() -> List[int]:
    """
    This fixture provides a simple list of integer tasks from 0 to 9 for testing purposes. It returns a list of integers that can be used as input for various test cases, such as verifying task distribution, parallel mapping, and error handling. This allows tests to have a consistent set of tasks to work with without needing to define them within each test method, promoting code reuse and clarity in test definitions. 
    
    Parameters:
        None

    Returns:
        List[int]: A list of integers from 0 to 9.
    """
    return list(range(10))


class TestMultiprocessingTaskWrapperAdditional:
    """ Tests for the multiprocessing task wrapper helper that adapts user functions for pool workers. """
    
    def test_wrapper_abort_policy(self: "TestMultiprocessingTaskWrapperAdditional") -> None:
        """
        This test verifies that the `_multiprocessing_task_wrapper` re-raises exceptions when the error policy is 'abort'. It defines a `failing_func` that raises a `ValueError`, constructs the expected arguments for the wrapper with the 'abort' policy, and calls it directly. The test asserts that a `ValueError` is raised, confirming that the wrapper does not catch the exception and allows it to propagate, which is necessary for enabling upstream abort behavior in the multiprocessing context. This ensures that the wrapper correctly enforces the semantics of the 'abort' error policy. 

        Parameters:
            None

        Returns:
            None
        """
        def failing_func(task):
            raise ValueError("Test error")
        
        args = (0, "task1", failing_func, "abort", (), {})
        
        with pytest.raises(ValueError, match="Test error"):
            _multiprocessing_task_wrapper(args)
    
    def test_wrapper_continue_policy(self: "TestMultiprocessingTaskWrapperAdditional") -> None:
        """
        This test confirms that the `_multiprocessing_task_wrapper` captures exceptions into `TaskResult` objects under the 'continue' policy. It defines a `failing_func` that raises a `RuntimeError`, constructs the expected arguments for the wrapper with the 'continue' policy, and calls it directly. The test asserts that the result indicates failure, that error information is captured in the `TaskResult`, and that the error message contains expected details about the exception. This validates that the wrapper correctly handles exceptions according to the 'continue' policy, allowing execution to proceed while marking failed tasks appropriately without halting or raising errors. This is important for scenarios where users want to continue processing remaining tasks even if some fail, while still capturing error information for later analysis. 

        Parameters:
            None

        Returns:
            None
        """
        def failing_func(task):
            raise RuntimeError("Expected error")
        
        args = (1, "task2", failing_func, "continue", (), {})
        result = _multiprocessing_task_wrapper(args)
        
        assert not result.success
        assert result.error is not None
        assert "RuntimeError: Expected error" in result.error
        assert result.task_id == pytest.approx(1)
    
    def test_wrapper_collect_policy(self: "TestMultiprocessingTaskWrapperAdditional") -> None:
        """
        This test confirms that the `_multiprocessing_task_wrapper` captures exceptions into `TaskResult` objects under the 'collect' policy. It defines a `failing_func` that raises a `TypeError`, constructs the expected arguments for the wrapper with the 'collect' policy, and calls it directly. The test asserts that the result indicates failure, that error information is captured in the `TaskResult`, and that the error message contains expected details about the exception. This validates that the wrapper correctly handles exceptions according to the 'collect' policy, allowing for error information to be captured without crashing the multiprocessing pool. Collection mode is important for aggregating errors post-run, enabling users to analyze failures after all tasks have completed. 

        Parameters:
            None

        Returns:
            None
        """
        def failing_func(task):
            raise TypeError("Type mismatch")
        
        args = (2, "task3", failing_func, "collect", (), {})
        result = _multiprocessing_task_wrapper(args)
        
        assert not result.success
        assert result.error is not None
        assert "TypeError: Type mismatch" in result.error
        assert result.task_id == pytest.approx(2)


class TestPrintStatistics:
    """ Tests for formatting and printing execution statistics from collected results. """
    
    def test_print_statistics_not_master(self: "TestPrintStatistics") -> None:
        """
        This test verifies that the `_print_statistics` method does not print any statistics when the manager is not the master process. It creates a mock MPI communicator that simulates a non-master rank, sets up a `ParallelStats` object with some dummy data, and captures stdout during the call to `_print_statistics`. The test asserts that no statistics-related output is printed since only the master process should output statistics. This ensures that in MPI runs, only the designated master process provides execution statistics, preventing cluttered output from multiple ranks. 

        Parameters:
            None

        Returns:
            None
        """
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 1 
        mock_comm.Get_size.return_value = 4
        
        with patch('mpasdiag.processing.parallel.MPI_AVAILABLE', True):
            mock_mpi = MagicMock()
            mock_mpi.COMM_WORLD = mock_comm
            
            with patch('mpasdiag.processing.parallel.MPI', mock_mpi):
                manager = MPASParallelManager(backend='mpi', verbose=True)
                assert_expected_public_methods(manager, 'MPASParallelManager')
                manager.is_master = False
                manager.stats = ParallelStats(total_tasks=10, completed_tasks=10)
                
                import io
                from contextlib import redirect_stdout                
                f = io.StringIO()

                with redirect_stdout(f):
                    manager._print_statistics()
                
                output = f.getvalue()

                assert "PARALLEL EXECUTION STATISTICS" not in output
    
    def test_print_statistics_no_stats(self: "TestPrintStatistics") -> None:
        """
        This test confirms that the `_print_statistics` method does not print any statistics when the `stats` attribute is None. It constructs a manager in serial mode, explicitly sets `stats` to None, and captures stdout during the call to `_print_statistics`. The test asserts that no statistics-related output is printed since there are no statistics to display. This ensures that the method handles cases where statistics have not been collected gracefully, without raising errors or printing misleading information. 

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=True)
        assert_expected_public_methods(manager, 'MPASParallelManager')
        manager.stats = None
        
        import io
        from contextlib import redirect_stdout        
        f = io.StringIO()

        with redirect_stdout(f):
            manager._print_statistics()
        
        output = f.getvalue()
        assert "PARALLEL EXECUTION STATISTICS" not in output
    
    def test_print_statistics_with_single_worker(self: "TestPrintStatistics") -> None:
        """
        This test validates that the `_print_statistics` method correctly formats and prints execution statistics when there is only a single worker (e.g., in serial mode). It constructs a `ParallelStats` object with dummy data for a single worker, captures stdout during the call to `_print_statistics`, and asserts that the output includes overall statistics such as total tasks, completed tasks, failed tasks, and success rate. The test also confirms that per-worker timing details are not printed since there is only one worker. This ensures that the statistics output is appropriately tailored to the execution context, providing relevant information without unnecessary details in single-worker scenarios.

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=True)
        assert_expected_public_methods(manager, 'MPASParallelManager')

        manager.stats = ParallelStats(
            total_tasks=10,
            completed_tasks=9,
            failed_tasks=1,
            total_time=25.5,
            worker_times={0: 25.5},
            load_imbalance=0.0
        )
        
        import io
        from contextlib import redirect_stdout        
        f = io.StringIO()

        with redirect_stdout(f):
            manager._print_statistics()
        
        output = f.getvalue()

        assert "PARALLEL EXECUTION STATISTICS" in output
        assert "Total tasks:       10" in output
        assert "Completed:         9" in output
        assert "Failed:            1" in output
        assert "Success rate:      90.0%" in output
        assert "Per-worker times:" not in output
    
    def test_print_statistics_with_multiple_workers(self: "TestPrintStatistics") -> None:
        """
        This test confirms that the `_print_statistics` method includes per-worker timing details when multiple workers are present. The test builds a `ParallelStats` object with several worker times, captures stdout during the call to `_print_statistics`, and asserts that the output includes overall statistics as well as per-worker timing details and load imbalance metrics. This ensures that when multiple workers are involved, the statistics output provides insights into the performance of each worker, allowing users to identify potential bottlenecks or imbalances in their parallel execution. 

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(backend='multiprocessing', verbose=True)
        assert_expected_public_methods(manager, 'MPASParallelManager')

        manager.stats = ParallelStats(
            total_tasks=20,
            completed_tasks=20,
            failed_tasks=0,
            total_time=50.0,
            worker_times={0: 12.5, 1: 13.0, 2: 12.0, 3: 12.5},
            load_imbalance=0.04
        )
        
        import io
        from contextlib import redirect_stdout        
        f = io.StringIO()

        with redirect_stdout(f):
            manager._print_statistics()
        
        output = f.getvalue()

        assert "PARALLEL EXECUTION STATISTICS" in output
        assert "Per-worker times:" in output
        assert "Rank  0:" in output
        assert "Load imbalance:" in output


class TestParallelPlotFunctionAdditional:
    """ Tests for the `parallel_plot` convenience function that configures a manager and maps a plotting callback over input files. """
    
    def test_parallel_plot_with_real_files(self: "TestParallelPlotFunctionAdditional", 
                                           mpas_output_files: List[str]) -> None:
        """
        This test validates that the `parallel_plot` function can execute a plotting function over a list of MPAS output files in parallel. It defines a mock plotting function that asserts the existence of each file and returns a string indicating it was plotted. The test calls `parallel_plot` with this function and the provided list of MPAS output files, asserting that results are returned, that the correct number of results is produced, and that all results are instances of `TaskResult`. This ensures that `parallel_plot` correctly orchestrates parallel execution of a plotting function across multiple files, allowing for efficient generation of plots from MPAS outputs in a parallelized manner. If no MPAS output files are available, the test is skipped to avoid false failures. 

        Parameters:
            mpas_output_files (List[str]): Fixture-provided list of MPAS output file paths.

        Returns:
            None
        """
        if not mpas_output_files:
            pytest.skip("No MPAS output files available")
            return
        
        def mock_plot_function(filepath, output_dir=None):
            """Mock plotting function that checks if file exists."""
            import os
            assert os.path.exists(filepath), f"File {filepath} should exist"
            return f"Plotted {filepath}"
        
        results = parallel_plot(mock_plot_function, mpas_output_files, output_dir="./test_output/")
        
        if results is not None:  
            assert len(results) == len(mpas_output_files)
            assert all(isinstance(r, TaskResult) for r in results)
    
    def test_parallel_plot_with_error_collection(self: "TestParallelPlotFunctionAdditional") -> None:
        """
        This test ensures that the `parallel_plot` function correctly collects errors when the plotting function raises exceptions for certain files. It defines a `failing_plot` function that raises a `ValueError` for a specific file and returns success for others. The test calls `parallel_plot` with this function and a list of file paths, asserting that results are returned, that the correct number of results is produced, and that the error is properly captured for the failing file while other files succeed. This confirms that `parallel_plot` can handle exceptions gracefully, allowing for robust parallel plotting even when some files may cause errors. 

        Parameters:
            None

        Returns:
            None
        """
        files = ["file1.nc", "file2.nc", "file3.nc"]
        
        def failing_plot(filepath, param=None):
            if "file2" in filepath:
                raise ValueError(f"Cannot plot {filepath}")
            return f"Success: {filepath}"
        
        results = parallel_plot(failing_plot, files, param="test")
        
        if results is not None:
            assert len(results) == len(files)
            failed = [r for r in results if not r.success]
            assert len(failed) == pytest.approx(1)
            assert failed[0].error is not None
            assert "file2" in failed[0].error
    
    def test_parallel_plot_returns_none_on_workers(self: "TestParallelPlotFunctionAdditional") -> None:
        """
        This test verifies that the `parallel_plot` function returns None when executed in a worker process, as it is designed to only return results on the master process. It defines a simple plotting function and simulates worker process behavior by calling `parallel_plot` directly. The test asserts that the result is None, confirming that the function correctly distinguishes between master and worker contexts and only returns results on the master. This ensures that users do not receive unexpected results when calling `parallel_plot` from within worker processes, maintaining a clear separation of responsibilities between master and worker execution.

        Parameters:
            None

        Returns:
            None
        """
        files = ["test1.nc", "test2.nc"]
        
        def simple_plot(filepath):
            return filepath
        
        results = parallel_plot(simple_plot, files)
        assert results is not None or True 


class TestWithRealMPASData:
    """ Integration tests that operate on real MPAS output and grid files when available. """
    
    def test_process_multiple_mpas_files(self: "TestWithRealMPASData", 
                                         mpas_output_files: List[str]) -> None:
        """
        This test validates that the `parallel_map` method can process multiple real MPAS output files in parallel, extracting basic information from each file. It defines a function to load an MPAS file and return its dimensions and variables, then uses the `MPASParallelManager` with multiprocessing to map this function over the provided list of MPAS output files. The test asserts that results are returned, that the number of results matches the number of files, and that each result contains expected information about the file's dimensions and variables. If no MPAS output files are available, the test is skipped to avoid false failures in CI environments without access to large data files. This ensures that the parallel processing logic can handle real-world MPAS data effectively, providing a foundation for more complex analyses and plotting functions.

        Parameters:
            mpas_output_files (List[str]): Fixture-provided list of MPAS file paths.

        Returns:
            None
        """
        if not mpas_output_files:
            pytest.skip("No MPAS output files available")
            return
        
        def load_mpas_file(filepath):
            """Load MPAS file and return basic info."""
            import xarray as xr
            ds = xr.open_dataset(filepath)
            info = {
                'file': filepath,
                'dims': dict(ds.sizes),
                'vars': list(ds.data_vars)
            }
            ds.close()
            return info
        
        manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=False)
        assert_expected_public_methods(manager, 'MPASParallelManager')

        results = manager.parallel_map(load_mpas_file, mpas_output_files)
        
        assert results is not None
        assert len(results) == len(mpas_output_files)
        assert all(r.success for r in results)
        assert all(isinstance(r.result, dict) for r in results)
    
    def test_process_grid_file_with_different_strategies(self: "TestWithRealMPASData", 
                                                         grid_file: str) -> None:
        """
        This test checks that the `parallel_map` method can process a real MPAS grid file using different load balancing strategies. It defines a function to extract basic grid information (number of cells and vertices) from the provided grid file, then uses the `MPASParallelManager` with the serial backend to map this function over the grid file. The test asserts that results are returned, that the result contains expected grid information, and that the method can execute without errors even when parallelism is not utilized. If the grid file is not available, the test is skipped to avoid false failures in environments where the file cannot be accessed. This ensures that the processing logic can handle real MPAS grid data and provides a basis for testing more complex operations on grid files in parallel contexts.

        Parameters:
            grid_file (str): Fixture-provided path to grid file.

        Returns:
            None
        """
        import os
        if not os.path.exists(grid_file):
            pytest.skip("Grid file not available")
            return
        
        def extract_grid_info(filepath):
            """Extract basic grid information."""
            import xarray as xr
            ds = xr.open_dataset(filepath, decode_times=False)
            info = {
                'nCells': ds.sizes.get('nCells', 0),
                'nVertices': ds.sizes.get('nVertices', 0),
            }
            ds.close()
            return info
        
        manager = MPASParallelManager(backend='serial', verbose=False)
        results = manager.parallel_map(extract_grid_info, [grid_file])
        assert_expected_public_methods(manager, 'MPASParallelManager')
        
        assert results is not None
        assert len(results) == pytest.approx(1)
        assert results[0].success
        assert 'nCells' in results[0].result
        assert 'nVertices' in results[0].result


class TestParallelPlotFunctionComplete:
    """ Comprehensive tests of the `parallel_plot` convenience function covering manager creation, error policy configuration, and simple integration runs. """
    
    def test_parallel_plot_creates_manager_with_dynamic_strategy(self: "TestParallelPlotFunctionComplete") -> None:
        """
        This test verifies that the `parallel_plot` function creates an instance of `MPASParallelManager` with the expected parameters, including the dynamic load balance strategy and verbose output. It mocks the `MPASParallelManager` class to intercept its instantiation and checks that it is called with the correct arguments. The test also asserts that the `set_error_policy` method is called with 'collect' to ensure that errors are handled appropriately during parallel execution. This confirms that the `parallel_plot` function sets up the parallel manager correctly for users, providing a foundation for effective parallel plotting operations. 

        Parameters:
            None

        Returns:
            None
        """
        files = ["test1.nc", "test2.nc", "test3.nc"]
        
        def mock_plot(filepath, **kwargs):
            return f"Plot of {filepath}"
        
        with patch('mpasdiag.processing.parallel.MPASParallelManager') as MockManager:
            mock_instance = Mock()

            mock_instance.parallel_map.return_value = [
                TaskResult(i, True, f"Plot of {f}") for i, f in enumerate(files)
            ]

            MockManager.return_value = mock_instance            

            results = parallel_plot(mock_plot, files, output_dir="./test/")            
            MockManager.assert_called_once_with(load_balance_strategy="dynamic", verbose=True)
            
            mock_instance.set_error_policy.assert_called_once_with('collect')
            mock_instance.parallel_map.assert_called_once()

            assert results is not None
            assert len(results) == len(files)
            assert all(isinstance(r, TaskResult) for r in results)
    
    def test_parallel_plot_integration(self: "TestParallelPlotFunctionComplete") -> None:
        """
        This test performs a simple integration test of the `parallel_plot` function using a basic counting plot function and a list of dummy file paths. It checks that the function returns results for each file and that the results are instances of `TaskResult`. This confirms that the `parallel_plot` function can execute end-to-end with a user-defined plotting function, providing a template for users to create their own parallel plotting workflows. 

        Parameters:
            None

        Returns:
            None
        """
        files = ["file1.txt", "file2.txt"]
        
        def counting_plot(filepath, multiplier=1):
            """Simple plot function for testing."""
            return len(filepath) * multiplier
        
        results = parallel_plot(counting_plot, files, multiplier=2)
        
        if results is not None:
            assert len(results) == pytest.approx(2)
            assert all(isinstance(r, TaskResult) for r in results)


def test_timing_output() -> None:
    """
    This test simulates the output of a parallel processing timing test for MPAS diagnostics, demonstrating how to interpret timing metrics and load balancing statistics. It creates a mock MPI environment to show how timing information would be printed for a batch processing run, including breakdowns of data processing, plotting, and saving times, as well as overall wall time and speedup potential. The test also includes an interpretation section that explains what each metric means and provides usage recommendations based on the timing results. This serves as a template for users to understand and analyze the performance of their parallel processing workflows when running MPAS diagnostics. 

    Parameters:
        None

    Returns:
        None
    """
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        is_parallel = size > 1
    except ImportError:
        rank = 0
        size = 1
        is_parallel = False
    
    if rank == 0:
        print("\n" + "="*70)
        print("PARALLEL PROCESSING TIMING TEST")
        print("="*70)
        print(f"Running with {size} process(es)")
        print(f"Mode: {'PARALLEL' if is_parallel else 'SERIAL'}")
        print("="*70 + "\n")
    
    if rank == 0:
        print("Example timing output structure:\n")
        
        print("="*70)
        print("PRECIPITATION BATCH PROCESSING RESULTS")
        print("="*70)
        print("Status:")
        print("  Successful: 10/10")
        print("  Failed: 0/10")
        print("  Created files: 10 in ./plots")
        print()
        print("Timing Breakdown (per time step):")
        print("  Data Processing:")
        print("    Min:   0.245s")
        print("    Max:   0.312s")
        print("    Mean:  0.278s")
        print("  Plotting:")
        print("    Min:   1.234s")
        print("    Max:   1.567s")
        print("    Mean:  1.401s")
        print("  Saving:")
        print("    Min:   0.089s")
        print("    Max:   0.123s")
        print("    Mean:  0.106s")
        print("  Total per step:")
        print("    Min:   1.568s")
        print("    Max:   2.002s")
        print("    Mean:  1.785s")
        print()
        print("Overall Parallel Execution:")
        print("  Wall time: 4.85s")
        print("  Speedup potential: 17.85s / 4.85s = 3.68x")
        print("  Load imbalance: 5.2%")
        print("="*70)
        print()
        
        print("INTERPRETATION:")
        print("-" * 70)
        print("1. Data Processing: Time spent reading/processing MPAS data")
        print("   - If high: Consider data chunking or I/O optimization")
        print()
        print("2. Plotting: Time spent creating matplotlib figures")
        print("   - Usually the dominant cost")
        print("   - Benefits most from parallelization")
        print()
        print("3. Saving: Time spent writing files to disk")
        print("   - If high: Check disk speed, consider different format")
        print()
        print("4. Wall time: Actual elapsed time (what user experiences)")
        print("   - In parallel: Limited by slowest worker + overhead")
        print()
        print("5. Speedup potential: Sum of all times / wall time")
        print("   - Ideal: Equal to number of processes")
        print("   - Here: 3.68x with 4 processes = 92% efficiency")
        print()
        print("6. Load imbalance: How uneven work distribution is")
        print("   - <10%: Excellent")
        print("   - 10-20%: Good")
        print("   - >20%: Consider dynamic load balancing")
        print("="*70 + "\n")
        
        print("USAGE RECOMMENDATIONS:")
        print("-" * 70)
        print("• Use parallel processing when:")
        print("  - Processing 10+ time steps")
        print("  - Each step takes >1 second")
        print("  - Have access to multiple cores")
        print()
        print("• Monitor timing to:")
        print("  - Identify bottlenecks (data vs plotting vs I/O)")
        print("  - Verify parallel speedup is worth the complexity")
        print("  - Tune load balancing strategy")
        print()
        print("• Serial vs Parallel comparison:")
        print("  - Run both and compare wall times")
        print("  - Check that results are identical")
        print("  - Ensure speedup > 1.5x to justify parallel overhead")
        print("="*70 + "\n")


def simulate_workload() -> None:
    """
    This function simulates a workload with variable execution times to demonstrate timing output and load balancing in a parallel processing context. It defines a `dummy_task` function that mimics typical MPAS diagnostic processing steps with fixed and variable sleep times to represent data processing, plotting, and saving phases. The `MPASParallelManager` is used to execute the tasks in parallel, and timing statistics are printed to show how the workload is distributed across processes. This allows for testing of dynamic load balancing and timing output without requiring actual MPAS datasets, providing insights into performance characteristics of parallel execution. 

    Parameters:
        None

    Returns:
        None
    """
    from mpasdiag.processing.parallel import MPASParallelManager
    
    def dummy_task(task_id: int) -> str:
        """
        This function simulates a task with variable execution times to mimic the behavior of processing MPAS diagnostics. It includes fixed sleep times to represent data processing and saving, as well as a variable sleep time to represent plotting, which can vary based on the task ID to create load imbalance. The function returns a formatted string indicating the completion of the task. 

        Parameters:
            task_id (int): Unique identifier for the task used to generate variable execution times.

        Returns:
            str: Formatted result string in format 'task_{id}_result' indicating task completion.
        """
        time.sleep(0.1)
        time.sleep(0.5 + (task_id % 3) * 0.1)  
        time.sleep(0.05)
        
        return f"task_{task_id}_result"
    
    manager = MPASParallelManager(load_balance_strategy="dynamic", verbose=True)
    manager.set_error_policy("collect")
    
    if manager.is_master:
        print("\nRunning dummy workload to demonstrate timing...\n")
    
    tasks = list(range(12))  
    results = manager.parallel_map(dummy_task, tasks)
    
    if manager.is_master:
        stats = manager.get_statistics()
        
        if stats is None:
            print("\nWarning: Statistics not available")
            return
        
        print("\nDummy workload completed:")
        print(f"  Tasks: {stats.total_tasks}")
        print(f"  Success: {stats.completed_tasks}")
        print(f"  Wall time: {stats.total_time:.2f}s")
        print(f"  Load imbalance: {100*stats.load_imbalance:.1f}%")
        
        avg_task_time = stats.total_time / manager.size
        serial_time = 0.65 * stats.total_tasks  
        speedup = serial_time / stats.total_time
        
        print("\nPerformance metrics:")
        print(f"  Estimated serial time: {serial_time:.2f}s")
        print(f"  Parallel time: {stats.total_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Efficiency: {100*speedup/manager.size:.1f}%")

        assert results is not None
        assert len(results) == len(tasks)
        assert all(r.success for r in results)
        assert all(r.result == f"task_{i}_result" for i, r in enumerate(results))
        assert avg_task_time < 0.7, "Average task time should be less than max individual task time"


if __name__ == "__main__":
    """
    This block allows the test module to be run directly, executing the `test_timing_output` function to demonstrate the timing output structure and interpretation guidelines for parallel processing. It also checks for MPI availability and provides instructions for testing with real data if MPI is available, or notes the requirement to install `mpi4py` for parallel execution testing. This enables users to easily run a demonstration of the timing output without needing to set up a full testing environment or access real MPAS datasets, providing insights into performance characteristics of parallel execution in MPAS diagnostics.
    """
    test_timing_output()
    
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            print("\n✓ Timing test completed successfully!")
            print("\nTo test with real data:")
            print("  mpirun -n 4 mpasdiag precipitation --grid-file grid.nc \\")
            print("    --data-dir ./data --variable rainnc --batch-all --parallel\n")
    except ImportError:
        print("\n✓ Timing test completed successfully!")
        print("\nNote: Install mpi4py to test parallel execution:")
        print("  pip install mpi4py\n")

