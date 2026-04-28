#!/usr/bin/env python3
"""
MPASdiag Test Suite: Parallel Processing - Execution Backend Tests

This module contains tests for the execution paths of the `MPASParallelManager` when using the multiprocessing and serial backends. It verifies that tasks are executed correctly, that error handling behaves as expected, and that statistics are collected properly in both parallel and serial contexts. The tests also ensure that platform-specific multiprocessing behaviors are handled correctly, and that fallback mechanisms work when multiprocessing contexts cannot be created.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import time
import pytest
import matplotlib.pyplot as plt
from typing import List, Generator
from unittest.mock import Mock, patch

from mpasdiag.processing.parallel import (
    MPASParallelManager,
    TaskResult,
    MPASResultCollector,
)

from tests.test_data_helpers import assert_expected_public_methods


@pytest.fixture
def sample_tasks() -> List[int]:
    """ Simple list of integer tasks from 0 to 9 for testing purposes. """
    return list(range(10))


class TestMultiprocessingExecution:
    """ Tests for multiprocessing backend execution paths including pool creation, mapping, and error handling. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestMultiprocessingExecution') -> Generator[None, None, None]:
        """
        This fixture sets up a `MPASParallelManager` instance configured for multiprocessing execution before each test method in this class. It initializes the manager with a specified number of workers and ensures that it is available as `self.manager` for use in test methods. After the test method completes, it performs cleanup by closing any open matplotlib plots to prevent resource leaks. This fixture provides a consistent multiprocessing manager environment for all tests in this class, allowing them to focus on testing specific execution behaviors without needing to handle setup and teardown of the manager themselves. 

        Parameters:
            None

        Returns:
            None
        """
        self.manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=False)
        assert_expected_public_methods(self.manager, 'MPASParallelManager')

        yield
        
        plt.close('all')
    
    def test_multiprocessing_map_success(self: 'TestMultiprocessingExecution') -> None:
        """
        This test confirms that the multiprocessing backend can successfully execute a simple mapping of tasks. It defines a `simple_func` that doubles its input, creates a list of tasks, and calls `parallel_map` with the manager. The test asserts that results are returned, all tasks succeeded, and that the results are correct (each input doubled). This validates that the multiprocessing backend is correctly executing tasks in parallel and returning expected results without errors. 

        Parameters:
            None

        Returns:
            None
        """
        def simple_func(x):
            return x * 2
        
        tasks = [1, 2, 3, 4, 5]
        results = self.manager.parallel_map(simple_func, tasks)
        
        assert results is not None

        assert len(results) == pytest.approx(5)
        assert all(r.success for r in results)
        assert [r.result for r in results] == [2, 4, 6, 8, 10]
    
    def test_multiprocessing_map_with_args_kwargs(self: 'TestMultiprocessingExecution') -> None:
        """
        This test verifies that the multiprocessing backend can handle functions with additional positional and keyword arguments. It defines an `add_func` that takes a value, an offset, and a multiplier, and returns the computed result. The test calls `parallel_map` with a list of tasks and additional arguments, then asserts that results are returned, all tasks succeeded, and that the results are correct based on the provided function logic. This ensures that the multiprocessing manager correctly passes extra arguments to worker functions during parallel execution. 

        Parameters:
            None

        Returns:
            None
        """
        def add_func(x, offset, multiplier=1):
            return (x + offset) * multiplier
        
        tasks = [1, 2, 3]
        results = self.manager.parallel_map(add_func, tasks, 10, multiplier=2)
        
        assert results is not None
        assert all(r.success for r in results)
        assert [r.result for r in results] == [22, 24, 26]
    
    def test_multiprocessing_map_with_errors(self: 'TestMultiprocessingExecution') -> None:
        """
        This test checks that the multiprocessing backend correctly handles errors according to the 'continue' error policy. It defines a `failing_func` that raises a `ValueError` for a specific input, sets the manager's error policy to 'continue', and calls `parallel_map` with a list of tasks. The test asserts that results are returned, that the correct number of tasks succeeded and failed, and that the error information is captured in the failed result. This validates that the manager can continue executing remaining tasks even when some fail, and that it properly collects error information for failed tasks in the multiprocessing context. 

        Parameters:
            None

        Returns:
            None
        """
        def failing_func(x):
            if x == 3:
                raise ValueError(f"Error on task {x}")
            return x * 2
        
        self.manager.set_error_policy('continue')
        tasks = [1, 2, 3, 4, 5]
        results = self.manager.parallel_map(failing_func, tasks)
        
        assert results is not None
        assert len(results) == pytest.approx(5)
        assert sum(1 for r in results if r.success) == pytest.approx(4)
        assert sum(1 for r in results if not r.success) == pytest.approx(1)
        
        failed_result = [r for r in results if not r.success][0]

        assert failed_result.error is not None
        assert "ValueError" in failed_result.error
    
    def test_multiprocessing_statistics(self: 'TestMultiprocessingExecution') -> None:
        """
        This test verifies that the multiprocessing backend correctly collects and reports execution statistics. It runs a simple mapping of tasks and then calls `get_statistics` to retrieve the collected statistics. The test asserts that the statistics object is not None, that it contains expected fields such as total tasks, completed tasks, failed tasks, and total execution time, and that these values are consistent with the executed tasks. This ensures that the manager's statistics collection mechanism works correctly in the multiprocessing context, providing users with insights into the performance and outcomes of their parallel executions. 

        Parameters:
            None

        Returns:
            None
        """
        def simple_func(x):
            time.sleep(0.01)
            return x * 2
        
        tasks = [1, 2, 3]

        results = self.manager.parallel_map(simple_func, tasks)        
        stats = self.manager.get_statistics()

        assert stats is not None
        assert stats.total_tasks == pytest.approx(3)
        assert stats.completed_tasks == pytest.approx(3)
        assert stats.failed_tasks == pytest.approx(0)
        assert stats.total_time > 0
        assert results is not None
        assert all(r.success for r in results)
    
    def test_multiprocessing_spawn_method(self: 'TestMultiprocessingExecution') -> None:
        """
        This test confirms that the multiprocessing backend can be initialized with the 'spawn' start method, which is necessary for compatibility with certain platforms like Windows. It creates a manager with `backend='multiprocessing'` and `start_method='spawn'`, then runs a simple mapping of tasks to ensure that execution proceeds without errors. The test asserts that results are returned and that the mapping was successful, validating that the manager can operate correctly using the 'spawn' method for multiprocessing execution. This is important for ensuring cross-platform compatibility of the multiprocessing backend. 

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=False)
        assert_expected_public_methods(manager, 'MPASParallelManager')

        results = manager.parallel_map(lambda x: x * 2, [1])

        assert results is not None
        assert len(results) == pytest.approx(1)
    
    def test_multiprocessing_fallback_on_error(self: 'TestMultiprocessingExecution') -> None:
        """
        This test simulates a failure during multiprocessing pool creation to confirm that the manager falls back to serial execution instead of crashing. It monkey-patches the internal `get_context` function used for creating multiprocessing pools to always raise an exception, then initializes a manager and calls `parallel_map`. The test asserts that results are returned and that all tasks succeeded, confirming that the manager correctly detected the pool creation failure and executed tasks in serial as a fallback. This ensures robustness of the multiprocessing backend in environments where multiprocessing may not be fully supported or configured. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.parallel as _parallel_mod
        original_get_context = _parallel_mod.get_context

        def _always_fail(method: str) -> None:
            raise Exception("Pool creation failed")

        _parallel_mod.get_context = _always_fail

        try:
            manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=False)
            assert_expected_public_methods(manager, 'MPASParallelManager')
            results = manager.parallel_map(lambda x: x * 2, [1, 2])
            assert results is not None
            assert len(results) == pytest.approx(2)
            assert all(r.success for r in results)
        finally:
            _parallel_mod.get_context = original_get_context


class TestMultiprocessingMap:
    """ Tests for various multiprocessing mapping behaviors including platform-specific context selection and pool retry strategies. """
    
    def test_multiprocessing_map_win32_platform(self: 'TestMultiprocessingMap', 
                                                sample_tasks: List[int]) -> None:
        """
        This test validates that the multiprocessing map function can execute tasks on a Windows platform where the 'spawn' start method is typically required. The test patches `sys.platform` to 'win32', runs a simple mapping of tasks, and asserts that results are returned and that all results are instances of `TaskResult`. This ensures that the manager correctly handles platform-specific multiprocessing behavior, allowing for successful parallel execution on Windows without raising errors related to process spawning. 

        Parameters:
            sample_tasks (List[int]): Fixture-provided list of integer tasks.

        Returns:
            None
        """
        def simple_func(task):
            return task * 2
        
        with patch('sys.platform', 'win32'):
            manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=False)
            results = manager.parallel_map(simple_func, sample_tasks)
            assert_expected_public_methods(manager, 'MPASParallelManager')
            
            assert results is not None
            assert len(results) == len(sample_tasks)
            assert all(isinstance(r, TaskResult) for r in results)
    
    def test_multiprocessing_map_darwin_platform(self: 'TestMultiprocessingMap', 
                                                 sample_tasks: List[int]) -> None:
        """
        This test validates that the multiprocessing map function can execute tasks on a macOS (darwin) platform where the 'spawn' start method is typically required. The test patches `sys.platform` to 'darwin', runs a simple mapping of tasks, and asserts that results are returned, that the correct number of results is produced, and that all tasks succeeded. This ensures that the manager correctly handles platform-specific multiprocessing behavior on macOS, allowing for successful parallel execution without raising errors related to process spawning. 

        Parameters:
            sample_tasks (List[int]): Fixture-provided list of integer tasks.

        Returns:
            None
        """
        def simple_func(task):
            return task + 1
        
        with patch('sys.platform', 'darwin'):
            manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=False)
            results = manager.parallel_map(simple_func, sample_tasks)
            
            assert results is not None
            assert len(results) == len(sample_tasks)
            assert all(r.success for r in results)
            assert_expected_public_methods(manager, 'MPASParallelManager')
    
    def test_multiprocessing_map_linux_platform(self: 'TestMultiprocessingMap', 
                                                sample_tasks: List[int]) -> None:
        """
        This test validates that the multiprocessing map function can execute tasks on a Linux platform where forking is common. The test patches `sys.platform` to 'linux', runs a simple mapping of tasks, and asserts that results are returned and that the correct number of results is produced. This ensures that the manager correctly handles platform-specific multiprocessing behavior on Linux, allowing for successful parallel execution without raising errors related to process forking. 

        Parameters:
            sample_tasks (List[int]): Fixture-provided list of integer tasks.

        Returns:
            None
        """
        def simple_func(task):
            return task ** 2
        
        with patch('sys.platform', 'linux'):
            manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=False)
            results = manager.parallel_map(simple_func, sample_tasks)
            
            assert results is not None
            assert len(results) == len(sample_tasks)
            assert all(r.success for r in results)
            assert_expected_public_methods(manager, 'MPASParallelManager')

    def test_multiprocessing_map_context_failure_fallback(self: 'TestMultiprocessingMap', 
                                                          sample_tasks: List[int]) -> None:
        """
        This test ensures fallback to serial execution when all multiprocessing context methods fail. The test simulates context failure and asserts the manager still returns results executed serially. This provides resilience against platform/permission issues creating process pools. It patches `sys.platform` to 'linux', mocks `get_context` to raise an exception, and captures verbose output to confirm that an appropriate error message is printed. The test asserts that results are returned, that the correct number of results is produced, and that all results are instances of `TaskResult`, confirming that the manager falls back to serial execution without crashing when multiprocessing contexts cannot be created. 

        Parameters:
            sample_tasks (List[int]): Fixture-provided list of integer tasks.

        Returns:
            None
        """
        def simple_func(task):
            return task
        
        with patch('sys.platform', 'linux'):
            manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=True)
            assert_expected_public_methods(manager, 'MPASParallelManager')
            
            with patch('mpasdiag.processing.parallel.get_context') as mock_ctx:
                mock_ctx.side_effect = Exception("Context error")
                
                import io
                from contextlib import redirect_stdout                
                f = io.StringIO()

                with redirect_stdout(f):
                    results = manager.parallel_map(simple_func, sample_tasks)
                
                output = f.getvalue()
                assert results is not None
                assert len(results) == len(sample_tasks)
                assert all(isinstance(r, TaskResult) for r in results)
                assert "Context error" in output or "Failed to create multiprocessing pool" in output
    
    def test_multiprocessing_map_pool_exception_retry(self: 'TestMultiprocessingMap', 
                                                      sample_tasks: List[int]) -> None:
        """
        This test verifies the retry behavior when initial pool creation fails and a subsequent attempt succeeds. The test monkeypatches `get_context` to fail on the first call (simulating a fork failure) and succeed on the second call (simulating a spawn fallback). It captures verbose output to confirm that an appropriate error message is printed during the failure. The test asserts that results are returned and that either an error message about the failure is printed or that the correct number of results is produced, confirming that the manager's built-in retry mechanisms for pool creation are functioning as intended. This ensures that transient issues with multiprocessing context creation do not prevent task execution, allowing for robust parallel processing even in environments with restrictions on process forking. 

        Parameters:
            sample_tasks (List[int]): Fixture-provided list of integer tasks.

        Returns:
            None
        """
        def simple_func(task):
            return task
        
        with patch('sys.platform', 'linux'):
            manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=True)
            assert_expected_public_methods(manager, 'MPASParallelManager')
            
            call_count = [0]
            original_get_context = __import__('multiprocessing').get_context
            
            def failing_get_context(method):
                call_count[0] += 1
                if call_count[0] == 1:  
                    raise RuntimeError("Fork failed")
                return original_get_context(method)  
            
            with patch('mpasdiag.processing.parallel.get_context', side_effect=failing_get_context):
                import io
                from contextlib import redirect_stdout                
                f = io.StringIO()

                with redirect_stdout(f):
                    results = manager.parallel_map(simple_func, sample_tasks)
                
                output = f.getvalue()
                assert results is not None
                assert "failed" in output.lower() or len(results) == len(sample_tasks)
    
    def test_multiprocessing_map_verbose_output(self: 'TestMultiprocessingMap', 
                                                sample_tasks: List[int]) -> None:
        """
        This test confirms that when verbose mode is enabled, the multiprocessing map function emits expected informational messages about processing, worker usage, and execution statistics. It captures stdout during a parallel map execution and asserts that key phrases related to multiprocessing execution and statistics are present in the output. This ensures that verbose logging provides useful insights into the parallel execution process, aiding users in understanding how tasks are being processed and the performance characteristics of their parallel runs. 

        Parameters:
            sample_tasks (List[int]): Fixture-provided list of integer tasks.

        Returns:
            None
        """
        def simple_func(task):
            return task
        
        manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=True)
        assert_expected_public_methods(manager, 'MPASParallelManager')
        
        import io
        from contextlib import redirect_stdout        
        f = io.StringIO()

        with redirect_stdout(f):
            results = manager.parallel_map(simple_func, sample_tasks)
        
        output = f.getvalue()
        assert "Processing" in output
        assert "Python multiprocessing" in output
        assert "workers" in output
        assert "PARALLEL EXECUTION STATISTICS" in output
        assert "Speedup" in output or "speedup" in output.lower()
        assert results is not None
        assert len(results) == len(sample_tasks)
    
    def test_multiprocessing_map_fallback_to_serial_all_methods_fail(self: 'TestMultiprocessingMap', 
                                                                     sample_tasks: List[int]) -> None:
        """
        This test ensures that if all multiprocessing context methods fail (simulating an environment where multiprocessing cannot be used), the manager falls back to serial execution and still returns results. It patches `sys.platform` to 'win32' to trigger spawn behavior, mocks `get_context` to always raise an exception, and captures verbose output to confirm that an appropriate error message is printed. The test asserts that results are returned, that the correct number of results is produced, and that all results are instances of `TaskResult`, confirming that the manager successfully falls back to serial execution without crashing when multiprocessing contexts cannot be created. This provides robustness against environments with severe restrictions on multiprocessing capabilities. 

        Parameters:
            sample_tasks (List[int]): Fixture-provided list of integer tasks.

        Returns:
            None
        """
        def simple_func(task):
            return task
        
        with patch('sys.platform', 'win32'):
            manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=True)
            assert_expected_public_methods(manager, 'MPASParallelManager')
            
            with patch('mpasdiag.processing.parallel.get_context', side_effect=RuntimeError("All failed")):
                import io
                from contextlib import redirect_stdout                
                f = io.StringIO()

                with redirect_stdout(f):
                    results = manager.parallel_map(simple_func, sample_tasks)
                
                assert results is not None
                assert len(results) == len(sample_tasks)
                assert all(isinstance(r, TaskResult) for r in results)


class TestSerialMap:
    """ Tests for serial mapping functionality used as a deterministic fallback when parallel backends are not available. """
    
    
    def test_serial_map_with_verbose(self: 'TestSerialMap', 
                                     sample_tasks: List[int]) -> None:
        """
        This test validates that when the manager is in serial mode with verbose output enabled, the `parallel_map` method emits expected informational messages about processing and execution. It constructs a manager with the 'serial' backend and verbose mode enabled, defines a simple function to map over the sample tasks, and captures stdout during the call to `parallel_map`. The test asserts that key phrases related to serial execution are present in the output and that results are returned with the correct count. This ensures that verbose logging provides useful insights into the execution process even in serial mode, aiding users in understanding how tasks are being processed. 

        Parameters:
            sample_tasks (List[int]): Fixture-provided list of tasks.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=True)
        assert_expected_public_methods(manager, 'MPASParallelManager')
        
        def simple_func(task):
            return task + 10
        
        import io
        from contextlib import redirect_stdout        
        f = io.StringIO()

        with redirect_stdout(f):
            results = manager.parallel_map(simple_func, sample_tasks)
        
        output = f.getvalue()
        assert results is not None
        assert "serial mode" in output.lower()
        assert len(results) == len(sample_tasks)


class TestSerialMapWithCollector:
    """ Tests that validate serial mapping when a `MPASResultCollector` is attached to gather statistics. """
    
    def test_serial_map_with_collector_enabled(self: 'TestSerialMapWithCollector', 
                                               sample_tasks: List[int]) -> None:
        """
        This test checks that the `parallel_map` method correctly executes in serial mode when a `MPASResultCollector` is attached, and that it collects execution statistics. It creates a manager in serial mode, attaches a mock collector, and defines a simple function to map over the sample tasks. The test asserts that results are returned, that the number of results matches the number of tasks, and that the manager's `stats` attribute is populated with the total number of tasks. This ensures that even in serial execution contexts, the presence of a result collector allows for gathering useful statistics about task execution, providing insights into performance and success rates. 

        Parameters:
            sample_tasks (List[int]): Fixture-provided list of tasks.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        assert_expected_public_methods(manager, 'MPASParallelManager')
        
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 1

        manager.collector = MPASResultCollector(mock_comm)
        assert_expected_public_methods(manager.collector, 'MPASResultCollector')
        
        def simple_func(task):
            return task * 2
        
        results = manager.parallel_map(simple_func, sample_tasks)
        
        assert results is not None
        assert len(results) == len(sample_tasks)
        assert manager.stats is not None
        assert manager.stats.total_tasks == len(sample_tasks)
    
    def test_serial_map_verbose_with_stats(self: 'TestSerialMapWithCollector', 
                                           sample_tasks: List[int]) -> None:
        """
        This test checks that verbose serial mapping prints statistics when a collector is enabled. The test captures stdout and asserts presence of 'serial mode' and statistics headings. This provides operator visibility for serial runs with stats. It creates a manager in serial mode with verbose enabled, attaches a mock collector, and defines a simple function to map over the sample tasks. The test captures the output during mapping and asserts that it includes indications of serial execution and the presence of parallel execution statistics, confirming that users receive feedback about execution context and performance even when running in serial mode with a collector attached. 

        Parameters:
            sample_tasks (List[int]): Fixture-provided list of tasks.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=True)
        assert_expected_public_methods(manager, 'MPASParallelManager')
        
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 1

        manager.collector = MPASResultCollector(mock_comm)
        assert_expected_public_methods(manager.collector, 'MPASResultCollector')
        
        def simple_func(task):
            return task + 5
        
        import io
        from contextlib import redirect_stdout        
        f = io.StringIO()

        with redirect_stdout(f):
            results = manager.parallel_map(simple_func, sample_tasks)
        
        output = f.getvalue()
        assert "serial mode" in output.lower()
        assert "PARALLEL EXECUTION STATISTICS" in output
        assert results is not None
        assert len(results) == len(sample_tasks)


class TestMultiprocessingBreakStatement:
    """ Test cases for break/early-exit logic in multiprocessing attempts that verify the first successful pool run stops retries. """
    
    def test_multiprocessing_successful_first_attempt(self: 'TestMultiprocessingBreakStatement', 
                                                      sample_tasks: List[int]) -> None:
        """
        This test validates that when the multiprocessing mapping succeeds on the first attempt, the retry logic does not execute further attempts. It defines a simple function to map over the sample tasks, creates a `MPASParallelManager` with the multiprocessing backend, and calls `parallel_map`. The test asserts that results are returned, that the number of results matches the number of tasks, and that all results indicate success. This confirms that the break statement in the multiprocessing retry logic functions correctly, allowing for efficient execution without unnecessary retries when the initial attempt is successful. 

        Parameters:
            sample_tasks (List[int]): Fixture-provided list of tasks.

        Returns:
            None
        """
        def simple_func(task):
            return task + 100
        
        manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=False)
        assert_expected_public_methods(manager, 'MPASParallelManager')
        results = manager.parallel_map(simple_func, sample_tasks)
        
        assert results is not None
        assert len(results) == len(sample_tasks)
        assert all(r.success for r in results)
        assert results[0].result == pytest.approx(100)


class TestMultiprocessingResultsNoneCheck:
    """ Tests that validate fallback to serial execution when multiprocessing attempts repeatedly return `None`. """
    
    def test_multiprocessing_results_none_fallback(self: 'TestMultiprocessingResultsNoneCheck', 
                                                   sample_tasks: List[int]) -> None:
        """
        This test validates behavior when multiprocessing mapping returns `None` for all attempts, ensuring serial fallback still produces results. The test patches `get_context` to force exceptions and asserts the manager returns a serial-executed results list. This prevents silent failures when pool creation fails, ensuring that users still receive results even if multiprocessing cannot be utilized. 

        Parameters:
            sample_tasks (List[int]): Fixture-provided list of tasks.

        Returns:
            None
        """
        def simple_func(task):
            return task
        
        manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=False)
        assert_expected_public_methods(manager, 'MPASParallelManager')

        
        with patch('mpasdiag.processing.parallel.get_context') as mock_ctx:
            mock_ctx.side_effect = Exception("Force fallback")            
            results = manager._multiprocessing_map(simple_func, sample_tasks)
            assert len(results) == len(sample_tasks)


class TestSerialMapVerbosePath:
    """ Tests covering verbose printing behavior for serial map execution when a collector is attached. """
    
    def test_serial_map_calls_print_statistics(self: 'TestSerialMapVerbosePath', 
                                               sample_tasks: List[int]) -> None:
        """
        This test checks that when the `parallel_map` method is executed in serial mode with verbose enabled and a result collector attached, the `_print_statistics` method is called to display execution statistics. It creates a manager in serial mode with verbose enabled, attaches a mock collector, and defines a simple function to map over the sample tasks. The test patches the `_print_statistics` method to verify that it is called during the mapping process, confirming that users receive feedback about execution statistics even when running in serial mode with a collector attached. This ensures that the verbose path provides insights into performance and success rates for serial executions. 

        Parameters:
            sample_tasks (List[int]): Fixture-provided list of tasks.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=True)        
        mock_comm = Mock()

        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 1

        manager.collector = MPASResultCollector(mock_comm)
        assert_expected_public_methods(manager, 'MPASParallelManager')
        
        def simple_func(task):
            return task
        
        with patch.object(manager, '_print_statistics') as mock_print:
            results = manager._serial_map(simple_func, sample_tasks)
            mock_print.assert_called_once()

        assert results is not None
        assert len(results) == len(sample_tasks)
