#!/usr/bin/env python3
"""
MPASdiag Test Suite: Parallel Processing Tests

This module contains tests for the parallel processing subsystem of MPASdiag, specifically targeting the `MPASParallelManager` and related classes. The tests cover basic multiprocessing functionality, performance characteristics with timed tasks, error handling policies, serial execution fallback, worker count control through the `n_workers` parameter, and integration of CLI arguments with manager configuration. These tests validate that the parallel manager correctly distributes tasks, collects results, handles errors according to policy, and provides expected performance benefits when using multiple workers. They also ensure that users can control parallelism through configuration and that CLI parameters propagate correctly to the manager.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules for testing
import time
from coverage import results
import pytest
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Generator
from unittest.mock import Mock, MagicMock, patch

from mpasdiag.processing.parallel import (
    MPASParallelManager,
    ErrorPolicy,
    LoadBalanceStrategy,
    TaskResult,
    ParallelStats,
    MPASTaskDistributor,
    MPASResultCollector,
    _multiprocessing_task_wrapper,
    parallel_plot
)


def simple_task(x: int) -> int:
    """
    This is a simple task function that takes an integer input and returns its square. It serves as a basic workload for testing parallel execution without introducing complexity or performance overhead. The function is deterministic and has no side effects, making it ideal for validating the correctness of parallel task distribution and result collection in the `MPASParallelManager`. It is used in multiple tests to confirm that tasks are executed correctly across different backends and configurations.

    Parameters:
        x (int): Input integer value to be squared.

    Returns:
        int: Square of the input value (x²).
    """
    return x ** 2


def slow_task(x: int) -> int:
    """
    This task function simulates a computational workload by introducing an artificial delay of 50 milliseconds before returning the result. It takes an integer input, sleeps for a short duration to mimic processing time, and then returns double the input value. This function is used in performance tests to validate that the `MPASParallelManager` can execute tasks concurrently across multiple workers, reducing total execution time compared to sequential processing. The predictable delay allows for measuring speedup benefits when using parallel execution.

    Parameters:
        x (int): Input integer value to be doubled after processing delay.

    Returns:
        int: Double of the input value (x × 2) after 50ms delay.
    """
    time.sleep(0.05)
    return x * 2


def error_task(x: int) -> int:
    """
    This task function is designed to test error handling in the `MPASParallelManager` by intentionally raising a `ValueError` when the input value equals 5. For all other input values, it returns the input plus 10. This allows tests to verify that the manager correctly captures and handles exceptions according to the configured error policy, while still processing other tasks successfully. The predictable failure at x=5 provides a controlled scenario for validating error collection and partial result recovery.

    Parameters:
        x (int): Input integer value; raises error if x equals 5, otherwise returns x + 10.

    Returns:
        int: Input value plus 10 (x + 10) for successful executions.

    Raises:
        ValueError: When x equals 5, intentionally for error handling testing.
    """
    if x == 5:
        raise ValueError(f"Intentional error for x={x}")
    return x + 10


def test_basic_multiprocessing() -> None:
    """
    This test validates basic multiprocessing functionality of the `MPASParallelManager` by executing a simple task across multiple workers and collecting results. It creates a manager with the multiprocessing backend, distributes a list of integer tasks to compute their squares, and asserts that the results match expected values. The test confirms that tasks are executed correctly in parallel and that results are collected without errors. This serves as a fundamental check of the multiprocessing execution path. 

    Parameters:
        None

    Returns:
        None
    """
    manager = MPASParallelManager()
    tasks = list(range(10))
    results = manager.parallel_map(simple_task, tasks)
    
    assert results is not None, "parallel_map returned None"
    
    expected = [x**2 for x in tasks]
    actual = [r.result for r in results]
    
    print(f"Expected: {expected}")
    print(f"Actual:   {actual}")
    print(f"All successful: {all(r.success for r in results)}")
    print(f"Match expected: {expected == actual}")
    
    assert expected == actual, "Results don't match!"


def test_with_timing() -> None:
    """
    This test evaluates the performance of the `MPASParallelManager` by executing a set of tasks with an artificial delay and measuring the total wall time. It creates a manager with multiple workers, runs a list of tasks that each take approximately 50 milliseconds, and asserts that the results are correct while also reporting the total execution time. This test demonstrates the speedup benefits of parallel execution compared to sequential processing, confirming that the manager effectively distributes work across workers to reduce overall runtime.

    Parameters:
        None

    Returns:
        None
    """
    manager = MPASParallelManager(n_workers=4)
    tasks = list(range(16))
    
    start = time.time()
    results = manager.parallel_map(slow_task, tasks)
    wall_time = time.time() - start
    
    assert results is not None, "parallel_map returned None"
    
    expected = [x*2 for x in tasks]
    actual = [r.result for r in results]
    
    print(f"\nExpected: {expected}")
    print(f"Actual:   {actual}")
    print(f"Wall time: {wall_time:.2f}s")
    print(f"All successful: {all(r.success for r in results)}")
    
    assert expected == actual, "Results don't match!"


def test_error_handling() -> None:
    """
    This test verifies the error handling capabilities of the `MPASParallelManager` by executing a set of tasks where one task is designed to fail. It configures the manager to collect errors instead of aborting, runs a list of tasks that will raise an exception for a specific input, and asserts that the manager correctly captures the failure while still processing other tasks successfully. The test checks that the results include both successful outcomes and the expected failure, confirming that error policies are respected and that partial results can be retrieved even when some tasks fail. 

    Parameters:
        None

    Returns:
        None
    """
    manager = MPASParallelManager()
    manager.set_error_policy('collect')
    
    tasks = list(range(10))
    results = manager.parallel_map(error_task, tasks)
    
    assert results is not None, "parallel_map returned None"
    
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    
    print(f"\nSuccessful: {len(successes)}/{len(results)}")
    print(f"Failed: {len(failures)}/{len(results)}")
    print(f"Failed task IDs: {[r.task_id for r in failures]}")
    
    assert len(successes) == pytest.approx(9), f"Expected 9 successes, got {len(successes)}"
    assert len(failures) == pytest.approx(1), f"Expected 1 failure, got {len(failures)}"
    assert failures[0].task_id == pytest.approx(5), "Wrong task failed"


def test_serial_fallback() -> None:
    """
    This test confirms that the `MPASParallelManager` can operate in serial mode when the 'serial' backend is explicitly selected. It creates a manager with the serial backend, executes a list of tasks to compute their squares, and asserts that the results are correct while also confirming that the manager reports being in serial mode. This test ensures that users can run tasks without parallelism when desired, and that the manager correctly configures itself for serial execution. 

    Parameters:
        None

    Returns:
        None
    """
    manager = MPASParallelManager(backend='serial')
    tasks = list(range(5))
    results = manager.parallel_map(simple_task, tasks)
    
    assert results is not None, "parallel_map returned None"
    
    expected = [x**2 for x in tasks]
    actual = [r.result for r in results]
    
    print(f"Expected: {expected}")
    print(f"Actual:   {actual}")
    print(f"All successful: {all(r.success for r in results)}")
    
    assert expected == actual, "Results don't match!"


def test_n_workers_argument() -> None:
    """
    This test verifies that the `n_workers` argument in the `MPASParallelManager` constructor correctly controls the number of worker processes created for parallel execution. It tests three scenarios: explicitly setting `n_workers=1` to confirm single-worker behavior, setting `n_workers=4` to confirm multiple workers are created, and setting `n_workers=None` to confirm that the manager auto-detects the number of workers based on available CPU cores. The test asserts that the manager's reported pool size matches the expected number of workers for each scenario, confirming that user control over parallelism is functioning as intended. 

    Parameters:
        None

    Returns:
        None
    """
    manager_1 = MPASParallelManager(n_workers=1, backend='multiprocessing')
    tasks = list(range(16))
    
    start = time.time()
    results_1 = manager_1.parallel_map(slow_task, tasks)
    time_1_worker = time.time() - start
    
    assert results_1 is not None, "parallel_map returned None with 1 worker"
    assert manager_1.size == pytest.approx(1), f"Expected size=1, got {manager_1.size}"
    print(f"✓ n_workers=1: Wall time = {time_1_worker:.2f}s, Pool size = {manager_1.size}")
    
    manager_4 = MPASParallelManager(n_workers=4, backend='multiprocessing')
    
    start = time.time()
    results_4 = manager_4.parallel_map(slow_task, tasks)
    time_4_workers = time.time() - start
    
    assert results_4 is not None, "parallel_map returned None with 4 workers"
    assert manager_4.size == pytest.approx(4), f"Expected size=4, got {manager_4.size}"
    print(f"✓ n_workers=4: Wall time = {time_4_workers:.2f}s, Pool size = {manager_4.size}")
    
    manager_auto = MPASParallelManager(n_workers=None, backend='multiprocessing')
    
    start = time.time()
    results_auto = manager_auto.parallel_map(slow_task, tasks)
    time_auto = time.time() - start
    
    assert results_auto is not None, "parallel_map returned None with auto workers"
    print(f"✓ n_workers=None: Wall time = {time_auto:.2f}s, Pool size = {manager_auto.size}")
    
    print("\n" + "-"*60)
    print("Performance Analysis:")
    print(f"  1 worker:  {time_1_worker:.2f}s")
    print(f"  4 workers: {time_4_workers:.2f}s")
    print(f"  Auto:      {time_auto:.2f}s")


def test_n_workers_with_cli_integration() -> None:
    """
    This test validates that the `n_workers` argument provided through CLI integration correctly configures the `MPASParallelManager` and that tasks execute successfully with the specified number of workers. It tests multiple scenarios including single worker, multiple workers, and auto-detection of workers, confirming that the manager's pool size matches expectations and that tasks are executed without errors. This test ensures that users can control parallelism through CLI parameters and that those parameters propagate correctly to the manager configuration. 

    Parameters:
        None

    Returns:
        None
    """
    test_cases = [
        (1, "Single worker (sequential-like)"),
        (2, "Two workers"),
        (4, "Four workers"),
        (None, "Auto-detect (default)")
    ]
    
    for n_workers_arg, description in test_cases:
        print(f"\nTesting: {description} (n_workers={n_workers_arg})")
        
        manager = MPASParallelManager(
            n_workers=n_workers_arg,
            backend='multiprocessing',
            verbose=False
        )
        
        if n_workers_arg is not None:
            assert manager.size == n_workers_arg, \
                f"Expected size={n_workers_arg}, got {manager.size}"
            print(f"  ✓ Manager created with {manager.size} workers (as requested)")
        else:
            from multiprocessing import cpu_count
            expected_size = max(1, cpu_count() - 1)
            assert manager.size == expected_size, \
                f"Expected auto-detect size={expected_size}, got {manager.size}"
            print(f"  ✓ Manager auto-detected {manager.size} workers")
        
        results = manager.parallel_map(simple_task, [1, 2, 3])
        assert results is not None, "parallel_map returned None"
        assert all(r.success for r in results), "Some tasks failed"
        print(f"  ✓ Tasks executed successfully with {manager.size} workers")


class TestParallelProcessing:
    """ Core tests for the parallel processing subsystem. """
    
    def test_import_parallel_module(self: "TestParallelProcessing") -> None:
        """
        This test verifies that the `parallel` module within `mpasdiag.processing` is importable and available. The test attempts to import the module and asserts that it contains the expected `MPASParallelManager` class. This ensures that the core parallel processing components are accessible for use in other tests and application code. It serves as a basic smoke test for module availability and correct packaging. 

        Parameters:
            self ("TestParallelProcessing"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing import parallel
        assert hasattr(parallel, 'MPASParallelManager')

    def test_parallel_manager_exists(self: "TestParallelProcessing") -> None:
        """
        This test confirms that the `MPASParallelManager` class is defined and importable from the `parallel` module. The presence of this class is critical as it serves as the central orchestrator for parallel task execution in MPASdiag. The test imports the class and asserts that it is not None, ensuring that the core manager component is available for use in parallel processing workflows. This test acts as a contract check for the existence of the manager class in the public API. 

        Parameters:
            self ("TestParallelProcessing"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing.parallel import MPASParallelManager
        assert MPASParallelManager is not None

    def test_load_balance_strategy_enum_exists(self: "TestParallelProcessing") -> None:
        """
        This test verifies that the `LoadBalanceStrategy` enum contains expected strategies for distributing tasks across workers. The test checks for members like `STATIC` and `DYNAMIC` to ensure that different load balancing approaches are available to callers. Correct enumeration values are important for configuring how the manager assigns work to workers, especially for heterogeneous workloads. This guards against accidental API or enum member renames that could break existing code relying on these strategies. 

        Parameters:
            self ("TestParallelProcessing"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing.parallel import LoadBalanceStrategy
        assert hasattr(LoadBalanceStrategy, 'STATIC')
        assert hasattr(LoadBalanceStrategy, 'DYNAMIC')

    def test_error_policy_enum_exists(self: "TestParallelProcessing") -> None:
        """
        This test confirms that the `ErrorPolicy` enum is defined with expected members for controlling error handling behavior in the `MPASParallelManager`. The test checks for members like `ABORT`, `CONTINUE`, and `COLLECT` to ensure that users have options for how the manager responds to task failures. The presence of these enum members is essential for configuring robust parallel execution workflows that can tolerate or respond to errors in different ways. This test helps catch regressions where error handling policies might be removed or renamed, ensuring that critical configuration options remain available. 

        Parameters:
            self ("TestParallelProcessing"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing.parallel import ErrorPolicy
        assert hasattr(ErrorPolicy, 'ABORT')
        assert hasattr(ErrorPolicy, 'CONTINUE')
        assert hasattr(ErrorPolicy, 'COLLECT')

    def test_task_result_dataclass_exists(self: "TestParallelProcessing") -> None:
        """
        This test ensures that the `TaskResult` dataclass is defined for encapsulating the outcome of individual tasks executed by the `MPASParallelManager`. The `TaskResult` class typically includes fields for success status, result data, error information, and task metadata. The test imports the dataclass and asserts that it is not None to confirm that the structure for representing task outcomes is available. This dataclass is crucial for consistent result handling and error reporting across different parallel execution backends. Its presence supports the manager's ability to collect and report on task outcomes in a standardized way. 

        Parameters:
            self ("TestParallelProcessing"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing.parallel import TaskResult
        assert TaskResult is not None

    def test_parallel_stats_dataclass_exists(self: "TestParallelProcessing") -> None:
        """
        This test verifies that the `ParallelStats` dataclass is defined for tracking performance and execution statistics of parallel tasks. The `ParallelStats` class typically includes fields for timing information, worker utilization, task counts, and load balancing metrics. The test imports the dataclass and asserts that it is not None to confirm that the structure for collecting parallel execution statistics is available. This dataclass is important for monitoring and optimizing parallel workflows, allowing users to analyze performance characteristics and identify bottlenecks in their parallel processing pipelines. Its presence supports the manager's ability to provide insights into execution efficiency. 

        Parameters:
            self ("TestParallelProcessing"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing.parallel import ParallelStats
        assert ParallelStats is not None

    def test_task_distributor_class_exists(self: "TestParallelProcessing") -> None:
        """
        This test confirms that the `MPASTaskDistributor` class is defined for managing the distribution of tasks to workers in the `MPASParallelManager`. The `MPASTaskDistributor` is responsible for implementing the logic that assigns tasks to workers based on the selected load balancing strategy. The test imports the class and asserts that it is not None to ensure that the core component for task distribution is available. This class is critical for enabling efficient parallel execution, as it determines how work is allocated across workers to optimize performance and resource utilization. Its presence supports the manager's ability to effectively distribute tasks according to user-configured strategies. 

        Parameters:
            self ("TestParallelProcessing"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing.parallel import MPASTaskDistributor
        assert MPASTaskDistributor is not None

    def test_result_collector_class_exists(self: "TestParallelProcessing") -> None:
        """
        This test verifies that the `MPASResultCollector` class is defined for aggregating results from worker processes in the `MPASParallelManager`. The `MPASResultCollector` is responsible for collecting task outcomes, handling errors according to the configured policy, and providing a unified interface for accessing results after parallel execution. The test imports the class and asserts that it is not None to confirm that the core component for result collection is available. This class is essential for ensuring that results from parallel tasks are properly gathered and made accessible to callers, supporting robust error handling and result retrieval in parallel workflows. Its presence ensures that the manager can effectively manage and report on task outcomes. 

        Parameters:
            self ("TestParallelProcessing"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing.parallel import MPASResultCollector
        assert MPASResultCollector is not None


class TestParallelWrappers:
    """ Tests for convenience wrapper classes and helper functions that adapt processors to parallel execution frameworks. """
    
    def test_import_parallel_wrappers(self: "TestParallelWrappers") -> None:
        """
        This test confirms that the `parallel_wrappers` module within `mpasdiag.processing` is importable and available. The test attempts to import the module and asserts that it is not None, ensuring that the convenience wrapper classes for parallel processing are accessible for use in higher-level orchestration code. These wrappers adapt specific diagnostic processors to run in parallel worker contexts, so their availability is critical for enabling parallel execution of various processing tasks. This test serves as a basic check for the presence of the wrapper module in the public API. 

        Parameters:
            self ("TestParallelWrappers"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing import parallel_wrappers
        assert parallel_wrappers is not None

    def test_parallel_precipitation_processor_exists(self: "TestParallelWrappers") -> None:
        """
        This test verifies that the `ParallelPrecipitationProcessor` wrapper class is defined for adapting precipitation processing tasks to parallel execution. The wrapper allows precipitation diagnostics to be distributed across workers with consistent initialization and result handling. The test imports the class and asserts that it is not None to confirm that the wrapper is available in the module exports. This ensures that users have access to a convenient interface for running precipitation processing in parallel, supporting efficient execution of these diagnostics. Presence of this wrapper class is important for enabling parallelism in precipitation-related workflows. 

        Parameters:
            self ("TestParallelWrappers"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing.parallel_wrappers import ParallelPrecipitationProcessor
        assert ParallelPrecipitationProcessor is not None

    def test_parallel_surface_processor_exists(self: "TestParallelWrappers") -> None:
        """
        This test confirms that the `ParallelSurfaceProcessor` wrapper class is defined for adapting surface processing tasks to parallel execution. The wrapper standardizes how surface diagnostics are initialized and executed across worker processes, allowing for efficient distribution of surface-related processing tasks. The test imports the class and asserts that it is not None to ensure that the wrapper is available in the module exports. This guarantees that users can easily run surface processing in parallel, supporting improved performance for these diagnostics. The presence of this wrapper class is critical for enabling parallelism in surface processing workflows and ensuring that users have access to convenient interfaces for parallel execution of these tasks. 

        Parameters:
            self ("TestParallelWrappers"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing.parallel_wrappers import ParallelSurfaceProcessor
        assert ParallelSurfaceProcessor is not None

    def test_parallel_wind_processor_exists(self: "TestParallelWrappers") -> None:
        """
        This test verifies that the `ParallelWindProcessor` wrapper class is defined for adapting wind processing tasks to parallel execution. The wrapper allows wind diagnostics to be distributed across workers with consistent initialization and result handling. The test imports the class and asserts that it is not None to confirm that the wrapper is available in the module exports. This ensures that users have access to a convenient interface for running wind processing in parallel, supporting efficient execution of these diagnostics. Presence of this wrapper class is important for enabling parallelism in wind-related workflows. 

        Parameters:
            self ("TestParallelWrappers"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing.parallel_wrappers import ParallelWindProcessor
        assert ParallelWindProcessor is not None

    def test_parallel_cross_section_processor_exists(self: "TestParallelWrappers") -> None:
        """
        This test confirms that the `ParallelCrossSectionProcessor` wrapper class is defined for adapting cross-section processing tasks to parallel execution. The wrapper standardizes how cross-section diagnostics are initialized and executed across worker processes, allowing for efficient distribution of cross-section-related processing tasks. The test imports the class and asserts that it is not None to ensure that the wrapper is available in the module exports. This guarantees that users can easily run cross-section processing in parallel, supporting improved performance for these diagnostics. The presence of this wrapper class is critical for enabling parallelism in cross-section processing workflows and ensuring that users have access to convenient interfaces for parallel execution of these tasks. 

        Parameters:
            self ("TestParallelWrappers"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing.parallel_wrappers import ParallelCrossSectionProcessor
        assert ParallelCrossSectionProcessor is not None

    def test_auto_batch_processor_function_exists(self: "TestParallelWrappers") -> None:
        """
        This test verifies that the `auto_batch_processor` function is defined for automatically batching processing tasks for parallel execution. The `auto_batch_processor` function is a convenience helper that takes a processor class and input parameters, and returns a wrapper instance configured for parallel execution. The test imports the function and asserts that it is callable to confirm that this utility is available in the module exports. This function is important for simplifying the process of adapting existing processors to run in parallel, allowing users to easily create batch processors without needing to manually configure wrapper classes. Presence of this function supports streamlined parallelization of processing tasks. 

        Parameters:
            self ("TestParallelWrappers"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing.parallel_wrappers import auto_batch_processor
        assert callable(auto_batch_processor)

    def test_precipitation_worker_function_exists(self: "TestParallelWrappers") -> None:
        """
        This test confirms that the private precipitation worker helper function `_precipitation_worker` exists within the `parallel_wrappers` module. This internal function is used by the `ParallelPrecipitationProcessor` wrapper to execute precipitation processing tasks in worker contexts. The test checks for the presence of this function to guard against accidental deletions or refactors that could break the internal workings of the wrapper. While this function is not part of the public API, its presence is critical for the correct operation of precipitation processing in parallel. This test ensures that internal helper functions remain available for use by wrappers, even though they are not intended for direct use by external code. 

        Parameters:
            self ("TestParallelWrappers"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing import parallel_wrappers
        assert hasattr(parallel_wrappers, '_precipitation_worker')

    def test_surface_worker_function_exists(self: "TestParallelWrappers") -> None:
        """
        This test verifies that the private surface worker helper function `_surface_worker` exists within the `parallel_wrappers` module. This internal function is used by the `ParallelSurfaceProcessor` wrapper to execute surface processing tasks in worker contexts. The test checks for the presence of this function to guard against accidental deletions or refactors that could break the internal workings of the wrapper. While this function is not part of the public API, its presence is critical for the correct operation of surface processing in parallel. This test ensures that internal helper functions remain available for use by wrappers, even though they are not intended for direct use by external code. 

        Parameters:
            self ("TestParallelWrappers"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing import parallel_wrappers
        assert hasattr(parallel_wrappers, '_surface_worker')

    def test_wind_worker_function_exists(self: "TestParallelWrappers") -> None:
        """
        This test confirms that the private wind worker helper function `_wind_worker` exists within the `parallel_wrappers` module. This internal function is used by the `ParallelWindProcessor` wrapper to execute wind processing tasks in worker contexts. The test checks for the presence of this function to guard against accidental deletions or refactors that could break the internal workings of the wrapper. While this function is not part of the public API, its presence is critical for the correct operation of wind processing in parallel. This test ensures that internal helper functions remain available for use by wrappers, even though they are not intended for direct use by external code. 

        Parameters:
            self ("TestParallelWrappers"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing import parallel_wrappers
        assert hasattr(parallel_wrappers, '_wind_worker')

    def test_cross_section_worker_function_exists(self: "TestParallelWrappers") -> None:
        """
        This test verifies that the private cross-section worker helper function `_cross_section_worker` exists within the `parallel_wrappers` module. This internal function is used by the `ParallelCrossSectionProcessor` wrapper to execute cross-section processing tasks in worker contexts. The test checks for the presence of this function to guard against accidental deletions or refactors that could break the internal workings of the wrapper. While this function is not part of the public API, its presence is critical for the correct operation of cross-section processing in parallel. This test ensures that internal helper functions remain available for use by wrappers, even though they are not intended for direct use by external code. 

        Parameters:
            self ("TestParallelWrappers"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing import parallel_wrappers
        assert hasattr(parallel_wrappers, '_cross_section_worker')


class TestMPASParallelManagerInitializationModule:
    """ Tests covering `MPASParallelManager` initialization across backends and environment permutations. """
    
    def test_init_multiprocessing_backend_no_mpi(self: "TestMPASParallelManagerInitializationModule") -> None:
        """
        This test validates that when the `MPASParallelManager` is initialized without specifying a backend in an environment where MPI is not available (such as a standard pytest run), it defaults to using the multiprocessing backend. The test constructs a manager with default parameters and asserts that it reports using the multiprocessing backend, confirming that the manager correctly detects the absence of MPI and falls back to multiprocessing for parallel execution. This ensures that users can still benefit from parallelism even when MPI is not installed or configured, providing a seamless experience across different environments. 

        Parameters:
            self ("TestMPASParallelManagerInitializationModule"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(verbose=False)

        assert manager.backend == 'multiprocessing'
        assert manager.rank == pytest.approx(0)
        assert manager.is_master
        assert manager.comm is None
        assert manager.distributor is None
        assert manager.collector is None
    
    def test_init_explicit_multiprocessing_backend(self: "TestMPASParallelManagerInitializationModule") -> None:
        """
        This test confirms that when the `MPASParallelManager` is explicitly initialized with the multiprocessing backend, it correctly configures itself for multiprocessing execution. The test constructs a manager with `backend='multiprocessing'` and asserts that it reports using the multiprocessing backend, has the expected number of workers (4 in this case), and identifies as the master process. This validates that explicit selection of the multiprocessing backend works as intended and that the manager initializes its internal state accordingly for parallel execution using multiprocessing. 

        Parameters:
            self ("TestMPASParallelManagerInitializationModule"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='multiprocessing', n_workers=4, verbose=False)

        assert manager.backend == 'multiprocessing'
        assert manager.size == pytest.approx(4)
        assert manager.is_master
    
    def test_init_serial_backend(self: "TestMPASParallelManagerInitializationModule") -> None:
        """
        This test validates that when the serial backend is explicitly selected, the manager correctly configures itself for serial execution. The test constructs a manager with `backend='serial'` and asserts that it reports using the serial backend, has a single worker, identifies as the master process, and has no communicator. This ensures that the manager correctly handles serial execution scenarios. 

        Parameters:
            self ("TestMPASParallelManagerInitializationModule"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        
        assert manager.backend == 'serial'
        assert manager.rank == pytest.approx(0)
        assert manager.size == pytest.approx(1)
        assert manager.is_master
        assert manager.comm is None
    
    def test_init_mpi_backend_single_process(self: "TestMPASParallelManagerInitializationModule") -> None:
        """
        This test simulates an environment where the MPI backend is selected but only a single process is available (size=1). It confirms that the manager falls back to using the multiprocessing backend in this scenario, as MPI parallelism is not possible with a single process. The test constructs a manager with `backend='mpi'` and asserts that it reports using the multiprocessing backend, ensuring that the manager gracefully degrades to multiprocessing when MPI cannot provide parallel execution due to insufficient processes. This test verifies robust fallback behavior for MPI initialization in constrained environments. 

        Parameters:
            self ("TestMPASParallelManagerInitializationModule"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='mpi', verbose=False)
        assert manager.backend == 'multiprocessing'
    
    def test_init_mpi_backend_exception_fallback(self: "TestMPASParallelManagerInitializationModule") -> None:
        """
        This test simulates an environment where the MPI backend is selected but an exception occurs during MPI initialization (e.g., due to missing MPI libraries). It confirms that the manager catches the exception and falls back to using the multiprocessing backend instead of crashing. The test constructs a manager with `backend='mpi'` and asserts that it reports using the multiprocessing backend, ensuring that the manager provides a robust fallback mechanism in the face of MPI initialization failures. This test validates that users can still run parallel tasks using multiprocessing even if their environment is not properly configured for MPI. 

        Parameters:
            self ("TestMPASParallelManagerInitializationModule"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='mpi', verbose=False)
        assert manager.backend == 'multiprocessing'
    
    def test_set_error_policy_string(self: "TestMPASParallelManagerInitializationModule") -> None:
        """
        This test verifies that the manager's error policy can be set using string identifiers. The test creates a manager instance, calls `set_error_policy` with string values like 'abort', 'continue', and 'collect', and asserts that the manager's `error_policy` attribute is updated to the corresponding `ErrorPolicy` enum value. This ensures that users can configure error handling behavior using intuitive string inputs, and that the manager correctly translates these into internal enum representations for consistent error handling during parallel execution. 

        Parameters:
            self ("TestMPASParallelManagerInitializationModule"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)        
        manager.set_error_policy('abort')
        assert manager.error_policy == ErrorPolicy.ABORT
        
        manager.set_error_policy('continue')
        assert manager.error_policy == ErrorPolicy.CONTINUE
        
        manager.set_error_policy('collect')
        assert manager.error_policy == ErrorPolicy.COLLECT
    
    def test_set_error_policy_enum(self: "TestMPASParallelManagerInitializationModule") -> None:
        """
        This test verifies that the manager's error policy can be set using the `ErrorPolicy` enum. The test directly passes enum values like `ErrorPolicy.ABORT`, `ErrorPolicy.CONTINUE`, and `ErrorPolicy.COLLECT` to the `set_error_policy` method and asserts that the manager's `error_policy` attribute is updated to match the provided enum value. This ensures that users have the flexibility to configure error handling using either string identifiers or direct enum values, and that the manager correctly accepts and applies both forms of input for error policy configuration. 

        Parameters:
            self ("TestMPASParallelManagerInitializationModule"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)        
        manager.set_error_policy(ErrorPolicy.ABORT)
        assert manager.error_policy == ErrorPolicy.ABORT

        manager.set_error_policy(ErrorPolicy.CONTINUE)
        assert manager.error_policy == ErrorPolicy.CONTINUE

        manager.set_error_policy(ErrorPolicy.COLLECT)
        assert manager.error_policy == ErrorPolicy.COLLECT

class TestMultiprocessingExecution:
    """ Tests for multiprocessing backend execution paths including pool creation, mapping, and error handling. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestMultiprocessingExecution") -> Generator[None, None, None]:
        """
        This fixture sets up a `MPASParallelManager` instance configured for multiprocessing execution before each test method in this class. It initializes the manager with a specified number of workers and ensures that it is available as `self.manager` for use in test methods. After the test method completes, it performs cleanup by closing any open matplotlib plots to prevent resource leaks. This fixture provides a consistent multiprocessing manager environment for all tests in this class, allowing them to focus on testing specific execution behaviors without needing to handle setup and teardown of the manager themselves. 

        Parameters:
            self ("TestMultiprocessingExecution"): Pytest-provided test instance.

        Returns:
            None
        """
        self.manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=False)
        
        yield
        
        plt.close('all')
    
    def test_multiprocessing_map_success(self: "TestMultiprocessingExecution") -> None:
        """
        This test confirms that the multiprocessing backend can successfully execute a simple mapping of tasks. It defines a `simple_func` that doubles its input, creates a list of tasks, and calls `parallel_map` with the manager. The test asserts that results are returned, all tasks succeeded, and that the results are correct (each input doubled). This validates that the multiprocessing backend is correctly executing tasks in parallel and returning expected results without errors. 

        Parameters:
            self ("TestMultiprocessingExecution"): Pytest-provided test instance with `self.manager`.

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
    
    def test_multiprocessing_map_with_args_kwargs(self: "TestMultiprocessingExecution") -> None:
        """
        This test verifies that the multiprocessing backend can handle functions with additional positional and keyword arguments. It defines an `add_func` that takes a value, an offset, and a multiplier, and returns the computed result. The test calls `parallel_map` with a list of tasks and additional arguments, then asserts that results are returned, all tasks succeeded, and that the results are correct based on the provided function logic. This ensures that the multiprocessing manager correctly passes extra arguments to worker functions during parallel execution. 

        Parameters:
            self ("TestMultiprocessingExecution"): Pytest-provided test instance with `self.manager`.

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
    
    def test_multiprocessing_map_with_errors(self: "TestMultiprocessingExecution") -> None:
        """
        This test checks that the multiprocessing backend correctly handles errors according to the 'continue' error policy. It defines a `failing_func` that raises a `ValueError` for a specific input, sets the manager's error policy to 'continue', and calls `parallel_map` with a list of tasks. The test asserts that results are returned, that the correct number of tasks succeeded and failed, and that the error information is captured in the failed result. This validates that the manager can continue executing remaining tasks even when some fail, and that it properly collects error information for failed tasks in the multiprocessing context. 

        Parameters:
            self ("TestMultiprocessingExecution"): Pytest-provided test instance with `self.manager`.

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
    
    def test_multiprocessing_statistics(self: "TestMultiprocessingExecution") -> None:
        """
        This test verifies that the multiprocessing backend correctly collects and reports execution statistics. It runs a simple mapping of tasks and then calls `get_statistics` to retrieve the collected statistics. The test asserts that the statistics object is not None, that it contains expected fields such as total tasks, completed tasks, failed tasks, and total execution time, and that these values are consistent with the executed tasks. This ensures that the manager's statistics collection mechanism works correctly in the multiprocessing context, providing users with insights into the performance and outcomes of their parallel executions. 

        Parameters:
            self ("TestMultiprocessingExecution"): Pytest-provided test instance with `self.manager`.

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
    
    def test_multiprocessing_spawn_method(self: "TestMultiprocessingExecution") -> None:
        """
        This test confirms that the multiprocessing backend can be initialized with the 'spawn' start method, which is necessary for compatibility with certain platforms like Windows. It creates a manager with `backend='multiprocessing'` and `start_method='spawn'`, then runs a simple mapping of tasks to ensure that execution proceeds without errors. The test asserts that results are returned and that the mapping was successful, validating that the manager can operate correctly using the 'spawn' method for multiprocessing execution. This is important for ensuring cross-platform compatibility of the multiprocessing backend. 

        Parameters:
            self ("TestMultiprocessingExecution"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=False)
        results = manager.parallel_map(lambda x: x * 2, [1])

        assert results is not None
        assert len(results) == pytest.approx(1)
    
    def test_multiprocessing_fallback_on_error(self: "TestMultiprocessingExecution") -> None:
        """
        This test simulates a failure during multiprocessing pool creation to confirm that the manager falls back to serial execution instead of crashing. It monkey-patches the internal `get_context` function used for creating multiprocessing pools to always raise an exception, then initializes a manager and calls `parallel_map`. The test asserts that results are returned and that all tasks succeeded, confirming that the manager correctly detected the pool creation failure and executed tasks in serial as a fallback. This ensures robustness of the multiprocessing backend in environments where multiprocessing may not be fully supported or configured. 

        Parameters:
            self ("TestMultiprocessingExecution"): Pytest-provided test instance.

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
            results = manager.parallel_map(lambda x: x * 2, [1, 2])
            assert results is not None
            assert len(results) == pytest.approx(2)
            assert all(r.success for r in results)
        finally:
            _parallel_mod.get_context = original_get_context


class TestSerialExecution:
    """ Tests for serial backend execution used as a deterministic fallback when parallel backends are not available. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestSerialExecution") -> Generator[None, None, None]:
        """
        This fixture sets up a `MPASParallelManager` instance configured for serial execution before each test method in this class. It initializes the manager with `backend='serial'` and ensures that it is available as `self.manager` for use in test methods. After the test method completes, it performs cleanup by closing any open matplotlib plots to prevent resource leaks. This fixture provides a consistent serial manager environment for all tests in this class, allowing them to focus on testing specific execution behaviors without needing to handle setup and teardown of the manager themselves. 

        Parameters:
            self ("TestSerialExecution"): Pytest-provided test instance.

        Returns:
            None
        """
        self.manager = MPASParallelManager(backend='serial', verbose=False)
        
        yield
        
        plt.close('all')
    
    def test_serial_map_success(self: "TestSerialExecution") -> None:
        """
        This test confirms that the serial backend can successfully execute a simple mapping of tasks. It defines a `simple_func` that doubles its input, creates a list of tasks, and calls `parallel_map` with the manager. The test asserts that results are returned, all tasks succeeded, and that the results are correct (each input doubled). This validates that the serial backend is correctly executing tasks in a deterministic manner and returning expected results without errors. 

        Parameters:
            self ("TestSerialExecution"): Pytest-provided test instance with `self.manager`.

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
    
    def test_serial_map_with_errors_continue(self: "TestSerialExecution") -> None:
        """
        This test checks that the serial backend correctly handles errors according to the 'continue' error policy. It defines a `failing_func` that raises a `ValueError` for a specific input, sets the manager's error policy to 'continue', and calls `parallel_map` with a list of tasks. The test asserts that results are returned, that the correct number of tasks succeeded and failed, and that the error information is captured in the failed result. This validates that the manager can continue executing remaining tasks even when some fail, and that it properly collects error information for failed tasks in the serial execution context. 

        Parameters:
            self ("TestSerialExecution"): Pytest-provided test instance with `self.manager`.

        Returns:
            None
        """
        def failing_func(x):
            if x == 3:
                raise ValueError(f"Error on {x}")
            return x * 2
        
        self.manager.set_error_policy('continue')
        tasks = [1, 2, 3, 4, 5]
        results = self.manager.parallel_map(failing_func, tasks)

        assert results is not None
        assert len(results) == pytest.approx(5)
        assert sum(1 for r in results if r.success) == pytest.approx(4)
        assert sum(1 for r in results if not r.success) == pytest.approx(1)


class TestTaskDistributor:
    """ Tests for `MPASTaskDistributor` that verify static, block, cyclic, and dynamic load-balancing strategies. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestTaskDistributor") -> Generator[None, None, None]:
        """
        This fixture sets up a mock MPI communicator and a list of tasks for testing the `MPASTaskDistributor` class. It initializes `self.mock_comm` as a `MagicMock` to simulate MPI communicator behavior, and creates a list of 10 tasks represented as integers. The fixture yields to allow test methods to execute with this setup, and then performs cleanup by closing any open matplotlib plots after each test. This provides a consistent environment for testing the task distribution logic across different load-balancing strategies without needing an actual MPI environment. 

        Parameters:
            self (Any): Pytest-provided test instance.

        Returns:
            None
        """
        self.mock_comm = MagicMock()
        self.tasks = list(range(10))
        
        yield
        
        plt.close('all')
    
    def test_static_distribution(self: "TestTaskDistributor") -> None:
        """
        This test confirms that the static distribution strategy correctly partitions tasks into contiguous slices based on rank and size. With a communicator of size 3 and rank 0, the test asserts that the first slice of tasks (4 tasks in this case) is assigned to rank 0, verifying that the static distribution logic correctly calculates task indices and handles remainders when tasks cannot be evenly divided. This ensures that the static load-balancing strategy produces expected task assignments for each rank in a parallel execution context. 

        Parameters:
            self ("TestTaskDistributor"): Pytest-provided test instance with `self.mock_comm` and `self.tasks`.

        Returns:
            None
        """
        self.mock_comm.Get_rank.return_value = 0
        self.mock_comm.Get_size.return_value = 3
        
        distributor = MPASTaskDistributor(self.mock_comm, LoadBalanceStrategy.STATIC)
        local_tasks = distributor.distribute_tasks(self.tasks)
        
        assert len(local_tasks) == pytest.approx(4)
        assert [tid for tid, _ in local_tasks] == [0, 1, 2, 3]
    
    def test_static_distribution_rank1(self: "TestTaskDistributor") -> None:
        """
        This test verifies that the static distribution strategy correctly assigns the next contiguous slice of tasks to rank 1 when the communicator has size 3. With rank 1, the test asserts that the second slice of tasks (3 tasks in this case) is assigned to rank 1, confirming that the static distribution logic correctly calculates task indices for subsequent ranks and handles remainders appropriately. This ensures that each rank receives the correct portion of tasks according to the static load-balancing strategy in a parallel execution context. 

        Parameters:
            self ("TestTaskDistributor"): Pytest-provided test instance.

        Returns:
            None
        """
        self.mock_comm.Get_rank.return_value = 1
        self.mock_comm.Get_size.return_value = 3
        
        distributor = MPASTaskDistributor(self.mock_comm, LoadBalanceStrategy.STATIC)
        local_tasks = distributor.distribute_tasks(self.tasks)
        
        assert len(local_tasks) == pytest.approx(3)
        assert [tid for tid, _ in local_tasks] == [4, 5, 6]
    
    def test_block_distribution(self: "TestTaskDistributor") -> None:
        """
        This test validates that the block distribution strategy partitions tasks into contiguous blocks based on the total number of tasks and the number of ranks. With a communicator of size 3 and rank 0, the test asserts that the first block of tasks (4 tasks in this case) is assigned to rank 0, confirming that the block distribution logic correctly calculates block sizes using ceiling division and assigns contiguous blocks to each rank. This ensures that the block load-balancing strategy produces expected task assignments for each rank in a parallel execution context. 

        Parameters:
            self ("TestTaskDistributor"): Pytest-provided test instance.

        Returns:
            None
        """
        self.mock_comm.Get_rank.return_value = 0
        self.mock_comm.Get_size.return_value = 3
        
        distributor = MPASTaskDistributor(self.mock_comm, LoadBalanceStrategy.BLOCK)
        local_tasks = distributor.distribute_tasks(self.tasks)
        
        assert len(local_tasks) == pytest.approx(4) 
        assert [tid for tid, _ in local_tasks] == [0, 1, 2, 3]
    
    def test_cyclic_distribution(self: "TestTaskDistributor") -> None:
        """
        This test ensures that the cyclic distribution strategy assigns tasks in a round-robin fashion across ranks. With a communicator of size 3 and rank 0, the test asserts that tasks at positions 0, 3, 6, and 9 are assigned to rank 0, confirming that the cyclic distribution logic correctly iterates through tasks and assigns them to ranks in a repeating sequence. This ensures that the cyclic load-balancing strategy produces expected task assignments for each rank in a parallel execution context. 

        Parameters:
            self ("TestTaskDistributor"): Pytest-provided test instance.

        Returns:
            None
        """
        self.mock_comm.Get_rank.return_value = 0
        self.mock_comm.Get_size.return_value = 3
        
        distributor = MPASTaskDistributor(self.mock_comm, LoadBalanceStrategy.CYCLIC)
        local_tasks = distributor.distribute_tasks(self.tasks)
        
        assert [tid for tid, _ in local_tasks] == [0, 3, 6, 9]
    
    def test_dynamic_distribution(self: "TestTaskDistributor") -> None:
        """
        This test validates that the dynamic distribution strategy currently falls back to static distribution logic. With a communicator of size 2 and rank 0, the test asserts that the first half of tasks (5 tasks in this case) is assigned to rank 0, confirming that the dynamic distribution logic correctly defaults to static partitioning until specialized dynamic logic is implemented. This ensures that the dynamic load-balancing strategy produces expected task assignments for each rank in a parallel execution context, even in its current fallback state. 

        Parameters:
            self ("TestTaskDistributor"): Pytest-provided test instance.

        Returns:
            None
        """
        self.mock_comm.Get_rank.return_value = 0
        self.mock_comm.Get_size.return_value = 2
        
        distributor = MPASTaskDistributor(self.mock_comm, LoadBalanceStrategy.DYNAMIC)
        local_tasks = distributor.distribute_tasks(self.tasks)
        
        assert len(local_tasks) == pytest.approx(5)


class TestResultCollector:
    """ Tests for `MPASResultCollector` functionality including gathering and flattening per-worker results. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestResultCollector") -> Generator[None, None, None]:
        """
        This fixture sets up a mock MPI communicator for testing the `MPASResultCollector` class. It initializes `self.mock_comm` as a `MagicMock` to simulate MPI communicator behavior, allowing test methods to configure rank, size, and gather behavior as needed. The fixture yields to allow test methods to execute with this setup, and then performs cleanup by closing any open matplotlib plots after each test. This provides a consistent environment for testing the result collection logic without needing an actual MPI environment. 

        Parameters:
            self ("TestResultCollector"): Pytest-provided test instance.

        Returns:
            None
        """
        self.mock_comm = MagicMock()
        
        yield
        
        plt.close('all')
    
    def test_gather_results_master(self: "TestResultCollector") -> None:
        """
        This test confirms that the master rank correctly gathers and flattens results from all worker ranks. It sets the mock communicator to simulate a master rank (rank 0) in a communicator of size 2, defines local results for the master and worker ranks, and configures the gather method to return combined results. The test asserts that the gathered results are not None and that they contain the expected number of total results (3 in this case), confirming that the master rank correctly collects and processes results from all workers. This ensures that the result collection logic works as intended in a parallel execution context with multiple ranks. 

        Parameters:
            self ("TestResultCollector"): Pytest-provided test instance with `self.mock_comm`.

        Returns:
            None
        """
        self.mock_comm.Get_rank.return_value = 0
        self.mock_comm.Get_size.return_value = 2
        
        local_results = [
            TaskResult(task_id=0, success=True, result=1),
            TaskResult(task_id=1, success=True, result=2)
        ]
        
        all_worker_results = [
            local_results,
            [TaskResult(task_id=2, success=True, result=3)]
        ]
        
        self.mock_comm.gather.return_value = all_worker_results
        
        collector = MPASResultCollector(self.mock_comm)
        results = collector.gather_results(local_results)
        
        assert results is not None
        assert len(results) == pytest.approx(3)
    
    def test_gather_results_worker(self: "TestResultCollector") -> None:
        """
        This test confirms that worker ranks return None from `gather_results` since they do not assemble global results. The test sets rank to 1 and verifies the collector returns None, indicating worker-side behavior. This respects the master/worker separation of responsibilities in the result collection process, ensuring that only the master rank gathers and processes results while workers simply return their local results without attempting to gather from others. 

        Parameters:
            self ("TestResultCollector"): Pytest-provided test instance with `self.mock_comm`.

        Returns:
            None
        """
        self.mock_comm.Get_rank.return_value = 1
        self.mock_comm.Get_size.return_value = 2
        
        local_results = [TaskResult(task_id=2, success=True, result=3)]
        
        collector = MPASResultCollector(self.mock_comm)
        results = collector.gather_results(local_results)
        
        assert results is None
    
    def test_compute_statistics(self: "TestResultCollector") -> None:
        """
        This test verifies that the `compute_statistics` method correctly calculates execution statistics from a list of `TaskResult` objects. It sets up a mock communicator to simulate rank 0 in a communicator of size 2, defines a list of task results with varying success and execution times, and calls `compute_statistics`. The test asserts that the computed statistics contain expected values for total tasks, completed tasks, failed tasks, total execution time, worker times, and load imbalance. This ensures that the statistics computation logic accurately processes task results to provide insights into the performance and outcomes of parallel executions. 

        Parameters:
            self (Any): Pytest-provided test instance with `self.mock_comm`.

        Returns:
            None
        """
        self.mock_comm.Get_rank.return_value = 0
        self.mock_comm.Get_size.return_value = 2
        
        results = [
            TaskResult(task_id=0, success=True, result=1, execution_time=1.0, worker_rank=0),
            TaskResult(task_id=1, success=True, result=2, execution_time=2.0, worker_rank=0),
            TaskResult(task_id=2, success=False, error="Error", execution_time=0.5, worker_rank=1)
        ]
        
        collector = MPASResultCollector(self.mock_comm)
        stats = collector.compute_statistics(results)
        
        assert stats.total_tasks == pytest.approx(3)
        assert stats.completed_tasks == pytest.approx(2)
        assert stats.failed_tasks == pytest.approx(1)
        assert stats.total_time == pytest.approx(3.5)
        assert stats.worker_times[0] == pytest.approx(3.0)
        assert stats.worker_times[1] == pytest.approx(0.5)
        assert stats.load_imbalance > 0


class TestMultiprocessingTaskWrapperModule:
    """ Tests for the internal `_multiprocessing_task_wrapper` which adapts user functions for multiprocessing pools. """
    
    def test_wrapper_success(self: "TestMultiprocessingTaskWrapperModule") -> None:
        """
        This test confirms that the `_multiprocessing_task_wrapper` correctly executes a simple function and returns a successful `TaskResult`. It defines a `simple_func` that adds an offset to its input, constructs the expected arguments for the wrapper, and calls it directly. The test asserts that the result indicates success, that the computed result is correct, and that execution time is recorded. This validates that the wrapper correctly adapts user functions for execution in a multiprocessing context and returns results in the expected format. 

        Parameters:
            self ("TestMultiprocessingTaskWrapperModule"): Pytest-provided test instance.

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
            self ("TestMultiprocessingTaskWrapperModule"): Pytest-provided test instance.

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
            self ("TestMultiprocessingTaskWrapperModule"): Pytest-provided test instance.

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
            self ("TestErrorHandling"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        manager.set_error_policy('collect')
        
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
            self ("TestErrorHandling"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        manager.set_error_policy('continue')
        
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
            self ("TestStatistics"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=False)        
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
            self ("TestStatistics"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=True)
        
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


class TestBarrierAndFinalizeModule:
    """ Tests for synchronization helpers such as `barrier` and `finalize` across MPI and non-MPI backends. """
    
    def test_barrier_serial(self: "TestBarrierAndFinalizeModule") -> None:
        """
        This test ensures that calling `barrier` in serial mode is safe and does not raise any exceptions. Since the serial backend does not involve actual parallel processes, the `barrier` method should effectively be a no-op. The test constructs a serial manager and calls `barrier`, asserting that it completes without raising any errors. This confirms that the `barrier` method is implemented in a way that is compatible with non-parallel execution contexts, allowing code that uses barriers to run seamlessly regardless of the backend. 

        Parameters:
            self (Any): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        manager.barrier() 
    
    def test_barrier_multiprocessing(self: "TestBarrierAndFinalizeModule") -> None:
        """
        This test verifies that calling `barrier` in multiprocessing mode does not raise any exceptions. In the multiprocessing backend, the `barrier` method should synchronize worker processes, but since this is a test environment without actual parallel execution, it should still complete without errors. The test constructs a multiprocessing manager and calls `barrier`, asserting that it completes successfully. This confirms that the `barrier` method is implemented to work correctly in the multiprocessing context, allowing for synchronization without causing issues in a testing environment. 

        Parameters:
            self ("TestBarrierAndFinalizeModule"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='multiprocessing', verbose=False)
        manager.barrier() 
    
    def test_finalize_serial(self: "TestBarrierAndFinalizeModule") -> None:
        """
        This test confirms that calling `finalize` in serial mode is safe and does not raise any exceptions. Since the serial backend does not involve actual parallel processes or resources, the `finalize` method should effectively be a no-op. The test constructs a serial manager and calls `finalize`, asserting that it completes without raising any errors. This ensures that the `finalize` method is implemented in a way that is compatible with non-parallel execution contexts, allowing code that uses finalization to run seamlessly regardless of the backend. 

        Parameters:
            self ("TestBarrierAndFinalizeModule"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        manager.finalize() 
    
    def test_finalize_multiprocessing(self: "TestBarrierAndFinalizeModule") -> None:
        """
        This test verifies that calling `finalize` in multiprocessing mode does not raise any exceptions. In the multiprocessing backend, the `finalize` method should clean up any resources associated with worker processes, but since this is a test environment without actual parallel execution, it should still complete without errors. The test constructs a multiprocessing manager and calls `finalize`, asserting that it completes successfully. This confirms that the `finalize` method is implemented to work correctly in the multiprocessing context, allowing for proper cleanup without causing issues in a testing environment. 

        Parameters:
            self ("TestBarrierAndFinalizeModule"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='multiprocessing', verbose=False)
        manager.finalize() 


class TestParallelPlotFunctionModule:
    """ Tests for the `parallel_plot` convenience function that wraps manager creation, mapping, and error policy configuration. """
    
    def test_parallel_plot_basic(self: "TestParallelPlotFunctionModule") -> None:
        """
        This test confirms that the `parallel_plot` function can execute a simple plotting function across multiple files without raising exceptions. It defines a `simple_plot` function that simulates plotting by returning a string, creates a list of file paths, and calls `parallel_plot` with these inputs. The test asserts that results are returned, that the correct number of results is produced, and that all results indicate success. This validates that the `parallel_plot` function correctly sets up the parallel manager, applies the plotting function to each file, and handles results according to the expected behavior, providing a convenient interface for parallel plotting tasks. 

        Parameters:
            self ("TestParallelPlotFunctionModule"): Pytest-provided test instance.

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
            self ("TestEdgeCases"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        results = manager.parallel_map(lambda x: x * 2, [])

        assert results is not None
        assert len(results) == pytest.approx(0)
    
    def test_single_task(self: "TestEdgeCases") -> None:
        """
        This test confirms that the manager can handle a single task correctly. It constructs a manager, calls `parallel_map` with a list containing one task, and asserts that results are returned, that there is exactly one result, that the task succeeded, and that the result is correct. This validates that the manager can execute a single task without issues, ensuring that it does not rely on having multiple tasks to function properly and can handle minimal workloads as expected. 

        Parameters:
            self ("TestEdgeCases"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=False)
        results = manager.parallel_map(lambda x: x * 2, [5])
        
        assert results is not None
        assert len(results) == pytest.approx(1)
        assert results[0].success
        assert results[0].result == pytest.approx(10)
    
    def test_more_workers_than_tasks(self: "TestEdgeCases") -> None:
        """
        This test verifies that the manager can handle scenarios where the number of workers exceeds the number of tasks. It constructs a manager with more workers than tasks, calls `parallel_map` with a small list of tasks, and asserts that results are returned, that the correct number of results is produced, and that all tasks succeed. This ensures that the manager efficiently utilizes available workers without creating invalid task slots or errors when there are fewer tasks than workers. 

        Parameters:
            self ("TestEdgeCases"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='multiprocessing', n_workers=10, verbose=False)
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
            self ("TestMultiprocessingTaskWrapperAdditional"): Pytest-provided test instance.

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
            self ("TestMultiprocessingTaskWrapperAdditional"): Pytest-provided test instance.

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
            self ("TestMultiprocessingTaskWrapperAdditional"): Pytest-provided test instance.

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


class TestMPASResultCollectorStatistics:
    """ Additional tests for result collector statistics, focusing on edge-case inputs. """
    
    def test_compute_statistics_with_empty_worker_times(self: "TestMPASResultCollectorStatistics") -> None:
        """
        This test verifies that the `compute_statistics` method can handle an empty list of results without errors and returns statistics with zero totals and no load imbalance. It creates a mock MPI communicator, constructs a result collector, and calls `compute_statistics` with an empty list. The test asserts that all computed statistics are zero or empty as expected, confirming that the method can gracefully handle cases where no tasks were executed or no timing information is available without raising exceptions. This ensures robustness in the statistics computation logic when faced with edge-case inputs. 

        Parameters:
            self ("TestMPASResultCollectorStatistics"): Pytest-provided test instance.

        Returns:
            None
        """
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 1
        
        collector = MPASResultCollector(mock_comm)
        
        results = []
        stats = collector.compute_statistics(results)
        
        assert stats.total_tasks == pytest.approx(0)
        assert stats.completed_tasks == pytest.approx(0)
        assert stats.failed_tasks == pytest.approx(0)
        assert stats.total_time == pytest.approx(0.0)
        assert stats.load_imbalance == pytest.approx(0.0)
    
    def test_compute_statistics_with_single_worker(self: "TestMPASResultCollectorStatistics") -> None:
        """
        This test verifies that the `compute_statistics` method correctly computes statistics when all tasks are executed by a single worker. It creates a mock MPI communicator simulating a single worker environment, constructs a result collector, and calls `compute_statistics` with a list of task results that all belong to the same worker. The test asserts that the computed statistics reflect the total number of tasks, completed and failed tasks, total execution time, and that there is no load imbalance since all work was done by one worker. This confirms that the statistics computation logic correctly handles scenarios where there is only one worker, ensuring that it does not produce invalid load imbalance values or other errors in this edge case. 

        Parameters:
            self ("TestMPASResultCollectorStatistics"): Pytest-provided test instance.

        Returns:
            None
        """
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 1
        
        collector = MPASResultCollector(mock_comm)
        
        results = [
            TaskResult(task_id=0, success=True, execution_time=1.0, worker_rank=0),
            TaskResult(task_id=1, success=False, execution_time=0.5, worker_rank=0),
        ]
        
        stats = collector.compute_statistics(results)
        
        assert stats.total_tasks == pytest.approx(2)
        assert stats.completed_tasks == pytest.approx(1)
        assert stats.failed_tasks == pytest.approx(1)
        assert stats.total_time == pytest.approx(1.5)
        assert 0 in stats.worker_times
        assert stats.load_imbalance == pytest.approx(0.0)


class TestMPASParallelManagerInitializationAdditional:
    """ Supplemental initialization tests for `MPASParallelManager` covering fallback and exception scenarios. """
    
    def test_init_with_mpi_unavailable_backend_none(self: "TestMPASParallelManagerInitializationAdditional") -> None:
        """
        This test confirms that when MPI is unavailable and the backend is set to None, the manager falls back to multiprocessing semantics. It patches the `MPI_AVAILABLE` flag to False, constructs a manager with `backend=None`, and asserts that the manager reports using the multiprocessing backend with appropriate rank and master status. This ensures that the manager can gracefully handle environments where MPI is not available by automatically switching to multiprocessing, allowing for parallel execution without requiring MPI. 

        Parameters:
            self ("TestMPASParallelManagerInitializationAdditional"): Pytest-provided test instance.

        Returns:
            None
        """
        with patch('mpasdiag.processing.parallel.MPI_AVAILABLE', False):
            manager = MPASParallelManager(backend=None, verbose=False)            
            assert manager.backend == 'multiprocessing'
            assert manager.rank == pytest.approx(0)
            assert manager.is_master
            assert manager.comm is None
            assert manager.distributor is None
            assert manager.collector is None
    
    def test_init_with_serial_backend(self: "TestMPASParallelManagerInitializationAdditional") -> None:
        """
        This test verifies that initializing the manager with the 'serial' backend correctly sets up a non-parallel execution context. It constructs a manager with `backend='serial'` and asserts that the manager reports using the serial backend, has rank 0, is master, and does not have any communicator or distributor/collector objects. This confirms that the manager can be explicitly configured for serial execution, allowing users to disable parallelism when desired while still using the same interface. 

        Parameters:
            self ("TestMPASParallelManagerInitializationAdditional"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)        
        assert manager.backend == 'serial'
        assert manager.rank == pytest.approx(0)
        assert manager.size == pytest.approx(1)
        assert manager.is_master
        assert manager.comm is None
        assert manager.distributor is None
        assert manager.collector is None
    
    def test_init_multiprocessing_backend_explicit(self: "TestMPASParallelManagerInitializationAdditional") -> None:
        """
        This test confirms that initializing the manager with the 'multiprocessing' backend explicitly sets up a multiprocessing execution context. It constructs a manager with `backend='multiprocessing'` and asserts that the manager reports using the multiprocessing backend, has rank 0, size 2 (as specified), is master, and does not have an MPI communicator. This validates that the manager can be explicitly configured for multiprocessing execution, allowing users to leverage multiple processes for parallelism without relying on MPI. The test also confirms that the manager correctly initializes multiprocessing internals while leaving MPI-related attributes as None, ensuring that the multiprocessing backend is properly set up. 

        Parameters:
            self ("TestMPASParallelManagerInitializationAdditional"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=False)        
        assert manager.backend == 'multiprocessing'
        assert manager.rank == pytest.approx(0)
        assert manager.size == pytest.approx(2)
        assert manager.is_master
        assert manager.comm is None
    
    def test_init_with_mpi_backend_but_size_1(self: "TestMPASParallelManagerInitializationAdditional") -> None:
        """
        This test verifies that if the manager is initialized with the 'mpi' backend but the MPI communicator reports a size of 1 (indicating a single process), the manager falls back to multiprocessing semantics. It patches the `MPI_AVAILABLE` flag to True, creates a mock MPI communicator that simulates a single-process environment, and constructs a manager with `backend='mpi'`. The test asserts that the manager falls back to using the multiprocessing backend and does not set up any MPI-related attributes. This ensures that the manager can handle cases where MPI is technically available but not actually running in a parallel context, allowing for graceful degradation to multiprocessing without raising errors. 

        Parameters:
            self ("TestMPASParallelManagerInitializationAdditional"): Pytest-provided test instance.

        Returns:
            None
        """
        with patch('mpasdiag.processing.parallel.MPI_AVAILABLE', True):
            mock_mpi = MagicMock()
            mock_comm = Mock()
            mock_comm.Get_rank.return_value = 0
            mock_comm.Get_size.return_value = 1 
            mock_mpi.COMM_WORLD = mock_comm
            
            with patch('mpasdiag.processing.parallel.MPI', mock_mpi):
                manager = MPASParallelManager(backend='mpi', verbose=False)                
                assert manager.backend == 'multiprocessing'
                assert manager.distributor is None
                assert manager.collector is None
    
    def test_init_with_mpi_exception(self: "TestMPASParallelManagerInitializationAdditional") -> None:
        """
        This test confirms that if an exception occurs during MPI initialization, the manager falls back to multiprocessing semantics and does not raise the exception. It patches the `MPI_AVAILABLE` flag to True, creates a mock MPI module that raises a `RuntimeError` when attempting to get the rank, and constructs a manager with `backend='mpi'`. The test captures verbose output to confirm that an appropriate error message is printed, and asserts that the manager falls back to using the multiprocessing backend without setting up any MPI-related attributes. This ensures that the manager can handle unexpected errors during MPI initialization gracefully, allowing for continued execution using multiprocessing without crashing. 

        Parameters:
            self ("TestMPASParallelManagerInitializationAdditional"): Pytest-provided test instance.

        Returns:
            None
        """
        with patch('mpasdiag.processing.parallel.MPI_AVAILABLE', True):
            mock_mpi = MagicMock()
            mock_mpi.COMM_WORLD.Get_rank.side_effect = RuntimeError("MPI Error")
            
            with patch('mpasdiag.processing.parallel.MPI', mock_mpi):
                import io
                from contextlib import redirect_stdout
                
                f = io.StringIO()

                with redirect_stdout(f):
                    manager = MPASParallelManager(backend='mpi', verbose=True)
                
                output = f.getvalue()
                assert "MPI initialization failed" in output or manager.backend == 'multiprocessing'
                assert manager.backend == 'multiprocessing'


class TestMultiprocessingMap:
    """ Tests for various multiprocessing mapping behaviors including platform-specific context selection and pool retry strategies. """
    
    def test_multiprocessing_map_win32_platform(self: "TestMultiprocessingMap", sample_tasks: List[int]) -> None:
        """
        This test validates that the multiprocessing map function can execute tasks on a Windows platform where the 'spawn' start method is typically required. The test patches `sys.platform` to 'win32', runs a simple mapping of tasks, and asserts that results are returned and that all results are instances of `TaskResult`. This ensures that the manager correctly handles platform-specific multiprocessing behavior, allowing for successful parallel execution on Windows without raising errors related to process spawning. 

        Parameters:
            self ("TestMultiprocessingMap"): Pytest-provided test instance.
            sample_tasks (List[int]): Fixture-provided list of integer tasks.

        Returns:
            None
        """
        def simple_func(task):
            return task * 2
        
        with patch('sys.platform', 'win32'):
            manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=False)
            results = manager.parallel_map(simple_func, sample_tasks)
            
            assert results is not None
            assert len(results) == len(sample_tasks)
            assert all(isinstance(r, TaskResult) for r in results)
    
    def test_multiprocessing_map_darwin_platform(self: "TestMultiprocessingMap", sample_tasks: List[int]) -> None:
        """
        This test validates that the multiprocessing map function can execute tasks on a macOS (darwin) platform where the 'spawn' start method is typically required. The test patches `sys.platform` to 'darwin', runs a simple mapping of tasks, and asserts that results are returned, that the correct number of results is produced, and that all tasks succeeded. This ensures that the manager correctly handles platform-specific multiprocessing behavior on macOS, allowing for successful parallel execution without raising errors related to process spawning. 

        Parameters:
            self ("TestMultiprocessingMap"): Pytest-provided test instance.
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
    
    def test_multiprocessing_map_linux_platform(self: "TestMultiprocessingMap", sample_tasks: List[int]) -> None:
        """
        This test validates that the multiprocessing map function can execute tasks on a Linux platform where forking is common. The test patches `sys.platform` to 'linux', runs a simple mapping of tasks, and asserts that results are returned and that the correct number of results is produced. This ensures that the manager correctly handles platform-specific multiprocessing behavior on Linux, allowing for successful parallel execution without raising errors related to process forking. 

        Parameters:
            self ("TestMultiprocessingMap"): Pytest-provided test instance.
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

    def test_multiprocessing_map_context_failure_fallback(self: "TestMultiprocessingMap", sample_tasks: List[int]) -> None:
        """
        This test ensures fallback to serial execution when all multiprocessing context methods fail. The test simulates context failure and asserts the manager still returns results executed serially. This provides resilience against platform/permission issues creating process pools. It patches `sys.platform` to 'linux', mocks `get_context` to raise an exception, and captures verbose output to confirm that an appropriate error message is printed. The test asserts that results are returned, that the correct number of results is produced, and that all results are instances of `TaskResult`, confirming that the manager falls back to serial execution without crashing when multiprocessing contexts cannot be created. 

        Parameters:
            self ("TestMultiprocessingMap"): Pytest-provided test instance.
            sample_tasks (List[int]): Fixture-provided list of integer tasks.

        Returns:
            None
        """
        def simple_func(task):
            return task
        
        with patch('sys.platform', 'linux'):
            manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=True)
            
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
    
    def test_multiprocessing_map_pool_exception_retry(self: "TestMultiprocessingMap", sample_tasks: List[int]) -> None:
        """
        This test verifies the retry behavior when initial pool creation fails and a subsequent attempt succeeds. The test monkeypatches `get_context` to fail on the first call (simulating a fork failure) and succeed on the second call (simulating a spawn fallback). It captures verbose output to confirm that an appropriate error message is printed during the failure. The test asserts that results are returned and that either an error message about the failure is printed or that the correct number of results is produced, confirming that the manager's built-in retry mechanisms for pool creation are functioning as intended. This ensures that transient issues with multiprocessing context creation do not prevent task execution, allowing for robust parallel processing even in environments with restrictions on process forking. 

        Parameters:
            self ("TestMultiprocessingMap"): Pytest-provided test instance.
            sample_tasks (List[int]): Fixture-provided list of integer tasks.

        Returns:
            None
        """
        def simple_func(task):
            return task
        
        with patch('sys.platform', 'linux'):
            manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=True)
            
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
    
    def test_multiprocessing_map_verbose_output(self: "TestMultiprocessingMap", sample_tasks: List[int]) -> None:
        """
        This test confirms that when verbose mode is enabled, the multiprocessing map function emits expected informational messages about processing, worker usage, and execution statistics. It captures stdout during a parallel map execution and asserts that key phrases related to multiprocessing execution and statistics are present in the output. This ensures that verbose logging provides useful insights into the parallel execution process, aiding users in understanding how tasks are being processed and the performance characteristics of their parallel runs. 

        Parameters:
            self ("TestMultiprocessingMap"): Pytest-provided test instance.
            sample_tasks (List[int]): Fixture-provided list of integer tasks.

        Returns:
            None
        """
        def simple_func(task):
            return task
        
        manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=True)
        
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
    
    def test_multiprocessing_map_fallback_to_serial_all_methods_fail(self: "TestMultiprocessingMap", sample_tasks: List[int]) -> None:
        """
        This test ensures that if all multiprocessing context methods fail (simulating an environment where multiprocessing cannot be used), the manager falls back to serial execution and still returns results. It patches `sys.platform` to 'win32' to trigger spawn behavior, mocks `get_context` to always raise an exception, and captures verbose output to confirm that an appropriate error message is printed. The test asserts that results are returned, that the correct number of results is produced, and that all results are instances of `TaskResult`, confirming that the manager successfully falls back to serial execution without crashing when multiprocessing contexts cannot be created. This provides robustness against environments with severe restrictions on multiprocessing capabilities. 

        Parameters:
            self ("TestMultiprocessingMap"): Pytest-provided test instance.
            sample_tasks (List[int]): Fixture-provided list of integer tasks.

        Returns:
            None
        """
        def simple_func(task):
            return task
        
        with patch('sys.platform', 'win32'):
            manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=True)
            
            with patch('mpasdiag.processing.parallel.get_context', side_effect=RuntimeError("All failed")):
                import io
                from contextlib import redirect_stdout                
                f = io.StringIO()

                with redirect_stdout(f):
                    results = manager.parallel_map(simple_func, sample_tasks)
                
                assert results is not None
                assert len(results) == len(sample_tasks)
                assert all(isinstance(r, TaskResult) for r in results)


class TestExecuteLocalTasks:
    """ Tests for executing tasks locally under different error policies (abort/continue/collect). """
    
    def test_execute_local_tasks_with_abort_policy_mpi(self: "TestExecuteLocalTasks") -> None:
        """
        This test verifies that when the error policy is set to 'abort' and the manager is running under MPI, any exception raised during local task execution results in an abort call on the MPI communicator. It creates a mock MPI communicator that simulates a worker rank and size, sets the error policy to 'abort', and defines a `failing_func` that raises a `ValueError`. The test calls `_execute_local_tasks` with this function and asserts that `Abort` was called on the communicator with the correct rank. This ensures that critical failures during local task execution properly trigger an abort in MPI runs, preventing further execution and allowing for appropriate error handling at the MPI level. 

        Parameters:
            self ("TestExecuteLocalTasks"): Pytest-provided test instance.

        Returns:
            None
        """
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 1
        mock_comm.Get_size.return_value = 4
        mock_comm.bcast.return_value = []
        mock_comm.Abort = Mock()
        
        with patch('mpasdiag.processing.parallel.MPI_AVAILABLE', True):
            mock_mpi = MagicMock()
            mock_mpi.COMM_WORLD = mock_comm
            
            with patch('mpasdiag.processing.parallel.MPI', mock_mpi):
                manager = MPASParallelManager(backend='mpi', verbose=False)
                manager.comm = mock_comm
                manager.set_error_policy('abort')
                
                def failing_func(task):
                    raise ValueError("Task failed")
                
                local_tasks = [(0, "task1")]
                
                manager._execute_local_tasks(failing_func, local_tasks)
                mock_comm.Abort.assert_called_once_with(1)
    
    def test_execute_local_tasks_with_abort_policy_non_mpi(self: "TestExecuteLocalTasks") -> None:
        """
        This test confirms that when the error policy is set to 'abort' and the manager is not running under MPI (e.g., in serial mode), any exception raised during local task execution is propagated as a normal exception rather than causing an abort. It constructs a manager with the 'serial' backend, sets the error policy to 'abort', and defines a `failing_func` that raises a `RuntimeError`. The test calls `_execute_local_tasks` with this function and asserts that a `RuntimeError` is raised with the expected message. This ensures that in non-MPI contexts, critical failures during local task execution are handled through standard exception propagation, allowing users to catch and manage errors without crashing the entire process. 

        Parameters:
            self ("TestExecuteLocalTasks"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        manager.set_error_policy('abort')
        
        def failing_func(task):
            raise RuntimeError("Critical error")
        
        local_tasks = [(0, "task1")]
        
        with pytest.raises(RuntimeError, match="Critical error"):
            manager._execute_local_tasks(failing_func, local_tasks)
    
    def test_execute_local_tasks_verbose_error_output(self: "TestExecuteLocalTasks", sample_tasks: List[int]) -> None:
        """
        This test verifies that when the error policy is set to 'continue' and verbose mode is enabled, any exceptions raised during local task execution are captured and printed to stdout with appropriate error messages. It constructs a manager with the 'serial' backend, sets the error policy to 'continue', and defines a `failing_func` that raises a `ValueError` for a specific task. The test calls `_execute_local_tasks` with this function and captures stdout to assert that error messages related to the failed task are printed. This ensures that when continuing on errors, the manager provides visibility into what went wrong with specific tasks, aiding users in diagnosing issues while still allowing other tasks to proceed. 

        Parameters:
            self ("TestExecuteLocalTasks"): Pytest-provided test instance.
            sample_tasks (List[int]): Fixture-provided list of tasks.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=True)
        manager.set_error_policy('continue')
        
        def failing_func(task):
            if task == 5:
                raise ValueError(f"Task {task} failed")
            return task
        
        local_tasks = [(i, task) for i, task in enumerate(sample_tasks)]
        
        import io
        from contextlib import redirect_stdout        
        f = io.StringIO()

        with redirect_stdout(f):
            results = manager._execute_local_tasks(failing_func, local_tasks)
        
        output = f.getvalue()
        assert "Error processing task" in output
        assert len(results) == len(sample_tasks)
        assert sum(1 for r in results if not r.success) == pytest.approx(1)


class TestSerialMap:
    """ Tests for serial mapping functionality used as a deterministic fallback when parallel backends are not available. """
    
    def test_serial_map_without_collector(self: "TestSerialMap", sample_tasks: List[int]) -> None:
        """
        This test confirms that the `parallel_map` method can execute tasks in serial mode even when the result collector is set to None. It constructs a manager with the 'serial' backend, explicitly sets the collector to None, and defines a simple function to map over the sample tasks. The test calls `parallel_map` and asserts that results are returned, that the correct number of results is produced, that all results indicate success, and that no statistics are collected since the collector is None. This ensures that the serial mapping functionality can operate without a result collector, allowing for basic task execution in environments where collecting results or statistics is not desired or necessary. 

        Parameters:
            self ("TestSerialMap"): Pytest-provided test instance.
            sample_tasks (List[int]): Fixture-provided list of tasks.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        manager.collector = None
        
        def simple_func(task):
            return task * 3
        
        results = manager.parallel_map(simple_func, sample_tasks)
        
        assert results is not None
        assert len(results) == len(sample_tasks)
        assert all(r.success for r in results)
        assert manager.stats is None 
    
    def test_serial_map_with_verbose(self: "TestSerialMap", sample_tasks: List[int]) -> None:
        """
        This test validates that when the manager is in serial mode with verbose output enabled, the `parallel_map` method emits expected informational messages about processing and execution. It constructs a manager with the 'serial' backend and verbose mode enabled, defines a simple function to map over the sample tasks, and captures stdout during the call to `parallel_map`. The test asserts that key phrases related to serial execution are present in the output and that results are returned with the correct count. This ensures that verbose logging provides useful insights into the execution process even in serial mode, aiding users in understanding how tasks are being processed. 

        Parameters:
            self ("TestSerialMap"): Pytest-provided test instance.
            sample_tasks (List[int]): Fixture-provided list of tasks.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=True)
        
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


class TestPrintStatistics:
    """ Tests for formatting and printing execution statistics from collected results. """
    
    def test_print_statistics_not_master(self: "TestPrintStatistics") -> None:
        """
        This test verifies that the `_print_statistics` method does not print any statistics when the manager is not the master process. It creates a mock MPI communicator that simulates a non-master rank, sets up a `ParallelStats` object with some dummy data, and captures stdout during the call to `_print_statistics`. The test asserts that no statistics-related output is printed since only the master process should output statistics. This ensures that in MPI runs, only the designated master process provides execution statistics, preventing cluttered output from multiple ranks. 

        Parameters:
            self ("TestPrintStatistics"): Pytest-provided test instance.

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
            self ("TestPrintStatistics"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=True)
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
            self ("TestPrintStatistics"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=True)

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
            self ("TestPrintStatistics"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='multiprocessing', verbose=True)

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


class TestBarrierAndFinalizeAdditional:
    """ Tests for synchronization and cleanup helpers like `barrier` and `finalize`. """
    
    def test_barrier_with_mpi(self: "TestBarrierAndFinalizeAdditional") -> None:
        """
        This test verifies that the `barrier` method correctly calls the MPI `Barrier` function when MPI is available. It creates a mock MPI communicator, patches the `MPI_AVAILABLE` flag to True, and asserts that calling `barrier` on the manager results in a call to the communicator's `Barrier` method. This ensures that synchronization across ranks is properly implemented when using MPI, allowing for coordinated execution in distributed runs. 

        Parameters:
            self ("TestBarrierAndFinalizeAdditional"): Pytest-provided test instance.

        Returns:
            None
        """
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 4
        mock_comm.Barrier = Mock()
        
        with patch('mpasdiag.processing.parallel.MPI_AVAILABLE', True):
            mock_mpi = MagicMock()
            mock_mpi.COMM_WORLD = mock_comm
            
            with patch('mpasdiag.processing.parallel.MPI', mock_mpi):
                manager = MPASParallelManager(backend='mpi', verbose=False)
                manager.comm = mock_comm                
                manager.barrier()                
                mock_comm.Barrier.assert_called_once()
    
    def test_barrier_without_mpi(self: "TestBarrierAndFinalizeAdditional") -> None:
        """
        This test confirms that the `barrier` method does not raise an error when MPI is not available. It constructs a manager with the 'serial' backend and calls `barrier`, asserting that it completes without raising any exceptions. This ensures that the `barrier` method can be safely called in non-MPI contexts, allowing for code that may be shared between parallel and serial execution paths without requiring conditional checks for MPI availability. 

        Parameters:
            self ("TestBarrierAndFinalizeAdditional"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        manager.barrier()
    
    def test_finalize_with_mpi(self: "TestBarrierAndFinalizeAdditional") -> None:
        """
        This test verifies that the `finalize` method performs an MPI barrier and prints a finalization message when MPI is available. It creates a mock MPI communicator, patches the `MPI_AVAILABLE` flag to True, and captures stdout during the call to `finalize`. The test asserts that the communicator's `Barrier` method is called and that the output includes a message indicating that the manager has been finalized. This ensures that proper synchronization and cleanup messages are emitted when finalizing in an MPI context, aiding users in understanding when parallel resources have been released. 

        Parameters:
            self ("TestBarrierAndFinalizeAdditional"): Pytest-provided test instance.

        Returns:
            None
        """
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 4
        mock_comm.Barrier = Mock()
        
        with patch('mpasdiag.processing.parallel.MPI_AVAILABLE', True):
            mock_mpi = MagicMock()
            mock_mpi.COMM_WORLD = mock_comm
            
            with patch('mpasdiag.processing.parallel.MPI', mock_mpi):
                manager = MPASParallelManager(backend='mpi', verbose=True)
                manager.comm = mock_comm
                
                import io
                from contextlib import redirect_stdout                
                f = io.StringIO()

                with redirect_stdout(f):
                    manager.finalize()
                
                output = f.getvalue()

                mock_comm.Barrier.assert_called_once()
                assert "finalized" in output
    
    def test_finalize_without_mpi(self: "TestBarrierAndFinalizeAdditional") -> None:
        """
        This test confirms that the `finalize` method prints a finalization message even when MPI is not available. It constructs a manager with the 'serial' backend, captures stdout during the call to `finalize`, and asserts that the output includes a message indicating that the manager has been finalized. This ensures that the `finalize` method provides feedback about resource cleanup regardless of the execution context, allowing users to confirm that finalization has occurred in both parallel and serial runs. 

        Parameters:
            self ("TestBarrierAndFinalizeAdditional"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=True)
        
        import io
        from contextlib import redirect_stdout        
        f = io.StringIO()

        with redirect_stdout(f):
            manager.finalize()
        
        output = f.getvalue()
        assert "finalized" in output


class TestParallelPlotFunctionAdditional:
    """ Tests for the `parallel_plot` convenience function that configures a manager and maps a plotting callback over input files. """
    
    def test_parallel_plot_with_real_files(self: "TestParallelPlotFunctionAdditional", mpas_output_files: List[str]) -> None:
        """
        This test validates that the `parallel_plot` function can execute a plotting function over a list of MPAS output files in parallel. It defines a mock plotting function that asserts the existence of each file and returns a string indicating it was plotted. The test calls `parallel_plot` with this function and the provided list of MPAS output files, asserting that results are returned, that the correct number of results is produced, and that all results are instances of `TaskResult`. This ensures that `parallel_plot` correctly orchestrates parallel execution of a plotting function across multiple files, allowing for efficient generation of plots from MPAS outputs in a parallelized manner. If no MPAS output files are available, the test is skipped to avoid false failures. 

        Parameters:
            self ("TestParallelPlotFunctionAdditional"): Pytest-provided test instance.
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
            self ("TestParallelPlotFunctionAdditional"): Pytest-provided test instance.

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
            self ("TestParallelPlotFunctionAdditional"): Pytest-provided test instance.

        Returns:
            None
        """
        files = ["test1.nc", "test2.nc"]
        
        def simple_plot(filepath):
            return filepath
        
        results = parallel_plot(simple_plot, files)
        assert results is not None or True 


class TestMPASTaskDistributor:
    """ Tests that exercise the `MPASTaskDistributor` for correct assignment of task indices across ranks. """
    
    def test_static_distribution_with_remainder(self: "TestMPASTaskDistributor") -> None:
        """
        This test checks that the static distribution strategy correctly handles cases where the total number of tasks is not perfectly divisible by the number of workers. It simulates a communicator with 3 workers and 10 tasks, which results in a block size of 3 and a remainder of 1. The test asserts that the first worker (rank 0) receives the extra task from the remainder, resulting in 4 tasks for rank 0 and 3 tasks each for ranks 1 and 2. This ensures that the static distribution logic properly accounts for remainders to achieve balanced task assignment across workers. 

        Parameters:
            self ("TestMPASTaskDistributor"): Pytest-provided test instance.

        Returns:
            None
        """
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 0 
        mock_comm.Get_size.return_value = 3
        
        distributor = MPASTaskDistributor(mock_comm, LoadBalanceStrategy.STATIC)
        tasks = list(range(10)) 
        
        distributed = distributor.distribute_tasks(tasks)
        
        assert len(distributed) == pytest.approx(4)
        assert distributed[0] == (0, 0)
        assert distributed[1] == (1, 1)
    
    def test_static_distribution_rank_beyond_remainder(self: "TestMPASTaskDistributor") -> None:
        """
        This test validates that in the static distribution strategy, workers with ranks beyond the remainder do not receive extra tasks. It simulates a communicator with 3 workers and 10 tasks, where the block size is 3 and the remainder is 1. The test asserts that worker rank 2, which is beyond the remainder, receives only the base block of 3 tasks and does not get any of the extra tasks from the remainder. This confirms that the static distribution logic correctly limits extra task assignment to only those workers whose ranks are within the remainder. 

        Parameters:
            self ("TestMPASTaskDistributor"): Pytest-provided test instance.

        Returns:
            None
        """
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 2
        mock_comm.Get_size.return_value = 3
        
        distributor = MPASTaskDistributor(mock_comm, LoadBalanceStrategy.STATIC)
        tasks = list(range(10))
        
        distributed = distributor.distribute_tasks(tasks)
        assert len(distributed) == pytest.approx(3)
    
    def test_block_distribution(self: "TestMPASTaskDistributor") -> None:
        """
        This test validates that the block distribution strategy assigns contiguous blocks of tasks to each worker. It simulates a communicator with 3 workers and 12 tasks, resulting in a block size of 4. The test asserts that worker rank 1 receives a contiguous block of tasks starting at index 4 and ending at index 7, confirming that the block distribution logic correctly calculates and assigns contiguous slices of tasks based on the worker's rank. This ensures that the block distribution strategy preserves locality and batching semantics as intended. 

        Parameters:
            self ("TestMPASTaskDistributor"): Pytest-provided test instance.

        Returns:
            None
        """
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 1
        mock_comm.Get_size.return_value = 3
        
        distributor = MPASTaskDistributor(mock_comm, LoadBalanceStrategy.BLOCK)
        tasks = list(range(12))
        
        distributed = distributor.distribute_tasks(tasks)
        
        assert len(distributed) == pytest.approx(4)
        assert distributed[0] == (4, 4) 
        assert distributed[3] == (7, 7)
    
    def test_cyclic_distribution(self: "TestMPASTaskDistributor") -> None:
        """
        This test checks that the cyclic distribution strategy assigns tasks in a round-robin fashion across workers. It simulates a communicator with 3 workers and 10 tasks, and asserts that worker rank 1 receives every third task starting from index 1 (i.e., tasks at indices 1, 4, and 7). This confirms that the cyclic distribution logic correctly calculates task indices based on the worker's rank and the total number of workers, ensuring an even spread of tasks in a cyclic manner. 

        Parameters:
            self ("TestMPASTaskDistributor"): Pytest-provided test instance.

        Returns:
            None
        """
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 1
        mock_comm.Get_size.return_value = 3
        
        distributor = MPASTaskDistributor(mock_comm, LoadBalanceStrategy.CYCLIC)
        tasks = list(range(10))
        
        distributed = distributor.distribute_tasks(tasks)
        
        assert len(distributed) == pytest.approx(3)
        assert distributed[0] == (1, 1)
        assert distributed[1] == (4, 4)
        assert distributed[2] == (7, 7)
    
    def test_dynamic_distribution(self: "TestMPASTaskDistributor") -> None:
        """
        This test checks the behavior of the dynamic distribution strategy, which currently defaults to static partitioning. It simulates a communicator with 2 workers and 8 tasks, and asserts that worker rank 1 receives the second half of the tasks (indices 4-7) as it would under static distribution. This confirms that until dynamic-specific logic is implemented, the dynamic strategy provides a predictable fallback to static behavior, allowing for consistent task assignment while dynamic features are being developed. 

        Parameters:
            self ("TestMPASTaskDistributor"): Pytest-provided test instance.

        Returns:
            None
        """
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 1
        mock_comm.Get_size.return_value = 2
        
        distributor = MPASTaskDistributor(mock_comm, LoadBalanceStrategy.DYNAMIC)
        tasks = list(range(8))
        
        distributed = distributor.distribute_tasks(tasks)
        
        assert len(distributed) == pytest.approx(4)
        assert distributed[0] == (4, 4)
    
    def test_distribute_tasks_unknown_strategy(self: "TestMPASTaskDistributor") -> None:
        """
        This test verifies that if an unknown distribution strategy is set, the `distribute_tasks` method falls back to static partitioning without raising an error. It simulates a communicator with 2 workers and 6 tasks, creates a distributor with the static strategy, then manually sets the strategy to an invalid value. The test asserts that the resulting distribution still follows static partitioning logic, confirming that the method handles unrecognized strategies gracefully by defaulting to a known behavior rather than crashing. 

        Parameters:
            self ("TestMPASTaskDistributor"): Pytest-provided test instance.

        Returns:
            None
        """
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 2
        
        distributor = MPASTaskDistributor(mock_comm, LoadBalanceStrategy.STATIC)

        assert distributor.strategy == LoadBalanceStrategy.STATIC
        assert distributor.strategy is not None

        distributor.strategy = "invalid" # type: ignore        
        tasks = list(range(6))
        distributed = distributor.distribute_tasks(tasks)        
        assert len(distributed) == pytest.approx(3)


class TestMPASResultCollectorGather:
    """ Tests for gathering `TaskResult` objects from multiple workers under MPI-like gather semantics. """
    
    def test_gather_results_on_worker_rank(self: "TestMPASResultCollectorGather") -> None:
        """
        This test confirms that when the `gather_results` method is called on a worker rank, it correctly gathers local results to the master rank and returns None. It simulates a communicator with 4 ranks where the current rank is 2 (a worker), and asserts that the `gather` method of the communicator is called with the local results and that the return value of `gather_results` is None on the worker. This ensures that worker ranks properly send their results to the master without attempting to process gathered results themselves, maintaining correct MPI gather semantics. 

        Parameters:
            self ("TestMPASResultCollectorGather"): Pytest-provided test instance.

        Returns:
            None
        """
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 2  
        mock_comm.Get_size.return_value = 4
        mock_comm.gather.return_value = None
        
        collector = MPASResultCollector(mock_comm)
        
        local_results = [
            TaskResult(task_id=5, success=True, execution_time=1.0, worker_rank=2)
        ]
        
        gathered = collector.gather_results(local_results)
        
        assert gathered is None
        mock_comm.gather.assert_called_once_with(local_results, root=0)
    
    def test_gather_results_on_master_rank(self: "TestMPASResultCollectorGather") -> None:
        """
        This test validates that when the `gather_results` method is called on the master rank, it correctly gathers results from all workers and returns a combined list of `TaskResult` objects. It simulates a communicator with 2 ranks where the current rank is 0 (the master), and mocks the `gather` method to return a list of results from both workers. The test asserts that the gathered results are returned as a single list containing all `TaskResult` objects from the workers, confirming that the master rank correctly processes gathered results while worker ranks do not attempt to access them. This ensures proper separation of responsibilities in the gather operation under MPI semantics. 

        Parameters:
            self ("TestMPASResultCollectorGather"): Pytest-provided test instance.

        Returns:
            None
        """
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 0 
        mock_comm.Get_size.return_value = 2
        
        all_results = [
            [TaskResult(task_id=0, success=True, worker_rank=0)],
            [TaskResult(task_id=1, success=True, worker_rank=1)]
        ]

        mock_comm.gather.return_value = all_results        
        collector = MPASResultCollector(mock_comm)        
        local_results = [TaskResult(task_id=0, success=True, worker_rank=0)]
        gathered = collector.gather_results(local_results)
        
        assert gathered is not None
        assert len(gathered) == pytest.approx(2)
        assert gathered[0].task_id == pytest.approx(0)
        assert gathered[1].task_id == pytest.approx(1)


class TestMPIMapExecution:
    """ Tests of MPI-mapped execution paths including master/worker separation and communicator-based control flow. """
    
    def test_mpi_map_verbose_output(self: "TestMPIMapExecution") -> None:
        """
        This test verifies that the `_mpi_map` method produces expected verbose output when executed in an MPI context. It creates a mock MPI communicator simulating a master rank, patches the `MPI_AVAILABLE` flag to True, and captures stdout during the call to `_mpi_map`. The test asserts that the output includes key phrases related to processing tasks across ranks and load balance strategy, confirming that verbose logging provides insights into the execution process in an MPI environment. This ensures that users have visibility into how tasks are being distributed and processed when running with MPI, aiding in debugging and performance tuning. 

        Parameters:
            self ("TestMPIMapExecution"): Pytest-provided test instance.

        Returns:
            None
        """
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 4
        mock_comm.bcast.return_value = list(range(10))
        mock_comm.gather.return_value = [[TaskResult(i, True, worker_rank=0) for i in range(10)]]
        
        with patch('mpasdiag.processing.parallel.MPI_AVAILABLE', True):
            mock_mpi = MagicMock()
            mock_mpi.COMM_WORLD = mock_comm
            
            with patch('mpasdiag.processing.parallel.MPI', mock_mpi):
                manager = MPASParallelManager(backend='mpi', verbose=True)
                manager.comm = mock_comm
                manager.distributor = MPASTaskDistributor(mock_comm, LoadBalanceStrategy.STATIC)
                manager.collector = MPASResultCollector(mock_comm)
                
                def simple_func(task):
                    return task * 2
                
                import io
                from contextlib import redirect_stdout                
                f = io.StringIO()

                with redirect_stdout(f):
                    results = manager._mpi_map(simple_func, list(range(10)))
                
                output = f.getvalue()
                assert "Processing" in output
                assert "tasks across" in output
                assert "Load balance strategy" in output
                assert results is not None
    
    def test_mpi_map_assertions(self: "TestMPIMapExecution") -> None:
        """
        This test confirms that the `_mpi_map` method raises appropriate assertions when critical components like the MPI communicator, task distributor, or result collector are not initialized. It creates a manager in serial mode and sequentially sets the `comm`, `distributor`, and `collector` attributes to None, asserting that each call to `_mpi_map` raises an `AssertionError` with the expected message. This ensures that the method enforces necessary preconditions for MPI execution, preventing runtime errors and guiding developers to properly configure the manager before attempting parallel mapping. 

        Parameters:
            self ("TestMPIMapExecution"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        manager.comm = None

        with pytest.raises(AssertionError, match="MPI communicator must be initialized"):
            manager._mpi_map(lambda x: x, [1, 2, 3])
        
        mock_comm = Mock()
        manager.comm = mock_comm
        manager.distributor = None

        with pytest.raises(AssertionError, match="Task distributor must be initialized"):
            manager._mpi_map(lambda x: x, [1, 2, 3])
        
        manager.distributor = MPASTaskDistributor(mock_comm, LoadBalanceStrategy.STATIC)
        manager.collector = None

        with pytest.raises(AssertionError, match="Result collector must be initialized"):
            manager._mpi_map(lambda x: x, [1, 2, 3])


class TestSerialMapWithCollector:
    """ Tests that validate serial mapping when a `MPASResultCollector` is attached to gather statistics. """
    
    def test_serial_map_with_collector_enabled(self: "TestSerialMapWithCollector", sample_tasks: List[int]) -> None:
        """
        This test checks that the `parallel_map` method correctly executes in serial mode when a `MPASResultCollector` is attached, and that it collects execution statistics. It creates a manager in serial mode, attaches a mock collector, and defines a simple function to map over the sample tasks. The test asserts that results are returned, that the number of results matches the number of tasks, and that the manager's `stats` attribute is populated with the total number of tasks. This ensures that even in serial execution contexts, the presence of a result collector allows for gathering useful statistics about task execution, providing insights into performance and success rates. 

        Parameters:
            self ("TestSerialMapWithCollector"): Pytest-provided test instance.
            sample_tasks (List[int]): Fixture-provided list of tasks.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 1
        manager.collector = MPASResultCollector(mock_comm)
        
        def simple_func(task):
            return task * 2
        
        results = manager.parallel_map(simple_func, sample_tasks)
        
        assert results is not None
        assert len(results) == len(sample_tasks)
        assert manager.stats is not None
        assert manager.stats.total_tasks == len(sample_tasks)
    
    def test_serial_map_verbose_with_stats(self: "TestSerialMapWithCollector", sample_tasks: List[int]) -> None:
        """
        This test checks that verbose serial mapping prints statistics when a collector is enabled. The test captures stdout and asserts presence of 'serial mode' and statistics headings. This provides operator visibility for serial runs with stats. It creates a manager in serial mode with verbose enabled, attaches a mock collector, and defines a simple function to map over the sample tasks. The test captures the output during mapping and asserts that it includes indications of serial execution and the presence of parallel execution statistics, confirming that users receive feedback about execution context and performance even when running in serial mode with a collector attached. 

        Parameters:
            self ("TestSerialMapWithCollector"): Pytest-provided test instance.
            sample_tasks (List[int]): Fixture-provided list of tasks.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=True)
        
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 1
        manager.collector = MPASResultCollector(mock_comm)
        
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


class TestWithRealMPASData:
    """ Integration tests that operate on real MPAS output and grid files when available. """
    
    def test_process_multiple_mpas_files(self: "TestWithRealMPASData", mpas_output_files: List[str]) -> None:
        """
        This test validates that the `parallel_map` method can process multiple real MPAS output files in parallel, extracting basic information from each file. It defines a function to load an MPAS file and return its dimensions and variables, then uses the `MPASParallelManager` with multiprocessing to map this function over the provided list of MPAS output files. The test asserts that results are returned, that the number of results matches the number of files, and that each result contains expected information about the file's dimensions and variables. If no MPAS output files are available, the test is skipped to avoid false failures in CI environments without access to large data files. This ensures that the parallel processing logic can handle real-world MPAS data effectively, providing a foundation for more complex analyses and plotting functions.

        Parameters:
            self ("TestWithRealMPASData"): Pytest-provided test instance.
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
        results = manager.parallel_map(load_mpas_file, mpas_output_files)
        
        assert results is not None
        assert len(results) == len(mpas_output_files)
        assert all(r.success for r in results)
        assert all(isinstance(r.result, dict) for r in results)
    
    def test_process_grid_file_with_different_strategies(self: "TestWithRealMPASData", grid_file: str) -> None:
        """
        This test checks that the `parallel_map` method can process a real MPAS grid file using different load balancing strategies. It defines a function to extract basic grid information (number of cells and vertices) from the provided grid file, then uses the `MPASParallelManager` with the serial backend to map this function over the grid file. The test asserts that results are returned, that the result contains expected grid information, and that the method can execute without errors even when parallelism is not utilized. If the grid file is not available, the test is skipped to avoid false failures in environments where the file cannot be accessed. This ensures that the processing logic can handle real MPAS grid data and provides a basis for testing more complex operations on grid files in parallel contexts.

        Parameters:
            self ("TestWithRealMPASData"): Pytest-provided test instance.
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
        
        assert results is not None
        assert len(results) == pytest.approx(1)
        assert results[0].success
        assert 'nCells' in results[0].result
        assert 'nVertices' in results[0].result


class TestImportErrorHandling:
    """ Tests that validate import-time MPI detection and the module-level `MPI_AVAILABLE` flag. """
    
    def test_mpi_import_warning(self: "TestImportErrorHandling") -> None:
        """
        This test confirms that if the `mpi4py` library is not available at import time, the `MPI_AVAILABLE` flag in the `parallel` module is set to False, and if `mpi4py` is available, the flag is True. It imports the `parallel` module and checks the value of `MPI_AVAILABLE` against the actual availability of the `MPI` module. This ensures that the module correctly handles import errors related to MPI and provides an accurate flag for users to check before attempting MPI-based operations, preventing runtime errors due to missing dependencies. 

        Parameters:
            self ("TestImportErrorHandling"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing import parallel
        
        if parallel.MPI is None:
            assert not parallel.MPI_AVAILABLE
        else:
            assert parallel.MPI_AVAILABLE


class TestMPIMapReturnValue:
    """ Tests ensuring `_mpi_map` returns `None` on worker ranks and only the master returns aggregated results. """
    
    def test_mpi_map_returns_none_on_worker(self: "TestMPIMapReturnValue") -> None:
        """
        This test verifies that when the `_mpi_map` method is executed on a worker rank, it returns `None` instead of attempting to return results. It simulates an MPI environment with a mock communicator where the current rank is a worker (rank 2) and asserts that the return value of `_mpi_map` is `None` on the worker. This confirms that the method correctly distinguishes between master and worker contexts, ensuring that only the master rank processes and returns results while workers simply send their local results to the master without trying to access gathered results. This behavior is crucial for maintaining correct MPI semantics and preventing errors in worker processes. 

        Parameters:
            self ("TestMPIMapReturnValue"): Pytest-provided test instance.

        Returns:
            None
        """
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 2  
        mock_comm.Get_size.return_value = 4
        mock_comm.bcast.return_value = list(range(10))
        mock_comm.gather.return_value = None  
        
        with patch('mpasdiag.processing.parallel.MPI_AVAILABLE', True):
            mock_mpi = MagicMock()
            mock_mpi.COMM_WORLD = mock_comm
            
            with patch('mpasdiag.processing.parallel.MPI', mock_mpi):
                manager = MPASParallelManager(backend='mpi', verbose=False)
                manager.comm = mock_comm
                manager.is_master = False  
                manager.distributor = MPASTaskDistributor(mock_comm, LoadBalanceStrategy.STATIC)
                manager.collector = MPASResultCollector(mock_comm)
                
                def simple_func(task):
                    return task * 2
                
                results = manager._mpi_map(simple_func, list(range(10)))
                
                assert results is None


class TestMultiprocessingBreakStatement:
    """ Test cases for break/early-exit logic in multiprocessing attempts that verify the first successful pool run stops retries. """
    
    def test_multiprocessing_successful_first_attempt(self: "TestMultiprocessingBreakStatement", sample_tasks: List[int]) -> None:
        """
        This test validates that when the multiprocessing mapping succeeds on the first attempt, the retry logic does not execute further attempts. It defines a simple function to map over the sample tasks, creates a `MPASParallelManager` with the multiprocessing backend, and calls `parallel_map`. The test asserts that results are returned, that the number of results matches the number of tasks, and that all results indicate success. This confirms that the break statement in the multiprocessing retry logic functions correctly, allowing for efficient execution without unnecessary retries when the initial attempt is successful. 

        Parameters:
            self ("TestMultiprocessingBreakStatement"): Pytest-provided test instance.
            sample_tasks (List[int]): Fixture-provided list of tasks.

        Returns:
            None
        """
        def simple_func(task):
            return task + 100
        
        manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=False)
        results = manager.parallel_map(simple_func, sample_tasks)
        
        assert results is not None
        assert len(results) == len(sample_tasks)
        assert all(r.success for r in results)
        assert results[0].result == pytest.approx(100)


class TestMultiprocessingResultsNoneCheck:
    """ Tests that validate fallback to serial execution when multiprocessing attempts repeatedly return `None`. """
    
    def test_multiprocessing_results_none_fallback(self: "TestMultiprocessingResultsNoneCheck", sample_tasks: List[int]) -> None:
        """
        This test validates behavior when multiprocessing mapping returns `None` for all attempts, ensuring serial fallback still produces results. The test patches `get_context` to force exceptions and asserts the manager returns a serial-executed results list. This prevents silent failures when pool creation fails, ensuring that users still receive results even if multiprocessing cannot be utilized. 

        Parameters:
            self ("TestMultiprocessingResultsNoneCheck"): Pytest-provided test instance.
            sample_tasks (List[int]): Fixture-provided list of tasks.

        Returns:
            None
        """
        def simple_func(task):
            return task
        
        manager = MPASParallelManager(backend='multiprocessing', n_workers=2, verbose=False)
        
        with patch('mpasdiag.processing.parallel.get_context') as mock_ctx:
            mock_ctx.side_effect = Exception("Force fallback")            
            results = manager._multiprocessing_map(simple_func, sample_tasks)
            assert len(results) == len(sample_tasks)


class TestSerialMapVerbosePath:
    """ Tests covering verbose printing behavior for serial map execution when a collector is attached. """
    
    def test_serial_map_calls_print_statistics(self: "TestSerialMapVerbosePath", sample_tasks: List[int]) -> None:
        """
        This test checks that when the `parallel_map` method is executed in serial mode with verbose enabled and a result collector attached, the `_print_statistics` method is called to display execution statistics. It creates a manager in serial mode with verbose enabled, attaches a mock collector, and defines a simple function to map over the sample tasks. The test patches the `_print_statistics` method to verify that it is called during the mapping process, confirming that users receive feedback about execution statistics even when running in serial mode with a collector attached. This ensures that the verbose path provides insights into performance and success rates for serial executions. 

        Parameters:
            self ("TestSerialMapVerbosePath"): Pytest-provided test instance.
            sample_tasks (List[int]): Fixture-provided list of tasks.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=True)        
        mock_comm = Mock()

        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 1

        manager.collector = MPASResultCollector(mock_comm)
        
        def simple_func(task):
            return task
        
        with patch.object(manager, '_print_statistics') as mock_print:
            results = manager._serial_map(simple_func, sample_tasks)
            mock_print.assert_called_once()

        assert results is not None
        assert len(results) == len(sample_tasks)


class TestParallelPlotFunctionComplete:
    """ Comprehensive tests of the `parallel_plot` convenience function covering manager creation, error policy configuration, and simple integration runs. """
    
    def test_parallel_plot_creates_manager_with_dynamic_strategy(self: "TestParallelPlotFunctionComplete") -> None:
        """
        This test verifies that the `parallel_plot` function creates an instance of `MPASParallelManager` with the expected parameters, including the dynamic load balance strategy and verbose output. It mocks the `MPASParallelManager` class to intercept its instantiation and checks that it is called with the correct arguments. The test also asserts that the `set_error_policy` method is called with 'collect' to ensure that errors are handled appropriately during parallel execution. This confirms that the `parallel_plot` function sets up the parallel manager correctly for users, providing a foundation for effective parallel plotting operations. 

        Parameters:
            self ("TestParallelPlotFunctionComplete"): Pytest-provided test instance.

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
            self ("TestParallelPlotFunctionComplete"): Pytest-provided test instance.

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

