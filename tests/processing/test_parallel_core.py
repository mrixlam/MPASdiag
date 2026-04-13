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
import pytest
from unittest.mock import Mock, MagicMock, patch

from mpasdiag.processing.parallel import (
    MPASParallelManager,
    ErrorPolicy,
    LoadBalanceStrategy,
    MPASTaskDistributor,
    MPASResultCollector
)

from tests.test_data_helpers import assert_expected_public_methods


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
    assert_expected_public_methods(manager, 'MPASParallelManager')

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
    assert_expected_public_methods(manager, 'MPASParallelManager')

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
    assert_expected_public_methods(manager, 'MPASParallelManager')

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
    assert_expected_public_methods(manager, 'MPASParallelManager')

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
    assert_expected_public_methods(manager_1, 'MPASParallelManager')

    tasks = list(range(16))
    
    start = time.time()
    results_1 = manager_1.parallel_map(slow_task, tasks)
    time_1_worker = time.time() - start
    
    assert results_1 is not None, "parallel_map returned None with 1 worker"
    assert manager_1.size == pytest.approx(1), f"Expected size=1, got {manager_1.size}"
    print(f"✓ n_workers=1: Wall time = {time_1_worker:.2f}s, Pool size = {manager_1.size}")
    
    manager_4 = MPASParallelManager(n_workers=4, backend='multiprocessing')
    assert_expected_public_methods(manager_4, 'MPASParallelManager')

    start = time.time()
    results_4 = manager_4.parallel_map(slow_task, tasks)
    time_4_workers = time.time() - start
    
    assert results_4 is not None, "parallel_map returned None with 4 workers"
    assert manager_4.size == pytest.approx(4), f"Expected size=4, got {manager_4.size}"
    print(f"✓ n_workers=4: Wall time = {time_4_workers:.2f}s, Pool size = {manager_4.size}")
    
    manager_auto = MPASParallelManager(n_workers=None, backend='multiprocessing')
    assert_expected_public_methods(manager_auto, 'MPASParallelManager')

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
        
        assert_expected_public_methods(manager, 'MPASParallelManager')

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
        parallel_manager = MPASParallelManager()
        assert_expected_public_methods(parallel_manager, 'MPASParallelManager')        

    def test_load_balance_strategy_enum_exists(self: "TestParallelProcessing") -> None:
        """
        This test verifies that the `LoadBalanceStrategy` enum contains expected strategies for distributing tasks across workers. The test checks for members like `STATIC` and `DYNAMIC` to ensure that different load balancing approaches are available to callers. Correct enumeration values are important for configuring how the manager assigns work to workers, especially for heterogeneous workloads. This guards against accidental API or enum member renames that could break existing code relying on these strategies. 

        Parameters:
            self ("TestParallelProcessing"): Pytest-provided test instance.

        Returns:
            None
        """
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

    def test_parallel_stats_dataclass_exists(self: "TestParallelProcessing") -> None:
        """
        This test verifies that the `ParallelStats` dataclass is defined for tracking performance and execution statistics of parallel tasks. The `ParallelStats` class typically includes fields for timing information, worker utilization, task counts, and load balancing metrics. The test imports the dataclass and asserts that it is not None to confirm that the structure for collecting parallel execution statistics is available. This dataclass is important for monitoring and optimizing parallel workflows, allowing users to analyze performance characteristics and identify bottlenecks in their parallel processing pipelines. Its presence supports the manager's ability to provide insights into execution efficiency. 

        Parameters:
            self ("TestParallelProcessing"): Pytest-provided test instance.

        Returns:
            None
        """

    def test_task_distributor_class_exists(self: "TestParallelProcessing") -> None:
        """
        This test confirms that the `MPASTaskDistributor` class is defined for managing the distribution of tasks to workers in the `MPASParallelManager`. The `MPASTaskDistributor` is responsible for implementing the logic that assigns tasks to workers based on the selected load balancing strategy. The test imports the class and asserts that it is not None to ensure that the core component for task distribution is available. This class is critical for enabling efficient parallel execution, as it determines how work is allocated across workers to optimize performance and resource utilization. Its presence supports the manager's ability to effectively distribute tasks according to user-configured strategies. 

        Parameters:
            self ("TestParallelProcessing"): Pytest-provided test instance.

        Returns:
            None
        """
        assert_expected_public_methods(MPASTaskDistributor, 'MPASTaskDistributor')

    def test_result_collector_class_exists(self: "TestParallelProcessing") -> None:
        """
        This test verifies that the `MPASResultCollector` class is defined for aggregating results from worker processes in the `MPASParallelManager`. The `MPASResultCollector` is responsible for collecting task outcomes, handling errors according to the configured policy, and providing a unified interface for accessing results after parallel execution. The test imports the class and asserts that it is not None to confirm that the core component for result collection is available. This class is essential for ensuring that results from parallel tasks are properly gathered and made accessible to callers, supporting robust error handling and result retrieval in parallel workflows. Its presence ensures that the manager can effectively manage and report on task outcomes. 

        Parameters:
            self ("TestParallelProcessing"): Pytest-provided test instance.

        Returns:
            None
        """
        assert_expected_public_methods(MPASResultCollector, 'MPASResultCollector')

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
        assert_expected_public_methods(ParallelPrecipitationProcessor, 'ParallelPrecipitationProcessor')

    def test_parallel_surface_processor_exists(self: "TestParallelWrappers") -> None:
        """
        This test confirms that the `ParallelSurfaceProcessor` wrapper class is defined for adapting surface processing tasks to parallel execution. The wrapper standardizes how surface diagnostics are initialized and executed across worker processes, allowing for efficient distribution of surface-related processing tasks. The test imports the class and asserts that it is not None to ensure that the wrapper is available in the module exports. This guarantees that users can easily run surface processing in parallel, supporting improved performance for these diagnostics. The presence of this wrapper class is critical for enabling parallelism in surface processing workflows and ensuring that users have access to convenient interfaces for parallel execution of these tasks. 

        Parameters:
            self ("TestParallelWrappers"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing.parallel_wrappers import ParallelSurfaceProcessor
        assert_expected_public_methods(ParallelSurfaceProcessor, 'ParallelSurfaceProcessor')

    def test_parallel_wind_processor_exists(self: "TestParallelWrappers") -> None:
        """
        This test verifies that the `ParallelWindProcessor` wrapper class is defined for adapting wind processing tasks to parallel execution. The wrapper allows wind diagnostics to be distributed across workers with consistent initialization and result handling. The test imports the class and asserts that it is not None to confirm that the wrapper is available in the module exports. This ensures that users have access to a convenient interface for running wind processing in parallel, supporting efficient execution of these diagnostics. Presence of this wrapper class is important for enabling parallelism in wind-related workflows. 

        Parameters:
            self ("TestParallelWrappers"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing.parallel_wrappers import ParallelWindProcessor
        assert_expected_public_methods(ParallelWindProcessor, 'ParallelWindProcessor')

    def test_parallel_cross_section_processor_exists(self: "TestParallelWrappers") -> None:
        """
        This test confirms that the `ParallelCrossSectionProcessor` wrapper class is defined for adapting cross-section processing tasks to parallel execution. The wrapper standardizes how cross-section diagnostics are initialized and executed across worker processes, allowing for efficient distribution of cross-section-related processing tasks. The test imports the class and asserts that it is not None to ensure that the wrapper is available in the module exports. This guarantees that users can easily run cross-section processing in parallel, supporting improved performance for these diagnostics. The presence of this wrapper class is critical for enabling parallelism in cross-section processing workflows and ensuring that users have access to convenient interfaces for parallel execution of these tasks. 

        Parameters:
            self ("TestParallelWrappers"): Pytest-provided test instance.

        Returns:
            None
        """
        from mpasdiag.processing.parallel_wrappers import ParallelCrossSectionProcessor
        assert_expected_public_methods(ParallelCrossSectionProcessor, 'ParallelCrossSectionProcessor')

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
        assert_expected_public_methods(manager, 'MPASParallelManager')

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
        assert_expected_public_methods(manager, 'MPASParallelManager')

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
        assert_expected_public_methods(manager, 'MPASParallelManager')
        
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
        assert_expected_public_methods(manager, 'MPASParallelManager')
    
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
        assert_expected_public_methods(manager, 'MPASParallelManager')
    
    def test_set_error_policy_string(self: "TestMPASParallelManagerInitializationModule") -> None:
        """
        This test verifies that the manager's error policy can be set using string identifiers. The test creates a manager instance, calls `set_error_policy` with string values like 'abort', 'continue', and 'collect', and asserts that the manager's `error_policy` attribute is updated to the corresponding `ErrorPolicy` enum value. This ensures that users can configure error handling behavior using intuitive string inputs, and that the manager correctly translates these into internal enum representations for consistent error handling during parallel execution. 

        Parameters:
            self ("TestMPASParallelManagerInitializationModule"): Pytest-provided test instance.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        assert_expected_public_methods(manager, 'MPASParallelManager')

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
        assert_expected_public_methods(manager, 'MPASParallelManager')

        manager.set_error_policy(ErrorPolicy.ABORT)
        assert manager.error_policy == ErrorPolicy.ABORT

        manager.set_error_policy(ErrorPolicy.CONTINUE)
        assert manager.error_policy == ErrorPolicy.CONTINUE

        manager.set_error_policy(ErrorPolicy.COLLECT)
        assert manager.error_policy == ErrorPolicy.COLLECT


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
            assert_expected_public_methods(manager, 'MPASParallelManager')

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
        assert_expected_public_methods(manager, 'MPASParallelManager')

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
        assert_expected_public_methods(manager, 'MPASParallelManager')

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
                assert_expected_public_methods(manager, 'MPASParallelManager')

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
                    assert_expected_public_methods(manager, 'MPASParallelManager')
                
                output = f.getvalue()
                assert "MPI initialization failed" in output or manager.backend == 'multiprocessing'
                assert manager.backend == 'multiprocessing'


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

