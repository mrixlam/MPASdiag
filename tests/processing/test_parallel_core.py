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

from mpasdiag.processing.parallel import (
    MPASParallelManager
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


def test_mpi_import_fallback_warning() -> None:
    """
    This test verifies that when mpi4py is not importable, parallel.py sets
    MPI_AVAILABLE to False and MPI to None, and issues a UserWarning (lines 28-31).
    It temporarily blocks the mpi4py import by inserting None into sys.modules, then
    reloads the module and asserts the expected module-level state. The module is
    restored to its original state afterward to avoid affecting subsequent tests.

    Parameters:
        None

    Returns:
        None
    """
    import sys
    import importlib
    from unittest.mock import patch
    import mpasdiag.processing.parallel as parallel_mod

    # Snapshot the module dict before any reload so we can restore the exact
    # same class objects afterward — reloading creates new class identities
    # which breaks isinstance() checks and enum equality in later tests.
    saved_state = dict(parallel_mod.__dict__)

    try:
        with patch.dict(sys.modules, {'mpi4py': None}):
            with pytest.warns(UserWarning, match="mpi4py is not available"):
                importlib.reload(parallel_mod)

            assert parallel_mod.MPI_AVAILABLE is False
            assert parallel_mod.MPI is None
    finally:
        # Restore original bindings (class objects, enums, etc.) so that
        # modules which imported these at load time continue to work.
        parallel_mod.__dict__.update(saved_state)
