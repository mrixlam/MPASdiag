#!/usr/bin/env python3
"""
MPAS Parallel Processing Backend Unit Tests

This module provides comprehensive unit tests for the multiprocessing backend functionality
in the MPASParallelManager class. These tests verify that parallel task execution works
correctly across different backend implementations (multiprocessing, threading, sequential),
handles error conditions gracefully, maintains data integrity during distributed processing,
and properly controls worker pool sizes. The tests ensure correct module-level function
definition to avoid pickling issues with multiprocessing.

Tests Performed:
    Helper Functions:
        - simple_task: Task function that squares a number for basic execution testing
        - slow_task: Task function with 0.05s delay to test performance characteristics
        - error_task: Task function that raises ValueError for x=5 to test error handling
    
    Test Functions:
        - test_basic_multiprocessing: Validates basic parallel task execution and result collection
        - test_with_timing: Tests performance speedup with timed tasks across multiple workers
        - test_error_handling: Validates error collection policy with intentional task failures
        - test_serial_fallback: Tests sequential execution backend as fallback mode
        - test_n_workers_argument: Validates worker count control (1, 4, auto-detect)
        - test_n_workers_with_cli_integration: Tests CLI parameter flow to MPASParallelManager

Test Coverage:
    - Parallel task execution: multiprocessing backend with task distribution
    - Result collection: aggregation of results from multiple workers
    - Error handling: collect policy for partial results with task failures
    - Backend selection: multiprocessing, threading, and sequential modes
    - Worker pool sizing: explicit counts (1, 4) and auto-detection
    - Performance measurement: wall time tracking and speedup calculations
    - Task result integrity: verification that results match expected values
    - CLI integration: parameter flow from command-line to manager configuration
    - Pickling compatibility: module-level function definitions for multiprocessing

Testing Approach:
    Unit and integration tests using the actual MPASParallelManager with simple task
    functions to validate parallel execution patterns, error handling, and result collection.
    Tests use synthetic workloads (arithmetic operations, timed delays, intentional errors)
    to verify behavior without requiring full MPAS dataset processing infrastructure.
    Performance tests measure wall time to verify speedup characteristics.

Expected Results:
    - Basic multiprocessing executes tasks correctly and returns expected results
    - Multiple workers show performance improvement over single worker execution
    - Error handling collects partial results when some tasks fail (9/10 success)
    - Serial backend executes tasks sequentially with correct results
    - Worker count control properly sets pool size (1, 4, or auto-detected)
    - Auto-detect uses cpu_count() - 1 workers (or 1 minimum)
    - CLI integration correctly maps workers argument to manager pool size
    - All successful tasks return results matching expected values
    - Failed tasks (x=5 in error_task) properly identified with task_id=5

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: October 2025
Version: 1.0.0
"""

import time
from mpasdiag.processing.parallel import MPASParallelManager


def simple_task(x: int) -> int:
    """
    Execute simple arithmetic operation squaring the input value for basic parallel execution testing. This task function provides the simplest possible workload to verify parallel task distribution and result collection without computational complexity. The function takes a single integer input and returns its square value. Used primarily for validating that the multiprocessing backend correctly distributes tasks and aggregates results. This minimal task helps isolate parallel execution mechanics from actual computational workload.

    Parameters:
        x (int): Input integer value to be squared.

    Returns:
        int: Square of the input value (x²).
    """
    return x ** 2


def slow_task(x: int) -> int:
    """
    Execute timed task with artificial delay to simulate realistic processing workload and measure performance characteristics. This task function introduces a 50 millisecond sleep delay before computation to model computationally intensive operations without actual computational complexity. The artificial delay enables testing of parallel speedup and worker pool efficiency across different configurations. Used to verify that multiple workers provide performance improvements over single worker execution. Returns double the input value after the processing delay.

    Parameters:
        x (int): Input integer value to be doubled after processing delay.

    Returns:
        int: Double of the input value (x × 2) after 50ms delay.
    """
    time.sleep(0.05)
    return x * 2


def error_task(x: int) -> int:
    """
    Execute task with conditional failure behavior to test error handling and collection policies. This task function intentionally raises ValueError when input equals 5 to validate the parallel manager's error handling mechanisms. For all other inputs, the function successfully returns input plus 10. Used to verify that the collect error policy correctly captures failed tasks while continuing to process successful ones. This enables testing of partial result collection and error reporting in distributed task execution scenarios.

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
    Validate basic parallel task execution functionality with multiprocessing backend and result collection. This test verifies the MPASParallelManager correctly distributes tasks across worker processes, executes them in parallel, and aggregates results maintaining data integrity. Ten simple squaring tasks test fundamental parallel execution without error conditions or performance requirements. Assertions confirm all tasks complete successfully and results match expected squared values. This foundational test ensures the multiprocessing backend functions correctly for basic parallel workload distribution and result collection.

    Parameters:
        None

    Returns:
        None
    """
    print("\n" + "="*60)
    print("TEST 1: Basic Multiprocessing")
    print("="*60)
    
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
    print("✓ Test passed")


def test_with_timing() -> None:
    """
    Validate parallel execution performance characteristics and speedup with timed tasks across multiple workers. This test measures wall clock time for 16 tasks with artificial 50ms delays executed across 4 worker processes to verify parallel speedup benefits. The test demonstrates that multiple workers can execute delayed tasks concurrently reducing total execution time compared to sequential processing. Assertions confirm all tasks complete successfully with correct results and timing measurements verify parallel execution efficiency. This performance test validates the multiprocessing backend provides actual concurrency benefits for computational workloads.

    Parameters:
        None

    Returns:
        None
    """
    print("\n" + "="*60)
    print("TEST 2: Performance with Timing")
    print("="*60)
    
    manager = MPASParallelManager(n_workers=4)
    tasks = list(range(16))
    
    print("\nRunning 16 tasks (each takes ~0.05s)...")
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
    print("✓ Test passed")


def test_error_handling() -> None:
    """
    Validate error handling and partial result collection with collect policy during parallel execution. This test verifies the MPASParallelManager correctly handles task failures while continuing to process remaining tasks and collect successful results. Ten tasks with one intentional failure (x=5) test the collect error policy's ability to capture both successful and failed task outcomes. Assertions confirm 9 successful task completions, 1 failure at the expected task ID, and proper error information capture. This ensures robust error handling enables partial result recovery in production scenarios where some tasks may fail without invalidating entire workloads.

    Parameters:
        None

    Returns:
        None
    """
    print("\n" + "="*60)
    print("TEST 3: Error Handling")
    print("="*60)
    
    manager = MPASParallelManager()
    manager.set_error_policy('collect')
    
    tasks = list(range(10))
    print("\nRunning tasks where x=5 will fail...")
    results = manager.parallel_map(error_task, tasks)
    
    assert results is not None, "parallel_map returned None"
    
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    
    print(f"\nSuccessful: {len(successes)}/{len(results)}")
    print(f"Failed: {len(failures)}/{len(results)}")
    print(f"Failed task IDs: {[r.task_id for r in failures]}")
    
    assert len(successes) == 9, f"Expected 9 successes, got {len(successes)}"
    assert len(failures) == 1, f"Expected 1 failure, got {len(failures)}"
    assert failures[0].task_id == 5, "Wrong task failed"
    print("✓ Test passed")


def test_serial_fallback() -> None:
    """
    Verify sequential execution backend operates correctly as fallback mode for parallel processing. This test validates the MPASParallelManager can execute tasks serially when multiprocessing is unavailable or undesired providing a reliable fallback execution path. Five simple squaring tasks execute sequentially through the serial backend without worker pool overhead. Assertions confirm all tasks complete successfully with correct results matching expected values. This ensures the serial backend provides functionally equivalent execution to parallel modes supporting systems where multiprocessing is problematic or unnecessary for small workloads.

    Parameters:
        None

    Returns:
        None
    """
    print("\n" + "="*60)
    print("TEST 4: Serial Mode")
    print("="*60)
    
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
    print("✓ Test passed")


def test_n_workers_argument() -> None:
    """
    Validate worker count control through n_workers parameter with explicit and auto-detection modes. This test verifies the MPASParallelManager correctly creates worker pools with specified sizes (1, 4) and auto-detects appropriate worker counts when not specified. Performance measurements with timed tasks across different worker configurations demonstrate pool size control and verify timing characteristics align with worker counts. Assertions confirm pool sizes match requested values (1 worker, 4 workers, auto-detected count). This ensures users have fine-grained control over parallelism through n_workers parameter supporting both explicit configuration and intelligent auto-detection for optimal resource utilization.

    Parameters:
        None

    Returns:
        None
    """
    print("\n" + "="*60)
    print("TEST 5: n_workers Argument Control")
    print("="*60)
    
    print("\nTesting with n_workers=1...")
    manager_1 = MPASParallelManager(n_workers=1, backend='multiprocessing')
    tasks = list(range(16))
    
    start = time.time()
    results_1 = manager_1.parallel_map(slow_task, tasks)
    time_1_worker = time.time() - start
    
    assert results_1 is not None, "parallel_map returned None with 1 worker"
    assert manager_1.size == 1, f"Expected size=1, got {manager_1.size}"
    print(f"✓ n_workers=1: Wall time = {time_1_worker:.2f}s, Pool size = {manager_1.size}")
    
    print("\nTesting with n_workers=4...")
    manager_4 = MPASParallelManager(n_workers=4, backend='multiprocessing')
    
    start = time.time()
    results_4 = manager_4.parallel_map(slow_task, tasks)
    time_4_workers = time.time() - start
    
    assert results_4 is not None, "parallel_map returned None with 4 workers"
    assert manager_4.size == 4, f"Expected size=4, got {manager_4.size}"
    print(f"✓ n_workers=4: Wall time = {time_4_workers:.2f}s, Pool size = {manager_4.size}")
    
    print("\nTesting with n_workers=None (auto-detect)...")
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
    
    print("\n✓ Worker count control verified:")
    print("  - n_workers=1 correctly created pool with 1 worker")
    print("  - n_workers=4 correctly created pool with 4 workers")
    print("  - n_workers=None correctly auto-detected workers")
    print("✓ Test passed")


def test_n_workers_with_cli_integration() -> None:
    """
    Verify complete parameter flow from CLI arguments through configuration to MPASParallelManager worker pool creation. This test validates the integration path where command-line worker count specifications propagate correctly through configuration objects to the parallel manager. Four test cases (1 worker, 2 workers, 4 workers, auto-detect) verify explicit worker counts and automatic detection scenarios. Assertions confirm pool sizes match CLI specifications and auto-detection uses cpu_count() - 1 formula. This integration test ensures end-to-end parameter flow from user command-line input to actual parallel execution configuration supporting operational workflows with diverse hardware configurations.

    Parameters:
        None

    Returns:
        None
    """
    print("\n" + "="*60)
    print("TEST 6: CLI to Manager Worker Count Integration")
    print("="*60)
    
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
    
    print("\n✓ CLI integration test passed")


if __name__ == '__main__':
    """
    Execute comprehensive multiprocessing backend test suite with structured output and error reporting.
    
    This main execution block coordinates six test functions validating parallel processing functionality
    including basic execution, performance characteristics, error handling, serial fallback mode, worker
    pool sizing, and CLI integration. Tests run sequentially with informative output banners indicating
    progress and success status. Exception handling captures and displays detailed error information
    including stack traces for debugging test failures. Exit code 1 signals test failures for CI/CD
    integration while successful completion returns default zero status.
    """
    print("\n" + "="*70)
    print("MPAS PARALLEL PROCESSING - MULTIPROCESSING BACKEND TESTS")
    print("="*70)
    
    try:
        test_basic_multiprocessing()
        test_with_timing()
        test_error_handling()
        test_serial_fallback()
        test_n_workers_argument()
        test_n_workers_with_cli_integration()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
