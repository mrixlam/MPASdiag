#!/usr/bin/env python3
"""
MPASdiag Test Suite: Parallel Processing - Task Distribution and
Result Collection Tests

This module contains tests for the execution paths of the `MPASParallelManager` when using the multiprocessing and serial backends. It verifies that tasks are executed correctly, that error handling behaves as expected, and that statistics are collected properly in both parallel and serial contexts. The tests also ensure that platform-specific multiprocessing behaviors are handled correctly, and that fallback mechanisms work when multiprocessing contexts cannot be created.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import pytest
import matplotlib.pyplot as plt
from typing import Generator
from unittest.mock import Mock, MagicMock

from mpasdiag.processing.parallel import (
    LoadBalanceStrategy,
    TaskResult,
    MPASTaskDistributor,
    MPASResultCollector,
)

from tests.test_data_helpers import assert_expected_public_methods


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
        assert_expected_public_methods(distributor, 'MPASTaskDistributor')
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
        assert_expected_public_methods(distributor, 'MPASTaskDistributor')
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
        assert_expected_public_methods(distributor, 'MPASTaskDistributor')
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
        assert_expected_public_methods(distributor, 'MPASTaskDistributor')
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
        assert_expected_public_methods(distributor, 'MPASTaskDistributor')

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
        assert_expected_public_methods(collector, 'MPASResultCollector')
        
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
        assert_expected_public_methods(collector, 'MPASResultCollector')

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
        assert_expected_public_methods(collector, 'MPASResultCollector')
        
        assert stats.total_tasks == pytest.approx(3)
        assert stats.completed_tasks == pytest.approx(2)
        assert stats.failed_tasks == pytest.approx(1)
        assert stats.total_time == pytest.approx(3.5)
        assert stats.worker_times[0] == pytest.approx(3.0)
        assert stats.worker_times[1] == pytest.approx(0.5)
        assert stats.load_imbalance > 0


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
        assert_expected_public_methods(collector, 'MPASResultCollector')
        
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
        assert_expected_public_methods(distributor, 'MPASTaskDistributor')
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
        assert_expected_public_methods(distributor, 'MPASTaskDistributor')
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
        assert_expected_public_methods(distributor, 'MPASTaskDistributor')

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
        assert_expected_public_methods(distributor, 'MPASTaskDistributor')
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
        assert_expected_public_methods(distributor, 'MPASTaskDistributor')
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
        assert_expected_public_methods(distributor, 'MPASTaskDistributor')

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
        assert_expected_public_methods(collector, 'MPASResultCollector')
        
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
        assert_expected_public_methods(collector, 'MPASResultCollector') 
        local_results = [TaskResult(task_id=0, success=True, worker_rank=0)]
        gathered = collector.gather_results(local_results)
        
        assert gathered is not None
        assert len(gathered) == pytest.approx(2)
        assert gathered[0].task_id == pytest.approx(0)
        assert gathered[1].task_id == pytest.approx(1)


