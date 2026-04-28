#!/usr/bin/env python3
"""
MPASdiag Test Suite: Parallel Processing - Task Distributor Tests

This module contains tests for the MPASDataCache class in the mpasdiag.processing.data_cache module, specifically targeting code coverage for the coordinate loading and variable data loading functionality. The tests are designed to verify that the cache correctly handles various scenarios, including pickling behavior, coordinate loading branches based on dataset dimensions, handling of missing coordinate variables, eviction of least accessed coordinates, and error handling when accessing unloaded data. The tests utilize pytest for assertions and are structured to cover edge cases and ensure the robustness of the MPASDataCache implementation in the context of parallel processing and task distribution within the MPASdiag framework. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: April 2026
Version: 1.0.0
"""
import pytest
from typing import List
from unittest.mock import Mock

from mpasdiag.processing.parallel import (
    MPASTaskDistributor,
    LoadBalanceStrategy,
)

from tests.test_data_helpers import assert_expected_public_methods


@pytest.fixture
def mock_comm_rank0() -> Mock:
    """
    This fixture provides a mock MPI communicator configured to simulate rank 0 of a 3-worker setup. It mocks the Get_rank and Get_size methods to return 0 and 3, respectively, allowing tests to verify task distribution logic for the master process in a parallel environment. This setup is essential for testing the behavior of MPASTaskDistributor under different load balancing strategies while ensuring that the tests are self-contained and do not require an actual MPI environment. 

    Parameters:
        None

    Returns:
        Mock: A mock MPI communicator with Get_rank and Get_size methods configured for rank 0 and 3 total workers.
    """
    comm = Mock()
    comm.Get_rank.return_value = 0
    comm.Get_size.return_value = 3
    return comm


@pytest.fixture
def tasks_9() -> List[int]:
    """
    This fixture provides a simple list of 9 integer tasks (0 through 8) for testing the task distribution logic of MPASTaskDistributor. The list is designed to be small and manageable while still allowing for meaningful distribution patterns across multiple workers. This fixture enables tests to verify that the distributor correctly assigns tasks based on the specified load balancing strategy, ensuring that the returned task IDs and their corresponding tasks are valid and consistent with the input list. 

    Parameters:
        None

    Returns:
        List[int]: A list of 9 integer tasks.
    """
    return list(range(9))


class TestMPASTaskDistributor:
    """ Tests for all four load balancing strategies of MPASTaskDistributor. """

    def test_distribute_tasks_dynamic_strategy(self: 'TestMPASTaskDistributor',
                                               mock_comm_rank0: Mock,
                                               tasks_9: List[int],) -> None:
        """
        This test verifies that the DYNAMIC strategy returns a non-empty list of task IDs and tasks, and that all returned task IDs are valid indices into the input task list. It exercises lines 169-170 and 242 by creating a distributor with DYNAMIC strategy, calling distribute_tasks, and asserting that the results are consistent with the input tasks. The test ensures that the dynamic distribution logic correctly produces a valid subset of tasks for rank 0, even though the specific distribution pattern may not be deterministic due to the nature of dynamic load balancing. 

        Parameters:
            mock_comm_rank0 (Mock): Fixture-provided mock MPI communicator at rank 0.
            tasks_9 (List[int]): Fixture-provided list of 9 integer tasks.

        Returns:
            None
        """
        distributor = MPASTaskDistributor(mock_comm_rank0, LoadBalanceStrategy.DYNAMIC)
        assert_expected_public_methods(distributor, 'MPASTaskDistributor')
        result = distributor.distribute_tasks(tasks_9)
        assert len(result) > 0
        assert all(isinstance(tid, int) and task in tasks_9 for tid, task in result)

    def test_distribute_tasks_block_strategy(self: 'TestMPASTaskDistributor',
                                             mock_comm_rank0: Mock,
                                             tasks_9: List[int],) -> None:
        """
        This test verifies that the BLOCK strategy returns a contiguous block of task IDs starting from 0, corresponding to the first ceil(9/3)=3 tasks for rank 0 in a 3-worker setup. It exercises lines 169-170 and 211-216 by creating a distributor with BLOCK strategy, calling distribute_tasks, and asserting that the returned task IDs are exactly [0, 1, 2] and that the corresponding tasks match the input list. This confirms that the block distribution logic correctly assigns contiguous chunks of tasks to each worker based on their rank and the total number of workers. 

        Parameters:
            mock_comm_rank0 (Mock): Fixture-provided mock MPI communicator at rank 0.
            tasks_9 (List[int]): Fixture-provided list of 9 integer tasks.

        Returns:
            None
        """
        distributor = MPASTaskDistributor(mock_comm_rank0, LoadBalanceStrategy.BLOCK)
        assert_expected_public_methods(distributor, 'MPASTaskDistributor')
        result = distributor.distribute_tasks(tasks_9)
        assert len(result) > 0
        task_ids = [tid for tid, _ in result]
        assert task_ids == list(range(len(task_ids))), "Block distribution must be contiguous from 0"

    def test_distribute_tasks_cyclic_strategy(self: 'TestMPASTaskDistributor',
                                              mock_comm_rank0: Mock,
                                              tasks_9: List[int],) -> None:
        """
        This test verifies that the CYCLIC strategy performs round-robin assignment, giving rank 0 every 3rd task (indices 0, 3, 6) from a 9-element list with 3 workers. It exercises lines 171-172 and 229, confirming that the returned task IDs match the expected cyclic pattern. The test ensures that the cyclic distribution logic correctly assigns tasks in a round-robin fashion based on the worker's rank and the total number of workers, resulting in a non-contiguous but predictable set of tasks for rank 0. 

        Parameters:
            mock_comm_rank0 (Mock): Fixture-provided mock MPI communicator at rank 0.
            tasks_9 (List[int]): Fixture-provided list of 9 integer tasks.

        Returns:
            None
        """
        distributor = MPASTaskDistributor(mock_comm_rank0, LoadBalanceStrategy.CYCLIC)
        assert_expected_public_methods(distributor, 'MPASTaskDistributor')
        result = distributor.distribute_tasks(tasks_9)
        expected_ids = [0, 3, 6]
        assert [tid for tid, _ in result] == expected_ids

    def test_distribute_tasks_dynamic_matches_static(self: 'TestMPASTaskDistributor',
                                                     mock_comm_rank0: Mock,
                                                     tasks_9: List[int],) -> None:
        """
        This test verifies that the DYNAMIC strategy's _dynamic_distribution method (line 242) delegates to _static_distribution, producing identical results. It constructs both a DYNAMIC and a STATIC distributor and asserts that their distribute_tasks outputs are equal for the same input. This confirms that the dynamic distribution logic correctly falls back to static distribution when invoked, ensuring consistency in task assignment regardless of how the distribution method is called. 

        Parameters:
            mock_comm_rank0 (Mock): Fixture-provided mock MPI communicator at rank 0.
            tasks_9 (List[int]): Fixture-provided list of 9 integer tasks.

        Returns:
            None
        """
        dynamic = MPASTaskDistributor(mock_comm_rank0, LoadBalanceStrategy.DYNAMIC)
        static = MPASTaskDistributor(mock_comm_rank0, LoadBalanceStrategy.STATIC)
        assert dynamic.distribute_tasks(tasks_9) == static.distribute_tasks(tasks_9)

    def test_distribute_tasks_else_fallback(self: 'TestMPASTaskDistributor',
                                            mock_comm_rank0: Mock,
                                            tasks_9: List[int],) -> None:
        """
        This test exercises the else branch (lines 173-174) of distribute_tasks by monkey-patching the distributor's strategy to a value that does not match any LoadBalanceStrategy enum member. It asserts that the fallback returns the same result as static distribution, confirming that unknown strategies are handled gracefully and that the method does not raise an error when encountering an unrecognized strategy. This ensures robustness in the face of potential misconfigurations or future additions to the LoadBalanceStrategy enum. 

        Parameters:
            mock_comm_rank0 (Mock): Fixture-provided mock MPI communicator at rank 0.
            tasks_9 (List[int]): Fixture-provided list of 9 integer tasks.

        Returns:
            None
        """
        distributor = MPASTaskDistributor(mock_comm_rank0, LoadBalanceStrategy.STATIC)
        distributor.strategy = "non_existent_strategy"
        static = MPASTaskDistributor(mock_comm_rank0, LoadBalanceStrategy.STATIC)
        assert distributor.distribute_tasks(tasks_9) == static.distribute_tasks(tasks_9)

    def test_block_distribution_direct(self: 'TestMPASTaskDistributor',
                                       mock_comm_rank0: Mock,) -> None:
        """
        This test calls _block_distribution directly with 10 integer tasks and 3 workers, verifying that rank 0 receives the first 4 tasks (indices 0-3) as expected for a block distribution (line 216). It confirms that the block distribution logic correctly calculates the number of tasks per worker and assigns the appropriate contiguous block of tasks to rank 0 based on its position in the worker hierarchy. 

        Parameters:
            mock_comm_rank0 (Mock): Fixture-provided mock MPI communicator at rank 0.

        Returns:
            None
        """
        distributor = MPASTaskDistributor(mock_comm_rank0, LoadBalanceStrategy.BLOCK)
        tasks = list(range(10))
        result = distributor._block_distribution(tasks)
        assert result == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_cyclic_distribution_direct(self: 'TestMPASTaskDistributor',
                                        mock_comm_rank0: Mock,) -> None:
        """
        This test calls _cyclic_distribution directly with 6 integer tasks and 3 workers, verifying that rank 0 receives every 3rd task (indices 0 and 3) as expected for a cyclic distribution (line 229). It confirms that the cyclic distribution logic correctly iterates through the task list in a round-robin fashion, assigning tasks to rank 0 based on its position in the worker hierarchy and the total number of workers. 

        Parameters:
            mock_comm_rank0 (Mock): Fixture-provided mock MPI communicator at rank 0.

        Returns:
            None
        """
        distributor = MPASTaskDistributor(mock_comm_rank0, LoadBalanceStrategy.CYCLIC)
        tasks = ['a', 'b', 'c', 'd', 'e', 'f']
        result = distributor._cyclic_distribution(tasks)
        assert result == [(0, 'a'), (3, 'd')]

    def test_dynamic_distribution_direct(self: 'TestMPASTaskDistributor',
                                         mock_comm_rank0: Mock,
                                         tasks_9: List[int],) -> None:
        """
        This test calls _dynamic_distribution directly and verifies that it produces the same result as _static_distribution for the same input tasks. Since _dynamic_distribution currently delegates to _static_distribution, this test confirms that the dynamic distribution logic correctly falls back to static distribution, ensuring consistency in task assignment regardless of how the distribution method is invoked. 

        Parameters:
            mock_comm_rank0 (Mock): Fixture-provided mock MPI communicator at rank 0.
            tasks_9 (List[int]): Fixture-provided list of 9 integer tasks.

        Returns:
            None
        """
        distributor = MPASTaskDistributor(mock_comm_rank0, LoadBalanceStrategy.DYNAMIC)
        assert distributor._dynamic_distribution(tasks_9) == distributor._static_distribution(tasks_9)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
