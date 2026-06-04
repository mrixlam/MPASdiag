#!/usr/bin/env python3

"""
MPASdiag Test Suite: Parallel Processing - MPI and Synchronization Tests

This module contains tests for the MPI execution paths and synchronization helpers of the `MPASParallelManager`. It verifies that the `barrier` and `finalize` methods work correctly in both MPI and non-MPI contexts, ensuring that they do not raise exceptions when called in serial mode and that they perform the expected synchronization when MPI is available. The tests also cover the behavior of local task execution under different error policies, confirming that exceptions are handled appropriately based on the policy and execution context. Additionally, the module includes tests for the `_mpi_map` method to ensure it produces expected verbose output, enforces necessary assertions, and returns `None` on worker ranks while only the master rank returns aggregated results. These tests are crucial for validating the correct implementation of parallel execution and synchronization in MPASdiag, ensuring robust performance across different backends. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import pytest
from typing import List
from unittest.mock import Mock, MagicMock, patch

from mpasdiag.processing.parallel import (
    MPASParallelManager,
    LoadBalanceStrategy,
    TaskResult,
    MPASTaskDistributor,
    MPASResultCollector,
)

from tests.test_data_helpers import assert_expected_public_methods


@pytest.fixture
def sample_tasks() -> List[int]:
    """Simple list of integer tasks from 0 to 9 for testing purposes."""
    return list(range(10))


class TestExecuteLocalTasks:
    """ Tests for executing tasks locally under different error policies (abort/continue/collect). """
    
    def test_execute_local_tasks_with_abort_policy_mpi(self: 'TestExecuteLocalTasks') -> None:
        """
        This test verifies that when the error policy is set to 'abort' and the manager is running under MPI, any exception raised during local task execution results in an abort call on the MPI communicator. It creates a mock MPI communicator that simulates a worker rank and size, sets the error policy to 'abort', and defines a `failing_func` that raises a `ValueError`. The test calls `_execute_local_tasks` with this function and asserts that `Abort` was called on the communicator with the correct rank. This ensures that critical failures during local task execution properly trigger an abort in MPI runs, preventing further execution and allowing for appropriate error handling at the MPI level. 

        Parameters:
            None

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
                assert_expected_public_methods(manager, 'MPASParallelManager')
                manager.comm = mock_comm
                manager.set_error_policy('abort')
                
                def failing_func(task):
                    raise ValueError("Task failed")
                
                local_tasks = [(0, "task1")]
                
                manager._execute_local_tasks(failing_func, local_tasks)
                mock_comm.Abort.assert_called_once_with(1)
    
    
    def test_execute_local_tasks_verbose_error_output(self: 'TestExecuteLocalTasks', 
                                                      sample_tasks: List[int]) -> None:
        """
        This test verifies that when the error policy is set to 'continue' and verbose mode is enabled, any exceptions raised during local task execution are captured and printed to stdout with appropriate error messages. It constructs a manager with the 'serial' backend, sets the error policy to 'continue', and defines a `failing_func` that raises a `ValueError` for a specific task. The test calls `_execute_local_tasks` with this function and captures stdout to assert that error messages related to the failed task are printed. This ensures that when continuing on errors, the manager provides visibility into what went wrong with specific tasks, aiding users in diagnosing issues while still allowing other tasks to proceed. 

        Parameters:
            sample_tasks (List[int]): Fixture-provided list of tasks.

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=True)
        assert_expected_public_methods(manager, 'MPASParallelManager')
        manager.set_error_policy('continue')
        
        def failing_func(task) -> int:
            """ 
            This function simulates a task that fails for a specific input (task == 5) by raising a ValueError. For all other tasks, it simply returns the task value. This allows us to test the error handling and verbose output when a task fails under the 'continue' error policy. The test will check that the error message is printed for the failed task while other tasks continue to execute successfully.

            Parameters:
                task: The input task to be processed.

            Returns:
                The original task value if it does not equal 5; otherwise, raises a ValueError indicating that the task failed.
            """
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
        assert_expected_public_methods(manager, 'MPASParallelManager')


class TestMPIMapExecution:
    """ Tests of MPI-mapped execution paths including master/worker separation and communicator-based control flow. """
    
    def test_mpi_map_verbose_output(self: 'TestMPIMapExecution') -> None:
        """
        This test verifies that the `_mpi_map` method produces expected verbose output when executed in an MPI context. It creates a mock MPI communicator simulating a master rank, patches the `MPI_AVAILABLE` flag to True, and captures stdout during the call to `_mpi_map`. The test asserts that the output includes key phrases related to processing tasks across ranks and load balance strategy, confirming that verbose logging provides insights into the execution process in an MPI environment. This ensures that users have visibility into how tasks are being distributed and processed when running with MPI, aiding in debugging and performance tuning. 

        Parameters:
            None

        Returns:
            None
        """
        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 4
        mock_comm.bcast.return_value = list(range(10))
        mock_comm.gather.return_value = [[TaskResult(i, True, worker_rank=0) for i in range(10)]]
        mock_comm.allreduce.return_value = 1.0  
        
        with patch('mpasdiag.processing.parallel.MPI_AVAILABLE', True):
            mock_mpi = MagicMock()
            mock_mpi.COMM_WORLD = mock_comm
            
            with patch('mpasdiag.processing.parallel.MPI', mock_mpi):
                manager = MPASParallelManager(backend='mpi', verbose=True)
                assert_expected_public_methods(manager, 'MPASParallelManager')
                manager.comm = mock_comm
                manager.distributor = MPASTaskDistributor(mock_comm, LoadBalanceStrategy.STATIC)
                assert_expected_public_methods(manager.distributor, 'MPASTaskDistributor')
                manager.collector = MPASResultCollector(mock_comm)
                assert_expected_public_methods(manager.collector, 'MPASResultCollector')
                
                def simple_func(task) -> int:
                    """
                    This is a simple function that takes a task (in this case, an integer) and returns its doubled value. It is used as the mapping function in the `_mpi_map` test to verify that the MPI mapping logic correctly processes tasks and produces results. The function is straightforward and serves as a stand-in for more complex processing that might occur in real use cases, allowing us to focus on testing the MPI execution flow and verbose output. 

                    Parameters:
                        task: The input task to be processed.

                    Returns:
                        The doubled value of the input task.
                    """
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
    

class TestMPIMapReturnValue:
    """ Tests ensuring `_mpi_map` returns `None` on worker ranks and only the master returns aggregated results. """
    
    def test_mpi_map_returns_none_on_worker(self: 'TestMPIMapReturnValue') -> None:
        """
        This test verifies that when the `_mpi_map` method is executed on a worker rank, it returns `None` instead of attempting to return results. It simulates an MPI environment with a mock communicator where the current rank is a worker (rank 2) and asserts that the return value of `_mpi_map` is `None` on the worker. This confirms that the method correctly distinguishes between master and worker contexts, ensuring that only the master rank processes and returns results while workers simply send their local results to the master without trying to access gathered results. This behavior is crucial for maintaining correct MPI semantics and preventing errors in worker processes. 

        Parameters:
            None

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
                assert_expected_public_methods(manager.distributor, 'MPASTaskDistributor')
                manager.collector = MPASResultCollector(mock_comm)
                assert_expected_public_methods(manager.collector, 'MPASResultCollector')
                assert_expected_public_methods(manager, 'MPASParallelManager')
                
                def simple_func(task) -> int:
                    """
                    This is a simple function that takes a task (in this case, an integer) and returns its doubled value. It is used as the mapping function in the `_mpi_map` test to verify that the MPI mapping logic correctly processes tasks and produces results. The function is straightforward and serves as a stand-in for more complex processing that might occur in real use cases, allowing us to focus on testing the MPI execution flow and verbose output. 

                    Parameters:
                        task: The input task to be processed.

                    Returns:
                        The doubled value of the input task.
                    """
                    return task * 2
                
                results = manager._mpi_map(simple_func, list(range(10)))

                assert results is None


class TestSetupMPIBackendFallback:
    """ Tests that an exception raised inside _setup_mpi_backend causes graceful fallback to multiprocessing. """

    def test_mpi_setup_exception_falls_back_to_multiprocessing(self: 'TestSetupMPIBackendFallback',) -> None:
        """
        This test verifies that if an exception occurs during the setup of the MPI backend (e.g., due to a failure in MPI initialization), the `MPASParallelManager` gracefully falls back to using the multiprocessing backend. It simulates an MPI initialization failure by having the mock communicator's `Get_rank` method raise a `RuntimeError`. The test captures stdout to confirm that appropriate error messages are printed, indicating the failure and the fallback to multiprocessing. Finally, it asserts that the manager's backend is set to 'multiprocessing' after the fallback, confirming that the error handling logic correctly switches to a functional backend without crashing. 

        Parameters:
            None

        Returns:
            None
        """
        import io
        from contextlib import redirect_stdout
        from unittest.mock import MagicMock, patch

        mock_comm = Mock()
        mock_comm.Get_rank.side_effect = RuntimeError("Simulated MPI failure")
        mock_comm.Get_size.return_value = 4

        with patch('mpasdiag.processing.parallel.MPI_AVAILABLE', True):
            mock_mpi = MagicMock()
            mock_mpi.COMM_WORLD = mock_comm

            with patch('mpasdiag.processing.parallel.MPI', mock_mpi):
                f = io.StringIO()
                with redirect_stdout(f):
                    manager = MPASParallelManager(backend='mpi', verbose=True)

        assert manager.backend == 'multiprocessing'
        output = f.getvalue()
        assert "MPI initialization failed" in output
        assert "Falling back to multiprocessing backend" in output
        assert_expected_public_methods(manager, 'MPASParallelManager')


class TestParallelMapMPIDispatch:
    """ Tests that parallel_map dispatches to _mpi_map when backend is 'mpi'. """

    def test_parallel_map_with_mpi_backend_calls_mpi_map(self: 'TestParallelMapMPIDispatch',) -> None:
        """
        This test verifies that when the `parallel_map` method is called on a manager with the 'mpi' backend, it correctly dispatches to the `_mpi_map` method. It sets up a mock MPI environment with a communicator and patches the `MPI_AVAILABLE` flag to True. The test defines a simple function to be mapped and calls `parallel_map`, asserting that the results are returned as expected and that the manager's public methods are intact. This confirms that the `parallel_map` method correctly routes execution to the MPI-specific mapping logic when the appropriate backend is selected, ensuring that parallel execution occurs as intended in an MPI context. 

        Parameters:
            None

        Returns:
            None
        """
        from unittest.mock import MagicMock, patch

        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 4
        mock_comm.bcast.return_value = list(range(6))
        mock_comm.gather.return_value = [[TaskResult(i, True, worker_rank=0) for i in range(6)]]

        with patch('mpasdiag.processing.parallel.MPI_AVAILABLE', True):
            mock_mpi = MagicMock()
            mock_mpi.COMM_WORLD = mock_comm

            with patch('mpasdiag.processing.parallel.MPI', mock_mpi):
                manager = MPASParallelManager(backend='mpi', verbose=False)
                manager.comm = mock_comm
                manager.distributor = MPASTaskDistributor(mock_comm, LoadBalanceStrategy.STATIC)
                manager.collector = MPASResultCollector(mock_comm)

                def simple_func(task: int) -> int:
                    """
                    This is a simple function that takes a task (in this case, an integer) and returns its doubled value. It is used as the mapping function in the `_mpi_map` test to verify that the MPI mapping logic correctly processes tasks and produces results. The function is straightforward and serves as a stand-in for more complex processing that might occur in real use cases, allowing us to focus on testing the MPI execution flow and verbose output. 

                    Parameters:
                        task: The input task to be processed.

                    Returns:
                        The doubled value of the input task.
                    """
                    return task * 2

                results = manager.parallel_map(simple_func, list(range(6)))

        assert results is not None
        assert_expected_public_methods(manager, 'MPASParallelManager')


class TestBarrierAndFinalize:
    """ Tests for the barrier() and finalize() methods across MPI and non-MPI backends. """

    def test_barrier_calls_comm_barrier_when_mpi_backend(self: 'TestBarrierAndFinalize',) -> None:
        """
        This test verifies that calling barrier() on a manager in MPI mode invokes comm.Barrier() exactly once (lines 766-767). It constructs an MPI-mode manager with a mock communicator and asserts that Barrier was called after invoking barrier(). This ensures that the barrier method correctly triggers MPI synchronization when running in an MPI environment, allowing for proper coordination between ranks. 

        Parameters:
            None

        Returns:
            None
        """
        from unittest.mock import MagicMock, patch

        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 4
        mock_comm.bcast.return_value = []

        with patch('mpasdiag.processing.parallel.MPI_AVAILABLE', True):
            mock_mpi = MagicMock()
            mock_mpi.COMM_WORLD = mock_comm

            with patch('mpasdiag.processing.parallel.MPI', mock_mpi):
                manager = MPASParallelManager(backend='mpi', verbose=False)
                manager.comm = mock_comm

        manager.barrier()
        mock_comm.Barrier.assert_called_once()
        assert_expected_public_methods(manager, 'MPASParallelManager')

    def test_barrier_is_noop_for_serial_backend(self: 'TestBarrierAndFinalize',) -> None:
        """
        This test confirms that calling barrier() on a manager with the 'serial' backend does not raise any exceptions and effectively acts as a no-op (lines 768-769). It creates a serial-mode manager and calls barrier(), asserting that no exceptions are raised and that the method can be called without issues in a non-MPI context. This ensures that the barrier method is implemented in a way that allows it to be safely called regardless of the backend, providing a consistent interface for synchronization even when MPI is not available. 

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        assert_expected_public_methods(manager, 'MPASParallelManager')
        manager.barrier()  # must not raise

    def test_finalize_mpi_calls_barrier_and_prints(self: 'TestBarrierAndFinalize',) -> None:
        """
        This test confirms that finalize() on an MPI-mode master manager calls comm.Barrier() and prints a finalization message (lines 771-773). It sets up a mock MPI environment with a master rank, captures stdout during the call to finalize(), and asserts that Barrier was called and that the output contains "finalized". This ensures that the finalize method performs necessary synchronization and provides user feedback when running in an MPI context, allowing for proper cleanup and communication of the finalization status across ranks. 

        Parameters:
            None

        Returns:
            None
        """
        import io
        from contextlib import redirect_stdout
        from unittest.mock import MagicMock, patch

        mock_comm = Mock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 4

        with patch('mpasdiag.processing.parallel.MPI_AVAILABLE', True):
            mock_mpi = MagicMock()
            mock_mpi.COMM_WORLD = mock_comm

            with patch('mpasdiag.processing.parallel.MPI', mock_mpi):
                manager = MPASParallelManager(backend='mpi', verbose=True)
                manager.comm = mock_comm
                manager.is_master = True

        f = io.StringIO()
        with redirect_stdout(f):
            manager.finalize()

        mock_comm.Barrier.assert_called()
        assert "finalized" in f.getvalue().lower()

    def test_finalize_serial_prints_message(self: 'TestBarrierAndFinalize',) -> None:
        """
        This test confirms that finalize() on a serial-mode manager prints a finalization message without raising exceptions. It creates a serial-mode manager, captures stdout during the call to finalize(), and asserts that the output contains "finalized". This ensures that the finalize method provides user feedback even in non-MPI contexts, allowing for consistent finalization behavior and messaging regardless of the execution backend. 

        Parameters:
            None

        Returns:
            None
        """
        import io
        from contextlib import redirect_stdout

        manager = MPASParallelManager(backend='serial', verbose=True)
        assert_expected_public_methods(manager, 'MPASParallelManager')

        f = io.StringIO()
        with redirect_stdout(f):
            manager.finalize()

        assert "finalized" in f.getvalue().lower()


class TestWorkStealing:
    """ Tests for the dynamic master/worker (work-stealing) MPI scheduler, driven with a mocked communicator so the dispatch protocol runs in a single process. """

    @staticmethod
    def _serial_manager() -> MPASParallelManager:
        """ 
        This helper method creates and returns a MPASParallelManager instance configured with the 'serial' backend and verbose mode disabled. It is used in the work-stealing tests to provide a manager instance that can be manipulated with a mocked communicator to simulate MPI behavior without requiring an actual MPI environment. This allows the tests to focus on verifying the logic of the master/worker dispatch protocol in a controlled, single-process context.

        Parameters:
            None

        Returns:
            MPASParallelManager: An instance of MPASParallelManager with 'serial' backend and verbose mode disabled.
        """
        return MPASParallelManager(backend='serial', verbose=False)

    def test_mpi_dispatch_master(self: 'TestWorkStealing') -> None:
        """
        This test verifies the dispatcher half of the work-stealing scheduler. It mocks the communicator so that two workers (world size 3) each send an initial readiness message and then return the result of one task, and asserts that _mpi_dispatch_master hands out both task indices, records both results in task order, and sends exactly one stop signal to each worker once the queue drains.

        Parameters:
            None

        Returns:
            None
        """
        r0 = TaskResult(0, True, result='r0')
        r1 = TaskResult(1, True, result='r1')

        mock_comm = Mock()
        mock_comm.recv.side_effect = [None, None, (0, r0), (1, r1)]

        mock_status = Mock()
        mock_status.Get_source.side_effect = [1, 2, 1, 2]

        manager = self._serial_manager()
        manager.comm = mock_comm
        manager.size = 3

        mock_mpi = MagicMock()
        mock_mpi.Status.return_value = mock_status

        with patch('mpasdiag.processing.parallel.MPI', mock_mpi):
            results = manager._mpi_dispatch_master(2)

        assert results == [r0, r1]
        assert mock_comm.send.call_count == 4

        stop_calls = [c for c in mock_comm.send.call_args_list
                      if c.kwargs.get('tag') == manager._TAG_STOP]
        
        assert len(stop_calls) == 2

    def test_mpi_dispatch_worker(self: 'TestWorkStealing') -> None:
        """
        This test verifies the worker half of the work-stealing scheduler. It mocks the communicator to hand back two task indices and then a stop signal, and asserts that _mpi_dispatch_worker executes the function on exactly the two assigned tasks in order and issues one readiness message per round-trip before exiting on the stop tag.

        Parameters:
            None

        Returns:
            None
        """
        manager = self._serial_manager()

        mock_comm = Mock()
        mock_comm.recv.side_effect = [0, 1, None]
        manager.comm = mock_comm

        mock_status = Mock()

        mock_status.Get_tag.side_effect = [
            manager._TAG_TASK, manager._TAG_TASK, manager._TAG_STOP,
        ]

        executed = []

        def func(task: str) -> str:
            """
            This is a simple function that takes a task (in this case, a string) and returns its uppercase version. It is used in the test for the worker half of the work-stealing scheduler to verify that the worker correctly executes the assigned tasks and produces results. The function also appends the original task to the `executed` list, allowing the test to assert that the correct tasks were executed in order. This helps confirm that the worker logic processes tasks as expected before receiving a stop signal.

            Parameters:
                task (str): The input task to be processed (a string).

            Returns:
                str: The uppercase version of the input task.
            """
            executed.append(task)
            return task.upper()

        mock_mpi = MagicMock()
        mock_mpi.Status.return_value = mock_status

        with patch('mpasdiag.processing.parallel.MPI', mock_mpi):
            manager._mpi_dispatch_worker(func, ['a', 'b'])

        assert executed == ['a', 'b']
        assert mock_comm.send.call_count == 3

    def test_mpi_work_stealing_master_branch(self: 'TestWorkStealing') -> None:
        """
        This test verifies that _mpi_work_stealing runs the master dispatch loop and returns the gathered results on the master rank. It mocks the manager to be the master rank and patches the _mpi_dispatch_master method to return a canned list of results, then asserts that _mpi_work_stealing returns those results as expected. This confirms that the master branch of the work-stealing scheduler correctly initiates the dispatch protocol and collects results from workers, ultimately returning the aggregated results to the caller.

        Parameters:
            None

        Returns:
            None
        """
        manager = self._serial_manager()
        manager.is_master = True
        canned = [TaskResult(0, True)]

        with patch.object(manager, '_mpi_dispatch_master', return_value=canned) as disp:
            result = manager._mpi_work_stealing(lambda t: t, ['a'])

        disp.assert_called_once_with(1)
        assert result == canned

    def test_mpi_work_stealing_worker_branch(self: 'TestWorkStealing') -> None:
        """
        This test verifies that _mpi_work_stealing runs the worker dispatch loop and returns None on worker ranks. It mocks the manager to be a worker rank and patches the _mpi_dispatch_worker method, then asserts that _mpi_work_stealing returns None on the worker. This confirms that the worker branch of the work-stealing scheduler correctly processes assigned tasks and does not attempt to return results, adhering to the expected behavior of workers in an MPI context.

        Parameters:
            None

        Returns:
            None
        """
        manager = self._serial_manager()
        manager.is_master = False

        with patch.object(manager, '_mpi_dispatch_worker') as disp:
            result = manager._mpi_work_stealing(lambda t: t, ['a', 'b'])

        disp.assert_called_once()
        assert result is None

    def test_mpi_map_uses_work_stealing(self: 'TestWorkStealing') -> None:
        """
        This test verifies that the _mpi_map method uses the work-stealing scheduler when the distributor is set to DYNAMIC. It mocks the manager to be the master rank, sets up a mock communicator, and patches the _mpi_work_stealing method to return a canned list of results. The test then calls _mpi_map and asserts that _mpi_work_stealing was called and that the results are returned as expected. This confirms that when the load balance strategy is set to dynamic, the MPI mapping logic correctly utilizes the work-stealing scheduler to distribute tasks and collect results across ranks.

        Parameters:
            None

        Returns:
            None
        """
        mock_comm = Mock()
        mock_comm.bcast.return_value = list(range(6))
        mock_comm.allreduce.return_value = 1.0

        canned = [TaskResult(i, True) for i in range(6)]

        manager = self._serial_manager()
        manager.backend = 'mpi'
        manager.comm = mock_comm
        manager.size = 4
        manager.is_master = True
        manager.distributor = MPASTaskDistributor(mock_comm, LoadBalanceStrategy.DYNAMIC)
        manager.collector = MPASResultCollector(mock_comm)

        mock_mpi = MagicMock()
        mock_mpi.Wtime.return_value = 0.0

        with patch('mpasdiag.processing.parallel.MPI', mock_mpi):
            with patch.object(manager, '_mpi_work_stealing', return_value=canned) as ws:
                results = manager._mpi_map(lambda t: t, list(range(6)))

        ws.assert_called_once()
        assert results is not None
        assert len(results) == 6


if __name__ == "__main__":
    pytest.main([__file__])
