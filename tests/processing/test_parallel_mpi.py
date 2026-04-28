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
                
                def simple_func(task):
                    return task * 2
                
                results = manager._mpi_map(simple_func, list(range(10)))

                assert results is None


class TestSetupMPIBackendFallback:
    """Tests that an exception raised inside _setup_mpi_backend causes graceful fallback to multiprocessing."""

    def test_mpi_setup_exception_falls_back_to_multiprocessing(
        self: 'TestSetupMPIBackendFallback',
    ) -> None:
        """
        This test verifies that when MPI is nominally available (MPI_AVAILABLE=True)
        but the underlying communicator raises an exception during Get_rank, the
        MPASParallelManager catches the exception and falls back to the multiprocessing
        backend (lines 379-383). It patches MPI_AVAILABLE to True and makes
        mock_comm.Get_rank raise a RuntimeError, then asserts that the manager ends
        up in multiprocessing mode and that the verbose fallback messages are printed.

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
    """Tests that parallel_map dispatches to _mpi_map when backend is 'mpi' (line 469)."""

    def test_parallel_map_with_mpi_backend_calls_mpi_map(
        self: 'TestParallelMapMPIDispatch',
    ) -> None:
        """
        This test verifies that calling parallel_map on a manager whose backend is
        'mpi' correctly dispatches to _mpi_map (line 469) rather than the
        multiprocessing or serial paths. It sets up a mock MPI communicator at rank 0
        with size 4, constructs an MPASParallelManager in MPI mode, and then calls
        parallel_map (not _mpi_map directly). The test asserts that results are
        returned from the master rank, confirming that the dispatch branch executes.

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
                    return task * 2

                results = manager.parallel_map(simple_func, list(range(6)))

        assert results is not None
        assert_expected_public_methods(manager, 'MPASParallelManager')


class TestBarrierAndFinalize:
    """Tests for the barrier() and finalize() methods across MPI and non-MPI backends."""

    def test_barrier_calls_comm_barrier_when_mpi_backend(
        self: 'TestBarrierAndFinalize',
    ) -> None:
        """
        This test verifies that calling barrier() on a manager in MPI mode invokes
        comm.Barrier() exactly once (lines 766-767). It constructs an MPI-mode manager
        with a mock communicator and asserts that Barrier was called after invoking
        barrier().

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

    def test_barrier_is_noop_for_serial_backend(
        self: 'TestBarrierAndFinalize',
    ) -> None:
        """
        This test confirms that barrier() is a no-op for a serial-mode manager — it
        completes without error and does not attempt any MPI communication. The serial
        backend has comm=None, so the MPI branch inside barrier() must not execute.

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        assert_expected_public_methods(manager, 'MPASParallelManager')
        manager.barrier()  # must not raise

    def test_finalize_mpi_calls_barrier_and_prints(
        self: 'TestBarrierAndFinalize',
    ) -> None:
        """
        This test verifies that finalize() on an MPI-mode manager calls barrier()
        (and therefore comm.Barrier()) and prints the finalization message when
        is_master=True and verbose=True (lines 779-783). It asserts that Barrier was
        called and that "finalized" appears in stdout.

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

    def test_finalize_serial_prints_message(
        self: 'TestBarrierAndFinalize',
    ) -> None:
        """
        This test confirms that finalize() on a serial-mode master manager prints
        the finalization message without invoking any MPI communication (lines 782-783
        reached via the non-MPI path). It captures stdout and asserts "finalized"
        appears in the output.

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
