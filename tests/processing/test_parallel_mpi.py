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
import matplotlib.pyplot as plt
from typing import List, Generator
from unittest.mock import Mock, MagicMock, patch

from mpasdiag.processing.parallel import (
    MPASParallelManager,
    LoadBalanceStrategy,
    TaskResult,
    ParallelStats,
    MPASTaskDistributor,
    MPASResultCollector,
)

from tests.test_data_helpers import assert_expected_public_methods


@pytest.fixture
def sample_tasks() -> List[int]:
    """Simple list of integer tasks from 0 to 9 for testing purposes."""
    return list(range(10))


class TestBarrierAndFinalizeModule:
    """ Tests for synchronization helpers such as `barrier` and `finalize` across MPI and non-MPI backends. """
    
    def test_barrier_serial(self: "TestBarrierAndFinalizeModule") -> None:
        """
        This test ensures that calling `barrier` in serial mode is safe and does not raise any exceptions. Since the serial backend does not involve actual parallel processes, the `barrier` method should effectively be a no-op. The test constructs a serial manager and calls `barrier`, asserting that it completes without raising any errors. This confirms that the `barrier` method is implemented in a way that is compatible with non-parallel execution contexts, allowing code that uses barriers to run seamlessly regardless of the backend. 

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        assert_expected_public_methods(manager, 'MPASParallelManager')
        manager.barrier() 
    
    def test_barrier_multiprocessing(self: "TestBarrierAndFinalizeModule") -> None:
        """
        This test verifies that calling `barrier` in multiprocessing mode does not raise any exceptions. In the multiprocessing backend, the `barrier` method should synchronize worker processes, but since this is a test environment without actual parallel execution, it should still complete without errors. The test constructs a multiprocessing manager and calls `barrier`, asserting that it completes successfully. This confirms that the `barrier` method is implemented to work correctly in the multiprocessing context, allowing for synchronization without causing issues in a testing environment. 

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(backend='multiprocessing', verbose=False)
        assert_expected_public_methods(manager, 'MPASParallelManager')
        manager.barrier() 
    
    def test_finalize_serial(self: "TestBarrierAndFinalizeModule") -> None:
        """
        This test confirms that calling `finalize` in serial mode is safe and does not raise any exceptions. Since the serial backend does not involve actual parallel processes or resources, the `finalize` method should effectively be a no-op. The test constructs a serial manager and calls `finalize`, asserting that it completes without raising any errors. This ensures that the `finalize` method is implemented in a way that is compatible with non-parallel execution contexts, allowing code that uses finalization to run seamlessly regardless of the backend. 

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        assert_expected_public_methods(manager, 'MPASParallelManager')
        manager.finalize() 
    
    def test_finalize_multiprocessing(self: "TestBarrierAndFinalizeModule") -> None:
        """
        This test verifies that calling `finalize` in multiprocessing mode does not raise any exceptions. In the multiprocessing backend, the `finalize` method should clean up any resources associated with worker processes, but since this is a test environment without actual parallel execution, it should still complete without errors. The test constructs a multiprocessing manager and calls `finalize`, asserting that it completes successfully. This confirms that the `finalize` method is implemented to work correctly in the multiprocessing context, allowing for proper cleanup without causing issues in a testing environment. 

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(backend='multiprocessing', verbose=False)
        assert_expected_public_methods(manager, 'MPASParallelManager')
        manager.finalize() 


class TestExecuteLocalTasks:
    """ Tests for executing tasks locally under different error policies (abort/continue/collect). """
    
    def test_execute_local_tasks_with_abort_policy_mpi(self: "TestExecuteLocalTasks") -> None:
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
    
    def test_execute_local_tasks_with_abort_policy_non_mpi(self: "TestExecuteLocalTasks") -> None:
        """
        This test confirms that when the error policy is set to 'abort' and the manager is not running under MPI (e.g., in serial mode), any exception raised during local task execution is propagated as a normal exception rather than causing an abort. It constructs a manager with the 'serial' backend, sets the error policy to 'abort', and defines a `failing_func` that raises a `RuntimeError`. The test calls `_execute_local_tasks` with this function and asserts that a `RuntimeError` is raised with the expected message. This ensures that in non-MPI contexts, critical failures during local task execution are handled through standard exception propagation, allowing users to catch and manage errors without crashing the entire process. 

        Parameters:
            None

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
        assert_expected_public_methods(manager, 'MPASParallelManager')
    
    def test_execute_local_tasks_verbose_error_output(self: "TestExecuteLocalTasks", 
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


class TestBarrierAndFinalizeAdditional:
    """ Tests for synchronization and cleanup helpers like `barrier` and `finalize`. """
    
    def test_barrier_with_mpi(self: "TestBarrierAndFinalizeAdditional") -> None:
        """
        This test verifies that the `barrier` method correctly calls the MPI `Barrier` function when MPI is available. It creates a mock MPI communicator, patches the `MPI_AVAILABLE` flag to True, and asserts that calling `barrier` on the manager results in a call to the communicator's `Barrier` method. This ensures that synchronization across ranks is properly implemented when using MPI, allowing for coordinated execution in distributed runs. 

        Parameters:
            None

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
                assert_expected_public_methods(manager, 'MPASParallelManager')
                manager.comm = mock_comm                
                manager.barrier()                
                mock_comm.Barrier.assert_called_once()
    
    def test_barrier_without_mpi(self: "TestBarrierAndFinalizeAdditional") -> None:
        """
        This test confirms that the `barrier` method does not raise an error when MPI is not available. It constructs a manager with the 'serial' backend and calls `barrier`, asserting that it completes without raising any exceptions. This ensures that the `barrier` method can be safely called in non-MPI contexts, allowing for code that may be shared between parallel and serial execution paths without requiring conditional checks for MPI availability. 

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        assert_expected_public_methods(manager, 'MPASParallelManager')
        manager.barrier()
    
    def test_finalize_with_mpi(self: "TestBarrierAndFinalizeAdditional") -> None:
        """
        This test verifies that the `finalize` method performs an MPI barrier and prints a finalization message when MPI is available. It creates a mock MPI communicator, patches the `MPI_AVAILABLE` flag to True, and captures stdout during the call to `finalize`. The test asserts that the communicator's `Barrier` method is called and that the output includes a message indicating that the manager has been finalized. This ensures that proper synchronization and cleanup messages are emitted when finalizing in an MPI context, aiding users in understanding when parallel resources have been released. 

        Parameters:
            None

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
                assert_expected_public_methods(manager, 'MPASParallelManager')
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
            None

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=True)
        assert_expected_public_methods(manager, 'MPASParallelManager')
        
        import io
        from contextlib import redirect_stdout        
        f = io.StringIO()

        with redirect_stdout(f):
            manager.finalize()
        
        output = f.getvalue()
        assert "finalized" in output


class TestMPIMapExecution:
    """ Tests of MPI-mapped execution paths including master/worker separation and communicator-based control flow. """
    
    def test_mpi_map_verbose_output(self: "TestMPIMapExecution") -> None:
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
    
    def test_mpi_map_assertions(self: "TestMPIMapExecution") -> None:
        """
        This test confirms that the `_mpi_map` method raises appropriate assertions when critical components like the MPI communicator, task distributor, or result collector are not initialized. It creates a manager in serial mode and sequentially sets the `comm`, `distributor`, and `collector` attributes to None, asserting that each call to `_mpi_map` raises an `AssertionError` with the expected message. This ensures that the method enforces necessary preconditions for MPI execution, preventing runtime errors and guiding developers to properly configure the manager before attempting parallel mapping. 

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(backend='serial', verbose=False)
        assert_expected_public_methods(manager, 'MPASParallelManager')

        manager.comm = None

        with pytest.raises(AssertionError, match="MPI communicator must be initialized"):
            manager._mpi_map(lambda x: x, [1, 2, 3])
        
        mock_comm = Mock()
        manager.comm = mock_comm
        manager.distributor = None

        with pytest.raises(AssertionError, match="Task distributor must be initialized"):
            manager._mpi_map(lambda x: x, [1, 2, 3])
        
        manager.distributor = MPASTaskDistributor(mock_comm, LoadBalanceStrategy.STATIC)
        assert_expected_public_methods(manager.distributor, 'MPASTaskDistributor')

        manager.collector = None

        with pytest.raises(AssertionError, match="Result collector must be initialized"):
            manager._mpi_map(lambda x: x, [1, 2, 3])


class TestMPIMapReturnValue:
    """ Tests ensuring `_mpi_map` returns `None` on worker ranks and only the master returns aggregated results. """
    
    def test_mpi_map_returns_none_on_worker(self: "TestMPIMapReturnValue") -> None:
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


