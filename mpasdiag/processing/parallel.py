#!/usr/bin/env python3

"""
MPAS Parallel Processing Manager

This module provides comprehensive parallel processing capabilities for MPAS visualization and analysis tasks enabling efficient distributed processing across multiple cores or compute nodes with automatic backend selection. It implements the MPASParallelManager class that coordinates parallel execution using either MPI-based parallelization (mpi4py) for distributed-memory systems or Python multiprocessing for shared-memory systems, with automatic fallback to serial execution when parallel backends are unavailable. The parallel manager handles dynamic load balancing for uneven workloads, task distribution across worker processes with memory-efficient communication, result gathering and aggregation from workers, fault tolerance with comprehensive error handling, and progress monitoring with detailed reporting. Core capabilities include automatic detection of available parallel backends (MPI via mpirun/mpiexec or multiprocessing), configurable worker count and load balancing strategies, support for both plotting and analysis task types, memory-efficient data transfer between processes, and graceful degradation to serial mode ensuring robust operation across diverse computing environments from laptops to HPC clusters.

Classes:
    MPASParallelManager: Main class for managing parallel processing with automatic backend selection and task coordination.
    MPASTaskDistributor: Load balancing and task distribution utilities for MPI-based parallel execution.
    MPASResultCollector: Result gathering and aggregation from distributed workers in MPI mode.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import os
import sys
import time
import warnings
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Pool, cpu_count, get_context
from functools import partial

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    MPI = None 
    warnings.warn(
        "mpi4py is not available. Parallel processing will use Python multiprocessing. "
        "To enable MPI parallelization, install mpi4py: pip install mpi4py",
        UserWarning
    )


class LoadBalanceStrategy(Enum):
    """
    Enumeration of available task distribution strategies for load balancing across workers. These strategies determine how computational work is divided among parallel processes to optimize throughput and minimize idle time. The choice of strategy depends on task characteristics such as execution time variability and data locality requirements. Different strategies excel in different scenarios from uniform to highly heterogeneous workloads.

    Attributes:
        STATIC (str): Equal distribution of tasks at initialization.
        DYNAMIC (str): Dynamic work stealing allowing workers to request new tasks.
        BLOCK (str): Contiguous blocks of tasks per worker for cache locality.
        CYCLIC (str): Round-robin allocation for heterogeneous task durations.
    """
    STATIC = "static"      # Equal distribution at start
    DYNAMIC = "dynamic"    # Dynamic work stealing
    BLOCK = "block"        # Contiguous blocks per worker
    CYCLIC = "cyclic"      # Round-robin distribution


class ErrorPolicy(Enum):
    """
    Enumeration of error handling strategies for managing task failures in parallel execution. These policies determine the manager's response when individual tasks encounter exceptions during processing. The ABORT policy provides fail-fast behavior for critical operations, CONTINUE enables best-effort completion, and COLLECT enables comprehensive error reporting. The selected policy affects both worker behavior and final result aggregation.

    Attributes:
        ABORT (str): Terminate all processes immediately upon first error.
        CONTINUE (str): Skip failed tasks and continue processing remaining tasks.
        COLLECT (str): Record all errors while completing all tasks for comprehensive reporting.
    """
    ABORT = "abort"        # Abort all on first error
    CONTINUE = "continue"  # Continue despite errors
    COLLECT = "collect"    # Collect all errors and report


@dataclass
class TaskResult:
    """
    Container for storing the outcome of a single task execution in parallel workflows. This dataclass encapsulates all relevant information about task completion including success status, result data, error messages, and performance metrics. It provides a standardized format for collecting and aggregating results from distributed workers in both MPI and multiprocessing backends. The structure enables comprehensive error tracking and performance analysis across parallel execution.

    Attributes:
        task_id (int): Unique identifier for the task within the batch.
        success (bool): Flag indicating whether task completed without errors.
        result (Any): Output data returned by the task function, None if failed.
        error (Optional[str]): Error message and traceback if task failed, None if successful.
        execution_time (float): Wall-clock time in seconds taken to execute the task.
        worker_rank (int): MPI rank or worker ID that processed this task.
    """
    task_id: int
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    worker_rank: int = 0


@dataclass
class ParallelStats:
    """
    Aggregated statistics for analyzing parallel execution performance and efficiency. This dataclass collects comprehensive metrics about task completion, timing, and load distribution across workers to evaluate parallel performance. It enables identification of bottlenecks, load imbalances, and overall efficiency of the parallel execution. The statistics support performance tuning and optimization of parallel MPAS workflows.

    Attributes:
        total_tasks (int): Total number of tasks distributed across all workers.
        completed_tasks (int): Number of tasks that finished successfully without errors.
        failed_tasks (int): Number of tasks that encountered errors during execution.
        total_time (float): Cumulative CPU time in seconds spent on all tasks.
        worker_times (Dict[int, float]): Dictionary mapping worker rank/ID to their total execution time.
        load_imbalance (float): Metric quantifying workload imbalance as (max_time - avg_time) / avg_time.
    """
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_time: float = 0.0
    worker_times: Dict[int, float] = field(default_factory=dict)
    load_imbalance: float = 0.0


def _multiprocessing_task_wrapper(args: Tuple[int, Any, Callable, str, Tuple, Dict]) -> 'TaskResult':
    """
    Wrap individual tasks for execution in multiprocessing workers with error handling and timing. This function serves as a picklable wrapper that can be passed to multiprocessing.Pool.map(). It executes the provided function with the given task and arguments, captures any exceptions, and returns a TaskResult object with execution details. The wrapper must be defined at module level to ensure it can be pickled by the multiprocessing module.

    Parameters:
        args (tuple): Packed arguments containing (task_id, task, func, error_policy_value, func_args, func_kwargs).
            - task_id (int): Unique identifier for this task.
            - task (Any): The task object to process.
            - func (Callable): The function to execute on the task.
            - error_policy_value (str): Error handling policy ('abort', 'continue', or 'collect').
            - func_args (tuple): Positional arguments to pass to func.
            - func_kwargs (dict): Keyword arguments to pass to func.

    Returns:
        TaskResult: Object containing execution results including success status, output, error messages, and execution time.
    """
    task_id, task, func, error_policy_value, func_args, func_kwargs = args
    result = TaskResult(task_id=task_id, success=False, worker_rank=0)
    task_start = time.time()
    
    try:
        output = func(task, *func_args, **func_kwargs)
        result.success = True
        result.result = output
    except Exception as e:
        result.success = False
        result.error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
        if error_policy_value == 'abort':
            raise
    finally:
        result.execution_time = time.time() - task_start
    
    return result


class MPASTaskDistributor:
    """
    Manages task distribution and load balancing across MPI workers for efficient parallel processing. This class implements multiple load balancing strategies to optimize work distribution based on workload characteristics and cluster topology. It handles the mapping of tasks to workers according to the selected strategy (STATIC, DYNAMIC, BLOCK, or CYCLIC) while accounting for process rank and total size. The distributor ensures each worker receives an appropriate subset of tasks to minimize idle time and maximize throughput.
    """
    
    def __init__(self, comm: Any, strategy: LoadBalanceStrategy = LoadBalanceStrategy.DYNAMIC) -> None:
        """
        Initialize the task distributor with MPI communicator and load balancing strategy. This constructor sets up the distributor to manage how computational tasks are divided among available MPI workers. The strategy parameter determines whether tasks are distributed statically at the beginning, dynamically as workers become available, in blocks, or cyclically round-robin. The communicator provides information about rank and total number of processes.

        Parameters:
            comm (MPI.Comm): MPI communicator object providing process rank and size information.
            strategy (LoadBalanceStrategy): Load balancing strategy to use (STATIC, DYNAMIC, BLOCK, or CYCLIC). Defaults to DYNAMIC.

        Returns:
            None
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.strategy = strategy
    
    def distribute_tasks(self, tasks: List[Any]) -> List[Tuple[int, Any]]:
        """
        Distribute computational tasks to this MPI worker based on the selected load balancing strategy. This method takes the complete list of tasks and returns only those tasks assigned to the current worker rank. The distribution strategy (STATIC, DYNAMIC, BLOCK, or CYCLIC) was chosen during initialization and determines how tasks are divided among workers. This ensures balanced workload across all available MPI processes.

        Parameters:
            tasks (List[Any]): Complete list of all tasks to be distributed across workers.

        Returns:
            List[Tuple[int, Any]]: List of (task_id, task) tuples assigned to this specific worker based on rank and strategy.
        """
        n_tasks = len(tasks)
        
        if self.strategy == LoadBalanceStrategy.STATIC:
            return self._static_distribution(tasks)
        elif self.strategy == LoadBalanceStrategy.DYNAMIC:
            return self._dynamic_distribution(tasks)
        elif self.strategy == LoadBalanceStrategy.BLOCK:
            return self._block_distribution(tasks)
        elif self.strategy == LoadBalanceStrategy.CYCLIC:
            return self._cyclic_distribution(tasks)
        else:
            return self._static_distribution(tasks)
    
    def _static_distribution(self, tasks: List[Any]) -> List[Tuple[int, Any]]:
        """
        Distribute tasks evenly among workers using static allocation at initialization. This strategy divides the total number of tasks as equally as possible among all workers at the start of execution. It handles remainder tasks by assigning one extra task to the first N workers where N is the remainder. This is the simplest and fastest distribution method but doesn't account for varying task execution times.

        Parameters:
            tasks (List[Any]): Complete list of tasks to distribute.

        Returns:
            List[Tuple[int, Any]]: List of (task_id, task) tuples assigned to this worker using static distribution.
        """
        n_tasks = len(tasks)
        tasks_per_worker = n_tasks // self.size
        remainder = n_tasks % self.size
        
        if self.rank < remainder:
            start = self.rank * (tasks_per_worker + 1)
            end = start + tasks_per_worker + 1
        else:
            start = self.rank * tasks_per_worker + remainder
            end = start + tasks_per_worker
        
        return [(i, tasks[i]) for i in range(start, min(end, n_tasks))]
    
    def _block_distribution(self, tasks: List[Any]) -> List[Tuple[int, Any]]:
        """
        Distribute tasks in contiguous blocks to each worker for cache efficiency. This strategy divides the task list into equal-sized contiguous blocks and assigns one block per worker. Block distribution is particularly effective when tasks have spatial or temporal locality that benefits from keeping related tasks on the same worker. The last block may be smaller if the total number of tasks doesn't divide evenly by the number of workers.

        Parameters:
            tasks (List[Any]): Complete list of tasks to distribute in blocks.

        Returns:
            List[Tuple[int, Any]]: List of (task_id, task) tuples forming a contiguous block for this worker.
        """
        n_tasks = len(tasks)
        block_size = (n_tasks + self.size - 1) // self.size
        start = self.rank * block_size
        end = min(start + block_size, n_tasks)
        
        return [(i, tasks[i]) for i in range(start, end)]
    
    def _cyclic_distribution(self, tasks: List[Any]) -> List[Tuple[int, Any]]:
        """
        Distribute tasks using round-robin allocation across workers for load balancing. This strategy assigns tasks in a cyclic pattern where worker 0 gets tasks 0, N, 2N..., worker 1 gets tasks 1, N+1, 2N+1..., and so on. Round-robin distribution is effective when task execution times vary significantly and you want to distribute short and long tasks evenly across workers. This pattern maximizes load balancing for heterogeneous workloads.

        Parameters:
            tasks (List[Any]): Complete list of tasks to distribute cyclically.

        Returns:
            List[Tuple[int, Any]]: List of (task_id, task) tuples assigned in round-robin pattern to this worker.
        """
        return [(i, tasks[i]) for i in range(self.rank, len(tasks), self.size)]
    
    def _dynamic_distribution(self, tasks: List[Any]) -> List[Tuple[int, Any]]:
        """
        Distribute tasks dynamically with potential for work stealing between workers. This strategy initially uses static distribution as a baseline but enables the parallel manager to implement dynamic work stealing on top of it. Dynamic distribution is most effective for workloads with highly variable task execution times where workers that finish early can steal work from busy workers. Currently implemented as static distribution with future work-stealing capability.

        Parameters:
            tasks (List[Any]): Complete list of tasks for dynamic distribution.

        Returns:
            List[Tuple[int, Any]]: List of (task_id, task) tuples initially distributed statically with dynamic rebalancing potential.
        """
        return self._static_distribution(tasks)


class MPASResultCollector:
    """
    Collects and aggregates computational results from distributed parallel workers into unified outputs. This class provides methods for gathering task results using MPI collective communication and computing comprehensive execution statistics. It handles the complexity of inter-process data collection and flattening nested result structures from multiple workers. The collector is essential for combining distributed computation outputs and analyzing parallel performance metrics on the master process.
    """
    
    def __init__(self, comm: Any) -> None:
        """
        Initialize the result collector with MPI communicator for gathering worker results. This constructor sets up the collector to receive and aggregate results from all parallel workers. The communicator provides the mechanism for inter-process communication and helps identify the master process that will collect all results. This class is essential for combining distributed computation results into a single unified output.

        Parameters:
            comm (MPI.Comm): MPI communicator object providing rank and communication capabilities.

        Returns:
            None
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
    
    def gather_results(self, local_results: List[TaskResult]) -> Optional[List[TaskResult]]:
        """
        Gather results from all workers to the master process using MPI collective communication. This method collects task results computed by each worker and aggregates them on the master process (rank 0). The MPI gather operation ensures all worker results are transmitted efficiently to the master. Only the master process receives the complete set of results, while other workers receive None. This pattern enables centralized result processing and statistics computation.

        Parameters:
            local_results (List[TaskResult]): Results computed by this specific worker process.

        Returns:
            Optional[List[TaskResult]]: Flattened list of all results from all workers (only on rank 0), None on other ranks.
        """
        all_results = self.comm.gather(local_results, root=0)
        
        if self.rank == 0:
            flattened = []
            for worker_results in all_results:
                flattened.extend(worker_results)
            return flattened
        return None
    
    def compute_statistics(self, results: List[TaskResult]) -> ParallelStats:
        """
        Compute comprehensive execution statistics from the collected task results. This method analyzes all completed tasks to calculate metrics including total/completed/failed task counts, cumulative execution time, and per-worker timing statistics. The statistics provide insights into parallel efficiency, load balancing, and error rates. These metrics are essential for performance tuning and identifying bottlenecks in parallel workflows.

        Parameters:
            results (List[TaskResult]): Complete list of all task results from all workers.

        Returns:
            ParallelStats: Statistics object containing execution metrics including task counts, timing data, and success rates.
        """
        stats = ParallelStats()
        stats.total_tasks = len(results)
        stats.completed_tasks = sum(1 for r in results if r.success)
        stats.failed_tasks = sum(1 for r in results if not r.success)
        stats.total_time = sum(r.execution_time for r in results)
        
        for result in results:
            rank = result.worker_rank
            stats.worker_times[rank] = stats.worker_times.get(rank, 0.0) + result.execution_time
        
        if stats.worker_times:
            max_time = max(stats.worker_times.values())
            avg_time = sum(stats.worker_times.values()) / len(stats.worker_times)
            stats.load_imbalance = (max_time - avg_time) / avg_time if avg_time > 0 else 0.0
        
        return stats


class MPASParallelManager:
    """
    High-level interface for parallel execution of MPAS visualization and analysis tasks. This class provides a unified API for distributed computing that automatically selects the best available backend (MPI, multiprocessing, or serial). It manages task distribution, worker coordination, error handling, and result collection while abstracting away backend-specific complexity. The manager supports multiple load balancing strategies, configurable error policies, and comprehensive performance monitoring for both small-scale and large-scale parallel workflows.
    
    Features:
    - Automatic MPI initialization and management
    - Multiple load balancing strategies
    - Fault tolerance with configurable error policies
    - Progress monitoring and logging
    - Performance statistics and load balancing metrics
    - Graceful fallback to serial execution if MPI unavailable
    
    Examples:
        >>> manager = MPASParallelManager()
        >>> def plot_file(filepath):
        ...     # Your plotting code
        ...     return filepath
        >>> files = ['file1.nc', 'file2.nc', 'file3.nc']
        >>> results = manager.parallel_map(plot_file, files)
    
        >>> manager.set_error_policy('continue')
        >>> results = manager.parallel_map(plot_file, files)
        >>> stats = manager.get_statistics()
        >>> print(f"Success: {stats.completed_tasks}/{stats.total_tasks}")
    """
    
    def __init__(
        self, 
        load_balance_strategy: Union[str, LoadBalanceStrategy] = "dynamic",
        verbose: bool = True,
        backend: Optional[str] = None,
        n_workers: Optional[int] = None
    ) -> None:
        """
        Initialize the parallel processing manager with backend selection and configuration. This constructor automatically detects and configures the best available parallelization backend (MPI, multiprocessing, or serial). It sets up the communicator, determines process rank and size, and initializes load balancing components. The backend selection follows a priority: MPI if available and requested, then multiprocessing, finally serial as fallback. The manager handles all the complexity of distributed computing setup.

        Parameters:
            load_balance_strategy (str or LoadBalanceStrategy): Strategy for distributing tasks across workers in MPI mode (STATIC, DYNAMIC, BLOCK, or CYCLIC). Defaults to DYNAMIC.
            verbose (bool): Enable verbose output for debugging and progress monitoring. Defaults to True.
            backend (str, optional): Force specific backend ('mpi', 'multiprocessing', or 'serial'). If None, automatically selects best available backend.
            n_workers (int, optional): Number of worker processes for multiprocessing backend. If None, uses cpu_count() - 1.

        Returns:
            None
        """
        self.verbose = verbose
        self.error_policy = ErrorPolicy.COLLECT
        self.backend = backend
        self.n_workers = n_workers
        
        if backend == 'mpi' or (backend is None and MPI_AVAILABLE):
            try:
                assert MPI_AVAILABLE and MPI is not None, "MPI not available"
                self.comm = MPI.COMM_WORLD
                self.rank = self.comm.Get_rank()
                self.size = self.comm.Get_size()
                
                if self.size > 1:
                    self.backend = 'mpi'
                    self.is_master = (self.rank == 0)
                    
                    if isinstance(load_balance_strategy, str):
                        load_balance_strategy = LoadBalanceStrategy(load_balance_strategy)
                    
                    self.distributor = MPASTaskDistributor(self.comm, load_balance_strategy)
                    self.collector = MPASResultCollector(self.comm)
                else:
                    self.backend = 'multiprocessing'
                    self.rank = 0
                    self.size = n_workers or max(1, cpu_count() - 1)
                    self.is_master = True
                    self.comm = None
                    self.distributor = None
                    self.collector = None
            except Exception as e:
                if verbose:
                    print(f"MPI initialization failed: {e}")
                    print("Falling back to multiprocessing backend")
                self.backend = 'multiprocessing'
                self.rank = 0
                self.size = n_workers or max(1, cpu_count() - 1)
                self.is_master = True
                self.comm = None
                self.distributor = None
                self.collector = None
        elif backend == 'multiprocessing' or (backend is None and not MPI_AVAILABLE):
            self.backend = 'multiprocessing'
            self.rank = 0
            self.size = n_workers or max(1, cpu_count() - 1)
            self.is_master = True
            self.comm = None
            self.distributor = None
            self.collector = None
        else:
            self.backend = 'serial'
            self.comm = None
            self.rank = 0
            self.size = 1
            self.is_master = True
            self.distributor = None
            self.collector = None
        
        self.stats = None
        
        if self.is_master and self.verbose:
            if self.backend == 'mpi':
                print(f"MPASParallelManager initialized in MPI mode with {self.size} processes")
            elif self.backend == 'multiprocessing':
                print(f"MPASParallelManager initialized in multiprocessing mode with {self.size} workers")
            else:
                print("MPASParallelManager initialized in serial mode")
    
    def set_error_policy(self, policy: Union[str, ErrorPolicy]) -> None:
        """
        Set the error handling policy for task execution failures across workers. This method configures how the parallel manager responds when individual tasks fail during execution. The ABORT policy stops all processing immediately, CONTINUE skips failed tasks and proceeds with remaining work, and COLLECT records all errors while completing all tasks. The policy applies uniformly across all workers and affects both MPI and multiprocessing backends.

        Parameters:
            policy (str or ErrorPolicy): Error handling policy ('abort', 'continue', or 'collect') as string or ErrorPolicy enum value.

        Returns:
            None
        """
        if isinstance(policy, str):
            policy = ErrorPolicy(policy)
        self.error_policy = policy
    
    def parallel_map(
        self,
        func: Callable,
        tasks: List[Any],
        *args,
        **kwargs
    ) -> Optional[List[TaskResult]]:
        """
        Execute a function in parallel across all tasks using the configured backend. This is the main entry point for parallel execution that distributes tasks across available workers, executes the provided function on each task, and collects results. The method automatically handles backend selection (MPI, multiprocessing, or serial), task distribution, error handling, and result aggregation. It returns TaskResult objects containing both successful outputs and any errors encountered.

        Parameters:
            func (Callable): Function to execute for each task, should accept a task as first argument.
            tasks (List[Any]): Complete list of tasks to process in parallel (e.g., file paths, parameters).
            *args (tuple): Additional positional arguments to pass to func after the task argument.
            **kwargs (dict): Additional keyword arguments to pass to func.

        Returns:
            Optional[List[TaskResult]]: List of TaskResult objects (only on master process), None on worker processes.

        Examples:
            >>> def process_file(filepath, output_dir):
            ...     # Process file and return result
            ...     return result
            >>> manager = MPASParallelManager()
            >>> files = ['file1.nc', 'file2.nc', 'file3.nc']
            >>> results = manager.parallel_map(process_file, files, output_dir='./output/')
        """
        if self.backend == 'mpi':
            return self._mpi_map(func, tasks, *args, **kwargs)
        elif self.backend == 'multiprocessing':
            return self._multiprocessing_map(func, tasks, *args, **kwargs)
        else:
            return self._serial_map(func, tasks, *args, **kwargs)
    
    def _mpi_map(
        self,
        func: Callable,
        tasks: List[Any],
        *args,
        **kwargs
    ) -> Optional[List[TaskResult]]:
        """
        Execute tasks using MPI backend with distributed memory parallelization across nodes. This method implements the MPI-based parallel execution path using mpi4py for inter-process communication. It broadcasts the task list to all workers, distributes tasks according to the load balancing strategy, executes assigned tasks locally, and gathers results back to the master process. This backend is optimal for distributed computing environments with multiple nodes and provides the best scalability for large-scale problems.

        Parameters:
            func (Callable): Function to execute on each task.
            tasks (List[Any]): Complete list of tasks to distribute across MPI workers.
            *args (tuple): Additional positional arguments for func.
            **kwargs (dict): Additional keyword arguments for func.

        Returns:
            Optional[List[TaskResult]]: Aggregated results from all MPI workers (only on master rank 0), None on other ranks.
        """
        assert self.comm is not None, "MPI communicator must be initialized"
        assert self.distributor is not None, "Task distributor must be initialized"
        assert self.collector is not None, "Result collector must be initialized"
        
        tasks = self.comm.bcast(tasks, root=0)        
        local_tasks = self.distributor.distribute_tasks(tasks)
        
        if self.is_master and self.verbose:
            print(f"\nProcessing {len(tasks)} tasks across {self.size} workers...")
            print(f"Load balance strategy: {self.distributor.strategy.value}")
            print(f"Error policy: {self.error_policy.value}")
        
        local_results = self._execute_local_tasks(func, local_tasks, *args, **kwargs)
        all_results = self.collector.gather_results(local_results)
        
        if self.is_master:
            assert all_results is not None, "Results should not be None on master"
            self.stats = self.collector.compute_statistics(all_results)
            
            if self.verbose:
                self._print_statistics()
            
            return all_results
        
        return None
    
    def _multiprocessing_map(
        self,
        func: Callable,
        tasks: List[Any],
        *args,
        **kwargs
    ) -> List[TaskResult]:
        """
        Execute tasks using Python multiprocessing backend with platform-aware context selection. This method implements shared-memory parallelization using Python's multiprocessing module as a fallback when MPI is unavailable. It automatically selects the appropriate start method (spawn on macOS/Windows, fork on Linux) to avoid platform-specific issues like fork() deprecation warnings. The method creates a worker pool, distributes tasks, and collects results with comprehensive error handling and timing statistics.

        Parameters:
            func (Callable): Function to execute on each task.
            tasks (List[Any]): Complete list of tasks to process across workers.
            *args (tuple): Additional positional arguments for func.
            **kwargs (dict): Additional keyword arguments for func.

        Returns:
            List[TaskResult]: Complete list of results from all tasks with execution metrics.
        """
        if self.verbose:
            print(f"\nProcessing {len(tasks)} tasks across {self.size} workers...")
            print("Backend: Python multiprocessing")
            print(f"Error policy: {self.error_policy.value}")
        
        start_time = time.time()        

        task_args = [
            (i, task, func, self.error_policy.value, args, kwargs)
            for i, task in enumerate(tasks)
        ]        

        results = None
        
        if sys.platform == 'win32':
            ctx_methods = ['spawn']
        elif sys.platform == 'darwin':
            ctx_methods = ['spawn']
        else:
            ctx_methods = ['fork', 'spawn']
        
        for ctx_method in ctx_methods:
            try:
                ctx = get_context(ctx_method)
                with ctx.Pool(processes=self.size) as pool:
                    results = pool.map(_multiprocessing_task_wrapper, task_args)
                break 
                
            except Exception as e:
                if self.verbose:
                    print(f"Multiprocessing with '{ctx_method}' failed: {e}")
                    if ctx_method != ctx_methods[-1]:
                        print("Trying next method...")
                    else:
                        print("Falling back to serial execution")
                
                if ctx_method == ctx_methods[-1]:
                    results = [_multiprocessing_task_wrapper(args) for args in task_args]
        
        if results is None:
            results = [_multiprocessing_task_wrapper(args) for args in task_args]
        
        stats = ParallelStats()
        stats.total_tasks = len(results)
        stats.completed_tasks = sum(1 for r in results if r.success)
        stats.failed_tasks = sum(1 for r in results if not r.success)
        stats.total_time = sum(r.execution_time for r in results)
        self.stats = stats
        
        if self.verbose:
            wall_time = time.time() - start_time
            print(f"\n{'='*60}")
            print("PARALLEL EXECUTION STATISTICS")
            print(f"{'='*60}")
            print(f"Total tasks:       {stats.total_tasks}")
            print(f"Completed:         {stats.completed_tasks}")
            print(f"Failed:            {stats.failed_tasks}")
            if stats.total_tasks > 0:
                print(f"Success rate:      {100*stats.completed_tasks/stats.total_tasks:.1f}%")
            print(f"Total time:        {stats.total_time:.2f} seconds")
            print(f"Wall time:         {wall_time:.2f} seconds")
            if wall_time > 0:
                print(f"Speedup:           {stats.total_time/wall_time:.2f}x")
            print(f"{'='*60}\n")
        
        return results
    
    def _execute_local_tasks(
        self,
        func: Callable,
        local_tasks: List[Tuple[int, Any]],
        *args,
        **kwargs
    ) -> List[TaskResult]:
        """
        Execute tasks assigned to this worker with comprehensive error handling and timing. This method processes each task sequentially within a worker, executing the provided function and capturing results or errors. It implements the configured error policy (abort, continue, or collect) and tracks execution time for each task. The method handles exceptions gracefully, logging errors while optionally continuing execution depending on the policy. All results are wrapped in TaskResult objects for consistent downstream processing.

        Parameters:
            func (Callable): Function to execute on each task.
            local_tasks (List[Tuple[int, Any]]): List of (task_id, task) tuples assigned to this specific worker.
            *args (tuple): Additional positional arguments to pass to func.
            **kwargs (dict): Additional keyword arguments to pass to func.

        Returns:
            List[TaskResult]: Results from all tasks processed by this worker including successes and failures.
        """
        results = []
        
        for task_id, task in local_tasks:
            start_time = time.time()
            result = TaskResult(task_id=task_id, success=False, worker_rank=self.rank)
            
            try:
                output = func(task, *args, **kwargs)
                result.success = True
                result.result = output
                
            except Exception as e:
                result.success = False
                result.error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                
                if self.error_policy == ErrorPolicy.ABORT:
                    if self.backend == 'mpi' and self.comm is not None:
                        self.comm.Abort(1)
                    else:
                        raise
                
                if self.verbose:
                    print(f"[Rank {self.rank}] Error processing task {task_id}: {str(e)}")
            
            finally:
                result.execution_time = time.time() - start_time
                results.append(result)
        
        return results
    
    def _serial_map(
        self,
        func: Callable,
        tasks: List[Any],
        *args,
        **kwargs
    ) -> List[TaskResult]:
        """
        Execute all tasks sequentially when parallel backends are unavailable. This method serves as a fallback when MPI or multiprocessing backends cannot be initialized, processing tasks one at a time in a single process. It wraps tasks with IDs for consistent result handling, delegates execution to _execute_local_tasks, and computes statistics when collection is enabled. The method provides verbose logging to inform users of serial execution mode and timing information.

        Parameters:
            func (Callable): Function to execute on each task.
            tasks (List[Any]): List of tasks to process serially.
            *args (tuple): Additional positional arguments to pass to func.
            **kwargs (dict): Additional keyword arguments to pass to func.

        Returns:
            List[TaskResult]: Results from all tasks executed sequentially.
        """
        if self.verbose:
            print(f"\nProcessing {len(tasks)} tasks in serial mode...")
        
        local_tasks = [(i, task) for i, task in enumerate(tasks)]
        results = self._execute_local_tasks(func, local_tasks, *args, **kwargs)
        
        self.stats = self.collector.compute_statistics(results) if self.collector else None
        
        if self.verbose and self.stats:
            self._print_statistics()
        
        return results
    
    def get_statistics(self) -> Optional[ParallelStats]:
        """
        Retrieve computed execution statistics for the completed parallel operation. This method returns performance metrics including total execution time, task success/failure counts, and per-worker load balance information. Statistics are only available after a parallel_map operation completes and when result collection is enabled. Returns None if no statistics have been computed yet or collection was disabled.

        Parameters:
            None

        Returns:
            Optional[ParallelStats]: ParallelStats object containing execution metrics or None if unavailable.
        """
        return self.stats
    
    def _print_statistics(self) -> None:
        """
        Display formatted execution statistics to console on the master process only. This method prints a comprehensive summary of parallel execution metrics including total time, task counts, success/failure rates, and load balance across workers. Output is formatted in a readable table structure with section separators for clarity. The method respects the is_master flag to prevent duplicate output in MPI environments. Statistics must be available before calling this method.

        Parameters:
            None

        Returns:
            None
        """
        if not self.is_master or not self.stats:
            return
        
        print("\n" + "="*60)
        print("PARALLEL EXECUTION STATISTICS")
        print("="*60)
        print(f"Total tasks:       {self.stats.total_tasks}")
        print(f"Completed:         {self.stats.completed_tasks}")
        print(f"Failed:            {self.stats.failed_tasks}")
        print(f"Success rate:      {100*self.stats.completed_tasks/self.stats.total_tasks:.1f}%")
        print(f"Total time:        {self.stats.total_time:.2f} seconds")
        
        if len(self.stats.worker_times) > 1:
            print("\nPer-worker times:")
            for rank, worker_time in sorted(self.stats.worker_times.items()):
                print(f"  Rank {rank:2d}:  {worker_time:8.2f} seconds")
            print(f"\nLoad imbalance:    {100*self.stats.load_imbalance:.1f}%")
        
        print("="*60 + "\n")
    
    def barrier(self) -> None:
        """
        Synchronize all processes at a barrier point for coordinated execution. This method provides a synchronization primitive that blocks each process until all processes in the communicator reach the barrier. When using the MPI backend, it calls MPI_Barrier on the communicator. For other backends, the method has no effect and returns immediately. Use this to ensure all workers complete a phase before proceeding to the next.

        Parameters:
            None

        Returns:
            None
        """
        if self.backend == 'mpi' and self.comm is not None:
            self.comm.Barrier()
    
    def finalize(self) -> None:
        """
        Clean up parallel execution resources and synchronize final state. This method performs graceful shutdown of the parallel manager by calling a barrier to ensure all processes complete their work before cleanup. It prints a finalization message on the master process when verbose mode is enabled. Note that for MPI backends, this does NOT call MPI.Finalize() as that should be handled by the user or automatically at program exit. Use this method at the end of parallel processing to ensure proper resource cleanup.

        Parameters:
            None

        Returns:
            None
        """
        if self.backend == 'mpi':
            self.barrier()
        
        if self.is_master and self.verbose:
            print("MPASParallelManager finalized")


def parallel_plot(
    plot_function: Callable,
    files: List[str],
    **kwargs
) -> Optional[List[TaskResult]]:
    """
    Execute plotting tasks in parallel with automatic backend selection and error collection. This convenience function simplifies the most common MPAS workflow of generating plots from multiple files concurrently. It automatically creates a parallel manager with dynamic load balancing, configures error collection policy, and executes the provided plotting function across all files. The function returns results only on the master process, making it suitable for both MPI and multiprocessing environments.

    Parameters:
        plot_function (Callable): Function that accepts a file path and produces a plot, signature should be func(filepath, **kwargs).
        files (List[str]): List of file paths to process in parallel.
        **kwargs (dict): Additional keyword arguments to pass to plot_function for customization.

    Returns:
        Optional[List[TaskResult]]: Results from all tasks on master process, None on worker processes.

    Examples:
        >>> from mpasdiag.processing.parallel import parallel_plot
        >>> def my_plot(filepath, output_dir):
        ...     # Plotting code here
        ...     pass
        >>> files = ['file1.nc', 'file2.nc', 'file3.nc']
        >>> results = parallel_plot(my_plot, files, output_dir='./plots/')
    """
    manager = MPASParallelManager(load_balance_strategy="dynamic", verbose=True)
    manager.set_error_policy('collect')
    return manager.parallel_map(plot_function, files, **kwargs)


if __name__ == "__main__":
    def test_function(task_id, delay=0.1):
        """Test function that simulates work."""
        import time
        time.sleep(delay)
        return f"Completed task {task_id}"
    
    manager = MPASParallelManager(verbose=True)
    tasks = list(range(20))
    results = manager.parallel_map(test_function, tasks, delay=0.1)
    
    if manager.is_master and results is not None:
        print(f"\nProcessed {len(results)} tasks")
        successes = sum(1 for r in results if r.success)
        print(f"Success rate: {100*successes/len(results):.1f}%")
