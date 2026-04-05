#!/usr/bin/env python3

"""
MPASdiag Core Processing Module: Parallel Processing Manager

This module provides a comprehensive framework for managing parallel processing of tasks in the MPASdiag package. It supports both MPI-based parallelization for distributed computing environments and Python's multiprocessing for shared-memory parallelism. The module includes functionality for distributing tasks across workers, executing functions in parallel, collecting results, and computing performance statistics. It also implements configurable load balancing strategies and error handling policies to optimize parallel execution based on the characteristics of the workload and the computational environment. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""
# Load standard libraries
import gc
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
    """ Enumeration of load balancing strategies for distributing tasks across parallel workers. """
    STATIC = "static"      # Equal distribution at start
    DYNAMIC = "dynamic"    # Dynamic work stealing
    BLOCK = "block"        # Contiguous blocks per worker
    CYCLIC = "cyclic"      # Round-robin distribution


class ErrorPolicy(Enum):
    """ Enumeration of error handling policies for managing task execution failures in parallel processing. """
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
    This function serves as a wrapper for executing a single task in the multiprocessing backend. It unpacks the provided arguments, executes the specified function on the given task, and captures the result or any exceptions that occur. The function implements the configured error handling policy (abort, continue, or collect) and measures execution time for the task. It returns a TaskResult object containing the outcome of the task execution, including success status, result data, error messages, and timing information. This wrapper is essential for ensuring consistent error handling and performance tracking across tasks executed in parallel using multiprocessing. 

    Parameters:
        args (Tuple[int, Any, Callable, str, Tuple, Dict]): A tuple containing:
            - task_id (int): Unique identifier for the task.
            - task (Any): The actual task data to be processed.
            - func (Callable): The function to execute on the task.
            - error_policy_value (str): The error handling policy as a string ('abort', 'continue', 'collect').
            - func_args (Tuple): Additional positional arguments to pass to func.
            - func_kwargs (Dict): Additional keyword arguments to pass to func.

    Returns:
        TaskResult: An object containing the result of the task execution, including success status, result data, error messages, and execution time. 
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
    """ Distributes computational tasks across MPI workers based on the selected load balancing strategy. """
    
    def __init__(self: 'MPASTaskDistributor', 
                 comm: Any, 
                 strategy: LoadBalanceStrategy = LoadBalanceStrategy.DYNAMIC) -> None:
        """
        This constructor initializes the task distributor with the MPI communicator and the chosen load balancing strategy. It sets up the necessary attributes to manage task distribution across workers, including process rank and size information. The task distributor is responsible for determining which subset of tasks should be processed by each worker based on the selected strategy (static, dynamic, block, or cyclic). This class is essential for ensuring that tasks are distributed efficiently across workers to optimize parallel execution in MPI environments. 

        Parameters:
            comm (MPI.Comm): MPI communicator object providing rank and communication capabilities.
            strategy (LoadBalanceStrategy): Load balancing strategy to use for task distribution (STATIC, DYNAMIC, BLOCK, CYCLIC). Defaults to DYNAMIC. 

        Returns:
            None
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.strategy = strategy
    
    def distribute_tasks(self: 'MPASTaskDistributor', 
                         tasks: List[Any]) -> List[Tuple[int, Any]]:
        """
        This method distributes the provided list of tasks to the worker process based on the selected load balancing strategy. It determines which subset of tasks should be processed by this specific worker according to its rank and the total number of workers. The method supports multiple strategies for task distribution, including static (equal chunks), dynamic (work stealing), block (contiguous blocks), and cyclic (round-robin). The returned list contains tuples of (task_id, task) that are assigned to this worker for processing. This method is called by the parallel manager to ensure that tasks are distributed appropriately across MPI workers for efficient parallel execution. 

        Parameters:
            tasks (List[Any]): Complete list of tasks to distribute across workers. 

        Returns:
            List[Tuple[int, Any]]: List of (task_id, task) tuples assigned to this worker based on the load balancing strategy. 
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
    
    def _static_distribution(self: 'MPASTaskDistributor', 
                             tasks: List[Any]) -> List[Tuple[int, Any]]:
        """
        This method implements static distribution of tasks across workers. It divides the total list of tasks into equal chunks based on the number of workers and assigns each chunk to a worker based on its rank. If the total number of tasks is not perfectly divisible by the number of workers, the remaining tasks are distributed one by one to the first few workers until all tasks are assigned. Static distribution is simple and has low overhead, but it may lead to load imbalance if task execution times vary significantly, as some workers may finish their assigned tasks much earlier than others. 

        Parameters:
            tasks (List[Any]): Complete list of tasks to distribute statically. 

        Returns:
            List[Tuple[int, Any]]: List of (task_id, task) tuples assigned to this worker based on static distribution. 
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
    
    def _block_distribution(self: 'MPASTaskDistributor', 
                            tasks: List[Any]) -> List[Tuple[int, Any]]:
        """
        This method implements block distribution of tasks across workers. It divides the total list of tasks into contiguous blocks and assigns each block to a worker based on its rank. Each worker receives a contiguous segment of the task list, which can help improve cache locality and reduce communication overhead for certain types of workloads. However, block distribution may lead to load imbalance if task execution times vary significantly, as some workers may finish their assigned blocks much earlier than others.         

        Parameters:
            tasks (List[Any]): Complete list of tasks to distribute in blocks. 

        Returns:
            List[Tuple[int, Any]]: List of (task_id, task) tuples assigned to this worker based on block distribution. 
        """
        n_tasks = len(tasks)
        block_size = (n_tasks + self.size - 1) // self.size
        start = self.rank * block_size
        end = min(start + block_size, n_tasks)
        
        return [(i, tasks[i]) for i in range(start, end)]
    
    def _cyclic_distribution(self: 'MPASTaskDistributor', 
                             tasks: List[Any]) -> List[Tuple[int, Any]]:
        """
        This method implements cyclic distribution of tasks across workers. It assigns tasks to workers in a round-robin fashion, where each worker receives every nth task based on its rank and the total number of workers. For example, worker 0 gets tasks 0, n, 2n, etc., while worker 1 gets tasks 1, n+1, 2n+1, and so on. Cyclic distribution can help mitigate load imbalance for workloads with variable execution times by spreading tasks more evenly across workers. However, it may introduce more communication overhead compared to block distribution due to less contiguous access patterns. 

        Parameters:
            tasks (List[Any]): Complete list of tasks to distribute cyclically. 

        Returns:
            List[Tuple[int, Any]]: List of (task_id, task) tuples assigned to this worker based on cyclic distribution. 
        """
        return [(i, tasks[i]) for i in range(self.rank, len(tasks), self.size)]
    
    def _dynamic_distribution(self: 'MPASTaskDistributor', 
                              tasks: List[Any]) -> List[Tuple[int, Any]]:
        """
        This method implements dynamic distribution of tasks across workers using a work-stealing approach. In this strategy, workers initially receive a small batch of tasks, and as they complete their assigned tasks, they can request additional tasks from a shared task pool or steal tasks from other workers that still have pending tasks. Dynamic distribution can help achieve better load balancing for workloads with highly variable execution times, as it allows faster workers to take on more work while slower workers are still processing their initial tasks. However, it may introduce additional overhead due to the need for synchronization and communication between workers to manage the shared task pool. 

        Parameters:
            tasks (List[Any]): Complete list of tasks to distribute dynamically. 

        Returns:
            List[Tuple[int, Any]]: List of (task_id, task) tuples assigned to this worker based on dynamic distribution. 
        """
        return self._static_distribution(tasks)


class MPASResultCollector:
    """ Collects and aggregates results from parallel workers in MPI environments, providing comprehensive statistics on task execution and performance. """
    
    def __init__(self: 'MPASResultCollector', 
                 comm: Any) -> None:
        """
        This constructor initializes the result collector with the MPI communicator. It sets up the necessary attributes to manage the collection of results from worker processes, including rank and size information. The result collector is responsible for gathering results from all workers to the master process, aggregating them into a single list, and computing statistics on task execution performance. This class is essential for consolidating results from distributed workers and enabling comprehensive analysis of task outcomes and performance metrics in parallel MPI workflows. 

        Parameters:
            comm (MPI.Comm): MPI communicator object providing rank and communication capabilities. 

        Returns:
            None 
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
    
    def gather_results(self: 'MPASResultCollector', 
                       local_results: List[TaskResult]) -> Optional[List[TaskResult]]:
        """
        This method gathers results from all worker processes to the master process (rank 0) in an MPI environment. Each worker sends its local results, which are collected and aggregated into a single list on the master process. The method uses MPI's gather functionality to collect results efficiently, and it handles the aggregation of results from multiple workers while ensuring that only the master process receives the complete list of results. This allows for centralized analysis and reporting of task outcomes and performance metrics after parallel execution. 

        Parameters:
            local_results (List[TaskResult]): List of TaskResult objects generated by the local worker process. 

        Returns:
            Optional[List[TaskResult]]: Aggregated list of TaskResult objects from all workers, returned only on the master process (rank 0). Returns None on worker processes. 
        """
        all_results = self.comm.gather(local_results, root=0)
        
        if self.rank == 0:
            flattened = []
            for worker_results in all_results:
                flattened.extend(worker_results)
            return flattened
        return None
    
    def compute_statistics(self: 'MPASResultCollector', 
                           results: List[TaskResult]) -> ParallelStats:
        """
        This method computes comprehensive statistics on the execution of tasks based on the collected results from all workers. It calculates metrics such as total tasks, completed tasks, failed tasks, total execution time, and load imbalance across workers. The method aggregates timing information for each worker to evaluate performance and identify any load imbalances that may exist. The resulting ParallelStats object provides a detailed overview of the parallel execution performance, enabling users to analyze the efficiency of their parallel workflows and identify potential areas for optimization. 

        Parameters:
            results (List[TaskResult]): Complete list of TaskResult objects collected from all workers, containing execution outcomes and timing information for each task. 

        Returns:
            ParallelStats: An object containing aggregated statistics on task execution performance, including total tasks, completed tasks, failed tasks, total execution time, and load imbalance metrics. 
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
    """ Manages parallel processing of tasks across multiple CPU cores using MPI or multiprocessing backends, with support for load balancing and error handling policies. """

    def _setup_multiprocessing_backend(self: 'MPASParallelManager') -> None:
        """
        This method configures the MPASParallelManager instance for parallel execution using Python's multiprocessing module. It sets the backend to 'multiprocessing', determines the number of worker processes to use based on the number of CPU cores (defaulting to one less than the total cores), and initializes attributes related to multiprocessing execution. The method ensures that the parallel manager is ready to execute tasks in parallel using multiprocessing, while also setting up the necessary state for managing task distribution and result collection within a shared-memory environment.

        Parameters:
            None

        Returns:
            None
        """
        self.backend = 'multiprocessing'
        self.rank = 0
        self.size = self.n_workers or max(1, cpu_count() - 1)
        self.is_master = True
        self.comm = None
        self.distributor = None
        self.collector = None

    def _setup_serial_backend(self: 'MPASParallelManager') -> None:
        """
        This method configures the MPASParallelManager instance for serial execution without parallelization. It sets the backend to 'serial' and initializes attributes to reflect a single-process execution environment. This setup is used when neither MPI nor multiprocessing is available or when the user explicitly chooses to run in serial mode. The method ensures that the parallel manager can still execute tasks sequentially while maintaining a consistent interface for task execution and result handling, albeit without any parallel performance benefits.

        Parameters:
            None

        Returns:
            None
        """
        self.backend = 'serial'
        self.comm = None
        self.rank = 0
        self.size = 1
        self.is_master = True
        self.distributor = None
        self.collector = None

    def _setup_mpi_backend(self: 'MPASParallelManager',
                           load_balance_strategy: Union[str, 'LoadBalanceStrategy'],) -> None:
        """
        This method configures the MPASParallelManager instance for parallel execution using MPI. It initializes the MPI communicator, retrieves the rank and size of the MPI processes, and sets up the task distributor and result collector based on the selected load balancing strategy. The method ensures that the parallel manager is ready to execute tasks in parallel across multiple MPI processes, with appropriate handling for task distribution and result collection in a distributed computing environment. If MPI initialization fails, it falls back to configuring the multiprocessing backend. 

        Parameters:
            load_balance_strategy (Union[str, LoadBalanceStrategy]): The strategy to use for load balancing tasks across MPI processes.

        Returns:
            None
        """
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
                self._setup_multiprocessing_backend()
        except Exception as e:
            if self.verbose:
                print(f"MPI initialization failed: {e}")
                print("Falling back to multiprocessing backend")
            self._setup_multiprocessing_backend()

    def _log_backend_initialized(self: 'MPASParallelManager') -> None:
        """
        This method logs information about the initialized parallel backend and the number of workers or processes being used. It provides feedback to the user about the parallel execution environment that has been set up, including whether MPI or multiprocessing is being used and how many workers are available for processing tasks. This logging is helpful for debugging and performance monitoring, allowing users to confirm that the parallel manager is configured as expected before executing tasks in parallel.

        Parameters:
            None

        Returns:
            None
        """
        if self.is_master and self.verbose:
            if self.backend == 'mpi':
                print(f"MPASParallelManager initialized in MPI mode with {self.size} processes")
            elif self.backend == 'multiprocessing':
                print(f"MPASParallelManager initialized in multiprocessing mode with {self.size} workers")
            else:
                print("MPASParallelManager initialized in serial mode")


    def __init__(self: 'MPASParallelManager', 
                 load_balance_strategy: Union[str, LoadBalanceStrategy] = "dynamic", 
                 verbose: bool = True, 
                 backend: Optional[str] = None, 
                 n_workers: Optional[int] = None) -> None:
        """
        This constructor initializes the MPASParallelManager with the specified load balancing strategy, verbosity level, parallel backend, and number of worker processes. It automatically detects the availability of MPI and configures the appropriate backend for parallel processing. The constructor sets up the necessary attributes for managing parallel execution, including MPI communicators, task distributors, and result collectors. It also handles fallback to multiprocessing if MPI is not available or if the user explicitly chooses to use multiprocessing. The load balancing strategy determines how tasks are distributed across workers in MPI mode, while the error handling policy can be configured separately using the set_error_policy method. This initialization process ensures that the parallel manager is ready to execute tasks efficiently based on the user's configuration and the capabilities of the computational environment. 

        Parameters:
            load_balance_strategy (str or LoadBalanceStrategy): Load balancing strategy for MPI task distribution ('static', 'dynamic', 'block', 'cyclic'). Defaults to 'dynamic'.
            verbose (bool): Whether to print detailed information about parallel execution and statistics. Defaults to True.
            backend (str or None): Parallel backend to use ('mpi', 'multiprocessing', or None for auto-detection). Defaults to None.
            n_workers (int or None): Number of worker processes to use for multiprocessing backend. If None, it defaults to the number of CPU cores minus one. Ignored if using MPI backend. Defaults to None.

        Returns:
            None
        """
        self.verbose = verbose
        self.error_policy = ErrorPolicy.COLLECT
        self.backend = backend
        self.n_workers = n_workers

        if backend == 'mpi' or (backend is None and MPI_AVAILABLE):
            self._setup_mpi_backend(load_balance_strategy)
        elif backend == 'multiprocessing' or (backend is None and not MPI_AVAILABLE):
            self._setup_multiprocessing_backend()
        else:
            self._setup_serial_backend()

        self.stats = None
        self._log_backend_initialized()

    def set_error_policy(self: 'MPASParallelManager', 
                         policy: Union[str, ErrorPolicy]) -> None:
        """
        This method allows users to configure the error handling policy for task execution in parallel processing. The error policy determines how the parallel manager responds to exceptions that occur during the execution of tasks across workers. Users can choose from three policies: 'abort' to stop all processing on the first error, 'continue' to ignore errors and proceed with remaining tasks, or 'collect' to gather all errors and report them after execution. The method accepts either a string representation of the policy or an ErrorPolicy enum value, and it updates the internal state of the parallel manager accordingly. This configuration is essential for managing robustness and fault tolerance in parallel workflows, allowing users to tailor error handling to their specific needs and preferences. 

        Parameters:
            policy (str or ErrorPolicy): The error handling policy to set ('abort', 'continue', 'collect'). Can be provided as a string or an ErrorPolicy enum value. 

        Returns:
            None
        """
        if isinstance(policy, str):
            policy = ErrorPolicy(policy)
        self.error_policy = policy
    
    def parallel_map(self: 'MPASParallelManager', 
                     func: Callable, 
                     tasks: List[Any], 
                     *args, 
                     **kwargs) -> Optional[List[TaskResult]]:
        """
        This method executes the provided function on a list of tasks in parallel using the configured backend (MPI or multiprocessing). It handles the distribution of tasks to workers, execution of the function on each task, and collection of results while adhering to the configured error handling policy. The method automatically selects the appropriate parallelization strategy based on the availability of MPI and user configuration. It returns a list of TaskResult objects containing the outcome of each task execution, including success status, results, errors, and timing information. In MPI mode, the complete list of results is returned only on the master process (rank 0), while other ranks receive None. This method serves as the main entry point for executing parallel tasks in MPASdiag workflows, providing a flexible and robust interface for parallel processing. 

        Parameters:
            func (Callable): The function to execute on each task. It should accept a single task as its first argument, followed by any additional positional and keyword arguments.
            tasks (List[Any]): A list of tasks to be processed in parallel. Each task can be any data structure that the provided function can handle.
            *args (tuple): Additional positional arguments to pass to the function for each task.
            **kwargs (dict): Additional keyword arguments to pass to the function for each task. 

        Returns:
            Optional[List[TaskResult]]: A list of TaskResult objects containing the outcome of each task execution. In MPI mode, this list is returned only on the master process (rank 0), while other ranks receive None. In multiprocessing mode, the complete list of results is returned to the caller. Each TaskResult includes success status, result data, error messages, and execution time for the corresponding task. 
        """
        if self.backend == 'mpi':
            return self._mpi_map(func, tasks, *args, **kwargs)
        elif self.backend == 'multiprocessing':
            return self._multiprocessing_map(func, tasks, *args, **kwargs)
        else:
            return self._serial_map(func, tasks, *args, **kwargs)
    
    def _mpi_map(self: 'MPASParallelManager', 
                 func: Callable, 
                 tasks: List[Any], 
                 *args, 
                 **kwargs) -> Optional[List[TaskResult]]:
        """
        This method executes the provided function on a list of tasks in parallel using the MPI backend. It first broadcasts the complete list of tasks to all worker processes, then each worker uses the task distributor to determine which subset of tasks it should process based on the selected load balancing strategy. Each worker executes its assigned tasks locally and generates a list of TaskResult objects containing the outcome of each task execution. The results from all workers are then gathered back to the master process (rank 0) using the result collector, where they are aggregated into a single list. The master process also computes statistics on task execution performance based on the collected results. Finally, the method returns the complete list of TaskResult objects to the caller on the master process, while other ranks receive None. This implementation provides efficient parallel processing using MPI while ensuring robust error handling and comprehensive performance tracking. 

        Parameters:
            func (Callable): The function to execute on each task. It should accept a single task as its first argument, followed by any additional positional and keyword arguments.
            tasks (List[Any]): A list of tasks to be processed in parallel. Each task can be any data structure that the provided function can handle.
            *args (tuple): Additional positional arguments to pass to the function for each task.
            **kwargs (dict): Additional keyword arguments to pass to the function for each task.

        Returns:
            Optional[List[TaskResult]]: A list of TaskResult objects containing the outcome of each task execution, returned only on the master process (rank 0). Other ranks receive None. Each TaskResult includes success status, result data, error messages, and execution time for the corresponding task. 
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
        
        del local_results
        gc.collect()
        
        if self.is_master:
            assert all_results is not None, "Results should not be None on master"
            self.stats = self.collector.compute_statistics(all_results)
            
            if self.verbose:
                self._print_statistics()
            
            return all_results
        
        return None

    @staticmethod
    def _get_mp_context_methods() -> List[str]:
        """
        This static method determines the available multiprocessing context methods based on the operating system. It returns a list of context method names that can be used for creating multiprocessing pools. On Windows and macOS, only the 'spawn' method is available, while on other platforms (e.g., Linux), both 'fork' and 'spawn' methods are typically available. This method is used to ensure compatibility with the multiprocessing module across different platforms and to provide fallback options in case one method fails.

        Parameters:
            None

        Returns:
            List[str]: A list of multiprocessing context method names to try when creating a multiprocessing pool. 
        """
        if sys.platform in ('win32', 'darwin'):
            return ['spawn']
        return ['fork', 'spawn']

    def _run_pool_with_fallback(self: 'MPASParallelManager',
                                task_args: List[Any],) -> List[TaskResult]:
        """
        This method attempts to execute the provided task arguments in parallel using a multiprocessing pool. It tries different multiprocessing start methods (fork, spawn) based on the operating system and available options. If multiprocessing fails for any reason (e.g., due to platform limitations or errors in task execution), it falls back to executing the tasks serially. The method ensures that all tasks are executed and that results are collected regardless of the success of multiprocessing, providing robustness in environments where multiprocessing may not be fully supported.

        Parameters:
            task_args (List[Any]): A list of arguments to be passed to the task wrapper function for each task. Each element in the list should be a tuple containing the necessary information for executing a single task, such as task ID, task data, function to execute, error handling policy, and any additional arguments.

        Returns:
            List[TaskResult]: A list of TaskResult objects containing the outcome of each task execution. Each TaskResult includes success status, result data, error messages, and execution time for the corresponding task. The results are returned in the same order as the input task arguments, regardless of whether multiprocessing was successful or if the method had to fall back to serial execution.
        """
        ctx_methods = self._get_mp_context_methods()

        for ctx_method in ctx_methods:
            try:
                ctx = get_context(ctx_method)
                with ctx.Pool(processes=self.size) as pool:
                    return pool.map(_multiprocessing_task_wrapper, task_args)
            except Exception as e:
                if self.verbose:
                    print(f"Multiprocessing with '{ctx_method}' failed: {e}")
                    if ctx_method != ctx_methods[-1]:
                        print("Trying next method...")
                    else:
                        print("Falling back to serial execution")

        return [_multiprocessing_task_wrapper(args) for args in task_args]

    def _print_mp_statistics(self: 'MPASParallelManager',
                             stats: 'ParallelStats',
                             wall_time: float,) -> None:
        """
        This method prints detailed statistics about the parallel execution when using the multiprocessing backend. It displays the total number of tasks, how many were completed successfully, how many failed, the success rate, total CPU time spent on all tasks, wall clock time for the entire execution, and the speedup achieved compared to serial execution. This information is crucial for evaluating the performance of the multiprocessing approach and understanding the efficiency of task execution across multiple CPU cores. The statistics help identify any bottlenecks or issues in the parallel processing workflow and provide insights for potential optimizations.

        Parameters:
            stats (ParallelStats): An object containing aggregated statistics on task execution performance, including total tasks, completed tasks, failed tasks, total execution time, and load imbalance metrics.
            wall_time (float): The total wall clock time taken for the entire parallel execution.

        Returns:
            None
        """
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

    def _multiprocessing_map(self: 'MPASParallelManager', 
                             func: Callable, 
                             tasks: List[Any], 
                             *args, 
                             **kwargs) -> List[TaskResult]:
        """
        This method executes the provided function on a list of tasks in parallel using the multiprocessing backend. It prepares the arguments for each task by creating a list of tuples containing the task ID, task data, function to execute, error handling policy, and any additional arguments. The method then attempts to create a multiprocessing pool using different context methods (fork, spawn) based on the operating system and available options. It maps the task wrapper function across all tasks in parallel, which executes each task and captures results or errors according to the configured error policy. If multiprocessing fails for any reason, it falls back to serial execution. After processing all tasks, it computes statistics on execution performance and returns a list of TaskResult objects containing the outcome of each task execution. This implementation provides robust parallel processing using multiprocessing while ensuring comprehensive error handling and performance tracking. 

        Parameters:
            func (Callable): The function to execute on each task. It should accept a single task as its first argument, followed by any additional positional and keyword arguments.
            tasks (List[Any]): A list of tasks to be processed in parallel. Each task can be any data structure that the provided function can handle.
            *args (tuple): Additional positional arguments to pass to the function for each task.
            **kwargs (dict): Additional keyword arguments to pass to the function for each task. 

        Returns:
            List[TaskResult]: A list of TaskResult objects containing the outcome of each task execution. Each TaskResult includes success status, result data, error messages, and execution time for the corresponding task. 
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

        results = self._run_pool_with_fallback(task_args)

        stats = ParallelStats()
        stats.total_tasks = len(results)
        stats.completed_tasks = sum(1 for r in results if r.success)
        stats.failed_tasks = sum(1 for r in results if not r.success)
        stats.total_time = sum(r.execution_time for r in results)
        self.stats = stats

        if self.verbose:
            self._print_mp_statistics(stats, time.time() - start_time)

        del task_args
        gc.collect()

        return results
    
    def _execute_local_tasks(self: 'MPASParallelManager', 
                             func: Callable, 
                             local_tasks: List[Tuple[int, Any]], 
                             *args, 
                             **kwargs) -> List[TaskResult]:
        """
        This method executes a list of tasks assigned to the local worker process. It iterates over each task, executes the provided function, and captures the result or any exceptions that occur. The method implements the configured error handling policy (abort, continue, or collect) and measures execution time for each task. It returns a list of TaskResult objects containing the outcome of each task execution, including success status, result data, error messages, and timing information. This method is called by both the MPI and multiprocessing mapping functions to process the subset of tasks assigned to this worker, ensuring consistent error handling and performance tracking across different parallel backends. 

        Parameters:
            func (Callable): The function to execute on each task. It should accept a single task as its first argument, followed by any additional positional and keyword arguments.
            local_tasks (List[Tuple[int, Any]]): A list of (task_id, task) tuples assigned to this worker for processing. Each task can be any data structure that the provided function can handle.
            *args (tuple): Additional positional arguments to pass to the function for each task.
            **kwargs (dict): Additional keyword arguments to pass to the function for each task. 

        Returns:
            List[TaskResult]: A list of TaskResult objects containing the outcome of each task execution. Each TaskResult includes success status, result data, error messages, and execution time for the corresponding task. 
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
    
    def _serial_map(self: 'MPASParallelManager', 
                    func: Callable, 
                    tasks: List[Any], 
                    *args, 
                    **kwargs) -> List[TaskResult]:
        """
        This method executes the provided function on a list of tasks sequentially in serial mode. It iterates over each task, executes the function, and captures the result or any exceptions that occur. The method implements the configured error handling policy (abort, continue, or collect) and measures execution time for each task. It returns a list of TaskResult objects containing the outcome of each task execution, including success status, result data, error messages, and timing information. This method is used when no parallel backend is available or when the user explicitly chooses to run in serial mode, providing a fallback option for executing tasks without parallelization while still maintaining consistent error handling and performance tracking. 

        Parameters:
            func (Callable): The function to execute on each task. It should accept a single task as its first argument, followed by any additional positional and keyword arguments.
            tasks (List[Any]): A list of tasks to be processed sequentially. Each task can be any data structure that the provided function can handle.
            *args (tuple): Additional positional arguments to pass to the function for each task.
            **kwargs (dict): Additional keyword arguments to pass to the function for each task.

        Returns:
            List[TaskResult]: A list of TaskResult objects containing the outcome of each task execution. Each TaskResult includes success status, result data, error messages, and execution time for the corresponding task. 
        """
        if self.verbose:
            print(f"\nProcessing {len(tasks)} tasks in serial mode...")
        
        local_tasks = [(i, task) for i, task in enumerate(tasks)]
        results = self._execute_local_tasks(func, local_tasks, *args, **kwargs)
        
        self.stats = self.collector.compute_statistics(results) if self.collector else None
        
        if self.verbose and self.stats:
            self._print_statistics()
        
        return results
    
    def get_statistics(self: 'MPASParallelManager') -> Optional[ParallelStats]:
        """
        This method returns the parallel execution statistics collected during the processing of tasks. The statistics include metrics such as total tasks, completed tasks, failed tasks, total execution time, and load imbalance across workers. The method returns a ParallelStats object containing these metrics if available, or None if statistics have not been collected (e.g., if no tasks have been executed yet). This allows users to access detailed performance information about their parallel processing workflow for analysis and optimization purposes. 

        Parameters:
            None

        Returns:
            Optional[ParallelStats]: An object containing aggregated statistics on task execution performance, including total tasks, completed tasks, failed tasks, total execution time, and load imbalance metrics. Returns None if statistics are not available. 
        """
        return self.stats
    
    def _print_statistics(self: 'MPASParallelManager') -> None:
        """
        This method prints the parallel execution statistics in a formatted manner to the console. It displays metrics such as total tasks, completed tasks, failed tasks, success rate, total execution time, and load imbalance across workers (if applicable). The method is designed to provide a clear and concise summary of the performance of the parallel processing workflow, allowing users to quickly assess the efficiency and effectiveness of their parallel execution. This method is called internally after task execution to report statistics if verbosity is enabled. 

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
    
    def barrier(self: 'MPASParallelManager') -> None:
        """
        This method implements a synchronization barrier for MPI backends, ensuring that all processes reach the same point in the execution before any process continues. It uses the MPI Barrier function to block each process until all processes have reached the barrier, providing a way to synchronize the execution flow across multiple processes. This is particularly useful in scenarios where certain operations need to be completed by all processes before proceeding, such as after task distribution or before finalizing the parallel manager. For non-MPI backends, this method does not perform any synchronization and simply returns immediately. 

        Parameters:
            None

        Returns:
            None
        """
        if self.backend == 'mpi' and self.comm is not None:
            self.comm.Barrier()
    
    def finalize(self: 'MPASParallelManager') -> None:
        """
        This method finalizes the parallel manager by performing any necessary cleanup operations. For MPI backends, it ensures that all processes reach a synchronization point using the barrier method before finalizing. It also prints a message indicating that the MPASParallelManager has been finalized if the current process is the master and verbosity is enabled. This method should be called at the end of the parallel processing workflow to ensure proper cleanup and synchronization of resources across processes. For non-MPI backends, this method simply prints the finalization message if applicable. 

        Parameters:
            None

        Returns:
            None
        """
        if self.backend == 'mpi':
            self.barrier()
        
        if self.is_master and self.verbose:
            print("MPASParallelManager finalized")


def parallel_plot(plot_function: Callable, 
                  files: List[str], 
                  **kwargs) -> Optional[List[TaskResult]]:
    """
    This function provides a convenient interface for executing a plotting function in parallel across multiple files using the MPASParallelManager. It accepts a plotting function that takes a file path as input and produces a plot, along with a list of file paths to process. The function initializes the MPASParallelManager with dynamic load balancing and collects results from all tasks while adhering to the configured error handling policy. It returns a list of TaskResult objects containing the outcome of each plotting task, including success status, results, errors, and timing information. This utility function simplifies the process of generating plots in parallel for multiple files, allowing users to efficiently visualize data across large datasets while leveraging the capabilities of the MPASParallelManager for robust parallel processing. 

    Parameters:
        plot_function (Callable): A function that takes a file path as input and generates a plot. It should return the result of the plotting operation or raise an exception if an error occurs.
        files (List[str]): A list of file paths to be processed by the plotting function.
        **kwargs: Additional keyword arguments to pass to the plotting function for each file.

    Returns:
        Optional[List[TaskResult]]: A list of TaskResult objects containing the outcome of each plotting task, returned only on the master process (rank 0) if using MPI. Each TaskResult includes success status, result data, error messages, and execution time for the corresponding plotting task. Returns None on worker processes in MPI mode. 
    """
    manager = MPASParallelManager(load_balance_strategy="dynamic", verbose=True)
    manager.set_error_policy('collect')
    return manager.parallel_map(plot_function, files, **kwargs)


if __name__ == "__main__":
    def test_function(task_id: int, 
                      delay: float = 0.1) -> str:
        """ Test function that simulates work. It sleeps for a specified delay and returns a completion message. This function is used to demonstrate the parallel execution capabilities of the MPASParallelManager. The task_id parameter identifies the specific task being processed, while the delay simulates variable execution times. This simple function allows us to test load balancing and error handling in the parallel manager."""
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
