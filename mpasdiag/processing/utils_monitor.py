#!/usr/bin/env python3

"""
MPAS Performance Monitoring Utilities

This module provides lightweight performance monitoring functionality for MPAS data analysis workflows including timing operations, memory usage tracking, and execution profiling. It implements the PerformanceMonitor class that captures elapsed time measurements for individual operations using context managers, stores timing data in memory for cumulative analysis, automatically reports operation durations to stdout when contexts exit, and provides summary statistics for completed workflows. The monitor is designed for short-running command-line tools and interactive scripts where lightweight profiling is needed without external profiling frameworks, supports named operations with automatic start/stop timing through Python's context manager protocol, accumulates timing data across multiple operations for workflow-level performance analysis, and prints human-readable timing reports suitable for development debugging and production monitoring. Core capabilities include context-based timing with automatic cleanup, flexible operation naming for organizing profiling data, cumulative duration tracking for identifying bottlenecks, and minimal overhead ensuring negligible impact on measured operation performance in data loading, computation, and visualization workflows.

Classes:
    PerformanceMonitor: Lightweight performance monitoring class providing timing utilities with context manager support for MPAS workflows.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

from datetime import datetime, timedelta
from typing import Dict
from contextlib import contextmanager


class PerformanceMonitor:
    """
    Performance monitoring utilities for measuring and reporting elapsed time of MPAS data analysis operations. This class provides a lightweight timing framework designed for short-running workflows and command-line tools, storing operation timings in memory with automatic reporting to stdout. The monitor supports context manager-based timing through the timer() method, which automatically captures start and end times and calculates durations for named operations. Individual timer durations are printed when the context exits, and cumulative summaries can be generated on demand via print_summary(). This simple timing infrastructure is useful for profiling data loading, computation, and visualization steps in MPASdiag workflows without requiring external profiling tools.
    """
    
    def __init__(self) -> None:
        """
        Initialize a new empty performance monitor with storage for operation timing data. This constructor creates two dictionaries to track timing information: start_times stores the datetime when each operation begins, and durations stores the computed timedelta for completed operations. The monitor starts with no active or completed timers, ready to measure operations as they are executed. This lightweight initialization ensures minimal overhead when creating performance monitors for profiling individual workflows or batch processing jobs.

        Returns:
            None: Initializes instance attributes self.start_times and self.durations as empty dictionaries.
        """
        self.start_times: Dict[str, datetime] = {}
        self.durations: Dict[str, timedelta] = {}
    
    @contextmanager
    def timer(self, operation_name: str):
        """
        Provide a context manager that automatically measures and reports elapsed time for named operations. This method captures the start datetime when entering the context, yields control back to the caller for operation execution, then calculates duration upon context exit regardless of success or exception. The elapsed time is immediately printed to stdout with the operation name and duration in seconds, and the timedelta is stored in the durations dictionary for later retrieval via get_summary(). This context manager pattern ensures timing is captured even if the operation raises an exception, providing reliable performance metrics for all executed operations in MPASdiag workflows.

        Parameters:
            operation_name (str): Descriptive name of the operation being timed for identification in timing reports.

        Yields:
            None: Control is yielded to the calling code to execute the timed operation within the context.
        """
        start_time = datetime.now()
        self.start_times[operation_name] = start_time
        
        try:
            yield
        finally:
            end_time = datetime.now()
            duration = end_time - start_time
            self.durations[operation_name] = duration
            
            print(f"{operation_name} completed in {duration.total_seconds():.2f} seconds")
    
    def get_summary(self) -> Dict[str, float]:
        """
        Return a dictionary mapping operation names to their elapsed execution times in seconds for programmatic access to timing data. This method converts all stored timedelta durations to floating-point seconds using total_seconds(), creating a simple key-value mapping suitable for serialization, logging, or further analysis. The returned dictionary includes only completed operations that were measured by the timer() context manager. This function is useful for exporting timing metrics to JSON, CSV, or other data formats, or for programmatic comparison of operation performance across different runs or configurations.

        Returns:
            Dict[str, float]: Dictionary with operation names as keys and execution durations in seconds as floating-point values.
        """
        return {name: duration.total_seconds()
                for name, duration in self.durations.items()}
    
    def print_summary(self) -> None:
        """
        Print a formatted human-readable performance timing summary to stdout showing all measured operations and total execution time. This method iterates through all stored operation durations, displaying each operation name with its elapsed time in seconds formatted to two decimal places. After listing individual operations, it calculates and displays the cumulative total time across all measured operations. This summary output is useful for quick performance assessment at the end of analysis workflows, providing immediate visibility into which operations consumed the most time during execution.

        Returns:
            None: Outputs formatted timing summary directly to stdout.
        """
        print("\n=== Performance Summary ===")
        for name, duration in self.durations.items():
            print(f"{name}: {duration.total_seconds():.2f} seconds")

        if self.durations:
            total_time = sum(d.total_seconds() for d in self.durations.values())
            print(f"Total time: {total_time:.2f} seconds")