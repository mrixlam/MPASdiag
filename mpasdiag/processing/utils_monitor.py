#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Core Processing Module: Performance Monitoring Utilities

This module provides the PerformanceMonitor class, which offers utilities for measuring and reporting the elapsed time of various operations within the MPASdiag data analysis workflows. The PerformanceMonitor class allows users to easily instrument their code to track the execution time of specific operations, providing insights into performance bottlenecks and helping to optimize the overall efficiency of MPASdiag processing tasks.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

from datetime import datetime, timedelta
from contextlib import contextmanager
from typing import Dict, Generator, Set

from .utils_logger import get_logger

logger = get_logger(__name__)


class PerformanceMonitor:
    """Performance monitoring utilities for measuring and reporting elapsed time of MPAS data analysis operations."""

    def __init__(self: "PerformanceMonitor", quiet: bool = False) -> None:
        """
        This constructor initializes a PerformanceMonitor instance, setting up internal data structures to track the start times and durations of various operations. The start_times attribute is a dictionary that will store the datetime when each operation begins, while the durations attribute will store the calculated timedelta for each operation once it completes. This setup allows for flexible and efficient performance monitoring across multiple operations within the MPASdiag processing workflows.

        Timers that run inside another timer are remembered as nested so that the total reported by print_summary counts only top-level operations and never double-counts the time spent in nested ones. The nested status is decided per timing instance and always describes the duration currently stored for the name, so a name that was once nested and later timed at top level is counted in the total. When quiet is True the per-operation completion messages are suppressed.

        Parameters:
            quiet (bool): Suppress the per-operation completion messages printed by the timer context manager (default: False).

        Returns:
            None: The constructor initializes the PerformanceMonitor instance without returning any value.
        """
        self.start_times: Dict[str, datetime] = {}
        self.durations: Dict[str, timedelta] = {}
        self.quiet: bool = quiet
        self._active: int = 0
        self._nested: Set[str] = set()

    @contextmanager
    def timer(
        self: "PerformanceMonitor", operation_name: str
    ) -> Generator[None, None, None]:
        """
        This context manager method allows users to measure the execution time of a specific operation by wrapping the code block that performs the operation within a with statement. When the context is entered, the current datetime is recorded as the start time for the specified operation name. Once the code block within the context is executed, the end time is recorded, and the duration of the operation is calculated as a timedelta. The duration is then stored in the durations dictionary under the corresponding operation name, and a formatted message is printed to stdout indicating the completion of the operation along with its execution time in seconds. This method provides a convenient and reusable way to instrument performance monitoring throughout MPASdiag processing workflows.

        Parameters:
            operation_name (str): A descriptive name for the operation being timed, which will be used as a key in the internal tracking dictionaries and included in the output messages.

        Returns:
            Generator[None, None, None]: This method is a context manager that yields control back to the caller after setting up the timing mechanism, allowing the wrapped code block to execute while the performance monitoring is active.
        """
        start_time = datetime.now()
        self.start_times[operation_name] = start_time

        # Nesting belongs to this particular timing instance, decided when it starts
        is_nested = self._active > 0
        self._active += 1

        try:
            yield
        finally:
            self._active -= 1
            end_time = datetime.now()
            duration = end_time - start_time
            self.durations[operation_name] = duration

            # The stored duration is this instance's, so the nested flag must follow it: a name that ran nested once and later at top level is counted in the total again
            if is_nested:
                self._nested.add(operation_name)
            else:
                self._nested.discard(operation_name)

            if not self.quiet:
                print(
                    f"{operation_name} completed in {duration.total_seconds():.2f} seconds"
                )

    def get_summary(self: "PerformanceMonitor") -> Dict[str, float]:
        """
        This method returns a summary of all recorded operation durations as a dictionary, where the keys are the operation names and the values are the execution durations in seconds as floating-point numbers. It iterates through the durations dictionary, converting each timedelta duration to its total seconds representation using the total_seconds() method. This summary provides a convenient way to access and analyze the performance metrics for various operations within MPASdiag processing workflows, allowing users to easily identify which steps may be taking longer and require optimization.

        Parameters:
            None: This method does not take any parameters and operates on the internal state of the PerformanceMonitor instance to access recorded durations.

        Returns:
            Dict[str, float]: A dictionary mapping operation names to their corresponding execution durations in seconds, providing a structured summary of performance metrics for the monitored operations.
        """
        return {
            name: duration.total_seconds() for name, duration in self.durations.items()
        }

    def print_summary(self: "PerformanceMonitor") -> None:
        """
        This method prints a formatted summary of all recorded operation durations to stdout. It iterates through the durations dictionary, printing each operation name along with its execution time in seconds formatted to two decimal places. Additionally, if there are any recorded durations, it calculates and prints the total execution time, counting only top-level operations so that timers nested inside another timer are not added twice. This method provides a user-friendly way to visualize the performance metrics for various operations within MPASdiag processing workflows, allowing users to quickly assess which steps may be taking longer and identify potential areas for optimization.

        Parameters:
            None: This method does not take any parameters and operates on the internal state of the PerformanceMonitor instance to access recorded durations.

        Returns:
            None: This method does not return any value, as its primary purpose is to print the performance summary to stdout.
        """
        print("=== Performance Summary ===")

        for name, duration in self.durations.items():
            print(f"{name}: {duration.total_seconds():.2f} seconds")

        if self.durations:
            total_time = sum(
                elapsed.total_seconds()
                for name, elapsed in self.durations.items()
                if name not in self._nested
            )
            print(f"Total time: {total_time:.2f} seconds")
