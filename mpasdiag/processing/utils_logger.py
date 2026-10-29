#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Core Processing Module: Error and Logging Utilities

This module provides a centralized logging utility for the MPASdiag processing code. It defines a custom logger class, MPASLogger, which configures the root logger with appropriate handlers and formatters to include MPI rank information when running in a parallel context. The MPIRankFilter is used to inject MPI rank and size into log records and to filter out INFO and DEBUG messages from non-zero ranks in a parallel execution environment. This ensures that log messages are informative and appropriately filtered based on the execution context, allowing for consistent logging practices across the MPASdiag processing codebase.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import os
import sys
import logging
from typing import Any, Dict, Optional

ROOT_LOGGER_NAME = "mpasdiag"

_configured: bool = False


_MPI_RANK_ENV_VARS = (
    "PMI_RANK",
    "OMPI_COMM_WORLD_RANK",
    "PMIX_RANK",
    "MV2_COMM_WORLD_RANK",
    "PALS_RANKID",
)
_MPI_SIZE_ENV_VARS = (
    "PMI_SIZE",
    "OMPI_COMM_WORLD_SIZE",
    "PMIX_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "PALS_NRANKS",
    "SLURM_STEP_NUM_TASKS",
    "SLURM_NTASKS",
)


def _env_int(name: str) -> Optional[int]:
    """
    This helper reads an environment variable and converts it to an integer, returning None when the variable is unset, empty, or not a valid integer.

    Parameters:
        name (str): Name of the environment variable.

    Returns:
        Optional[int]: The integer value, or None if it is missing or malformed.
    """
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _detect_mpi_rank_and_size() -> "tuple[int, int]":
    """
    This function detects the MPI rank and size from the environment variables that the standard MPI launchers export (PMI_RANK/PMI_SIZE for MPICH and Intel MPI, OMPI_COMM_WORLD_* for Open MPI, MV2_COMM_WORLD_* for MVAPICH2, PMIX_RANK/PMIX_SIZE, PALS_RANKID/PALS_NRANKS for Cray PALS, and SLURM_PROCID with SLURM_STEP_NUM_TASKS/SLURM_NTASKS under srun; the size may come from a different launcher's variable than the rank, e.g. PMIX_RANK with SLURM_NTASKS under srun --mpi=pmix), without importing mpi4py. Importing mpi4py initialises MPI as a side effect, which on some installations aborts every process (even a serial one) at exit, so the logging system must never trigger it. It defaults to a rank of 0 and a size of 1, indicating a serial execution context, when no launcher variables are found. This allows the logging system to adapt its behavior based on whether it is running in a parallel environment, ensuring that log messages are appropriately filtered and formatted based on the execution context.

    Parameters:
        None

    Returns:
        tuple[int, int]: A tuple containing the MPI rank and size. Defaults to (0, 1) when no MPI launcher is detected.
    """
    rank: Optional[int] = None
    for name in _MPI_RANK_ENV_VARS:
        rank = _env_int(name)
        if rank is not None:
            break

    size: Optional[int] = None
    for name in _MPI_SIZE_ENV_VARS:
        size = _env_int(name)
        if size is not None:
            break

    if rank is None:
        slurm_tasks = _env_int("SLURM_NTASKS")
        slurm_rank = _env_int("SLURM_PROCID")
        if slurm_rank is not None and slurm_tasks is not None and slurm_tasks > 1:
            rank, size = slurm_rank, slurm_tasks

    if rank is None:
        return 0, 1
    if size is None:
        # A launcher set a rank but no known size variable. Every rank must reach the same verdict
        # (rank 0 calling itself serial while the others pick MPI would split the run), so assume a
        # parallel world; mpi4py then reports the true size and a one-rank world runs without MPI.
        size = max(2, rank + 1)
    return rank, max(1, size)


def mpi_launcher_detected() -> bool:
    """
    This function reports whether the process was started by an MPI launcher with more than one rank, judged only from environment variables (no mpi4py import). It is the gate for importing mpi4py lazily: serial and multiprocessing runs never import it.

    Parameters:
        None

    Returns:
        bool: True if launcher variables are present and the world size is greater than one.
    """
    return _detect_mpi_rank_and_size()[1] > 1


def is_root_process() -> bool:
    """
    This function reports whether the current process is the root (rank 0) process of an MPI run, or a serial/multiprocessing parent. It is derived from the launcher environment variables only, so calling it never imports mpi4py. Callers use it to print plain-text summaries from a single rank.

    Parameters:
        None

    Returns:
        bool: True for rank 0 and for any non-MPI process, False for non-zero MPI ranks.
    """
    return _detect_mpi_rank_and_size()[0] == 0


class MPIRankFilter(logging.Filter):
    """Logging filter that injects MPI rank/size and drops INFO/DEBUG records on non-zero ranks."""

    def __init__(self: "MPIRankFilter") -> None:
        """
        This constructor initializes the MPIRankFilter by detecting the MPI rank and size using the _detect_mpi_rank_and_size function. The detected rank and size are stored as instance attributes, which are then used in the filter method to determine whether to allow or suppress log records based on their severity level and the MPI rank. This ensures that only WARNING and above messages are logged from non-zero ranks in a parallel execution context, while all messages are logged in a serial context.

        Parameters:
            None

        Returns:
            None
        """
        super().__init__()
        self.rank, self.size = _detect_mpi_rank_and_size()

    def filter(self: "MPIRankFilter", record: logging.LogRecord) -> bool:
        """
        This method is called for each log record to determine whether it should be logged or not. It injects the MPI rank and size into the log record, allowing the formatter to include this information in the log messages. Additionally, if the rank is non-zero and the log level is below WARNING, it returns False to suppress the log record, ensuring that only important messages are logged from non-zero ranks in a parallel execution context. In a serial context (size=1), all messages are allowed regardless of rank.

        Parameters:
            record (logging.LogRecord): The log record to be filtered.

        Returns:
            bool: True if the record should be logged, False otherwise.
        """
        record.rank = self.rank
        record.size = self.size
        if self.rank != 0 and record.levelno < logging.WARNING:
            return False
        return True


def _build_formatter(size: int) -> logging.Formatter:
    """
    This function builds a logging formatter that includes the MPI rank in the log messages if the size of the MPI communicator is greater than 1. If running in a parallel context, the log format will include the rank information to help distinguish messages from different processes. In a serial context (size=1), the log format will omit the rank information for cleaner output. This allows the logging system to adapt its message formatting based on whether it is running in a parallel environment, ensuring that log messages are informative and appropriately formatted for both contexts.

    Parameters:
        size (int): The size of the MPI communicator.

    Returns:
        logging.Formatter: Configured formatter instance.
    """
    if size > 1:
        fmt = "%(asctime)s - [rank %(rank)d] - %(name)s - %(levelname)s - %(message)s"
    else:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")


def get_logger(name: str = ROOT_LOGGER_NAME) -> logging.Logger:
    """
    This function retrieves a logger with the specified name. If the name is the root logger name or starts with it, it returns the logger directly. Otherwise, it returns a child logger under the root logger. This allows modules to obtain loggers that are properly namespaced under the root "mpasdiag" logger, ensuring that log messages from different modules are organized and can be filtered or formatted consistently based on their hierarchical relationship to the root logger.

    Parameters:
        name (str): The name of the logger to retrieve. Defaults to the root logger name "mpasdiag". Most callers should not override this.

    Returns:
        logging.Logger: The requested logger instance.
    """
    if name == ROOT_LOGGER_NAME or name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def snapshot_logging_config() -> Dict[str, Any]:
    """
    This function captures the parent process's "mpasdiag" logger configuration (effective level, log-file path, and whether a console handler is attached) as a small picklable dictionary. The parallel manager passes it to every pool worker through the pool initializer so that worker processes, which start with an unconfigured logger, honour the same level, --log-file, and --quiet settings as the parent.

    Parameters:
        None

    Returns:
        Dict[str, Any]: Dictionary with keys 'level' (int), 'log_file' (Optional[str]), and 'console' (bool).
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)
    log_file: Optional[str] = None
    console = False
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            log_file = log_file or handler.baseFilename
        elif isinstance(handler, logging.StreamHandler):
            console = True
    return {"level": root.getEffectiveLevel(), "log_file": log_file, "console": console}


def configure_worker_logging(config: Optional[Dict[str, Any]]) -> None:
    """
    This function configures the "mpasdiag" logger inside a pool worker process from a snapshot produced by snapshot_logging_config. It sets the level, replaces any handlers inherited through fork with fresh ones, and attaches a console handler and/or a file handler in append mode. Every worker record carries the process id in its format so interleaved lines from different workers can be told apart. A None or empty config leaves the logger untouched.

    Parameters:
        config (Optional[Dict[str, Any]]): Snapshot with keys 'level', 'log_file', and 'console', or None.

    Returns:
        None
    """
    if not config:
        return

    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(config.get("level", logging.INFO))

    for handler in root.handlers[:]:
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter(
        "%(asctime)s - [pid %(process)d] - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    added = False

    if config.get("console"):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)
        added = True

    log_file = config.get("log_file")
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
            added = True
        except OSError:
            pass

    if added:
        root.propagate = False


class MPASLogger:
    """Root logger configurator for MPASdiag processing code."""

    def __init__(
        self: "MPASLogger",
        name: str = ROOT_LOGGER_NAME,
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """
        This constructor initializes the MPASLogger by configuring the root logger with the specified name, log level, optional file handler, and console handler based on the verbose flag. It first removes any existing handlers from the logger to ensure a clean configuration. Then, it creates an MPIRankFilter instance to inject MPI rank and size information into log records and to filter out INFO/DEBUG messages from non-zero ranks in a parallel context. The constructor sets up a formatter that includes rank information if running in parallel and attaches the appropriate handlers (console and/or file) with the configured formatter and filter. Finally, it disables propagation to prevent duplicate logging in child loggers. This setup ensures that log messages are informative, properly formatted, and appropriately filtered based on the execution context (serial vs parallel) for all modules that use this logger.

        Parameters:
            name (str): The name of the logger to configure. Defaults to the root logger name "mpasdiag". Most callers should not override this.
            level (int): The logging level threshold. Defaults to logging.INFO.
            log_file (Optional[str]): Optional file path to write logs to. If None, no file handler is added. Defaults to None.
            verbose (bool): If True, adds a console handler that outputs to stdout. If False, no console handler is added. Defaults to True.

        Returns:
            None
        """
        global _configured

        self.verbose = verbose
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Replaced handlers must be closed as well as detached, otherwise every reconfiguration leaks the file
        # descriptor of a previous FileHandler.
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

        # Handlers live on this logger only, so records must not also bubble up to the root logger's handlers
        # (that would print every line twice whenever the application configured logging.basicConfig).
        self.logger.propagate = False

        rank_filter = MPIRankFilter()
        formatter = _build_formatter(rank_filter.size)

        if verbose:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            console_handler.addFilter(rank_filter)
            self.logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            file_handler.addFilter(rank_filter)
            self.logger.addHandler(file_handler)

        _configured = True

    def info(self: "MPASLogger", message: str) -> None:
        """
        This method logs a message at INFO severity on the configured root logger. It provides a convenient way for users of the MPASLogger class to log informational messages without needing to directly interact with the underlying logger instance. By using this method, users can ensure that their info messages are properly formatted and filtered according to the configuration set up in the constructor, allowing for consistent logging practices across the MPASdiag processing codebase. Note that in a parallel execution context, INFO messages will only be logged from rank 0 due to the filtering implemented in the MPIRankFilter.

        Parameters:
            message (str): The message to be logged at INFO severity.

        Returns:
            None
        """
        self.logger.info(message)

    def warning(self: "MPASLogger", message: str) -> None:
        """
        This method logs a message at WARNING severity on the configured root logger. Similar to the info method, it provides a convenient way for users of the MPASLogger class to log warning messages without needing to directly interact with the underlying logger instance. By using this method, users can ensure that their warning messages are properly formatted and filtered according to the configuration set up in the constructor, allowing for consistent logging practices across the MPASdiag processing codebase. Note that in a parallel execution context, WARNING messages will be logged from all ranks, but INFO and DEBUG messages will only be logged from rank 0 due to the filtering implemented in the MPIRankFilter.

        Parameters:
            message (str): The message to be logged at WARNING severity.

        Returns:
            None
        """
        self.logger.warning(message)

    def error(self: "MPASLogger", message: str) -> None:
        """
        This method logs a message at ERROR severity on the configured root logger. Like the other logging methods, it provides a convenient interface for users of the MPASLogger class to log error messages without needing to directly access the underlying logger instance. By calling this method, users can ensure that their error messages are properly formatted and filtered according to the configuration set up in the constructor, maintaining consistent logging practices across the MPASdiag processing codebase. Note that in a parallel execution context, ERROR messages will be logged from all ranks, similar to WARNING messages, while INFO and DEBUG messages will only be logged from rank 0 due to the filtering implemented in the MPIRankFilter.

        Parameters:
            message (str): The message to be logged at ERROR severity.

        Returns:
            None
        """
        self.logger.error(message)

    def debug(self: "MPASLogger", message: str) -> None:
        """
        This method logs a message at DEBUG severity on the configured root logger. Similar to the other logging methods, it provides a convenient way for users of the MPASLogger class to log debug messages without needing to directly interact with the underlying logger instance. By using this method, users can ensure that their debug messages are properly formatted and filtered according to the configuration set up in the constructor, allowing for consistent logging practices across the MPASdiag processing codebase. Note that in a parallel execution context, DEBUG messages will only be logged from rank 0 due to the filtering implemented in the MPIRankFilter.

        Parameters:
            message (str): The message to be logged at DEBUG severity.

        Returns:
            None
        """
        self.logger.debug(message)
