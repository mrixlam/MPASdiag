#!/usr/bin/env python3

"""
MPASdiag Core Processing Module: Error and Logging Utilities

This module provides a centralized logging utility for the MPASdiag processing code. It defines a custom logger class, MPASLogger, which configures the root logger with appropriate handlers and formatters to include MPI rank information when running in a parallel context. The MPIRankFilter is used to inject MPI rank and size into log records and to filter out INFO and DEBUG messages from non-zero ranks in a parallel execution environment. This ensures that log messages are informative and appropriately filtered based on the execution context, allowing for consistent logging practices across the MPASdiag processing codebase. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.1.0
"""

import sys
import logging
from typing import Optional

ROOT_LOGGER_NAME = "mpasdiag"

_configured: bool = False


def _detect_mpi_rank_and_size() -> "tuple[int, int]":
    """
    This function attempts to detect the MPI rank and size using the mpi4py library. If mpi4py is available and can be imported successfully, it retrieves the rank and size from the MPI communicator. If mpi4py is not available or an error occurs during import, it defaults to returning a rank of 0 and a size of 1, indicating a serial execution context. This allows the logging system to adapt its behavior based on whether it is running in a parallel environment, ensuring that log messages are appropriately filtered and formatted based on the execution context. 

    Parameters:
        None

    Returns:
        tuple[int, int]: A tuple containing the MPI rank and size. Defaults to (0, 1) if MPI is not available.
    """
    try:
        from mpi4py import MPI  # type: ignore
        comm = MPI.COMM_WORLD
        return comm.Get_rank(), comm.Get_size()
    except Exception:
        return 0, 1


class MPIRankFilter(logging.Filter):
    """ Logging filter that injects MPI rank/size and drops INFO/DEBUG records on non-zero ranks. """

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

    def filter(self: "MPIRankFilter",
               record: logging.LogRecord) -> bool:
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


class MPASLogger:
    """ Root logger configurator for MPASdiag processing code. """

    def __init__(self: "MPASLogger",
                 name: str = ROOT_LOGGER_NAME,
                 level: int = logging.INFO,
                 log_file: Optional[str] = None,
                 verbose: bool = True) -> None:
        """
        This constructor initializes the MPASLogger by configuring the root logger with the specified name, log level, optional file handler, and console handler based on the verbose flag. It first removes any existing handlers from the logger to ensure a clean configuration. Then, it creates an MPIRankFilter instance to inject MPI rank and size information into log records and to filter out INFO/DEBUG messages from non-zero ranks in a parallel context. The constructor sets up a formatter that includes rank information if running in parallel and attaches the appropriate handlers (console and/or file) with the configured formatter and filter. Finally, it disables propagation to prevent duplicate logging in child loggers. This setup ensures that log messages are informative, properly formatted, and appropriately filtered based on the execution context (serial vs parallel) for all modules that use this logger. 

        Parameters:
            name (str): The name of the logger to configure. Defaults to the root logger name "mpasdiag". Most callers should not override this.
            level (int): The logging level threshold. Defaults to logging.INFO.
            log_file (Optional[str]): Optional file path to write logs to. If None, no file handler is added. Defaults to None.
            verbose (bool): If True, adds a console handler that outputs to stdout. If False, no console handler is added. Defaults to True.
        """
        global _configured

        self.verbose = verbose
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)

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

    def info(self: "MPASLogger",
             message: str) -> None:
        """
        This method logs a message at INFO severity on the configured root logger. It provides a convenient way for users of the MPASLogger class to log informational messages without needing to directly interact with the underlying logger instance. By using this method, users can ensure that their info messages are properly formatted and filtered according to the configuration set up in the constructor, allowing for consistent logging practices across the MPASdiag processing codebase. Note that in a parallel execution context, INFO messages will only be logged from rank 0 due to the filtering implemented in the MPIRankFilter.

        Parameters:
            message (str): The message to be logged at INFO severity.

        Returns:
            None
        """
        self.logger.info(message)

    def warning(self: "MPASLogger",
                message: str) -> None:
        """
        This method logs a message at WARNING severity on the configured root logger. Similar to the info method, it provides a convenient way for users of the MPASLogger class to log warning messages without needing to directly interact with the underlying logger instance. By using this method, users can ensure that their warning messages are properly formatted and filtered according to the configuration set up in the constructor, allowing for consistent logging practices across the MPASdiag processing codebase. Note that in a parallel execution context, WARNING messages will be logged from all ranks, but INFO and DEBUG messages will only be logged from rank 0 due to the filtering implemented in the MPIRankFilter.

        Parameters:
            message (str): The message to be logged at WARNING severity.

        Returns:
            None
        """
        self.logger.warning(message)

    def error(self: "MPASLogger",
              message: str) -> None:
        """
        This method logs a message at ERROR severity on the configured root logger. Like the other logging methods, it provides a convenient interface for users of the MPASLogger class to log error messages without needing to directly access the underlying logger instance. By calling this method, users can ensure that their error messages are properly formatted and filtered according to the configuration set up in the constructor, maintaining consistent logging practices across the MPASdiag processing codebase. Note that in a parallel execution context, ERROR messages will be logged from all ranks, similar to WARNING messages, while INFO and DEBUG messages will only be logged from rank 0 due to the filtering implemented in the MPIRankFilter.

        Parameters:
            message (str): The message to be logged at ERROR severity.

        Returns:
            None
        """
        self.logger.error(message)

    def debug(self: "MPASLogger",
              message: str) -> None:
        """
        This method logs a message at DEBUG severity on the configured root logger. Similar to the other logging methods, it provides a convenient way for users of the MPASLogger class to log debug messages without needing to directly interact with the underlying logger instance. By using this method, users can ensure that their debug messages are properly formatted and filtered according to the configuration set up in the constructor, allowing for consistent logging practices across the MPASdiag processing codebase. Note that in a parallel execution context, DEBUG messages will only be logged from rank 0 due to the filtering implemented in the MPIRankFilter. 

        Parameters:
            message (str): The message to be logged at DEBUG severity.

        Returns:
            None
        """
        self.logger.debug(message)