#!/usr/bin/env python3

"""
MPAS Logging Utilities

This module provides comprehensive logging functionality for MPAS data analysis workflows including console and file output configuration, severity level management, and message formatting. It implements the MPASLogger class as a lightweight convenience wrapper around Python's standard logging module, simplifying logger setup and message routing for MPASdiag processing and visualization scripts. The logger supports simultaneous output to both console (stdout) and file destinations with independent severity thresholds, provides standardized timestamp formatting for all log messages with clear level indicators, includes convenience methods (info, warning, error, debug) that forward to the underlying logger for clean API usage, and handles logger initialization with automatic cleanup of existing handlers to prevent duplicate messages. Core capabilities include configurable verbosity for controlling console output in interactive vs batch modes, optional file logging for persistent run records and debugging, integration with Python's logging hierarchy for compatibility with other libraries, and minimal overhead suitable for both development environments and production operational workflows requiring detailed execution logs and error tracking.

Classes:
    MPASLogger: Logging utility wrapper class providing simplified configuration and message routing for MPAS analysis workflows.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import sys
import logging
from typing import Optional


class MPASLogger:
    """
    Logging utility wrapper for MPAS analysis scripts with configurable console and file output handlers. This class provides a lightweight convenience layer around Python's standard logging module, simplifying logger configuration and message routing for MPASdiag workflows. The logger supports simultaneous output to both console (stdout) and file destinations, with configurable severity thresholds to control message filtering. Simple forwarding methods (info, warning, error, debug) expose the underlying logger's functionality while maintaining a clean interface suitable for examples, tests, and production analysis scripts.
    """
    
    def __init__(self, name: str = "mpas_analysis", level: int = logging.INFO,
                 log_file: Optional[str] = None, verbose: bool = True) -> None:
        """
        Initialize and configure an MPAS logging instance with console and optional file output handlers using Python's standard logging framework. This constructor creates a named logger, sets the logging level threshold, clears any existing handlers to prevent duplicate messages, and configures a standardized timestamp formatter for all log messages. When verbose mode is enabled, a console handler directs logs to stdout for real-time monitoring during analysis runs. When a log file path is provided, a file handler writes persistent logs to disk for post-run review and debugging. This dual-output configuration supports both interactive development and production batch processing workflows in MPASdiag.

        Parameters:
            name (str): Logger name for identification in multi-logger applications (default: "mpas_analysis").
            level (int): Minimum logging level threshold from logging module constants (DEBUG=10, INFO=20, WARNING=30, ERROR=40) (default: logging.INFO).
            log_file (Optional[str]): Absolute or relative path to log file for persistent storage, None disables file logging (default: None).
            verbose (bool): Enable console output handler to stdout for real-time log display during execution (default: True).

        Returns:
            None: Initializes instance attributes self.logger with configured handlers and formatters.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        self.logger.handlers.clear()
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if verbose:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str) -> None:
        """
        Log an informational message at INFO level (20) to configured output handlers for documenting normal operation and analysis progress. This method forwards the message string to the underlying Python logger's info() method, which writes to console and/or file handlers based on the logger configuration established during initialization. INFO-level messages are used for reporting significant events during workflow execution such as file loading completion, processing milestones, configuration summaries, and successful operation outcomes. These messages appear in standard logging output and provide users with runtime feedback about analysis progression without overwhelming them with debug-level details.

        Parameters:
            message (str): Human-readable informational message string to log, typically describing workflow progress or completion events.

        Returns:
            None: Writes message to configured log handlers with INFO severity level.
        """
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """
        Log a warning message at WARNING level (30) to configured output handlers for alerting users to potential issues or unexpected conditions that do not prevent execution. This method forwards the message string to the underlying Python logger's warning() method, which writes to console and/or file handlers based on logger configuration. WARNING-level messages are used for non-critical issues such as missing optional configuration parameters, deprecated function usage, data quality concerns, unusual but handled conditions, or recoverable errors where the analysis can continue with reduced functionality or default values. These messages help users identify potential problems that may affect results without halting workflow execution.

        Parameters:
            message (str): Human-readable warning message string describing the potential issue or unexpected condition encountered.

        Returns:
            None: Writes message to configured log handlers with WARNING severity level.
        """
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """
        Log an error message at ERROR level (40) to configured output handlers for reporting critical failures or conditions that prevent successful completion of requested operations. This method forwards the message string to the underlying Python logger's error() method, which writes to console and/or file handlers based on logger configuration. ERROR-level messages are used for serious problems such as missing required input files, invalid data format, failed file I/O operations, configuration errors, unrecoverable exceptions, or conditions that cause workflow termination. These high-priority messages alert users to failures requiring immediate attention and typically appear before raising exceptions or returning error codes in MPASdiag workflows.

        Parameters:
            message (str): Human-readable error message string describing the critical failure or condition that prevents successful execution.

        Returns:
            None: Writes message to configured log handlers with ERROR severity level.
        """
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """
        Log a debug-level message at DEBUG level (10) to configured output handlers for detailed diagnostic information useful during development and troubleshooting. This method forwards the message string to the underlying Python logger's debug() method, which only writes to handlers when the logger level is set to DEBUG or lower. DEBUG-level messages provide verbose operational details such as variable values, intermediate computation results, function entry/exit points, loop iterations, and detailed execution flow information. These low-priority messages are typically disabled in production runs to reduce log volume but are invaluable for developers diagnosing issues or understanding code behavior during testing and debugging workflows.

        Parameters:
            message (str): Human-readable debug message string containing detailed diagnostic information for troubleshooting.

        Returns:
            None: Writes message to configured log handlers with DEBUG severity level if logger threshold permits.
        """
        self.logger.debug(message)