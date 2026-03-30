#!/usr/bin/env python3

"""
MPASdiag Core Processing Module: Error and Logging Utilities

This module defines the MPASLogger class, which provides a standardized logging interface for MPAS diagnostic processing workflows. The MPASLogger class enables consistent reporting of informational messages, warnings, and errors throughout the execution of MPAS diagnostic scripts, facilitating easier debugging and monitoring of workflow progress. By utilizing Python's built-in logging module, MPASLogger allows users to configure logging levels, output formats, and destinations (console and/or file) to suit their specific needs. This utility is designed to enhance the usability and maintainability of MPAS diagnostic processing code by providing a centralized mechanism for handling log messages in a structured and informative manner. 

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
    """ MPASLogger provides a standardized logging interface for MPAS diagnostic processing workflows, enabling consistent reporting of informational messages, warnings, and errors. """
    
    def __init__(self: "MPASLogger", 
                 name: str = "MPASdiag", 
                 level: int = logging.INFO,
                 log_file: Optional[str] = None, 
                 verbose: bool = True) -> None:
        """
        This constructor initializes an instance of the MPASLogger class with specified logging configuration parameters. It sets up the underlying Python logger with the provided name, logging level, and output handlers based on the log_file and verbose parameters. The logger is configured to format log messages with timestamps, logger name, severity level, and message content for clear and informative output. Depending on the parameters, the logger can write messages to the console (stdout) for real-time monitoring and/or to a specified log file for persistent storage and later review. This setup allows users to easily track workflow progress, identify potential issues, and maintain a record of execution details for MPAS diagnostic processing tasks. 

        Parameters:
            name (str): Name of the logger instance, used to identify log messages from this source. Default is "MPASdiag".
            level (int): Logging level threshold (e.g., logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR) that determines which messages are recorded. Default is logging.INFO.
            log_file (Optional[str]): Path to a log file where messages should be written. If None, no file output is configured. Default is None.
            verbose (bool): If True, enables console output of log messages to stdout. If False, disables console output. Default is True. 

        Returns:
            None: Initializes the logger instance with the specified configuration for handling log messages. 
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
    
    def info(self: "MPASLogger", 
             message: str) -> None:
        """
        This method logs an informational message at INFO level (20) to configured output handlers for reporting general workflow progress, completion events, or other non-critical information. It forwards the message string to the underlying Python logger's info() method, which writes to console and/or file handlers based on logger configuration. INFO-level messages are used for routine updates such as successful completion of processing steps, key milestones reached, or any other relevant information that helps users understand the flow of execution without indicating any problems or issues. These messages provide users with insights into the normal operation of the MPAS diagnostic processing workflow and can be helpful for tracking progress and confirming successful execution. 

        Parameters:
            message (str): Human-readable informational message string describing the workflow progress or relevant details. 

        Returns:
            None: Writes message to configured log handlers with INFO severity level. 
        """
        self.logger.info(message)
    
    def warning(self: "MPASLogger", 
                message: str) -> None:
        """
        This method logs a warning message at WARNING level (30) to configured output handlers for reporting potential issues, unexpected conditions, or non-critical problems that may require user attention but do not prevent successful execution of the workflow. It forwards the message string to the underlying Python logger's warning() method, which writes to console and/or file handlers based on logger configuration. WARNING-level messages are used for situations such as missing optional input files, deprecated configuration settings, minor data inconsistencies, or any other conditions that may indicate a potential issue but do not necessarily cause workflow failure. These messages provide users with important information about potential concerns or areas that may require further investigation without indicating a critical failure. 

        Parameters:
            message (str): Human-readable warning message string describing the potential issue or unexpected condition that may require user attention. 

        Returns:
            None: Writes message to configured log handlers with WARNING severity level. 
        """
        self.logger.warning(message)
    
    def error(self: "MPASLogger", 
              message: str) -> None:
        """
        This method logs an error message at ERROR level (40) to configured output handlers for reporting critical failures, exceptions, or conditions that prevent successful execution of the workflow. It forwards the message string to the underlying Python logger's error() method, which writes to console and/or file handlers based on logger configuration. ERROR-level messages are used for situations such as missing required input files, invalid configuration settings, unhandled exceptions, or any other conditions that indicate a critical failure in the workflow. These messages provide users with clear and concise information about the nature of the failure, allowing them to identify and address the underlying issue to restore successful execution of the MPAS diagnostic processing workflow. 

        Parameters:
            message (str): Human-readable error message string describing the critical failure or condition that prevents successful execution of the workflow. 

        Returns:
            None: Writes message to configured log handlers with ERROR severity level. 
        """
        self.logger.error(message)
    
    def debug(self: "MPASLogger", 
              message: str) -> None:
        """
        This method logs a debug message at DEBUG level (10) to configured output handlers for reporting detailed diagnostic information, variable values, execution flow details, or any other information that may be useful for troubleshooting and debugging purposes. It forwards the message string to the underlying Python logger's debug() method, which writes to console and/or file handlers based on logger configuration. DEBUG-level messages are typically used during development or when troubleshooting specific issues, as they can provide in-depth insights into the internal workings of the workflow without overwhelming users with too much information during normal execution. These messages can include details such as variable states, function entry and exit points, loop iterations, or any other relevant information that helps developers understand the behavior of the code and identify potential issues. 

        Parameters:
            message (str): Human-readable debug message string describing the detailed diagnostic information or execution flow details that may be useful for troubleshooting and debugging purposes. 

        Returns:
            None: Writes message to configured log handlers with DEBUG severity level. 
        """
        self.logger.debug(message)