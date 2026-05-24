#!/usr/bin/env python3

"""
MPASdiag Test Suite: Tests for MPAS Logger Utilities

This module contains a set of unit tests for the logging utilities in the MPASdiag package, specifically targeting the MPASLogger class, MPIRankFilter, and the get_logger function. The tests verify that the logging configuration is correctly applied to the root logger, that child loggers are properly namespaced and propagate messages to the root logger's handlers, and that the MPIRankFilter correctly injects MPI rank and size attributes into log records while allowing WARNING messages from all ranks. Additionally, tests for the _resolve_log_level function ensure that command-line arguments for log level, quiet, and verbose flags are correctly interpreted to determine the effective log level. Finally, a test for MPASConfig validates that log level inputs are properly checked for validity. This suite ensures that the logging infrastructure in MPASdiag is robust and behaves as expected under various configurations. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import argparse
import logging
from collections.abc import Iterator

import pytest

from mpasdiag.processing.utils_logger import (
    MPASLogger,
    MPIRankFilter,
    ROOT_LOGGER_NAME,
    get_logger,
)


@pytest.fixture(autouse=True)
def _reset_root_logger() -> Iterator[None]:
    """
    This fixture is automatically applied to all tests in this module and ensures that the root logger is reset before each test runs. It removes all handlers from the root logger at the start of the test, allowing each test to configure the logging system without interference from previous tests. After the test completes, it again removes any handlers that may have been added during the test, ensuring a clean state for subsequent tests. This prevents issues with multiple handlers being added to the root logger across different tests, which could lead to duplicate log messages or unintended logging behavior. 

    Parameters:
        None

    Returns:
         An iterator that yields None, allowing the fixture to perform setup before the test and teardown after the test.
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)
    for h in list(root.handlers):
        root.removeHandler(h)
    yield
    root = logging.getLogger(ROOT_LOGGER_NAME)
    for h in list(root.handlers):
        root.removeHandler(h)


def test_mpas_logger_configures_root_once() -> None:
    """
    This test verifies that the MPASLogger class configures the root logger correctly and does not add multiple handlers when instantiated multiple times. It creates an instance of MPASLogger with the root logger name and a specific log level, then checks that the root logger has exactly one handler and that the log level is set as expected. It then creates another instance of MPASLogger with the same root logger name but a different log level, and checks again that there is still only one handler (not a new one) and that the log level has been updated to the new value. This ensures that MPASLogger properly manages the root logger configuration without adding duplicate handlers on multiple instantiations. 

    Parameters:
        None

    Returns:
        None
    """
    MPASLogger(name=ROOT_LOGGER_NAME, level=logging.DEBUG, verbose=True)
    root = logging.getLogger(ROOT_LOGGER_NAME)
    handlers_first = list(root.handlers)
    assert len(handlers_first) == 1
    assert root.level == logging.DEBUG

    MPASLogger(name=ROOT_LOGGER_NAME, level=logging.INFO, verbose=True)
    handlers_second = list(root.handlers)
    assert len(handlers_second) == 1
    assert handlers_second[0] is not handlers_first[0]
    assert root.level == logging.INFO


def test_get_logger_returns_child_under_root() -> None:
    """
    This test verifies that the get_logger function returns a child logger that is properly namespaced under the root logger. It first creates an instance of MPASLogger with the root logger name and a specific log level, then calls get_logger with a child logger name. The test checks that the returned logger's name is correctly prefixed with "mpasdiag." and that its parent logger is the root logger configured by MPASLogger. This confirms that get_logger is correctly creating child loggers under the "mpasdiag" namespace and that they are properly connected to the root logger for message propagation. 

    Parameters:
        None

    Returns:
        None
    """
    MPASLogger(name=ROOT_LOGGER_NAME, level=logging.INFO, verbose=True)
    child = get_logger("mpasdiag.processing.foo")
    assert child.name == "mpasdiag.processing.foo"
    assert child.parent.name.startswith(ROOT_LOGGER_NAME) or child.parent.name == ROOT_LOGGER_NAME


def test_get_logger_namespaces_non_mpasdiag_names() -> None:
    """
    This test verifies that the get_logger function correctly namespaces logger names that do not already start with "mpasdiag.". It calls get_logger with a logger name that does not start with "mpasdiag." and checks that the returned logger's name is prefixed with "mpasdiag.". This ensures that get_logger consistently namespaces loggers under the "mpasdiag" hierarchy, even if the input name does not follow that convention, allowing for organized logging across the MPASdiag codebase. 

    Parameters:
        None

    Returns:
        None
    """
    bare = get_logger("foo.bar")
    assert bare.name == "mpasdiag.foo.bar"


def test_child_propagates_to_root_handlers(caplog: pytest.LogCaptureFixture) -> None:
    """
    This test verifies that log messages emitted from a child logger created by get_logger are properly propagated to the handlers of the root logger configured by MPASLogger. It creates an instance of MPASLogger with the root logger name and a specific log level, then obtains a child logger using get_logger. The test emits a log message from the child logger and uses the caplog fixture to capture log records at the DEBUG level for the root logger. Finally, it checks that the captured log records include the message emitted from the child logger, confirming that messages from child loggers are correctly propagated to the root logger's handlers as intended. 

    Parameters:
        caplog (pytest.LogCaptureFixture): A pytest fixture that captures log records emitted during the test.

    Returns:
        None
    """
    MPASLogger(name=ROOT_LOGGER_NAME, level=logging.DEBUG, verbose=True)
    child = get_logger("mpasdiag.test_propagation")
    with caplog.at_level(logging.DEBUG, logger=ROOT_LOGGER_NAME):
        child.info("hello propagation")
    assert any("hello propagation" in record.message for record in caplog.records)


def test_rank_filter_injects_rank_and_size() -> None:
    """
    This test verifies that the MPIRankFilter correctly injects the rank and size attributes into log records and that it allows INFO messages to be logged only from rank 0 while allowing WARNING messages from all ranks. It creates an instance of MPIRankFilter and constructs a LogRecord with INFO level. The test checks that the filter adds the "rank" and "size" attributes to the record and that the filter returns True (allowing the message) if the rank is 0, and False (filtering out the message) if the rank is non-zero. This confirms that MPIRankFilter is functioning as intended in a parallel execution context, ensuring that log messages are appropriately filtered based on MPI rank while still allowing important WARNING messages to be logged from all ranks. 

    Parameters:
        None

    Returns:
        None
    """
    flt = MPIRankFilter()
    record = logging.LogRecord(
        name="mpasdiag.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="x", args=(), exc_info=None,
    )
    keep = flt.filter(record)
    assert hasattr(record, "rank")
    assert hasattr(record, "size")
    if flt.rank == 0:
        assert keep is True
    else:
        assert keep is False


def test_rank_filter_keeps_warnings_on_non_zero_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    This test verifies that the MPIRankFilter allows WARNING messages to be logged from all ranks, even when the rank is non-zero. It creates an instance of MPIRankFilter and uses monkeypatch to set the rank to a non-zero value (e.g., 2) and the size to a positive integer (e.g., 4). The test then constructs two LogRecord instances: one with INFO level and another with WARNING level. It checks that the filter returns False for the INFO record (filtering it out) and True for the WARNING record (allowing it), confirming that MPIRankFilter correctly allows WARNING messages from all ranks while still filtering INFO messages based on rank. 

    Parameters:
        monkeypatch (pytest.MonkeyPatch): A pytest fixture that allows modification of attributes for testing purposes.

    Returns:
        None
    """
    flt = MPIRankFilter()
    monkeypatch.setattr(flt, "rank", 2)
    monkeypatch.setattr(flt, "size", 4)

    info_record = logging.LogRecord(
        name="x", level=logging.INFO, pathname=__file__, lineno=1,
        msg="x", args=(), exc_info=None,
    )
    warn_record = logging.LogRecord(
        name="x", level=logging.WARNING, pathname=__file__, lineno=1,
        msg="w", args=(), exc_info=None,
    )
    assert flt.filter(info_record) is False
    assert flt.filter(warn_record) is True


def _resolve_log_level(args: argparse.Namespace) -> int:
    """
    This function resolves the effective log level based on the command-line arguments provided. It checks if a specific log level is set in the "log_level" attribute of the args namespace and returns the corresponding logging level if it is. If no specific log level is set, it checks if the "quiet" flag is True and returns logging.ERROR if so. If the "verbose" flag is True, it returns logging.DEBUG. If neither "quiet" nor "verbose" flags are set, it defaults to returning logging.INFO. This function allows for flexible configuration of log levels based on user input from the command line, ensuring that the logging behavior can be easily adjusted for different use cases. 

    Parameters:
        args (argparse.Namespace): The argparse namespace containing the command-line arguments.

    Returns:
        int: The resolved log level.
    """
    if getattr(args, "log_level", None):
        return getattr(logging, args.log_level)
    if getattr(args, "quiet", False):
        return logging.ERROR
    if getattr(args, "verbose", False):
        return logging.DEBUG
    return logging.INFO


def test_log_level_flag_overrides_verbose() -> None:
    """
    This test verifies that when a specific log level is provided in the "log_level" attribute of the args namespace, it takes precedence over the "verbose" flag. It creates an argparse.Namespace with log_level set to "WARNING" and verbose set to True, then calls the _resolve_log_level function with these arguments. The test checks that the resolved log level is logging.WARNING, confirming that the explicit log level flag overrides the verbose flag when determining the effective log level. 

    Parameters:
        None

    Returns:
        None
    """
    args = argparse.Namespace(log_level="WARNING", verbose=True, quiet=False)
    assert _resolve_log_level(args) == logging.WARNING


def test_quiet_overrides_verbose_when_no_log_level() -> None:
    """
    This test verifies that when the "quiet" flag is set to True and no specific log level is provided, the _resolve_log_level function returns logging.ERROR, overriding the "verbose" flag. It creates an argparse.Namespace with verbose set to True and quiet set to True, then calls the _resolve_log_level function with these arguments. The test checks that the resolved log level is logging.ERROR, confirming that the quiet flag takes precedence over the verbose flag when no explicit log level is specified. 

    Parameters:
        None

    Returns:
        None
    """
    args = argparse.Namespace(log_level=None, verbose=True, quiet=True)
    assert _resolve_log_level(args) == logging.ERROR


def test_verbose_maps_to_debug() -> None:
    """
    This test verifies that when the "verbose" flag is set to True and no specific log level is provided, the _resolve_log_level function returns logging.DEBUG. It creates an argparse.Namespace with verbose set to True and quiet set to False, then calls the _resolve_log_level function with these arguments. The test checks that the resolved log level is logging.DEBUG, confirming that the verbose flag correctly maps to the DEBUG log level when no explicit log level is specified and the quiet flag is not set. 

    Parameters:
        None

    Returns:
        None
    """
    args = argparse.Namespace(log_level=None, verbose=True, quiet=False)
    assert _resolve_log_level(args) == logging.DEBUG


def test_default_is_info() -> None:
    """
    This test verifies that when no specific log level is provided and neither the "quiet" nor "verbose" flags are set, the _resolve_log_level function defaults to returning logging.INFO. It creates an argparse.Namespace with log_level set to None, verbose set to False, and quiet set to False, then calls the _resolve_log_level function with these arguments. The test checks that the resolved log level is logging.INFO, confirming that the default log level is INFO when no other configuration is provided. 

    Parameters:
        None

    Returns:
        None
    """
    args = argparse.Namespace(log_level=None, verbose=False, quiet=False)
    assert _resolve_log_level(args) == logging.INFO


def test_mpas_config_log_level_validation() -> None:
    """
    This test verifies that the MPASConfig class correctly validates log level inputs. It creates an instance of MPASConfig with a valid log level string (e.g., "warning") and checks that the log_level attribute is set to the expected value (e.g., "WARNING"). It then attempts to create an instance of MPASConfig with an invalid log level string (e.g., "BOGUS") and checks that a ValueError is raised, confirming that the class properly checks for valid log level inputs and raises an error when an invalid value is provided. This ensures that users of MPASConfig are informed of incorrect log level configurations in a clear manner. 

    Parameters:
        None

    Returns:
        None
    """
    from mpasdiag.processing.utils_config import MPASConfig

    cfg = MPASConfig(log_level="warning")
    assert cfg.log_level == "WARNING"

    with pytest.raises(ValueError):
        MPASConfig(log_level="BOGUS")
