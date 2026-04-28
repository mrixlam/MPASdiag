#!/usr/bin/env python3
"""
MPASdiag Test Suite: CLI Logging Tests

This module contains unit tests for the logging setup functionality of the MPASdiag CLI. It verifies that the logging configuration behaves as expected under different verbosity settings and that loggers are properly attached to the CLI instance. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load standard libraries
import os
import pytest


class TestLoggingSetup:
    """ Test logging setup functionality for different verbosity modes. """

    def test_setup_logging_normal_mode(self: 'TestLoggingSetup') -> None:
        """
        This test verifies that the `setup_logging` method of the MPASUnifiedCLI class correctly initializes a logger and attaches it to the CLI instance in normal mode (neither verbose nor quiet). 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig

        cli = MPASUnifiedCLI()
        config = MPASConfig(grid_file='test.nc', data_dir='data/')

        logger = cli.setup_logging(config)

        assert logger is not None
        assert cli.logger is logger

    def test_setup_logging_verbose_mode(self: 'TestLoggingSetup') -> None:
        """
        This test checks that when `verbose=True` is set in the configuration, the `setup_logging` method configures a logger with a low numeric log level (DEBUG or lower). This ensures that verbose mode results in detailed logging output.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig

        cli = MPASUnifiedCLI()
        config = MPASConfig(grid_file='test.nc', data_dir='data/', verbose=True)

        logger = cli.setup_logging(config)

        assert logger is not None
        assert logger.logger.level <= 10  

    def test_setup_logging_quiet_mode(self: 'TestLoggingSetup') -> None:
        """
        This test verifies that when `quiet=True` is set in the configuration, the `setup_logging` method configures a logger with a high numeric log level (ERROR or higher). This ensures that quiet mode suppresses most logging output.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig

        cli = MPASUnifiedCLI()
        config = MPASConfig(grid_file='test.nc', data_dir='data/', quiet=True)

        logger = cli.setup_logging(config)

        assert logger is not None
        assert logger.logger.level >= 40  


class TestSetupLoggingEdgeCases:
    """ Test logging setup with log file. """
    
    def test_setup_logging_with_log_file(self: 'TestSetupLoggingEdgeCases') -> None:
        """
        This test verifies that the `setup_logging` method can accept a log file path and properly configure a file handler for logging output. It checks that the logger is created and that the specified log file is created on the filesystem.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        import tempfile
        
        cli = MPASUnifiedCLI()
        
        config = MPASConfig(
            grid_file='test.nc',
            data_dir='data/',
            verbose=True
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = f.name
        
        try:
            logger = cli.setup_logging(config, log_file=log_file)
            
            assert logger is not None
            assert os.path.exists(log_file)
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
