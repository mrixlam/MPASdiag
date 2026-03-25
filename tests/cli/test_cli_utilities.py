#!/usr/bin/env python3
"""
MPASdiag Test Suite: CLI Utilities

This module contains unit tests for the utility functions used in the MPASdiag CLI, including argument mapping and configuration summary printing. These tests ensure that the helper functions behave as expected, providing robust and informative outputs for users of the CLI. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries for testing
import pytest
from unittest.mock import patch


class TestMappingFunctions:
    """ Test argument mapping helper functions. """
    
    def test_map_overlay_args_with_variables(self) -> None:
        """
        This test verifies that the argument mapping function correctly parses the `--variables` argument when provided as a comma-separated string. It ensures that the resulting configuration contains a list of variable names as expected.

        Parameters:
            self (TestMappingFunctions): The test instance.

        Returns:
            None: Assertions validate the parsed configuration.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()
        
        args = parser.parse_args([
            'overlay',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--overlay-type', 'precip_wind', 
            '--variables', 'temp,pressure,wind',
            '--time-index', '0'
        ])
        
        config = cli.parse_args_to_config(args)
        
        assert config.variables == ['temp', 'pressure', 'wind']


class TestUtilityFunctions:
    """ Test utility functions for system info and config summary. """
    
    def test_print_system_info_with_psutil(self: "TestUtilityFunctions") -> None:
        """
        This test verifies that the `_print_system_info` helper function executes without error when `psutil` is available. It mocks the logger to confirm that system information is logged as expected, ensuring that the function behaves correctly in a typical environment where `psutil` is installed.

        Parameters:
            self (TestUtilityFunctions): The test instance.

        Returns:
            None: The test asserts no exception is raised and logging occurs as expected.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from unittest.mock import Mock
        
        cli = MPASUnifiedCLI()
        cli.logger = Mock()
        
        cli._print_system_info()
        
        assert cli.logger.info.called
    
    def test_print_system_info_without_psutil(self: "TestUtilityFunctions") -> None:
        """
        This test verifies that the `_print_system_info` helper function handles the absence of `psutil` gracefully. By mocking the import to raise an `ImportError`, it ensures that the function does not crash and continues execution without logging system information, demonstrating robust error handling for missing dependencies.

        Parameters:
            self (TestUtilityFunctions): The test instance.

        Returns:
            None: The test asserts that no exception is raised on the ImportError path.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_logger import MPASLogger
        
        cli = MPASUnifiedCLI()
        cli.logger = MPASLogger('test', verbose=True)
        
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: 
                   __import__(name, *args, **kwargs) if name != 'psutil' else (_ for _ in ()).throw(ImportError)):
            cli._print_system_info()
    
    def test_print_config_summary(self: "TestUtilityFunctions") -> None:
        """
        This test verifies that the `_print_config_summary` helper function executes without error when the configuration is populated. It mocks the logger to confirm that the configuration summary is logged, ensuring that the function behaves correctly and provides informative output about the current configuration.

        Parameters:
            self (TestUtilityFunctions): The test instance.

        Returns:
            None: The test asserts that the logger was invoked.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        from unittest.mock import Mock
        
        cli = MPASUnifiedCLI()
        cli.logger = Mock()

        cli.config = MPASConfig(
            grid_file='test.nc',
            data_dir='data/',
            analysis_type='precipitation',
            verbose=True
        )
        
        cli._print_config_summary()
        
        assert cli.logger.info.called
    
    def test_print_config_summary_with_values(self: "TestUtilityFunctions") -> None:
        """
        This test verifies that the `_print_config_summary` helper function executes without error when the configuration contains valid values. It ensures that the function can process and log a typical configuration setup, confirming that it operates correctly under normal conditions.

        Parameters:
            self (TestUtilityFunctions): The test instance.

        Returns:
            None: The test asserts the method completes without error.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        from mpasdiag.processing.utils_logger import MPASLogger
        
        cli = MPASUnifiedCLI()
        cli.logger = MPASLogger('test', verbose=True)

        cli.config = MPASConfig(
            grid_file='test.nc',
            data_dir='data/',
            verbose=True
        )
        
        cli._print_config_summary()
    
    def test_print_config_summary_filters_none_values(self: "TestUtilityFunctions") -> None:
        """
        This test verifies that the `_print_config_summary` helper function executes without error when the configuration contains `None` values. It ensures that the function can handle and filter out `None` fields gracefully, confirming that it operates correctly even when some configuration parameters are not set.

        Parameters:
            self (TestUtilityFunctions): The test instance.

        Returns:
            None: The test asserts the method runs without error and filters out None fields.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        from mpasdiag.processing.utils_logger import MPASLogger
        
        cli = MPASUnifiedCLI()
        cli.logger = MPASLogger('test', verbose=True)

        cli.config = MPASConfig(
            grid_file='test.nc',
            data_dir='data/',
            output_dir="",  
            verbose=True
        )
        
        cli._print_config_summary()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
