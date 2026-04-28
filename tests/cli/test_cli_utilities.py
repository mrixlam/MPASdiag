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


class TestMappingFunctions:
    """ Test argument mapping helper functions. """
    
    def test_map_overlay_args_with_variables(self: 'TestMappingFunctions') -> None:
        """
        This test verifies that the argument mapping function correctly parses the `--variables` argument when provided as a comma-separated string. It ensures that the resulting configuration contains a list of variable names as expected.

        Parameters:
            None

        Returns:
            None
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
    
    
    def test_print_config_summary_with_values(self: 'TestUtilityFunctions') -> None:
        """
        This test verifies that the `_print_config_summary` helper function executes without error when the configuration contains valid values. It ensures that the function can process and log a typical configuration setup, confirming that it operates correctly under normal conditions.

        Parameters:
            None

        Returns:
            None
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
    
    def test_print_config_summary_filters_none_values(self: 'TestUtilityFunctions') -> None:
        """
        This test verifies that the `_print_config_summary` helper function executes without error when the configuration contains `None` values. It ensures that the function can handle and filter out `None` fields gracefully, confirming that it operates correctly even when some configuration parameters are not set.

        Parameters:
            None

        Returns:
            None
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
