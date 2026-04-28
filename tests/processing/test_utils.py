#!/usr/bin/env python3
"""
MPASdiag Test Suite: Tests for MPAS Utility Classes and Helper Functions

This module contains unit tests for the utility classes and helper functions defined in `mpasdiag.processing.utils_parser` and `mpasdiag.processing.utils_config`. The tests cover the creation of argument parsers for general, surface, wind, and cross-section plotting, as well as the conversion of parsed arguments into configuration objects. The tests ensure that the parsers correctly recognize required and optional arguments, apply defaults, and that the conversion functions properly map argument namespaces to `MPASConfig` instances with the expected attributes and types.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules for testing
import sys
import pytest
from pathlib import Path

from mpasdiag.processing.utils_file import FileManager
from mpasdiag.processing.utils_parser import ArgumentParser

package_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_dir))

file_manager = FileManager()
get_available_memory = file_manager.get_available_memory
format_file_size = file_manager.format_file_size
create_output_filename = file_manager.create_output_filename


def create_mock_memory_getter(available_bytes: int):
    """
    This helper function creates a mock memory getter function that simulates available memory in bytes for testing purposes. It returns a lambda function that converts the provided byte value to gigabytes when called. This allows tests to simulate different memory availability scenarios without relying on actual system memory conditions. 

    Parameters:
        available_bytes (int): Simulated available memory in bytes.

    Returns:
        Callable: Function that returns memory in gigabytes.
    """
    return lambda: available_bytes / (1024**3)


class TestArgumentParser:
    """ Tests for ArgumentParser utility class behavior, specifically the creation of argument parsers and the conversion of parsed arguments to configuration objects. """
    
    def test_create_parser(self: 'TestArgumentParser') -> None:
        """
        This test validates that ArgumentParser.create_parser successfully creates an argument parser instance with the expected command-line arguments for grid file, data directory, and variable selection. The test confirms that the parser can parse a sample set of arguments and that the resulting namespace contains the correct values for each argument. This parser creation testing ensures that the command-line interface is properly defined and that users can specify necessary parameters for processing through CLI arguments. The test checks that required arguments are recognized and that their values are correctly assigned in the parsed namespace. 

        Parameters:
            None

        Returns:
            None
        """
        parser = ArgumentParser.create_parser()
        
        assert parser is not None
        
        args = parser.parse_args([
            '--grid-file', 'grid.nc',
            '--data-dir', './data',
            '--var', 'rainc'
        ])
        
        assert args.grid_file == 'grid.nc'
        assert args.data_dir == './data'
        assert args.var == 'rainc'
    
    def test_parse_args_to_config(self: 'TestArgumentParser') -> None:
        """
        This test confirms that ArgumentParser.parse_args_to_config correctly converts a parsed argument namespace into an MPASConfig object with the expected field values. The test creates a sample set of arguments, parses them, and then converts the parsed namespace to a configuration object. The test validates that the resulting MPASConfig instance has attributes matching the provided arguments (grid_file, data_dir, variable, dpi, verbose) confirming that the conversion process maps CLI arguments to configuration fields accurately. This conversion testing ensures that users can seamlessly transition from command-line input to structured configuration objects for use in processing workflows. The test checks that all specified parameters are correctly reflected in the created configuration object. 

        Parameters:
            None

        Returns:
            None
        """
        parser = ArgumentParser.create_parser()

        args = parser.parse_args([
            '--grid-file', 'grid.nc',
            '--data-dir', './data',
            '--var', 'total',
            '--dpi', '400',
            '--verbose'
        ])
        
        config = ArgumentParser.parse_args_to_config(args)
        
        assert config.grid_file == 'grid.nc'
        assert config.data_dir == './data'
        assert config.variable == 'total'
        assert config.dpi == pytest.approx(400)
        assert config.verbose is True


if __name__ == "__main__":
    pytest.main([__file__])
