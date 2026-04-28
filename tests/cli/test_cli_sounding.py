#!/usr/bin/env python3

"""
MPASdiag Test Suite: CLI Sounding Analysis Tests

This module contains a suite of tests for the command-line interface (CLI) related to sounding analyses in the MPASdiag package. The tests are designed to verify that the CLI correctly parses arguments for sounding analyses, maps those arguments to the appropriate configuration fields, and dispatches the analysis to the correct function without errors. The tests cover various aspects of the CLI, including subcommand recognition, argument parsing, default values, and dispatch logic. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: March 2026
Version: 1.0.0
"""
# Load necessary libraries for testing
import pytest
import argparse

from mpasdiag.processing.cli_unified import MPASUnifiedCLI


@pytest.fixture
def cli() -> MPASUnifiedCLI:
    """ 
    This fixture creates and returns an instance of the MPASUnifiedCLI class, which is used for testing the CLI parsing and dispatch logic related to sounding analyses. It allows tests to access the CLI instance directly, enabling them to create argument parsers, parse command-line arguments into configuration objects, and test the dispatching of analysis functions based on the specified analysis type.

    Parameters:
        None

    Returns:
        MPASUnifiedCLI: An instance of the MPASUnifiedCLI class, which contains methods for creating the argument parser, parsing arguments into a configuration object, and dispatching analysis functions based on the specified analysis type.
    """
    return MPASUnifiedCLI()


@pytest.fixture
def parser(cli: 'MPASUnifiedCLI') -> argparse.Namespace:
    """
    This fixture creates and returns the argument parser from the MPASUnifiedCLI instance. It allows tests to access the parser directly, enabling them to simulate command-line inputs and verify that the parser correctly accepts and parses the 'sounding' subcommand and its associated options.

    Parameters:
        cli ('MPASUnifiedCLI'): The CLI instance provided by the cli fixture.

    Returns:
        argparse.Namespace: The argument parser created by the MPASUnifiedCLI instance.
    """
    return cli.create_main_parser()


class TestSoundingArgMapping:
    """ Confirm that the CLI parsing logic correctly maps parsed arguments from the 'sounding' subcommand to the appropriate fields in the configuration object used for analysis. """

    def test_parse_args_to_config_maps_fields(self: 'TestSoundingArgMapping', 
                                              cli: 'MPASUnifiedCLI', 
                                              parser: 'argparse.Namespace') -> None:
        """ 
        This test verifies that the parse_args_to_config method of the CLI correctly maps the parsed arguments from a 'sounding' subcommand input to the corresponding fields in the configuration object. It simulates a command-line input that includes the 'sounding' subcommand along with various options, parses the arguments, and then asserts that the resulting configuration object has the expected values for analysis type, longitude, latitude, and flags for showing indices and parcel information.

        Parameters:
            cli ('MPASUnifiedCLI'): The CLI instance provided by the cli fixture.
            parser ('argparse.Namespace'): The argument parser provided by the parser fixture.

        Returns:
            None
        """
        args = parser.parse_args([
            'sounding',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--lon', '-97.5',
            '--lat', '36.0',
            '--time-index', '3',
            '--show-indices',
            '--show-parcel',
        ])

        config = cli.parse_args_to_config(args)

        assert config.analysis_type == 'sounding'
        assert config.sounding_lat == pytest.approx(36.0)
        assert config.sounding_lon == pytest.approx(-97.5)
        assert config.show_indices is True
        assert config.show_parcel is True

    def test_parse_args_to_config_skewt_alias(self: 'TestSoundingArgMapping', 
                                               cli: 'MPASUnifiedCLI', 
                                               parser: 'argparse.Namespace') -> None:
        """ 
        This test verifies that the parse_args_to_config method correctly maps the 'skewt' alias to the 'sounding' analysis type in the configuration object. It simulates a command-line input that includes the 'skewt' subcommand along with required options, parses the arguments, and then asserts that the resulting configuration object has 'skewt' as the analysis type, confirming that the alias is preserved in the configuration.

        Parameters:
            cli ('MPASUnifiedCLI'): The CLI instance provided by the cli fixture.
            parser ('argparse.Namespace'): The argument parser provided by the parser fixture.

        Returns:
            None
        """
        args = parser.parse_args([
            'skewt',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--lon', '10', '--lat', '20',
            '--time-index', '0',
        ])

        config = cli.parse_args_to_config(args)
        assert config.analysis_type == 'skewt'

    def test_time_range_sets_batch_mode(self: 'TestSoundingArgMapping', 
                                        cli: 'MPASUnifiedCLI', 
                                        parser: 'argparse.Namespace') -> None:
        """
        This test verifies that the parse_args_to_config method correctly sets the batch_mode flag when a time range is specified. It simulates a command-line input that includes the 'sounding' subcommand along with a time range, parses the arguments, and then asserts that the resulting configuration object has batch_mode set to True.

        Parameters:
            cli ('MPASUnifiedCLI'): The CLI instance provided by the cli fixture.
            parser ('argparse.Namespace'): The argument parser provided by the parser fixture.

        Returns:
            None
        """
        args = parser.parse_args([
            'sounding',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--lon', '0', '--lat', '0',
            '--time-range', '0', '5',
        ])

        config = cli.parse_args_to_config(args)
        assert config.batch_mode is True


if __name__ == "__main__":
    pytest.main([__file__])
