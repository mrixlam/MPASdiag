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
from unittest.mock import patch

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
def parser(cli) -> argparse.Namespace:
    """
    This fixture creates and returns the argument parser from the MPASUnifiedCLI instance. It allows tests to access the parser directly, enabling them to simulate command-line inputs and verify that the parser correctly accepts and parses the 'sounding' subcommand and its associated options.

    Parameters:
        cli (MPASUnifiedCLI): The CLI instance provided by the cli fixture.

    Returns:
        argparse.Namespace: The argument parser created by the MPASUnifiedCLI instance.
    """
    return cli.create_main_parser()


class TestSoundingParser:
    """ Verify that the CLI argument parser correctly accepts and parses the 'sounding' subcommand and its associated options. """

    def test_sounding_subcommand_accepted(self: "TestSoundingParser", parser: "argparse.Namespace") -> None:
        """ 
        This test verifies that the 'sounding' subcommand is correctly registered in the argument parser. It simulates a command-line input that includes the 'sounding' subcommand along with required options such as grid file, data directory, longitude, latitude, and time index. The test asserts that the parsed arguments include the correct analysis command, confirming that the subcommand is recognized by the parser.

        Parameters:
            parser (argparse.Namespace): The argument parser provided by the parser fixture.

        Returns:
            None: The test will pass if the assertion is true, and fail if it is not.
        """
        args = parser.parse_args([
            'sounding',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--lon', '-100.0',
            '--lat', '35.0',
            '--time-index', '0',
        ])

        assert args.analysis_command == 'sounding'

    def test_skewt_alias_accepted(self: "TestSoundingParser", parser: "argparse.Namespace") -> None:
        """ 
        This test verifies that the 'skewt' alias is correctly registered in the argument parser. It simulates a command-line input that includes the 'skewt' subcommand along with required options such as grid file, data directory, longitude, latitude, and time index. The test asserts that the parsed arguments include the correct analysis command, confirming that the alias is recognized by the parser.

        Parameters:
            parser (argparse.Namespace): The argument parser provided by the parser fixture.

        Returns:
            None: The test will pass if the assertion is true, and fail if it is not.
        """
        args = parser.parse_args([
            'skewt',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--lon', '-100.0',
            '--lat', '35.0',
            '--time-index', '0',
        ])

        assert args.analysis_command == 'skewt'

    def test_profile_alias_accepted(self: "TestSoundingParser", parser: "argparse.Namespace") -> None:
        """ 
        This test verifies that the 'profile' alias is correctly registered in the argument parser. It simulates a command-line input that includes the 'profile' subcommand along with required options such as grid file, data directory, longitude, latitude, and time index. The test asserts that the parsed arguments include the correct analysis command, confirming that the alias is recognized by the parser.

        Parameters:
            parser (argparse.Namespace): The argument parser provided by the parser fixture.

        Returns:
            None: The test will pass if the assertion is true, and fail if it is not.
        """
        args = parser.parse_args([
            'profile',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--lon', '-100.0',
            '--lat', '35.0',
            '--time-index', '0',
        ])

        assert args.analysis_command == 'profile'

    def test_lon_lat_required(self: "TestSoundingParser", parser: "argparse.Namespace") -> None:
        """
        This test verifies that the parser raises a SystemExit error when the required --lon and --lat arguments are missing. It simulates a command-line input that includes the 'sounding' subcommand along with required options such as grid file, data directory, and time index, but omits the longitude and latitude arguments.

        Parameters:
            parser (argparse.Namespace): The argument parser provided by the parser fixture.

        Returns:
            None: The test will pass if the SystemExit error is raised, and fail if it is not.
        """
        import io
        import sys

        captured = io.StringIO()
        sys_stderr = sys.stderr
        sys.stderr = captured

        try:
            with pytest.raises(SystemExit):
                parser.parse_args([
                    'sounding',
                    '--grid-file', 'test.nc',
                    '--data-dir', 'data/',
                    '--time-index', '0',
                    # missing --lon and --lat
                ])
            output = captured.getvalue()
        finally:
            sys.stderr = sys_stderr

        assert "the following arguments are required: --lon, --lat" in output

    def test_lon_parsed_as_float(self: "TestSoundingParser", parser: "argparse.Namespace") -> None:
        """ 
        This test verifies that the --lon and --lat arguments are correctly parsed as float values. It simulates a command-line input that includes the 'sounding' subcommand along with required options such as grid file, data directory, longitude, latitude, and time index. The test asserts that the parsed longitude and latitude values are of type float.

        Parameters:
            parser (argparse.Namespace): The argument parser provided by the parser fixture.

        Returns:
            None: The test will pass if the assertion is true, and fail if it is not.
        """
        args = parser.parse_args([
            'sounding',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--lon', '-97.5',
            '--lat', '36.25',
            '--time-index', '0',
        ])

        assert args.lon == pytest.approx(-97.5)
        assert args.lat == pytest.approx(36.25)

    def test_show_indices_default_false(self: "TestSoundingParser", parser: "argparse.Namespace") -> None:
        """
        This test verifies that the --show-indices flag is correctly parsed and defaults to False when not provided. It simulates a command-line input that includes the 'sounding' subcommand along with required options such as grid file, data directory, longitude, latitude, and time index, but omits the --show-indices flag.

        Parameters:
            parser (argparse.Namespace): The argument parser provided by the parser fixture.

        Returns:
            None: The test will pass if the assertion is true, and fail if it is not.
        """
        args = parser.parse_args([
            'sounding',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--lon', '0', '--lat', '0',
            '--time-index', '0',
        ])

        assert args.show_indices is False

    def test_show_indices_flag(self: "TestSoundingParser", parser: "argparse.Namespace") -> None:
        """ 
        This test verifies that the --show-indices flag is correctly parsed and set to True when provided. It simulates a command-line input that includes the 'sounding' subcommand along with required options such as grid file, data directory, longitude, latitude, time index, and the --show-indices flag. The test asserts that the parsed show_indices value is True.

        Parameters:
            parser (argparse.Namespace): The argument parser provided by the parser fixture.

        Returns:
            None: The test will pass if the assertion is true, and fail if it is not.
        """
        args = parser.parse_args([
            'sounding',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--lon', '0', '--lat', '0',
            '--time-index', '0',
            '--show-indices',
        ])

        assert args.show_indices is True

    def test_show_parcel_flag(self: "TestSoundingParser", parser: "argparse.Namespace") -> None:
        """ 
        This test verifies that the --show-parcel flag is correctly parsed and set to True when provided. It simulates a command-line input that includes the 'sounding' subcommand along with required options such as grid file, data directory, longitude, latitude, time index, and the --show-parcel flag. The test asserts that the parsed show_parcel value is True.

        Parameters:
            parser (argparse.Namespace): The argument parser provided by the parser fixture.

        Returns:
            None: The test will pass if the assertion is true, and fail if it is not.
        """
        args = parser.parse_args([
            'sounding',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--lon', '0', '--lat', '0',
            '--time-index', '0',
            '--show-parcel',
        ])

        assert args.show_parcel is True


class TestSoundingArgMapping:
    """ Confirm that the CLI parsing logic correctly maps parsed arguments from the 'sounding' subcommand to the appropriate fields in the configuration object used for analysis. """

    def test_parse_args_to_config_maps_fields(self: "TestSoundingArgMapping", cli: "MPASUnifiedCLI", parser: "argparse.Namespace") -> None:
        """ 
        This test verifies that the parse_args_to_config method of the CLI correctly maps the parsed arguments from a 'sounding' subcommand input to the corresponding fields in the configuration object. It simulates a command-line input that includes the 'sounding' subcommand along with various options, parses the arguments, and then asserts that the resulting configuration object has the expected values for analysis type, longitude, latitude, and flags for showing indices and parcel information.

        Parameters:
            cli (MPASUnifiedCLI): The CLI instance provided by the cli fixture.
            parser (argparse.Namespace): The argument parser provided by the parser fixture.

        Returns:
            None: The test will pass if all assertions are true, and fail if any assertion is not true.
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

    def test_parse_args_to_config_skewt_alias(self: "TestSoundingArgMapping", cli: "MPASUnifiedCLI", parser: "argparse.Namespace") -> None:
        """ 
        This test verifies that the parse_args_to_config method correctly maps the 'skewt' alias to the 'sounding' analysis type in the configuration object. It simulates a command-line input that includes the 'skewt' subcommand along with required options, parses the arguments, and then asserts that the resulting configuration object has 'skewt' as the analysis type, confirming that the alias is preserved in the configuration.

        Parameters:
            cli (MPASUnifiedCLI): The CLI instance provided by the cli fixture.
            parser (argparse.Namespace): The argument parser provided by the parser fixture.

        Returns:
            None: The test will pass if all assertions are true, and fail if any assertion is not true.
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

    def test_time_range_sets_batch_mode(self: "TestSoundingArgMapping", cli: "MPASUnifiedCLI", parser: "argparse.Namespace") -> None:
        """
        This test verifies that the parse_args_to_config method correctly sets the batch_mode flag when a time range is specified. It simulates a command-line input that includes the 'sounding' subcommand along with a time range, parses the arguments, and then asserts that the resulting configuration object has batch_mode set to True.

        Parameters:
            cli (MPASUnifiedCLI): The CLI instance provided by the cli fixture.
            parser (argparse.Namespace): The argument parser provided by the parser fixture.

        Returns:
            None: The test will pass if all assertions are true, and fail if any assertion is not true.
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


class TestSoundingDispatch:
    """ Verify that the CLI dispatch method correctly routes the 'sounding' analysis type to the appropriate analysis function without raising errors. """

    def test_sounding_in_analysis_map(self: "TestSoundingDispatch", cli: "MPASUnifiedCLI") -> None:
        """
        This test verifies that the _dispatch_analysis method of the CLI correctly routes the 'sounding' analysis type to the appropriate analysis function. It creates a configuration object with the analysis type set to 'sounding' and then calls the dispatch method. The test uses patching to mock the actual execution of the sounding analysis function, allowing it to confirm that the correct function is called without requiring actual data or execution.

        Parameters:
            cli (MPASUnifiedCLI): The CLI instance provided by the cli fixture.

        Returns:
            None: The test will pass if the correct analysis function is called without raising a KeyError, and fail if a KeyError is raised or if the wrong function is called.
        """
        from mpasdiag.processing.utils_config import MPASConfig

        config = MPASConfig(
            grid_file='test.nc',
            data_dir='data/',
            analysis_type='sounding',
            time_index=0,
            sounding_lon=-100.0,
            sounding_lat=35.0,
        )

        with patch.object(cli, '_run_sounding_analysis', return_value=True) as mock_run:
            result = cli._dispatch_analysis('sounding', config)
            mock_run.assert_called_once_with(config)
            assert result is True


if __name__ == "__main__":
    pytest.main([__file__])