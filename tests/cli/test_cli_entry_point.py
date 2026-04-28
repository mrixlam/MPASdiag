#!/usr/bin/env python3
"""
MPASdiag Test Suite: CLI Entry Point and Unified CLI Tests

This module contains unit tests for the command-line interface (CLI) entry point and the `MPASUnifiedCLI` class defined in `cli_unified.py`. The tests validate that the package exposes a callable `main` function, that it correctly delegates to the unified CLI implementation, and that the `MPASUnifiedCLI` class provides the expected parser construction and subcommand behavior. These tests help ensure that the CLI remains functional and consistent as the codebase evolves.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load standard libraries
import pytest


class TestUnifiedCLI:
    """ Tests for the `MPASUnifiedCLI` class and its associated parser construction in `mpasdiag.processing.cli_unified`. """


    def test_create_main_parser_returns_parser(self: 'TestUnifiedCLI') -> None:
        """
        This test validates that the `create_main_parser` method of `MPASUnifiedCLI` returns an instance of `argparse.ArgumentParser` with the expected program name. This ensures that the CLI parser is constructed correctly and is ready to handle command-line arguments.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()

        assert parser is not None
        assert parser.prog == 'mpasdiag'

    def test_parser_has_subcommands(self: 'TestUnifiedCLI') -> None:
        """
        This test checks that the main parser created by `MPASUnifiedCLI` includes the expected analysis subcommands such as `precipitation`, `surface`, `wind`, `cross`, and `overlay`. By inspecting the help text of the parser, this test ensures that the subcommand registration is intact and that users will see these options when invoking the CLI.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()

        help_text = parser.format_help()
        assert 'precipitation' in help_text
        assert 'surface' in help_text
        assert 'wind' in help_text
        assert 'cross' in help_text
        assert 'overlay' in help_text

    def test_precipitation_command_parser(self: 'TestUnifiedCLI') -> None:
        """
        This test verifies that the `precipitation` subcommand can be parsed with the expected arguments. By providing a representative argument list to the parser, this test checks that the resulting namespace contains the correct `analysis_command` and other common options such as `grid_file` and `data_dir`. This ensures that the precipitation command is properly registered and that its arguments are correctly handled by the parser.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()

        args = parser.parse_args([
            'precipitation',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--output-dir', 'output/',
            '--time-index', '0'
        ])

        assert args.analysis_command == 'precipitation'
        assert args.grid_file == 'test.nc'
        assert args.data_dir == 'data/'

    def test_surface_command_parser(self: 'TestUnifiedCLI') -> None:
        """
        This test verifies that the `surface` subcommand can be parsed with the expected arguments. By providing a representative argument list to the parser, this test checks that the resulting namespace contains the correct `analysis_command` and other common options such as `variable` and `time_index`. This ensures that the surface command is properly registered and that its arguments are correctly handled by the parser.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()

        args = parser.parse_args([
            'surface',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--variable', 'temperature',
            '--time-index', '0'
        ])

        assert args.analysis_command == 'surface'
        assert args.variable == 'temperature'

    def test_wind_command_parser(self: 'TestUnifiedCLI') -> None:
        """
        This test verifies that the `wind` subcommand can be parsed with the expected arguments, including wind-specific options like `--wind-plot-type`. By providing a representative argument list to the parser, this test checks that the resulting namespace contains the correct `analysis_command` and that the `wind_plot_type` option is correctly parsed. This ensures that the wind command is properly registered and that its unique arguments are handled by the parser.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()

        args = parser.parse_args([
            'wind',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--wind-plot-type', 'barbs',
            '--time-index', '0'
        ])

        assert args.analysis_command == 'wind'
        assert args.wind_plot_type == 'barbs'

    def test_cross_command_parser(self: 'TestUnifiedCLI') -> None:
        """
        This test verifies that the `cross` (cross-section) subcommand can be parsed with the expected arguments. By providing a representative argument list to the parser, this test checks that the resulting namespace contains the correct `analysis_command` and other common options such as `variable`, `start_lat`, `start_lon`, `end_lat`, and `end_lon`. This ensures that the cross command is properly registered and that its arguments are correctly handled by the parser.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()

        args = parser.parse_args([
            'cross',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--variable', 'temperature',
            '--time-index', '0',
            '--start-lat', '30',
            '--start-lon', '-100',
            '--end-lat', '40',
            '--end-lon', '-90'
        ])

        assert args.analysis_command == 'cross'
        assert args.variable == 'temperature'

    def test_overlay_command_parser(self: 'TestUnifiedCLI') -> None:
        """
        This test verifies that the `overlay` subcommand can be parsed with the expected arguments. By providing a representative argument list to the parser, this test checks that the resulting namespace contains the correct `analysis_command` and that the `overlay_type` option is correctly parsed. This ensures that the overlay command is properly registered and that its unique arguments are handled by the parser.

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
            '--time-index', '0',
            '--overlay-type', 'precip_wind'
        ])

        assert args.analysis_command == 'overlay'

    def test_verbose_flag_parsing(self: 'TestUnifiedCLI') -> None:
        """
        This test verifies that the global `--verbose` flag is correctly parsed and stored on the resulting namespace. By invoking the parser with `--verbose` and a valid subcommand, this test checks that the `verbose` attribute on the parsed arguments is set to `True`, ensuring that global flags are applied consistently across all subcommands.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()

        args = parser.parse_args([
            '--verbose',
            'surface',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--variable', 'temp',
            '--time-index', '0'
        ])

        assert args.verbose is True

    def test_workers_argument_parsing(self: 'TestUnifiedCLI') -> None:
        """
        This test verifies that the `--workers` argument is correctly parsed and stored as an integer on the resulting namespace. By invoking the parser with `--workers` and a valid subcommand, this test checks that the `workers` attribute on the parsed arguments is set to the expected integer value, ensuring that this common option is handled properly by the parser.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()

        args = parser.parse_args([
            'surface',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--variable', 'temp',
            '--workers', '8'
        ])

        assert args.workers == pytest.approx(8)

    def test_invalid_subcommand_raises_error(self: 'TestUnifiedCLI') -> None:
        """
        This test ensures that providing an invalid subcommand to the parser results in a `SystemExit` exception, which is the expected behavior when argument parsing fails. By invoking the parser with an unrecognized command, this test confirms that the CLI correctly handles user errors and provides appropriate feedback.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(['invalid_command'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
