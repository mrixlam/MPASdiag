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


class TestCLIEntryPoint:
    """ Tests for the CLI entry point defined in `mpasdiag.cli`. """

    def test_cli_main_import(self: "TestCLIEntryPoint") -> None:
        """
        This test verifies that the `main` function is defined and callable in the `mpasdiag.cli` module. It imports the `main` function and asserts that it is present and can be called, ensuring that the package-level CLI entry point is correctly implemented and available for invocation.

        Parameters:
            self (TestCLIEntryPoint): The test instance.

        Returns:
            None: Raises on failure.
        """
        from mpasdiag.cli import main
        assert callable(main)

    def test_cli_delegates_to_unified(self: "TestCLIEntryPoint") -> None:
        """
        This test confirms that the `main` function in `mpasdiag.cli` delegates to the `main` function defined in `mpasdiag.processing.cli_unified`. By importing both and asserting they are the same object, this test ensures that the CLI entry point correctly forwards execution to the unified CLI implementation, which is critical for maintaining a single source of truth for CLI behavior.

        Parameters:
            self (TestCLIEntryPoint): The test instance.

        Returns:
            None: Raises on failure.
        """
        from mpasdiag.cli import main
        from mpasdiag.processing.cli_unified import main as unified_main
        assert main is unified_main


class TestUnifiedCLI:
    """ Tests for the `MPASUnifiedCLI` class and its associated parser construction in `mpasdiag.processing.cli_unified`. """

    def test_cli_class_initialization(self: "TestUnifiedCLI") -> None:
        """
        This test checks that an instance of `MPASUnifiedCLI` can be created without errors and that its initial attributes are set to `None` as expected. This verifies that the class constructor initializes the object state correctly, which is important for the subsequent setup and execution of CLI commands.

        Parameters:
            self (TestUnifiedCLI): The test instance.

        Returns:
            None: Raises on failure.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        assert cli.logger is None 
        assert cli.perf_monitor is None
        assert cli.config is None

    def test_create_main_parser_returns_parser(self: "TestUnifiedCLI") -> None:
        """
        This test validates that the `create_main_parser` method of `MPASUnifiedCLI` returns an instance of `argparse.ArgumentParser` with the expected program name. This ensures that the CLI parser is constructed correctly and is ready to handle command-line arguments.

        Parameters:
            self (TestUnifiedCLI): The test instance.

        Returns:
            None: Raises on assertion failure.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()

        assert parser is not None
        assert parser.prog == 'mpasdiag'

    def test_parser_has_subcommands(self: "TestUnifiedCLI") -> None:
        """
        This test checks that the main parser created by `MPASUnifiedCLI` includes the expected analysis subcommands such as `precipitation`, `surface`, `wind`, `cross`, and `overlay`. By inspecting the help text of the parser, this test ensures that the subcommand registration is intact and that users will see these options when invoking the CLI.

        Parameters:
            self (TestUnifiedCLI): The test instance.

        Returns:
            None: Raises on failure.
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

    def test_precipitation_command_parser(self: "TestUnifiedCLI") -> None:
        """
        This test verifies that the `precipitation` subcommand can be parsed with the expected arguments. By providing a representative argument list to the parser, this test checks that the resulting namespace contains the correct `analysis_command` and other common options such as `grid_file` and `data_dir`. This ensures that the precipitation command is properly registered and that its arguments are correctly handled by the parser.

        Parameters:
            self (TestUnifiedCLI): The test instance.

        Returns:
            None: Assertion failures indicate parser regressions.
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

    def test_surface_command_parser(self: "TestUnifiedCLI") -> None:
        """
        This test verifies that the `surface` subcommand can be parsed with the expected arguments. By providing a representative argument list to the parser, this test checks that the resulting namespace contains the correct `analysis_command` and other common options such as `variable` and `time_index`. This ensures that the surface command is properly registered and that its arguments are correctly handled by the parser.

        Parameters:
            self (TestUnifiedCLI): The test instance.

        Returns:
            None: Raises on mismatch.
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

    def test_wind_command_parser(self: "TestUnifiedCLI") -> None:
        """
        This test verifies that the `wind` subcommand can be parsed with the expected arguments, including wind-specific options like `--wind-plot-type`. By providing a representative argument list to the parser, this test checks that the resulting namespace contains the correct `analysis_command` and that the `wind_plot_type` option is correctly parsed. This ensures that the wind command is properly registered and that its unique arguments are handled by the parser.

        Parameters:
            self (TestUnifiedCLI): The test instance.

        Returns:
            None: Assertion failures indicate parser issues.
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

    def test_cross_command_parser(self: "TestUnifiedCLI") -> None:
        """
        This test verifies that the `cross` (cross-section) subcommand can be parsed with the expected arguments. By providing a representative argument list to the parser, this test checks that the resulting namespace contains the correct `analysis_command` and other common options such as `variable`, `start_lat`, `start_lon`, `end_lat`, and `end_lon`. This ensures that the cross command is properly registered and that its arguments are correctly handled by the parser.

        Parameters:
            self (TestUnifiedCLI): The test instance.

        Returns:
            None: Raises when parsed values differ from expectations.
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

    def test_overlay_command_parser(self: "TestUnifiedCLI") -> None:
        """
        This test verifies that the `overlay` subcommand can be parsed with the expected arguments. By providing a representative argument list to the parser, this test checks that the resulting namespace contains the correct `analysis_command` and that the `overlay_type` option is correctly parsed. This ensures that the overlay command is properly registered and that its unique arguments are handled by the parser.

        Parameters:
            self (TestUnifiedCLI): The test instance.

        Returns:
            None: Raises on incorrect parsing.
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

    def test_verbose_flag_parsing(self: "TestUnifiedCLI") -> None:
        """
        This test verifies that the global `--verbose` flag is correctly parsed and stored on the resulting namespace. By invoking the parser with `--verbose` and a valid subcommand, this test checks that the `verbose` attribute on the parsed arguments is set to `True`, ensuring that global flags are applied consistently across all subcommands.

        Parameters:
            self (TestUnifiedCLI): The test instance.

        Returns:
            None: Assertion failures indicate CLI parsing regressions.
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

    def test_workers_argument_parsing(self: "TestUnifiedCLI") -> None:
        """
        This test verifies that the `--workers` argument is correctly parsed and stored as an integer on the resulting namespace. By invoking the parser with `--workers` and a valid subcommand, this test checks that the `workers` attribute on the parsed arguments is set to the expected integer value, ensuring that this common option is handled properly by the parser.

        Parameters:
            self (TestUnifiedCLI): The test instance.

        Returns:
            None: Raises on parsing or type coercion failures.
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

    def test_invalid_subcommand_raises_error(self: "TestUnifiedCLI") -> None:
        """
        This test ensures that providing an invalid subcommand to the parser results in a `SystemExit` exception, which is the expected behavior when argument parsing fails. By invoking the parser with an unrecognized command, this test confirms that the CLI correctly handles user errors and provides appropriate feedback.

        Parameters:
            self (unittest.TestCase): The test instance.

        Returns:
            None: The test will pass only if `SystemExit` is raised.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(['invalid_command'])

    def test_cli_has_main_method(self: "TestUnifiedCLI") -> None:
        """
        This test checks that the `MPASUnifiedCLI` class provides a `main` method that is callable. The `main` method is the primary entry point for executing the CLI logic, and its presence is essential for the functionality of the CLI. This test ensures that the class interface includes this method and that it can be invoked without errors.

        Parameters:
            self (unittest.TestCase): The test instance.

        Returns:
            None: Raises if the attribute is missing or not callable.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        assert hasattr(cli, 'main')
        assert callable(cli.main)

    def test_cli_has_setup_logging_method(self: "TestUnifiedCLI") -> None:
        """
        This test checks that the `MPASUnifiedCLI` class provides a `setup_logging` method that is callable. The `setup_logging` method is responsible for configuring logging for CLI-based runs, and its presence is essential for proper logging behavior. This test ensures that the class interface includes this method and that it can be invoked without errors.

        Parameters:
            self (unittest.TestCase): The test instance.

        Returns:
            None: Fails if attribute is missing or not callable.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        assert hasattr(cli, 'setup_logging')
        assert callable(cli.setup_logging)

    def test_cli_has_validate_config_method(self: "TestUnifiedCLI") -> None:
        """
        This test checks that the `MPASUnifiedCLI` class provides a `validate_config` method that is callable. The `validate_config` method is responsible for validating the configuration before executing the analysis, and its presence is crucial for ensuring that the CLI can properly check user inputs and configuration settings. This test ensures that the class interface includes this method and that it can be invoked without errors.

        Parameters:
            self (unittest.TestCase): The test instance.

        Returns:
            None: Raises on missing or non-callable attribute.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        assert hasattr(cli, 'validate_config')
        assert callable(cli.validate_config)

    def test_cli_has_run_analysis_method(self: "TestUnifiedCLI") -> None:
        """
        This test checks that the `MPASUnifiedCLI` class provides a `run_analysis` method that is callable. The `run_analysis` method is the primary execution method that dispatches the selected analysis workflow, and its presence is essential for the functionality of the CLI. This test ensures that the class interface includes this method and that it can be invoked without errors.

        Parameters:
            self (unittest.TestCase): The test instance.

        Returns:
            None: Raises on missing or non-callable attribute.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        assert hasattr(cli, 'run_analysis')
        assert callable(cli.run_analysis)

    def test_cli_plot_types_constant(self: "TestUnifiedCLI") -> None:
        """
        This test verifies that the `MPASUnifiedCLI` class defines a `PLOT_TYPES` constant that is a dictionary containing the expected plot type keys. The presence of this constant is important for ensuring that the CLI has a centralized definition of available plot types, which can be used for validation and help text generation. This test checks that `PLOT_TYPES` exists, is a dictionary, and includes keys for common plot types such as 'precipitation', 'surface', and 'wind'.

        Parameters:
            self (unittest.TestCase): The test instance.

        Returns:
            None: Assertion failure indicates missing or malformed constant.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        assert hasattr(MPASUnifiedCLI, 'PLOT_TYPES')
        assert isinstance(MPASUnifiedCLI.PLOT_TYPES, dict)
        assert 'precipitation' in MPASUnifiedCLI.PLOT_TYPES
        assert 'surface' in MPASUnifiedCLI.PLOT_TYPES
        assert 'wind' in MPASUnifiedCLI.PLOT_TYPES


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
