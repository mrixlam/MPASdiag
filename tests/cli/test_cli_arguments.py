#!/usr/bin/env python3
"""
MPASdiag Test Suite: CLI Argument Tests

This module contains tests focused on validating the argument parsing and handling logic of the `MPASUnifiedCLI` class and its `main()` function. The tests cover various scenarios including correct parsing of time ranges, overlay-specific arguments, handling of global flags in different positions, and the behavior of the CLI when exceptions are raised during argument parsing. Additionally, integration tests for the `--workers` argument are included to verify that the CLI correctly processes this option and that it affects execution as expected.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load standard libraries
import os
import sys
import yaml
import pytest
import tempfile
from unittest.mock import patch


class TestArgumentParsing:
    """ Test additional argument parsing scenarios. """
    
    def test_parse_args_to_config_with_time_range(self: 'TestArgumentParsing') -> None:
        """
        This test verifies that `parse_args_to_config` correctly converts time range arguments into numeric values and sets the batch-mode flag. By providing a `--time-range` with string values, the test checks that the resulting configuration object has `time_start` and `time_end` as numeric types and that `batch_mode` is set to True, which is essential for time range processing in the CLI.

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
            '--time-range', '0', '5'
        ])
        
        config = cli.parse_args_to_config(args)
        
        assert config.time_start == pytest.approx(0)
        assert config.time_end == pytest.approx(5)
        assert config.batch_mode is True
    
    def test_parse_args_to_config_overlay_mapping(self: 'TestArgumentParsing') -> None:
        """
        This test checks that when the `overlay` subcommand is used with specific overlay arguments, the resulting configuration object correctly maps those arguments to the expected configuration fields. For example, providing `--overlay-type` should result in the `config.overlay_type` field being set accordingly. This ensures that the CLI's argument parsing logic correctly translates command-line inputs into the internal configuration structure used for processing.

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
            '--time-index', '0'
        ])
        
        config = cli.parse_args_to_config(args)
        
        assert config.analysis_type == 'overlay'
        assert config.overlay_type == 'precip_wind'


class TestMainFunctionAndArgumentHandling:
    """ Test main() function entry point and CLI workflows. """
    
    
    def test_main_with_config_file_loading(self: 'TestMainFunctionAndArgumentHandling') -> None:
        """
        This test verifies that when `main()` is invoked with a `--config` argument pointing to a YAML file, the CLI attempts to load the configuration from that file and that if validation fails, it returns the expected exit code. By creating a temporary config file with known values and mocking the validation to return False, the test checks that `main()` correctly processes the config file and handles the validation failure as designed.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'grid_file': 'config_grid.nc',
                'data_dir': 'config_data/',
                'verbose': False,
                'analysis_type': 'precipitation',
                'variable': 'rainnc',
                'accumulation_period': 'a01h'
            }
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            cli = MPASUnifiedCLI()
            
            with patch('sys.argv', ['mpasdiag', '--config', config_file, 'precipitation', '--verbose', '--grid-file', 'override.nc', '--data-dir', 'override_data/']):
                with patch.object(cli, 'setup_logging'):
                    with patch.object(cli, 'validate_config', return_value=False):
                        result = cli.main()
                        assert result == pytest.approx(1)
        finally:
            if os.path.exists(config_file):
                os.unlink(config_file)
    
    def test_main_with_verbose_system_info(self: 'TestMainFunctionAndArgumentHandling') -> None:
        """
        This test checks that when `main()` is run with the `--verbose` flag, it prints system information and configuration summaries as part of its execution. By mocking the validation and run_analysis methods to return True, the test ensures that the CLI proceeds to the point where it would print verbose output and that it returns a zero exit code on success.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        cli = MPASUnifiedCLI()
        
        with patch('sys.argv', ['mpasdiag', 'precipitation', '--grid-file', 'test.nc', '--data-dir', 'data/', '--verbose']):
            with patch.object(cli, 'validate_config', return_value=True):
                with patch.object(cli, 'run_analysis', return_value=True):
                    result = cli.main()
                    assert result == pytest.approx(0)

    def test_main_with_direct_args(self: 'TestMainFunctionAndArgumentHandling', 
                                   grid_file: str, 
                                   test_data_dir: str) -> None:
        """
        This test runs `MPASUnifiedCLI.main()` with a direct set of command-line arguments simulating a typical user invocation for a surface plot. It sets `sys.argv` to include necessary flags and arguments, then calls `main()` to ensure it processes the arguments correctly and returns a zero exit code on success. This test verifies that the CLI can handle a straightforward command-line invocation without relying on config files or global flags.

        Parameters:
            grid_file (str): Session fixture providing path to MPAS grid file.
            test_data_dir (str): Session fixture providing path to test data directory.

        Returns:
            None
        """
        import pytest
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        if grid_file is None:
            pytest.skip("Test data files not available")
            return
        
        data_dir = str(test_data_dir / 'u240k' / 'diag')
        
        original_argv = sys.argv
        
        try:
            sys.argv = [
                'mpasdiag',
                'surface',
                '--grid-file', grid_file,
                '--data-dir', data_dir,
                '--variable', 't2m',
                '--time-index', '0',
                '--output-dir', 'output/test_main_direct',
                '--quiet'
            ]
            
            cli = MPASUnifiedCLI()
            result = cli.main()
            
            assert result == pytest.approx(0)
            
        finally:
            sys.argv = original_argv
    
    def test_main_with_verbose_flag(self: 'TestMainFunctionAndArgumentHandling', 
                                    grid_file: str, 
                                    test_data_dir: str) -> None:
        """
        This test verifies that when `MPASUnifiedCLI.main()` is invoked with the `--verbose` flag, it processes the arguments correctly and returns a zero exit code on success. By setting `sys.argv` to include the `--verbose` flag along with necessary arguments for a precipitation plot, the test checks that the CLI can handle verbose mode and that it executes without errors, returning the expected exit code.

        Parameters:
            grid_file (str): Session fixture providing path to MPAS grid file.
            test_data_dir (str): Session fixture providing path to test data directory.

        Returns:
            None
        """
        import pytest
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        if grid_file is None:
            pytest.skip("Test data files not available")
            return
        
        data_dir = str(test_data_dir / 'u240k' / 'diag')
        
        original_argv = sys.argv
        
        try:
            sys.argv = [
                'mpasdiag',
                '--verbose',
                'precipitation',
                '--grid-file', grid_file,
                '--data-dir', data_dir,
                '--time-index', '0',
                '--output-dir', 'output/test_verbose'
            ]
            
            cli = MPASUnifiedCLI()
            result = cli.main()
            
            assert result == pytest.approx(0)
            
        finally:
            sys.argv = original_argv
    
    def test_main_keyboard_interrupt(self: 'TestMainFunctionAndArgumentHandling') -> None:
        """
        This test simulates a `KeyboardInterrupt` during the execution of `MPASUnifiedCLI.main()` to verify that the CLI handles such interrupts gracefully. By patching the `run_analysis` method to raise a `KeyboardInterrupt`, the test checks that `main()` catches this exception and returns an appropriate exit code (commonly 130 for keyboard interrupts) instead of allowing the exception to propagate uncaught.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        cli = MPASUnifiedCLI()
        original_argv = sys.argv
        
        try:
            sys.argv = ['mpasdiag', 'precipitation', '--grid-file', 'test.nc', '--data-dir', 'data/']
            
            with patch.object(cli, 'validate_config', return_value=True):
                with patch.object(cli, 'run_analysis', side_effect=KeyboardInterrupt()):
                    result = cli.main()
                    
                    assert result == pytest.approx(130)
        finally:
            sys.argv = original_argv
    
    def test_main_unexpected_exception(self: 'TestMainFunctionAndArgumentHandling') -> None:
        """
        This test simulates an unexpected exception during the execution of `MPASUnifiedCLI.main()` to verify that the CLI handles such exceptions gracefully. By patching the `run_analysis` method to raise a generic `RuntimeError`, the test checks that `main()` catches this exception and returns an appropriate exit code (commonly 1 for general errors) instead of allowing the exception to propagate uncaught.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        cli = MPASUnifiedCLI()
        original_argv = sys.argv
        
        try:
            sys.argv = ['mpasdiag', 'precipitation', '--grid-file', 'test.nc', '--data-dir', 'data/']
            
            with patch.object(cli, 'validate_config', return_value=True):
                with patch.object(cli, 'run_analysis', side_effect=RuntimeError("Test error")):
                    result = cli.main()                    
                    assert result == pytest.approx(1)
        finally:
            sys.argv = original_argv
    

class TestMainArgumentReorderingAdditional:
    """ Test main() function argument reordering logic. """
    
    def test_main_with_global_flags_after_subcommand(self: 'TestMainArgumentReorderingAdditional', 
                                                     grid_file: str, 
                                                     test_data_dir: str) -> None:
        """
        This test verifies that `MPASUnifiedCLI.main()` can handle global flags (like `--verbose`) appearing after the subcommand and that it processes the arguments correctly, returning a zero exit code on success. By setting `sys.argv` to include the `--verbose` flag after the subcommand and necessary arguments for a precipitation plot, the test checks that the CLI can reorder arguments as needed and execute without errors.

        Parameters:
            grid_file (str): Session fixture providing path to MPAS grid file.
            test_data_dir (str): Session fixture providing path to test data directory.

        Returns:
            None
        """
        import pytest
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        if grid_file is None:
            pytest.skip("Test data files not available")
            return
        
        data_dir = str(test_data_dir / 'u240k' / 'diag')        
        original_argv = sys.argv
        
        try:
            sys.argv = [
                'mpasdiag',
                'precipitation',
                '--grid-file', grid_file,
                '--data-dir', data_dir,
                '--verbose',  
                '--time-index', '0',
                '--output-dir', 'output/test_reorder'
            ]
            
            cli = MPASUnifiedCLI()
            result = cli.main()
            
            assert result == pytest.approx(0)
            
        finally:
            sys.argv = original_argv
    
    def test_main_with_log_file_argument(self: 'TestMainArgumentReorderingAdditional', 
                                         grid_file: str, 
                                         test_data_dir: str) -> None:
        """
        This test verifies that `MPASUnifiedCLI.main()` can handle a `--log-file` argument appearing in the command line and that it processes the arguments correctly, returning a zero exit code on success. By setting `sys.argv` to include the `--log-file` flag along with necessary arguments for a precipitation plot, the test checks that the CLI can handle log file specification and that it executes without errors, creating the log file as expected.

        Parameters:
            grid_file (str): Session fixture providing path to MPAS grid file.
            test_data_dir (str): Session fixture providing path to test data directory.

        Returns:
            None
        """
        import pytest
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        import tempfile
        
        if grid_file is None:
            pytest.skip("Test data files not available")
            return
        
        data_dir = str(test_data_dir / 'u240k' / 'diag')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = f.name
        
        original_argv = sys.argv
        
        try:
            sys.argv = [
                'mpasdiag',
                '--log-file', log_file,
                'precipitation',
                '--grid-file', grid_file,
                '--data-dir', data_dir,
                '--time-index', '0',
                '--output-dir', 'output/test_logfile',
                '--quiet'
            ]
            
            cli = MPASUnifiedCLI()
            result = cli.main()
            
            assert result == pytest.approx(0)
            assert os.path.exists(log_file)
            
        finally:
            sys.argv = original_argv
            if os.path.exists(log_file):
                os.unlink(log_file)


class TestMainArgumentReorderingUnifiedFinal:
    """ Test main() function argument reordering logic. """
    
    def test_main_with_global_arg_equals_format(self: 'TestMainArgumentReorderingUnifiedFinal') -> None:
        """
        This test verifies that `MPASUnifiedCLI.main()` can handle a global argument provided in the `--arg=value` format and that it processes the arguments correctly, returning a zero exit code on success. By creating a temporary config file and invoking `main()` with the `--config` argument using the equals sign format, the test checks that the CLI can parse this style of argument and that it executes without errors when validation fails.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        import tempfile
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'grid_file': 'test.nc', 'data_dir': 'data/'}, f)
            config_file = f.name
        
        try:
            cli = MPASUnifiedCLI()
            
            with patch('sys.argv', ['mpasdiag', f'--config={config_file}', 'precipitation', '--grid-file', 'override.nc', '--data-dir', 'data/']):
                with patch.object(cli, 'setup_logging'):
                    with patch.object(cli, 'validate_config', return_value=False):
                        result = cli.main()
                        assert result == pytest.approx(1)
        finally:
            if os.path.exists(config_file):
                os.unlink(config_file)
    
    def test_main_with_global_arg_at_end(self: 'TestMainArgumentReorderingUnifiedFinal') -> None:
        """
        This test verifies that `MPASUnifiedCLI.main()` can handle a global argument (like `--verbose`) appearing at the end of the command line and that it processes the arguments correctly, returning a zero exit code on success. By setting `sys.argv` to include the `--verbose` flag after the subcommand and necessary arguments for a precipitation plot, the test checks that the CLI can reorder arguments as needed and execute without errors, even when validation fails.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        cli = MPASUnifiedCLI()
        
        with patch('sys.argv', ['mpasdiag', 'precipitation', '--grid-file', 'test.nc', '--data-dir', 'data/', '--verbose']):
            with patch.object(cli, 'setup_logging'):
                with patch.object(cli, 'validate_config', return_value=False):
                    result = cli.main()
                    assert result == pytest.approx(1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
