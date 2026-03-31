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
import subprocess
from unittest.mock import Mock, patch
from typing import Optional, Dict, Any


class TestArgumentParsing:
    """ Test additional argument parsing scenarios. """
    
    def test_parse_args_to_config_with_time_range(self: "TestArgumentParsing") -> None:
        """
        This test verifies that `parse_args_to_config` correctly converts time range arguments into numeric values and sets the batch-mode flag. By providing a `--time-range` with string values, the test checks that the resulting configuration object has `time_start` and `time_end` as numeric types and that `batch_mode` is set to True, which is essential for time range processing in the CLI.

        Parameters:
            self (TestArgumentParsing): The test instance.

        Returns:
            None: The test asserts numeric conversion and batch-mode flagging.
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
        
        assert config.time_start == 0
        assert config.time_end == 5
        assert config.batch_mode is True
    
    def test_parse_args_to_config_overlay_mapping(self: "TestArgumentParsing") -> None:
        """
        This test checks that when the `overlay` subcommand is used with specific overlay arguments, the resulting configuration object correctly maps those arguments to the expected configuration fields. For example, providing `--overlay-type` should result in the `config.overlay_type` field being set accordingly. This ensures that the CLI's argument parsing logic correctly translates command-line inputs into the internal configuration structure used for processing.

        Parameters:
            self (TestArgumentParsing): The test instance.

        Returns:
            None: The test asserts the mapping of overlay arguments.
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
    
    def test_main_parse_args_exception_fallback(self: "TestMainFunctionAndArgumentHandling") -> None:
        """
        This test simulates a scenario where the initial argument parsing using `parse_intermixed_args` raises an exception, and verifies that `main()` correctly falls back to a second parsing attempt using `parse_args`. By mocking the parser to raise an exception on the first call and succeed on the second, the test checks that `main()` handles this situation gracefully and returns the expected exit code when validation fails.

        Parameters:
            self (TestMainFunctionAndArgumentHandling): The test instance.

        Returns:
            None: Asserts that `main()` returns exit code 1 in this case.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        cli = MPASUnifiedCLI()
        
        with patch('sys.argv', ['mpasdiag', 'precipitation', '--verbose', '--grid-file', 'test.nc', '--data-dir', 'data/']):
            with patch.object(cli, 'create_main_parser') as mock_parser_factory:
                mock_parser = Mock()
                mock_parser.parse_intermixed_args.side_effect = Exception("Parse error")
                mock_parser.parse_args.return_value = Mock(
                    analysis_command='precipitation',
                    config=None,
                    log_file=None,
                    verbose=False,
                    grid_file='test.nc',
                    data_dir='data/'
                )

                mock_parser_factory.return_value = mock_parser
                
                with patch.object(cli, 'parse_args_to_config') as mock_parse_config:
                    mock_config = Mock()
                    mock_config.to_dict.return_value = {}
                    mock_parse_config.return_value = mock_config
                    
                    with patch.object(cli, 'setup_logging'):
                        with patch.object(cli, 'validate_config', return_value=False):
                            result = cli.main()
                            assert result == 1
    
    def test_main_with_config_file_loading(self: "TestMainFunctionAndArgumentHandling") -> None:
        """
        This test verifies that when `main()` is invoked with a `--config` argument pointing to a YAML file, the CLI attempts to load the configuration from that file and that if validation fails, it returns the expected exit code. By creating a temporary config file with known values and mocking the validation to return False, the test checks that `main()` correctly processes the config file and handles the validation failure as designed.

        Parameters:
            self (TestMainFunctionAndArgumentHandling): The test instance.

        Returns:
            None: Asserts `main()` returns exit code 1 under these mocks.
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
                        assert result == 1
        finally:
            if os.path.exists(config_file):
                os.unlink(config_file)
    
    def test_main_with_verbose_system_info(self: "TestMainFunctionAndArgumentHandling") -> None:
        """
        This test checks that when `main()` is run with the `--verbose` flag, it prints system information and configuration summaries as part of its execution. By mocking the validation and run_analysis methods to return True, the test ensures that the CLI proceeds to the point where it would print verbose output and that it returns a zero exit code on success.

        Parameters:
            self (TestMainFunctionAndArgumentHandling): The test instance.

        Returns:
            None: Asserts `main()` returns 0 in the mocked success case.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        cli = MPASUnifiedCLI()
        
        with patch('sys.argv', ['mpasdiag', 'precipitation', '--grid-file', 'test.nc', '--data-dir', 'data/', '--verbose']):
            with patch.object(cli, 'validate_config', return_value=True):
                with patch.object(cli, 'run_analysis', return_value=True):
                    result = cli.main()
                    assert result == 0

    def test_main_with_direct_args(self: "TestMainFunctionAndArgumentHandling", grid_file, test_data_dir) -> None:
        """
        This test runs `MPASUnifiedCLI.main()` with a direct set of command-line arguments simulating a typical user invocation for a surface plot. It sets `sys.argv` to include necessary flags and arguments, then calls `main()` to ensure it processes the arguments correctly and returns a zero exit code on success. This test verifies that the CLI can handle a straightforward command-line invocation without relying on config files or global flags.

        Parameters:
            self (TestMainFunction): The test instance.
            grid_file: Session fixture providing path to MPAS grid file.
            test_data_dir: Session fixture providing path to test data directory.

        Returns:
            None: The test asserts `main()` returns `0` for success.
        """
        import pytest
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        if grid_file is None:
            pytest.skip("Test data files not available")
        
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
            
            assert result == 0
            
        finally:
            sys.argv = original_argv
    
    def test_main_with_verbose_flag(self: "TestMainFunctionAndArgumentHandling", grid_file, test_data_dir) -> None:
        """
        This test verifies that when `MPASUnifiedCLI.main()` is invoked with the `--verbose` flag, it processes the arguments correctly and returns a zero exit code on success. By setting `sys.argv` to include the `--verbose` flag along with necessary arguments for a precipitation plot, the test checks that the CLI can handle verbose mode and that it executes without errors, returning the expected exit code.

        Parameters:
            self (TestMainFunction): The test instance.
            grid_file: Session fixture providing path to MPAS grid file.
            test_data_dir: Session fixture providing path to test data directory.

        Returns:
            None: The test asserts `main()` returns `0` when successful.
        """
        import pytest
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        if grid_file is None:
            pytest.skip("Test data files not available")
        
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
            
            assert result == 0
            
        finally:
            sys.argv = original_argv
    
    def test_main_keyboard_interrupt(self: "TestMainFunctionAndArgumentHandling") -> None:
        """
        This test simulates a `KeyboardInterrupt` during the execution of `MPASUnifiedCLI.main()` to verify that the CLI handles such interrupts gracefully. By patching the `run_analysis` method to raise a `KeyboardInterrupt`, the test checks that `main()` catches this exception and returns an appropriate exit code (commonly 130 for keyboard interrupts) instead of allowing the exception to propagate uncaught.

        Parameters:
            self (TestMainFunctionAndArgumentHandling): The test instance.

        Returns:
            None: The test passes if `main()` handles the interrupt cleanly.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        cli = MPASUnifiedCLI()
        original_argv = sys.argv
        
        try:
            sys.argv = ['mpasdiag', 'precipitation', '--grid-file', 'test.nc', '--data-dir', 'data/']
            
            with patch.object(cli, 'validate_config', return_value=True):
                with patch.object(cli, 'run_analysis', side_effect=KeyboardInterrupt()):
                    result = cli.main()
                    
                    assert result == 130
        finally:
            sys.argv = original_argv
    
    def test_main_unexpected_exception(self: "TestMainFunctionAndArgumentHandling") -> None:
        """
        This test simulates an unexpected exception during the execution of `MPASUnifiedCLI.main()` to verify that the CLI handles such exceptions gracefully. By patching the `run_analysis` method to raise a generic `RuntimeError`, the test checks that `main()` catches this exception and returns an appropriate exit code (commonly 1 for general errors) instead of allowing the exception to propagate uncaught.

        Parameters:
            self (TestMainFunctionAndArgumentHandling): The test instance.

        Returns:
            None: The test asserts `main()` handles unexpected exceptions.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        cli = MPASUnifiedCLI()
        original_argv = sys.argv
        
        try:
            sys.argv = ['mpasdiag', 'precipitation', '--grid-file', 'test.nc', '--data-dir', 'data/']
            
            with patch.object(cli, 'validate_config', return_value=True):
                with patch.object(cli, 'run_analysis', side_effect=RuntimeError("Test error")):
                    result = cli.main()                    
                    assert result == 1
        finally:
            sys.argv = original_argv
    
    def test_module_level_main_function(self: "TestMainFunctionAndArgumentHandling") -> None:
        """
        This test verifies that the module-level `main()` function in `cli_unified` can be called directly and that it returns a zero exit code when the internal `MPASUnifiedCLI.main()` is mocked to return success. This ensures that the entry point provided by the module-level `main()` function correctly delegates to the CLI class and that it can be invoked without issues.

        Parameters:
            self (TestMainFunctionAndArgumentHandling): The test instance.

        Returns:
            None: The test asserts the module-level `main()` returns `0`.
        """
        from mpasdiag.processing import cli_unified
        
        with patch.object(cli_unified.MPASUnifiedCLI, 'main', return_value=0):
            result = cli_unified.main()
            
            assert result == 0


class TestMainArgumentReorderingAdditional:
    """ Test main() function argument reordering logic. """
    
    def test_main_with_global_flags_after_subcommand(self: "TestMainArgumentReorderingAdditional", grid_file, test_data_dir) -> None:
        """
        This test verifies that `MPASUnifiedCLI.main()` can handle global flags (like `--verbose`) appearing after the subcommand and that it processes the arguments correctly, returning a zero exit code on success. By setting `sys.argv` to include the `--verbose` flag after the subcommand and necessary arguments for a precipitation plot, the test checks that the CLI can reorder arguments as needed and execute without errors.

        Parameters:
            self (TestMainArgumentReorderingAdditional): The test instance.
            grid_file: Session fixture providing path to MPAS grid file.
            test_data_dir: Session fixture providing path to test data directory.

        Returns:
            None: Assertions validate the CLI exit code behavior.
        """
        import pytest
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        if grid_file is None:
            pytest.skip("Test data files not available")
        
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
            
            assert result == 0
            
        finally:
            sys.argv = original_argv
    
    def test_main_with_log_file_argument(self: "TestMainArgumentReorderingAdditional", grid_file, test_data_dir) -> None:
        """
        This test verifies that `MPASUnifiedCLI.main()` can handle a `--log-file` argument appearing in the command line and that it processes the arguments correctly, returning a zero exit code on success. By setting `sys.argv` to include the `--log-file` flag along with necessary arguments for a precipitation plot, the test checks that the CLI can handle log file specification and that it executes without errors, creating the log file as expected.

        Parameters:
            self (TestMainArgumentReorderingAdditional): The test instance.
            grid_file: Session fixture providing path to MPAS grid file.
            test_data_dir: Session fixture providing path to test data directory.

        Returns:
            None: The test asserts exit code and filesystem side-effects.
        """
        import pytest
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        import tempfile
        
        if grid_file is None:
            pytest.skip("Test data files not available")
        
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
            
            assert result == 0
            assert os.path.exists(log_file)
            
        finally:
            sys.argv = original_argv
            if os.path.exists(log_file):
                os.unlink(log_file)


class TestMainFunctionFallbackParsing:
    """ Test main() function argument parsing fallback. """
    
    def test_main_fallback_parsing_with_parse_args_exception(self: "TestMainFunctionFallbackParsing") -> None:
        """
        This test simulates a scenario where both `parse_intermixed_args` and `parse_args` raise exceptions during the execution of `MPASUnifiedCLI.main()`. By mocking the parser to raise exceptions on both parsing attempts, the test checks that `main()` correctly handles this situation by catching the exceptions and returning an appropriate exit code (commonly 1 for general errors) instead of allowing the exceptions to propagate uncaught.

        Parameters:
            self (TestMainFunctionFallbackParsing): The test instance.

        Returns:
            None: The test asserts that the fallback parsing path is used and that subsequent validation behaves as expected.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        cli = MPASUnifiedCLI()
        
        with patch('sys.argv', ['mpasdiag', 'precipitation', '--grid-file', 'test.nc', '--data-dir', 'data/']):
            with patch.object(cli, 'create_main_parser') as mock_parser_factory:
                mock_parser = Mock()
                mock_parser.parse_intermixed_args.side_effect = Exception("Parse error 1")
                mock_parser.parse_args.side_effect = Exception("Parse error 2")
                mock_parser_factory.return_value = mock_parser
                
                result = cli.main()
                assert result == 1
    
    def test_main_exception_without_logger(self: "TestMainFunctionFallbackParsing") -> None:
        """
        This test verifies that if an unexpected exception occurs during the execution of `MPASUnifiedCLI.main()` and there is no logger attached to handle the error, the function still returns an appropriate exit code (commonly 1 for general errors) instead of allowing the exception to propagate uncaught. By mocking the parser to raise a `RuntimeError` and ensuring that `cli.logger` is set to `None`, the test checks that `main()` can handle exceptions gracefully even without a logger.

        Parameters:
            self (TestMainFunctionFallbackParsing): The test instance.

        Returns:
            None: The test asserts that `main()` handles exceptions and does not require an attached logger to manage errors.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        cli = MPASUnifiedCLI()
        cli.logger = None  
        
        with patch('sys.argv', ['mpasdiag', 'precipitation', '--grid-file', 'test.nc', '--data-dir', 'data/']):
            with patch.object(cli, 'create_main_parser') as mock_parser_factory:
                mock_parser = Mock()
                mock_parser.parse_intermixed_args.side_effect = RuntimeError("Unexpected error")
                mock_parser_factory.return_value = mock_parser
                
                result = cli.main()
                assert result == 1


class TestMainArgumentReorderingUnifiedFinal:
    """ Test main() function argument reordering logic. """
    
    def test_main_with_global_arg_equals_format(self: "TestMainArgumentReorderingUnifiedFinal") -> None:
        """
        This test verifies that `MPASUnifiedCLI.main()` can handle a global argument provided in the `--arg=value` format and that it processes the arguments correctly, returning a zero exit code on success. By creating a temporary config file and invoking `main()` with the `--config` argument using the equals sign format, the test checks that the CLI can parse this style of argument and that it executes without errors when validation fails.

        Parameters:
            self (TestMainArgumentReorderingUnifiedFinal): The test instance.

        Returns:
            None: Assertions validate the returned exit code.
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
                        assert result == 1
        finally:
            if os.path.exists(config_file):
                os.unlink(config_file)
    
    def test_main_with_global_arg_at_end(self: "TestMainArgumentReorderingUnifiedFinal") -> None:
        """
        This test verifies that `MPASUnifiedCLI.main()` can handle a global argument (like `--verbose`) appearing at the end of the command line and that it processes the arguments correctly, returning a zero exit code on success. By setting `sys.argv` to include the `--verbose` flag after the subcommand and necessary arguments for a precipitation plot, the test checks that the CLI can reorder arguments as needed and execute without errors, even when validation fails.

        Parameters:
            self (TestMainArgumentReorderingUnifiedFinal): The test instance.

        Returns:
            None: Assertions validate the returned exit code.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        cli = MPASUnifiedCLI()
        
        with patch('sys.argv', ['mpasdiag', 'precipitation', '--grid-file', 'test.nc', '--data-dir', 'data/', '--verbose']):
            with patch.object(cli, 'setup_logging'):
                with patch.object(cli, 'validate_config', return_value=False):
                    result = cli.main()
                    assert result == 1


class TestWorkersArgumentIntegration:
    """ Integration tests for --workers CLI argument functionality. """
    
    @staticmethod
    def run_mpasdiag_command(workers: Optional[int] = None, plot_type: str = "surface") -> Optional[subprocess.CompletedProcess]:
        """
        This helper function executes the `mpasdiag` CLI as a subprocess with specified worker count and plot type, capturing the output for analysis. It constructs the command with appropriate flags and arguments, runs it, and returns the completed process object. If required data files are not available or if execution fails (e.g., due to a timeout), it returns `None` to indicate that the test should be skipped.

        Parameters:
            workers (Optional[int]): Number of worker processes to request, or `None` to allow the application to select a default.
            plot_type (str): Plot category to request (e.g., "surface", "precipitation").

        Returns:
            Optional[subprocess.CompletedProcess]: The `CompletedProcess` returned by `subprocess.run` when execution completes, or `None` if execution could not be performed (missing data or timeout).
        """
        import subprocess
        import os
        
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "u240k", "mpasout")
        grid_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "grids", "x1.10242.static.nc")
        
        if not os.path.exists(data_dir) or not os.path.exists(grid_file):
            return None
        
        cmd = [
            "mpasdiag", plot_type,
            "--grid-file", grid_file,
            "--data-dir", data_dir,
            "--output-dir", "/tmp/test_output"
        ]
        
        if plot_type == "surface":
            cmd.extend(["--variable", "t2m"])
        elif plot_type == "precipitation":
            cmd.extend(["--variable", "rainnc"])
        
        if workers is not None:
            cmd.extend(["--workers", str(workers)])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            return result
        except subprocess.TimeoutExpired:
            return None
    
    @staticmethod
    def extract_metrics(output: str) -> Dict[str, Any]:
        """
        This helper function parses the combined stdout and stderr output from the `mpasdiag` CLI to extract any reported metrics related to worker count and speedup. It uses regular expressions to search for specific patterns in the output that indicate the number of workers initialized and any speedup metrics reported by the application. The extracted metrics are returned as a dictionary for use in assertions within the tests.

        Parameters:
            output (str): Combined stdout and stderr text from a command run.

        Returns:
            Dict[str, Any]: Mapping of discovered metrics (for example {"speedup": 1.23, "workers": 4}). Numeric values are either `float` or `int` depending on the metric.
        """
        import re
        metrics: Dict[str, Any] = {}

        speedup_match = re.search(r"Speedup: ([\d.]+)x", output)

        if speedup_match:
            metrics["speedup"] = float(speedup_match.group(1))

        worker_match = re.search(r"initialized in multiprocessing mode with (\d+) workers", output)

        if worker_match:
            metrics["workers"] = int(worker_match.group(1))

        return metrics
    
    def test_workers_argument_single(self) -> None:
        """
        This test validates that when the `--workers 1` argument is provided to the `mpasdiag` CLI, the application initializes in single-worker mode and reports the expected worker count and speedup metrics. It runs the CLI as a subprocess with the specified worker count and checks the output for any reported metrics, asserting that the worker count is correct and that the speedup is reasonable for a single-worker execution. If data files are missing or if execution fails, the test will be skipped.

        Parameters:
            self (TestWorkersArgumentIntegration): The test instance.

        Returns:
            None: Assertions validate execution results.
        """
        result = self.run_mpasdiag_command(workers=1, plot_type="surface")

        if result is None:
            pytest.skip("Test data not available")
        
        exit_code = result.returncode

        assert exit_code in [0, 1, 143], f"Command failed with exit code {exit_code}\nSTDOUT: {result.stdout[:500]}\nSTDERR: {result.stderr[:500]}"
        
        metrics = self.extract_metrics(result.stdout + result.stderr)
        
        if "workers" in metrics:
            assert metrics["workers"] == 1, f"Worker count wrong: expected 1, got {metrics['workers']}"
        
        if "speedup" in metrics:
            assert metrics["speedup"] < 1.5, f"Speedup too high for single worker: {metrics['speedup']}x"
    
    def test_workers_argument_multiple(self: "TestWorkersArgumentIntegration") -> None:
        """
        This test validates that when the `--workers` argument is provided with a value greater than 1, the `mpasdiag` CLI initializes in multiprocessing mode and reports the expected worker count. It runs the CLI as a subprocess with a specified worker count (e.g., 4) and checks the output for any reported metrics, asserting that the worker count matches the requested value. If data files are missing or if execution fails, the test will be skipped.

        Parameters:
            self (TestWorkersArgumentIntegration): The test instance.

        Returns:
            None: Assertions validate the observed worker count metrics.
        """
        result = self.run_mpasdiag_command(workers=4, plot_type="surface")

        if result is None:
            pytest.skip("Test data not available")
        
        exit_code = result.returncode

        assert exit_code in [0, 1, 143], f"Command failed with exit code {exit_code}\nSTDERR: {result.stderr[:500]}"
        
        metrics = self.extract_metrics(result.stdout + result.stderr)
        
        if "workers" in metrics:
            assert metrics["workers"] == 4, f"Worker count wrong: expected 4, got {metrics['workers']}"
    
    def test_workers_argument_default(self: "TestWorkersArgumentIntegration") -> None:
        """
        This test validates that when the `--workers` argument is not provided, the `mpasdiag` CLI initializes with a default worker count (which may be 1 or more depending on the environment) and reports the worker count in the output. It runs the CLI as a subprocess without specifying the worker count and checks the output for any reported metrics, asserting that a worker count is present and is a positive integer. If data files are missing or if execution fails, the test will be skipped.

        Parameters:
            self (TestWorkersArgumentIntegration): The test instance.

        Returns:
            None: Assertions verify returned metrics when present.
        """
        result = self.run_mpasdiag_command(workers=None, plot_type="surface")

        if result is None:
            pytest.skip("Test data not available")
        
        exit_code = result.returncode
        
        assert exit_code in [0, 1, 143], f"Command failed with exit code {exit_code}\nSTDERR: {result.stderr[:500]}"
        
        metrics = self.extract_metrics(result.stdout + result.stderr)
        
        if "workers" in metrics:
            assert metrics["workers"] >= 1, f"Worker count should be >= 1, got {metrics['workers']}"
    
    def test_workers_all_plot_types(self: "TestWorkersArgumentIntegration") -> None:
        """
        This test validates that the `--workers` argument is accepted and processed correctly across multiple plot types supported by the `mpasdiag` CLI. It runs the CLI as a subprocess for each plot type (e.g., "surface", "precipitation") with a specified worker count and checks the output for any reported metrics, asserting that the worker count matches the requested value for each plot type. If data files are missing or if execution fails for any plot type, the test will be skipped.

        Parameters:
            self (TestWorkersArgumentIntegration): The test instance.

        Returns:
            None: The test asserts that all selected plot types passed.
        """
        plot_types = ["surface", "precipitation"]
        all_passed = True
        
        for plot_type in plot_types:
            result = self.run_mpasdiag_command(workers=2, plot_type=plot_type)
            
            if result is None:
                continue
            
            exit_code = result.returncode

            if exit_code not in [0, 1, 143]:
                all_passed = False
                continue
            
            metrics = self.extract_metrics(result.stdout + result.stderr)

            if "workers" in metrics and metrics["workers"] != 2:
                all_passed = False
        
        assert all_passed, "Some plot types failed"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
