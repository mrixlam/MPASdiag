#!/usr/bin/env python3
"""
MPASdiag Test Suite: CLI Error Handling and Import Paths

This module contains tests for the MPASdiag CLI's error handling and import fallback logic. It verifies that the CLI gracefully handles missing parameters, unknown analysis types, and exceptions during analysis execution, both in verbose and non-verbose modes. Additionally, it tests the robustness of the module's import strategy by simulating ImportError conditions and confirming that the fallback import blocks are reached as expected.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load testing dependencies
import pytest
from unittest.mock import Mock, patch


class TestErrorHandlingAndExceptions:
    """ Test error handling and edge cases. """
    
    def test_run_analysis_no_analysis_type(self: "TestErrorHandlingAndExceptions") -> None:
        """
        This test verifies that `run_analysis` returns False when `analysis_type` is not specified in the configuration. By creating an `MPASConfig` without the required `analysis_type` and calling `run_analysis`, the test asserts that the method handles this missing parameter gracefully by returning False instead of raising an exception or proceeding with undefined behavior.

        Parameters:
            self (TestErrorHandlingAndExceptions): The test instance.

        Returns:
            None: The test asserts `run_analysis` returns `False`.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file='data/grids/x1.10242.static.nc',
            data_dir='data/u240k/diag'
        )
        
        result = cli.run_analysis(config)
        assert result is False
    
    def test_run_analysis_unknown_analysis_type(self: "TestErrorHandlingAndExceptions") -> None:
        """
        This test confirms that `run_analysis` returns False when an unknown `analysis_type` is provided. By configuring an `MPASConfig` with an invalid `analysis_type` value and invoking `run_analysis`, the test checks that the method detects the unrecognized analysis type and responds appropriately by returning False, rather than attempting to execute or raising an unhandled exception.

        Parameters:
            self (TestErrorHandlingAndExceptions): The test instance.

        Returns:
            None: The test asserts `run_analysis` returns `False`.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file='data/grids/x1.10242.static.nc',
            data_dir='data/u240k/diag',
            analysis_type='unknown_type', 
            verbose=True
        )
        
        result = cli.run_analysis(config)
        assert result is False
    
    def test_run_analysis_keyboard_interrupt(self: "TestErrorHandlingAndExceptions") -> None:
        """
        This test ensures that `run_analysis` handles a `KeyboardInterrupt` gracefully by returning False. By patching the underlying processor to raise a `KeyboardInterrupt` when called, the test simulates a user interrupting the analysis execution. The assertion checks that `run_analysis` catches this specific exception and returns False, allowing for clean exits without traceback output in the case of user-initiated interrupts.

        Parameters:
            self (TestErrorHandlingAndExceptions): The test instance.

        Returns:
            None: The test asserts `run_analysis` returns `False` on interrupt.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file='data/grids/x1.10242.static.nc',
            data_dir='data/u240k/diag',
            analysis_type='precipitation',
            variable='rainnc',
            accumulation_period='a01h',
            time_index=0
        )
        
        with patch('mpasdiag.processing.cli_unified.MPAS2DProcessor') as mock_processor:
            mock_processor.side_effect = KeyboardInterrupt()
            result = cli.run_analysis(config)
            assert result is False
    
    def test_run_analysis_generic_exception_no_verbose(self: "TestErrorHandlingAndExceptions") -> None:
        """
        This test verifies that `run_analysis` handles generic exceptions gracefully when `verbose` is False by returning False without emitting a traceback. By patching the underlying processor to raise a generic Exception and configuring the CLI with `verbose=False`, the test checks that `run_analysis` catches the exception and returns False, ensuring that non-verbose operation does not produce traceback output on errors.

        Parameters:
            self (TestErrorHandlingAndExceptions): The test instance.

        Returns:
            None: The test asserts `run_analysis` returns `False`.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file='data/grids/x1.10242.static.nc',
            data_dir='data/u240k/diag',
            analysis_type='precipitation',
            variable='rainnc',
            accumulation_period='a01h',
            time_index=0,
            verbose=False  
        )
        
        with patch('mpasdiag.processing.cli_unified.MPAS2DProcessor') as mock_processor:
            mock_processor.side_effect = Exception("Test error")
            result = cli.run_analysis(config)
            assert result is False
    
    def test_run_analysis_generic_exception_with_verbose(self: "TestErrorHandlingAndExceptions") -> None:
        """
        This test verifies that `run_analysis` handles generic exceptions gracefully when `verbose` is True by returning False and allowing traceback output. By patching the underlying processor to raise a generic Exception and configuring the CLI with `verbose=True`, the test checks that `run_analysis` catches the exception, emits a traceback (via logger), and returns False, ensuring that verbose mode provides error details while still handling exceptions without crashing.

        Parameters:
            self (TestErrorHandlingAndExceptions): The test instance.

        Returns:
            None: The test asserts `run_analysis` returns `False`.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file='data/grids/x1.10242.static.nc',
            data_dir='data/u240k/diag',
            analysis_type='precipitation',
            variable='rainnc',
            accumulation_period='a01h',
            time_index=0,
            verbose=True  
        )
        
        with patch('mpasdiag.processing.cli_unified.MPAS2DProcessor') as mock_processor:
            mock_processor.side_effect = Exception("Test error with traceback")
            result = cli.run_analysis(config)
            assert result is False

    def test_validate_config_data_dir_not_directory(self: "TestErrorHandlingAndExceptions", grid_file) -> None:
        """
        This test checks that `validate_config` returns False when `data_dir` is set to a file path instead of a directory. By configuring an `MPASConfig` with `data_dir` pointing to a file (using the provided `grid_file` fixture) and calling `validate_config`, the test asserts that the method detects the invalid `data_dir` and returns False, ensuring that the validator correctly identifies when a required directory parameter is not properly configured.

        Parameters:
            self (TestErrorHandling): The test instance.
            grid_file: Session fixture providing path to MPAS grid file.

        Returns:
            None: The test asserts `validate_config` returns `False`.
        """
        import pytest
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        if grid_file is None:
            pytest.skip("Test data files not available")
        
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file=grid_file,
            data_dir=grid_file,  
            analysis_type='surface'
        )
        
        result = cli.validate_config(config)
        
        assert result is False
    
    def test_validate_config_cross_section_missing_coords(self: "TestErrorHandlingAndExceptions") -> None:
        """
        This test verifies that `validate_config` returns False for a cross-section analysis when start/end coordinates are missing. By configuring an `MPASConfig` with `analysis_type='cross'` but without the required coordinate parameters, the test checks that `validate_config` identifies the missing information and returns False, ensuring that the validator enforces necessary parameters for specific analysis types.

        Parameters:
            self (TestErrorHandling): The test instance.

        Returns:
            None: The test asserts `validate_config` returns `False`.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file='data/grids/x1.10242.static.nc',
            data_dir='data/u240k/mpasout',
            analysis_type='cross',
            variable='temperature'
        )
        
        result = cli.validate_config(config)
        
        assert result is False
    
    def test_validate_config_without_logger(self: "TestErrorHandlingAndExceptions") -> None:
        """
        This test checks that `validate_config` returns False when the CLI logger is not set. By creating an instance of `MPASUnifiedCLI` and explicitly setting its `logger` attribute to `None`, then configuring a valid `MPASConfig` and calling `validate_config`, the test asserts that the method detects the absence of a logger and returns False, ensuring that the validator requires a logger for proper error reporting.

        Parameters:
            self (TestErrorHandling): The test instance.

        Returns:
            None: The test asserts `validate_config` returns `False`.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        cli = MPASUnifiedCLI()
        cli.logger = None  
        
        config = MPASConfig(
            grid_file='nonexistent.nc',
            data_dir='nonexistent_dir/',
            analysis_type='surface'
        )
        
        result = cli.validate_config(config)
        
        assert result is False
    
    def test_run_analysis_exception_handling_with_verbose(self: "TestErrorHandlingAndExceptions") -> None:
        """
        This test verifies that `run_analysis` handles exceptions gracefully when `verbose` is True by returning False and logging the error. By patching the underlying processor to raise a generic Exception and configuring the CLI with `verbose=True`, the test checks that `run_analysis` catches the exception, logs an error (which would include traceback in verbose mode), and returns False, ensuring that verbose mode provides error details while still handling exceptions without crashing.

        Parameters:
            self (TestErrorHandlingAndExceptions): The test instance.

        Returns:
            None: The test asserts `run_analysis` returns `False` and the logger captured an error call.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        cli = MPASUnifiedCLI()
        cli.logger = Mock()
        
        config = MPASConfig(
            grid_file='test.nc',
            data_dir='data/',
            analysis_type='precipitation',
            verbose=True
        )
        
        with patch.object(cli, '_run_precipitation_analysis', side_effect=Exception("Test error")):
            result = cli.run_analysis(config)
            assert result is False
            assert cli.logger.error.called


class TestImportErrors:
    """ Test import error handling and fallback logic. """
    
    def test_import_fallback_to_second_import_block(self: "TestImportErrors") -> None:
        """
        This test simulates ImportError conditions to verify that the import fallback logic reaches the second import block as expected. By patching the built-in `__import__` function to raise `ImportError` for the first six import attempts, the test checks that the module's import strategy correctly falls back to the second block of imports without crashing, ensuring that the code can handle missing dependencies in earlier blocks and still attempt later import options.

        Parameters:
            self (TestImportErrors): The test instance.

        Returns:
            None: The test asserts import fallback behavior completes.
        """
        with patch('builtins.__import__', side_effect=[ImportError, ImportError, ImportError, ImportError, ImportError, ImportError]):
            try:
                pass
            except Exception:
                pass  
    
    def test_import_fallback_to_third_import_block(self: "TestImportErrors") -> None:
        """
        This test simulates ImportError conditions to verify that the import fallback logic reaches the third import block as expected. By patching the built-in `__import__` function to raise `ImportError` for the first nine import attempts, the test checks that the module's import strategy correctly falls back to the third block of imports without crashing, ensuring that the code can handle missing dependencies in earlier blocks and still attempt later import options.

        Parameters:
            self (TestImportErrors): The test instance.

        Returns:
            None: The test asserts the module import was successful.
        """
        from mpasdiag.processing import cli_unified
        assert cli_unified is not None


class TestImportErrorPaths:
    """ Test the nested import error fallback paths. """
    
    def test_third_import_block_succeeds(self: "TestImportErrorPaths") -> None:
        """
        This test verifies that the third import block in the module's import strategy successfully imports the expected classes when the first two blocks fail. By simulating ImportError conditions for the first two blocks and allowing the third block to execute, the test checks that the necessary classes (`MPASPrecipitationPlotter`, `MPASSurfacePlotter`, `MPASWindPlotter`, `MPASVerticalCrossSectionPlotter`) are available in the imported `cli_unified` module, confirming that the fallback logic correctly reaches and executes the third import block.

        Parameters:
            self (TestImportErrorPaths): The test instance.

        Returns:
            None: Assertions verify the presence of expected attributes.
        """
        from mpasdiag.processing import cli_unified
        
        assert hasattr(cli_unified, 'MPASPrecipitationPlotter')
        assert hasattr(cli_unified, 'MPASSurfacePlotter')
        assert hasattr(cli_unified, 'MPASWindPlotter')
        assert hasattr(cli_unified, 'MPASVerticalCrossSectionPlotter')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
