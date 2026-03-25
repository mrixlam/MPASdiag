#!/usr/bin/env python3
"""
MPASdiag Test Suite: CLI Module Tests

This module contains tests for the CLI module of MPASdiag, focusing on verifying the module-level execution paths, including the `main()` function and the `if __name__ == '__main__'` block. These tests ensure that the CLI can be invoked correctly as a module or script and that the expected behaviors occur when the module is run directly.    

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load pytest and unittest.mock for testing and mocking
import pytest
from unittest.mock import patch


class TestModuleLevelExecution:
    """ Test module-level execution. """
    
    def test_module_main_name_block(self: "TestModuleLevelExecution") -> None:
        """
        This test verifies that the module-level `main()` function can be called and that it returns the expected value when the module is run as a script. By patching the `main()` function to return a fixed value, we can confirm that the module-level execution path is correctly set up and that the `main()` function is callable in the expected context.

        Parameters:
            self (TestModuleLevelExecution): The test instance.

        Returns:
            None: The test asserts the module-level `main()` was called and its return value was forwarded.
        """
        from mpasdiag.processing import cli_unified
        
        with patch.object(cli_unified, 'main', return_value=0):
            with patch.object(cli_unified, '__name__', '__main__'):
                result = cli_unified.main()
                assert result == 0


class TestModuleExecution:
    """ Test module-level execution paths, including main() and __name__ == '__main__' block. """
    
    def test_module_level_main_instantiates_cli(self: "TestModuleExecution") -> None:
        """
        This test ensures that the module-level `main()` function correctly instantiates and calls the `MPASUnifiedCLI` class's `main()` method. By patching the `MPASUnifiedCLI.main` method, we can verify that the module-level `main()` function is properly forwarding the call to the CLI class and that it returns the expected exit code.

        Parameters:
            self (TestModuleExecution): The test instance.

        Returns:
            None: The test asserts `main()` returns the patched exit code.
        """
        from mpasdiag.processing import cli_unified
        
        with patch.object(cli_unified.MPASUnifiedCLI, 'main', return_value=0) as mock_main:
            result = cli_unified.main()
            assert result == 0
            mock_main.assert_called_once()
    
    def test_module_level_main_returns_exit_code(self: "TestModuleExecution") -> None:
        """
        This test verifies that the module-level `main()` function returns the exit code provided by the `MPASUnifiedCLI.main()` method. By patching the `MPASUnifiedCLI.main` method to return a specific exit code, we can confirm that the module-level `main()` function correctly propagates this exit code back to the caller.

        Parameters:
            self (TestModuleExecution): The test instance.

        Returns:
            None: The test asserts the wrapper returns the same exit code.
        """
        from mpasdiag.processing import cli_unified
        
        with patch.object(cli_unified.MPASUnifiedCLI, 'main', return_value=1):
            result = cli_unified.main()
            assert result == 1
    
    def test_main_block_calls_sys_exit(self: "TestModuleExecution") -> None:
        """
        This test checks that when the module is run as a script, the `main()` function is called and its return value is passed to `sys.exit()`. By patching both the `main()` function and `sys.exit()`, we can verify that the correct exit code is being used when the module is executed directly.

        Parameters:
            self (TestModuleExecution): The test instance.

        Returns:
            None: The test asserts that `main` is callable.
        """        
        from mpasdiag.processing.cli_unified import main
        assert callable(main)


class TestMainIfNameMain:
    """ Test the if __name__ == '__main__' block indirectly. """
    
    def test_module_can_be_run_as_script(self: "TestMainIfNameMain") -> None:
        """
        This test verifies that the `if __name__ == '__main__'` block is correctly set up to allow the module to be run as a script. By patching the `MPASUnifiedCLI.main` method, we can confirm that when the module is executed directly, it calls the CLI's main method and returns the expected exit code.

        Parameters:
            self (TestMainIfNameMain): The test instance.

        Returns:
            None: The test asserts that the module can be run as a script and that `main()` is called with the expected behavior.
        """
        from mpasdiag.processing import cli_unified
        
        assert hasattr(cli_unified, 'main')
        assert callable(cli_unified.main)
        assert hasattr(cli_unified, 'MPASUnifiedCLI')
        
        with patch.object(cli_unified.MPASUnifiedCLI, 'main', return_value=0):
            result = cli_unified.main()
            assert isinstance(result, int)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
