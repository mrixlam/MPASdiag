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


class TestModuleExecution:
    """ Test module-level execution paths. """

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
            assert result == pytest.approx(0)
            mock_main.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
