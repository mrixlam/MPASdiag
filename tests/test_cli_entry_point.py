#!/usr/bin/env python3

"""
MPASdiag Test Suite: CLI Entry Point Tests

This module contains tests for the command-line interface (CLI) entry point of the MPASdiag package. It verifies that the main function is properly defined and that the module can be executed as a script, ensuring that the CLI entry point is functional. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import runpy
from unittest.mock import patch


class TestCliEntryPoint:
    """ Test suite for verifying the CLI entry point of the MPASdiag package. """

    def test_import_exposes_main(self: 'TestCliEntryPoint') -> None:
        """
        This test verifies that the cli module can be imported and that it exposes a main function, which is expected to be the entry point for the CLI.

        Parameters: 
            None

        Returns:
            None
        """
        import mpasdiag.cli as cli_mod
        assert hasattr(cli_mod, "main")
        assert callable(cli_mod.main)

    def test_module_runs_via_runpy(self: 'TestCliEntryPoint') -> None:
        """
        This test verifies that the cli module can be executed as a script using runpy, which simulates running the module directly from the command line. It checks that the main function is called when the module is run. 

        Parameters:
            None

        Returns:
            None
        """
        with patch("mpasdiag.processing.cli_unified.main") as mock_main:
            runpy.run_module("mpasdiag.cli", run_name="__main__", alter_sys=False)
        mock_main.assert_called_once()

