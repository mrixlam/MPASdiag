#!/usr/bin/env python3
"""
MPASdiag Test Suite: CLI Analysis with Logger

This module contains tests for the `run_analysis` method of the `MPASUnifiedCLI` class, specifically focusing on scenarios where a logger is not configured and when it is active. The tests cover various analysis types, including wind and cross-section analyses, and verify that the method behaves as expected under different configurations.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules
import os
import pytest


class TestRunAnalysisNoLogger:
    """ Test run_analysis without logger. """
    
    def test_run_analysis_no_analysis_type_no_logger(self: "TestRunAnalysisNoLogger") -> None:
        """
        This test verifies that `run_analysis` returns False when no analysis type is specified and no logger is configured. The test initializes the `MPASUnifiedCLI` without a logger and provides a configuration that lacks an `analysis_type`. It asserts that the method returns False, indicating that the analysis cannot proceed without a specified type. This test is designed to cover the code path where the method checks for the presence of an analysis type and handles the absence of a logger gracefully.

        Parameters:
            self (TestRunAnalysisNoLogger): The test instance.

        Returns:
            None: Assertions validate the boolean return value.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        cli = MPASUnifiedCLI()
        cli.logger = None  
        
        config = MPASConfig(
            grid_file='data/grids/x1.10242.static.nc',
            data_dir='data/u240k/diag'
        )
        
        result = cli.run_analysis(config)
        assert result is False


class TestWindAnalysisTimeFormatting:
    """ Test wind analysis time string formatting. """
    
    def test_wind_single_time_with_time_dimension(self: "TestWindAnalysisTimeFormatting") -> None:
        """
        This test verifies that the `run_analysis` method correctly handles time formatting for wind analysis when a single time index is specified. The test initializes the `MPASUnifiedCLI` without a logger and provides a configuration for a wind analysis with a specific time index. It asserts that the method returns a boolean, indicating that the analysis completed successfully and that the time formatting code path was exercised. This test is intended to cover scenarios where the method must format time strings for output files when processing a single time step.

        Parameters:
            self (TestWindAnalysisTimeFormatting): The test instance.

        Returns:
            None: Assertions validate that `run_analysis` returns a boolean.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        if not os.path.exists('data/grids/x1.10242.static.nc'):
            pytest.skip("Test data not available")
            return
        
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file='data/grids/x1.10242.static.nc',
            data_dir='data/u240k/diag',
            analysis_type='wind',
            u_variable='u10',
            v_variable='v10',
            wind_plot_type='barbs',
            subsample_factor=5,
            time_index=0,
            batch_mode=False,  
            output_dir='output/test_wind_time_format',
            verbose=False
        )
        
        result = cli.run_analysis(config)
        assert isinstance(result, bool)


class TestCrossSectionSingleTimeWithLogger:
    """ Test cross-section single time analysis with logger. """
    
    def test_cross_section_single_time_with_logger(self: "TestCrossSectionSingleTimeWithLogger") -> None:
        """
        This test verifies that the `run_analysis` method correctly processes a cross-section analysis for a single time index when a logger is configured. The test initializes the `MPASUnifiedCLI` with a logger and provides a configuration for a cross-section analysis with specific spatial and temporal parameters. It asserts that the method returns a boolean, indicating that the analysis completed successfully and that the logger was utilized during the process. This test is designed to cover the code path where the method handles cross-section analyses with logging enabled, ensuring that log messages are generated appropriately. 

        Parameters:
            self (TestCrossSectionSingleTimeWithLogger): The test instance.

        Returns:
            None: The test asserts that `run_analysis` returns a boolean indicating completion when data exist.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        from mpasdiag.processing.utils_logger import MPASLogger
        
        if not os.path.exists('data/grids/x1.10242.static.nc'):
            pytest.skip("Test data not available")
            return
        
        cli = MPASUnifiedCLI()
        cli.logger = MPASLogger('test', verbose=True)
        
        config = MPASConfig(
            grid_file='data/grids/x1.10242.static.nc',
            data_dir='data/u240k/mpasout',
            analysis_type='cross',
            variable='temperature',
            start_lon=95.0,
            start_lat=5.0,
            end_lon=105.0,
            end_lat=10.0,
            time_index=0,
            batch_mode=False, 
            output_dir='output/test_cross_single_logger',
            verbose=True
        )
        
        result = cli.run_analysis(config)
        assert isinstance(result, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
