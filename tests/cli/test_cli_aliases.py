#!/usr/bin/env python3

"""
MPASdiag Test Suite: Command Line Interface Aliases

This module contains tests for validating the behavior of command aliases in the MPASdiag CLI. It ensures that alternative subcommand names (e.g., `precip`, `surf`, `vector`) are correctly recognized by the argument parser and map to the expected analysis commands. The tests cover aliases for precipitation, surface, wind, and cross-section analyses, as well as overlay analysis types.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load standard libraries
import os
import pytest


class TestCommandAliases:
    """ Test command aliases for MPASdiag CLI. """

    def test_precipitation_aliases(self: "TestCommandAliases") -> None:
        """
        This test verifies that the precipitation analysis command aliases are accepted by the CLI parser. It checks that `precipitation`, `precip`, and `rain` are all recognized as valid subcommands and that the parser correctly sets the `analysis_command` attribute to the invoked alias.

        Parameters:
            self (unittest.TestCase): The test instance.

        Returns:
            None: Raises on parsing mismatch.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()

        for alias in ['precipitation', 'precip', 'rain']:
            args = parser.parse_args([
                alias,
                '--grid-file', 'test.nc',
                '--data-dir', 'data/',
                '--time-index', '0'
            ])
            assert args.analysis_command == alias

    def test_surface_aliases(self: "TestCommandAliases") -> None:
        """
        This test confirms that surface analysis command aliases are properly recognized by the CLI parser. It iterates through known surface aliases such as `surface`, `surf`, and `2d`, parsing arguments for each and asserting that the `analysis_command` attribute matches the invoked alias.

        Parameters:
            self (unittest.TestCase): The test instance.

        Returns:
            None: Raises on mismatch.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()

        for alias in ['surface', 'surf', '2d']:
            args = parser.parse_args([
                alias,
                '--grid-file', 'test.nc',
                '--data-dir', 'data/',
                '--variable', 'temp',
                '--time-index', '0'
            ])
            assert args.analysis_command == alias

    def test_wind_aliases(self: "TestCommandAliases") -> None:
        """
        This test verifies that wind analysis command aliases are accepted by the CLI parser. It checks that `wind`, `vector`, and `winds` are all recognized as valid subcommands for wind analysis and that the parser sets the `analysis_command` attribute to the invoked alias.

        Parameters:
            self (unittest.TestCase): The test instance.

        Returns:
            None: Raises on mismatch.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()

        for alias in ['wind', 'vector', 'winds']:
            args = parser.parse_args([
                alias,
                '--grid-file', 'test.nc',
                '--data-dir', 'data/',
                '--time-index', '0'
            ])
            assert args.analysis_command == alias

    def test_cross_section_aliases(self: "TestCommandAliases") -> None:
        """
        This test verifies that cross-section analysis command aliases are accepted by the CLI parser. It checks that `cross`, `xsec`, `3d`, and `vertical` are all recognized as valid subcommands for cross-section analysis and that the parser sets the `analysis_command` attribute to the invoked alias.

        Parameters:
            self (unittest.TestCase): The test instance.

        Returns:
            None: Raises on parsing mismatch.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()

        for alias in ['cross', 'xsec', '3d', 'vertical']:
            args = parser.parse_args([
                alias,
                '--grid-file', 'test.nc',
                '--data-dir', 'data/',
                '--variable', 'temp',
                '--start-lat', '30',
                '--start-lon', '-100',
                '--end-lat', '40',
                '--end-lon', '-90',
                '--time-index', '0'
            ])
            assert args.analysis_command == alias


class TestAnalysisTypeAliases:
    """ Test analysis type aliases for MPASdiag CLI. """
    
    def test_precipitation_analysis_precip_alias(self: "TestAnalysisTypeAliases") -> None:
        """
        This test exercises the precipitation analysis workflow using the `precip` alias. It constructs an `MPASConfig` with `analysis_type='precip'` and invokes `run_analysis`. The test is skipped if example data are not available; otherwise it asserts that the call returns a boolean result indicating whether the analysis attempted to run successfully.

        Parameters:
            self (TestAnalysisTypeAliases): The test instance.

        Returns:
            None: The test asserts the return value is a boolean.
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
            analysis_type='precip',  # Alias
            variable='rainnc',
            accumulation_period='a01h',
            time_index=0,
            output_dir='output/test_precip_alias',
            verbose=False
        )
        
        result = cli.run_analysis(config)
        assert isinstance(result, bool)
    
    def test_precipitation_analysis_rain_alias(self: "TestAnalysisTypeAliases") -> None:
        """
        This test exercises the precipitation analysis workflow using the `rain` alias. It builds an `MPASConfig` with `analysis_type='rain'` and calls `run_analysis`. If sample data are unavailable, the test is skipped; otherwise it asserts that the call returns a boolean indicating whether the analysis attempted to run.

        Parameters:
            self (TestAnalysisTypeAliases): The test instance.

        Returns:
            None: The test asserts the return value is a boolean.
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
            analysis_type='rain',  # Alias
            variable='rainnc',
            accumulation_period='a01h',
            time_index=0,
            output_dir='output/test_rain_alias',
            verbose=False
        )
        
        result = cli.run_analysis(config)
        assert isinstance(result, bool)
    
    def test_surface_analysis_surf_alias(self: "TestAnalysisTypeAliases") -> None:
        """
        This test exercises the surface analysis workflow using the `surf` alias. It constructs an `MPASConfig` with `analysis_type='surf'` and invokes `run_analysis`. The test is skipped if example data are not available; otherwise it asserts that the call returns a boolean result indicating whether the analysis attempted to run successfully.

        Parameters:
            self (TestAnalysisTypeAliases): The test instance.

        Returns:
            None: The test asserts that `run_analysis` returns a boolean indicating whether the analysis attempted to run.
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
            analysis_type='surf',  # Alias
            variable='t2m',
            plot_type='scatter',
            time_index=0,
            output_dir='output/test_surf_alias',
            verbose=False
        )
        
        result = cli.run_analysis(config)
        assert isinstance(result, bool)
    
    def test_surface_analysis_2d_alias(self: "TestAnalysisTypeAliases") -> None:
        """
        This test exercises the surface analysis workflow using the `2d` alias. It builds an `MPASConfig` with `analysis_type='2d'` and calls `run_analysis`. If sample data are unavailable, the test is skipped; otherwise it asserts that the call returns a boolean indicating whether the analysis attempted to run.

        Parameters:
            self (TestAnalysisTypeAliases): The test instance.

        Returns:
            None: The test asserts the returned result is a boolean.
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
            analysis_type='2d',  # Alias
            variable='t2m',
            plot_type='scatter',
            time_index=0,
            output_dir='output/test_2d_alias',
            verbose=False
        )
        
        result = cli.run_analysis(config)
        assert isinstance(result, bool)
    
    def test_wind_analysis_vector_alias(self: "TestAnalysisTypeAliases") -> None:
        """
        This test exercises the wind analysis workflow using the `vector` alias. It constructs an `MPASConfig` with `analysis_type='vector'` and invokes `run_analysis`. The test is skipped if example data are not available; otherwise it asserts that the call returns a boolean result indicating whether the analysis attempted to run successfully.

        Parameters:
            self (TestAnalysisTypeAliases): The test instance.

        Returns:
            None: The test asserts the returned value is a boolean.
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
            analysis_type='vector',  # Alias
            u_variable='u10',
            v_variable='v10',
            wind_plot_type='barbs',
            subsample_factor=5,
            time_index=0,
            output_dir='output/test_vector_alias',
            verbose=False
        )
        
        result = cli.run_analysis(config)
        assert isinstance(result, bool)
    
    def test_wind_analysis_winds_alias(self: "TestAnalysisTypeAliases") -> None:
        """
        This test exercises the wind analysis workflow using the `winds` alias. It constructs an `MPASConfig` with `analysis_type='winds'` and invokes `run_analysis`. The test is skipped if example data are not available; otherwise it asserts that the call returns a boolean result indicating whether the analysis attempted to run successfully.

        Parameters:
            self (TestAnalysisTypeAliases): The test instance.

        Returns:
            None: The test asserts the returned result is a boolean.
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
            analysis_type='winds',  # Alias
            u_variable='u10',
            v_variable='v10',
            wind_plot_type='barbs',
            subsample_factor=5,
            time_index=0,
            output_dir='output/test_winds_alias',
            verbose=False
        )
        
        result = cli.run_analysis(config)
        assert isinstance(result, bool)
    
    def test_overlay_analysis_complex_alias(self: "TestAnalysisTypeAliases") -> None:
        """
        This test exercises overlay analysis using the `complex` alias. It constructs an `MPASConfig` with `analysis_type='complex'` and calls `run_analysis`. If sample data are unavailable, the test is skipped; otherwise it asserts that the call returns True to indicate the overlay workflow completed successfully.

        Parameters:
            self (TestOverlayAnalysisAliases): The test instance.

        Returns:
            None: Asserts `run_analysis` returns True when data exist.
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
            analysis_type='complex',  # Alias
            overlay_type='precip_wind',
            time_index=0,
            output_dir='output/test_complex_alias',
            verbose=False
        )
        
        result = cli.run_analysis(config)
        assert result is True
    
    def test_overlay_analysis_multi_alias(self: "TestAnalysisTypeAliases") -> None:
        """
        This test exercises overlay analysis using the `multi` alias. It constructs an `MPASConfig` with `analysis_type='multi'` and calls `run_analysis`. If sample data are unavailable, the test is skipped; otherwise it asserts that the call returns True to indicate the multi-overlay workflow completed successfully.

        Parameters:
            self (TestAnalysisTypeAliases): The test instance.

        Returns:
            None: Asserts `run_analysis` returns True when data exist.
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
            analysis_type='multi',  # Alias
            overlay_type='temp_pressure',
            time_index=0,
            output_dir='output/test_multi_alias',
            verbose=False
        )
        
        result = cli.run_analysis(config)
        assert result is True
    
    def test_overlay_analysis_composite_alias(self: "TestAnalysisTypeAliases") -> None:
        """
        This test exercises overlay analysis using the `composite` alias. It constructs an `MPASConfig` with `analysis_type='composite'` and calls `run_analysis`. If sample data are unavailable, the test is skipped; otherwise it asserts that the call returns True to indicate the composite overlay workflow completed successfully.

        Parameters:
            self (TestAnalysisTypeAliases): The test instance.

        Returns:
            None: Asserts `run_analysis` returns True when data exist.
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
            analysis_type='composite',  # Alias
            overlay_type='precip_wind',
            time_index=0,
            output_dir='output/test_composite_alias',
            verbose=False
        )
        
        result = cli.run_analysis(config)
        assert result is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
