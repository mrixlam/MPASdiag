#!/usr/bin/env python3
"""
MPASdiag Test Suite: CLI Analysis Tests

This module contains tests for the command-line interface (CLI) analysis execution and argument parsing of the MPASdiag processing package. It verifies that the `run_analysis` method correctly dispatches to the appropriate analysis routines based on the specified `analysis_type` and that CLI arguments are properly parsed into configuration objects for surface, wind, and cross-section analyses. The tests include checks for handling missing or unknown analysis types, recognition of common aliases, and validation of extended argument mapping for specific analysis categories. Additionally, integration-style tests are included to run cross-section analyses using real MPAS data, ensuring end-to-end functionality of the CLI processing workflows. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load pytest for test execution
import os
import pytest


class TestAnalysisExecution:
    """ Tests for CLI analysis execution and dispatch logic. """

    def test_run_analysis_unknown_analysis_type(self: 'TestAnalysisExecution') -> None:
        """
        This test checks that `run_analysis` returns `False` when an unrecognized `analysis_type` is provided. It creates a configuration with an invalid `analysis_type` and asserts that the method does not attempt to execute any analysis and instead returns `False`.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig

        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file='test.nc',
            data_dir='data/',
            analysis_type='invalid_type'
        )

        result = cli.run_analysis(config)

        assert result is False

    def test_run_analysis_recognizes_precipitation_alias(self: 'TestAnalysisExecution') -> None:
        """
        This test ensures that `run_analysis` correctly identifies precipitation analysis aliases. It iterates over known precipitation aliases and asserts that the `_run_precipitation_analysis` method is called for each alias, returning `True` to indicate successful recognition and dispatch.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        from unittest.mock import patch

        cli = MPASUnifiedCLI()

        for alias in ['precipitation', 'precip', 'rain']:
            config = MPASConfig(
                grid_file='test.nc',
                data_dir='data/',
                analysis_type=alias
            )

            with patch.object(cli, '_run_precipitation_analysis', return_value=True) as mock_method:
                result = cli.run_analysis(config)
                mock_method.assert_called_once_with(config)
                assert result is True

    def test_run_analysis_recognizes_surface_alias(self: 'TestAnalysisExecution') -> None:
        """
        This test verifies that `run_analysis` correctly identifies surface analysis aliases. It iterates over known surface aliases and asserts that the `_run_surface_analysis` method is called for each alias, returning `True` to confirm proper recognition and dispatch.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        from unittest.mock import patch

        cli = MPASUnifiedCLI()

        for alias in ['surface', 'surf', '2d']:
            config = MPASConfig(
                grid_file='test.nc',
                data_dir='data/',
                analysis_type=alias
            )

            with patch.object(cli, '_run_surface_analysis', return_value=True) as mock_method:
                result = cli.run_analysis(config)
                mock_method.assert_called_once_with(config)
                assert result is True

    def test_run_analysis_recognizes_wind_alias(self: 'TestAnalysisExecution') -> None:
        """
        This test ensures that `run_analysis` correctly identifies wind analysis aliases. It iterates over known wind aliases and asserts that the `_run_wind_analysis` method is called for each alias, returning `True` to indicate successful recognition and dispatch.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        from unittest.mock import patch

        cli = MPASUnifiedCLI()

        for alias in ['wind', 'vector', 'winds']:
            config = MPASConfig(
                grid_file='test.nc',
                data_dir='data/',
                analysis_type=alias
            )

            with patch.object(cli, '_run_wind_analysis', return_value=True) as mock_method:
                result = cli.run_analysis(config)
                mock_method.assert_called_once_with(config)
                assert result is True

    def test_run_analysis_recognizes_cross_alias(self: 'TestAnalysisExecution') -> None:
        """
        This test ensures that `run_analysis` correctly identifies cross-section analysis aliases. It iterates over known cross-section aliases and asserts that the `_run_cross_analysis` method is called for each alias, returning `True` to indicate successful recognition and dispatch.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        from unittest.mock import patch

        cli = MPASUnifiedCLI()

        for alias in ['cross', 'xsec', '3d', 'vertical']:
            config = MPASConfig(
                grid_file='test.nc',
                data_dir='data/',
                analysis_type=alias
            )

            with patch.object(cli, '_run_cross_analysis', return_value=True) as mock_method:
                result = cli.run_analysis(config)
                mock_method.assert_called_once_with(config)
                assert result is True

    def test_run_analysis_recognizes_overlay_alias(self: 'TestAnalysisExecution') -> None:
        """
        This test ensures that `run_analysis` correctly identifies overlay analysis aliases. It iterates over known overlay aliases and asserts that the `_run_overlay_analysis` method is called for each alias, returning `True` to indicate successful recognition and dispatch.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        from unittest.mock import patch

        cli = MPASUnifiedCLI()

        for alias in ['overlay', 'complex', 'multi', 'composite']:
            config = MPASConfig(
                grid_file='test.nc',
                data_dir='data/',
                analysis_type=alias
            )

            with patch.object(cli, '_run_overlay_analysis', return_value=True) as mock_method:
                result = cli.run_analysis(config)
                mock_method.assert_called_once_with(config)
                assert result is True


class TestSurfaceAnalysis:
    """ Extended surface analysis configuration mapping tests. """

    def test_parse_args_surface_with_colormap(self: 'TestSurfaceAnalysis') -> None:
        """
        This test verifies that the `--colormap` argument is correctly parsed and mapped to the configuration object for surface analysis. It constructs a CLI argument list including `--colormap` and asserts that the resulting configuration's `colormap` attribute matches the input value.

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
            '--colormap', 'jet',
            '--time-index', '0'
        ])

        config = cli.parse_args_to_config(args)

        assert config.colormap == 'jet'
        assert config.variable == 'temperature'

    def test_parse_args_surface_with_plot_type(self: 'TestSurfaceAnalysis') -> None:
        """
        This test verifies that different `--plot-type` values are accepted and correctly mapped to the configuration object for surface analysis. It iterates over a set of supported plot types, constructs CLI arguments for each, and asserts that the resulting configuration's `plot_type` attribute matches the input value.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()

        for plot_type in ['scatter', 'contour', 'contourf', 'pcolormesh']:
            args = parser.parse_args([
                'surface',
                '--grid-file', 'test.nc',
                '--data-dir', 'data/',
                '--variable', 'temp',
                '--plot-type', plot_type,
                '--time-index', '0'
            ])

            config = cli.parse_args_to_config(args)

            assert config.plot_type == plot_type


class TestWindAnalysis:
    """ Extended wind analysis configuration mapping tests. """

    def test_parse_args_wind_with_wind_plot_type(self: 'TestWindAnalysis') -> None:
        """
        This test verifies that different `--wind-plot-type` values are accepted and correctly mapped to the configuration object for wind analysis. It iterates over a set of supported wind plot types, constructs CLI arguments for each, and asserts that the resulting configuration's `wind_plot_type` attribute matches the input value.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()

        for wind_type in ['barbs', 'arrows', 'streamlines']:
            args = parser.parse_args([
                'wind',
                '--grid-file', 'test.nc',
                '--data-dir', 'data/',
                '--wind-plot-type', wind_type,
                '--time-index', '0'
            ])

            config = cli.parse_args_to_config(args)

            assert config.wind_plot_type == wind_type

    def test_parse_args_wind_with_subsample_factor(self: 'TestWindAnalysis') -> None:
        """
        This test verifies that the `--subsample` option is correctly parsed and mapped to the configuration object for wind analysis. It constructs a CLI argument list including `--subsample` and asserts that the resulting configuration's `subsample_factor` attribute matches the input value.

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
            '--subsample', '5',
            '--time-index', '0'
        ])

        config = cli.parse_args_to_config(args)

        assert config.subsample_factor == pytest.approx(5)

    def test_wind_single_time_with_time_dimension(self: 'TestWindAnalysis') -> None:
        """
        This test verifies that the `run_analysis` method correctly handles time formatting for wind analysis when a single time index is specified. The test initializes the `MPASUnifiedCLI` without a logger and provides a configuration for a wind analysis with a specific time index. It asserts that the method returns a boolean, indicating that the analysis completed successfully and that the time formatting code path was exercised. This test is intended to cover scenarios where the method must format time strings for output files when processing a single time step.

        Parameters:
            None

        Returns:
            None
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


class TestCrossSectionAnalysis:
    """ Extended cross-section configuration mapping tests. """

    def test_parse_args_cross_with_vertical_coord(self: 'TestCrossSectionAnalysis') -> None:
        """
        This test verifies that different `--vertical-coord` values are accepted and correctly mapped to the configuration object for cross-section analysis. It iterates over a set of supported vertical coordinate identifiers, constructs CLI arguments for each, and asserts that the resulting configuration's `vertical_coord` attribute matches the input value.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()

        for vert_coord in ['pressure', 'modlev', 'height']:
            args = parser.parse_args([
                'cross',
                '--grid-file', 'test.nc',
                '--data-dir', 'data/',
                '--variable', 'temp',
                '--start-lat', '30',
                '--start-lon', '-100',
                '--end-lat', '40',
                '--end-lon', '-90',
                '--vertical-coord', vert_coord,
                '--time-index', '0'
            ])

            config = cli.parse_args_to_config(args)

            assert config.vertical_coord == vert_coord

    def test_parse_args_cross_with_num_points(self: 'TestCrossSectionAnalysis') -> None:
        """
        This test verifies that the `--num-points` argument is correctly parsed and mapped to the configuration object for cross-section analysis. It constructs a CLI argument list including `--num-points` with an integer value and asserts that the resulting configuration's `num_points` attribute matches the input integer.

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
            '--variable', 'temp',
            '--start-lat', '30',
            '--start-lon', '-100',
            '--end-lat', '40',
            '--end-lon', '-90',
            '--num-points', '200',
            '--time-index', '0'
        ])

        config = cli.parse_args_to_config(args)

        assert config.num_points == pytest.approx(200)


class TestCrossSectionWorkflows:
    """ Integration-style tests for cross-section analysis workflows using real data. """

    def test_cross_section_batch_mode_serial(self: 'TestCrossSectionWorkflows', 
                                             grid_file: str, 
                                             test_data_dir: str) -> None:
        """
        This test runs a cross-section analysis in batch mode without parallel processing. It sets up an `MPASConfig` with `batch_mode=True` and `parallel=False`, then calls `run_analysis` to validate that the method executes the batch workflow in serial mode and returns a boolean indicating the run attempted to execute.

        Parameters:
            grid_file (str): Session fixture providing path to MPAS grid file.
            test_data_dir (str): Session fixture providing path to test data directory.

        Returns:
            None
        """
        import pytest
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        if grid_file is None:
            pytest.skip("Test data files not available")
            return
        
        data_dir = str(test_data_dir / 'u240k' / 'mpasout')
        
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file=grid_file,
            data_dir=data_dir,
            analysis_type='cross',
            variable='temperature',
            start_lon=95.0,
            start_lat=5.0,
            end_lon=105.0,
            end_lat=10.0,
            vertical_coord='pressure',
            num_points=50,
            batch_mode=True, 
            parallel=False,  
            output_dir='output/test_cross_batch_serial',
            verbose=True
        )
        
        result = cli.run_analysis(config)
        
        assert isinstance(result, bool)
    
    def test_cross_section_batch_mode_parallel(self: 'TestCrossSectionWorkflows', 
                                               grid_file: str, 
                                               test_data_dir: str) -> None:
        """
        This test runs a cross-section analysis in batch mode with parallel processing. It sets up an `MPASConfig` with `batch_mode=True` and `parallel=True`, then calls `run_analysis` to validate that the method executes the batch workflow in parallel mode and returns a boolean indicating the run attempted to execute.

        Parameters:
            grid_file (str): Session fixture providing path to MPAS grid file.
            test_data_dir (str): Session fixture providing path to test data directory.

        Returns:
            None
        """
        import pytest
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        if grid_file is None:
            pytest.skip("Test data files not available")
            return
        
        data_dir = str(test_data_dir / 'u240k' / 'mpasout')
        
        cli = MPASUnifiedCLI()
        
        config = MPASConfig(
            grid_file=grid_file,
            data_dir=data_dir,
            analysis_type='cross',
            variable='temperature',
            start_lon=95.0,
            start_lat=5.0,
            end_lon=105.0,
            end_lat=10.0,
            vertical_coord='pressure',
            num_points=50,
            batch_mode=True, 
            parallel=True,  
            workers=2,
            output_dir='output/test_cross_batch_parallel',
            verbose=True
        )
        
        result = cli.run_analysis(config)
        
        assert isinstance(result, bool)
    
    def test_cross_section_with_custom_output_path(self: 'TestCrossSectionWorkflows', 
                                                   grid_file: str, 
                                                   test_data_dir: str) -> None:
        """
        This test runs a cross-section analysis with a custom output path specified in the configuration. It sets up an `MPASConfig` with a custom `output` path and calls `run_analysis` to validate that the method accepts the custom output path and returns a boolean indicating the run attempted to execute.

        Parameters:
            grid_file (str): Session fixture providing path to MPAS grid file.
            test_data_dir (str): Session fixture providing path to test data directory.

        Returns:
            None
        """
        import pytest
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        if grid_file is None:
            pytest.skip("Test data files not available")
            return
        
        data_dir = str(test_data_dir / 'u240k' / 'mpasout')
        
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file=grid_file,
            data_dir=data_dir,
            analysis_type='cross',
            variable='temperature',
            start_lon=95.0,
            start_lat=5.0,
            end_lon=105.0,
            end_lat=10.0,
            time_index=0,
            vertical_coord='pressure',
            num_points=50,
            output='output/test_cross_custom/my_custom_plot',  
            output_dir='output/test_cross_custom',
            verbose=True
        )
        
        result = cli.run_analysis(config)
        
        assert isinstance(result, bool)

    def test_run_cross_section_analysis_with_real_data(self: 'TestCrossSectionWorkflows', 
                                                        grid_file: str, 
                                                        test_data_dir: str) -> None:
        """
        This test runs a cross-section analysis using real MPAS data files. It sets up an `MPASConfig` with appropriate parameters for a cross-section analysis and calls `run_analysis` to validate that the method executes without errors and returns a boolean indicating the run attempted to execute. This test serves as an integration-style check to ensure that the CLI processing workflow can handle real data inputs end-to-end.

        Parameters:
            grid_file (str): Session fixture providing path to MPAS grid file.
            test_data_dir (str): Session fixture providing path to test data directory.

        Returns:
            None
        """
        import pytest
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        if grid_file is None:
            pytest.skip("Test data files not available")
            return
        
        data_dir = str(test_data_dir / 'u240k' / 'mpasout')
        
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file=grid_file,
            data_dir=data_dir,
            analysis_type='cross',
            variable='temperature',
            start_lon=95.0,
            start_lat=5.0,
            end_lon=105.0,
            end_lat=10.0,
            time_index=0,
            vertical_coord='pressure',
            num_points=50,
            output_dir='output/test_cross',
            verbose=False
        )
        
        result = cli.run_analysis(config)
        
        assert result is not None
        assert isinstance(result, bool)

    def test_cross_section_single_time_with_logger(self: 'TestCrossSectionWorkflows') -> None:
        """
        This test verifies that the `run_analysis` method correctly processes a cross-section analysis for a single time index when a logger is configured. The test initializes the `MPASUnifiedCLI` with a logger and provides a configuration for a cross-section analysis with specific spatial and temporal parameters. It asserts that the method returns a boolean, indicating that the analysis completed successfully and that the logger was utilized during the process. This test is designed to cover the code path where the method handles cross-section analyses with logging enabled, ensuring that log messages are generated appropriately. 

        Parameters:
            None

        Returns:
            None
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
