#!/usr/bin/env python3
"""
MPASdiag Test Suite: CLI Integration Tests

This module contains integration tests for the MPASdiag CLI unified interface, verifying that the command-line interface can successfully execute end-to-end workflows using real MPAS diagnostic files. The tests cover various analysis types (precipitation, surface, wind) and ensure that the CLI can parse arguments, dispatch to the correct analysis functions, and complete without errors. Sample data files are required for these tests, and they will be skipped if the necessary files are not available on the test host. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load standard libraries
import sys
import pytest
from pathlib import Path


class TestCLIIntegration:
    """ Integration tests for the MPASdiag CLI unified interface. """

    def test_help_exits_cleanly(self: 'TestCLIIntegration') -> None:
        """
        This test checks that invoking the CLI with the `--help` flag results in a clean exit with a zero exit code. It uses `unittest.mock.patch` to temporarily set `sys.argv` to simulate a command-line invocation of the help flag. The test then asserts that a `SystemExit` exception is raised and that the exit code is zero, which indicates that the help message was displayed successfully without errors.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()

        with pytest.raises(SystemExit) as exc_info:
            cli.create_main_parser().parse_args(['--help'])

        assert exc_info.value.code == pytest.approx(0)


class TestIntegrationWithRealData:
    """ Integration tests using actual MPAS data files. """
    
    def test_run_precipitation_analysis_with_real_data(self: 'TestIntegrationWithRealData', 
                                                       grid_file: str, 
                                                       test_data_dir: str) -> None:
        """
        This test executes the precipitation analysis workflow using real MPAS diagnostic files. It constructs a `MPASConfig` with parameters for a precipitation analysis, including the grid file and data directory provided by fixtures. The test then calls the CLI's `run_analysis` method and asserts that it returns `True`, indicating successful completion of the workflow. If the required sample data files are not available, the test will be skipped to avoid false failures on machines without access to the dataset.

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
        
        data_dir = str(Path(test_data_dir) / 'u240k' / 'diag')
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file=grid_file,
            data_dir=data_dir,
            analysis_type='precipitation',
            variable='rainnc',
            accumulation_period='a01h',
            time_index=0,
            output_dir='output/test_precip',
            verbose=False
        )
        
        result = cli.run_analysis(config)
        assert result is True
    
    def test_run_surface_analysis_with_real_data(self: 'TestIntegrationWithRealData', 
                                                 grid_file: str, 
                                                 test_data_dir: str) -> None:
        """
        This test validates the surface analysis workflow using real MPAS diagnostic files. It sets up an `MPASConfig` for a surface analysis of the 2-meter temperature variable, specifying the grid file and data directory from fixtures. The test invokes the CLI's `run_analysis` method and asserts that it returns `True`, indicating that the analysis completed successfully. If the necessary sample data files are not present, the test will be skipped to maintain reliability across different testing environments.

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
        
        data_dir = str(Path(test_data_dir) / 'u240k' / 'diag')
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file=grid_file,
            data_dir=data_dir,
            analysis_type='surface',
            variable='t2m',
            plot_type='scatter',
            time_index=0,
            output_dir='output/test_surface',
            verbose=False
        )
        
        result = cli.run_analysis(config)
        assert result is True
    
    def test_run_wind_analysis_with_real_data(self: 'TestIntegrationWithRealData', 
                                              grid_file: str, 
                                              test_data_dir: str) -> None:
        """
        This test checks the wind analysis workflow using real MPAS diagnostic files. It configures an `MPASConfig` for a wind analysis of 10-meter u and v components, including subsampling and output directory settings. The test calls the CLI's `run_analysis` method and asserts that it returns `True`, indicating that the wind analysis completed successfully. If the required sample data files are not available, the test will be skipped to prevent false negatives in environments without access to the dataset.

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
        
        data_dir = str(Path(test_data_dir) / 'u240k' / 'diag')
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file=grid_file,
            data_dir=data_dir,
            analysis_type='wind',
            u_variable='u10',
            v_variable='v10',
            wind_plot_type='barbs',
            subsample_factor=5,
            time_index=0,
            output_dir='output/test_wind',
            verbose=False
        )
        
        result = cli.run_analysis(config)
        assert result is True
    
    def test_run_precipitation_analysis_batch_mode(self: 'TestIntegrationWithRealData', 
                                                   grid_file: str, 
                                                   test_data_dir: str) -> None:
        """
        This test exercises the batch-mode control flow of the precipitation analysis, ensuring that multiple files or time indices are processed without error. It uses the same sample-data guard as other integration tests and will be skipped if required files are not found on disk. The test constructs a configuration for batch processing and asserts that the `run_analysis` method returns `True`, indicating successful completion of the batch workflow.

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
        
        data_dir = str(Path(test_data_dir) / 'u240k' / 'diag')
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file=grid_file,
            data_dir=data_dir,
            analysis_type='precipitation',
            variable='rainnc',
            accumulation_period='a01h',
            batch_mode=True,
            output_dir='output/test_precip_batch',
            verbose=False
        )
        
        result = cli.run_analysis(config)
        assert result is True
    
    def test_run_surface_analysis_with_spatial_bounds(self: 'TestIntegrationWithRealData', 
                                                      grid_file: str, 
                                                      test_data_dir: str) -> None:
        """
        This test verifies that the surface analysis correctly accepts and applies spatial bounds (latitude and longitude minima/maxima) when constructing plots and subsetting data. It uses sample MPAS files and will be skipped when those files are not available in the test host environment to avoid spurious failures. The test configures a surface analysis with specified spatial bounds and asserts that the `run_analysis` method returns `True`, indicating successful completion of the analysis with spatial constraints.

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
        
        data_dir = str(Path(test_data_dir) / 'u240k' / 'diag')
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file=grid_file,
            data_dir=data_dir,
            analysis_type='surface',
            variable='t2m',
            plot_type='scatter',
            time_index=0,
            lat_min=0.0,
            lat_max=10.0,
            lon_min=95.0,
            lon_max=105.0,
            output_dir='output/test_surface_bounds',
            verbose=False
        )
        
        result = cli.run_analysis(config)
        assert result is True
    
    def test_main_function_with_precipitation_args(self: 'TestIntegrationWithRealData', 
                                                   grid_file: str, 
                                                   test_data_dir: str) -> None:
        """
        This test simulates a CLI invocation by setting `sys.argv` to a precipitation command with required options and asserts that the top-level `main()` returns a zero exit code on success. It is useful for verifying argument parsing, dispatch, and exit-code conventions when running the CLI as a process would normally do. The test will be skipped if sample data files required for the run are not present in the test environment.

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
        
        data_dir = str(Path(test_data_dir) / 'u240k' / 'diag')
        original_argv = sys.argv
        
        try:
            sys.argv = [
                'mpasdiag',
                'precipitation',
                '--grid-file', grid_file,
                '--data-dir', data_dir,
                '--time-index', '0',
                '--output-dir', 'output/test_main_precip',
                '--quiet'
            ]
            
            cli = MPASUnifiedCLI()
            result = cli.main()
            
            assert result == pytest.approx(0)
            
        finally:
            sys.argv = original_argv
    
    def test_main_function_with_surface_args(self: 'TestIntegrationWithRealData', 
                                             grid_file: str, 
                                             test_data_dir: str) -> None:
        """
        This test validates that `MPASUnifiedCLI.main()` can execute a surface analysis invocation. By manipulating `sys.argv` to represent a surface subcommand call, the test checks that the CLI can parse surface-specific options and complete successfully, returning the conventional zero exit code for success. The test will be skipped if sample data files required for the run are not present in the test environment.

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
        
        data_dir = str(Path(test_data_dir) / 'u240k' / 'diag')        
        original_argv = sys.argv
        
        try:
            sys.argv = [
                'mpasdiag',
                'surface',
                '--grid-file', grid_file,
                '--data-dir', data_dir,
                '--variable', 't2m',
                '--time-index', '0',
                '--output-dir', 'output/test_main_surface',
                '--quiet'
            ]
            
            cli = MPASUnifiedCLI()
            result = cli.main()
            
            assert result == pytest.approx(0)
            
        finally:
            sys.argv = original_argv
    
    def test_main_function_with_wind_args(self: 'TestIntegrationWithRealData', 
                                          grid_file: str, 
                                          test_data_dir: str) -> None:
        """
        This test confirms that `MPASUnifiedCLI.main()` can handle a wind subcommand invocation. This test sets `sys.argv` to emulate a wind-analysis CLI call, including options such as subsampling and output directory, and asserts that the `main()` method completes with a zero return code. It verifies the integration of parsing, dispatch, and underlying wind-analysis logic under representative parameters. The test will be skipped if sample data files required for the run are not present in the test environment.

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
        
        data_dir = str(Path(test_data_dir) / 'u240k' / 'diag')        
        original_argv = sys.argv
        
        try:
            sys.argv = [
                'mpasdiag',
                'wind',
                '--grid-file', grid_file,
                '--data-dir', data_dir,
                '--time-index', '0',
                '--subsample', '5',
                '--output-dir', 'output/test_main_wind',
                '--quiet'
            ]
            
            cli = MPASUnifiedCLI()
            result = cli.main()            
            assert result == pytest.approx(0)
            
        finally:
            sys.argv = original_argv


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
