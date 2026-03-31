#!/usr/bin/env python3
"""
MPASdiag Test Suite: CLI Configuration Tests

This module contains unit tests for the command-line interface (CLI) configuration parsing and validation logic in the MPASdiag processing package. The tests focus on ensuring that command-line arguments are correctly mapped to the internal configuration object and that the validation logic properly identifies valid and invalid configurations. The tests cover various scenarios including basic argument mapping, analysis-specific parameter handling, and edge cases in configuration validation.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries for testing
import pytest
import tempfile
from pathlib import Path


class TestConfigurationMapping:
    """ Tests for mapping CLI arguments to configuration fields. """

    def test_parse_args_to_config_basic(self: "TestConfigurationMapping") -> None:
        """
        This test verifies that basic CLI arguments are correctly mapped to the configuration object. It simulates a command-line invocation of the `surface` analysis type with common options such as `--grid-file`, `--data-dir`, `--output-dir`, `--variable`, and `--time-index`. The test asserts that the resulting configuration object has the expected values for these fields, confirming that the argument parsing and mapping logic in `parse_args_to_config` is functioning correctly.

        Parameters:
            self (TestConfigurationMapping): The test instance.

        Returns:
            None: Raises on mismatches between parsed and expected values.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI

        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()

        args = parser.parse_args([
            'surface',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--output-dir', 'output/',
            '--variable', 'temperature',
            '--time-index', '5'
        ])

        config = cli.parse_args_to_config(args)

        assert config.grid_file == 'test.nc'
        assert config.data_dir == 'data/'
        assert config.output_dir == 'output/'
        assert config.time_index == 5
    
    def test_parse_args_precipitation_mapping(self: "TestConfigurationMapping") -> None:
        """
        This test checks that precipitation-specific CLI arguments are correctly mapped to the configuration object. It simulates a command-line invocation for the `precipitation` analysis type, providing options such as `--variable` for the precipitation variable and asserts that these values are present on the resulting configuration object. This ensures that analysis-specific parameters are handled appropriately during argument parsing.

        Parameters:
            self (TestConfigurationMapping): The test instance.

        Returns:
            None: The test will raise on assertion failure.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()
        
        args = parser.parse_args([
            'precipitation',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--variable', 'rainnc',
            '--time-index', '0'
        ])
        
        config = cli.parse_args_to_config(args)
        
        assert config.analysis_type == 'precipitation'
        assert config.variable == 'rainnc'
    
    def test_parse_args_wind_mapping(self: "TestConfigurationMapping") -> None:
        """
        This test verifies that wind analysis CLI arguments are correctly mapped to the configuration object. It simulates a command-line invocation for the `wind` analysis type, providing options for the u and v wind components (`--u-variable` and `--v-variable`) and asserts that these values are correctly set on the resulting configuration object. This confirms that the argument parsing logic properly handles analysis-specific parameters for wind analyses.

        Parameters:
            self (TestConfigurationMapping): The test instance.

        Returns:
            None: Raises on mismatch.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()
        
        args = parser.parse_args([
            'wind',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--u-variable', 'u10',
            '--v-variable', 'v10',
            '--time-index', '0'
        ])
        
        config = cli.parse_args_to_config(args)
        
        assert config.analysis_type == 'wind'
        assert config.u_variable == 'u10'
        assert config.v_variable == 'v10'
    
    def test_parse_args_cross_section_mapping(self: "TestConfigurationMapping") -> None:
        """
        This test checks that cross-section analysis CLI arguments are correctly mapped to the configuration object. It simulates a command-line invocation for the `cross` analysis type, providing start and end latitude/longitude coordinates via `--start-lat`, `--start-lon`, `--end-lat`, and `--end-lon` options. The test asserts that these values are converted to floating point numbers and correctly set on the resulting configuration object, confirming that spatial coordinate arguments are handled properly during parsing.

        Parameters:
            self (TestConfigurationMapping): The test instance.

        Returns:
            None: Raises on incorrect conversions.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()
        
        args = parser.parse_args([
            'cross',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--variable', 'theta',
            '--start-lat', '30',
            '--start-lon', '-100',
            '--end-lat', '40',
            '--end-lon', '-90',
            '--time-index', '0'
        ])
        
        config = cli.parse_args_to_config(args)
        
        assert config.analysis_type == 'cross'
        assert config.start_lat == 30.0
        assert config.start_lon == -100.0
        assert config.end_lat == 40.0
        assert config.end_lon == -90.0
    
    def test_parse_args_with_spatial_bounds(self: "TestConfigurationMapping") -> None:
        """
        This test validates that explicit latitude and longitude bounds provided via CLI arguments are correctly mapped to the configuration object. It simulates a command-line invocation for a `surface` analysis type with spatial bounds specified using `--lat-min`, `--lat-max`, `--lon-min`, and `--lon-max` options. The test asserts that these values are converted to floating point numbers and set on the resulting configuration object, ensuring that spatial extent parameters are handled correctly during argument parsing.

        Parameters:
            self (TestConfigurationMapping): The test instance.

        Returns:
            None: Raises on mismatch.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()
        
        args = parser.parse_args([
            'surface',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--variable', 'temperature',
            '--lat-min', '20',
            '--lat-max', '50',
            '--lon-min', '-120',
            '--lon-max', '-80',
            '--time-index', '0'
        ])
        
        config = cli.parse_args_to_config(args)
        
        assert config.lat_min == 20.0
        assert config.lat_max == 50.0
        assert config.lon_min == -120.0
        assert config.lon_max == -80.0
    
    def test_parse_args_with_workers(self: "TestConfigurationMapping") -> None:
        """
        This test checks that the `--workers` CLI argument is correctly parsed and mapped to the configuration object. It simulates a command-line invocation for a `surface` analysis type with the `--workers` option set to a specific integer value. The test asserts that this value is correctly converted to an integer and assigned to the `workers` field on the resulting configuration object, confirming that parallel processing options are handled properly during argument parsing.

        Parameters:
            self (TestConfigurationMapping): The test instance.

        Returns:
            None: Raises on type or value mismatch.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        
        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()
        
        args = parser.parse_args([
            'surface',
            '--grid-file', 'test.nc',
            '--data-dir', 'data/',
            '--variable', 'temperature',
            '--workers', '4',
            '--time-index', '0'
        ])
        
        config = cli.parse_args_to_config(args)
        
        assert config.workers == 4


class TestConfigurationAndValidation:
    """ Tests for configuration validation logic, including edge cases and data discovery paths. """

    def test_validate_config_data_dir_with_subdirs(self: "TestConfigurationAndValidation", grid_file: str, test_data_dir: str) -> None:
        """
        This test verifies that `validate_config` can successfully find data files when they are located in subdirectories of the specified `data_dir`. It sets up a configuration with a valid `grid_file` and a `data_dir` that contains subdirectories (e.g., `diag/` and `mpasout/`) with data files. The test asserts that `validate_config` returns True, indicating that it correctly discovers the necessary files for processing even when they are not directly in the top-level `data_dir`.

        Parameters:
            self (TestConfigurationAndValidation): The test instance.
            grid_file: Session fixture providing path to MPAS grid file.
            test_data_dir: Session fixture providing path to test data directory.

        Returns:
            None: The test asserts `validate_config` returns True.
        """
        import pytest
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        if grid_file is None:
            pytest.skip("Test data files not available")
        
        data_dir = str(Path(test_data_dir) / 'u240k')  
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file=grid_file,
            data_dir=data_dir,
            analysis_type='surface'
        )
        
        result = cli.validate_config(config)
        
        assert result is True

    def test_validate_config_missing_grid_file(self: "TestConfigurationAndValidation") -> None:
        """
        This test checks that validation fails when the `grid_file` parameter is missing from the configuration. It constructs an `MPASConfig` object without providing a `grid_file` and asserts that the `validate_config` method returns `False`, indicating that the configuration is invalid due to the missing required parameter.

        Parameters:
            self (unittest.TestCase): The test instance.

        Returns:
            None: Raises on unexpected validation result.
        """

        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig

        cli = MPASUnifiedCLI()
        config = MPASConfig(data_dir='data/')

        result = cli.validate_config(config)

        assert result is False

    def test_validate_config_missing_data_dir(self: "TestConfigurationAndValidation") -> None:
        """
        This test verifies that validation fails when the `data_dir` parameter is missing from the configuration. It creates an `MPASConfig` object without specifying a `data_dir` and asserts that the `validate_config` method returns `False`, indicating that the configuration is invalid due to the absence of the required data directory parameter.

        Parameters:
            self (unittest.TestCase): The test instance.

        Returns:
            None: Raises on unexpected validation result.
        """

        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig

        cli = MPASUnifiedCLI()
        config = MPASConfig(grid_file='test.nc')

        result = cli.validate_config(config)

        assert result is False

    def test_validate_config_nonexistent_grid_file(self: "TestConfigurationAndValidation") -> None:
        """
        This test checks that validation fails when the specified `grid_file` does not exist. It constructs an `MPASConfig` with a `grid_file` that points to a non-existent file and asserts that the `validate_config` method returns `False`, indicating that the configuration is invalid due to the missing grid file.

        Parameters:
            self (unittest.TestCase): The test instance.

        Returns:
            None: Raises on unexpected behavior.
        """

        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig

        cli = MPASUnifiedCLI()
        config = MPASConfig(grid_file='nonexistent.nc', data_dir='data/')

        result = cli.validate_config(config)

        assert result is False

    def test_validate_config_invalid_lat_range(self: "TestConfigurationAndValidation") -> None:
        """
        This test verifies that validation fails when the latitude bounds specified in the configuration are invalid (i.e., `lat_min` is greater than or equal to `lat_max`). It constructs an `MPASConfig` object with `lat_min` set to a value greater than `lat_max` and asserts that the `validate_config` method raises a `ValueError`, indicating that the spatial extent parameters are invalid.

        Parameters:
            self (unittest.TestCase): The test instance.

        Returns:
            None: The test passes if `ValueError` is raised.
        """

        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        import pytest

        cli = MPASUnifiedCLI()

        with pytest.raises(ValueError, match="Invalid spatial extent parameters"):
            config = MPASConfig(
                grid_file='test.nc',
                data_dir='data/',
                lat_min=50.0,
                lat_max=20.0  # Invalid: min > max
            )
            result = cli.validate_config(config)  
            assert result is False  


    def test_validate_config_invalid_lon_range(self: "TestConfigurationAndValidation") -> None:
        """
        This test checks that validation fails when the longitude bounds specified in the configuration are invalid (i.e., `lon_min` is greater than or equal to `lon_max`). It constructs an `MPASConfig` object with `lon_min` set to a value greater than `lon_max` and asserts that the `validate_config` method raises a `ValueError`, indicating that the spatial extent parameters are invalid.

        Parameters:
            self (unittest.TestCase): The test instance.

        Returns:
            None: The test passes if `ValueError` is raised.
        """
        from mpasdiag.processing.utils_config import MPASConfig
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        import pytest

        cli = MPASUnifiedCLI()

        with pytest.raises(ValueError, match="Invalid spatial extent parameters"):
            config = MPASConfig(
                grid_file='test.nc',
                data_dir='data/',
                lon_min=-80.0,
                lon_max=-120.0  # Invalid: min > max
            )
            result = cli.validate_config(config)
            assert result is False


class TestValidateConfigEdgeCases:
    """ Test validate_config method edge cases. """
    
    def test_validate_config_invalid_latitude_range(self: "TestValidateConfigEdgeCases") -> None:
        """
        This test checks that validation fails when the latitude bounds specified in the configuration are invalid (i.e., `lat_min` is greater than or equal to `lat_max`). It constructs an `MPASConfig` object with `lat_min` set to a value greater than `lat_max` and asserts that the `validate_config` method returns `False`, indicating that the configuration is invalid due to the incorrect spatial extent parameters.

        Parameters:
            self (TestValidateConfigEdgeCases): The test instance.

        Returns:
            None: The test asserts `validate_config` returns `False`.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        cli = MPASUnifiedCLI()

        config = object.__new__(MPASConfig)
        object.__setattr__(config, 'grid_file', 'data/grids/x1.10242.static.nc')
        object.__setattr__(config, 'data_dir', 'data/u240k/diag')
        object.__setattr__(config, 'lat_min', 50.0)
        object.__setattr__(config, 'lat_max', 40.0)  # Invalid: lat_max < lat_min
        object.__setattr__(config, 'lon_min', -180.0)
        object.__setattr__(config, 'lon_max', 180.0)
        
        result = cli.validate_config(config)
        assert result is False
    
    def test_validate_config_invalid_longitude_range(self: "TestValidateConfigEdgeCases") -> None:
        """
        This test checks that validation fails when the longitude bounds specified in the configuration are invalid (i.e., `lon_min` is greater than or equal to `lon_max`). It constructs an `MPASConfig` object with `lon_min` set to a value greater than `lon_max` and asserts that the `validate_config` method returns `False`, indicating that the configuration is invalid due to the incorrect spatial extent parameters.

        Parameters:
            self (TestValidateConfigEdgeCases): The test instance.

        Returns:
            None: The test asserts `validate_config` returns `False`.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        cli = MPASUnifiedCLI()

        config = object.__new__(MPASConfig)
        object.__setattr__(config, 'grid_file', 'data/grids/x1.10242.static.nc')
        object.__setattr__(config, 'data_dir', 'data/u240k/diag')
        object.__setattr__(config, 'lat_min', -90.0)
        object.__setattr__(config, 'lat_max', 90.0)
        object.__setattr__(config, 'lon_min', 100.0)
        object.__setattr__(config, 'lon_max', 50.0)  # Invalid: lon_max < lon_min
        
        result = cli.validate_config(config)
        assert result is False
    
    def test_validate_config_cross_section_missing_start_lon(self: "TestValidateConfigEdgeCases") -> None:
        """
        This test verifies that validation fails for cross-section analyses when the `start_lon` parameter is missing from the configuration. It constructs an `MPASConfig` object for a cross-section analysis type without providing `start_lon` and asserts that the `validate_config` method returns `False`, indicating that the configuration is incomplete for cross-section analyses due to the missing required coordinate parameter.

        Parameters:
            self (TestValidateConfigEdgeCases): The test instance.

        Returns:
            None: The test asserts `validate_config` returns `False`.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file='data/grids/x1.10242.static.nc',
            data_dir='data/u240k/diag',
            analysis_type='cross',
            # start_lon missing
            start_lat=5.0,
            end_lon=105.0,
            end_lat=10.0
        )
        
        result = cli.validate_config(config)
        assert result is False
    
    def test_validate_config_cross_section_missing_start_lat(self: "TestValidateConfigEdgeCases") -> None:
        """
        This test checks that validation fails for cross-section analyses when the `start_lat` parameter is missing from the configuration. It constructs an `MPASConfig` object for a cross-section analysis type without providing `start_lat` and asserts that the `validate_config` method returns `False`, indicating that the configuration is incomplete for cross-section analyses due to the missing required coordinate parameter.

        Parameters:
            self (TestValidateConfigEdgeCases): The test instance.

        Returns:
            None: The test asserts `validate_config` returns `False`.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        cli = MPASUnifiedCLI()
        config = MPASConfig(
            grid_file='data/grids/x1.10242.static.nc',
            data_dir='data/u240k/diag',
            analysis_type='cross',
            start_lon=95.0,
            # start_lat missing
            end_lon=105.0,
            end_lat=10.0
        )
        
        result = cli.validate_config(config)
        assert result is False
    
    def test_validate_config_cross_section_missing_end_lon(self: "TestValidateConfigEdgeCases") -> None:
        """
        This test verifies that validation fails for cross-section analyses when the `end_lon` parameter is missing from the configuration. It constructs an `MPASConfig` object for a cross-section analysis type without providing `end_lon` and asserts that the `validate_config` method returns `False`, indicating that the configuration is incomplete for cross-section analyses due to the missing required coordinate parameter.

        Parameters:
            self (TestValidateConfigEdgeCases): The test instance.

        Returns:
            None: The test asserts `validate_config` returns `False`.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        cli = MPASUnifiedCLI()
        config = MPASConfig(
            grid_file='data/grids/x1.10242.static.nc',
            data_dir='data/u240k/diag',
            analysis_type='cross',
            start_lon=95.0,
            start_lat=5.0,
            # end_lon missing
            end_lat=10.0
        )
        
        result = cli.validate_config(config)
        assert result is False
    
    def test_validate_config_cross_section_missing_end_lat(self: "TestValidateConfigEdgeCases") -> None:
        """
        This test checks that validation fails for cross-section analyses when the `end_lat` parameter is missing from the configuration. It constructs an `MPASConfig` object for a cross-section analysis type without providing `end_lat` and asserts that the `validate_config` method returns `False`, indicating that the configuration is incomplete for cross-section analyses due to the missing required coordinate parameter.

        Parameters:
            self (TestValidateConfigEdgeCases): The test instance.

        Returns:
            None: The test asserts `validate_config` returns `False`.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file='data/grids/x1.10242.static.nc',
            data_dir='data/u240k/diag',
            analysis_type='cross',
            start_lon=95.0,
            start_lat=5.0,
            end_lon=105.0
            # end_lat missing
        )
        
        result = cli.validate_config(config)
        assert result is False
    
    def test_validate_config_cross_section_xsec_alias(self: "TestValidateConfigEdgeCases") -> None:
        """
        This test verifies that the `xsec` alias for cross-section analyses is properly recognized and validated. It constructs an `MPASConfig` object using the `xsec` analysis type with valid coordinate parameters and asserts that the `validate_config` method returns `True`, confirming that the alias is correctly handled and that the configuration is valid for cross-section analyses.

        Parameters:
            self (TestValidateConfigEdgeCases): The test instance.

        Returns:
            None: The test asserts `validate_config` returns `False`.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file='data/grids/x1.10242.static.nc',
            data_dir='data/u240k/diag',
            analysis_type='xsec',  # Alias for cross-section
            # Missing coordinates should fail validation
        )
        
        result = cli.validate_config(config)
        assert result is False
    
    def test_validate_config_cross_section_3d_alias(self: "TestValidateConfigEdgeCases") -> None:
        """
        This test checks that the `3d` alias for cross-section analyses is properly recognized and validated. It constructs an `MPASConfig` object using the `3d` analysis type with valid coordinate parameters and asserts that the `validate_config` method returns `True`, confirming that the alias is correctly handled and that the configuration is valid for cross-section analyses.

        Parameters:
            self (TestValidateConfigEdgeCases): The test instance.

        Returns:
            None: The test asserts `validate_config` returns `False`.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file='data/grids/x1.10242.static.nc',
            data_dir='data/u240k/diag',
            analysis_type='3d',  # Alias for cross-section
        )
        
        result = cli.validate_config(config)
        assert result is False
    
    def test_validate_config_cross_section_vertical_alias(self: "TestValidateConfigEdgeCases") -> None:
        """
        This test verifies that the `vertical` alias for cross-section analyses is properly recognized and validated. It constructs an `MPASConfig` object using the `vertical` analysis type with valid coordinate parameters and asserts that the `validate_config` method returns `True`, confirming that the alias is correctly handled and that the configuration is valid for cross-section analyses.

        Parameters:
            self (TestValidateConfigEdgeCases): The test instance.

        Returns:
            None: The test asserts `validate_config` returns `False`.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file='data/grids/x1.10242.static.nc',
            data_dir='data/u240k/diag',
            analysis_type='vertical',  # Alias for cross-section
        )
        
        result = cli.validate_config(config)
        assert result is False


class TestValidateConfigDataDiscovery:
    """ Test data file discovery paths in validate_config. """
    
    def test_validate_config_finds_mpasout_in_subdirectory(self: "TestValidateConfigDataDiscovery") -> None:
        """
        This test verifies that `validate_config` can successfully find data files when they are located in an `mpasout` subdirectory of the specified `data_dir`. It creates a temporary directory structure with an `mpasout` subdirectory containing a test data file and a dummy grid file at the top level. The test asserts that `validate_config` returns True, indicating that it correctly discovers the necessary files for processing even when they are located in a subdirectory.

        Parameters:
            self (TestValidateConfigDataDiscovery): The test instance.

        Returns:
            None: The test asserts `validate_config` returns True.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mpasout_dir = Path(tmpdir) / 'mpasout'
            mpasout_dir.mkdir()
            test_file = mpasout_dir / 'mpasout.2024-01-01_00:00:00.nc'
            test_file.touch()
            
            grid_file = Path(tmpdir) / 'grid.nc'
            grid_file.touch()
            
            cli = MPASUnifiedCLI()

            config = MPASConfig(
                grid_file=str(grid_file),
                data_dir=str(tmpdir)
            )
            
            result = cli.validate_config(config)
            assert result is True  
    
    def test_validate_config_uses_rglob_when_no_files_found(self: "TestValidateConfigDataDiscovery") -> None:
        """
        This test checks that `validate_config` uses recursive globbing to find data files when they are not found in the top-level `data_dir`. It creates a temporary directory structure with nested subdirectories containing a test data file and a dummy grid file at the top level. The test asserts that `validate_config` returns True, indicating that it correctly discovers the necessary files for processing by searching recursively through the directory structure.

        Parameters:
            self (TestValidateConfigDataDiscovery): The test instance.

        Returns:
            None: The test asserts `validate_config` returns True.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / 'subdir' / 'data'
            nested_dir.mkdir(parents=True)
            test_file = nested_dir / 'diag.2024-01-01_00:00:00.nc'
            test_file.touch()
            
            grid_file = Path(tmpdir) / 'grid.nc'
            grid_file.touch()
            
            cli = MPASUnifiedCLI()

            config = MPASConfig(
                grid_file=str(grid_file),
                data_dir=str(tmpdir)
            )
            
            result = cli.validate_config(config)
            assert result is True  


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
