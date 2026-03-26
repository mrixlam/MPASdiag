#!/usr/bin/env python3

"""
MPASdiag Test Suite: Tests for ArgumentParser Factory and Config Conversion

This module contains unit tests for the `ArgumentParser` class in the `mpasdiag.processing.utils_parser` module. The tests cover the creation of argument parsers for general, surface, wind, and cross-section plotting, as well as the conversion of parsed arguments into `MPASConfig` instances. The tests ensure that the parsers are correctly configured to handle expected command-line arguments and that the conversion logic properly maps parsed values onto the configuration objects used by the plotting functions. Each test is designed to validate specific functionality and edge cases to ensure the robustness of the argument parsing and configuration conversion processes. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules for testing
import pytest
import argparse
from mpasdiag.processing.utils_parser import ArgumentParser
from mpasdiag.processing.utils_config import MPASConfig


class TestCreateParser:
    """ Test general parser creation. """
    
    def test_create_parser_returns_argument_parser(self: "TestCreateParser") -> None:
        """
        This test verifies that the `create_parser` factory method returns an instance of `argparse.ArgumentParser`. The returned parser should be properly initialized and ready to accept command-line arguments for the general plotting workflow. This test confirms the type of the returned object to ensure it is a valid argument parser. 

        Parameters:
            None

        Returns:
            None: Validated through isinstance/assert checks.
        """
        parser = ArgumentParser.create_parser()        
        assert isinstance(parser, argparse.ArgumentParser)
        assert parser is not None
    
    def test_create_parser_has_required_arguments(self: "TestCreateParser") -> None:
        """
        This test checks that the parser created by `create_parser` correctly defines required arguments such as `--grid-file` and `--data-dir`. By parsing a minimal set of required arguments, we can confirm that the parser is structured to accept these essential inputs and that they are accessible as attributes on the parsed namespace. This ensures that users will be prompted for necessary information when using the CLI. 

        Parameters:
            None

        Returns:
            None: Confirmed via assertions on parsed namespace attributes.
        """
        parser = ArgumentParser.create_parser()        
        args = parser.parse_args(['--grid-file', 'test.nc', '--data-dir', './data'])        
        assert args.grid_file == 'test.nc'
        assert args.data_dir == './data'


class TestParseArgsToConfig:
    """ Test general argument-to-config conversion. """
    
    def test_parse_args_basic(self: "TestParseArgsToConfig") -> None:
        """
        This test verifies that the `parse_args_to_config` method correctly converts a parsed argument namespace into an `MPASConfig` instance. By providing a comprehensive set of arguments, we can confirm that each relevant field is properly mapped onto the configuration object, including type conversions for numeric values and handling of optional parameters. This test ensures that the conversion logic is robust and produces a configuration object that downstream plotting functions can utilize effectively. 

        Parameters:
            None

        Returns:
            None: Verified via type checks and selected attribute assertions.
        """
        args = argparse.Namespace(
            grid_file='grid.nc',
            data_dir='./data',
            output_dir='./output',
            lat_min=-10.0,
            lat_max=15.0,
            lon_min=90.0,
            lon_max=115.0,
            var='rainnc',
            accum='a01h',
            time_index=5,
            colormap='viridis',
            dpi=150,
            formats=['png', 'pdf'],
            batch_all=True,
            verbose=True,
            quiet=False,
            parallel=True
        )
        
        config = ArgumentParser.parse_args_to_config(args)
        
        assert isinstance(config, MPASConfig)
        assert config.grid_file == 'grid.nc'
        assert config.data_dir == './data'
        assert config.lat_min == -10.0
        assert config.variable == 'rainnc'
        assert config.dpi == 150
    
    def test_parse_args_with_figure_size(self: "TestParseArgsToConfig") -> None:
        """
        This test checks that the `figure_size` argument is correctly converted into a tuple on the `MPASConfig`. When a list of two floats is provided for `figure_size`, the converter should transform it into a tuple of (width, height) and assign it to the appropriate attributes on the configuration object. This ensures that figure sizing options are properly handled and can be used by plotting functions to set the dimensions of generated figures. 

        Parameters:
            None

        Returns:
            None: Verified by assertions on `figure_width` and `figure_height`.
        """
        args = argparse.Namespace(
            figure_size=[12.0, 8.0],
            grid_file='grid.nc',
            data_dir='./data'
        )
        
        config = ArgumentParser.parse_args_to_config(args)
        
        assert config.figure_width == 12.0
        assert config.figure_height == 8.0
    
    def test_parse_args_with_xarray_data_type(self: "TestParseArgsToConfig") -> None:
        """
        This test ensures that when `data_type='xarray'` is specified in the arguments, the resulting configuration has `use_pure_xarray` set to True. This flag indicates that the application should use pure xarray-based data handling for MPAS files, which may affect how data is loaded and processed. The test confirms that the converter correctly interprets the `data_type` argument and sets the appropriate configuration attribute to enable xarray functionality. 

        Parameters:
            None

        Returns:
            None: Verified via boolean assertion on `use_pure_xarray`.
        """
        args = argparse.Namespace(
            data_type='xarray',
            grid_file='grid.nc',
            data_dir='./data'
        )
        
        config = ArgumentParser.parse_args_to_config(args)        
        assert config.use_pure_xarray is True
    
    def test_parse_args_with_uxarray_data_type(self: "TestParseArgsToConfig") -> None:
        """
        This test verifies that when `data_type='uxarray'` is specified, the resulting configuration does not have `use_pure_xarray` set to True. The 'uxarray' data type may indicate a different data handling approach that does not rely on pure xarray functionality. This test confirms that the converter correctly distinguishes between 'xarray' and 'uxarray' data types and does not erroneously set the xarray flag when 'uxarray' is chosen. 

        Parameters:
            None

        Returns:
            None: Verified by checking the absence or false value of the flag.
        """
        args = argparse.Namespace(
            data_type='uxarray',
            grid_file='grid.nc',
            data_dir='./data'
        )
        
        config = ArgumentParser.parse_args_to_config(args)
        assert not (hasattr(config, 'use_pure_xarray') and config.use_pure_xarray)
    
    def test_parse_args_with_real_mpas_paths(self: "TestParseArgsToConfig", grid_file, test_data_dir) -> None:
        """
        This test performs an integration check by parsing arguments that include real MPAS grid file and data directory paths provided by session fixtures. It verifies that the `parse_args_to_config` method can handle actual file system paths and correctly populate the configuration object with these values. This test ensures that the argument parsing and config conversion logic works as expected in a realistic scenario where users provide valid MPAS file paths for processing. 

        Parameters:
            grid_file: Session fixture providing path to real MPAS grid file.
            test_data_dir: Session fixture providing path to test data directory.

        Returns:
            None: Verified by assertions on config paths and types.
        """
        if grid_file is None:
            pytest.skip("MPAS grid file not available")
        
        data_dir = str(test_data_dir / "u120k" / "diag")
        
        args = argparse.Namespace(
            grid_file=grid_file,
            data_dir=data_dir,
            output_dir='./output',
            lat_min=-10.0,
            lat_max=15.0,
            lon_min=90.0,
            lon_max=115.0,
            var='rainnc',
            time_index=0,
            verbose=False
        )
        
        config = ArgumentParser.parse_args_to_config(args)
        
        assert isinstance(config, MPASConfig)
        assert config.grid_file == grid_file
        assert config.data_dir == data_dir
        assert config.variable == 'rainnc'


class TestCreateSurfaceParser:
    """ Test surface parser creation. """
    
    def test_create_surface_parser_returns_parser(self: "TestCreateSurfaceParser") -> None:
        """
        This test verifies that the `create_surface_parser` factory method returns an instance of `argparse.ArgumentParser`. The returned parser should be configured to accept command-line arguments specific to surface plotting, such as variable selection, plotting options, and output settings. This test confirms the type of the returned object to ensure it is a valid argument parser ready for use in the surface plotting workflow. 

        Parameters:
            None

        Returns:
            None: Verified by isinstance assertion.
        """
        parser = ArgumentParser.create_surface_parser()        
        assert isinstance(parser, argparse.ArgumentParser)
    
    def test_create_surface_parser_required_args(self: "TestCreateSurfaceParser") -> None:
        """
        This test checks that the surface parser correctly handles required arguments such as `--grid-file`, `--data-dir`, and `--variable`. By parsing a minimal set of required arguments, we can confirm that the parser is structured to accept these essential inputs for surface plotting and that they are accessible as attributes on the parsed namespace. This ensures that users will be prompted for necessary information when using the surface plotting CLI. 

        Parameters:
            None

        Returns:
            None: Verified via assertions on parsed namespace attributes.
        """
        parser = ArgumentParser.create_surface_parser()
        
        args = parser.parse_args([
            '--grid-file', 'grid.nc',
            '--data-dir', './data',
            '--variable', 't2m'
        ])
        
        assert args.grid_file == 'grid.nc'
        assert args.data_dir == './data'
        assert args.variable == 't2m'
    
    def test_create_surface_parser_optional_args(self: "TestCreateSurfaceParser") -> None:
        """
        This test verifies that the surface parser correctly handles a comprehensive set of optional arguments related to surface plotting. By providing a rich set of command-line options, we can confirm that the parser accepts and correctly types these arguments, making them available on the parsed namespace for use in the plotting workflow. This test ensures that users have flexibility in customizing their surface plots through various CLI options and that these options are properly parsed. 

        Parameters:
            None

        Returns:
            None: Verified through assertions on argument values and types.
        """
        parser = ArgumentParser.create_surface_parser()
        
        args = parser.parse_args([
            '--grid-file', 'grid.nc',
            '--data-dir', './data',
            '--variable', 'surface_pressure',
            '--plot-type', 'contour',
            '--colormap', 'plasma',
            '--time-index', '10',
            '--lat-min', '0',
            '--lat-max', '20',
            '--lon-min', '100',
            '--lon-max', '120',
            '--clim-min', '900',
            '--clim-max', '1020',
            '--output', 'pressure_plot',
            '--dpi', '300',
            '--verbose',
            '--batch-all'
        ])
        
        assert args.plot_type == 'contour'
        assert args.colormap == 'plasma'
        assert args.time_index == 10
        assert args.verbose is True
        assert args.batch_all is True


class TestParseSurfaceArgsToConfig:
    """ Test surface argument-to-config conversion. """
    
    def test_parse_surface_args_basic(self: "TestParseSurfaceArgsToConfig") -> None:
        """
        This test verifies that the `parse_surface_args_to_config` method correctly converts a parsed argument namespace specific to surface plotting into an `MPASConfig` instance. By providing a comprehensive set of surface-related arguments, we can confirm that each relevant field is properly mapped onto the configuration object, including type conversions for numeric values and handling of optional parameters. This test ensures that the conversion logic for surface plotting arguments is robust and produces a configuration object that downstream plotting functions can utilize effectively. 

        Parameters:
            None

        Returns:
            None: Verified via type checks and multiple attribute assertions.
        """
        args = argparse.Namespace(
            grid_file='grid.nc',
            data_dir='./data',
            output_dir='./output',
            variable='t2m',
            time_index=5,
            plot_type='scatter',
            colormap='coolwarm',
            title='Temperature Analysis',
            lat_min=-5.0,
            lat_max=10.0,
            lon_min=95.0,
            lon_max=110.0,
            clim_min=280.0,
            clim_max=310.0,
            grid_resolution=200,
            output='temp_plot',
            dpi=200,
            formats=['png', 'pdf'],
            verbose=True,
            quiet=False,
            batch_all=False,
            figure_size=[10.0, 12.0]
        )
        
        config = ArgumentParser.parse_surface_args_to_config(args)
        
        assert isinstance(config, MPASConfig)
        assert config.grid_file == 'grid.nc'
        assert config.variable == 't2m'
        assert config.time_index == 5
        assert config.plot_type == 'scatter'
        assert config.colormap == 'coolwarm'
        assert config.clim_min == 280.0
        assert config.clim_max == 310.0
        assert config.grid_resolution == 200
        assert config.figure_size == (10.0, 12.0)
    
    def test_parse_surface_args_with_batch_all(self: "TestParseSurfaceArgsToConfig") -> None:
        """
        This test ensures that when the `batch_all` flag is set to True in the parsed arguments, the resulting configuration has `batch_mode` set to True. This indicates that the application should run in batch processing mode, which may affect how plots are generated and saved. The test confirms that the converter correctly interprets the `batch_all` argument and sets the appropriate configuration attribute to enable batch processing functionality. 

        Parameters:
            None

        Returns:
            None: Verified by asserting `config.batch_mode is True`.
        """
        args = argparse.Namespace(
            grid_file='grid.nc',
            data_dir='./data',
            variable='t2m',
            batch_all=True,
            figure_size=None
        )
        
        config = ArgumentParser.parse_surface_args_to_config(args)        
        assert config.batch_mode is True
    
    def test_parse_surface_args_default_plot_type(self: "TestParseSurfaceArgsToConfig") -> None:
        """
        This test checks that when no `plot_type` is specified in the parsed arguments, the resulting configuration defaults to 'scatter'. This verifies that the converter applies a sensible default for the type of plot to generate when the user does not explicitly choose one. The test confirms that the default behavior for surface plotting is correctly implemented in the argument-to-config conversion logic. 

        Parameters:
            None

        Returns:
            None: Verified by asserting `config.plot_type == 'scatter'`.
        """
        args = argparse.Namespace(
            grid_file='grid.nc',
            data_dir='./data',
            variable='t2m',
            batch_all=False,
            figure_size=None
        )
        
        config = ArgumentParser.parse_surface_args_to_config(args)        
        assert config.plot_type == 'scatter'
    
    def test_parse_surface_args_default_time_index(self: "TestParseSurfaceArgsToConfig") -> None:
        """
        This test verifies that when no `time_index` is provided in the parsed arguments, the resulting configuration defaults to 0. This ensures that if the user does not specify a time index for plotting, the application will use the first time step by default. The test confirms that the default value for `time_index` is correctly applied during the argument-to-config conversion process.

        Parameters:
            None

        Returns:
            None: Verified by asserting `config.time_index == 0`.
        """
        args = argparse.Namespace(
            grid_file='grid.nc',
            data_dir='./data',
            variable='t2m',
            batch_all=False,
            figure_size=None
        )
        
        config = ArgumentParser.parse_surface_args_to_config(args)        
        assert config.time_index == 0


    def test_parse_surface_args_with_real_mpas_paths(self: "TestParseSurfaceArgsToConfig", grid_file, test_data_dir) -> None:
        """
        This test performs an integration check by parsing surface plotting arguments that include real MPAS grid file and data directory paths provided by session fixtures. It verifies that the `parse_surface_args_to_config` method can handle actual file system paths and correctly populate the configuration object with these values, along with surface-specific plotting parameters. This test ensures that the argument parsing and config conversion logic for surface plotting works as expected in a realistic scenario where users provide valid MPAS file paths for processing. 

        Parameters:
            grid_file: Session fixture providing path to real MPAS grid file.
            test_data_dir: Session fixture providing path to test data directory.

        Returns:
            None: Verified by assertions on config attributes.
        """
        if grid_file is None:
            pytest.skip("MPAS grid file not available")
        
        data_dir = str(test_data_dir / "u120k" / "diag")
        
        args = argparse.Namespace(
            grid_file=grid_file,
            data_dir=data_dir,
            variable='t2m',
            output_dir='./output',
            batch_all=False,
            time_index=0,
            plot_type='contourf',
            figure_size=[12.0, 8.0],
            verbose=False
        )
        
        config = ArgumentParser.parse_surface_args_to_config(args)
        
        assert isinstance(config, MPASConfig)
        assert config.grid_file == grid_file
        assert config.data_dir == data_dir
        assert config.variable == 't2m'
        assert config.plot_type == 'contourf'


class TestCreateWindParser:
    """ Test wind parser creation. """
    
    def test_create_wind_parser_returns_parser(self: "TestCreateWindParser") -> None:
        """
        This test verifies that the `create_wind_parser` factory method returns an instance of `argparse.ArgumentParser`. The returned parser should be configured to accept command-line arguments specific to wind plotting, such as variable selection, level options, plotting styles, and output settings. This test confirms the type of the returned object to ensure it is a valid argument parser ready for use in the wind plotting workflow. 

        Parameters:
            None

        Returns:
            None: Verified via isinstance assertion.
        """
        parser = ArgumentParser.create_wind_parser()        
        assert isinstance(parser, argparse.ArgumentParser)
    
    def test_create_wind_parser_defaults(self: "TestCreateWindParser") -> None:
        """
        This test checks that the wind parser correctly applies default values for optional arguments when they are not provided. By parsing a minimal set of required arguments, we can confirm that the parser assigns sensible defaults for parameters such as variable names, level, plotting style, sampling options, and time index. This ensures that users can run the wind plotting CLI with minimal input and still get a functional plot with reasonable defaults. 

        Parameters:
            None

        Returns:
            None: Verified via attribute assertions on parsed namespace.
        """
        parser = ArgumentParser.create_wind_parser()
        
        args = parser.parse_args([
            '--grid_file', 'grid.nc',
            '--data_dir', './data'
        ])
        
        assert args.u_variable == 'u10'
        assert args.v_variable == 'v10'
        assert args.wind_level == 'surface'
        assert args.wind_plot_type == 'barbs'
        assert args.subsample_factor == 0
        assert args.time_index == 0
    
    def test_create_wind_parser_custom_args(self: "TestCreateWindParser") -> None:
        """
        This test verifies that the wind parser correctly handles a comprehensive set of custom arguments related to wind plotting. By providing a rich set of command-line options, we can confirm that the parser accepts and correctly types these arguments, making them available on the parsed namespace for use in the plotting workflow. This test ensures that users have flexibility in customizing their wind plots through various CLI options and that these options are properly parsed. 

        Parameters:
            None

        Returns:
            None: Validated via assertions on parsed attributes.
        """
        parser = ArgumentParser.create_wind_parser()
        
        args = parser.parse_args([
            '--grid_file', 'grid.nc',
            '--data_dir', './data',
            '--u-variable', 'u850',
            '--v-variable', 'v850',
            '--wind-level', '850mb',
            '--wind-plot-type', 'arrows',
            '--subsample', '5',
            '--wind-scale', '100.0',
            '--show-background',
            '--background-colormap', 'plasma',
            '--time-index', '12',
            '--extent', '-100', '-90', '30', '45',
            '--output', 'wind_plot',
            '--title', 'Wind Analysis',
            '--dpi', '300',
            '--verbose',
            '--batch-all'
        ])
        
        assert args.u_variable == 'u850'
        assert args.v_variable == 'v850'
        assert args.wind_level == '850mb'
        assert args.wind_plot_type == 'arrows'
        assert args.subsample_factor == 5
        assert args.wind_scale == 100.0
        assert args.show_background is True
        assert args.background_colormap == 'plasma'
        assert args.extent == [-100.0, -90.0, 30.0, 45.0]
        assert args.verbose is True
        assert args.batch_all is True


class TestParseWindArgsToConfig:
    """ Test wind argument-to-config conversion. """
    
    def test_parse_wind_args_with_extent(self: "TestParseWindArgsToConfig") -> None:
        """
        This test verifies that when a specific `extent` is provided in the parsed arguments, the resulting configuration correctly maps the longitude and latitude bounds onto the appropriate attributes. The converter should take the list of extent values and assign them to `lon_min`, `lon_max`, `lat_min`, and `lat_max` on the configuration object. This ensures that spatial subsetting for wind plots is properly configured based on user input. 

        Parameters:
            None

        Returns:
            None: Verified via multiple attribute assertions on `config`.
        """
        args = argparse.Namespace(
            extent=[-100.0, -90.0, 30.0, 45.0],
            grid_file='grid.nc',
            data_dir='./data',
            u_variable='u10',
            v_variable='v10',
            wind_level='surface',
            wind_plot_type='barbs',
            subsample_factor=3,
            wind_scale=50.0,
            show_background=True,
            background_colormap='viridis',
            time_index=10,
            output='wind_output',
            output_dir='./plots',
            output_formats=['png'],
            title='Wind Field',
            figure_size=[12.0, 10.0],
            dpi=150,
            verbose=True,
            batch_all=False
        )
        
        config = ArgumentParser.parse_wind_args_to_config(args)
        
        assert isinstance(config, MPASConfig)
        assert config.grid_file == 'grid.nc'
        assert config.u_variable == 'u10'
        assert config.v_variable == 'v10'
        assert config.wind_level == 'surface'
        assert config.wind_plot_type == 'barbs'
        assert config.subsample_factor == 3
        assert config.wind_scale == 50.0
        assert config.show_background is True
        assert config.background_colormap == 'viridis'
        assert config.lon_min == -100.0
        assert config.lon_max == -90.0
        assert config.lat_min == 30.0
        assert config.lat_max == 45.0
        assert config.figure_size == (12.0, 10.0)
    
    def test_parse_wind_args_without_extent(self: "TestParseWindArgsToConfig") -> None:
        """
        This test checks that when no `extent` is provided in the parsed arguments, the resulting configuration defaults to a global extent. The converter should assign default values of `lon_min=-180.0`, `lon_max=180.0`, `lat_min=-90.0`, and `lat_max=90.0` to the configuration object when the `extent` argument is None. This ensures that if users do not specify a spatial extent for their wind plots, the application will use a global view by default. 

        Parameters:
            None

        Returns:
            None: Verified by assertions on `lon_min/lon_max/lat_min/lat_max`.
        """
        args = argparse.Namespace(
            extent=None,
            grid_file='grid.nc',
            data_dir='./data',
            u_variable='u10',
            v_variable='v10',
            wind_level='surface',
            wind_plot_type='barbs',
            subsample_factor=0,
            wind_scale=None,
            show_background=False,
            background_colormap='viridis',
            time_index=0,
            output=None,
            output_dir='.',
            output_formats=['png'],
            title=None,
            figure_size=[10.0, 13.0],
            dpi=100,
            verbose=False,
            batch_all=False
        )
        
        config = ArgumentParser.parse_wind_args_to_config(args)
        
        assert config.lon_min == -180.0
        assert config.lon_max == 180.0
        assert config.lat_min == -90.0
        assert config.lat_max == 90.0
    
    def test_parse_wind_args_with_batch_all(self: "TestParseWindArgsToConfig") -> None:
        """
        This test ensures that when the `batch_all` flag is set to True in the parsed arguments, the resulting configuration has `batch_mode` set to True. This indicates that the application should run in batch processing mode for wind plotting, which may affect how plots are generated and saved. The test confirms that the converter correctly interprets the `batch_all` argument and sets the appropriate configuration attribute to enable batch processing functionality for wind plots. 

        Parameters:
            None

        Returns:
            None: Verified via assertion that `config.batch_mode is True`.
        """
        args = argparse.Namespace(
            extent=None,
            grid_file='grid.nc',
            data_dir='./data',
            u_variable='u10',
            v_variable='v10',
            wind_level='surface',
            wind_plot_type='barbs',
            subsample_factor=0,
            wind_scale=None,
            show_background=False,
            background_colormap='viridis',
            time_index=0,
            output=None,
            output_dir='.',
            output_formats=['png'],
            title=None,
            figure_size=[10.0, 13.0],
            dpi=100,
            verbose=False,
            batch_all=True
        )
        
        config = ArgumentParser.parse_wind_args_to_config(args)        
        assert config.batch_mode is True
    
    def test_parse_wind_args_with_real_mpas_paths(self: "TestParseWindArgsToConfig", grid_file, test_data_dir) -> None:
        """
        This test performs an integration check by parsing wind plotting arguments that include real MPAS grid file and data directory paths provided by session fixtures. It verifies that the `parse_wind_args_to_config` method can handle actual file system paths and correctly populate the configuration object with these values, along with wind-specific plotting parameters. This test ensures that the argument parsing and config conversion logic for wind plotting works as expected in a realistic scenario where users provide valid MPAS file paths for processing. 

        Parameters:
            grid_file: Session fixture providing path to real MPAS grid file.
            test_data_dir: Session fixture providing path to test data directory.

        Returns:
            None: Verified by assertions on config attributes.
        """
        if grid_file is None:
            pytest.skip("MPAS grid file not available")
        
        data_dir = str(test_data_dir / "u120k" / "diag")
        
        args = argparse.Namespace(
            grid_file=grid_file,
            data_dir=data_dir,
            extent=[-100.0, -80.0, 30.0, 50.0],
            u_variable='u10',
            v_variable='v10',
            wind_level='10m',
            wind_plot_type='barbs',
            subsample_factor=5,
            wind_scale=None,
            show_background=False,
            background_colormap='viridis',
            time_index=0,
            output=None,
            output_dir='./output',
            output_formats=['png'],
            title=None,
            figure_size=[10.0, 8.0],
            dpi=100,
            batch_all=False,
            verbose=False
        )
        
        config = ArgumentParser.parse_wind_args_to_config(args)
        
        assert isinstance(config, MPASConfig)
        assert config.grid_file == grid_file
        assert config.data_dir == data_dir
        assert config.wind_plot_type == 'barbs'
        assert config.subsample_factor == 5


class TestCreateCrosssectionParser:
    """ Test cross-section parser creation. """
    
    def test_create_crosssection_parser_returns_parser(self: "TestCreateCrosssectionParser") -> None:
        """
        This test verifies that the `create_crosssection_parser` factory method returns an instance of `argparse.ArgumentParser`. The returned parser should be configured to accept command-line arguments specific to cross-section plotting, such as start/end coordinates, vertical options, plotting styles, and output settings. This test confirms the type of the returned object to ensure it is a valid argument parser ready for use in the cross-section plotting workflow. 

        Parameters:
            None

        Returns:
            None: Verified by isinstance assertion.
        """
        parser = ArgumentParser.create_crosssection_parser()        
        assert isinstance(parser, argparse.ArgumentParser)
    
    def test_create_crosssection_parser_required_args(self: "TestCreateCrosssectionParser") -> None:
        """
        This test checks that the cross-section parser correctly handles required arguments such as `--grid-file`, `--data-dir`, `--variable`, and the start/end coordinates. By parsing a minimal set of required arguments, we can confirm that the parser is structured to accept these essential inputs for cross-section plotting and that they are accessible as attributes on the parsed namespace. This ensures that users will be prompted for necessary information when using the cross-section plotting CLI. 

        Parameters:
            None

        Returns:
            None: Verified via assertions on parsed namespace attributes.
        """
        parser = ArgumentParser.create_crosssection_parser()
        
        args = parser.parse_args([
            '--grid-file', 'grid.nc',
            '--data-dir', './data',
            '--variable', 'theta',
            '--start-lon', '-105.0',
            '--start-lat', '39.7',
            '--end-lon', '-94.6',
            '--end-lat', '39.1'
        ])
        
        assert args.grid_file == 'grid.nc'
        assert args.data_dir == './data'
        assert args.variable == 'theta'
        assert args.start_lon == -105.0
        assert args.start_lat == 39.7
        assert args.end_lon == -94.6
        assert args.end_lat == 39.1
    
    def test_create_crosssection_parser_defaults(self: "TestCreateCrosssectionParser") -> None:
        """
        This test verifies that the cross-section parser correctly applies default values for optional arguments when they are not provided. By parsing a minimal set of required arguments, we can confirm that the parser assigns sensible defaults for parameters such as time index, vertical coordinate, number of points, plotting style, colormap, and output settings. This ensures that users can run the cross-section plotting CLI with minimal input and still get a functional plot with reasonable defaults. 

        Parameters:
            None

        Returns:
            None: Verified via assertions on defaulted parsed values.
        """
        parser = ArgumentParser.create_crosssection_parser()
        
        args = parser.parse_args([
            '--grid-file', 'grid.nc',
            '--data-dir', './data',
            '--variable', 'theta',
            '--start-lon', '-100',
            '--start-lat', '30',
            '--end-lon', '-90',
            '--end-lat', '40'
        ])
        
        assert args.time_index == 0
        assert args.vertical_coord == 'pressure'
        assert args.num_points == 100
        assert args.plot_type == 'filled_contour'
        assert args.colormap == 'viridis'
        assert args.extend == 'both'
        assert args.output_dir == './output'
        assert args.dpi == 100
    
    def test_create_crosssection_parser_custom_args(self: "TestCreateCrosssectionParser") -> None:
        """
        This test verifies that the cross-section parser correctly handles a comprehensive set of custom arguments related to cross-section plotting. By providing a rich set of command-line options, we can confirm that the parser accepts and correctly types these arguments, making them available on the parsed namespace for use in the plotting workflow. This test ensures that users have flexibility in customizing their cross-section plots through various CLI options and that these options are properly parsed. 

        Parameters:
            None

        Returns:
            None: Verified via assertions on parsed attribute values.
        """
        parser = ArgumentParser.create_crosssection_parser()
        
        args = parser.parse_args([
            '--grid-file', 'grid.nc',
            '--data-dir', './data',
            '--variable', 'temperature',
            '--start-lon', '-110',
            '--start-lat', '35',
            '--end-lon', '-80',
            '--end-lat', '45',
            '--time-index', '24',
            '--vertical-coord', 'model_levels',
            '--num-points', '200',
            '--max-height', '15.0',
            '--plot-type', 'contour',
            '--colormap', 'plasma',
            '--levels', '250', '260', '270', '280', '290', '300',
            '--extend', 'max',
            '--output', 'xsec_plot',
            '--output-dir', './cross_sections',
            '--output-formats', 'png', 'pdf',
            '--title', 'Temperature Cross-Section',
            '--figure-size', '16', '10',
            '--dpi', '300',
            '--verbose'
        ])
        
        assert args.time_index == 24
        assert args.vertical_coord == 'model_levels'
        assert args.num_points == 200
        assert args.max_height == 15.0
        assert args.plot_type == 'contour'
        assert args.colormap == 'plasma'
        assert args.levels == [250.0, 260.0, 270.0, 280.0, 290.0, 300.0]
        assert args.extend == 'max'
        assert args.verbose is True


class TestParseCrosssectionArgsToConfig:
    """ Test cross-section argument-to-config conversion. """
    
    def test_parse_crosssection_args_basic(self: "TestParseCrosssectionArgsToConfig") -> None:
        """
        This test verifies that the `parse_crosssection_args_to_config` method correctly converts a parsed argument namespace specific to cross-section plotting into an `MPASConfig` instance. By providing a comprehensive set of cross-section-related arguments, we can confirm that each relevant field is properly mapped onto the configuration object, including type conversions for numeric values and handling of optional parameters. This test ensures that the conversion logic for cross-section plotting arguments is robust and produces a configuration object that downstream plotting functions can utilize effectively. 

        Parameters:
            None

        Returns:
            None: Verified through multiple attribute assertions on `config`.
        """
        args = argparse.Namespace(
            grid_file='grid.nc',
            data_dir='./data',
            variable='theta',
            time_index=12,
            output='xsec',
            output_dir='./output',
            output_formats=['png', 'pdf'],
            title='Cross-Section',
            figure_size=[14.0, 8.0],
            dpi=200,
            verbose=True,
            start_lon=-100.0,
            start_lat=30.0,
            end_lon=-90.0,
            end_lat=40.0,
            vertical_coord='pressure',
            num_points=150,
            max_height=12.0,
            plot_type='filled_contour',
            colormap='viridis',
            levels=None,
            extend='both'
        )
        
        config = ArgumentParser.parse_crosssection_args_to_config(args)
        
        assert isinstance(config, MPASConfig)
        assert config.grid_file == 'grid.nc'
        assert config.data_dir == './data'
        assert config.variable == 'theta'
        assert config.time_index == 12
        assert config.start_lon == -100.0
        assert config.start_lat == 30.0
        assert config.end_lon == -90.0
        assert config.end_lat == 40.0
        assert config.vertical_coord == 'pressure'
        assert config.num_points == 150
        assert config.max_height == 12.0
        assert config.plot_type == 'filled_contour'
        assert config.colormap == 'viridis'
        assert config.levels is None
        assert config.extend == 'both'
        assert config.figure_size == (14.0, 8.0)
    
    def test_parse_crosssection_args_with_levels(self: "TestParseCrosssectionArgsToConfig") -> None:
        """
        This test checks that when specific `levels` are provided in the parsed arguments, the resulting configuration correctly includes these levels as a list of floats. The converter should take the list of level values from the arguments and assign them to the `levels` attribute on the configuration object. This ensures that if users specify particular contour levels for their cross-section plots, these levels are properly reflected in the configuration for use during plotting. 

        Parameters:
            None

        Returns:
            None: Verified by comparing `config.levels` to the provided list.
        """
        args = argparse.Namespace(
            grid_file='grid.nc',
            data_dir='./data',
            variable='theta',
            time_index=0,
            output=None,
            output_dir='./output',
            output_formats=['png'],
            title=None,
            figure_size=[14.0, 8.0],
            dpi=100,
            verbose=False,
            start_lon=-100.0,
            start_lat=30.0,
            end_lon=-90.0,
            end_lat=40.0,
            vertical_coord='pressure',
            num_points=100,
            max_height=None,
            plot_type='filled_contour',
            colormap='viridis',
            levels=[250.0, 260.0, 270.0, 280.0, 290.0],
            extend='both'
        )
        
        config = ArgumentParser.parse_crosssection_args_to_config(args)        
        assert config.levels == [250.0, 260.0, 270.0, 280.0, 290.0]
    
    def test_parse_crosssection_args_without_max_height(self: "TestParseCrosssectionArgsToConfig") -> None:
        """
        This test verifies that when no `max_height` is provided in the parsed arguments, the resulting configuration has `max_height` set to None. This ensures that if users do not specify a maximum height for their cross-section plots, the application will not apply any height limit by default. The test confirms that the absence of the `max_height` argument is correctly handled during the argument-to-config conversion process, allowing for flexible plotting without height constraints when desired. 

        Parameters:
            None

        Returns:
            None: Verified by asserting `config.max_height is None`.
        """
        args = argparse.Namespace(
            grid_file='grid.nc',
            data_dir='./data',
            variable='theta',
            time_index=0,
            output=None,
            output_dir='./output',
            output_formats=['png'],
            title=None,
            figure_size=[14.0, 8.0],
            dpi=100,
            verbose=False,
            start_lon=-100.0,
            start_lat=30.0,
            end_lon=-90.0,
            end_lat=40.0,
            vertical_coord='height',
            num_points=100,
            plot_type='filled_contour',
            colormap='viridis',
            levels=None,
            extend='both'
        )
        
        config = ArgumentParser.parse_crosssection_args_to_config(args)        
        assert config.max_height is None
    
    def test_parse_crosssection_args_with_real_mpas_paths(self: "TestParseCrosssectionArgsToConfig", grid_file, test_data_dir) -> None:
        """
        This test performs an integration check by parsing cross-section plotting arguments that include real MPAS grid file and data directory paths provided by session fixtures. It verifies that the `parse_crosssection_args_to_config` method can handle actual file system paths and correctly populate the configuration object with these values, along with cross-section-specific plotting parameters. This test ensures that the argument parsing and config conversion logic for cross-section plotting works as expected in a realistic scenario where users provide valid MPAS file paths for processing. 

        Parameters:
            grid_file: Session fixture providing path to real MPAS grid file.
            test_data_dir: Session fixture providing path to test data directory.

        Returns:
            None: Verified by assertions on config attributes.
        """
        if grid_file is None:
            pytest.skip("MPAS grid file not available")
        
        data_dir = str(test_data_dir / "u120k" / "mpasout")
        
        args = argparse.Namespace(
            grid_file=grid_file,
            data_dir=data_dir,
            variable='theta',
            time_index=0,
            start_lon=-100.0,
            start_lat=30.0,
            end_lon=-90.0,
            end_lat=40.0,
            vertical_coord='pressure',
            num_points=100,
            max_height=None,
            plot_type='filled_contour',
            colormap='viridis',
            levels=None,
            extend='both',
            output=None,
            output_dir='./output',
            output_formats=['png'],
            title=None,
            figure_size=[14.0, 8.0],
            dpi=100,
            verbose=False
        )
        
        config = ArgumentParser.parse_crosssection_args_to_config(args)
        
        assert isinstance(config, MPASConfig)
        assert config.grid_file == grid_file
        assert config.data_dir == data_dir
        assert config.variable == 'theta'
        assert config.start_lon == -100.0
        assert config.end_lon == -90.0
        assert config.vertical_coord == 'pressure'

if __name__ == "__main__": 
    pytest.main([__file__]) 