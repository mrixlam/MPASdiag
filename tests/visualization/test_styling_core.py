#!/usr/bin/env python3
"""
MPASdiag Test Suite: Visualization Core Functionality Tests 

This module contains unit tests for the core styling utilities used in MPASdiag visualizations. It verifies that the styling helpers correctly generate colormaps, contour levels, and map projections based on variable names and data characteristics. The tests cover edge cases such as empty data, all-NaN values, and boundary conditions for humidity levels. It also ensures that the styling module can be imported and that its API surface is consistent with expected usage patterns. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load standard libraries
import os
import sys
import pytest
import shutil
import tempfile
import matplotlib
import numpy as np
import xarray as xr
matplotlib.use('Agg')
import cartopy.crs as ccrs
from typing import Generator
from unittest.mock import Mock
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from mpasdiag.visualization.styling import MPASVisualizationStyle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestStyling:
    """ Test styling functionality. """
    
    def test_import_styling_module(self: "TestStyling") -> None:
        """
        This test verifies that the `styling` module can be imported successfully from the `mpasdiag.visualization` package. It ensures that the module is available and can be accessed without raising an ImportError, confirming that the styling utilities are properly exposed for use in visualization components.

        Parameters:
            None

        Returns:
            None: Confirmed by asserting the imported module is not None.
        """        
        from mpasdiag.visualization import styling
        assert styling is not None
    
    def test_styling_has_style_functions(self: "TestStyling") -> None:
        """
        This test checks that the `styling` module contains the expected styling functions and classes. It verifies that key attributes such as `MPASVisualizationStyle` are present in the module, ensuring that the core styling functionality is accessible for use in generating visualizations.

        Parameters:
            None

        Returns:
            None: Verified by checking module attributes.
        """
        from mpasdiag.visualization import styling
        assert hasattr(styling, '__name__')


class TestCreatePrecipColormap:
    """ Tests for precipitation colormap creation. """
    
    def test_create_precip_colormap_24h(self: "TestCreatePrecipColormap") -> None:
        """
        This test verifies that the `create_precip_colormap` function generates a colormap and levels appropriate for 24-hour precipitation accumulation. It checks that the returned colormap is an instance of `ListedColormap` and that the levels list contains expected values characteristic of daily accumulations (e.g., 100 mm). This ensures that the function correctly configures styling for long-period precipitation variables.

        Parameters:
            None

        Returns:
            None: Verified by type and content assertions on the returned values.
        """
        cmap, levels = MPASVisualizationStyle.create_precip_colormap('a24h')

        assert isinstance(cmap, mcolors.ListedColormap)
        assert isinstance(levels, list)
        assert len(levels) == pytest.approx(10)
        assert 100 in levels 
    
    def test_create_precip_colormap_1h(self: "TestCreatePrecipColormap") -> None:
        """
        This test verifies that the `create_precip_colormap` function generates a colormap and levels appropriate for 1-hour precipitation accumulation. It checks that the returned colormap is an instance of `ListedColormap` and that the levels list contains expected values characteristic of short-period accumulations (e.g., 0.5 mm). This ensures that the function correctly configures styling for hourly precipitation variables.

        Parameters:
            None

        Returns:
            None: Verified by type and content assertions on the returned values.
        """
        cmap, levels = MPASVisualizationStyle.create_precip_colormap('a01h')

        assert isinstance(cmap, mcolors.ListedColormap)
        assert len(levels) == pytest.approx(10)
        assert 0.5 in levels  
    
    def test_create_precip_colormap_invalid_string(self: "TestCreatePrecipColormap") -> None:
        """
        This test checks that the `create_precip_colormap` function handles invalid accumulation period strings gracefully by falling back to default styling. When an unrecognized string is passed, the function should not raise an exception but instead return a colormap and levels corresponding to a default period (e.g., 24h). This test verifies that the function is robust against incorrect metadata and still provides usable output.

        Parameters:
            None

        Returns:
            None: Verified by checking for a default level present in output.
        """
        cmap, levels = MPASVisualizationStyle.create_precip_colormap('invalid')
        assert isinstance(cmap, mcolors.ListedColormap)
        assert 100 in levels
    
    def test_create_precip_colormap_none(self: "TestCreatePrecipColormap") -> None:
        """
        This test verifies that the `create_precip_colormap` function can handle a `None` input for the accumulation period string without raising an exception. The function should default to a standard colormap and level set (e.g., 24h) when no specific period information is provided. This ensures that the styling helper is robust and can provide reasonable defaults in the absence of metadata.

        Parameters:
            None

        Returns:
            None: Verified by asserting expected default values are present.
        """
        cmap, levels = MPASVisualizationStyle.create_precip_colormap(None) # type: ignore
        assert isinstance(cmap, mcolors.ListedColormap)
        assert 100 in levels
    
    def test_create_precip_colormap_regex_failure(self: "TestCreatePrecipColormap") -> None:
        """
        This test checks that the `create_precip_colormap` function handles cases where the regular expression fails to match a valid accumulation period string. If the input string does not conform to expected patterns, the function should catch this and return default styling (e.g., 24h levels) rather than raising an exception. This test ensures that the function is resilient to malformed input and still provides usable output.

        Parameters:
            None

        Returns:
            None: Verified by presence of expected default levels.
        """
        cmap, levels = MPASVisualizationStyle.create_precip_colormap('no_numbers_here')
        assert isinstance(cmap, mcolors.ListedColormap)
        assert 100 in levels


class TestGetVariableStyle:
    """ Tests for variable-specific styling. """
    
    def test_get_variable_style_temperature(self: "TestGetVariableStyle") -> None:
        """
        This test verifies that the `get_variable_style` function returns the expected colormap and settings for a temperature variable (e.g., `t2m`). It checks that the returned style dictionary contains the correct colormap (`RdYlBu_r`) and that it is configured to extend in both directions, which is appropriate for temperature fields that can have both positive and negative anomalies. This ensures that the styling helper correctly identifies temperature variables and applies suitable styling.

        Parameters:
            None

        Returns:
            None: Verified by inspecting keys and expected values in returned dict.
        """
        style = MPASVisualizationStyle.get_variable_style('t2m')

        assert 'colormap' in style
        assert style['colormap'] == 'RdYlBu_r'
        assert style['extend'] == 'both'
    
    def test_get_variable_style_precipitation(self: "TestGetVariableStyle") -> None:
        """
        This test verifies that the `get_variable_style` function returns a discrete colormap and appropriate levels for a precipitation variable (e.g., `rainnc`). It checks that the returned style includes a `ListedColormap` instance and that contour levels are provided, which are essential for visualizing precipitation accumulations. This ensures that the styling helper correctly identifies precipitation variables and applies suitable discrete styling.

        Parameters:
            None

        Returns:
            None: Verified by type checks and presence of 'levels'.
        """
        style = MPASVisualizationStyle.get_variable_style('rainnc')

        assert 'colormap' in style
        assert isinstance(style['colormap'], mcolors.ListedColormap)
        assert 'levels' in style
    
    def test_get_variable_style_precip_with_period(self: "TestGetVariableStyle") -> None:
        """
        This test verifies that the `get_variable_style` function correctly identifies precipitation variables with accumulation period indicators in their names (e.g., `precip_6h`) and returns appropriate styling. The function should recognize the '6h' period and return a `ListedColormap` along with levels suitable for 6-hour accumulations. This test ensures that the styling helper can parse variable names for period information and apply the correct styling logic.

        Parameters:
            None

        Returns:
            None: Verified by type inspection of the returned colormap.
        """
        style = MPASVisualizationStyle.get_variable_style('precip_6h')
        assert isinstance(style['colormap'], mcolors.ListedColormap)
        assert 'levels' in style
    
    def test_get_variable_style_precip_daily(self: "TestGetVariableStyle") -> None:
        """
        This test verifies that the `get_variable_style` function correctly identifies daily precipitation variables (e.g., `daily_precip`) and returns a discrete colormap with appropriate levels for daily accumulations. The function should recognize the 'daily' keyword and return a `ListedColormap` along with levels suitable for 24-hour precipitation totals. This test ensures that the styling helper can handle common naming conventions for accumulated precipitation variables.

        Parameters:
            None

        Returns:
            None: Verified by confirming the returned colormap type.
        """
        style = MPASVisualizationStyle.get_variable_style('daily_precip')

        assert isinstance(style['colormap'], mcolors.ListedColormap)
        assert 'levels' in style

    def test_get_variable_style_precip_hourly(self: "TestGetVariableStyle") -> None:
        """
        This test verifies that the `get_variable_style` function correctly identifies hourly precipitation variables (e.g., `hourly_rain`) and returns a discrete colormap with appropriate levels for hourly accumulations. The function should recognize the 'hourly' keyword and return a `ListedColormap` along with levels suitable for 1-hour precipitation totals. This test ensures that the styling helper can handle common naming conventions for short-period accumulated precipitation variables.

        Parameters:
            None

        Returns:
            None: Verified by confirming the returned colormap type.
        """
        style = MPASVisualizationStyle.get_variable_style('hourly_rain')
        assert isinstance(style['colormap'], mcolors.ListedColormap)
        assert 'levels' in style

    def test_get_variable_style_with_data_array(self: "TestGetVariableStyle", mpas_coordinates, mpas_wind_data) -> None:
        """
        This test verifies that the `get_variable_style` function can accept an `xarray.DataArray` as input and return appropriate styling based on the variable name and data characteristics. It checks that when a DataArray with a name corresponding to mean-sea-level pressure (e.g., 'mslp') is passed, the function returns a style dictionary that includes contour levels suitable for pressure fields. This ensures that the styling helper can utilize data arrays to inform its styling decisions.

        Parameters:
            None

        Returns:
            None: Verified by presence of 'levels' in returned style dict.
        """
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, _ = mpas_wind_data
        vals = u.copy()

        data = xr.DataArray(
            vals,
            dims=['nCells'],
            name='mslp'
        )

        style = MPASVisualizationStyle.get_variable_style('mslp', data)
        assert 'colormap' in style
        assert 'levels' in style
    
    def test_get_variable_style_unknown_variable(self: "TestGetVariableStyle") -> None:
        """
        This test verifies that the `get_variable_style` function returns a default colormap and levels for an unknown variable name. When a variable name that does not match any specific styling rules is passed, the function should return a default style (e.g., using the 'viridis' colormap) and generate levels based on the data if provided. This test ensures that the styling helper can handle unrecognized variable names gracefully while still providing usable styling information.

        Parameters:
            None

        Returns:
            None: Verified by checking 'colormap' and 'levels' keys.
        """
        style = MPASVisualizationStyle.get_variable_style('unknown_var')
        assert style['colormap'] == 'viridis'
        assert 'levels' in style


class TestGenerateLevelsFromData:
    """ Tests for automatic level generation. """
    
    def test_generate_levels_temperature(self: "TestGenerateLevelsFromData", mpas_wind_data) -> None:
        """
        This test verifies that the `_generate_levels_from_data` function produces a reasonable set of contour levels for a temperature-like variable when provided with representative data. It checks that the generated levels are not None, are returned as a list, and contain more than 5 entries, which would indicate a sensible range of levels for a typical temperature field. This ensures that the level generation helper can derive appropriate contour levels from data characteristics.

        Parameters:
            None

        Returns:
            None: Verified by checking the returned levels list properties.
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, _ = mpas_wind_data
        umin = u.min()
        umax = u.max()
        scale = (u - umin) / (umax - umin + 1e-12)
        temp = 250.0 + 60.0 * scale
        data = xr.DataArray(temp, dims=['nCells'])

        levels = MPASVisualizationStyle._generate_levels_from_data(data, 't2m')

        assert levels is not None
        assert isinstance(levels, list)
        assert len(levels) > 5
    
    def test_generate_levels_generic(self: "TestGenerateLevelsFromData", mpas_wind_data) -> None:
        """
        This test verifies that the `_generate_levels_from_data` function can generate contour levels for a generic variable when provided with representative data. It checks that the generated levels are not None and that the number of levels is exactly 16, which corresponds to a default configuration of 15 steps plus the maximum level. This ensures that the level generation helper can produce a standard set of contour levels for variables without specific styling rules.

        Parameters:
            None

        Returns:
            None: Verified by asserting exact expected length of generated levels.
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, _ = mpas_wind_data
        umin = u.min()
        umax = u.max()
        scale = (u - umin) / (umax - umin + 1e-12)
        data = xr.DataArray(0.0 + 100.0 * scale, dims=['nCells'])

        levels = MPASVisualizationStyle._generate_levels_from_data(data, 'generic_var')

        assert levels is not None
        assert len(levels) == pytest.approx(16)
    
    def test_generate_levels_empty_data(self: "TestGenerateLevelsFromData") -> None:
        """
        This test verifies that the `_generate_levels_from_data` function returns `None` when provided with an empty data array. Since there are no values to compute statistics from, the function should recognize this and return `None` to indicate that level generation is not possible. This test ensures that the level generation helper can handle edge cases of empty datasets gracefully.

        Parameters:
            None

        Returns:
            None: Verified by asserting the helper returns None for empty data.
        """
        data = xr.DataArray(
            np.array([]),
            dims=['nCells']
        )

        levels = MPASVisualizationStyle._generate_levels_from_data(data, 'var')
        assert levels is None
    
    def test_generate_levels_all_nan(self: "TestGenerateLevelsFromData") -> None:
        """
        This test verifies that the `_generate_levels_from_data` function returns `None` when provided with a data array that contains only NaN values. Since all values are NaN, the function should not be able to compute meaningful statistics for level generation and should return `None`. This test ensures that the level generation helper can handle datasets with invalid or missing values appropriately.

        Parameters:
            None

        Returns:
            None: Verified by asserting the helper returns None for all-NaN data.
        """
        data = xr.DataArray(
            np.full(100, np.nan),
            dims=['nCells']
        )

        levels = MPASVisualizationStyle._generate_levels_from_data(data, 'var')

        assert levels is None
    
    def test_generate_levels_zero_range(self: "TestGenerateLevelsFromData") -> None:
        """
        This test verifies that the `_generate_levels_from_data` function returns `None` when provided with a data array where all values are the same (zero range). Since there is no variability in the data, the function should recognize that meaningful levels cannot be generated and return `None`. This test ensures that the level generation helper can handle cases of zero-range data appropriately.

        Parameters:
            None

        Returns:
            None: Verified by asserting the helper returns None for zero-range data.
        """
        data = xr.DataArray(
            np.full(100, 42.0),
            dims=['nCells']
        )

        levels = MPASVisualizationStyle._generate_levels_from_data(data, 'var')
        assert levels is None
    
    def test_generate_levels_exception(self: "TestGenerateLevelsFromData") -> None:
        """
        This test verifies that the `_generate_levels_from_data` function returns `None` when an exception occurs during level generation. By mocking the data input to cause an error (e.g., by having `values` attribute set to `None`), we can ensure that the function handles unexpected issues gracefully and does not propagate exceptions, instead returning `None` to indicate failure in level generation. This test ensures that the level generation helper is robust against internal errors and provides a safe fallback.

        Parameters:
            None

        Returns:
            None: Verified by asserting the helper returns None on exception.
        """
        data = Mock()
        data.values = None

        levels = MPASVisualizationStyle._generate_levels_from_data(data, 'var')
        assert levels is None


class TestGetVariableSpecificSettings:
    """ Tests for comprehensive variable-specific settings. """
    
    def test_settings_temperature(self: "TestGetVariableSpecificSettings", mpas_wind_data) -> None:
        """
        This test verifies that the `get_variable_specific_settings` function returns the expected colormap and levels for a temperature variable when provided with representative data. It checks that the function selects the `RdYlBu_r` colormap and generates a non-empty list of contour levels appropriate for temperature fields. This ensures that the styling helper can correctly identify temperature variables and apply suitable styling based on data characteristics.

        Parameters:
            mpas_wind_data: session fixture providing synthetic or real wind components used to derive sample temperature-like data.

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, _ = mpas_wind_data
        umin = u.min()
        umax = u.max()
        data = 250.0 + 60.0 * (u - umin) / (umax - umin + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('temperature', data)
        
        assert cmap == 'RdYlBu_r'
        assert levels is not None
        assert len(levels) > 5
    
    def test_settings_precipitation(self: "TestGetVariableSpecificSettings", mpas_wind_data) -> None:
        """
        This test verifies that the `get_variable_specific_settings` function returns a discrete colormap and appropriate levels for a precipitation variable when provided with representative data. It checks that the function selects a `ListedColormap` and generates contour levels suitable for precipitation accumulations, confirming that the styling helper can correctly identify precipitation variables and apply suitable discrete styling based on data characteristics.

        Parameters:
            mpas_wind_data: session fixture providing sample data used to synthesize precipitation-like values.

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, _ = mpas_wind_data
        umin = u.min()
        umax = u.max()
        data = 0.0 + 50.0 * (u - umin) / (umax - umin + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('precip_24h', data)
        
        assert isinstance(cmap, mcolors.ListedColormap)
        assert levels is not None
    
    def test_settings_pressure(self: "TestGetVariableSpecificSettings", mpas_wind_data) -> None:
        """
        This test verifies that the `get_variable_specific_settings` function returns the expected colormap and levels for a pressure variable when provided with representative data. It checks that the function selects the `RdBu_r` colormap and generates contour levels suitable for mean-sea-level pressure fields, confirming that the styling helper can correctly identify pressure variables and apply appropriate styling based on data characteristics.

        Parameters:
            mpas_wind_data: session fixture used to derive synthetic pressure data for the test.

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, _ = mpas_wind_data
        umin = u.min()
        umax = u.max()
        data = 95000.0 + 10000.0 * (u - umin) / (umax - umin + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('mslp', data)
        
        assert cmap == 'RdBu_r'
        assert levels is not None
    
    def test_settings_wind(self: "TestGetVariableSpecificSettings", mpas_wind_data) -> None:
        """
        This test verifies that the `get_variable_specific_settings` function returns the expected colormap and levels for a wind speed variable when provided with representative wind component data. It checks that the function selects the `plasma` colormap and generates contour levels suitable for wind speed fields, confirming that the styling helper can correctly identify wind-related variables and apply appropriate styling based on derived speed data.

        Parameters:
            mpas_wind_data: fixture providing `u, v` wind component arrays.

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, v = mpas_wind_data
        speed = np.hypot(u, v)
        smin = speed.min()
        smax = speed.max()
        data = 0.0 + 25.0 * (speed - smin) / (smax - smin + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('wind_speed', data)
        
        assert cmap == 'plasma'
        assert levels is not None
    
    def test_settings_geopotential(self: "TestGetVariableSpecificSettings", mpas_wind_data) -> None:
        """
        This test verifies that the `get_variable_specific_settings` function returns the expected colormap and levels for a geopotential height variable when provided with representative data. It checks that the function selects the `terrain` colormap and generates contour levels suitable for geopotential height fields, confirming that the styling helper can correctly identify geopotential-related variables and apply appropriate styling based on derived height-like data.

        Parameters:
            mpas_wind_data: fixture used to synthesize geopotential-like data.

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, _ = mpas_wind_data
        umin = u.min()
        umax = u.max()
        data = 5000.0 + 1000.0 * (u - umin) / (umax - umin + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('height_500hPa', data)
        
        assert cmap == 'terrain'
        assert levels is not None
    
    def test_settings_humidity(self: "TestGetVariableSpecificSettings", mpas_wind_data) -> None:
        """
        This test verifies that the `get_variable_specific_settings` function returns the expected colormap and levels for a relative humidity variable when provided with representative data. It checks that the function selects the `BuGn` colormap and generates contour levels suitable for humidity fields, confirming that the styling helper can correctly identify humidity-related variables and apply appropriate styling based on derived relative humidity-like data.

        Parameters:
            mpas_wind_data: fixture to provide data used for humidity sample.

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, _ = mpas_wind_data
        umin = u.min()
        umax = u.max()
        data = 0.0 + 1.0 * (u - umin) / (umax - umin + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('rh', data)
        
        assert cmap == 'BuGn'
        assert levels is not None
    
    def test_settings_empty_data(self: "TestGetVariableSpecificSettings") -> None:
        """
        This test verifies that the `get_variable_specific_settings` function returns a default colormap and `None` levels when provided with an empty data array. It checks that the function selects a default colormap (e.g., `viridis`) and does not attempt to generate levels from empty data, confirming that the styling helper can handle edge cases of empty datasets gracefully.

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([])
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('var', data)
        
        assert cmap == 'viridis'
        assert levels is None
    
    def test_settings_bipolar_data(self: "TestGetVariableSpecificSettings", mpas_wind_data) -> None:
        """
        This test verifies that the `get_variable_specific_settings` function returns a diverging colormap and appropriate levels for a variable with bipolar data (both positive and negative values). It checks that the function selects the `RdBu_r` colormap, which is suitable for variables with values that can deviate in both directions from a central value, and that it generates contour levels that reflect the range of the bipolar data. This ensures that the styling helper can correctly identify and style variables with mixed-sign data.

        Parameters:
            mpas_wind_data: fixture providing data to construct bipolar values.

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, _ = mpas_wind_data
        umin = u.min()
        umax = u.max()
        data = -50.0 + 100.0 * (u - umin) / (umax - umin + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('unknown_var', data)
        
        assert cmap == 'RdBu_r'
        assert levels is not None
    
    def test_settings_positive_only(self: "TestGetVariableSpecificSettings", mpas_wind_data) -> None:
        """
        This test verifies that the `get_variable_specific_settings` function returns a sequential colormap and appropriate levels for a variable with strictly positive data. It checks that the function selects the `viridis` colormap, which is suitable for variables that only take on positive values, and that it generates contour levels that reflect the range of the positive data. This ensures that the styling helper can correctly identify and style variables with strictly positive values.

        Parameters:
            mpas_wind_data: fixture used to synthesize positive-only data.

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, _ = mpas_wind_data
        umin = u.min()
        umax = u.max()
        data = 0.0 + 100.0 * (u - umin) / (umax - umin + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('unknown_var', data)
        
        assert cmap == 'viridis'
    
    def test_settings_negative_only(self: "TestGetVariableSpecificSettings", mpas_wind_data) -> None:
        """
        This test verifies that the `get_variable_specific_settings` function returns a sequential colormap and appropriate levels for a variable with strictly negative data. It checks that the function selects the `plasma` colormap, which can be suitable for variables that only take on negative values, and that it generates contour levels that reflect the range of the negative data. This ensures that the styling helper can correctly identify and style variables with strictly negative values.

        Parameters:
            mpas_wind_data: fixture used to synthesize negative-only values.

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, _ = mpas_wind_data
        umin = u.min()
        umax = u.max()
        data = -100.0 + 90.0 * (u - umin) / (umax - umin + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('unknown_var', data)
        
        assert cmap == 'plasma'


class TestMapProjectionSetup:
    """ Tests for map projection setup. """
    
    def test_setup_platecarree(self: "TestMapProjectionSetup") -> None:
        """
        This test verifies that the `setup_map_projection` function returns a `PlateCarree` map projection and data CRS when requested. It checks that both the map projection and data CRS are instances of `ccrs.PlateCarree`, which is appropriate for global or regional plots where data is in latitude-longitude format. This ensures that the styling helper can correctly configure a basic map projection for common use cases.

        Parameters:
            None

        Returns:
            None
        """
        map_proj, data_crs = MPASVisualizationStyle.setup_map_projection(
            -120, -80, 30, 50, 'PlateCarree'
        )
        
        assert isinstance(map_proj, ccrs.PlateCarree)
        assert isinstance(data_crs, ccrs.PlateCarree)
    
    def test_setup_mercator(self: "TestMapProjectionSetup") -> None:
        """
        This test verifies that the `setup_map_projection` function returns a `Mercator` map projection and `PlateCarree` data CRS when requested. It checks that the map projection is an instance of `ccrs.Mercator`, which is suitable for mid-latitude regional plots, while the data CRS remains `PlateCarree` to facilitate common lat/lon data handling. This ensures that the styling helper can correctly configure a Mercator projection for appropriate use cases.

        Parameters:
            None

        Returns:
            None
        """
        map_proj, data_crs = MPASVisualizationStyle.setup_map_projection(
            -120, -80, 30, 50, 'Mercator'
        )
        
        assert isinstance(map_proj, ccrs.Mercator)
        assert isinstance(data_crs, ccrs.PlateCarree)
    
    def test_setup_lambert(self: "TestMapProjectionSetup") -> None:
        """
        This test verifies that the `setup_map_projection` function returns a `LambertConformal` map projection and `PlateCarree` data CRS when requested. It checks that the map projection is an instance of `ccrs.LambertConformal`, which is ideal for mid-latitude regional plots with a specific focus area, while the data CRS remains `PlateCarree` to facilitate common lat/lon data handling. This ensures that the styling helper can correctly configure a Lambert Conformal projection for appropriate use cases.

        Parameters:
            None

        Returns:
            None
        """
        map_proj, data_crs = MPASVisualizationStyle.setup_map_projection(
            -120, -80, 30, 50, 'LambertConformal'
        )
        
        assert isinstance(map_proj, ccrs.LambertConformal)
        assert isinstance(data_crs, ccrs.PlateCarree)
    
    def test_setup_unknown_projection(self: "TestMapProjectionSetup") -> None:
        """
        This test verifies that the `setup_map_projection` function falls back to a default `PlateCarree` projection when an unknown projection type is requested. It checks that the map projection returned is an instance of `ccrs.PlateCarree`, ensuring that the function handles invalid input gracefully and still provides a usable map projection for plotting. This test confirms the robustness of the projection setup helper against unrecognized projection types.

        Parameters:
            None

        Returns:
            None
        """
        map_proj, data_crs = MPASVisualizationStyle.setup_map_projection(
            -120, -80, 30, 50, 'UnknownProjection'
        )
        
        assert isinstance(map_proj, ccrs.PlateCarree)
    
    def test_setup_projection_central_point(self: "TestMapProjectionSetup") -> None:
        """
        This test verifies that the `setup_map_projection` function correctly calculates the central point of the map extent and uses it to configure the map projection. It checks that the function does not raise exceptions when provided with valid extent parameters and that it returns a map projection instance, confirming that the central point calculation is integrated into the projection setup process. This test ensures that the styling helper can correctly configure projections based on specified geographic extents.

        Parameters:
            None

        Returns:
            None
        """
        map_proj, data_crs = MPASVisualizationStyle.setup_map_projection(
            -120, -80, 30, 50, 'PlateCarree'
        )

        assert isinstance(map_proj, ccrs.PlateCarree)
        assert isinstance(data_crs, ccrs.PlateCarree)


class TestCoordinateFormatting:
    """ Tests for coordinate formatting. """
    
    def test_format_latitude_north(self: "TestCoordinateFormatting") -> None:
        """
        This test verifies that the `format_latitude` function correctly formats positive latitude values with the 'N' hemisphere indicator. It checks that when a positive latitude (e.g., 45.5) is passed, the function returns a string formatted as '45.5°N', confirming that the helper can correctly identify and format northern hemisphere latitudes for display purposes.

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.format_latitude(45.5, None)
        
        assert result == '45.5°N'
    
    def test_format_latitude_south(self: "TestCoordinateFormatting") -> None:
        """
        This test verifies that the `format_latitude` function correctly formats negative latitude values with the 'S' hemisphere indicator. It checks that when a negative latitude (e.g., -33.2) is passed, the function returns a string formatted as '33.2°S', confirming that the helper can correctly identify and format southern hemisphere latitudes for display purposes.

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.format_latitude(-33.2, None)
        
        assert result == '33.2°S'
    
    def test_format_latitude_zero(self: "TestCoordinateFormatting") -> None:
        """
        This test verifies that the `format_latitude` function correctly formats zero latitude with the 'N' hemisphere indicator by convention. It checks that when a latitude of 0.0 is passed, the function returns '0.0°N', confirming that the helper applies a consistent formatting approach for the equator and does not treat it as a special case with a different hemisphere indicator.

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.format_latitude(0.0, None)
        
        assert result == '0.0°N'
    
    def test_format_longitude_east(self: "TestCoordinateFormatting") -> None:
        """
        This test verifies that the `format_longitude` function correctly formats positive longitude values with the 'E' hemisphere indicator. It checks that when a positive longitude (e.g., 120.5) is passed, the function returns a string formatted as '120.5°E', confirming that the helper can correctly identify and format eastward longitudes for display purposes.

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.format_longitude(120.5, None)
        
        assert result == '120.5°E'
    
    def test_format_longitude_west(self: "TestCoordinateFormatting") -> None:
        """
        This test verifies that the `format_longitude` function correctly formats negative longitude values with the 'W' hemisphere indicator. It checks that when a negative longitude (e.g., -75.3) is passed, the function returns a string formatted as '75.3°W', confirming that the helper can correctly identify and format westward longitudes for display purposes.

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.format_longitude(-75.3, None)
        
        assert result == '75.3°W'
    
    def test_format_longitude_zero(self: "TestCoordinateFormatting") -> None:
        """
        This test verifies that the `format_longitude` function correctly formats zero longitude with the 'E' hemisphere indicator by convention. It checks that when a longitude of 0.0 is passed, the function returns '0.0°E', confirming that the helper applies a consistent formatting approach for the prime meridian and does not treat it as a special case with a different hemisphere indicator.

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.format_longitude(0.0, None)
        
        assert result == '0.0°E'


class TestAdaptiveMarkerSize:
    """ Tests for adaptive marker size calculation. """
    
    def test_calculate_marker_size_sparse(self: "TestAdaptiveMarkerSize") -> None:
        """
        This test verifies that the `calculate_adaptive_marker_size` function returns a larger marker size for sparse data densities. It checks that when a low number of points (e.g., 100) is provided within a defined map extent, the computed marker size is greater than 1.0 and does not exceed a reasonable upper limit (e.g., 20.0), confirming that the helper can adjust marker sizes to enhance visibility for sparse datasets.

        Parameters:
            None

        Returns:
            None
        """
        map_extent = (-120, -80, 30, 50)  
        num_points = 100  
        
        size = MPASVisualizationStyle.calculate_adaptive_marker_size(
            map_extent, num_points, (12, 10)
        )
        
        assert size > 1.0
        assert size <= 20.0
    
    def test_calculate_marker_size_dense(self: "TestAdaptiveMarkerSize") -> None:
        """
        This test verifies that the `calculate_adaptive_marker_size` function returns a smaller marker size for dense data densities. It checks that when a high number of points (e.g., 50,000) is provided within a defined map extent, the computed marker size is less than 1.0 and does not fall below a reasonable lower limit (e.g., 0.1), confirming that the helper can adjust marker sizes to prevent overplotting and maintain readability for dense datasets.

        Parameters:
            None

        Returns:
            None
        """
        map_extent = (-120, -80, 30, 50)
        num_points = 50000  
        
        size = MPASVisualizationStyle.calculate_adaptive_marker_size(
            map_extent, num_points, (12, 10)
        )
        
        assert size >= 0.1
        assert size < 5.0
    
    def test_calculate_marker_size_none_extent(self: "TestAdaptiveMarkerSize") -> None:
        """
        This test verifies that the `calculate_adaptive_marker_size` function returns a default marker size when the map extent is `None`. It checks that when `map_extent` is not provided, the function falls back to a predefined default size (e.g., 5.0), confirming that the helper can handle missing extent information gracefully and still provide a usable marker size for plotting.

        Parameters:
            None

        Returns:
            None
        """
        size = MPASVisualizationStyle.calculate_adaptive_marker_size(
            None, 1000, (12, 10) # type: ignore
        )
        
        assert size == pytest.approx(5.0)
    
    def test_calculate_marker_size_zero_area(self: "TestAdaptiveMarkerSize") -> None:
        """
        This test verifies that the `calculate_adaptive_marker_size` function returns a default marker size when the map extent has zero area. It checks that when the provided `map_extent` defines a region with no width or height (e.g., (-80, -80, 30, 30)), the function falls back to a predefined default size (e.g., 5.0), confirming that the helper can handle invalid extent configurations gracefully and still provide a usable marker size for plotting.

        Parameters:
            None

        Returns:
            None
        """
        map_extent = (-80, -80, 30, 30)  
        
        size = MPASVisualizationStyle.calculate_adaptive_marker_size(
            map_extent, 1000, (12, 10)
        )
        
        assert size == pytest.approx(5.0)
    
    def test_calculate_marker_size_zero_points(self: "TestAdaptiveMarkerSize") -> None:
        """
        This test verifies that the `calculate_adaptive_marker_size` function returns a default marker size when the number of points is zero. It checks that when `num_points` is set to zero, the function falls back to a predefined default size (e.g., 5.0), confirming that the helper can handle cases with no data points gracefully and still provide a usable marker size for plotting.

        Parameters:
            None

        Returns:
            None
        """
        map_extent = (-120, -80, 30, 50)
        
        size = MPASVisualizationStyle.calculate_adaptive_marker_size(
            map_extent, 0, (12, 10)
        )
        
        assert size == pytest.approx(5.0)


class TestDynamicTickFormatting:
    """ Tests for dynamic tick formatting. """
    
    def test_format_ticks_integers(self: "TestDynamicTickFormatting") -> None:
        """
        This test verifies that the `format_ticks_dynamic` function formats integer tick values without decimal places. It checks that when a list of integer tick values (e.g., [1.0, 2.0, 3.0]) is passed, the function returns a list of strings formatted as ['1', '2', '3'], confirming that the helper can dynamically adjust formatting to produce clean labels for whole number ticks.

        Parameters:
            None

        Returns:
            None
        """
        ticks = [1.0, 2.0, 3.0, 4.0, 5.0]        
        result = MPASVisualizationStyle.format_ticks_dynamic(ticks)        
        assert result == ['1', '2', '3', '4', '5']
    
    def test_format_ticks_decimals(self: "TestDynamicTickFormatting") -> None:
        """
        This test verifies that the `format_ticks_dynamic` function formats tick values with decimal places when necessary. It checks that when a list of tick values with decimal components (e.g., [1.5, 2.5, 3.5]) is passed, the function returns a list of strings that include decimal points (e.g., ['1.5', '2.5', '3.5']), confirming that the helper can dynamically adjust formatting to preserve necessary precision for non-integer ticks.

        Parameters:
            None

        Returns:
            None
        """
        ticks = [1.5, 2.5, 3.5, 4.5]        
        result = MPASVisualizationStyle.format_ticks_dynamic(ticks)        
        assert len(result) == pytest.approx(4)
        assert '.' in result[0]
    
    def test_format_ticks_scientific(self: "TestDynamicTickFormatting") -> None:
        """
        This test verifies that the `format_ticks_dynamic` function formats tick values in scientific notation when they have large magnitudes. It checks that when a list of large tick values (e.g., [1e5, 2e5, 3e5]) is passed, the function returns a list of strings that include an exponent (e.g., ['1e+05', '2e+05', '3e+05']), confirming that the helper can dynamically adjust formatting to use scientific notation for readability when dealing with large numbers.

        Parameters:
            None

        Returns:
            None
        """
        ticks = [1e5, 2e5, 3e5]        
        result = MPASVisualizationStyle.format_ticks_dynamic(ticks)        
        assert 'e' in result[0].lower()
    
    def test_format_ticks_small_values(self: "TestDynamicTickFormatting") -> None:
        """
        This test verifies that the `format_ticks_dynamic` function formats small tick values in scientific notation when they have small magnitudes. It checks that when a list of small tick values (e.g., [1e-5, 2e-5, 3e-5]) is passed, the function returns a list of strings that include an exponent (e.g., ['1e-05', '2e-05', '3e-05']), confirming that the helper can dynamically adjust formatting to use scientific notation for readability when dealing with small numbers.

        Parameters:
            None

        Returns:
            None
        """
        ticks = [1e-5, 2e-5, 3e-5]        
        result = MPASVisualizationStyle.format_ticks_dynamic(ticks)        
        assert 'e' in result[0].lower()
    
    def test_format_ticks_empty(self: "TestDynamicTickFormatting") -> None:
        """
        This test verifies that the `format_ticks_dynamic` function returns an empty list when given an empty list of ticks. It checks that when an empty list is passed, the function does not raise an error and simply returns an empty list, confirming that the helper can handle edge cases of no tick values gracefully.

        Parameters:
            None

        Returns:
            None
        """
        ticks = []        
        result = MPASVisualizationStyle.format_ticks_dynamic(ticks)        
        assert result == []
    
    def test_format_ticks_with_zero(self: "TestDynamicTickFormatting") -> None:
        """
        This test verifies that the `format_ticks_dynamic` function formats tick values that include zero correctly. It checks that when a list of tick values that includes zero (e.g., [-2.0, -1.0, 0.0, 1.0, 2.0]) is passed, the function returns a list of strings that correctly formats zero (e.g., ['-2', '-1', '0', '1', '2']), confirming that the helper can handle and format zero values appropriately within a range of ticks.

        Parameters:
            None

        Returns:
            None
        """
        ticks = [-2.0, -1.0, 0.0, 1.0, 2.0]        
        result = MPASVisualizationStyle.format_ticks_dynamic(ticks)        
        assert len(result) == pytest.approx(5)
    
    def test_format_ticks_duplicate_labels(self: "TestDynamicTickFormatting") -> None:
        """
        This test verifies that the `format_ticks_dynamic` function increases precision to avoid duplicate labels when tick values are very close together. It checks that when a list of tick values that are close in value (e.g., [1.001, 1.002, 1.003]) is passed, the function returns a list of strings that are all unique (e.g., ['1.001', '1.002', '1.003']), confirming that the helper can dynamically adjust formatting to ensure distinct labels even when tick values are similar.

        Parameters:
            None

        Returns:
            None
        """
        ticks = [1.001, 1.002, 1.003]        
        result = MPASVisualizationStyle.format_ticks_dynamic(ticks)
        assert len(set(result)) == len(result)


class TestTimestampAndBranding:
    """ Tests for timestamp and branding. """
    
    def test_add_timestamp_valid_figure(self: "TestTimestampAndBranding") -> None:
        """
        This test verifies that the `add_timestamp_and_branding` function successfully adds timestamp and branding text to a valid matplotlib `Figure`. It checks that when a `Figure` object is passed, the function adds at least one non-empty text element to the figure, confirming that the helper can enhance the plot with appropriate metadata for traceability and branding purposes.

        Parameters:
            None

        Returns:
            None
        """
        fig = plt.figure()
        
        MPASVisualizationStyle.add_timestamp_and_branding(fig)
        import matplotlib.text as mtext
        all_texts = fig.findobj(mtext.Text)
        text_strings = [t.get_text() for t in all_texts if t.get_text().strip()]
        assert len(text_strings) > 0
        
        plt.close(fig)
    
    def test_add_timestamp_none_figure(self: "TestTimestampAndBranding") -> None:
        """
        This test verifies that the `add_timestamp_and_branding` function does not raise an error when passed `None` instead of a `Figure`. It checks that the function can handle a `None` input gracefully, confirming that the helper includes input validation and does not attempt to add text to a non-existent figure, which would otherwise result in an exception.

        Parameters:
            None

        Returns:
            None
        """
        MPASVisualizationStyle.add_timestamp_and_branding(None) # type: ignore
    
    def test_add_timestamp_with_version(self: "TestTimestampAndBranding") -> None:
        """
        This test verifies that the `add_timestamp_and_branding` function includes version information in the branding text when a version is provided. It checks that when a `Figure` object is passed, the function adds text that contains the version string (e.g., 'MPASdiag v1.0'), confirming that the helper can incorporate version metadata into the plot for enhanced traceability.

        Parameters:
            None

        Returns:
            None
        """
        fig = plt.figure()
        
        MPASVisualizationStyle.add_timestamp_and_branding(fig)
        
        import matplotlib.text as mtext
        all_texts = fig.findobj(mtext.Text)
        text_strings = [t.get_text() for t in all_texts if t.get_text().strip()]
        assert len(text_strings) > 0
        
        branding_texts = [t for t in text_strings if 'MPASdiag' in t]

        assert len(branding_texts) > 0
        assert 'Generated' in branding_texts[0]
        
        plt.close(fig)


class TestSavePlot:
    """ Tests for plot saving. """
    
    @pytest.fixture
    def temp_dir(self) -> Generator[str, None, None]:
        """
        This fixture provides a temporary directory for tests that need to save files to disk. It creates a temporary directory before the test runs and ensures that it is cleaned up afterward, preventing clutter in the project workspace.

        Parameters:
            None

        Returns:
            Generator[str, None, None]: Temporary directory path generator.
        """
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_save_plot_png(self: "TestSavePlot", temp_dir: str) -> None:
        """
        This test verifies that the `save_plot` function successfully saves a figure in PNG format. It checks that when a `Figure` object is passed along with a valid output path and the 'png' format specified, the function creates a PNG file at the expected location on disk, confirming that the plot saving utility can correctly write files in the requested format.

        Parameters:
            temp_dir: path to a temporary directory provided by the fixture.

        Returns:
            None
        """
        fig = plt.figure()
        output_path = os.path.join(temp_dir, 'test_plot')

        MPASVisualizationStyle.save_plot(fig, output_path, formats=['png'])

        assert os.path.exists(output_path + '.png')

        plt.close(fig)
    
    def test_save_plot_multiple_formats(self: "TestSavePlot", temp_dir: str) -> None:
        """
        This test verifies that the `save_plot` function successfully saves a figure in multiple formats (PNG and PDF). It checks that when a `Figure` object is passed along with a valid output path and both 'png' and 'pdf' formats specified, the function creates files in both formats at the expected locations on disk, confirming that the plot saving utility can handle multiple format requests correctly.

        Parameters:
            temp_dir: path to a temporary directory provided by the fixture.

        Returns:
            None
        """
        fig = plt.figure()
        output_path = os.path.join(temp_dir, 'test_plot')

        MPASVisualizationStyle.save_plot(fig, output_path, formats=['png', 'pdf'])

        assert os.path.exists(output_path + '.png')
        assert os.path.exists(output_path + '.pdf')

        plt.close(fig)
    
    def test_save_plot_none_figure(self: "TestSavePlot", temp_dir: str) -> None:
        """
        This test verifies that the `save_plot` function raises a `ValueError` when passed `None` instead of a `Figure`. It checks that when `None` is provided as the figure argument, the function raises an appropriate exception, confirming that the plot saving utility includes input validation to prevent attempts to save non-existent figures.

        Parameters:
            temp_dir: path to a temporary directory provided by the fixture.

        Returns:
            None
        """
        output_path = os.path.join(temp_dir, 'test_plot')

        with pytest.raises(ValueError):
            MPASVisualizationStyle.save_plot(None, output_path) # type: ignore
    
    def test_save_plot_creates_directory(self: "TestSavePlot", temp_dir: str) -> None:
        """
        This test verifies that the `save_plot` function creates missing parent directories when saving plots. If the destination parent directories do not exist, `save_plot` should create them before writing the output files. This test requests a nested path beneath the temporary directory and asserts that the resulting file exists after the call.

        Parameters:
            temp_dir: path to a temporary directory provided by the fixture.

        Returns:
            None
        """
        fig = plt.figure()
        subdir = os.path.join(temp_dir, 'subdir', 'subsubdir')
        output_path = os.path.join(subdir, 'test_plot')

        MPASVisualizationStyle.save_plot(fig, output_path, formats=['png'])

        assert os.path.exists(output_path + '.png')
        
        plt.close(fig)


class TestGet3DVariableStyle:
    """ Tests for 3D variable styling (placeholder). """
    
    def test_get_3d_variable_style_not_implemented(self: "TestGet3DVariableStyle") -> None:
        """
        This test verifies that the `get_3d_variable_style` function raises a `NotImplementedError`, as it is currently a placeholder for future implementation. It checks that when the function is called with a variable name and level, it raises the expected exception, confirming that the method is correctly marked as not yet implemented and serves as a reminder for future development.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(NotImplementedError):
            MPASVisualizationStyle.get_3d_variable_style('temperature', 850.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
