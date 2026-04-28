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
import matplotlib
import xarray as xr
matplotlib.use('Agg')
import matplotlib.colors as mcolors


from mpasdiag.visualization.styling import MPASVisualizationStyle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestGetVariableStyle:
    """ Tests for variable-specific styling. """
    
    def test_get_variable_style_temperature(self: 'TestGetVariableStyle') -> None:
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
    
    def test_get_variable_style_precipitation(self: 'TestGetVariableStyle') -> None:
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
    
    def test_get_variable_style_precip_with_period(self: 'TestGetVariableStyle') -> None:
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
    
    def test_get_variable_style_precip_daily(self: 'TestGetVariableStyle') -> None:
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

    def test_get_variable_style_precip_hourly(self: 'TestGetVariableStyle') -> None:
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

    def test_get_variable_style_with_data_array(self: 'TestGetVariableStyle', mpas_coordinates, mpas_wind_data) -> None:
        """
        This test verifies that the `get_variable_style` function can accept an `xarray.DataArray` as input and return appropriate styling based on the variable name and data characteristics. It checks that when a DataArray with a name corresponding to mean-sea-level pressure (e.g., 'mslp') is passed, the function returns a style dictionary that includes contour levels suitable for pressure fields. This ensures that the styling helper can utilize data arrays to inform its styling decisions.

        Parameters:
            None

        Returns:
            None: Verified by presence of 'levels' in returned style dict.
        """
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
            return
        
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
    
    def test_get_variable_style_unknown_variable(self: 'TestGetVariableStyle') -> None:
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
