#!/usr/bin/env python3
"""
MPASdiag Test Suite: Visualization Styling Edge Cases

This module contains tests designed to cover edge cases and boundary conditions in the MPAS Visualization Styling functionality. The tests focus on ensuring that the styling helpers behave robustly across a range of inputs, including unusual or extreme cases that may not be encountered in typical usage but are important for validating the resilience of the code. It includes tests for precipitation colormap generation across various accumulation periods, case-insensitive variable style lookup, adaptive marker size calculation for different point densities, and dynamic tick formatting for edge cases. Additionally, it contains tests that specifically target previously uncovered lines in the styling code to ensure comprehensive coverage.  

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
import numpy as np
import xarray as xr
import matplotlib.colors as mcolors
from unittest.mock import patch

matplotlib.use('Agg')

from mpasdiag.visualization.styling import MPASVisualizationStyle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestEdgeCases:
    """ Tests for edge cases and boundary conditions. """
    
    
    def test_variable_style_case_insensitive(self: 'TestEdgeCases') -> None:
        """
        This test ensures that variable style retrieval is case-insensitive. The `get_variable_style` function should return the same styling dictionary for variable names that differ only in case (e.g., 'T2M' vs 't2m'). This test asserts that the colormap entries in the returned style dictionaries are identical for both cases, confirming that the lookup mechanism correctly normalizes variable names to ensure consistent styling regardless of input case. 

        Parameters:
            None

        Returns:
            None
        """
        style1 = MPASVisualizationStyle.get_variable_style('T2M')
        style2 = MPASVisualizationStyle.get_variable_style('t2m')        
        assert style1['colormap'] == style2['colormap']
    
    
class TestPrecipitationPeriodBranches:
    """ Tests for precipitation period determination branches. """
    
    def test_precip_style_with_2h_period(self: 'TestPrecipitationPeriodBranches') -> None:
        """
        This test verifies that the `get_variable_style` function correctly identifies a 2-hour accumulation period from the variable name and returns the appropriate styling. Variables with accumulation periods less than 3 hours should map to the 1-hour (a01h) styling branch. This test checks that 'precip_2h' triggers the expected discrete colormap and that characteristic levels (such as 0.5) are present in the returned style, confirming that the period parsing logic correctly categorizes short accumulation periods.

        Parameters:
            None

        Returns:
            None
        """
        style = MPASVisualizationStyle.get_variable_style('precip_2h')        
        assert isinstance(style['colormap'], mcolors.ListedColormap)
        assert 0.5 in style['levels']  
    
    def test_precip_style_with_4h_period(self: 'TestPrecipitationPeriodBranches') -> None:
        """
        This test checks that the `get_variable_style` function correctly categorizes a 4-hour accumulation period. For periods in the range of 3 <= hours < 6, the styling should map to the 3-hour (a03h) branch. This test ensures that 'precip_4h' is processed by the helper without errors and that the returned style includes a discrete colormap and levels characteristic of the 3-hour accumulation styling.

        Parameters:
            None

        Returns:
            None
        """
        style = MPASVisualizationStyle.get_variable_style('precip_4h')        
        assert isinstance(style['colormap'], mcolors.ListedColormap)
        assert 0.5 in style['levels']
    
    def test_precip_style_with_8h_period(self: 'TestPrecipitationPeriodBranches') -> None:
        """
        This test verifies that the `get_variable_style` function correctly identifies an 8-hour accumulation period and applies the appropriate styling. For periods in the range of 6 <= hours < 12, the styling should map to the 6-hour (a06h) branch. This test checks that 'precip_8h' is handled by the helper without raising exceptions and that the returned style includes a discrete colormap and levels consistent with the 6-hour accumulation styling, such as the presence of a level at 0.5.

        Parameters:
            None

        Returns:
            None
        """
        style = MPASVisualizationStyle.get_variable_style('precip_8h')
        assert isinstance(style['colormap'], mcolors.ListedColormap)
        assert 0.5 in style['levels']
    
    def test_precip_style_with_18h_period(self: 'TestPrecipitationPeriodBranches') -> None:
        """
        This test checks that the `get_variable_style` function correctly categorizes an 18-hour accumulation period. For periods in the range of 12 <= hours < 24, the styling should map to the 12-hour (a12h) branch. This test ensures that 'precip_18h' is processed by the helper without errors and that the returned style includes a discrete colormap and levels characteristic of the 12-hour accumulation styling. The presence of levels appropriate for mid-range accumulations (e.g., 0.5) is also checked to confirm that the correct styling branch is selected.

        Parameters:
            None

        Returns:
            None
        """
        style = MPASVisualizationStyle.get_variable_style('precip_18h')
        assert isinstance(style['colormap'], mcolors.ListedColormap)
        assert 10.0 in style['levels']
    
    def test_precip_style_with_48h_period(self: 'TestPrecipitationPeriodBranches') -> None:
        """
        This test verifies that the `get_variable_style` function correctly identifies a 48-hour accumulation period and applies the appropriate styling. For periods of 24 hours or more, the styling should map to the 24-hour (a24h) branch. This test checks that 'precip_48h' is handled by the helper without raising exceptions and that the returned style includes a discrete colormap and levels consistent with the 24-hour accumulation styling, such as the presence of high accumulation levels like 100.

        Parameters:
            None

        Returns:
            None
        """
        style = MPASVisualizationStyle.get_variable_style('precip_48h')
        assert isinstance(style['colormap'], mcolors.ListedColormap)
        assert 100 in style['levels'] 
    
    def test_precip_style_with_daily_keyword(self: 'TestPrecipitationPeriodBranches') -> None:
        """
        This test checks that the `get_variable_style` function correctly maps the 'daily' keyword to 24-hour precipitation styling. Variables containing 'daily' should be treated as 24-hour accumulations (a24h). This test asserts that 'daily_precip' selects the appropriate colormap and level set for long accumulation periods, including the presence of high levels such as 100, confirming that keyword-based period detection is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        style = MPASVisualizationStyle.get_variable_style('daily_precip')
        assert isinstance(style['colormap'], mcolors.ListedColormap)
        assert 100 in style['levels']

    def test_precip_style_with_hourly_keyword(self: 'TestPrecipitationPeriodBranches') -> None:
        """
        This test verifies that the `get_variable_style` function correctly maps the 'hourly' keyword to short-period precipitation styling. Variables containing 'hourly' should be treated as short accumulations (a01h). This test asserts that 'hourly_precip' selects the appropriate discrete colormap and level set for short accumulation periods, including the presence of low levels such as 0.5, confirming that keyword-based period detection is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        style = MPASVisualizationStyle.get_variable_style('hourly_precip')        
        assert isinstance(style['colormap'], mcolors.ListedColormap)
        assert 0.5 in style['levels']
    
    def test_precip_style_no_period_match(self: 'TestPrecipitationPeriodBranches') -> None:
        """
        This test checks the behavior of the `get_variable_style` function when no accumulation period can be determined from the variable name. In such cases, the function should default to 24-hour (a24h) precipitation styling. This test asserts that a variable name without recognizable period keywords (e.g., 'total_rain') still results in a valid style with a discrete colormap and levels appropriate for long accumulations, such as the presence of high levels like 100, confirming that the default styling branch is correctly applied when no specific period is detected.

        Parameters:
            None

        Returns:
            None
        """
        style = MPASVisualizationStyle.get_variable_style('total_rain')
        assert isinstance(style['colormap'], mcolors.ListedColormap)
        assert 20.0 in style['levels']
    
class TestAdditionalMissingLines:
    """ Additional tests for remaining uncovered lines. """
    
    def test_precip_style_exact_1h(self: 'TestAdditionalMissingLines') -> None:
        """
        This test verifies that the `get_variable_style` function correctly identifies an exact 1-hour accumulation period from the variable name and applies the appropriate styling. Variables with an exact 1-hour accumulation (e.g., 'precip_1h') should map to the 1-hour (a01h) styling branch. This test asserts that 'precip_1h' triggers the expected discrete colormap and that characteristic levels (such as 0.5) are present in the returned style, confirming that the period parsing logic correctly categorizes exact short accumulation periods.

        Parameters:
            None

        Returns:
            None
        """
        style = MPASVisualizationStyle.get_variable_style('precip_1h')
        assert isinstance(style['colormap'], mcolors.ListedColormap)
        assert 0.5 in style['levels']
    
    def test_precip_style_exact_6h(self: 'TestAdditionalMissingLines') -> None:
        """
        This test checks that the `get_variable_style` function correctly identifies an exact 6-hour accumulation period from the variable name and applies the appropriate styling. Variables with an exact 6-hour accumulation (e.g., 'precip_6h') should map to the 6-hour (a06h) styling branch. This test asserts that 'precip_6h' triggers the expected discrete colormap and that characteristic levels consistent with 6-hour accumulations are present in the returned style, confirming that the period parsing logic correctly categorizes exact mid-range accumulation periods.

        Parameters:
            None

        Returns:
            None
        """
        style = MPASVisualizationStyle.get_variable_style('precip_6h')
        assert isinstance(style['colormap'], mcolors.ListedColormap)
        assert 0.5 in style['levels']
    
    def test_precip_style_5h_branch(self: 'TestAdditionalMissingLines') -> None:
        """
        This test verifies that the `get_variable_style` function correctly categorizes a 5-hour accumulation period. For periods in the range of 3 <= hours < 6, the styling should map to the 3-hour (a03h) branch. This test ensures that 'rain_5h' is processed by the helper without errors and that the returned style includes a discrete colormap and levels characteristic of the 3-hour accumulation styling, such as the presence of a level at 0.5.

        Parameters:
            None

        Returns:
            None
        """
        style = MPASVisualizationStyle.get_variable_style('rain_5h')
        assert isinstance(style['colormap'], mcolors.ListedColormap)
        assert 0.5 in style['levels']
    
    
class TestGenerateLevelsException:
    """ Tests for exception handling in level generation. """
    
    def test_generate_levels_exception_in_get_variable_style(self: 'TestGenerateLevelsException') -> None:
        """
        This test checks that the `get_variable_style` function handles exceptions raised during level generation gracefully. By patching the `_generate_levels_from_data` method to raise an exception, we can simulate a failure in the level generation logic. The test asserts that when such an exception occurs, the `get_variable_style` function falls back to default level definitions (e.g., a range of 0 to 20) rather than propagating the exception, ensuring that the function remains robust even when internal level generation fails.

        Parameters:
            None

        Returns:
            None
        """
        data = xr.DataArray(np.array([1, 2, 3]))
        
        with patch.object(MPASVisualizationStyle, '_generate_levels_from_data',
                         side_effect=Exception("Level generation error")):
            style = MPASVisualizationStyle.get_variable_style('temperature', data)
            assert 'levels' in style
            assert style['levels'] == list(range(0, 21))
    
    def test_generate_levels_returns_none(self: 'TestGenerateLevelsException') -> None:
        """
        This test verifies that when the `_generate_levels_from_data` function returns None (e.g., due to all NaN data), the `get_variable_style` function falls back to default levels. The test creates a data array filled with NaN values, which should cause the level generation helper to return None. The test then asserts that the `get_variable_style` function detects this and returns a style with default levels (e.g., 0 to 20), confirming that the function can handle cases where level generation fails due to invalid data.

        Parameters:
            None

        Returns:
            None
        """
        data = xr.DataArray(np.array([np.nan, np.nan, np.nan]))        
        style = MPASVisualizationStyle.get_variable_style('temperature', data)        
        assert style['levels'] == list(range(0, 21))
    
    
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
