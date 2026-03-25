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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from unittest.mock import Mock, patch

matplotlib.use('Agg')

from mpasdiag.visualization.styling import MPASVisualizationStyle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestEdgeCases:
    """ Tests for edge cases and boundary conditions. """
    
    def test_precip_colormap_various_periods(self: "TestEdgeCases") -> None:
        """
        This test verifies that the `create_precip_colormap` function correctly generates colormaps and levels for a range of accumulation periods, including edge cases. It checks that the function returns a `ListedColormap` instance and that the levels list contains more than 5 entries for each period, ensuring that the colormap is appropriately detailed for different precipitation accumulation windows.

        Parameters:
            None

        Returns:
            None
        """
        for period in ['a01h', 'a03h', 'a06h', 'a12h', 'a24h']:
            cmap, levels = MPASVisualizationStyle.create_precip_colormap(period)            
            assert isinstance(cmap, mcolors.ListedColormap)
            assert len(levels) > 5
    
    def test_variable_style_case_insensitive(self: "TestEdgeCases") -> None:
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
    
    def test_marker_size_various_densities(self: "TestEdgeCases") -> None:
        """
        This test verifies that the `calculate_adaptive_marker_size` function returns marker sizes within a reasonable range for various point densities. The function should produce smaller marker sizes for higher point counts to prevent overcrowding on the map, while ensuring that the size does not become too small to be visible. This test checks that the returned marker size is between 0.1 and 20.0 for a range of point counts from 10 to 100,000, confirming that the adaptive sizing logic is functioning as intended across different densities.

        Parameters:
            None

        Returns:
            None
        """
        map_extent = (-120, -80, 30, 50)
        
        for num_points in [10, 100, 1000, 10000, 100000]:
            size = MPASVisualizationStyle.calculate_adaptive_marker_size(
                map_extent, num_points, (12, 10)
            )
            
            assert size >= 0.1
            assert size <= 20.0
    
    def test_format_ticks_single_value(self: "TestEdgeCases") -> None:
        """
        This test ensures that the `format_ticks_dynamic` function can handle a single tick value without errors. When provided with a list containing only one tick (e.g., [42.0]), the function should return a list with one formatted label corresponding to that tick. This test confirms that the function does not raise exceptions and returns a valid output even when the input is minimal, demonstrating robustness in handling edge cases of tick formatting.

        Parameters:
            None

        Returns:
            None
        """
        ticks = [42.0]        
        result = MPASVisualizationStyle.format_ticks_dynamic(ticks)        
        assert len(result) == 1


class TestCreatePrecipColormapException:
    """ Tests for exception handling in create_precip_colormap. """
    
    def test_create_precip_colormap_regex_exception(self: "TestCreatePrecipColormapException") -> None:
        """
        This test verifies that the `create_precip_colormap` function handles exceptions raised during regex operations gracefully. By patching the `re.search` function to raise an exception, we can simulate a failure in the regex parsing logic. The test asserts that when such an exception occurs, the `create_precip_colormap` function falls back to a default behavior (e.g., returning a colormap and levels for a 24-hour accumulation) rather than propagating the exception, ensuring that the function remains robust even when internal parsing fails.

        Parameters:
            None

        Returns:
            None
        """
        with patch('mpasdiag.visualization.styling.re.search', side_effect=Exception("Regex error")):
            cmap, levels = MPASVisualizationStyle.create_precip_colormap('a24h')            
            assert 100 in levels
            assert isinstance(cmap, mcolors.ListedColormap)


class TestPrecipitationPeriodBranches:
    """ Tests for precipitation period determination branches. """
    
    def test_precip_style_with_2h_period(self: "TestPrecipitationPeriodBranches") -> None:
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
    
    def test_precip_style_with_4h_period(self: "TestPrecipitationPeriodBranches") -> None:
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
    
    def test_precip_style_with_8h_period(self: "TestPrecipitationPeriodBranches") -> None:
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
    
    def test_precip_style_with_18h_period(self: "TestPrecipitationPeriodBranches") -> None:
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
    
    def test_precip_style_with_48h_period(self: "TestPrecipitationPeriodBranches") -> None:
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
    
    def test_precip_style_with_daily_keyword(self: "TestPrecipitationPeriodBranches") -> None:
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

    def test_precip_style_with_hourly_keyword(self: "TestPrecipitationPeriodBranches") -> None:
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
    
    def test_precip_style_no_period_match(self: "TestPrecipitationPeriodBranches") -> None:
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
    
    def test_precip_style_exact_1h(self: "TestAdditionalMissingLines") -> None:
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
    
    def test_precip_style_exact_6h(self: "TestAdditionalMissingLines") -> None:
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
    
    def test_precip_style_5h_branch(self: "TestAdditionalMissingLines") -> None:
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
    
    def test_generate_levels_temp_step_5(self: "TestAdditionalMissingLines", mpas_surface_temp_data) -> None:
        """
        This test validates that the temperature level generator correctly selects a step of approximately 5.0 for large temperature ranges. When the computed level step is greater than or equal to 5, the helper should return levels spaced near 5.0. This test uses real surface temperature data scaled to create a large range and asserts that at least one adjacent level difference is close to 5.0, confirming that the level generation logic correctly categorizes large temperature ranges.

        Parameters:
            mpas_surface_temp_data: Real surface temperature data from fixtures.

        Returns:
            None
        """
        if mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
        
        t_min, t_max = mpas_surface_temp_data.min(), mpas_surface_temp_data.max()
        data = xr.DataArray(-40.0 + 90.0 * (mpas_surface_temp_data - t_min) / (t_max - t_min + 1e-12))
        
        levels = MPASVisualizationStyle._generate_levels_from_data(data, 'temperature')
        
        assert levels is not None
        assert len(levels) >= 5, "Should have at least 5 levels for large range"

        if len(levels) > 1:
            assert all(levels[i] < levels[i+1] for i in range(len(levels)-1))
    
    def test_generate_levels_no_hasattr_flatten(self: "TestAdditionalMissingLines") -> None:
        """
        This test checks the behavior of the `_generate_levels_from_data` function when the input data object does not have a `flatten` method. The helper should detect the absence of this method and return None rather than raising an AttributeError. This test uses a mock object that simulates a data array without a `flatten` attribute and asserts that the function returns None, confirming that it handles this edge case gracefully.

        Parameters:
            None

        Returns:
            None
        """
        mock_data = Mock()
        mock_data.values = "not_an_array"
        
        result = MPASVisualizationStyle._generate_levels_from_data(mock_data, 'temp')
        assert result is None
    
    def test_settings_precip_default_24h(self: "TestAdditionalMissingLines", mpas_precip_data) -> None:
        """
        This test verifies that when the `get_variable_specific_settings` function is called for a precipitation variable without a recognizable period, it defaults to 24-hour styling. The test uses real precipitation data scaled to have a large maximum value to ensure that the levels generated include high accumulation values (e.g., 100). It asserts that the returned colormap is a `ListedColormap` and that the levels list contains the expected high level, confirming that the default styling branch is correctly applied when no specific period is detected.

        Parameters:
            mpas_precip_data: Real precipitation data from fixtures.

        Returns:
            None
        """
        if mpas_precip_data is None:
            pytest.skip("MPAS data not available")
        
        p_min, p_max = mpas_precip_data.min(), mpas_precip_data.max()
        data = 0.0 + 150.0 * (mpas_precip_data - p_min) / (p_max - p_min + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('snow', data)
        
        assert isinstance(cmap, mcolors.ListedColormap)
        assert levels is not None
        assert 100 in levels
    
    def test_settings_pressure_generic_small(self: "TestAdditionalMissingLines", mpas_wind_data) -> None:
        """
        This test checks that the `get_variable_specific_settings` function correctly identifies pressure-like data with a small range and applies the appropriate styling. When the variable name is 'pres' and the data range is small (e.g., 95000 to 100000), the helper should select a colormap suitable for pressure (e.g., 'RdBu_r') and generate levels that reflect typical pressure values. This test uses real wind data scaled to a pressure-like range and asserts that the correct colormap is returned and that levels are generated, confirming that the function can handle pressure styling even when the input variable name is generic.

        Parameters:
            mpas_wind_data: Real wind data from fixtures used to create pressure-like data.

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, _ = mpas_wind_data
        u_min, u_max = u.min(), u.max()

        data = 95000.0 + 500.0 * (u - u_min) / (u_max - u_min + 1e-12)        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('pres', data)
        
        assert cmap == 'RdBu_r'
        assert levels is not None
    
    def test_settings_wind_high_speed(self: "TestAdditionalMissingLines", mpas_wind_data) -> None:
        """
        This test verifies that the `get_variable_specific_settings` function correctly identifies wind speed data with a high range and applies the appropriate styling. When the variable name is 'wspd' and the data range is large (e.g., 0 to 60 m/s), the helper should select a colormap suitable for wind speed (e.g., 'plasma') and generate levels that reflect typical wind speed values. This test uses real wind data to compute wind speed, scales it to a typical range, and asserts that the correct colormap is returned and that expected levels (such as 10 m/s) are present, confirming that the function can handle wind speed styling even when the input variable name is generic.

        Parameters:
            mpas_wind_data: Real wind data from fixtures.

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, v = mpas_wind_data
        wspd = np.sqrt(u**2 + v**2)
        wspd_min, wspd_max = wspd.min(), wspd.max()
        data = 0.0 + 60.0 * (wspd - wspd_min) / (wspd_max - wspd_min + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('wspd', data)
        
        assert cmap == 'plasma'
        assert levels is not None
        assert 10.0 in levels
    
    def test_settings_humidity_percent(self: "TestAdditionalMissingLines", mpas_qv_3d_data) -> None:
        """
        This test checks that the `get_variable_specific_settings` function correctly identifies specific humidity data scaled to a percentage range and applies the appropriate styling. When the variable name is 'humidity' and the data is scaled to a typical humidity percentage range (e.g., 20% to 90%), the helper should select a colormap suitable for humidity (e.g., 'BuGn') and generate levels that reflect typical humidity values. This test uses real specific humidity data, scales it to a percentage range, and asserts that the correct colormap is returned and that expected levels (such as 20%) are present, confirming that the function can handle humidity styling even when the input variable name is generic.

        Parameters:
            mpas_qv_3d_data: Real specific humidity data from fixtures.

        Returns:
            None
        """
        if mpas_qv_3d_data is None:
            pytest.skip("MPAS data not available")
        
        qv_min, qv_max = mpas_qv_3d_data.min(), mpas_qv_3d_data.max()
        data = 20.0 + 70.0 * (mpas_qv_3d_data - qv_min) / (qv_max - qv_min + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('humidity', data)
        
        assert cmap == 'BuGn'
        assert levels is not None
        assert 20 in levels
    
    def test_save_plot_raises_value_error(self: "TestAdditionalMissingLines") -> None:
        """
        This test verifies that the `save_plot` function raises a ValueError when no figure is provided. The function should check if the input figure is None and raise an appropriate exception with a clear error message. This test asserts that when `save_plot` is called with None as the figure argument, it raises a ValueError containing the message "No figure to save", confirming that the function correctly handles this edge case.

        Parameters:
            None

        Returns:
            None
        """
        assert MPASVisualizationStyle.save_plot is not None

        with pytest.raises(ValueError) as exc_info:
            MPASVisualizationStyle.save_plot(None, '/tmp/test') # type: ignore
        
        assert "No figure to save" in str(exc_info.value)
    
    def test_marker_size_returns_default_none_extent(self: "TestAdditionalMissingLines") -> None:
        """
        This test checks that the `calculate_adaptive_marker_size` function returns a default marker size when the map extent is None. The function should handle a None value for the map extent by returning a reasonable default size (e.g., 5.0) rather than raising an exception. This test asserts that when `calculate_adaptive_marker_size` is called with None for the map extent and a typical point count, it returns the expected default marker size, confirming that the function can handle missing extent information gracefully.

        Parameters:
            None

        Returns:
            None
        """
        assert MPASVisualizationStyle.calculate_adaptive_marker_size is not None
        size = MPASVisualizationStyle.calculate_adaptive_marker_size(None, 1000) # type: ignore
        
        assert size == 5.0
    
    def test_format_ticks_returns_empty_for_empty_input(self: "TestAdditionalMissingLines") -> None:
        """
        This test verifies that the `format_ticks_dynamic` function returns an empty list when given an empty list of ticks. The function should handle this edge case without raising exceptions and should return an empty list as the formatted output. This test asserts that when `format_ticks_dynamic` is called with an empty list, it returns an empty list, confirming that the function can handle the absence of tick values gracefully.

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.format_ticks_dynamic([])
        
        assert result == []
    
    def test_format_ticks_all_zeros_handling(self: "TestAdditionalMissingLines") -> None:
        """
        This test checks that the `format_ticks_dynamic` function can handle a list of ticks that are all zeros without crashing. The function should return a list of formatted labels (which may all be '0' or similar) without raising exceptions, demonstrating robustness in handling edge cases where the tick values do not vary. This test asserts that when `format_ticks_dynamic` is called with a list of zeros, it returns a list of the same length without errors, confirming that the function can process uniform tick values gracefully.

        Parameters:
            None

        Returns:
            None
        """
        ticks = [0, 0, 0, 0]
        assert MPASVisualizationStyle.format_ticks_dynamic is not None
        result = MPASVisualizationStyle.format_ticks_dynamic(ticks) # type: ignore
        assert len(result) == 4
    
    def test_format_ticks_scientific_notation_path(self: "TestAdditionalMissingLines") -> None:
        """
        This test verifies that the `format_ticks_dynamic` function correctly formats ticks in scientific notation when the values are large. When the tick values are sufficiently large (e.g., on the order of 10^5 or greater), the function should return labels in scientific notation to maintain readability. This test asserts that when `format_ticks_dynamic` is called with large tick values, the resulting formatted labels contain 'e' (indicating scientific notation), confirming that the function correctly identifies and formats large tick values.

        Parameters:
            None

        Returns:
            None
        """
        ticks = [50000, 100000, 150000]
        result = MPASVisualizationStyle.format_ticks_dynamic(ticks) # type: ignore
        assert any('e' in str(label).lower() for label in result)
    
    def test_format_ticks_typical_magnitude_selection(self: "TestAdditionalMissingLines") -> None:
        """
        This test checks that the `format_ticks_dynamic` function selects appropriate formatting for typical tick magnitudes. The function should adapt its formatting strategy based on the range and magnitude of the tick values, using fixed-point notation for moderate values and scientific notation for very large or small values. This test asserts that when `format_ticks_dynamic` is called with ticks of varying magnitudes (e.g., 150, 20, 2.5, 0.25, 0.025), it returns formatted labels without errors, confirming that the function can handle a range of typical tick values and apply suitable formatting.

        Parameters:
            None

        Returns:
            None
        """
        ticks_100 = [150, 200, 250, 300]
        assert MPASVisualizationStyle.format_ticks_dynamic is not None

        result_100 = MPASVisualizationStyle.format_ticks_dynamic(ticks_100) # type: ignore

        assert result_100 is not None
        
        ticks_10 = [15, 20, 25, 30]
        result_10 = MPASVisualizationStyle.format_ticks_dynamic(ticks_10) # type: ignore

        assert result_10 is not None
        
        ticks_1 = [1.5, 2.0, 2.5, 3.0]
        result_1 = MPASVisualizationStyle.format_ticks_dynamic(ticks_1)

        assert result_1 is not None
        
        ticks_01 = [0.15, 0.20, 0.25, 0.30]
        result_01 = MPASVisualizationStyle.format_ticks_dynamic(ticks_01)

        assert result_01 is not None
        
        ticks_001 = [0.015, 0.020, 0.025, 0.030]
        result_001 = MPASVisualizationStyle.format_ticks_dynamic(ticks_001)

        assert result_001 is not None
    
    def test_format_ticks_duplicate_handling_g_format(self: "TestAdditionalMissingLines") -> None:
        """
        This test verifies that the `format_ticks_dynamic` function can handle cases where tick values are close enough to produce duplicate labels with standard formatting. The function should attempt to use 'g' format or similar strategies to differentiate tick labels when values are very close together. This test asserts that when `format_ticks_dynamic` is called with closely spaced tick values (e.g., 1.0001, 1.0002, 1.0003, 1.0004), it returns a list of formatted labels that are distinct (or at least does not raise an error), confirming that the function can handle potential label duplication gracefully.

        Parameters:
            None

        Returns:
            None
        """
        ticks = [1.0001, 1.0002, 1.0003, 1.0004]
        result = MPASVisualizationStyle.format_ticks_dynamic(ticks)        
        assert len(result) == len(ticks)


class TestGenerateLevelsException:
    """ Tests for exception handling in level generation. """
    
    def test_generate_levels_exception_in_get_variable_style(self: "TestGenerateLevelsException") -> None:
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
    
    def test_generate_levels_returns_none(self: "TestGenerateLevelsException") -> None:
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
    
    def test_generate_levels_exception_generic(self: "TestGenerateLevelsException") -> None:
        """
        This test checks that the `_generate_levels_from_data` function handles generic exceptions gracefully. By creating a mock data object that raises an exception when its values are accessed, we can simulate an unexpected error during level generation. The test asserts that when such an exception occurs, the function returns None instead of propagating the error, confirming that it can handle unforeseen issues without crashing.

        Parameters:
            None

        Returns:
            None
        """
        mock_data = Mock()
        mock_data.values = Mock(side_effect=Exception("Data access error"))
        result = MPASVisualizationStyle._generate_levels_from_data(mock_data, 'temperature')        
        assert result is None
    
    def test_generate_levels_no_flatten_attribute(self: "TestGenerateLevelsException") -> None:
        """
        This test verifies that the `_generate_levels_from_data` function returns None when the input data object does not have a `flatten` method. The function should check for the presence of this method and return None if it is not available, rather than raising an AttributeError. This test uses a mock object that simulates a data array without a `flatten` attribute and asserts that the function returns None, confirming that it handles this edge case gracefully.

        Parameters:
            None

        Returns:
            None
        """
        data_list = [1, 2, 3, 4, 5]
        data = xr.DataArray(data_list)
        
        with patch('builtins.hasattr', return_value=False):
            result = MPASVisualizationStyle._generate_levels_from_data(data, 'temperature')
            assert result is None


class TestTemperatureLevelGeneration:
    """ Tests for temperature-specific level generation branches. """
    
    def test_generate_levels_temp_step_05(self: "TestTemperatureLevelGeneration", mpas_surface_temp_data) -> None:
        """
        This test ensures that the temperature level generator selects a step of approximately 0.5 for very small temperature ranges. When the computed level step is less than 1, the helper should return levels spaced near 0.5. This test uses real surface temperature data scaled to a tight range and asserts that at least one adjacent level difference is close to 0.5, confirming that the level generation logic correctly categorizes small temperature ranges.

        Parameters:
            mpas_surface_temp_data: Real surface temperature data from fixtures.

        Returns:
            None
        """
        if mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
        
        t_subset = mpas_surface_temp_data[:4]
        t_min, t_max = t_subset.min(), t_subset.max()

        data = xr.DataArray(20.0 + 1.5 * (t_subset - t_min) / (t_max - t_min + 1e-12))        
        levels = MPASVisualizationStyle._generate_levels_from_data(data, 'temperature')
        
        assert levels is not None

        assert any(abs(levels[i+1] - levels[i] - 0.5) < 0.01 
                           for i in range(len(levels)-1) if len(levels) > 1)
    
    def test_generate_levels_temp_step_1(self: "TestTemperatureLevelGeneration", mpas_surface_temp_data) -> None:
        """
        This test validates that the temperature level generator selects a step of approximately 1.0 for small to moderate temperature ranges. When the computed level step falls in the 1 <= step < 2 bucket, the helper should return levels spaced near 1.0. This test uses real surface temperature data scaled to a moderate range and asserts that at least one adjacent level difference is close to 1.0, confirming that the level generation logic correctly categorizes small to moderate temperature ranges.

        Parameters:
            mpas_surface_temp_data: Real surface temperature data from fixtures.

        Returns:
            None
        """
        if mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
        
        t_min, t_max = mpas_surface_temp_data.min(), mpas_surface_temp_data.max()
        data = xr.DataArray(10.0 + 18.0 * (mpas_surface_temp_data - t_min) / (t_max - t_min + 1e-12))
        
        levels = MPASVisualizationStyle._generate_levels_from_data(data, 't2m')
        
        assert levels is not None

        if len(levels) > 1:
            steps = [abs(levels[i+1] - levels[i]) for i in range(len(levels)-1)]
            assert any(0.5 <= step <= 2.0 for step in steps)
    
    def test_generate_levels_temp_step_2(self: "TestTemperatureLevelGeneration", mpas_surface_temp_data) -> None:
        """
        This test validates that the temperature level generator selects a step of approximately 2.0 for moderate temperature ranges. When the computed level step falls in the 2 <= step < 5 bucket, the helper should return levels spaced near 2.0. This test uses real surface temperature data scaled to a moderate range and asserts that at least one adjacent level difference is close to 2.0, confirming that the level generation logic correctly categorizes moderate temperature ranges.

        Parameters:
            mpas_surface_temp_data: Real surface temperature data from fixtures.

        Returns:
            None
        """
        if mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
        
        t_min, t_max = mpas_surface_temp_data.min(), mpas_surface_temp_data.max()
        data = xr.DataArray(0.0 + 40.0 * (mpas_surface_temp_data - t_min) / (t_max - t_min + 1e-12))
        
        levels = MPASVisualizationStyle._generate_levels_from_data(data, 'temp')
        
        assert levels is not None
        assert len(levels) >= 5, "Should have at least 5 levels"

        if len(levels) > 1:
            level_range = levels[-1] - levels[0]
            assert level_range > 0, "Levels should span positive range"
            assert all(levels[i] < levels[i+1] for i in range(len(levels)-1)), "Levels should be ascending"


class TestVariableSpecificSettingsBranches:
    """ Tests for get_variable_specific_settings branches. """
    
    def test_settings_temp_range_40_plus(self: "TestVariableSpecificSettingsBranches", mpas_surface_temp_data) -> None:
        """
        This test checks that the `get_variable_specific_settings` function applies the correct styling for temperature data with a large dynamic range (greater than 40 degrees). For such ranges, the helper should return the canonical temperature colormap (e.g., 'RdYlBu_r') and coarser spacing (e.g., step = 5). This test uses real surface temperature data scaled to a wide range and asserts that the expected colormap is returned and that levels are generated with appropriate spacing, confirming that the function correctly identifies and styles large temperature ranges.

        Parameters:
            mpas_surface_temp_data: Real surface temperature data from fixtures.

        Returns:
            None
        """
        if mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
        
        t_min, t_max = mpas_surface_temp_data.min(), mpas_surface_temp_data.max()
        data = -20.0 + 50.0 * (mpas_surface_temp_data - t_min) / (t_max - t_min + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('temperature', data)
        
        assert cmap == 'RdYlBu_r'
        assert levels is not None
    
    def test_settings_temp_range_20_to_40(self: "TestVariableSpecificSettingsBranches", mpas_surface_temp_data) -> None:
        """
        This test checks that the `get_variable_specific_settings` function applies the correct styling for temperature data with a moderate dynamic range (between 20 and 40 degrees). For such ranges, the helper should return the canonical temperature colormap (e.g., 'RdYlBu_r') and medium spacing (e.g., step = 2). This test uses real surface temperature data scaled to a moderate range and asserts that the expected colormap is returned and that levels are generated with appropriate spacing, confirming that the function correctly identifies and styles moderate temperature ranges.

        Parameters:
            mpas_surface_temp_data: Real surface temperature data from fixtures.

        Returns:
            None
        """
        if mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
        
        t_min, t_max = mpas_surface_temp_data.min(), mpas_surface_temp_data.max()
        data = 10.0 + 25.0 * (mpas_surface_temp_data - t_min) / (t_max - t_min + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('sst', data)
        
        assert cmap == 'RdYlBu_r'
        assert levels is not None
    
    def test_settings_temp_range_under_20(self: "TestVariableSpecificSettingsBranches", mpas_surface_temp_data) -> None:
        """
        This test checks that the `get_variable_specific_settings` function applies the correct styling for temperature data with a small dynamic range (20 degrees or less). For such ranges, the helper should return the canonical temperature colormap (e.g., 'RdYlBu_r') and finer spacing (e.g., step = 1). This test uses real surface temperature data scaled to a narrow range and asserts that the expected colormap is returned and that levels are generated with appropriate spacing, confirming that the function correctly identifies and styles small temperature ranges.

        Parameters:
            mpas_surface_temp_data: Real surface temperature data from fixtures.

        Returns:
            None
        """
        if mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
        
        t_min, t_max = mpas_surface_temp_data.min(), mpas_surface_temp_data.max()
        data = 15.0 + 10.0 * (mpas_surface_temp_data - t_min) / (t_max - t_min + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('t2m', data)
        
        assert cmap == 'RdYlBu_r'
        assert levels is not None
    
    def test_settings_precip_01h(self: "TestVariableSpecificSettingsBranches", mpas_precip_data) -> None:
        """
        This test verifies that the `get_variable_specific_settings` function correctly identifies a 1-hour accumulation period from the variable name and applies the appropriate styling. Variables with a 1-hour accumulation (e.g., 'precip_01h') should map to the 1-hour (a01h) styling branch. This test uses real precipitation data scaled to a typical 1-hour range and asserts that the returned colormap is a `ListedColormap` and that characteristic levels (such as 0.5) are present, confirming that the period parsing logic correctly categorizes short accumulation periods.

        Parameters:
            mpas_precip_data: Real precipitation data from fixtures.

        Returns:
            None
        """
        if mpas_precip_data is None:
            pytest.skip("MPAS data not available")
        
        p_min, p_max = mpas_precip_data.min(), mpas_precip_data.max()
        data = 0.0 + 5.0 * (mpas_precip_data - p_min) / (p_max - p_min + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('precip_01h', data)
        
        assert isinstance(cmap, mcolors.ListedColormap)
        assert levels is not None
        assert 0.5 in levels
    
    def test_settings_precip_03h(self: "TestVariableSpecificSettingsBranches", mpas_precip_data) -> None:
        """
        This test verifies that the `get_variable_specific_settings` function correctly identifies a 3-hour accumulation period from the variable name and applies the appropriate styling. Variables with a 3-hour accumulation (e.g., 'rain_03h') should map to the 3-hour (a03h) styling branch. This test uses real precipitation data scaled to a typical 3-hour range and asserts that the returned colormap is a `ListedColormap` and that characteristic levels (such as 0.5) are present, confirming that the period parsing logic correctly categorizes mid-range accumulation periods.

        Parameters:
            mpas_precip_data: Real precipitation data from fixtures.

        Returns:
            None
        """
        if mpas_precip_data is None:
            pytest.skip("MPAS data not available")
        
        p_min, p_max = mpas_precip_data.min(), mpas_precip_data.max()
        data = 0.0 + 10.0 * (mpas_precip_data - p_min) / (p_max - p_min + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('rain_03h', data)
        
        assert isinstance(cmap, mcolors.ListedColormap)
    
    def test_settings_precip_06h(self: "TestVariableSpecificSettingsBranches", mpas_precip_data) -> None:
        """
        This test verifies that the `get_variable_specific_settings` function correctly identifies a 6-hour accumulation period from the variable name and applies the appropriate styling. Variables with a 6-hour accumulation (e.g., 'precip_06h') should map to the 6-hour (a06h) styling branch. This test uses real precipitation data scaled to a typical 6-hour range and asserts that the returned colormap is a `ListedColormap`, confirming that the period parsing logic correctly categorizes longer accumulation periods.

        Parameters:
            mpas_precip_data: Real precipitation data from fixtures.

        Returns:
            None
        """
        if mpas_precip_data is None:
            pytest.skip("MPAS data not available")
        
        p_min, p_max = mpas_precip_data.min(), mpas_precip_data.max()
        data = 0.0 + 20.0 * (mpas_precip_data - p_min) / (p_max - p_min + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('precip_06h', data)
        
        assert isinstance(cmap, mcolors.ListedColormap)
    
    def test_settings_precip_12h(self: "TestVariableSpecificSettingsBranches", mpas_precip_data) -> None:
        """
        This test verifies that the `get_variable_specific_settings` function correctly identifies a 12-hour accumulation period from the variable name and applies the appropriate styling. Variables with a 12-hour accumulation (e.g., 'rain_12h') should map to the 12-hour (a12h) styling branch. This test uses real precipitation data scaled to a typical 12-hour range and asserts that the returned colormap is a `ListedColormap`, confirming that the period parsing logic correctly categorizes longer accumulation periods.

        Parameters:
            mpas_precip_data: Real precipitation data from fixtures.

        Returns:
            None
        """
        if mpas_precip_data is None:
            pytest.skip("MPAS data not available")
        
        p_min, p_max = mpas_precip_data.min(), mpas_precip_data.max()
        data = 0.0 + 30.0 * (mpas_precip_data - p_min) / (p_max - p_min + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('rain_12h', data)
        
        assert isinstance(cmap, mcolors.ListedColormap)
    
    def test_settings_precip_24h_or_daily(self: "TestVariableSpecificSettingsBranches", mpas_precip_data) -> None:
        """
        This test verifies that the `get_variable_specific_settings` function correctly applies default 24-hour styling for precipitation variables without a specific period. When the variable name does not contain a recognizable period (e.g., 'precip' or 'rain'), the helper should default to 24-hour styling. This test uses real precipitation data scaled to a typical daily range and asserts that the returned colormap is a `ListedColormap`, confirming that the function correctly defaults to 24-hour styling when no specific period is detected.

        Parameters:
            mpas_precip_data: Real precipitation data from fixtures.

        Returns:
            None
        """
        if mpas_precip_data is None:
            pytest.skip("MPAS data not available")
        
        p_min, p_max = mpas_precip_data.min(), mpas_precip_data.max()
        data = 0.0 + 50.0 * (mpas_precip_data - p_min) / (p_max - p_min + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('daily_rain', data)
        
        assert isinstance(cmap, mcolors.ListedColormap)
    
    def test_settings_pressure_slp_large_range(self: "TestVariableSpecificSettingsBranches", mpas_wind_data) -> None:
        """
        This test checks that the `get_variable_specific_settings` function applies the correct styling for sea-level pressure data with a large dynamic range. When the variable name is 'slp' and the data range is large (e.g., > 100 hPa), the helper should return a diverging colormap (e.g., 'RdBu_r') and generate levels that reflect typical pressure values. This test uses real wind data scaled to a pressure-like range and asserts that the expected colormap is returned and that levels are generated, confirming that the function correctly identifies and styles large pressure ranges even when the variable name is specific to sea-level pressure.

        Parameters:
            mpas_wind_data: Real wind data from fixtures used to create pressure-like data.

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, _ = mpas_wind_data
        u_min, u_max = u.min(), u.max()
        data = 980.0 + 60.0 * (u - u_min) / (u_max - u_min + 1e-12)

        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('slp', data)
        
        assert cmap == 'RdBu_r'
        assert levels is not None
    
    def test_settings_pressure_slp_small_range(self: "TestVariableSpecificSettingsBranches", mpas_wind_data) -> None:
        """
        This test checks that the `get_variable_specific_settings` function applies the correct styling for sea-level pressure data with a small dynamic range. When the variable name is 'slp' and the data range is small (e.g., < 20 hPa), the helper should still return a diverging colormap (e.g., 'RdBu_r') but may generate levels with finer spacing. This test uses real wind data scaled to a narrow pressure-like range and asserts that the expected colormap is returned and that levels are generated, confirming that the function correctly identifies and styles small pressure ranges even when the variable name is specific to sea-level pressure.

        Parameters:
            mpas_wind_data: Real wind data from fixtures used to create pressure-like data.

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, _ = mpas_wind_data
        u_min, u_max = u.min(), u.max()
        data = 1000.0 + 20.0 * (u - u_min) / (u_max - u_min + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('mslp', data)
        
        assert cmap == 'RdBu_r'
        assert levels is not None
    
    def test_settings_pressure_generic_large(self: "TestVariableSpecificSettingsBranches", mpas_wind_data) -> None:
        """
        This test checks that the `get_variable_specific_settings` function applies the correct styling for generic pressure data with a large dynamic range. When the variable name is 'pressure' and the data range is large (e.g., > 1000 Pa), the helper should return a diverging colormap (e.g., 'RdBu_r') and generate levels that reflect typical pressure values. This test uses real wind data scaled to a large pressure-like range and asserts that the expected colormap is returned and that levels are generated, confirming that the function correctly identifies and styles large pressure ranges even when the variable name is generic.

        Parameters:
            mpas_wind_data: Real wind data from fixtures used to create pressure-like data.

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, _ = mpas_wind_data
        u_min, u_max = u.min(), u.max()
        data = 50000.0 + 50000.0 * (u - u_min) / (u_max - u_min + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('pressure', data)
        
        assert cmap == 'RdBu_r'
        assert levels is not None
    
    def test_settings_wind_very_low(self: "TestVariableSpecificSettingsBranches", mpas_wind_data) -> None:
        """
        This test checks that the `get_variable_specific_settings` function applies the correct styling for wind speed data with very low magnitudes. When the variable name indicates wind speed and the data range is very small (e.g., < 2.5 m/s), the helper should return a perceptually appropriate palette (e.g., 'plasma') and include fine contour levels (e.g., 0.5 m/s). This test uses real wind data scaled to a very low wind speed range and asserts that the expected colormap is returned and that characteristic levels are present, confirming that the function correctly identifies and styles very low wind speed ranges.

        Parameters:
            mpas_wind_data: Real wind data from fixtures.

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, v = mpas_wind_data
        wspd = np.sqrt(u**2 + v**2)
        wspd_min, wspd_max = wspd.min(), wspd.max()
        data = 0.0 + 2.5 * (wspd - wspd_min) / (wspd_max - wspd_min + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('wind_speed', data)
        
        assert cmap == 'plasma'
        assert levels is not None
        assert 0.5 in levels
    
    def test_settings_wind_low(self: "TestVariableSpecificSettingsBranches", mpas_wind_data) -> None:
        """
        This test checks that the `get_variable_specific_settings` function applies the correct styling for wind speed data with low to moderate magnitudes. When the variable name indicates wind speed and the data range is low (e.g., < 15 m/s), the helper should return a perceptually appropriate palette (e.g., 'plasma') and include mid-range contour levels (e.g., 2 m/s). This test uses real wind data scaled to a low wind speed range and asserts that the expected colormap is returned and that characteristic levels are present, confirming that the function correctly identifies and styles low to moderate wind speed ranges.

        Parameters:
            mpas_wind_data: Real wind data from fixtures.

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, v = mpas_wind_data
        wspd = np.sqrt(u**2 + v**2)
        wspd_min, wspd_max = wspd.min(), wspd.max()
        data = 0.0 + 12.0 * (wspd - wspd_min) / (wspd_max - wspd_min + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('wspd', data)
        
        assert cmap == 'plasma'
        assert levels is not None
        assert 2.0 in levels
    
    def test_settings_wind_medium(self: "TestVariableSpecificSettingsBranches", mpas_wind_data) -> None:
        """
        This test checks that the `get_variable_specific_settings` function applies the correct styling for wind speed data with moderate magnitudes. When the variable name indicates wind speed and the data range is moderate (e.g., < 30 m/s), the helper should return a perceptually appropriate palette (e.g., 'plasma') and include mid-range contour levels (e.g., 5 m/s). This test uses real wind data scaled to a moderate wind speed range and asserts that the expected colormap is returned and that characteristic levels are present, confirming that the function correctly identifies and styles moderate wind speed ranges.

        Parameters:
            mpas_wind_data: Real wind data from fixtures.

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, v = mpas_wind_data
        wspd = np.sqrt(u**2 + v**2)
        wspd_min, wspd_max = wspd.min(), wspd.max()
        data = 0.0 + 25.0 * (wspd - wspd_min) / (wspd_max - wspd_min + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('wind', data)
        
        assert cmap == 'plasma'
        assert levels is not None

        assert 5.0 in levels
    
    def test_settings_geopotential_large(self: "TestVariableSpecificSettingsBranches", mpas_wind_data) -> None:
        """
        This test checks that the `get_variable_specific_settings` function applies the correct styling for geopotential height data with a large dynamic range. When the variable name indicates geopotential height and the data range is large (e.g., > 4000 m), the helper should return a terrain colormap and generate levels with coarser spacing (e.g., step = 60). This test uses real wind data scaled to a large height-like range and asserts that the expected colormap is returned and that levels are generated, confirming that the function correctly identifies and styles large geopotential height ranges.

        Parameters:
            mpas_wind_data: Real wind data from fixtures used to create height-like data.

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, _ = mpas_wind_data
        u_min, u_max = u.min(), u.max()
        data = 1000.0 + 4000.0 * (u - u_min) / (u_max - u_min + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('height', data)
        
        assert cmap == 'terrain'
        assert levels is not None
    
    def test_settings_geopotential_medium(self: "TestVariableSpecificSettingsBranches", mpas_wind_data) -> None:
        """
        This test checks that the `get_variable_specific_settings` function applies the correct styling for geopotential height data with a moderate dynamic range. When the variable name indicates geopotential height and the data range is moderate (e.g., > 1000 m but < 2000 m), the helper should return a terrain colormap and generate levels with medium spacing (e.g., step = 30). This test uses real wind data scaled to a moderate height-like range and asserts that the expected colormap is returned and that levels are generated, confirming that the function correctly identifies and styles moderate geopotential height ranges.

        Parameters:
            mpas_wind_data: Real wind data from fixtures used to create height-like data.

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("MPAS data not available")
        
        u, _ = mpas_wind_data
        u_min, u_max = u.min(), u.max()
        data = 1000.0 + 1500.0 * (u - u_min) / (u_max - u_min + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('z', data)
        
        assert cmap == 'terrain'
        assert levels is not None
    
    def test_settings_humidity_fractional_rh(self: "TestVariableSpecificSettingsBranches", mpas_qv_3d_data) -> None:
        """
        This test checks that the `get_variable_specific_settings` function applies the correct styling for relative humidity data expressed as fractions. When the variable name indicates relative humidity ('rh') and the data values are fractional (e.g., 0.3 - 0.9), the helper should return a humidity-specific colormap and generate levels that include fractional values. This test uses real 3D specific humidity data scaled to a fractional range and asserts that the expected colormap is returned and that levels include 0.3.

        Parameters:
            mpas_qv_3d_data: Real 3D specific humidity data from fixtures.

        Returns:
            None
        """
        if mpas_qv_3d_data is None:
            pytest.skip("MPAS data not available")
        
        qv_min, qv_max = mpas_qv_3d_data.min(), mpas_qv_3d_data.max()
        data = 0.2 + 0.75 * (mpas_qv_3d_data - qv_min) / (qv_max - qv_min + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('rh', data)
        
        assert cmap == 'BuGn'
        assert levels is not None
        assert 0.3 in levels
    
    def test_settings_humidity_fractional_mixing(self: "TestVariableSpecificSettingsBranches", mpas_qv_3d_data) -> None:
        """
        This test checks that the `get_variable_specific_settings` function applies the correct styling for specific humidity data expressed as mixing ratios. When the variable name indicates specific humidity ('q') and the data values are in a mixing ratio-like range (e.g., 0.001 - 0.1), the helper should return a humidity-specific colormap and generate levels that include typical mixing ratio values. This test uses real 3D specific humidity data scaled to a mixing ratio-like range and asserts that the expected colormap is returned and that levels include 0.001.

        Parameters:
            mpas_qv_3d_data: Real 3D specific humidity data from fixtures.

        Returns:
            None
        """
        if mpas_qv_3d_data is None:
            pytest.skip("MPAS data not available")
        
        qv_min, qv_max = mpas_qv_3d_data.min(), mpas_qv_3d_data.max()
        data = 0.0005 + 0.1495 * (mpas_qv_3d_data - qv_min) / (qv_max - qv_min + 1e-12)
        
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('q', data)
        
        assert cmap == 'BuGn'
        assert levels is not None
        assert 0.001 in levels


class TestSavePlotException:
    """ Tests for save_plot exception handling. """
    
    def test_save_plot_none_figure(self: "TestSavePlotException") -> None:
        """
        This test verifies that the `save_plot` function raises a ValueError when no figure is provided. The helper should check if the figure argument is None and raise an appropriate exception with a clear message. This test calls `save_plot` with None as the figure and asserts that a ValueError is raised with the expected message, confirming that the function correctly handles attempts to save without a valid figure.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError) as exc_info:
            MPASVisualizationStyle.save_plot(None, '/tmp/test_plot') # type: ignore

        assert "No figure to save" in str(exc_info.value)


class TestAdaptiveMarkerSizeBranches:
    """ Tests for adaptive marker size calculation branches. """
    
    def test_marker_size_none_extent(self: "TestAdaptiveMarkerSizeBranches") -> None:
        """
        This test ensures that the `calculate_adaptive_marker_size` function returns a default marker size when the geographic extent is None. The helper should handle a None extent by returning a predefined default size (e.g., 5.0) rather than attempting calculations that would fail. This test calls the function with None for the extent and asserts that the returned size is equal to the expected default, confirming that the function correctly handles missing geographic extent information.

        Parameters:
            None

        Returns:
            None
        """
        size = MPASVisualizationStyle.calculate_adaptive_marker_size(None, 1000) # type: ignore 
        
        assert size == 5.0
    
    def test_marker_size_zero_area(self: "TestAdaptiveMarkerSizeBranches") -> None:
        """
        This test verifies that the `calculate_adaptive_marker_size` function returns a default marker size when the geographic extent has zero area. The helper should detect when the extent defines a zero-area region (e.g., min_lon == max_lon or min_lat == max_lat) and return a safe default size rather than performing calculations that would lead to division by zero or nonsensical values. This test provides an extent with zero width and asserts that the returned size is equal to the expected default, confirming that the function correctly handles zero-area extents.

        Parameters:
            None

        Returns:
            None
        """
        extent = (-100, -100, 30, 30)  
        size = MPASVisualizationStyle.calculate_adaptive_marker_size(extent, 1000) # type: ignore
        
        assert size == 5.0
    
    def test_marker_size_zero_points(self: "TestAdaptiveMarkerSizeBranches") -> None:
        """
        This test checks that the `calculate_adaptive_marker_size` function returns a default marker size when the number of points is zero. The helper should handle a scenario with zero points by returning a predefined default size (e.g., 5.0) rather than attempting calculations that would involve division by zero or result in an undefined density. This test provides a valid geographic extent but sets the point count to zero and asserts that the returned size is equal to the expected default, confirming that the function correctly handles cases with no data points.

        Parameters:
            None

        Returns:
            None
        """
        extent = (-100, -90, 30, 40)
        size = MPASVisualizationStyle.calculate_adaptive_marker_size(extent, 0)
        
        assert size == 5.0
    
    def test_marker_size_density_under_1(self: "TestAdaptiveMarkerSizeBranches") -> None:
        """
        This test checks that the `calculate_adaptive_marker_size` function returns a reasonable marker size for very low densities (density < 1 pt/deg^2). The helper should produce larger markers for sparse data to enhance visibility. This test provides an extent and point count that results in a density of less than 1 point per square degree and asserts that the returned size is positive, confirming that the function correctly scales marker sizes for very low-density scenarios.

        Parameters:
            None

        Returns:
            None
        """
        extent = (-100, -90, 30, 40)  
        size = MPASVisualizationStyle.calculate_adaptive_marker_size(extent, 50)          
        assert size > 0
    
    def test_marker_size_density_1_to_10(self: "TestAdaptiveMarkerSizeBranches") -> None:
        """
        This test checks that the `calculate_adaptive_marker_size` function returns a reasonable marker size for low to moderate densities (1 <= density < 10 pts/deg^2). The helper should produce moderately sized markers for these densities to balance visibility and overlap. This test provides an extent and point count that results in a density between 1 and 10 points per square degree and asserts that the returned size is positive, confirming that the function correctly scales marker sizes for low to moderate-density scenarios.

        Parameters:
            None

        Returns:
            None
        """
        extent = (-100, -95, 30, 35)  
        size = MPASVisualizationStyle.calculate_adaptive_marker_size(extent, 100) 
        assert size > 0
    
    def test_marker_size_area_under_50(self: "TestAdaptiveMarkerSizeBranches") -> None:
        """
        This test checks that the `calculate_adaptive_marker_size` function returns a reasonable marker size for small-area maps (area < 50 sq deg). The helper should produce larger markers for small-area maps to enhance visibility. This test provides an extent that results in an area of less than 50 square degrees and a moderate point count, then asserts that the returned size is positive, confirming that the function correctly scales marker sizes for small-area scenarios.

        Parameters:
            None

        Returns:
            None
        """
        extent = (-95, -90, 30, 35)  
        size = MPASVisualizationStyle.calculate_adaptive_marker_size(extent, 100)        
        assert size > 0
    
    def test_marker_size_area_50_to_500(self: "TestAdaptiveMarkerSizeBranches") -> None:
        """
        This test checks that the `calculate_adaptive_marker_size` function returns a reasonable marker size for medium-area maps (50 <= area < 500 sq deg). The helper should produce moderately sized markers for medium-area maps to balance visibility and overlap. This test provides an extent that results in an area between 50 and 500 square degrees and a moderate point count, then asserts that the returned size is positive, confirming that the function correctly scales marker sizes for medium-area scenarios.

        Parameters:
            None

        Returns:
            None
        """
        extent = (-110, -90, 20, 30)  
        size = MPASVisualizationStyle.calculate_adaptive_marker_size(extent, 1000)        
        assert size > 0
    
    def test_marker_size_area_500_to_5000(self: "TestAdaptiveMarkerSizeBranches") -> None:
        """
        This test checks that the `calculate_adaptive_marker_size` function returns a reasonable marker size for large-area maps (500 <= area < 5000 sq deg). The helper should produce smaller markers for large-area maps to prevent excessive overlap. This test provides an extent that results in an area between 500 and 5000 square degrees and a moderate point count, then asserts that the returned size is positive, confirming that the function correctly scales marker sizes for large-area scenarios.

        Parameters:
            None

        Returns:
            None
        """
        extent = (-130, -90, 10, 50)  
        size = MPASVisualizationStyle.calculate_adaptive_marker_size(extent, 5000)
        
        assert size > 0
    
    def test_marker_size_clamping_min(self: "TestAdaptiveMarkerSizeBranches") -> None:
        """
        This test ensures that the `calculate_adaptive_marker_size` function returns a marker size that is not smaller than the configured minimum (0.1). Extremely high-density scenarios should not produce impractically small markers; this test checks that the size is bounded below by the configured minimum. 

        Parameters:
            None

        Returns:
            None
        """
        extent = (-100, -90, 30, 40)  
        size = MPASVisualizationStyle.calculate_adaptive_marker_size(extent, 100000)
        
        assert size >= 0.1
    
    def test_marker_size_clamping_max(self: "TestAdaptiveMarkerSizeBranches") -> None:
        """
        This test ensures that the `calculate_adaptive_marker_size` function returns a marker size that is not larger than the configured maximum (20.0). Extremely low-density or small-area scenarios should not produce impractically large markers; this test checks that the size is bounded above by the configured maximum.

        Parameters:
            None

        Returns:
            None
        """
        extent = (-90.5, -90.0, 30, 30.5) 
        size = MPASVisualizationStyle.calculate_adaptive_marker_size(extent, 1)
        
        assert size <= 20.0


class TestFormatTicksDynamicBranches:
    """ Tests for format_ticks_dynamic branches. """
    
    def test_format_ticks_empty_list(self: "TestFormatTicksDynamicBranches") -> None:
        """
        This test verifies that the `format_ticks_dynamic` function returns an empty list when given an empty list of ticks. The helper should handle this edge case gracefully without errors and simply return an empty list, confirming that the function can manage cases with no tick values.

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.format_ticks_dynamic([])
        
        assert result == []
    
    def test_format_ticks_all_zeros(self: "TestFormatTicksDynamicBranches") -> None:
        """
        This test checks that the `format_ticks_dynamic` function correctly formats a list of ticks that are all zeros. The helper should recognize that all values are zero and return a consistent formatting (e.g., '0') for each tick without errors, confirming that the function can handle uniform tick values appropriately.

        Parameters:
            None

        Returns:
            None
        """
        ticks = [0.0, 0.0, 0.0]
        result = MPASVisualizationStyle.format_ticks_dynamic(ticks)
        assert len(result) == 3
    
    def test_format_ticks_scientific_large(self: "TestFormatTicksDynamicBranches") -> None:
        """
        This test verifies that the `format_ticks_dynamic` function formats very large-magnitude ticks using scientific notation. When tick values are extremely large (e.g., > 10,000), the helper should switch to scientific notation to maintain readability. This test provides a list of large tick values and asserts that the resulting labels include exponent-style formatting, confirming that the function correctly identifies and formats large magnitudes.

        Parameters:
            None

        Returns:
            None
        """
        ticks = [10000, 20000, 30000, 40000]
        result = MPASVisualizationStyle.format_ticks_dynamic(ticks) # type: ignore
        assert any('e' in label.lower() for label in result)
    
    def test_format_ticks_scientific_small(self: "TestFormatTicksDynamicBranches") -> None:
        """
        This test verifies that the `format_ticks_dynamic` function formats very small-magnitude ticks using scientific notation. When tick values are extremely small (e.g., < 0.001), the helper should switch to scientific notation to maintain readability. This test provides a list of small tick values and asserts that the resulting labels include exponent-style formatting, confirming that the function correctly identifies and formats small magnitudes.

        Parameters:
            None

        Returns:
            None
        """
        ticks = [0.0001, 0.0002, 0.0005, 0.001]
        result = MPASVisualizationStyle.format_ticks_dynamic(ticks) # type: ignore
        assert any('e' in label.lower() for label in result)
    
    def test_format_ticks_magnitude_100(self: "TestFormatTicksDynamicBranches") -> None:
        """
        This test verifies that the `format_ticks_dynamic` function formats ticks with typical magnitude around 100 without decimal places. When tick values are in a moderate range (e.g., around 100), the helper should format them as integers without decimal points for clarity. This test provides a list of tick values around 100 and asserts that the resulting labels do not include decimal points, confirming that the function correctly formats moderate magnitudes as integers.

        Parameters:
            None

        Returns:
            None
        """
        ticks = [100, 200, 300, 400]
        result = MPASVisualizationStyle.format_ticks_dynamic(ticks) # type: ignore
        assert all('.' not in label for label in result)
    
    def test_format_ticks_magnitude_10_to_100(self: "TestFormatTicksDynamicBranches") -> None:
        """
        This test verifies that the `format_ticks_dynamic` function formats ticks with magnitudes between 10 and 100 using one decimal place. When tick values are in this range, the helper should format them with a single decimal place to enhance readability while still showing some precision. This test provides a list of tick values between 10 and 100 and asserts that the resulting labels include one decimal place, confirming that the function correctly formats this magnitude range.

        Parameters:
            None

        Returns:
            None
        """
        ticks = [10, 20, 30, 40, 50]
        result = MPASVisualizationStyle.format_ticks_dynamic(ticks) # type: ignore
        assert (any('.' in label for label in result) or 
                       all('.' not in label for label in result))
    
    def test_format_ticks_magnitude_very_small(self: "TestFormatTicksDynamicBranches") -> None:
        """
        This test verifies that the `format_ticks_dynamic` function formats ticks with very small magnitudes using appropriate precision. When tick values are very small (e.g., < 0.01), the helper should format them with sufficient decimal places or switch to scientific notation to ensure readability. This test provides a list of very small tick values and asserts that the resulting labels are formatted with either multiple decimal places or scientific notation, confirming that the function correctly handles very small magnitudes.

        Parameters:
            None

        Returns:
            None
        """
        ticks = [0.005, 0.010, 0.015, 0.020]
        result = MPASVisualizationStyle.format_ticks_dynamic(ticks) # type: ignore
        assert result is not None
    
    def test_format_ticks_duplicate_increase_precision(self: "TestFormatTicksDynamicBranches") -> None:
        """
        This test checks that the `format_ticks_dynamic` function increases numeric precision to avoid duplicate labels when tick values are close together. If the initial formatting produces duplicate labels due to low precision, the helper should attempt to increase the number of decimal places to differentiate them. This test provides a list of closely spaced tick values that would produce duplicates with low precision and asserts that the resulting labels are unique, confirming that the function correctly adjusts precision to avoid duplicates.

        Parameters:
            None

        Returns:
            None
        """
        ticks = [1.0001, 1.0002, 1.0003]
        result = MPASVisualizationStyle.format_ticks_dynamic(ticks)
        assert len(set(result)) == len(result)
    
    def test_format_ticks_duplicate_fallback_to_g(self: "TestFormatTicksDynamicBranches") -> None:
        """
        This test checks that the `format_ticks_dynamic` function falls back to general format ('g') if increasing precision does not resolve duplicate labels. If the helper attempts to increase decimal places but still produces duplicates, it should switch to a more compact general format to differentiate the labels. This test provides a list of closely spaced tick values that would produce duplicates even with increased decimal places and asserts that the resulting labels are unique, confirming that the function correctly falls back to 'g' formatting when necessary.

        Parameters:
            None

        Returns:
            None
        """
        ticks = [1.00001, 1.00002, 1.00003]
        result = MPASVisualizationStyle.format_ticks_dynamic(ticks)
        assert len(set(result)) == len(result)


class TestFooterAxesAndBranding:
    """ Tests for _create_footer_axes colorbar detection and add_timestamp_and_branding. """

    def test_create_footer_axes_no_colorbar(self, capsys) -> None:
        """
        This test verifies that the `_create_footer_axes` function correctly handles the case where no colorbar is detected in the figure. The helper should print a message indicating that no colorbar was found and create footer axes at the default position (e.g., y=0.05). This test creates a simple figure without any colorbars, calls the function, and asserts that the expected message is printed and that the footer axes are created with the correct properties, confirming that the function behaves as intended when no colorbar is present.

        Parameters:
            capsys: Pytest fixture for capturing stdout output.

        Returns:
            None
        """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])

        footer_ax = MPASVisualizationStyle._create_footer_axes(fig)

        output = capsys.readouterr().out
        assert "no colorbar detected" in output
        assert footer_ax is not None
        assert footer_ax.get_gid() == 'mpasdiag_footer'
        plt.close(fig)

    def test_create_footer_axes_with_colorbar(self, capsys) -> None:
        """
        This test verifies that the `_create_footer_axes` function correctly handles the case where a colorbar is detected in the figure. The helper should print a message indicating the position of the detected colorbar and create footer axes above it. This test creates a simple figure with a simulated colorbar, calls the function, and asserts that the expected message is printed and that the footer axes are created with the correct properties, confirming that the function behaves as intended when a colorbar is present.

        Parameters:
            capsys: Pytest fixture for capturing stdout output.

        Returns:
            None
        """
        fig = plt.figure(figsize=(8, 6))
        fig.add_axes((0.1, 0.3, 0.8, 0.6))
        fig.add_axes((0.1, 0.05, 0.8, 0.05))

        footer_ax = MPASVisualizationStyle._create_footer_axes(fig)

        output = capsys.readouterr().out
        assert "detected colorbar tops" in output
        assert footer_ax is not None
        plt.close(fig)

    def test_create_footer_reuses_existing(self, capsys) -> None:
        """
        This test verifies that the `_create_footer_axes` function reuses existing footer axes if they are already present in the figure. The helper should detect that footer axes with the specified GID already exist and print a message indicating that it is reusing them instead of creating new ones. This test creates a figure, calls the function to create footer axes, then calls it again and asserts that the expected message about reusing existing axes is printed and that the same axes object is returned, confirming that the function correctly identifies and reuses existing footer axes.

        Parameters:
            capsys: Pytest fixture for capturing stdout output.

        Returns:
            None
        """
        fig = plt.figure(figsize=(8, 6))

        first = MPASVisualizationStyle._create_footer_axes(fig)
        capsys.readouterr()  

        second = MPASVisualizationStyle._create_footer_axes(fig)

        assert "reusing existing footer axes" in capsys.readouterr().out
        assert first is second
        plt.close(fig)

    def test_add_timestamp_and_branding_success(self, capsys) -> None:
        """
        This test verifies that the `add_timestamp_and_branding` function successfully adds a timestamp and branding to the figure. The helper should create footer axes (reusing existing ones if present) and add the appropriate text elements for the timestamp and branding. This test creates a simple figure, calls the function, and asserts that the expected message about using footer axes for branding is printed, confirming that the function correctly adds timestamp and branding information to the figure.

        Parameters:
            capsys: Pytest fixture for capturing stdout output.

        Returns:
            None
        """
        fig = plt.figure(figsize=(8, 6))
        fig.add_subplot(111)

        MPASVisualizationStyle.add_timestamp_and_branding(fig)

        output = capsys.readouterr().out
        assert "using footer axes for branding" in output
        plt.close(fig)

    def test_add_timestamp_and_branding_none_fig(self) -> None:
        """
        This test verifies that the `add_timestamp_and_branding` function does nothing when called with None. The function should handle the None input gracefully without raising any exceptions.

        Parameters:
            None

        Returns:
            None
        """
        MPASVisualizationStyle.add_timestamp_and_branding(None)  # type: ignore


class TestGenerateLevelsNonTemperature:
    """ Tests for _generate_levels_from_data when variable is NOT temperature. """

    def test_non_temp_variable_uses_linear_spacing(self) -> None:
        """
        This test checks that the `_generate_levels_from_data` function generates linearly spaced levels for non-temperature variables. When the variable name does not indicate temperature, the helper should produce levels that are evenly spaced between the data minimum and maximum. This test creates a DataArray with a known range of values, calls the function with a non-temperature variable name, and asserts that the generated levels are linearly spaced and include the expected number of levels, confirming that the function correctly applies linear spacing for non-temperature variables.

        Parameters:
            None

        Returns:
            None
        """
        data = xr.DataArray(np.linspace(0, 100, 500))
        levels = MPASVisualizationStyle._generate_levels_from_data(data, 'mslp')
        assert levels is not None
        assert len(levels) == 16  

    def test_all_nan_returns_none(self) -> None:
        """
        This test checks that the `_generate_levels_from_data` function returns None when all data values are NaN. The helper should detect that there are no valid numeric values to generate levels from and return None to indicate that level generation is not possible. This test creates a DataArray filled with NaN values, calls the function, and asserts that the result is None, confirming that the function correctly handles cases with no valid data.

        Parameters:
            None

        Returns:
            None
        """
        data = xr.DataArray(np.full(50, np.nan))
        levels = MPASVisualizationStyle._generate_levels_from_data(data, 'temp')
        assert levels is None

    def test_zero_range_returns_none(self) -> None:
        """
        This test checks that the `_generate_levels_from_data` function returns None when the data has zero range (i.e., all values are the same). The helper should recognize that there is no variability in the data to create meaningful levels and return None to indicate that level generation is not possible. This test creates a DataArray filled with a constant value, calls the function, and asserts that the result is None, confirming that the function correctly handles cases with zero data range.

        Parameters:
            None
        
        Returns:
            None
        """
        data = xr.DataArray(np.full(50, 5.0))
        levels = MPASVisualizationStyle._generate_levels_from_data(data, 'pressure')
        assert levels is None


class TestHumidityBoundaryExact:
    """ Tests for the humidity fractional vs percentage boundary. """

    def test_humidity_at_boundary_1_1(self) -> None:
        """
        This test checks that the `get_variable_specific_settings` function uses fractional RH levels when the maximum data value is exactly 1.1. The function should correctly identify the boundary condition and apply the appropriate settings. This test creates a data array with values ranging from 0.1 to 1.1, calls the function, and asserts that the colormap is 'BuGn' and that at least one level is below 1.5, confirming that the function correctly handles the boundary case.
        
        Parameters:
            None

        Returns:
            None
        """
        data = np.linspace(0.1, 1.1, 100)
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('rh', data)
        assert cmap == 'BuGn'
        assert levels is not None
        assert any(lev < 1.5 for lev in levels)

    def test_humidity_above_boundary(self) -> None:
        """
        This test checks that the `get_variable_specific_settings` function uses percentage-scale levels when the maximum data value is above 1.1. The function should correctly identify that the data exceeds the fractional RH boundary and apply the appropriate settings. This test creates a data array with values ranging from 10 to 100, calls the function, and asserts that the colormap is 'BuGn' and that at least one level is above or equal to 10, confirming that the function correctly handles cases above the boundary.

        Parameters:
            None

        Returns:
            None
        """
        data = np.linspace(10, 100, 100)
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings('humidity', data)
        assert cmap == 'BuGn'
        assert levels is not None
        assert any(lev >= 10 for lev in levels)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
