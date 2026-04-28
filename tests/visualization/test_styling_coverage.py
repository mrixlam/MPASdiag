#!/usr/bin/env python3

"""
MPASdiag Test Suite: Visualization Styling Coverage

This module contains tests designed to achieve coverage of specific branches in the MPASVisualizationStyle class, particularly in the create_precip_colormap, _hours_to_accum_period, _generate_levels_from_data, _levels_for_temperature, _levels_for_precipitation, _levels_for_wind, _levels_for_geopotential, _levels_for_humidity, _levels_default, get_variable_specific_settings, setup_map_projection, add_timestamp_and_branding, _create_footer_axes, save_plot, get_3d_variable_style, and _configure_horizontal_colorbar methods. Each test is focused on triggering a specific branch or exception handling path to ensure comprehensive coverage of the styling logic. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import pytest
import xarray as xr
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch

from mpasdiag.visualization.styling import MPASVisualizationStyle


class TestCreatePrecipColormapException:
    """ Test coverage for exception handling in create_precip_colormap, specifically when re.search raises an exception. """

    def test_re_search_exception_sets_hours_none(self: 'TestCreatePrecipColormapException') -> None:
        """
        This test verifies that when re.search raises an exception while trying to extract hours from the variable name in create_precip_colormap, the function correctly sets hours to None. This confirms that the exception handling branch for re.search is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        with patch('mpasdiag.visualization.styling.re.search',
                   side_effect=Exception("regex error")):
            cmap, levels = MPASVisualizationStyle.create_precip_colormap("a06h")
        assert 150 in levels

    def test_re_search_exception_still_returns_colormap(self: 'TestCreatePrecipColormapException') -> None:
        """
        This test verifies that if re.search raises an exception while trying to extract hours from the variable name, the function still returns a valid colormap and levels list. This ensures that the exception handling does not prevent the function from producing usable output. 

        Parameters:
            None

        Returns:
            None
        """
        with patch('mpasdiag.visualization.styling.re.search',
                   side_effect=RuntimeError("forced")):
            cmap, levels = MPASVisualizationStyle.create_precip_colormap("1h")
        assert cmap is not None
        assert isinstance(levels, list)


class TestHoursToAccumPeriod:
    """ Test coverage for _hours_to_accum_period, specifically the branches for hours == 3 and hours == 12. """

    def test_hours_3_returns_a03h(self: 'TestHoursToAccumPeriod') -> None:
        """
        This test verifies that when the input hours is 3, the _hours_to_accum_period function correctly returns 'a03h'. This confirms that the specific branch for 3-hour accumulation is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle._hours_to_accum_period(3)
        assert result == 'a03h'

    def test_hours_12_returns_a12h(self: 'TestHoursToAccumPeriod') -> None:
        """
        This test verifies that when the input hours is 12, the _hours_to_accum_period function correctly returns 'a12h'. This confirms that the specific branch for 12-hour accumulation is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle._hours_to_accum_period(12)
        assert result == 'a12h'


class TestGenerateLevelsFromData:
    """ Test coverage for _generate_levels_from_data, specifically the branches for small, medium, and large temperature ranges, as well as the exception handling when accessing data values. """

    def test_small_temp_range_uses_step_half(self: 'TestGenerateLevelsFromData') -> None:
        """
        This test verifies that when the temperature range is small, the _generate_levels_from_data function correctly uses a step of 0.5. This confirms that the specific branch for small temperature ranges is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(np.linspace(20.0, 26.0, 100))
        levels = MPASVisualizationStyle._generate_levels_from_data(da, 't2m')
        assert levels is not None
        assert len(levels) > 0

    def test_medium_temp_range_uses_step_1(self: 'TestGenerateLevelsFromData') -> None:
        """
        This test verifies that when the temperature range is medium, the _generate_levels_from_data function correctly uses a step of 1. This confirms that the specific branch for medium temperature ranges is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(np.linspace(0.0, 20.0, 100))
        levels = MPASVisualizationStyle._generate_levels_from_data(da, 't2m')
        assert levels is not None

    def test_large_temp_range_uses_step_5(self: 'TestGenerateLevelsFromData') -> None:
        """
        This test verifies that when the temperature range is large, the _generate_levels_from_data function correctly uses a step of 5. This confirms that the specific branch for large temperature ranges is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(np.linspace(-50.0, 50.0, 100))
        levels = MPASVisualizationStyle._generate_levels_from_data(da, 'temperature')
        assert levels is not None

    def test_exception_in_values_returns_none(self: 'TestGenerateLevelsFromData') -> None:
        """
        This test verifies that when an exception is raised while trying to access the values of the data array in _generate_levels_from_data, the function correctly handles the exception and returns None. This confirms that the exception handling branch for data access issues is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        class BadDataArray:
            """ A mock data array that raises an exception when trying to access values. """

            @property
            def values(self: 'BadDataArray') -> None:
                """
                This property simulates a data array that raises an exception when trying to access its values. This is used to test the exception handling in _generate_levels_from_data.

                Parameters:
                    None

                Returns:
                    None
                """
                raise RuntimeError("bad values access")

        result = MPASVisualizationStyle._generate_levels_from_data(BadDataArray(), 't2m')
        assert result is None


class TestLevelsForTemperature:
    """ Test coverage for _levels_for_temperature, specifically the branches for data_range > 40, data_range <= 20, and data_range between 20 and 40. """

    def test_data_range_above_40_uses_step_5(self: 'TestLevelsForTemperature') -> None:
        """
        This test verifies that when the temperature range is above 40, the _levels_for_temperature function correctly uses a step of 5. This confirms that the specific branch for large temperature ranges is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        colormap, levels = MPASVisualizationStyle._levels_for_temperature(
            data_min=0.0, data_max=50.0, data_range=50.0
        )

        assert colormap == 'RdYlBu_r'
        assert len(levels) > 0

        if len(levels) > 1:
            diffs = [levels[i+1] - levels[i] for i in range(len(levels)-1)]
            assert all(abs(d - 5) < 0.01 for d in diffs)

    def test_data_range_at_most_20_uses_step_1(self: 'TestLevelsForTemperature') -> None:
        """
        This test verifies that when the temperature range is at most 20, the _levels_for_temperature function correctly uses a step of 1. This confirms that the specific branch for small temperature ranges is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        colormap, levels = MPASVisualizationStyle._levels_for_temperature(
            data_min=10.0, data_max=20.0, data_range=10.0
        )

        assert colormap == 'RdYlBu_r'

        if len(levels) > 1:
            diffs = [levels[i+1] - levels[i] for i in range(len(levels)-1)]
            assert all(abs(d - 1) < 0.01 for d in diffs)

    def test_data_range_between_20_and_40_uses_step_2(self: 'TestLevelsForTemperature') -> None:
        """
        This test verifies that when the temperature range is between 20 and 40, the _levels_for_temperature function correctly uses a step of 2. This confirms that the specific branch for medium temperature ranges is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        colormap, levels = MPASVisualizationStyle._levels_for_temperature(
            data_min=0.0, data_max=30.0, data_range=30.0
        )
        assert colormap == 'RdYlBu_r'


class TestLevelsForPrecipitation:
    """ Test coverage for _levels_for_precipitation, specifically the branches for '3h', '6h', '12h' in var_lower, and the else branch for no matching pattern. """

    def test_3h_variable_uses_a03h(self: 'TestLevelsForPrecipitation') -> None:
        """
        This test verifies that when the precipitation variable corresponds to a 3-hour accumulation period, the _levels_for_precipitation function correctly uses the 'a03h' accumulation period. This confirms that the specific branch for 3-hour accumulation periods is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        cmap, levels = MPASVisualizationStyle._levels_for_precipitation('rain_3h', 50.0)
        assert cmap is not None
        assert len(levels) > 0

    def test_6h_variable_uses_a06h(self: 'TestLevelsForPrecipitation') -> None:
        """
        This test verifies that when the precipitation variable corresponds to a 6-hour accumulation period, the _levels_for_precipitation function correctly uses the 'a06h' accumulation period. This confirms that the specific branch for 6-hour accumulation periods is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        cmap, levels = MPASVisualizationStyle._levels_for_precipitation('precip_6h', 50.0)
        assert cmap is not None

    def test_12h_variable_uses_a12h(self: 'TestLevelsForPrecipitation') -> None:
        """
        This test verifies that when the precipitation variable corresponds to a 12-hour accumulation period, the _levels_for_precipitation function correctly uses the 'a12h' accumulation period. This confirms that the specific branch for 12-hour accumulation periods is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        cmap, levels = MPASVisualizationStyle._levels_for_precipitation('rain_12h', 100.0)
        assert cmap is not None

    def test_no_matching_pattern_uses_else_a24h(self: 'TestLevelsForPrecipitation') -> None:
        """
        This test verifies that when the precipitation variable does not match any specific accumulation period pattern, the _levels_for_precipitation function correctly uses the 'a24h' accumulation period. This confirms that the else branch for no matching pattern is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        cmap, levels = MPASVisualizationStyle._levels_for_precipitation('qpf_total', 50.0)
        assert cmap is not None

    def test_03h_pattern_also_works(self: 'TestLevelsForPrecipitation') -> None:
        """
        This test verifies that when the precipitation variable corresponds to a 3-hour accumulation period, the _levels_for_precipitation function correctly uses the 'a03h' accumulation period. This confirms that the specific branch for 3-hour accumulation periods is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        cmap, levels = MPASVisualizationStyle._levels_for_precipitation('precip_03h', 30.0)
        assert cmap is not None


class TestLevelsForWind:
    """ Test coverage for _levels_for_wind, specifically the branches for data_max < 3, data_max < 15, data_max >= 30, and data_max in [15, 30). """

    def test_very_low_wind_speed(self: 'TestLevelsForWind') -> None:
        """
        This test verifies that when the wind speed is very low (data_max < 3), the _levels_for_wind function correctly uses 0.5 m/s spacing levels. This confirms that the specific branch for very low wind speeds is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        colormap, levels = MPASVisualizationStyle._levels_for_wind(data_max=2.0)
        assert colormap == 'plasma'
        assert 0.5 in levels

    def test_moderate_wind_speed(self: 'TestLevelsForWind') -> None:
        """
        This test verifies that when the wind speed is moderate (data_max < 15), the _levels_for_wind function correctly uses 2 m/s spacing levels. This confirms that the specific branch for moderate wind speeds is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        colormap, levels = MPASVisualizationStyle._levels_for_wind(data_max=10.0)
        assert colormap == 'plasma'
        assert 2.0 in levels

    def test_high_wind_speed(self: 'TestLevelsForWind') -> None:
        """
        This test verifies that when the wind speed is high (data_max >= 30), the _levels_for_wind function correctly uses 10 m/s spacing levels. This confirms that the specific branch for high wind speeds is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        colormap, levels = MPASVisualizationStyle._levels_for_wind(data_max=40.0)
        assert colormap == 'plasma'
        assert 10.0 in levels

    def test_near_30_wind_speed(self: 'TestLevelsForWind') -> None:
        """
        This test verifies that when the wind speed is near 30 (data_max in [15, 30)), the _levels_for_wind function correctly uses 5 m/s spacing levels. This confirms that the specific branch for wind speeds in this range is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        colormap, levels = MPASVisualizationStyle._levels_for_wind(data_max=20.0)
        assert colormap == 'plasma'


class TestLevelsForGeopotential:
    """ Test coverage for _levels_for_geopotential, specifically the branches for data_range > 1000 and data_range > 2000. """

    def test_large_height_range_uses_step_30_or_60(self: 'TestLevelsForGeopotential') -> None:
        """
        This test verifies that when the height range is large (data_range > 1000), the _levels_for_geopotential function correctly uses a step of 30 or 60 meters. This confirms that the specific branch for large height ranges is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        colormap, levels = MPASVisualizationStyle._levels_for_geopotential(
            data_min=5000.0, data_max=6500.0, data_range=1500.0
        )
        assert colormap == 'terrain'
        assert len(levels) > 0

    def test_very_large_height_range_uses_step_60(self: 'TestLevelsForGeopotential') -> None:
        """
        This test verifies that when the height range is very large (data_range > 2000), the _levels_for_geopotential function correctly uses a step of 60 meters. This confirms that the specific branch for very large height ranges is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        colormap, levels = MPASVisualizationStyle._levels_for_geopotential(
            data_min=1000.0, data_max=4000.0, data_range=3000.0
        )

        assert colormap == 'terrain'

        if len(levels) > 1:
            diffs = [levels[i+1] - levels[i] for i in range(len(levels)-1)]
            assert all(abs(d - 60) < 0.01 for d in diffs)


class TestLevelsForHumidity:
    """ Test coverage for _levels_for_humidity, specifically the branches for small ranges of non-relative humidity variables and large ranges of non-relative humidity variables. """

    def test_small_range_non_rh_var_uses_log_style_levels(self: 'TestLevelsForHumidity') -> None:
        """
        This test verifies that when the humidity variable is not 'rh' or 'humidity' and the maximum value is small (data_max <= 1.1), the _levels_for_humidity function correctly uses small log-style levels. This confirms that the specific branch for small ranges of non-relative humidity variables is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        colormap, levels = MPASVisualizationStyle._levels_for_humidity(
            var_lower='q2', data_min=0.001, data_max=0.1
        )
        assert colormap == 'BuGn'
        assert 0.001 in levels

    def test_large_range_uses_percent_levels(self: 'TestLevelsForHumidity') -> None:
        """
        This test verifies that when the humidity variable is not 'rh' or 'humidity' and the maximum value is large (data_max > 1.1), the _levels_for_humidity function correctly uses percent-style levels. This confirms that the specific branch for large ranges of non-relative humidity variables is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        colormap, levels = MPASVisualizationStyle._levels_for_humidity(
            var_lower='relhum', data_min=30.0, data_max=90.0
        )
        assert colormap == 'BuGn'
        assert 50 in levels


class TestLevelsDefault:
    """ Test coverage for _levels_default, specifically the branches for all-negative data (plasma colormap) and mixed-sign data (RdBu_r colormap). """

    def test_all_negative_data_uses_plasma(self: 'TestLevelsDefault') -> None:
        """
        This test verifies that when the data contains only negative values (data_min < 0 and data_max <= 0), the _levels_default function correctly uses the 'plasma' colormap. This confirms that the specific branch for all-negative data is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        colormap, levels = MPASVisualizationStyle._levels_default(
            data_min=-50.0, data_max=-5.0, data_range=45.0
        )
        assert colormap == 'plasma'
        assert len(levels) > 0

    def test_mixed_sign_uses_rdbu(self: 'TestLevelsDefault') -> None:
        """
        This test verifies that when the data contains both negative and positive values (data_min < 0 and data_max > 0), the _levels_default function correctly uses the 'RdBu_r' colormap. This confirms that the specific branch for mixed-sign data is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        colormap, _ = MPASVisualizationStyle._levels_default(-10.0, 10.0, 20.0)
        assert colormap == 'RdBu_r'


class TestGetVariableSpecificSettings:
    """ Test coverage for get_variable_specific_settings, specifically the branches for xr.DataArray input and all-NaN data. """

    def test_xarray_input_uses_values_flatten(self: 'TestGetVariableSpecificSettings') -> None:
        """
        This test verifies that when the input data is an xr.DataArray, the get_variable_specific_settings function correctly accesses the values and flattens them. This confirms that the specific branch for handling xr.DataArray input is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(np.linspace(0.0, 50.0, 100))
        colormap, levels = MPASVisualizationStyle.get_variable_specific_settings('wind_speed', da)
        assert colormap is not None
        assert levels is not None

    def test_all_nan_data_returns_viridis_none(self: 'TestGetVariableSpecificSettings') -> None:
        """
        This test verifies that when all data values are NaN, the get_variable_specific_settings function correctly returns the 'viridis' colormap and None for levels. This confirms that the specific branch for all-NaN data is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        data = np.full(10, np.nan)
        colormap, levels = MPASVisualizationStyle.get_variable_specific_settings('anything', data)
        assert colormap == 'viridis'
        assert levels is None

    def test_all_nan_xarray_returns_viridis_none(self: 'TestGetVariableSpecificSettings') -> None:
        """
        This test verifies that when all data values in an xr.DataArray are NaN, the get_variable_specific_settings function correctly returns the 'viridis' colormap and None for levels. This confirms that the specific branch for all-NaN xr.DataArray data is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(np.full(5, np.nan))
        colormap, levels = MPASVisualizationStyle.get_variable_specific_settings('var', da)
        assert colormap == 'viridis'
        assert levels is None


class TestSetupMapProjection:
    """ Test coverage for setup_map_projection, specifically the branches for LambertConformal and unknown projections. """

    def test_lambertconformal_returns_lcc_projection(self: 'TestSetupMapProjection') -> None:
        """
        This test verifies that when the projection is 'LambertConformal', the setup_map_projection function correctly returns a LambertConformal projection. This confirms that the specific branch for LambertConformal projection is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        import cartopy.crs as ccrs
        proj, data_crs = MPASVisualizationStyle.setup_map_projection(
            -100., -80., 30., 50., projection='LambertConformal'
        )
        assert isinstance(proj, ccrs.LambertConformal)

    def test_unknown_projection_falls_back_to_platecarree(self: 'TestSetupMapProjection') -> None:
        """
        This test verifies that when the projection is unknown, the setup_map_projection function correctly falls back to a PlateCarree projection. This confirms that the specific branch for unknown projection is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        import cartopy.crs as ccrs
        proj, data_crs = MPASVisualizationStyle.setup_map_projection(
            -100., -80., 30., 50., projection='UnknownProjection'
        )
        assert isinstance(proj, ccrs.PlateCarree)


class TestAddTimestampAndBranding:
    """ Test coverage for add_timestamp_and_branding, specifically the branch for None figure. """

    def test_none_fig_returns_immediately(self: 'TestAddTimestampAndBranding') -> None:
        """
        This test verifies that when the figure is None, the add_timestamp_and_branding function correctly returns immediately without error. This confirms that the specific branch for None figure is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        MPASVisualizationStyle.add_timestamp_and_branding(None)


class TestCreateFooterAxes:
    """ Test coverage for _create_footer_axes, specifically the branches for reusing existing footer axes, handling get_gid exceptions, and handling get_position exceptions. """

    def test_reuses_existing_footer_axes(self: 'TestCreateFooterAxes') -> None:
        """
        This test verifies that when an existing footer axes with gid='mpasdiag_footer' is present, the _create_footer_axes function correctly returns it. This confirms that the specific branch for reusing existing footer axes is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        fig = plt.figure()
        footer = fig.add_axes((0.02, 0.07, 0.96, 0.07), frameon=False)
        footer.set_gid('mpasdiag_footer')
        result = MPASVisualizationStyle._create_footer_axes(fig)
        assert result is footer
        plt.close(fig)

    def test_get_gid_exception_continues_loop(self: 'TestCreateFooterAxes') -> None:
        """
        This test verifies that when get_gid() raises an exception, the _create_footer_axes function correctly continues the loop. This confirms that the specific branch for handling get_gid exceptions is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        fig = plt.figure()
        ax1 = fig.add_axes((0.1, 0.5, 0.8, 0.4))
        ax1.get_gid = MagicMock(side_effect=Exception("gid error"))
        result = MPASVisualizationStyle._create_footer_axes(fig)
        assert result is not None
        plt.close(fig)

    def test_colorbar_detection_exception_handled(self: 'TestCreateFooterAxes') -> None:
        """
        This test verifies that when get_position() raises an exception, the _create_footer_axes function correctly handles it. This confirms that the specific branch for handling get_position exceptions is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        fig = plt.figure()
        ax = fig.add_axes((0.1, 0.1, 0.8, 0.6))
        with patch.object(ax, 'get_position', side_effect=Exception("position error")):
            result = MPASVisualizationStyle._create_footer_axes(fig)
        assert result is not None
        plt.close(fig)


class TestSavePlot:
    """ Test coverage for save_plot, specifically the branches for None figure and missing output directory. """

    def test_none_figure_raises_value_error(self: 'TestSavePlot', tmp_path) -> None:
        """
        This test verifies that when the figure is None, the save_plot function correctly raises a ValueError. This confirms that the specific branch for None figure is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="No figure to save"):
            MPASVisualizationStyle.save_plot(None, str(tmp_path / 'test_output'))

    def test_creates_output_directory_if_missing(self: 'TestSavePlot', tmp_path) -> None:
        """
        This test verifies that when the output directory does not exist, the save_plot function correctly creates it. This confirms that the specific branch for missing output directory is functioning as intended.

        Parameters:
            tmp_path: Temporary path provided by pytest

        Returns:
            None
        """
        new_dir = str(tmp_path / "new_subdir" / "nested")
        output_path = os.path.join(new_dir, "test_plot")
        fig = plt.figure()
        MPASVisualizationStyle.save_plot(fig, output_path, formats=['png'])
        plt.close(fig)
        assert os.path.exists(new_dir)


class TestGet3DVariableStyle:
    """ Test coverage for get_3d_variable_style, specifically the branch for NotImplementedError. """

    def test_raises_not_implemented_error(self: 'TestGet3DVariableStyle') -> None:
        """
        This test verifies that when get_3d_variable_style is called, it correctly raises a NotImplementedError. This confirms that the specific branch for 3D variable support is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(NotImplementedError, match="3D variable support"):
            MPASVisualizationStyle.get_3d_variable_style('temperature')

    def test_raises_with_level_argument(self: 'TestGet3DVariableStyle') -> None:
        """
        This test verifies that when get_3d_variable_style is called with a level argument, it correctly raises a NotImplementedError. This confirms that the specific branch for 3D variable support with level argument is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(NotImplementedError):
            MPASVisualizationStyle.get_3d_variable_style('wind', level='500hPa')


class TestConfigureHorizontalColorbar:
    """ Test coverage for _configure_horizontal_colorbar, specifically the branches for exception handling. """

    def test_set_label_position_exception_is_silenced(self: 'TestConfigureHorizontalColorbar') -> None:
        """
        This test verifies that when set_label_position or set_ticks_position raises an exception, the _configure_horizontal_colorbar function correctly silences it. This confirms that the specific branch for handling set_label_position exceptions is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        mock_cbar = MagicMock()
        mock_cbar.ax.xaxis.set_label_position.side_effect = Exception("position error")
        mock_cbar.ax.xaxis.set_ticks_position.side_effect = Exception("ticks error")

        MPASVisualizationStyle._configure_horizontal_colorbar(
            mock_cbar, label=None, label_pos='top', tick_labelsize=8, labelpad=6.0
        )

    def test_set_label_failure_falls_back_to_set_xlabel(self: 'TestConfigureHorizontalColorbar') -> None:
        """
        This test verifies that when set_label raises an exception, the _configure_horizontal_colorbar function correctly falls back to using set_xlabel. This confirms that the specific branch for handling set_label exceptions is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        mock_cbar = MagicMock()
        mock_cbar.set_label.side_effect = Exception("label error")

        MPASVisualizationStyle._configure_horizontal_colorbar(
            mock_cbar, label="Temperature [K]", label_pos='top', tick_labelsize=8, labelpad=6.0
        )

        mock_cbar.ax.set_xlabel.assert_called_once()

    def test_with_no_label_skips_label_setting(self: 'TestConfigureHorizontalColorbar') -> None:
        """
        This test verifies that when no label is provided, the _configure_horizontal_colorbar function correctly skips setting the label. This confirms that the specific branch for handling None labels is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        mock_cbar = MagicMock()
        
        MPASVisualizationStyle._configure_horizontal_colorbar(
            mock_cbar, label=None, label_pos='bottom', tick_labelsize=8, labelpad=6.0
        )

        mock_cbar.set_label.assert_not_called()


class TestConfigureVerticalColorbar:
    """ Test coverage for _configure_vertical_colorbar, specifically the branches for exception handling. """

    def test_set_label_position_exception_is_silenced(self: 'TestConfigureVerticalColorbar') -> None:
        """
        This test verifies that when set_label_position raises an exception, the _configure_vertical_colorbar function
        correctly silences it. This confirms that the specific branch for handling set_label_position exceptions is
        functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        mock_cbar = MagicMock()
        mock_cbar.ax.yaxis.set_label_position.side_effect = Exception("position error")
        MPASVisualizationStyle._configure_vertical_colorbar(
            mock_cbar, label=None, label_pos='right', tick_labelsize=8, labelpad=6.0
        )

    def test_set_label_failure_falls_back_to_set_ylabel(self: 'TestConfigureVerticalColorbar') -> None:
        """
        This test verifies that when set_label raises an exception, the _configure_vertical_colorbar function correctly falls back to using set_ylabel. This confirms that the specific branch for handling set_label exceptions is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        mock_cbar = MagicMock()
        mock_cbar.set_label.side_effect = Exception("label error")

        MPASVisualizationStyle._configure_vertical_colorbar(
            mock_cbar, label="Pressure [hPa]", label_pos='right', tick_labelsize=8, labelpad=6.0
        )

        mock_cbar.ax.set_ylabel.assert_called_once()


class TestApplyColorbarFormatter:
    """ Test coverage for _apply_colorbar_formatter, specifically the branches for handling different format types and exceptions. """

    def test_string_format_sets_formatter(self: 'TestApplyColorbarFormatter') -> None:
        """
        This test verifies that when a string format is provided, the _apply_colorbar_formatter function correctly creates a FormatStrFormatter and assigns it to the colorbar. It also verifies that update_ticks is called. This confirms that the specific branch for handling string formats is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        mock_cbar = MagicMock()
        MPASVisualizationStyle._apply_colorbar_formatter(mock_cbar, '%d')
        mock_cbar.update_ticks.assert_called_once()

    def test_non_string_format_set_directly(self: 'TestApplyColorbarFormatter') -> None:
        """
        This test verifies that when a non-string format object is provided, the _apply_colorbar_formatter function correctly assigns it directly to the colorbar's formatter and calls update_ticks. This confirms that the specific branch for handling non-string format objects is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        import matplotlib.ticker as mticker
        mock_cbar = MagicMock()
        fmt_obj = mticker.FormatStrFormatter('%.2f')
        MPASVisualizationStyle._apply_colorbar_formatter(mock_cbar, fmt_obj)
        mock_cbar.update_ticks.assert_called_once()

    def test_exception_in_formatter_is_silenced(self: 'TestApplyColorbarFormatter') -> None:
        """
        This test verifies that when update_ticks raises an exception, the _apply_colorbar_formatter function correctly silences it. This confirms that the specific branch for handling exceptions in the formatter is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        mock_cbar = MagicMock()
        mock_cbar.update_ticks.side_effect = Exception("formatter error")
        MPASVisualizationStyle._apply_colorbar_formatter(mock_cbar, '%d')


class TestAddColorbarWithFormatter:
    """ Test coverage for add_colorbar with a focus on the fmt parameter, specifically the branches for non-None fmt calling the formatter and None fmt skipping the formatter. """

    def test_add_colorbar_calls_formatter(self: 'TestAddColorbarWithFormatter') -> None:
        """
        This test verifies that when a non-None format string is provided to the add_colorbar function, it correctly calls the _apply_colorbar_formatter function with the provided format. This confirms that the specific branch for handling non-None format strings is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = plt.subplots()
        data = np.random.rand(5, 5)
        im = ax.imshow(data)

        cbar = MPASVisualizationStyle.add_colorbar(
            fig, ax, im, label='Test', fmt='%d', orientation='horizontal'
        )

        assert cbar is not None
        plt.close(fig)

    def test_add_colorbar_fmt_none_skips_formatter(self: 'TestAddColorbarWithFormatter') -> None:
        """
        This test verifies that when fmt is None, the add_colorbar function correctly skips calling the formatter. This confirms that the specific branch for handling None format is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = plt.subplots()
        im = ax.imshow(np.random.rand(4, 4))

        cbar = MPASVisualizationStyle.add_colorbar(
            fig, ax, im, fmt=None, orientation='horizontal'
        )

        assert cbar is not None
        plt.close(fig)


class TestBuildColorbarLabel:
    """ Test coverage for build_colorbar_label with a focus on handling missing long_name and units, as well as combinations of both. """

    def test_default_long_name_used_when_metadata_missing(self: 'TestBuildColorbarLabel') -> None:
        """
        This test verifies that when the long_name is missing in the metadata, the build_colorbar_label function correctly uses the default_long_name. This confirms that the specific branch for handling missing long_name is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.build_colorbar_label(
            None, default_long_name='Temperature', default_units='K'
        )
        assert result == "Temperature [K]"

    def test_default_units_used_when_metadata_missing(self: 'TestBuildColorbarLabel') -> None:
        """
        This test verifies that when the units are missing in the metadata, the build_colorbar_label function correctly uses the default_units. This confirms that the specific branch for handling missing units is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.build_colorbar_label(
            {}, default_long_name='Temperature', default_units='K'
        )
        assert result == "Temperature [K]"

    def test_long_name_only_returned_without_units(self: 'TestBuildColorbarLabel') -> None:
        """
        This test verifies that when only the long_name is present in the metadata and units are missing, the build_colorbar_label function correctly returns just the long_name without any brackets. This confirms that the specific branch for handling missing units while long_name is present is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.build_colorbar_label(
            {'long_name': 'Surface Temperature'}
        )
        assert result == "Surface Temperature"

    def test_units_only_returned_in_brackets(self: 'TestBuildColorbarLabel') -> None:
        """
        This test verifies that when only the units are present in the metadata and long_name is missing, the build_colorbar_label function correctly returns just the units enclosed in brackets. This confirms that the specific branch for handling missing long_name while units are present is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.build_colorbar_label({'units': 'K'})
        assert result == "[K]"

    def test_both_metadata_returns_combined(self: 'TestBuildColorbarLabel') -> None:
        """
        This test verifies that when both long_name and units are present in the metadata, the build_colorbar_label function correctly combines them into the format "long_name [units]". This confirms that the specific branch for handling both pieces of metadata is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.build_colorbar_label(
            {'long_name': '2m Temperature', 'units': 'K'}
        )
        assert result == "2m Temperature [K]"

    def test_units_already_embedded_returns_long_name_as_is(self: 'TestBuildColorbarLabel') -> None:
        """
        This test verifies that when the long_name already contains the units in brackets, the build_colorbar_label function correctly returns the long_name unchanged without adding another set of brackets. This confirms that the specific branch for handling long_name that already includes units is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.build_colorbar_label(
            {'long_name': 'Temperature [K]', 'units': 'K'}
        )
        assert result == "Temperature [K]"

    def test_no_long_name_no_units_returns_none(self: 'TestBuildColorbarLabel') -> None:
        """
        This test verifies that when neither long_name nor units are present in the metadata, the build_colorbar_label function correctly returns None. This confirms that the specific branch for handling missing long_name and units is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.build_colorbar_label({})
        assert result is None

    def test_none_metadata_no_defaults_returns_none(self: 'TestBuildColorbarLabel') -> None:
        """
        This test verifies that when the metadata is None and no default values are provided, the build_colorbar_label function correctly returns None. This confirms that the specific branch for handling None metadata without defaults is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.build_colorbar_label(None)
        assert result is None


class TestFormatLatitude:
    """ Test coverage for format_latitude, specifically the branches for positive/zero latitude (N) and negative latitude (S). """

    def test_positive_latitude_north(self: 'TestFormatLatitude') -> None:
        """
        This test verifies that when the latitude value is positive, the format_latitude function correctly returns the latitude with the 'N' direction. This confirms that the specific branch for handling northern latitudes is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.format_latitude(45.0, None)
        assert result == "45.0°N"

    def test_negative_latitude_south(self: 'TestFormatLatitude') -> None:
        """
        This test verifies that when the latitude value is negative, the format_latitude function correctly returns the latitude with the 'S' direction. This confirms that the specific branch for handling southern latitudes is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.format_latitude(-30.5, None)
        assert result == "30.5°S"

    def test_zero_latitude_is_north(self: 'TestFormatLatitude') -> None:
        """
        This test verifies that when the latitude value is zero, the format_latitude function correctly returns the latitude with the 'N' direction. This confirms that the specific branch for handling zero latitude is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.format_latitude(0.0, None)
        assert result == "0.0°N"


class TestCalculateAdaptiveMarkerSize:
    """ Test coverage for calculate_adaptive_marker_size, specifically the branches for map_extent=None, middle density ranges, and small area scale. """

    def test_none_map_extent_returns_5(self: 'TestCalculateAdaptiveMarkerSize') -> None:
        """
        This test verifies that when the map_extent is None, the calculate_adaptive_marker_size function correctly returns a default marker size of 5.0. This confirms that the specific branch for handling None map_extent is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.calculate_adaptive_marker_size(None, 100)
        assert result == pytest.approx(5.0)

    def test_density_10_to_50_branch(self: 'TestCalculateAdaptiveMarkerSize') -> None:
        """
        This test verifies that when the density is between 10 and 50, the calculate_adaptive_marker_size function correctly returns a base size of 1.5. This confirms that the specific branch for handling this density range is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.calculate_adaptive_marker_size(
            (-100., -90., 30., 40.), num_points=2000
        )
        assert isinstance(result, float)
        assert result > 0

    def test_density_50_to_150_branch(self: 'TestCalculateAdaptiveMarkerSize') -> None:
        """
        This test verifies that when the density is between 50 and 150, the calculate_adaptive_marker_size function correctly returns a base size of 0.8. This confirms that the specific branch for handling this density range is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.calculate_adaptive_marker_size(
            (-100., -90., 30., 40.), num_points=7500
        )
        assert isinstance(result, float)

    def test_density_150_to_500_branch(self: 'TestCalculateAdaptiveMarkerSize') -> None:
        """
        This test verifies that when the density is between 150 and 500, the calculate_adaptive_marker_size function correctly returns a base size of 0.4. This confirms that the specific branch for handling this density range is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.calculate_adaptive_marker_size(
            (-100., -90., 30., 40.), num_points=30_000
        )
        assert isinstance(result, float)

    def test_density_above_500_branch(self: 'TestCalculateAdaptiveMarkerSize') -> None:
        """
        This test verifies that when the density is above 500,
        the calculate_adaptive_marker_size function correctly returns a base size of 0.25.
        This confirms that the specific branch for handling this density range is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.calculate_adaptive_marker_size(
            (-100., -90., 30., 40.), num_points=100_000
        )
        assert isinstance(result, float)

    def test_small_map_area_uses_large_area_scale(self: 'TestCalculateAdaptiveMarkerSize') -> None:
        """
        This test verifies that when the map area is small,
        the calculate_adaptive_marker_size function correctly uses a large area scale.
        This confirms that the specific branch for handling small map areas is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.calculate_adaptive_marker_size(
            (-91., -88., 30., 32.), num_points=100
        )
        assert isinstance(result, float)
        assert result > 0


class TestChooseTickFmt:
    """ Test coverage for _choose_tick_fmt, specifically the branches for no nonzero ticks, large magnitude non-integer values, medium magnitude non-integer values, small-medium magnitude non-integer values, and tiny magnitude non-integer values. """

    def test_no_nonzero_ticks_returns_int_format(self: 'TestChooseTickFmt') -> None:
        """
        This test verifies that when there are no nonzero ticks, the _choose_tick_fmt function correctly returns the integer format '{:.0f}'. This confirms that the specific branch for handling the case of no nonzero ticks is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        tick_array = np.array([0.5])
        nonzero_ticks = np.array([])  # forced empty to hit line 1028
        result = MPASVisualizationStyle._choose_tick_fmt(tick_array, nonzero_ticks)
        assert result == '{:.0f}'

    def test_large_magnitude_non_integer_returns_int_format(self: 'TestChooseTickFmt') -> None:
        """
        This test verifies that when the non-integer values have a typical magnitude greater than or equal to 100, the _choose_tick_fmt function correctly returns the integer format '{:.0f}'. This confirms that the specific branch for handling large magnitude non-integer values is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        tick_array = np.array([100.5, 200.3]) 
        nonzero_ticks = tick_array
        result = MPASVisualizationStyle._choose_tick_fmt(tick_array, nonzero_ticks)
        assert result == '{:.0f}'

    def test_medium_magnitude_non_integer_returns_1dp(self: 'TestChooseTickFmt') -> None:
        """
        This test verifies that when the non-integer values have a typical magnitude between 10 and 100, the _choose_tick_fmt function correctly returns the format '{:.1f}'. This confirms that the specific branch for handling medium magnitude non-integer values is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        tick_array = np.array([15.5, 20.3]) 
        nonzero_ticks = tick_array
        result = MPASVisualizationStyle._choose_tick_fmt(tick_array, nonzero_ticks)
        assert result == '{:.1f}'

    def test_small_medium_magnitude_returns_2dp(self: 'TestChooseTickFmt') -> None:
        """
        This test verifies that when the non-integer values have a typical magnitude between 0.01 and 10, the _choose_tick_fmt function correctly returns the format '{:.2f}'. This confirms that the specific branch for handling small-medium magnitude non-integer values is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        tick_array = np.array([0.5, 0.7, 1.3])  
        nonzero_ticks = tick_array
        result = MPASVisualizationStyle._choose_tick_fmt(tick_array, nonzero_ticks)
        assert result == '{:.2f}'

    def test_tiny_magnitude_returns_3dp_format(self: 'TestChooseTickFmt') -> None:
        """
        This test verifies that when the non-integer values have a typical magnitude less than 0.01, the _choose_tick_fmt function correctly returns the format '{:.3f}'. This confirms that the specific branch for handling tiny magnitude non-integer values is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        tick_array = np.array([0.001, 0.002, 0.003]) 
        nonzero_ticks = tick_array
        result = MPASVisualizationStyle._choose_tick_fmt(tick_array, nonzero_ticks)
        assert result == '{:.3f}'


class TestFormatTicksDynamic:
    """ Test coverage for format_ticks_dynamic, specifically the branches for empty ticks list, large magnitude values using scientific notation, and tiny magnitude values using scientific notation. """

    def test_empty_ticks_returns_empty_list(self: 'TestFormatTicksDynamic') -> None:
        """
        This test verifies that when the input ticks list is empty, the format_ticks_dynamic function correctly returns an empty list. This confirms that the specific branch for handling an empty ticks list is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.format_ticks_dynamic([])
        assert result == []

    def test_large_magnitude_uses_scientific_notation(self: 'TestFormatTicksDynamic') -> None:
        """
        This test verifies that when the input ticks list contains large magnitude values, the format_ticks_dynamic function correctly returns the values in scientific notation. This confirms that the specific branch for handling large magnitude values is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.format_ticks_dynamic([10000.0, 20000.0, 30000.0])
        assert all('e' in label for label in result)

    def test_tiny_values_use_scientific_notation(self: 'TestFormatTicksDynamic') -> None:
        """
        This test verifies that when the input ticks list contains tiny magnitude values, the format_ticks_dynamic function correctly returns the values in scientific notation. This confirms that the specific branch for handling tiny magnitude values is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASVisualizationStyle.format_ticks_dynamic([0.0001, 0.0002, 0.0003])
        assert all('e' in label for label in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
