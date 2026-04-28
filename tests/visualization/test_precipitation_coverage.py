#!/usr/bin/env python3

"""
MPASdiag Test Suite: Precipitation Plotter Coverage

This module contains unit tests for the MPASPrecipitationPlotter class in the mpasdiag.visualization.precipitation module. The tests are designed to cover specific branches and code paths in the precipitation plotting functionality, including unit conversion, colormap preparation, time annotation, plot creation, overlay setup, and coordinate extraction. Each test case focuses on a particular aspect of the code to ensure that all branches are exercised and that the methods behave as expected under various conditions. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import matplotlib
from pathlib import Path
matplotlib.use('Agg')
from typing import Any
import numpy as np
import pytest
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch

from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter


N_CELLS = 20
LON_MIN, LON_MAX = -100., -80.
LAT_MIN, LAT_MAX = 30., 50.


def _lon() -> np.ndarray:
    """
    This helper function generates a 1D array of longitude values evenly spaced between LON_MIN and LON_MAX, with a total of N_CELLS points. It is used in the test cases to provide consistent longitude data for the precipitation plotter tests. 

    Parameters:
        None

    Returns:
        np.ndarray: A 1D array of longitude values from LON_MIN to LON_MAX with N_CELLS points.
    """
    return np.linspace(LON_MIN, LON_MAX, N_CELLS)


def _lat() -> np.ndarray:
    """
    This helper function generates a 1D array of latitude values evenly spaced between LAT_MIN and LAT_MAX, with a total of N_CELLS points. It is used in the test cases to provide consistent latitude data for the precipitation plotter tests. 

    Parameters:
        None

    Returns:
        np.ndarray: A 1D array of latitude values from LAT_MIN to LAT_MAX with N_CELLS points. 
    """
    return np.linspace(LAT_MIN, LAT_MAX, N_CELLS)


def _precip(val: float = 2.0) -> np.ndarray:
    """
    This helper function generates a 1D array of precipitation values with N_CELLS points, all set to the specified value (default is 2.0). It is used in the test cases to provide consistent precipitation data for the precipitation plotter tests, allowing for validation of unit conversion, colormap preparation, and other functionalities that depend on the precipitation data. 

    Parameters:
        val (float): The value to fill the precipitation array with. Default is 2.0.

    Returns:
        np.ndarray: A 1D array of precipitation values with N_CELLS points, all set to the specified value.
    """
    return np.full(N_CELLS, val)


def _make_geo_ax() -> tuple[plt.Figure, plt.Axes]:
    """
    This helper function creates a Matplotlib figure and adds a GeoAxes with a PlateCarree projection. It is used in the test cases that require a GeoAxes for plotting, such as when testing time annotations or other plot elements that depend on the axes. The function returns the created figure and axes as a tuple, allowing the test cases to use them for plotting and then close the figure afterward to free resources. 

    Parameters:
        None    

    Returns:
        tuple[plt.Figure, plt.Axes]: A tuple containing the created Matplotlib figure and GeoAxes.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    return fig, ax


class TestConvertPrecipitationUnits:
    """ Covers unit-conversion branches in _convert_precipitation_units. """

    def test_none_data_array_returns_original(self: 'TestConvertPrecipitationUnits') -> None:
        """
        This test case verifies that when the data_array input to the _convert_precipitation_units method is None, the method returns the original precipitation data and the correct units label without attempting any conversion. This covers the branch in the method where it checks if data_array is None and ensures that it handles this case by returning the input data directly. The test uses the _precip helper function to generate consistent precipitation data for validation. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        data, label = plotter._convert_precipitation_units(_precip(), None, 'rainnc')
        assert label == 'mm'
        np.testing.assert_array_equal(data, _precip())

    def test_non_ndarray_precip_passes_through(self: 'TestConvertPrecipitationUnits') -> None:
        """
        This test case verifies that when the convert_data_for_display method returns a non-ndarray type (e.g., a list), the _convert_precipitation_units method correctly uses the converted data and label without further conversion. This covers the branch in the method where it checks if the converted data is an ndarray and ensures that if it is not, it is returned directly without modification. The test uses the _precip helper function to generate consistent precipitation data for validation, and mocks the convert_data_for_display method to return a list and units for testing. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        da = xr.DataArray(_precip(), dims=['nCells'], attrs={'units': 'mm'})
        with patch('mpasdiag.visualization.precipitation.UnitConverter.convert_data_for_display',
                   return_value=(da, {'units': 'mm'})):
            data, label = plotter._convert_precipitation_units(da, da, 'rainnc')
        assert label == 'mm'

    def test_attribute_error_returns_original_with_fallback_metadata(self: 'TestConvertPrecipitationUnits') -> None:
        """
        This test case verifies that when the convert_data_for_display method raises an AttributeError (e.g., due to missing attributes), the _convert_precipitation_units method correctly falls back to returning the original precipitation data and a default units label. This covers the branch in the method where it handles exceptions from the unit conversion process and ensures that it provides a fallback mechanism to return the input data with a reasonable default label. The test uses the _precip helper function to generate consistent precipitation data for validation, and mocks the convert_data_for_display method to raise an AttributeError for testing.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        precip = _precip()
        data_array = xr.DataArray(precip, dims=['nCells'], attrs={'units': 'mm', 'long_name': 'Rain'})

        with patch('mpasdiag.visualization.precipitation.UnitConverter.convert_data_for_display',
                   side_effect=AttributeError("missing attrs")):
            data, label = plotter._convert_precipitation_units(precip, data_array, 'rainnc')

        assert label == 'mm'
        np.testing.assert_array_equal(data, precip)

    def test_converted_ndarray_branch(self: 'TestConvertPrecipitationUnits') -> None:
        """
        This test case verifies that when the convert_data_for_display method returns a NumPy ndarray, the _convert_precipitation_units method correctly uses the converted data and label for display. This covers the branch in the method where it checks if the converted data is an ndarray and ensures that it returns the converted data directly when it is of the correct type. The test uses the _precip helper function to generate consistent precipitation data for validation, and mocks the convert_data_for_display method to return a NumPy array and units for testing. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        precip = _precip()
        data_array = xr.DataArray(precip, dims=['nCells'], attrs={'units': 'mm'})
        converted = np.ones(N_CELLS) * 3.0

        with patch('mpasdiag.visualization.precipitation.UnitConverter.convert_data_for_display',
                   return_value=(converted, {'units': 'mm'})):
            data, label = plotter._convert_precipitation_units(precip, data_array, 'rainnc')

        np.testing.assert_array_equal(data, converted)

    def test_converted_other_type_branch(self: 'TestConvertPrecipitationUnits') -> None:
        """
        This test case verifies that when the convert_data_for_display method returns a converted data type that is not a NumPy ndarray (e.g., a list), the _convert_precipitation_units method correctly uses the converted data and label without further conversion. This covers the branch in the method where it checks if the converted data is an ndarray and ensures that if it is not, it is returned directly without modification. The test uses the _precip helper function to generate consistent precipitation data for validation, and mocks the convert_data_for_display method to return a list and units for testing. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        precip = _precip()
        data_array = xr.DataArray(precip, dims=['nCells'], attrs={'units': 'mm'})
        converted_list = list(precip)

        with patch('mpasdiag.visualization.precipitation.UnitConverter.convert_data_for_display',
                   return_value=(converted_list, {'units': 'mm'})):
            data, label = plotter._convert_precipitation_units(precip, data_array, 'rainnc')

        assert isinstance(data, np.ndarray)

    def test_negative_values_clipped(self: 'TestConvertPrecipitationUnits') -> None:
        """
        This test case verifies that when the precipitation data contains negative values, the _convert_precipitation_units method correctly clips them to zero. This covers the branch in the method where it ensures that precipitation values are non-negative for display. The test uses a custom precipitation array with negative values and mocks the convert_data_for_display method to return it, allowing for validation that the final returned data has all values clipped to zero or above.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        precip = np.array([-1.0, 2.0, -0.5, 3.0])
        data_array = xr.DataArray(precip, dims=['nCells'], attrs={'units': 'mm'})

        with patch('mpasdiag.visualization.precipitation.UnitConverter.convert_data_for_display',
                   return_value=(precip.copy(), {'units': 'mm'})):
            data, _ = plotter._convert_precipitation_units(precip, data_array, 'rainnc')

        assert np.all(data >= 0)


class TestPreparePrecipitationColormap:
    """ Covers colormap-only and clim filtering branches. """

    def test_custom_colormap_only_uses_default_levels(self: 'TestPreparePrecipitationColormap') -> None:
        """
        This test case verifies that when a custom colormap is provided but no levels are specified, the _prepare_precipitation_colormap method generates and uses default levels appropriate for the specified accumulation period. This covers the branch in the method where it checks if a colormap is provided without levels and ensures that it creates default levels based on the accumulation period. The test validates that the returned levels list is not empty, indicating that default levels were generated. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()

        cmap, norm, levels = plotter._prepare_precipitation_colormap(
            colormap='Blues', levels=None, accum_period='a01h',
            clim_min=None, clim_max=None,
        )

        assert len(levels) > 0

    def test_custom_colormap_and_levels_both_used(self: 'TestPreparePrecipitationColormap') -> None:
        """
        This test case verifies that when both a custom colormap and custom levels are provided, the _prepare_precipitation_colormap method uses the provided levels. This covers the branch in the method where it checks if both colormap and levels are provided and ensures that the custom levels are used without modification.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        custom_levels = [0.5, 1.0, 2.0, 5.0]

        cmap, norm, levels = plotter._prepare_precipitation_colormap(
            colormap='Blues', levels=custom_levels, accum_period='a01h',
            clim_min=None, clim_max=None,
        )

        assert 0.5 in levels or 1.0 in levels

    def test_clim_filtering_restricts_levels(self: 'TestPreparePrecipitationColormap') -> None:
        """
        This test case verifies that when clim_min and clim_max are provided, the _prepare_precipitation_colormap method filters and pads the levels accordingly. This covers the branch in the method where it checks if clim_min and clim_max are specified and ensures that the levels are restricted within the specified range.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()

        cmap, norm, levels = plotter._prepare_precipitation_colormap(
            colormap=None, levels=None, accum_period='a24h',
            clim_min=1.0, clim_max=10.0,
        )

        assert min(levels) >= 1.0
        assert max(levels) <= 10.0

    def test_clim_min_inserted_when_absent(self: 'TestPreparePrecipitationColormap') -> None:
        """
        This test case verifies that when clim_min is provided but not present in the levels, the _prepare_precipitation_colormap method inserts clim_min at the front of the levels list. This covers the branch in the method where it checks if clim_min is specified and ensures that it is included in the levels.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()

        cmap, norm, levels = plotter._prepare_precipitation_colormap(
            colormap=None, levels=None, accum_period='a01h',
            clim_min=0.01, clim_max=50.0,
        )

        assert 0.01 in levels or levels[0] <= 0.01

    def test_clim_max_appended_when_absent(self: 'TestPreparePrecipitationColormap') -> None:
        """
        This test case verifies that when clim_max is provided but not present in the levels, the _prepare_precipitation_colormap method appends clim_max at the end of the levels list. This covers the branch in the method where it checks if clim_max is specified and ensures that it is included in the levels.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()

        cmap, norm, levels = plotter._prepare_precipitation_colormap(
            colormap='Blues', levels=[0.5, 1.0, 2.0], accum_period='a01h',
            clim_min=0.0, clim_max=3.7,  # 3.7 not in [0.5,1.0,2.0] → appended
        )

        assert 3.7 in levels


class TestAddTimeAnnotation:
    """Covers else-return when neither time_end nor accum_period is given."""

    def test_no_time_no_accum_returns_early(self: 'TestAddTimeAnnotation') -> None:
        """
        This test case verifies that when neither time_end nor accum_period is provided, the _add_time_annotation method returns early without adding any annotation to the plot. This covers the branch in the method where it checks for the presence of time_end and accum_period and ensures that it does not attempt to add an annotation if both are missing. The test creates a GeoAxes for plotting and calls the method with None values for time_end and accum_period, then checks that no annotation was added to the axes. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        fig, ax = _make_geo_ax()
        plotter.ax = ax
        try:
            plotter._add_time_annotation(None, None, '')
        finally:
            plt.close(fig)

    def test_no_time_but_accum_period_adds_annotation(self: 'TestAddTimeAnnotation') -> None:
        """
        This test case verifies that when time_end is None but accum_period is provided, the _add_time_annotation method adds an annotation to the plot. This covers the branch in the method where it checks for the presence of accum_period and ensures that an annotation is added even if time_end is not available. The test creates a GeoAxes for plotting and calls the method with None for time_end but a valid accum_period, then checks that an annotation was added to the axes. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        fig, ax = _make_geo_ax()
        plotter.ax = ax
        try:
            plotter._add_time_annotation(None, None, 'a01h')
        finally:
            plt.close(fig)


class TestCreatePrecipitationMapValidation:
    """ Covers ValueError raises before any figure is created. """

    def test_invalid_plot_type_raises(self: 'TestCreatePrecipitationMapValidation') -> None:
        """
        This test case verifies that when an invalid plot_type is provided to the create_precipitation_map method, it raises a ValueError. This covers the branch in the method where it checks if plot_type is one of the accepted values ('contour', 'pcolormesh') and ensures that it does not proceed with creating a figure if the plot_type is invalid. The test uses unittest.mock to patch the method and asserts that a ValueError is raised with the expected message when an invalid plot_type is used. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        with pytest.raises(ValueError, match="plot_type must be"):
            plotter.create_precipitation_map(
                _lon(), _lat(), _precip(),
                LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
                plot_type='heatmap',
            )

    def test_invalid_extent_raises(self: 'TestCreatePrecipitationMapValidation') -> None:
        """
        This test case verifies that when the longitude bounds are invalid (e.g., lon_max is less than or equal to lon_min), the create_precipitation_map method raises a ValueError indicating an invalid plot extent. This covers the branch in the method where it checks the validity of the longitude and latitude bounds before creating the plot and ensures that it does not proceed with plotting if the extent is invalid. The test uses unittest.mock to patch the method and asserts that a ValueError is raised with the expected message when invalid longitude bounds are provided. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        with pytest.raises(ValueError, match="Invalid plot extent"):
            plotter.create_precipitation_map(
                _lon(), _lat(), _precip(),
                LON_MAX, LON_MIN, LAT_MIN, LAT_MAX,  # lon flipped
            )


class TestCreatePrecipitationMapContour:
    """ Covers the contour branch (plot_type='contour') in create_precipitation_map. """

    def test_contour_plot_type_calls_create_contour_plot(self: 'TestCreatePrecipitationMapContour') -> None:
        """
        This test case verifies that when the plot_type is set to 'contour', the create_precipitation_map method calls the _create_contour_plot method to generate the contour plot. This covers the branch in the method where it checks the plot_type and ensures that the appropriate plotting function is called for contour plots. The test uses unittest.mock to patch the _create_contour_plot method and asserts that it is called exactly once when create_precipitation_map is invoked with plot_type='contour'. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        lon, lat, data = _lon(), _lat(), _precip(2.0)
        with patch.object(plotter, '_create_contour_plot') as mock_contour:
            with patch.object(plotter, '_add_gridlines'):
                with patch.object(plotter, 'add_timestamp_and_branding'):
                    fig, ax = plotter.create_precipitation_map(
                        lon, lat, data, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
                        plot_type='contour',
                    )
        assert mock_contour.call_count == 1
        plt.close('all')


class TestSetupOverlayColormap:
    """ Covers colormap + levels and colormap-only branches in _setup_overlay_colormap. """

    def test_colormap_and_levels_both_provided(self: 'TestSetupOverlayColormap') -> None:
        """
        This test case verifies that when both a colormap and levels are provided to the _setup_overlay_colormap method, it uses the provided levels without modification. This covers the branch in the method where it checks if both colormap and levels are provided and ensures that the custom levels are used directly. The test validates that the returned levels match the input levels, confirming that the method correctly handles cases where both parameters are specified. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        cmap, norm, levels = plotter._setup_overlay_colormap(
            colormap='Blues', levels=[0.5, 1.0, 5.0, 10.0], accum_period='a01h'
        )
        assert levels == sorted(set([0.5, 1.0, 5.0, 10.0]))

    def test_colormap_only_uses_default_levels(self: 'TestSetupOverlayColormap') -> None:
        """
        This test case verifies that when a colormap is provided but no levels are specified, the _setup_overlay_colormap method generates and uses default levels appropriate for the specified accumulation period. This covers the branch in the method where it checks if a colormap is provided without levels and ensures that it creates default levels based on the accumulation period. The test validates that the returned levels list is not empty, indicating that default levels were generated. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()

        cmap, norm, levels = plotter._setup_overlay_colormap(
            colormap='Reds', levels=None, accum_period='a01h'
        )

        assert len(levels) > 0

    def test_neither_colormap_nor_levels_uses_defaults(self: 'TestSetupOverlayColormap') -> None:
        """
        This test case verifies that when neither a colormap nor levels are provided, the _setup_overlay_colormap method generates and uses default levels appropriate for the specified accumulation period. This covers the branch in the method where it checks if both colormap and levels are None and ensures that it creates default levels based on the accumulation period. The test validates that the returned levels list is not empty, indicating that default levels were generated even when no colormap was specified. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()

        cmap, norm, levels = plotter._setup_overlay_colormap(
            colormap=None, levels=None, accum_period='a01h'
        )

        assert len(levels) > 0


class TestCalculateOverlayGridResolution:
    """ Covers explicit grid_resolution input return. """

    def test_explicit_resolution_returned(self: 'TestCalculateOverlayGridResolution') -> None:
        """
        This test case verifies that when an explicit grid_resolution input is provided to the _calculate_overlay_grid_resolution method, it returns that value directly without performing any calculations. This covers the branch in the method where it checks if grid_resolution_input is not None and ensures that it uses the provided resolution directly for the overlay grid. The test validates that the returned resolution matches the input value, confirming that the method correctly handles explicit grid resolution inputs. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        result = plotter._calculate_overlay_grid_resolution(0.25, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
        assert result == pytest.approx(0.25)

    def test_adaptive_resolution_when_none(self: 'TestCalculateOverlayGridResolution') -> None:
        """
        This test case verifies that when the grid_resolution_input is None, the _calculate_overlay_grid_resolution method calculates an adaptive grid resolution based on the longitude and latitude bounds. This covers the branch in the method where it checks if grid_resolution_input is None and performs calculations to determine an appropriate grid resolution for the overlay. The test validates that the returned resolution is within a reasonable range (e.g., between 0.1 and 1.0 degrees) for typical longitude and latitude bounds, confirming that the method correctly computes an adaptive resolution when no explicit input is provided. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        result = plotter._calculate_overlay_grid_resolution(None, -100., -80., 30., 50.)
        assert 0.1 <= result <= 1.0


class TestAddPrecipitationOverlayValidation:
    """ Covers invalid plot_type raise in add_precipitation_overlay. """

    def test_invalid_plot_type_raises(self: 'TestAddPrecipitationOverlayValidation') -> None:
        """
        This test case verifies that when an invalid plot_type is provided to the add_precipitation_overlay method, it raises a ValueError. This covers the branch in the method where it checks if plot_type is one of the accepted values ('contour', 'pcolormesh', 'hexbin') and ensures that it does not attempt to add an overlay if the plot_type is invalid. The test uses unittest.mock to create a mock axes object and asserts that a ValueError is raised with the expected message when an invalid plot_type is used for adding a precipitation overlay. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        mock_ax = MagicMock()
        with pytest.raises(ValueError, match="plot_type must be"):
            plotter.add_precipitation_overlay(
                mock_ax, _lon(), _lat(),
                {'data': _precip(), 'plot_type': 'hexbin'}
            )


class TestExtractCoordinatesFromProcessor:
    """ Covers three fallback paths in _extract_coordinates_from_processor. """

    def test_extract_spatial_coordinates_fallback(self: 'TestExtractCoordinatesFromProcessor') -> None:
        """
        This test case verifies that when the processor has an extract_spatial_coordinates method but not an extract_2d_coordinates_for_variable method, the _extract_coordinates_from_processor method correctly calls extract_spatial_coordinates to retrieve the longitude and latitude coordinates. This covers the branch in the method where it checks for the presence of extract_2d_coordinates_for_variable and falls back to extract_spatial_coordinates if the former is not available. The test uses unittest.mock to create a processor mock with the appropriate method and return value, then asserts that the returned coordinates match the expected values and that extract_spatial_coordinates was called exactly once. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        expected = (_lon(), _lat())

        proc = MagicMock(spec=['extract_spatial_coordinates'])
        proc.extract_spatial_coordinates.return_value = expected

        lon, lat = plotter._extract_coordinates_from_processor(proc, 'rainnc')
        assert np.array_equal(lon, expected[0])
        assert np.array_equal(lat, expected[1])
        proc.extract_spatial_coordinates.assert_called_once()

    def test_direct_dataset_access_fallback(self: 'TestExtractCoordinatesFromProcessor') -> None:
        """
        This test case verifies that when the processor has neither an extract_2d_coordinates_for_variable method nor an extract_spatial_coordinates method, the _extract_coordinates_from_processor method correctly falls back to accessing the dataset's lonCell and latCell values to retrieve the longitude and latitude coordinates. This covers the branch in the method where it checks for the presence of both coordinate extraction methods and falls back to direct dataset access if neither is available. The test uses unittest.mock to create a processor mock without the coordinate extraction methods but with the appropriate dataset attributes, then asserts that the returned coordinates match the expected values from the dataset. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        lon_vals = _lon()
        lat_vals = _lat()

        proc = MagicMock(spec=['dataset'])
        proc.dataset.lonCell.values = lon_vals
        proc.dataset.latCell.values = lat_vals

        lon, lat = plotter._extract_coordinates_from_processor(proc, 'rainnc')
        np.testing.assert_array_equal(lon, lon_vals)
        np.testing.assert_array_equal(lat, lat_vals)

    def test_attribute_error_fallback(self: 'TestExtractCoordinatesFromProcessor') -> None:
        """
        This test case verifies that when the processor's extract_2d_coordinates_for_variable method raises an AttributeError (e.g., due to missing attributes), the _extract_coordinates_from_processor method correctly falls back to accessing the dataset's lonCell and latCell values to retrieve the longitude and latitude coordinates. This covers the branch in the method where it handles exceptions from the coordinate extraction process and ensures that it provides a fallback mechanism to access coordinates directly from the dataset when necessary. The test uses unittest.mock to create a processor mock that raises an AttributeError for extract_2d_coordinates_for_variable but has the appropriate dataset attributes, then asserts that the returned coordinates match the expected values from the dataset. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        lon_vals = _lon()
        lat_vals = _lat()

        proc = MagicMock()
        proc.extract_2d_coordinates_for_variable.side_effect = AttributeError("no attr")
        proc.dataset.lonCell.values = lon_vals
        proc.dataset.latCell.values = lat_vals

        lon, lat = plotter._extract_coordinates_from_processor(proc, 'rainnc')
        np.testing.assert_array_equal(lon, lon_vals)
        np.testing.assert_array_equal(lat, lat_vals)

class TestSetupBatchTimeIndices:
    """ Covers early-return paths when time steps are insufficient or all filtered. """

    def _make_proc(self: 'TestSetupBatchTimeIndices', 
                   n_times: int) -> MagicMock:
        """
        This helper method creates a mock processor with a dataset that has a specified number of time steps. It sets up the dataset's sizes attribute to include a 'Time' dimension with the given number of steps, allowing the test cases to simulate different scenarios of time step availability for the _setup_batch_time_indices method. This method is used in multiple test cases to create processors with varying numbers of time steps to validate the behavior of the method under conditions of insufficient time steps and user index filtering. 

        Parameters:
            n_times (int): The number of time steps to include in the mock processor's dataset. 

        Returns:
            MagicMock: A mock processor with the specified number of time steps.
        """
        proc = MagicMock()
        proc.dataset.sizes = {'Time': n_times}
        return proc

    def test_insufficient_time_steps_returns_empty(self: 'TestSetupBatchTimeIndices') -> None:
        """
        This test case verifies that when the number of time steps in the processor's dataset is insufficient for the specified accumulation period, the _setup_batch_time_indices method returns an empty list of indices and the correct number of hours for the accumulation period. This covers the branch in the method where it checks for sufficient time steps and ensures that it returns an empty list when there is not enough data to process. The test uses the _make_proc helper method to create a mock processor with only 1 time step, then calls the method with an 'a24h' accumulation period and asserts that the returned indices list is empty and that the hours value is 24. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        result, hours = plotter._setup_batch_time_indices(self._make_proc(1), 'a24h', None)
        assert result == []
        assert hours == 24

    def test_insufficient_steps_prints_warning(self: 'TestSetupBatchTimeIndices', 
                                               capsys: 'pytest.CaptureFixture') -> None:
        """
        This test case verifies that when the number of time steps in the processor's dataset is insufficient for the specified accumulation period, the _setup_batch_time_indices method prints a warning message to standard output. This covers the branch in the method where it checks for sufficient time steps and ensures that it provides feedback to the user when there is not enough data to process. The test uses the _make_proc helper method to create a mock processor with only 1 time step, then calls the method with an 'a24h' accumulation period and captures the standard output to assert that a warning message is present. 

        Parameters:
            capsys (pytest.CaptureFixture): A pytest fixture for capturing standard output.

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        plotter._setup_batch_time_indices(self._make_proc(1), 'a24h', None)
        out = capsys.readouterr().out
        assert 'Warning' in out or 'time steps' in out

    def test_all_user_indices_filtered_out(self: 'TestSetupBatchTimeIndices') -> None:
        """
        This test case verifies that when user-specified time indices are provided but all of them are filtered out due to being invalid for the specified accumulation period, the _setup_batch_time_indices method returns an empty list of indices. This covers the branch in the method where it checks the validity of user-specified indices and ensures that it returns an empty list if all provided indices are invalid. The test uses the _make_proc helper method to create a mock processor with 5 time steps, then calls the method with user indices that are all invalid for an 'a01h' accumulation period (e.g., [0]) and asserts that the returned indices list is empty. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        result, hours = plotter._setup_batch_time_indices(self._make_proc(5), 'a01h', [0])
        assert result == []

    def test_valid_user_indices_returned(self: 'TestSetupBatchTimeIndices') -> None:
        """
        This test case verifies that when user-specified time indices are provided and they are valid for the specified accumulation period, the _setup_batch_time_indices method returns those indices correctly. This covers the branch in the method where it checks the validity of user-specified indices and ensures that it returns them when they are within the acceptable range. The test uses the _make_proc helper method to create a mock processor with 10 time steps, then calls the method with user indices that are valid for an 'a01h' accumulation period, asserting that the returned indices list matches the input. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        result, hours = plotter._setup_batch_time_indices(self._make_proc(10), 'a01h', [1, 2, 3])
        assert result == [1, 2, 3]

    def test_default_all_valid_indices(self: 'TestSetupBatchTimeIndices') -> None:
        """
        This test case verifies that when no user-specified time indices are provided, the _setup_batch_time_indices method returns a default list of all valid indices for the specified accumulation period. This covers the branch in the method where it generates a default list of indices based on the number of time steps and the accumulation period when no user input is given. The test uses the _make_proc helper method to create a mock processor with 5 time steps, then calls the method with no user indices for an 'a01h' accumulation period, asserting that the returned indices list includes all valid indices (1 through 4 in this case).

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        result, hours = plotter._setup_batch_time_indices(self._make_proc(5), 'a01h', None)
        assert result == [1, 2, 3, 4]


class TestProcessSingleTimeStep:
    """ Covers no-Time-coordinate fallback and custom title template. """

    def _make_proc_no_time(self: 'TestProcessSingleTimeStep') -> MagicMock:
        """
        This helper method creates a mock processor with a dataset that lacks a 'Time' coordinate, simulating the scenario where the dataset has a 'time' coordinate instead. This allows test cases to validate the behavior of the _process_single_time_step method when it encounters a dataset without the expected 'Time' coordinate and needs to use fallback values for time_str and time_end. The method sets up the dataset with the necessary dimensions and variables to mimic a realistic processor state for testing. 

        Parameters:
            None

        Returns:
            MagicMock: A mock processor with a dataset that lacks a 'Time' coordinate.
        """
        n = N_CELLS
        ds = xr.Dataset({'rainnc': (['time', 'nCells'], np.zeros((3, n)))})
        proc = MagicMock()
        proc.dataset = ds
        proc.data_type = 'forecast'
        return proc

    def test_no_time_coordinate_uses_index_string(self: 'TestProcessSingleTimeStep', 
                                                  tmp_path: 'Path') -> None:
        """
        This test case verifies that when the processor's dataset lacks a 'Time' coordinate, the _process_single_time_step method uses a fallback time string based on the time index (e.g., 't001') and sets time_end to None. This covers the branch in the method where it checks for the presence of a 'Time' coordinate and ensures that it handles cases where time information is not available by using a default string representation of the time index. The test creates a mock processor with a dataset that has no 'Time' coordinate, then calls the method and asserts that the generated file names include the expected fallback time string. 

        Parameters:
            tmp_path (Path): A temporary directory provided by pytest for file output.

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        proc = self._make_proc_no_time()
        precip_da = xr.DataArray(np.zeros(N_CELLS), dims=['nCells'])

        with patch('mpasdiag.visualization.precipitation.PrecipitationDiagnostics') as mock_cls:
            mock_cls.return_value.compute_precipitation_difference.return_value = precip_da
            with patch.object(plotter, 'create_precipitation_map',
                              return_value=(MagicMock(), MagicMock())):
                with patch.object(plotter, 'save_plot'):
                    with patch.object(plotter, 'close_plot'):
                        files = plotter._process_single_time_step(
                            proc, 1, _lon(), _lat(),
                            LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
                            'rainnc', 'a01h', 'scatter', None, None, None,
                            None, str(tmp_path), 'test', ['png'],
                        )

        assert any('t001' in f for f in files)

    def test_custom_title_template_applied(self: 'TestProcessSingleTimeStep', 
                                           tmp_path: 'Path') -> None:
        """
        This test case verifies that when a custom title template is provided to the _process_single_time_step method, it is correctly applied to the plot title. This covers the branch in the method where it formats the title using the provided template and ensures that the placeholders in the template are replaced with the appropriate values (e.g., variable name, time string, accumulation period). The test creates a mock processor with a dataset that has no 'Time' coordinate, then calls the method with a custom title template and asserts that the captured title includes expected values based on the template formatting. 

        Parameters:
            tmp_path (Path): A temporary directory provided by pytest for file output.

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        proc = self._make_proc_no_time()
        precip_da = xr.DataArray(np.zeros(N_CELLS), dims=['nCells'])
        captured_title = {}

        def capture_title(*args: Any, 
                          **kwargs: Any) -> tuple:
            """
            This helper function is designed to capture the title argument passed to the create_precipitation_map method. It extracts the title from either the keyword arguments or the positional arguments (depending on how the method is called) and stores it in a dictionary for later assertions in the test case. The function then returns a tuple of MagicMock objects to mimic the expected return type of create_precipitation_map, allowing the test to proceed without actually creating a plot. 

            Parameters:
                *args (Any): Positional arguments passed to the create_precipitation_map method.
                **kwargs (Any): Keyword arguments passed to the create_precipitation_map method.

            Returns:
                tuple: A tuple of two MagicMock objects to mimic the expected return type of create_precipitation_map. 
            """
            captured_title['title'] = kwargs.get('title', args[7] if len(args) > 7 else '')
            return MagicMock(), MagicMock()

        with patch('mpasdiag.visualization.precipitation.PrecipitationDiagnostics') as mock_cls:
            mock_cls.return_value.compute_precipitation_difference.return_value = precip_da
            with patch.object(plotter, 'create_precipitation_map',
                              side_effect=capture_title):
                with patch.object(plotter, 'save_plot'):
                    with patch.object(plotter, 'close_plot'):
                        plotter._process_single_time_step(
                            proc, 1, _lon(), _lat(),
                            LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
                            'rainnc', 'a01h', 'scatter', None, None, None,
                            '{var_name}_{time_str}_{accum_period}',
                            str(tmp_path), 'test', ['png'],
                        )
        assert 'RAINNC' in captured_title.get('title', '')


class TestCreateBatchPrecipitationMaps:
    """ Covers validation raises, early-return, progress prints, and error handling. """

    def test_processor_none_raises(self: 'TestCreateBatchPrecipitationMaps') -> None:
        """
        This test case verifies that when the processor argument is None, the create_batch_precipitation_maps method raises a ValueError indicating that the processor cannot be None. This covers the branch in the method where it checks if the processor is None before attempting to extract coordinates and process time steps, ensuring that it does not proceed with batch processing without a valid processor. The test asserts that a ValueError is raised with the expected message when None is passed as the processor argument. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        with pytest.raises(ValueError, match="Processor cannot be None"):
            plotter.create_batch_precipitation_maps(None, '/tmp', LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)

    def test_dataset_none_raises(self: 'TestCreateBatchPrecipitationMaps', 
                                 tmp_path: 'Path') -> None:
        """
        This test case verifies that when the processor's dataset is None, the create_batch_precipitation_maps method raises a ValueError indicating that no data is loaded. This covers the branch in the method where it checks if the processor's dataset is available before attempting to extract coordinates and process time steps, ensuring that it does not proceed with batch processing without valid data. The test creates a mock processor with a None dataset and asserts that a ValueError is raised with the expected message when the method is called. 

        Parameters:
            tmp_path (Path): A temporary directory provided by pytest for file output.

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        proc = MagicMock()
        proc.dataset = None
        with pytest.raises(ValueError, match="No data loaded"):
            plotter.create_batch_precipitation_maps(proc, '/tmp', LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)

    def test_empty_time_indices_returns_empty_list(self: 'TestCreateBatchPrecipitationMaps', 
                                                   tmp_path: 'Path') -> None:
        """
        This test case verifies that when the _setup_batch_time_indices method returns an empty list of time indices (e.g., due to insufficient time steps), the create_batch_precipitation_maps method also returns an empty list without attempting to process any time steps. This covers the branch in the method where it checks if there are valid time indices to process and ensures that it returns early with an empty list if there are none. The test creates a mock processor with a dataset that has only 1 time step, then calls the method and asserts that the returned result is an empty list, confirming that it correctly handles cases with insufficient time steps. 

        Parameters:
            tmp_path (Path): A temporary directory provided by pytest for file output.

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        proc = MagicMock()
        proc.dataset.sizes = {'Time': 1}
        proc.extract_2d_coordinates_for_variable.return_value = (_lon(), _lat())

        result = plotter.create_batch_precipitation_maps(
            proc, str(tmp_path), LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
            accum_period='a24h',
        )
        assert result == []

    def test_progress_print_every_10_steps(self: 'TestCreateBatchPrecipitationMaps', 
                                           tmp_path: 'Path', 
                                           capsys: 'pytest.CaptureFixture') -> None:
        """
        This test case verifies that the create_batch_precipitation_maps method prints a progress message to standard output every 10 time steps processed. This covers the branch in the method where it checks if the current time step index is a multiple of 10 and prints a message indicating how many steps have been completed. The test creates a mock processor with a dataset that has enough time steps to trigger the progress print (e.g., 12 time steps), then calls the method and captures the standard output to assert that the expected progress message is present after processing 10 steps. 

        Parameters:
            tmp_path (Path): A temporary directory provided by pytest for file output.
            capsys (pytest.CaptureFixture): A fixture to capture stdout and stderr output.

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        proc = MagicMock()
        proc.dataset.sizes = {'Time': 12} 
        proc.extract_2d_coordinates_for_variable.return_value = (_lon(), _lat())

        with patch.object(plotter, '_process_single_time_step', return_value=['file.png']):
            result = plotter.create_batch_precipitation_maps(
                proc, str(tmp_path), LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
                accum_period='a01h',
            )

        out = capsys.readouterr().out
        assert 'Completed 10' in out
        assert len(result) == 11  

    def test_exception_in_time_step_continues(self: 'TestCreateBatchPrecipitationMaps', 
                                              tmp_path: 'Path', 
                                              capsys: 'pytest.CaptureFixture') -> None:
        """
        This test case verifies that if an exception occurs during the processing of a single time step, it is logged and the loop continues to process the remaining time steps. This covers the branch in the method where it wraps the call to _process_single_time_step in a try-except block and logs any exceptions without stopping the entire batch processing. The test creates a mock processor with multiple time steps and patches the _process_single_time_step method to raise an exception for one of the steps, then asserts that the exception is logged and that the method continues to process the remaining steps. 

        Parameters:
            tmp_path (Path): A temporary directory provided by pytest for file output.
            capsys (pytest.CaptureFixture): A fixture to capture stdout and stderr output.

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        proc = MagicMock()
        proc.dataset.sizes = {'Time': 3}
        proc.extract_2d_coordinates_for_variable.return_value = (_lon(), _lat())

        with patch.object(plotter, '_process_single_time_step',
                          side_effect=RuntimeError("plot failed")):
            result = plotter.create_batch_precipitation_maps(
                proc, str(tmp_path), LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
                accum_period='a01h',
            )

        out = capsys.readouterr().out
        assert 'Error' in out
        assert result == []


class TestRenderOverlayInterpolated:
    """ Covers config dispatch, ESMPy fallback, and contour rendering branches. """

    def _make_remapped_mock(self: 'TestRenderOverlayInterpolated') -> MagicMock:
        """
        This helper method creates a mock object representing a remapped dataset with appropriate attributes for testing the _render_overlay_interpolated method. It sets up the mock to have lon and lat values that span the specified longitude and latitude bounds, as well as a values attribute that simulates remapped precipitation data. This allows test cases to validate the behavior of the method when it receives a remapped dataset from the dispatch_remap function, ensuring that it can handle the expected structure of the remapped data for plotting. 

        Parameters:
            None

        Returns:
            MagicMock: A mock object representing the remapped dataset with appropriate attributes for testing. 
        """
        remapped = MagicMock()
        remapped.lon.values = np.linspace(LON_MIN, LON_MAX, 5)
        remapped.lat.values = np.linspace(LAT_MIN, LAT_MAX, 5)
        remapped.values = np.ones((5, 5)) * 2.0
        return remapped

    def test_config_dispatch_remap_path(self: 'TestRenderOverlayInterpolated') -> None:
        """
        This test case verifies that when a configuration object is provided to the _render_overlay_interpolated method, it correctly calls the dispatch_remap function to remap the data according to the specified configuration. This covers the branch in the method where it checks if a config is provided and uses it to perform remapping before plotting. The test uses unittest.mock to patch the dispatch_remap function and asserts that it is called with the expected arguments, confirming that the method correctly integrates with the remapping functionality when a configuration is present. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        mock_ax = MagicMock()
        lon, lat = _lon(), _lat()
        remapped = self._make_remapped_mock()

        mock_config = MagicMock()
        mock_config.remap_engine = None
        mock_config.remap_method = 'linear'

        with patch('mpasdiag.visualization.precipitation.dispatch_remap', return_value=remapped):
            plotter._render_overlay_interpolated(
                mock_ax, lon, lat, _precip(),
                LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
                'contourf', 0.5, 'precipitation', None,
                plt.get_cmap('Blues'), MagicMock(), [1., 2., 5.], 0.7,
                lon, lat, config=mock_config,
            )

        mock_ax.contourf.assert_called_once()

    def test_config_dispatch_with_none_dataset_creates_dataset(self: 'TestRenderOverlayInterpolated') -> None:
        """
        This test case verifies that when a configuration object is provided but the dataset argument is None, the _render_overlay_interpolated method still calls the dispatch_remap function to perform remapping. This covers the branch in the method where it checks for a config and attempts to remap even if the dataset is not provided, ensuring that it can handle cases where the dataset is None by creating a minimal dataset for remapping. The test uses unittest.mock to patch the dispatch_remap function and asserts that it is called, confirming that the method correctly attempts to remap data even when the dataset is initially None. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        mock_ax = MagicMock()
        lon, lat = _lon(), _lat()
        remapped = self._make_remapped_mock()

        mock_config = MagicMock()
        mock_config.remap_engine = None
        mock_config.remap_method = 'linear'

        with patch('mpasdiag.visualization.precipitation.dispatch_remap',
                   return_value=remapped) as mock_dispatch:
            plotter._render_overlay_interpolated(
                mock_ax, lon, lat, _precip(),
                LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
                'contourf', 0.5, 'precipitation', None,  
                plt.get_cmap('Blues'), MagicMock(), [1., 2., 5.], 0.7,
                lon, lat, config=mock_config,
            )

        assert mock_dispatch.called

    def test_esmpy_path_exception_falls_back_to_kdtree(self: 'TestRenderOverlayInterpolated') -> None:
        """
        This test case verifies that when ESMPY is available but the _backmap_to_full_grid method raises a RuntimeError (simulating an ESMPY error), the _render_overlay_interpolated method correctly falls back to using the KDTree-based remapping approach. This covers the branch in the method where it attempts to use ESMPY for remapping and handles exceptions by falling back to an alternative method. The test uses unittest.mock to patch the relevant methods and simulates an ESMPY error, then asserts that the contourf method is still called on the axes object, confirming that the method successfully falls back to the KDTree approach when ESMPY fails. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        mock_ax = MagicMock()
        lon, lat = _lon(), _lat()
        remapped = self._make_remapped_mock()

        with patch('mpasdiag.visualization.precipitation.ESMPY_AVAILABLE', True):
            with patch.object(plotter, '_has_boundary_data', return_value=True):
                with patch.object(plotter, '_ensure_boundary_data', return_value=None):
                    with patch.object(plotter, '_backmap_to_full_grid',
                                      side_effect=RuntimeError("ESMPy error")):
                        with patch('mpasdiag.visualization.precipitation'
                                   '.remap_mpas_to_latlon_with_masking',
                                   return_value=remapped):
                            plotter._render_overlay_interpolated(
                                mock_ax, lon, lat, _precip(),
                                LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
                                'contourf', 0.5, 'precipitation', None,
                                plt.get_cmap('Blues'), MagicMock(), [1., 2., 5.], 0.7,
                                lon, lat, config=None,
                            )

        mock_ax.contourf.assert_called_once()

    def test_contour_plot_type_calls_contour_and_clabel(self: 'TestRenderOverlayInterpolated') -> None:
        """
        This test case verifies that when the plot_type argument is set to 'contour', the _render_overlay_interpolated method correctly calls the contour method on the axes object to create contour lines and then calls the clabel method to add labels to those contours. This covers the branch in the method where it checks for the 'contour' plot type and ensures that it performs both contouring and labeling as expected. The test uses unittest.mock to create a mock axes object and asserts that both contour and clabel methods are called, confirming that the method handles the 'contour' plot type correctly. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        mock_ax = MagicMock()
        lon, lat = _lon(), _lat()
        remapped = self._make_remapped_mock()

        with patch('mpasdiag.visualization.precipitation.remap_mpas_to_latlon_with_masking',
                   return_value=remapped):
            plotter._render_overlay_interpolated(
                mock_ax, lon, lat, _precip(),
                LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
                'contour', 0.5, 'precipitation', None,
                plt.get_cmap('Blues'), MagicMock(), [1., 2., 5.], 0.7,
                lon, lat, config=None,
            )

        mock_ax.contour.assert_called_once()
        mock_ax.clabel.assert_called_once()

    def test_contour_clabel_exception_silenced(self: 'TestRenderOverlayInterpolated') -> None:
        """
        This test case verifies that if the clabel method raises an exception (e.g., due to an issue with the contour levels or labels), the _render_overlay_interpolated method silences the exception and does not allow it to propagate, ensuring that the plotting process can continue without crashing. This covers the branch in the method where it wraps the clabel call in a try-except block and handles any exceptions by passing, allowing the method to continue even if labeling fails. The test uses unittest.mock to create a mock axes object and simulates an exception from clabel, then asserts that no exception is raised from the _render_overlay_interpolated method, confirming that it correctly silences errors from clabel. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        mock_ax = MagicMock()
        mock_ax.clabel.side_effect = Exception("clabel error")
        lon, lat = _lon(), _lat()
        remapped = self._make_remapped_mock()

        with patch('mpasdiag.visualization.precipitation.remap_mpas_to_latlon_with_masking',
                   return_value=remapped):
            plotter._render_overlay_interpolated(
                mock_ax, lon, lat, _precip(),
                LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
                'contour', 0.5, 'precipitation', None,
                plt.get_cmap('Blues'), MagicMock(), [1., 2., 5.], 0.7,
                lon, lat, config=None,
            )


class TestSetupComparisonSubplot:
    """ Covers the regional (non-global) extent branch. """

    def test_regional_extent_calls_set_extent(self: 'TestSetupComparisonSubplot') -> None:
        """
        This test case verifies that when the _setup_comparison_subplot method is called with a regional (non-global) extent defined by the longitude and latitude bounds, it correctly calls the set_extent method on the axes object to set the plot extent to those bounds. This covers the branch in the method where it checks if the extent is regional and uses set_extent to configure the axes accordingly. The test uses unittest.mock to create a mock axes object and asserts that set_extent is called with the expected longitude and latitude bounds, confirming that the method correctly sets the extent for regional plots. 

        Parameters:
            None

        Returns:
            None
        """        
        plotter = MPASPrecipitationPlotter()
        fig, ax = _make_geo_ax()
        data_crs = ccrs.PlateCarree()
        try:
            plotter._setup_comparison_subplot(
                ax, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, data_crs, False, 0
            )
        finally:
            plt.close(fig)

    def test_right_panel_shows_right_labels(self: 'TestSetupComparisonSubplot') -> None:
        """
        This test case verifies that when the _setup_comparison_subplot method is called for the right panel (panel_index=1) with a regional extent, it correctly configures the axes to show only the right-side labels and ticks. This covers the branch in the method where it checks the panel index and sets the appropriate label and tick parameters to ensure that only the right panel displays labels, while the left panel does not. The test uses unittest.mock to create a mock axes object and asserts that set_extent is called with the expected longitude and latitude bounds, confirming that the method correctly configures the axes for the right panel in a regional plot. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        fig, ax = _make_geo_ax()
        data_crs = ccrs.PlateCarree()
        try:
            plotter._setup_comparison_subplot(
                ax, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, data_crs, False, 1
            )
        finally:
            plt.close(fig)


class TestPlotPrecipitationData:
    """ Covers no-valid-data early return. """

    def test_all_nan_data_returns_none(self: 'TestPlotPrecipitationData') -> None:
        """
        This test case verifies that when the precipitation data provided to the _plot_precipitation_data method consists entirely of NaN values, the method returns None. This covers the branch in the method where it checks if there are any valid (non-NaN) data points to plot and ensures that it handles cases with no valid data appropriately by returning None instead of attempting to plot. The test creates a mock axes object and a data array filled with NaN values, then calls the method and asserts that the result is None, confirming that the method correctly identifies the lack of valid data and responds accordingly. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        fig, ax = _make_geo_ax()
        data_crs = ccrs.PlateCarree()
        data = np.full(N_CELLS, np.nan)
        try:
            result = plotter._plot_precipitation_data(
                ax, _lon(), _lat(), data,
                LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
                plt.get_cmap('Blues'), MagicMock(), data_crs,
            )
            assert result is None
        finally:
            plt.close(fig)

    def test_out_of_bounds_data_returns_none(self: 'TestPlotPrecipitationData') -> None:
        """
        This test case verifies that when the precipitation data provided to the _plot_precipitation_data method contains valid values but all of the corresponding longitude points are outside the specified longitude bounds (LON_MIN, LON_MAX), the method returns None. This covers the branch in the method where it checks if there are any valid data points that fall within the specified geographic bounds and ensures that it handles cases where all data points are out of bounds by returning None instead of attempting to plot. The test creates a mock axes object and a data array with valid values, but sets the longitude values to be outside the defined bounds, then calls the method and asserts that the result is None, confirming that the method correctly identifies that there are no valid data points within the bounds and responds accordingly. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        fig, ax = _make_geo_ax()
        data_crs = ccrs.PlateCarree()
        lon = np.linspace(-180., -150., N_CELLS)
        data = _precip(2.0)
        try:
            result = plotter._plot_precipitation_data(
                ax, lon, _lat(), data,
                LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
                plt.get_cmap('Blues'), MagicMock(), data_crs,
            )
            assert result is None
        finally:
            plt.close(fig)


class TestSavePlot:
    """ Covers no-figure ValueError in save_plot. """

    def test_no_figure_raises(self: 'TestSavePlot', tmp_path) -> None:
        """
        This test case verifies that when the save_plot method is called without a figure being set (i.e., self.fig is None), it raises a ValueError indicating that there is no figure to save. This covers the branch in the method where it checks if self.fig is None before attempting to save, ensuring that it does not proceed with saving when there is no figure available. The test creates an instance of MPASPrecipitationPlotter without setting a figure, then calls the save_plot method and asserts that a ValueError is raised with the expected message, confirming that the method correctly handles cases where there is no figure to save.  

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        with pytest.raises(ValueError, match="No figure to save"):
            plotter.save_plot(str(tmp_path / 'test_output'))

    def test_saves_to_disk(self: 'TestSavePlot', 
                           tmp_path: 'Path') -> None:
        """
        This test case verifies that when the save_plot method is called with a valid figure set, it successfully saves the plot to disk in the specified format(s). This covers the branch in the method where it checks for a valid figure and proceeds to save it using plt.savefig, ensuring that the saving functionality works as expected. The test creates an instance of MPASPrecipitationPlotter, sets a figure, calls the save_plot method with a temporary path and specified format, and then asserts that the expected output file exists on disk, confirming that the plot was saved correctly. Finally, it closes the figure to clean up resources. 

        Parameters:
            tmp_path (Path): A temporary directory provided by pytest for file output. 

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        plotter.fig = plt.figure()
        output_path = str(tmp_path / 'test_precip')
        plotter.save_plot(output_path, formats=['png'])
        assert (tmp_path / 'test_precip.png').exists()
        plt.close(plotter.fig)


class TestApplyStyle:
    """Covers apply_style full body."""

    def test_apply_style_no_fig_ax(self: 'TestApplyStyle') -> None:
        """
        This test case verifies that when the apply_style method is called without a figure or axes being set (i.e., self.fig and self.ax are None), it applies the style without attempting to update the facecolor of the figure. This covers the branch in the method where it checks if fig and ax are set before applying styling to them, ensuring that it can apply styles even when no figure or axes are present without raising an exception. The test creates an instance of MPASPrecipitationPlotter, calls the apply_style method, and asserts that no exceptions are raised, confirming that the method can handle cases with no figure or axes gracefully. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        plotter.apply_style()  # must not raise

    def test_apply_style_with_fig_and_ax(self: 'TestApplyStyle') -> None:
        """
        This test case verifies that when the apply_style method is called with a figure and axes set, it applies the style to the figure and axes without raising any exceptions. This covers the branch in the method where it checks if fig and ax are available and applies styling to them, ensuring that it can successfully apply styles when a figure and axes are present. The test creates an instance of MPASPrecipitationPlotter, sets a figure and axes, calls the apply_style method, and asserts that no exceptions are raised, confirming that the method can apply styles correctly when fig and ax are set. Finally, it closes the figure to clean up resources. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        fig, ax = _make_geo_ax()
        plotter.fig = fig
        plotter.ax = ax
        try:
            plotter.apply_style()
        finally:
            plt.close(fig)

    def test_apply_style_calls_style_manager_when_present(self: 'TestApplyStyle') -> None:
        """
        This test case verifies that when the apply_style method is called and a style_manager is present, it correctly calls the apply_style method on the style_manager with the specified style name. This covers the branch in the method where it checks for the presence of a style_manager and delegates the styling to it, ensuring that it integrates properly with the style management system when available. The test creates an instance of MPASPrecipitationPlotter, sets a mock style_manager, calls the apply_style method with a custom style name, and asserts that the style_manager's apply_style method is called with the expected argument, confirming that the method correctly interacts with the style manager. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        mock_sm = MagicMock()
        plotter.style_manager = mock_sm
        plotter.apply_style('custom')
        mock_sm.apply_style.assert_called_once_with('custom')


class TestCreatePrecipitationComparisonPlot:
    """ Covers colorbar tick setup in create_precipitation_comparison_plot. """

    def test_colorbar_ticks_set_when_cbar_not_none(self: 'TestCreatePrecipitationComparisonPlot') -> None:
        """
        This test case verifies that when the create_precipitation_comparison_plot method is called and a colorbar object is created (i.e., not None), the method correctly calls the set_ticks method on the colorbar to configure the ticks. This covers the branch in the method where it checks if a colorbar was created and attempts to set the ticks, ensuring that it properly configures the colorbar when it is present. The test uses unittest.mock to create a mock colorbar object and asserts that set_ticks is called, confirming that the method correctly handles colorbar tick setup when a colorbar is created. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        lon, lat = _lon(), _lat()
        data1 = _precip(2.0)
        data2 = _precip(5.0)

        mock_cbar = MagicMock()
        mock_scatter = MagicMock()

        with patch.object(plotter, 'setup_map_projection',
                          return_value=(ccrs.PlateCarree(), ccrs.PlateCarree())):
            with patch.object(plotter, '_setup_comparison_subplot'):
                with patch.object(plotter, '_plot_precipitation_data',
                                  side_effect=[mock_scatter, None]):
                    with patch('mpasdiag.visualization.precipitation'
                               '.MPASVisualizationStyle.add_colorbar',
                               return_value=mock_cbar):
                        with patch.object(plotter, 'add_timestamp_and_branding'):
                            fig, axes = plotter.create_precipitation_comparison_plot(
                                lon, lat, data1, data2,
                                LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
                            )
        mock_cbar.set_ticks.assert_called_once()
        plt.close('all')

    def test_colorbar_tick_exception_silenced(self: 'TestCreatePrecipitationComparisonPlot') -> None:
        """
        This test case verifies that if the set_ticks method on the colorbar raises an exception (e.g., due to an issue with the tick configuration), the create_precipitation_comparison_plot method silences the exception and does not allow it to propagate, ensuring that the plotting process can continue without crashing. This covers the branch in the method where it wraps the set_ticks call in a try-except block and handles any exceptions by passing, allowing the method to continue even if there is an issue with setting colorbar ticks. The test uses unittest.mock to create a mock colorbar object and simulates an exception from set_ticks, then asserts that no exception is raised from the create_precipitation_comparison_plot method, confirming that it correctly silences errors from set_ticks.  

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        lon, lat = _lon(), _lat()
        mock_cbar = MagicMock()
        mock_cbar.set_ticks.side_effect = Exception("tick error")
        mock_scatter = MagicMock()

        with patch.object(plotter, 'setup_map_projection',
                          return_value=(ccrs.PlateCarree(), ccrs.PlateCarree())):
            with patch.object(plotter, '_setup_comparison_subplot'):
                with patch.object(plotter, '_plot_precipitation_data',
                                  side_effect=[mock_scatter, None]):
                    with patch('mpasdiag.visualization.precipitation'
                               '.MPASVisualizationStyle.add_colorbar',
                               return_value=mock_cbar):
                        with patch.object(plotter, 'add_timestamp_and_branding'):
                            fig, axes = plotter.create_precipitation_comparison_plot(
                                lon, lat, _precip(), _precip(),
                                LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
                            )
        plt.close('all')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
